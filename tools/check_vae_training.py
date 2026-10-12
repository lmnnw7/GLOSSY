"""Exercise a real VAE config/data/checkpoint for a few isolated training steps."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from AutoEncoder.model.REV_VQVAE import REV_VQVAE
from train.vae_training import (build_data_loaders, build_trainer, checkpoint_stage,
                                load_config, load_training_checkpoint)
from runtime import project_path
from config_io import add_raw_arguments, vae_input_options


class TrainingAudit(pl.Callback):
    def __init__(self):
        self.losses = []
        self.optimizer_steps = {'autoencoder': 0, 'discriminator': 0}

    def on_fit_start(self, trainer, model):
        self.frozen_codes = {name: tensor.clone() for name, tensor in model.codebook.state_dict().items()}
        self.discriminator_ids = {id(parameter) for parameter in model.volume_discriminator.parameters()}

    def on_before_optimizer_step(self, trainer, model, optimizer):
        parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
        label = 'discriminator' if id(parameters[0]) in self.discriminator_ids else 'autoencoder'
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f'{label}: missing or non-finite gradients')
        self.optimizer_steps[label] += 1

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        self.losses.append(float(loss))
        if model.perceptual_loss.training:
            raise RuntimeError('Perceptual features unexpectedly entered training mode')
        if model.stage == 2:
            if model.encoder.training or model.codebook.training:
                raise RuntimeError('Stage-2 encoder/codebook unexpectedly entered training mode')
            for name, tensor in model.codebook.state_dict().items():
                if not torch.equal(tensor, self.frozen_codes[name]):
                    raise RuntimeError(f'Stage-2 codebook changed: {name}')


def main(args):
    if args.steps < 2 or args.batch_size <= 0:
        raise ValueError('Use at least two steps and a positive batch size')
    config_path = project_path(args.config)
    cfg = load_config(config_path, int(OmegaConf.load(config_path).model.stage), **vae_input_options(args))
    destination = project_path(args.output_dir)
    if destination.exists():
        raise FileExistsError(f'Use a new output directory for each smoke run: {destination}')
    checkpoint = load_training_checkpoint(cfg.model.resume_from_checkpoint) if cfg.model.resume_from_checkpoint else None
    saved_stage = checkpoint_stage(checkpoint) if checkpoint else None
    if saved_stage not in (None, 1, 2) or (cfg.model.stage == 1 and saved_stage == 2):
        raise ValueError('Incompatible VAE checkpoint stage')
    full_resume = saved_stage == cfg.model.stage
    start_step = int(checkpoint.get('global_step', 0)) if full_resume else 0
    cfg.model.default_root_dir = str(destination)
    cfg.model.max_steps = start_step + args.steps
    cfg.model.max_epochs = -1
    cfg.model.batch_size = args.batch_size
    cfg.model.num_workers = 0
    cfg.model.log_max_volumes = 0
    cfg.model.gpus = 1
    if args.cpu:
        cfg.model.accelerator = 'cpu'
        cfg.model.precision = 32
    if args.test_gan:
        if cfg.model.volume_gan_weight <= 0:
            raise ValueError('--test-gan requires volume_gan_weight > 0')
        cfg.model.discriminator_iter_start = start_step
    pl.seed_everything(cfg.model.seed, workers=True)
    model = REV_VQVAE(cfg, initialize_perceptual=checkpoint is None)
    if checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    train_loader, validation_loader = build_data_loaders(cfg)
    audit = TrainingAudit()
    # Keep the check short; optimizer alternation continues across epochs.
    trainer = build_trainer(cfg, check_val_every_n_epoch=1,
                            extra_callback_factory=lambda: audit,
                            limit_train_batches=2, limit_val_batches=1,
                            num_sanity_val_steps=1)
    trainer.fit(model, train_loader, validation_loader,
                ckpt_path=cfg.model.resume_from_checkpoint if full_resume else None)
    validation = trainer.validate(model, validation_loader, verbose=False)[0]
    if not all(torch.isfinite(torch.tensor(value)) for value in validation.values()):
        raise FloatingPointError('Non-finite validation metrics')
    if args.test_gan and not audit.optimizer_steps['discriminator']:
        raise RuntimeError('GAN smoke run did not reach a discriminator step')
    trainer.save_checkpoint(str(destination / 'smoke.ckpt'))
    report = dict(stage=int(model.stage), start_step=start_step,
                  end_step=trainer.global_step, optimizer_steps=audit.optimizer_steps,
                  losses=audit.losses, validation=validation,
                  config=OmegaConf.to_container(cfg, resolve=True),
                  accuracy_verified=False)
    (destination / 'smoke_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Real-data training smoke check completed: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-path', help='Direct source directory/file; overrides the YAML data path')
    parser.add_argument('--ckpt', help='Override the initialization/resume checkpoint')
    add_raw_arguments(parser)
    parser.add_argument('--output-dir', required=True, help='New directory reserved for this smoke run')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--test-gan', action='store_true', help='Enable GAN immediately in the isolated smoke config')
    main(parser.parse_args())
