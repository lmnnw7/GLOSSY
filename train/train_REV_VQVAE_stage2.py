"""Initialize stage 2 from stage 1, or resume a complete stage-2 checkpoint."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytorch_lightning as pl
from AutoEncoder.model.REV_VQVAE import REV_VQVAE
from utils.config_io import add_vae_input_arguments, vae_input_options
from train.vae_training import (build_data_loaders, build_trainer, print_training_settings,
                                load_config, load_training_checkpoint, checkpoint_stage)


def main(cfg_path, **input_options):
    cfg = load_config(cfg_path, stage=2, **input_options)
    checkpoint_path = cfg.model.resume_from_checkpoint
    checkpoint = load_training_checkpoint(checkpoint_path)
    saved_stage = checkpoint_stage(checkpoint)
    if saved_stage not in (1, 2):
        raise ValueError('Expected a stage-1 or stage-2 VAE checkpoint')
    pl.seed_everything(cfg.model.seed, workers=True)
    train_loader, val_loader = build_data_loaders(cfg)
    print_training_settings(cfg)
    model = REV_VQVAE(cfg, initialize_perceptual=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    trainer = build_trainer(cfg, check_val_every_n_epoch=1)
    # Stage-1 optimizers include the encoder and cannot be restored into stage 2.
    resume_path = checkpoint_path if saved_stage == 2 else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_path)
    final_path = Path(cfg.model.default_root_dir) / 'checkpoints' / 'stage2-final.ckpt'
    final_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(final_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    add_vae_input_arguments(parser)
    args = parser.parse_args()
    main(args.config, **vae_input_options(args))



