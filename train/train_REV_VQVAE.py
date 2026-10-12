"""Train or resume REV-VQVAE stage 1."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytorch_lightning as pl
from AutoEncoder.model.REV_VQVAE import REV_VQVAE
from config_io import add_vae_input_arguments, vae_input_options
from train.vae_training import (build_data_loaders, build_trainer, print_training_settings,
                                load_config, load_training_checkpoint, checkpoint_stage)


def main(cfg_path, **input_options):
    cfg = load_config(cfg_path, stage=1, **input_options)
    checkpoint_path = cfg.model.resume_from_checkpoint
    if checkpoint_path and checkpoint_stage(load_training_checkpoint(checkpoint_path)) != 1:
        raise ValueError('Stage 1 can only resume a stage-1 checkpoint')
    pl.seed_everything(cfg.model.seed, workers=True)
    train_loader, val_loader = build_data_loaders(cfg)
    print_training_settings(cfg)
    model = REV_VQVAE(cfg, initialize_perceptual=not bool(checkpoint_path))
    trainer = build_trainer(cfg, check_val_every_n_epoch=2)
    trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path or None)
    final_path = Path(cfg.model.default_root_dir) / 'checkpoints' / 'stage1-final.ckpt'
    final_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(final_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    add_vae_input_arguments(parser)
    args = parser.parse_args()
    main(args.config, **vae_input_options(args))



