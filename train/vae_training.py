"""Shared, checkpoint-aware training setup for both REV-VQVAE stages."""
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from dataset.core_volume import CoreVolumeDataset, CoreVolumeDataset4x
from runtime import project_path
from train.callbacks import VolumeLogger


def load_config(path, stage, data_path=None, results_dir=None, checkpoint_path=None, raw_meta=None):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(project_path(path))
    if data_path is not None:
        cfg.dataset.root_dir = data_path
    if results_dir is not None:
        cfg.model.default_root_dir = results_dir
    if checkpoint_path is not None:
        cfg.model.resume_from_checkpoint = checkpoint_path
    if raw_meta is not None:
        cfg.dataset.raw_meta = raw_meta
    if int(cfg.model.stage) != stage:
        raise ValueError(f'This entry point requires model.stage={stage}')
    ratios = tuple(int(value) for value in cfg.model.downsample)
    if len(ratios) != 3 or len(set(ratios)) != 1 or ratios[0] not in (4, 8):
        raise ValueError('The supplied VAE supports isotropic [4,4,4] or [8,8,8] compression')
    if cfg.dataset.patch_size <= 0 or cfg.dataset.patch_size % ratios[0]:
        raise ValueError('patch_size must be a positive multiple of the compression ratio')
    if cfg.model.n_hiddens <= 0 or cfg.model.n_codes <= 0 or cfg.model.embedding_dim <= 0:
        raise ValueError('Model channel and codebook sizes must be positive')
    if cfg.model.norm_type == 'group' and cfg.model.n_hiddens % cfg.model.num_groups:
        raise ValueError('num_groups must divide n_hiddens')
    cfg.dataset.root_dir = str(project_path(cfg.dataset.root_dir))
    cfg.model.default_root_dir = str(project_path(cfg.model.default_root_dir))
    checkpoint = cfg.model.resume_from_checkpoint
    if checkpoint:
        cfg.model.resume_from_checkpoint = str(project_path(checkpoint))
        if not Path(cfg.model.resume_from_checkpoint).is_file():
            raise FileNotFoundError(f'VAE checkpoint does not exist: {cfg.model.resume_from_checkpoint}')
    elif stage == 2:
        raise ValueError('Stage 2 requires a compatible stage-1 or stage-2 checkpoint')
    return cfg


def checkpoint_stage(checkpoint):
    if 'vae_stage' in checkpoint:
        return int(checkpoint['vae_stage'])
    saved_cfg = checkpoint.get('hyper_parameters', {}).get('cfg', {})
    model_cfg = saved_cfg.get('model', {}) if hasattr(saved_cfg, 'get') else {}
    return int(model_cfg.get('stage', 1))


def load_training_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected a Lightning VAE checkpoint with state_dict')
    return checkpoint


def build_data_loaders(cfg):
    from omegaconf import OmegaConf
    raw_meta = getattr(cfg.dataset, 'raw_meta', None)
    if OmegaConf.is_config(raw_meta):
        raw_meta = OmegaConf.to_container(raw_meta, resolve=True)
    dataset_type = CoreVolumeDataset4x if cfg.model.downsample[0] == 4 else CoreVolumeDataset
    kwargs = dict(
        root_dir=cfg.dataset.root_dir, patch_size=cfg.dataset.patch_size,
        validation_count=getattr(cfg.dataset, 'validation_count', None),
        split_seed=getattr(cfg.dataset, 'split_seed', 216),
        stage2_crop_shape=getattr(cfg.dataset, 'stage2_crop_shape', None),
        validation_crop_shape=getattr(cfg.dataset, 'validation_crop_shape', None), raw_meta=raw_meta)
    train_dataset = dataset_type(augmentation=True, split='train', stage=cfg.model.stage, **kwargs)
    val_dataset = dataset_type(augmentation=False, split='val', stage=cfg.model.stage, **kwargs)
    common = dict(num_workers=cfg.model.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.model.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.model.seed), **common)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        generator=torch.Generator().manual_seed(cfg.model.seed + 1), **common)
    return train_loader, val_loader


def print_training_settings(cfg):
    print(f'Learning rate {cfg.model.lr:.2e}; batch size {cfg.model.batch_size}; devices {cfg.model.gpus}')


def build_trainer(cfg, check_val_every_n_epoch, extra_callback_factory=None, **overrides):
    callbacks = [
        ModelCheckpoint(monitor='val/recon_loss', save_top_k=3, save_last=True,
                        mode='min', filename='best-{epoch}-{step}', auto_insert_metric_name=False),
        ModelCheckpoint(every_n_train_steps=3000, save_top_k=-1,
                        filename='periodic-{epoch}-{step}', auto_insert_metric_name=False),
        ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                        filename='archive-{epoch}-{step}', auto_insert_metric_name=False),
        VolumeLogger(batch_frequency=getattr(cfg.model, 'volume_log_frequency', 1500),
                     max_volumes=getattr(cfg.model, 'log_max_volumes', 4), clamp=True),
    ]
    if extra_callback_factory is not None:
        callbacks.append(extra_callback_factory())
    devices = cfg.model.gpus
    if not isinstance(devices, int):
        devices = list(devices)
    count = len(devices) if isinstance(devices, list) else devices
    if count <= 0:
        raise ValueError('Configure a positive number of training devices')
    accelerator = getattr(cfg.model, 'accelerator', 'gpu')
    strategy = DDPStrategy(
        find_unused_parameters=True,
        process_group_backend='gloo' if os.name == 'nt' or accelerator == 'cpu' else 'nccl') if count > 1 else 'auto'
    precision = cfg.model.precision
    if str(precision) in ('bf16', '16'):
        precision = 'bf16-mixed' if str(precision) == 'bf16' else '16-mixed'
    logger = TensorBoardLogger(cfg.model.default_root_dir, name='REV_VQVAE')
    settings = dict(
        accelerator=accelerator, devices=devices,
        default_root_dir=cfg.model.default_root_dir, strategy=strategy,
        callbacks=callbacks, max_steps=cfg.model.max_steps, max_epochs=cfg.model.max_epochs,
        precision=precision, check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=2, logger=logger,
    )
    settings.update(overrides)
    return pl.Trainer(**settings)
