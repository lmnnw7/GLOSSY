"""Deterministic file splits and bounded VAE patch sampling."""
import random

import torch
from torch.utils.data.dataset import Dataset
import torchio as tio

from dataset.io_utils import collect_volume_files, load_scalar_image
from utils.config_io import image_roots


def split_volume_files(files, split, validation_count=None, seed=216):
    if split not in ('train', 'val'):
        raise ValueError("split must be 'train' or 'val'")
    files = sorted(set(files))
    if len(files) < 2:
        raise ValueError('VAE training requires at least two independent volumes for train/validation splits')
    if validation_count is None:
        validation_count = 40 if len(files) > 40 else max(1, len(files) // 5)
    validation_count = int(validation_count)
    if not 0 < validation_count < len(files):
        raise ValueError('validation_count must leave at least one training and one validation volume')
    random.Random(seed).shuffle(files)
    return files[:-validation_count] if split == 'train' else files[-validation_count:]


class CoreVolumeDataset(Dataset):
    """8x defaults with configurable, patch-aligned crop sizes."""
    stage2_crop_shape = (192, 192, 192)
    validation_crop_shape = (256, 256, 256)

    def __init__(self, root_dir=None, augmentation=False, split='train', stage=1,
                 patch_size=64, validation_count=None, split_seed=216,
                 stage2_crop_shape=None, validation_crop_shape=None,
                 max_patch_attempts=16, raw_meta=None):
        if stage not in (1, 2) or patch_size <= 0 or max_patch_attempts <= 0:
            raise ValueError('Invalid VAE stage, patch size or sampling limit')
        paths = image_roots(root_dir)
        self.raw_meta = raw_meta
        self.file_names = split_volume_files(
            collect_volume_files(paths), split, validation_count, split_seed)
        self.stage, self.split = stage, split
        self.augmentation = bool(augmentation and split == 'train')
        self.patch_size = int(patch_size)
        self.max_patch_attempts = int(max_patch_attempts)
        self.stage2_crop_shape = tuple(stage2_crop_shape or self.stage2_crop_shape)
        self.validation_crop_shape = tuple(validation_crop_shape or self.validation_crop_shape)
        for shape in (self.stage2_crop_shape, self.validation_crop_shape):
            if len(shape) != 3 or any(size < patch_size or size % patch_size for size in shape):
                raise ValueError('Crop sizes must be three positive multiples of patch_size')
        self.patch_sampler = tio.data.UniformSampler(self.patch_size)
        self.randomflip = tio.RandomFlip(axes=(0, 1), flip_probability=0.5)
        print(f'{split}: {len(self.file_names)} volumes; patch size {self.patch_size}')

    def __len__(self):
        return len(self.file_names)

    def _aligned_crop(self, image, target):
        shape = image.shape[1:]
        if any(size < self.patch_size for size in shape):
            raise ValueError(f'Volume dimensions {shape} are smaller than patch_size={self.patch_size}')
        crop_shape = tuple(min(size, limit) // self.patch_size * self.patch_size
                           for size, limit in zip(shape, target))
        if crop_shape != tuple(shape):
            if self.split == 'val':
                margins = []
                for size, target_size in zip(shape, crop_shape):
                    difference = size - target_size
                    margins.extend((difference // 2, difference - difference // 2))
                image = tio.Crop(tuple(margins))(image)
            else:
                sampler = tio.data.UniformSampler(crop_shape)
                image = next(sampler(tio.Subject(image=image)))['image']
        return image

    def __getitem__(self, index):
        path = self.file_names[index]
        image = load_scalar_image(path, raw_meta=self.raw_meta)
        if any(size < self.patch_size for size in image.shape[1:]):
            raise ValueError(f'{path}: every spatial dimension must be at least {self.patch_size}')
        if self.stage == 1 and self.split == 'train':
            for _ in range(self.max_patch_attempts):
                patch = next(self.patch_sampler(tio.Subject(image=image)))['image']
                if patch.data.sum().item() != 0:
                    break
            # A constant zero volume is valid; do not loop forever searching it.
            image = patch
        else:
            target = self.validation_crop_shape if self.split == 'val' else self.stage2_crop_shape
            image = self._aligned_crop(image, target)
        if self.augmentation:
            image = self.randomflip(image)
        tensor = image.data
        if self.augmentation and random.random() > 0.5:
            tensor = torch.rot90(tensor, dims=(1, 2))
        tensor = (tensor * 2 - 1).transpose(1, 3).transpose(2, 3).to(torch.float32)
        if self.split == 'val':
            return {'data': tensor, 'affine': image.affine, 'path': path}
        return {'data': tensor}


class CoreVolumeDataset4x(CoreVolumeDataset):
    """4x defaults share the same split and sampling rules."""
    stage2_crop_shape = (256, 256, 128)
    validation_crop_shape = (256, 256, 128)
