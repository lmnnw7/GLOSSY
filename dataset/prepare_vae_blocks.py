"""Prepare non-overlapping uint16 RAW blocks and file partitions for VAE training."""
import argparse
from itertools import product
from pathlib import Path
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.core_subvolume_extraction import extract_subvolumes, source_sha256, write_manifest


def prepare(args):
    source = Path(args.input).expanduser().resolve()
    destination = Path(args.output_root).expanduser().resolve()
    shape = tuple(args.shape)
    storage_order = getattr(args, 'storage_order', 'XYZ')
    if storage_order not in ('XYZ', 'ZYX'):
        raise ValueError('storage-order must be XYZ or ZYX')
    side = args.block_size
    if len(shape) != 3 or side <= 0 or any(size <= 0 or size % side for size in shape):
        raise ValueError('Shape must contain three positive multiples of block-size')
    if args.validation_count <= 0 or args.test_count <= 0:
        raise ValueError('Validation and test counts must be positive')
    xyz_shape = shape if storage_order == 'XYZ' else tuple(reversed(shape))
    positions = list(product(*(range(0, size, side) for size in xyz_shape)))
    if args.validation_count + args.test_count >= len(positions):
        raise ValueError('Leave at least one training block')
    dtype = np.dtype('<u2' if args.byte_order == 'little' else '>u2')
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if source.stat().st_size != expected_bytes:
        raise ValueError(f'RAW size is {source.stat().st_size} bytes; expected {expected_bytes}')
    if destination.exists():
        raise FileExistsError(f'Choose a new output root: {destination}')

    # Reserve a spatial end region before the VAE's deterministic file split.
    test_positions = set(positions[-args.test_count:])
    names = {position: f'block_{position[0]:04d}_{position[1]:04d}_{position[2]:04d}.raw'
             for position in positions}
    pool_names = sorted(names[position] for position in positions if position not in test_positions)
    random.Random(args.seed).shuffle(pool_names)
    validation_names = set(pool_names[-args.validation_count:])
    train_names = set(pool_names[:-args.validation_count])
    folders = {name: destination / name for name in ('all_blocks', 'vae_pool', 'train', 'validation', 'test')}
    for path in folders.values():
        path.mkdir(parents=True)
    extract_subvolumes(source, folders['all_blocks'], shape, (side,) * 3,
                       dtype='uint16', byte_order=args.byte_order, storage_order=storage_order)
    rows = []
    for position in positions:
        x, y, z = position
        name = names[position]
        canonical = folders['all_blocks'] / name
        if position in test_positions:
            split = 'test'
        else:
            split = 'validation' if name in validation_names else 'train'
            shutil.copyfile(canonical, folders['vae_pool'] / name)
        shutil.copyfile(canonical, folders[split] / name)
        rows.append(dict(file=name, split=split, x=x, y=y, z=z, block_size=side))
    write_manifest(destination / 'split.csv', rows)
    digest = source_sha256(source)
    (destination / 'source.txt').write_text(
        f'source={source}\nsha256={digest}\nshape={shape}\ndtype={dtype.str}\nsource_order={storage_order}\naxis_order=XYZ\n'
        f'block_size={side}\nvalidation_count={args.validation_count}\nsplit_seed={args.seed}\n'
        'intensity_normalization=not_applied; declare it when loading\n', encoding='utf-8')
    print(f'Input: {source}; shape={shape}; dtype={dtype.str}; output storage order=XYZ')
    print(f'Blocks: {len(positions)}; train={len(train_names)}; validation={len(validation_names)}; '
          f'test={len(test_positions)}; VAE pool={len(pool_names)}')
    print(f'Output: {destination}')
    print('Set dataset.validation_count and dataset.split_seed to the same values in the VAE YAML.')
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='One headerless uint16 RAW file')
    parser.add_argument('--output-root', required=True, help='New directory; existing directories are rejected')
    parser.add_argument('--shape', nargs=3, type=int, default=[256, 256, 256], help='Source dimensions in storage-order')
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--byte-order', choices=['little', 'big'], default='little')
    parser.add_argument('--storage-order', choices=['XYZ', 'ZYX'], default='XYZ', help='Source shape/bytes convention; output is XYZ')
    parser.add_argument('--validation-count', type=int, default=8)
    parser.add_argument('--test-count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=216)
    return parser


if __name__ == '__main__':
    prepare(build_parser().parse_args())
