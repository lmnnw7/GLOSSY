"""Extract RAW blocks/crops using the slab slicing from the original notebook.

Source dimensions follow storage_order. Outputs always use C-order XYZ storage,
the convention expected by the GLOSSY image loader. Intensities are unchanged.
"""
import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np


def _source(path, shape, dtype, byte_order, storage_order):
    path = Path(path).expanduser().resolve()
    shape = tuple(int(size) for size in shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('shape must contain three positive source dimensions')
    if storage_order not in ('XYZ', 'ZYX') or byte_order not in ('little', 'big'):
        raise ValueError('Use storage_order XYZ/ZYX and byte_order little/big')
    dtype = np.dtype(dtype).newbyteorder('<' if byte_order == 'little' else '>')
    if dtype.kind not in 'uif':
        raise ValueError('RAW dtype must be an integer or floating-point type')
    expected = int(np.prod(shape)) * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f'{path}: {path.stat().st_size} bytes, expected {expected} for {shape}/{dtype}')
    return path, shape, dtype


def _xyz(values, storage_order):
    return tuple(values) if storage_order == 'XYZ' else tuple(reversed(values))


def extract_subvolumes(input_path, output_dir, shape, block_shape=(64, 64, 64),
                       stride=None, dtype='uint16', byte_order='little', storage_order='XYZ'):
    """Read one source-axis slab at a time; slice complete blocks as in the notebook."""
    source, shape, dtype = _source(input_path, shape, dtype, byte_order, storage_order)
    block_shape = tuple(int(size) for size in block_shape)
    stride = tuple(int(size) for size in (stride or block_shape))
    if len(block_shape) != 3 or len(stride) != 3 or min((*block_shape, *stride)) <= 0:
        raise ValueError('block_shape and stride must each contain three positive sizes')
    if any(block > size for block, size in zip(block_shape, shape)):
        raise ValueError('A block is larger than the source volume')
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Choose a new or empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    positions = [range(0, size - block + 1, step) for size, block, step in zip(shape, block_shape, stride)]
    rows = []
    with source.open('rb') as stream:
        for first in positions[0]:
            count = block_shape[0] * shape[1] * shape[2] * dtype.itemsize
            stream.seek(first * shape[1] * shape[2] * dtype.itemsize)
            slab = stream.read(count)
            if len(slab) != count:
                raise ValueError(f'Incomplete source slab: {len(slab)} bytes, expected {count}')
            data = np.frombuffer(slab, dtype=dtype).reshape((block_shape[0], shape[1], shape[2]))
            for second, third in product(positions[1], positions[2]):
                block = data[:, second:second + block_shape[1], third:third + block_shape[2]]
                if storage_order == 'ZYX':
                    block = block.transpose(2, 1, 0)
                x, y, z = _xyz((first, second, third), storage_order)
                name = f'block_{x:04d}_{y:04d}_{z:04d}.raw'
                block.tofile(output / name)
                sx, sy, sz = block.shape
                rows.append(dict(file=name, x=x, y=y, z=z, size_x=sx, size_y=sy, size_z=sz))
    return rows


def extract_crop(input_path, output_path, shape, start, crop_shape,
                 dtype='uint16', byte_order='little', storage_order='XYZ'):
    """Crop the original array bounds without reading the complete source into RAM."""
    source, shape, dtype = _source(input_path, shape, dtype, byte_order, storage_order)
    start = tuple(int(value) for value in start)
    crop_shape = tuple(int(value) for value in crop_shape)
    if len(start) != 3 or len(crop_shape) != 3 or min(start) < 0 or min(crop_shape) <= 0:
        raise ValueError('start must be nonnegative; crop_shape must be positive; both need three values')
    if any(begin + size > limit for begin, size, limit in zip(start, crop_shape, shape)):
        raise ValueError('Crop extends beyond the source bounds')
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    volume = np.memmap(source, dtype=dtype, mode='r', shape=shape, order='C')
    crop = volume[tuple(slice(begin, begin + size) for begin, size in zip(start, crop_shape))]
    if storage_order == 'ZYX':
        crop = crop.transpose(2, 1, 0)
    crop.tofile(output)
    output_shape = tuple(crop.shape)
    del crop, volume
    return output_shape


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, help='Block directory or one crop RAW path')
    parser.add_argument('--shape', nargs=3, type=int, required=True, help='Source dimensions in storage-order')
    parser.add_argument('--storage-order', choices=['XYZ', 'ZYX'], default='XYZ')
    parser.add_argument('--dtype', choices=['uint16', 'int16', 'float32', 'uint8', 'int8'], default='uint16')
    parser.add_argument('--byte-order', choices=['little', 'big'], default='little')
    parser.add_argument('--block-shape', nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument('--stride', nargs=3, type=int)
    parser.add_argument('--crop-start', nargs=3, type=int)
    parser.add_argument('--crop-shape', nargs=3, type=int)
    return parser


def main(args):
    options = dict(dtype=args.dtype, byte_order=args.byte_order, storage_order=args.storage_order)
    if (args.crop_start is None) != (args.crop_shape is None):
        raise ValueError('Specify both crop-start and crop-shape')
    if args.crop_shape is not None:
        shape = extract_crop(args.input, args.output, args.shape, args.crop_start, args.crop_shape, **options)
        print(f'Saved XYZ crop: {args.output}; shape={shape}')
    else:
        rows = extract_subvolumes(args.input, args.output, args.shape, args.block_shape, args.stride, **options)
        with (Path(args.output) / 'blocks.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'Saved {len(rows)} XYZ blocks in {args.output}; original intensities and byte order retained')


if __name__ == '__main__':
    main(build_parser().parse_args())
