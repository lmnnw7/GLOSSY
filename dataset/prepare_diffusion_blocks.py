"""Prepare equal-sized uint16 RAW blocks with half-block strides for diffusion."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.core_subvolume_extraction import extract_subvolumes, source_sha256, write_manifest


def prepare_diffusion_blocks(input_path, output_dir, shape=(256, 256, 256),
                             block_size=64, byte_order='little', storage_order='XYZ'):
    """Cut one parent core using the original notebook's slab/slice implementation.

    Every axis advances by block_size // 2; only complete blocks are saved.
    Overlapping blocks belong to one dataset. Split independent parent cores
    before extraction when preparing validation or test data.
    """
    if not isinstance(block_size, int) or block_size <= 0 or block_size % 2:
        raise ValueError('block_size must be a positive even integer for half-block strides')
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Choose a new output directory: {destination}')
    stride = block_size // 2
    rows = extract_subvolumes(source, destination, shape, (block_size,) * 3,
                             (stride,) * 3, dtype='uint16', byte_order=byte_order,
                             storage_order=storage_order)
    write_manifest(destination / 'blocks.csv', rows)
    dtype = '<u2' if byte_order == 'little' else '>u2'
    (destination / 'source.txt').write_text(
        f'source={source}\nsha256={source_sha256(source)}\nshape={tuple(shape)}\n'
        f'dtype={dtype}\nsource_order={storage_order}\naxis_order=XYZ\n'
        f'block_size={block_size}\nstride={stride}\nblock_count={len(rows)}\n'
        'intensity_normalization=not_applied; declare it when loading\n', encoding='utf-8')
    print(f'Saved {len(rows)} XYZ blocks in {destination}; block_size={block_size}; stride={stride}')
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='One headerless uint16 RAW parent core')
    parser.add_argument('--output-root', required=True, help='New directory for one dataset')
    parser.add_argument('--shape', nargs=3, type=int, default=[256, 256, 256],
                        help='Source dimensions in storage-order')
    parser.add_argument('--block-size', type=int, default=64, help='Even side length; stride is half this size')
    parser.add_argument('--byte-order', choices=['little', 'big'], default='little')
    parser.add_argument('--storage-order', choices=['XYZ', 'ZYX'], default='XYZ',
                        help='Source shape/bytes convention; output is XYZ')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    prepare_diffusion_blocks(args.input, args.output_root, args.shape, args.block_size,
                             args.byte_order, args.storage_order)
