"""Export REV-VQVAE latents in patches without placing the whole core on the GPU."""
import argparse
from itertools import product
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchio as tio
from utils.config_io import read_data_roots, add_raw_arguments, raw_options
from dataset.io_utils import collect_volume_files, load_scalar_image, to_model_tensor
from ddpm.factory import codebook_digest
from utils.runtime import project_path, resolve_device, load_autoencoder, normalize_latent


@torch.inference_mode()
def encode_volume(sample, ae, patch_size, patch_batch_size, device, quantize):
    ratios = tuple(int(v) for v in ae.cfg.model.downsample)
    if patch_size <= 0 or any(patch_size % r for r in ratios):
        raise ValueError('patch-size must be a positive multiple of each VAE compression ratio')
    if any(s % patch_size for s in sample.shape[-3:]):
        raise ValueError('Source dimensions must be divisible by patch-size; crop explicitly before export')
    latent_shape = tuple(s // r for s, r in zip(sample.shape[-3:], ratios))
    latent = torch.empty(ae.embedding_dim, *latent_shape, dtype=torch.float32)
    positions = list(product(*(range(0, s, patch_size) for s in sample.shape[-3:])))
    for offset in range(0, len(positions), patch_batch_size):
        block_positions = positions[offset:offset+patch_batch_size]
        patches = torch.stack([sample[:, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                               for d, h, w in block_positions]).to(device)
        output = ae.encode(patches, include_embeddings=True, quantize=quantize)
        z = output[0] if quantize else output
        z = normalize_latent(z, ae).float().cpu()
        for index, (d, h, w) in enumerate(block_positions):
            starts = (d//ratios[0], h//ratios[1], w//ratios[2])
            ld, lh, lw = z.shape[-3:]
            zd, zh, zw = starts
            latent[:, zd:zd+ld, zh:zh+lh, zw:zw+lw] = z[index]
    return latent


def generate(args):
    if args.patch_batch_size <= 0:
        raise ValueError('patch-batch-size must be positive')
    raw_meta = raw_options(args)
    roots = read_data_roots(args.data_path, numeric_labels=True, direct_kind='images',
                            class_id=getattr(args, 'class_id', 0))
    device = resolve_device(args.device)
    ae = load_autoencoder(args.AE_ckpt, device)
    digest = codebook_digest(ae)
    manifest = {}
    planned, occupied = [], set()
    for label, entry in roots.items():
        if int(label) < 0:
            raise ValueError('Class IDs must be nonnegative integers')
        sources = entry.get('images') if isinstance(entry, dict) else entry
        sources = sources if isinstance(sources, list) else [sources]
        output_roots = []
        for root_index, source in enumerate(sources):
            source = project_path(source)
            output_root = (Path(args.output_root).resolve() / str(label) / str(root_index)
                           if args.output_root else Path(str(source if source.is_dir() else source.parent)+'_latents'))
            output_roots.append(str(output_root.resolve()))
            files = collect_volume_files([source])
            if not files:
                raise FileNotFoundError(f'No source images found: {source}')
            for path in files:
                name = Path(path).name
                stem = name[:-7] if name.lower().endswith('.nii.gz') else Path(name).stem
                output = output_root / (stem + '.nii.gz')
                if output in occupied:
                    raise ValueError(f'Source names produce duplicate latent output: {output}')
                if output.exists() and not args.overwrite:
                    raise FileExistsError(f'Latent already exists: {output}; use --overwrite to re-encode')
                occupied.add(output)
                planned.append((path, output))
        manifest[label] = dict(images=[str(project_path(s)) for s in sources],
                               latents=output_roots, downsample=list(ae.cfg.model.downsample),
                               ae_codebook_sha256=digest, quantized=args.quantize,
                               latent_normalization='codebook_minmax', latent_axis_order='CDHW')
    for path, output in planned:
        image = load_scalar_image(path, raw_meta=raw_meta)
        sample = to_model_tensor(image)
        latent = encode_volume(sample, ae, args.patch_size, args.patch_batch_size, device, args.quantize)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Latent axes intentionally use model D,H,W; the diffusion loader reads them directly.
        tio.ScalarImage(tensor=latent).save(str(output))
        info = dict(source=str(path), shape=list(latent.shape), ae_codebook_sha256=digest,
                    quantized=args.quantize, latent_axis_order='CDHW',
                    downsample=list(ae.cfg.model.downsample), latent_normalization='codebook_minmax')
        torch.save(info, str(output) + '.meta.pt')
        print(f'{path} -> {output}, shape={tuple(latent.shape)}')
    if args.manifest or project_path(args.data_path).suffix.lower() == '.json':
        manifest_path = project_path(args.manifest) if args.manifest else project_path(args.data_path).with_name('training_latents.json')
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Optional training manifest: {manifest_path.resolve()}')
    else:
        print('Pass the exported latent directory directly to diffusion --data-path.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True, help='Source directory/file or optional legacy JSON mapping')
    parser.add_argument('--class-id', type=int, default=0, help='Class label for a direct source directory/file')
    add_raw_arguments(parser)
    parser.add_argument('--AE-ckpt', required=True)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--patch-batch-size', '--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=16, help='Compatibility argument; export iterates volumes serially')
    parser.add_argument('--quantize', action=argparse.BooleanOptionalAction, default=False,
                        help='Continuous latents retain the original export; --quantize selects codebook embeddings')
    parser.add_argument('--output-root', help='Optional output directory, organized by class and source root')
    parser.add_argument('--manifest', help='Optional output JSON mapping; directory input does not need it')
    parser.add_argument('--overwrite', action='store_true')
    generate(parser.parse_args())
