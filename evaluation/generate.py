"""Generate grayscale digital cores from an MSP-DiffNet checkpoint."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from ddpm.factory import add_model_arguments, from_checkpoint, codebook_digest
from runtime import resolve_device, load_autoencoder, decode_latent, denormalize_latent
from evaluation.volume_output import save_volume
from sampling_config import (add_sampling_arguments, resolve_sampling_options,
                             decode_options, validate_decode_settings)


def main(args):
    resolve_sampling_options(args)
    if args.num_samples <= 0 or min(args.volume_size) <= 0:
        raise ValueError('num-samples and volume-size must be positive')
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ae = load_autoencoder(args.AE_ckpt, device)
    checkpoint = torch.load(args.model_ckpt, map_location='cpu')
    digest = codebook_digest(ae)
    if checkpoint.get('ae_codebook_sha256', digest) != digest:
        raise ValueError('Generation requires the VAE used to encode the diffusion training latents')
    model, diffusion, cfg = from_checkpoint(checkpoint, args, device)
    if cfg['channels'] != ae.embedding_dim:
        raise ValueError('Diffusion channels and VAE embedding dimension differ')
    classes = cfg.get('cond_classes')
    if cfg.get('cond_dim') is not None:
        if args.condition is None or len(args.condition) != cfg['cond_dim']:
            raise ValueError('Continuous checkpoints require --condition with cond_dim numeric values')
        y = torch.tensor([args.condition], device=device, dtype=torch.float32)
    elif classes is not None:
        if not 0 <= args.class_id < classes:
            raise ValueError(f'class-id must lie in [0, {classes-1}]')
        y = torch.tensor([args.class_id], device=device, dtype=torch.long)
    else:
        y = None
    ratios = tuple(int(v) for v in ae.cfg.model.downsample)
    if any(n % r for n, r in zip(args.volume_size, ratios)):
        raise ValueError('Output voxel dimensions must be divisible by the VAE compression ratios')
    resolution = tuple(n // r for n, r in zip(args.volume_size, ratios))
    if any(n % d for n, d in zip(resolution, model.spatial_divisor)):
        raise ValueError(f'Latent size {resolution} must be divisible by {model.spatial_divisor}')
    if args.sampling_strategy == 'ddim' and not 1 <= args.ddim_steps <= diffusion.num_timesteps:
        raise ValueError('ddim-steps must lie in [1, checkpoint timesteps]')
    decoding = decode_options(args)
    effective_patch = validate_decode_settings((1, cfg['channels'], *resolution), ratios, **decoding)
    out = Path(args.output_dir) / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    for index in range(args.num_samples):
        base = out / f'core_class{args.class_id}_{"x".join(map(str, args.volume_size))}_{index:04d}'
        if any(Path(str(base)+suffix).exists() for suffix in ('.raw', '.nii.gz')):
            raise FileExistsError(f'Output already exists: {base}; choose another seed or output directory')
        with torch.inference_mode():
            z = torch.randn(1, cfg['channels'], *resolution, device=device)
            sampled = diffusion.sample(model, z, y=y, res=z.new_tensor(resolution)/64.,
                                       strategy=args.sampling_strategy,
                                       ddim_steps=args.ddim_steps, eta=args.eta)
            volume = decode_latent(ae, denormalize_latent(sampled, ae), **decoding)
            save_volume(volume[0], base, args.sample_save_format,
                        args.sample_raw_dtype, args.sample_raw_byte_order, args.spacing)
        info = dict(seed=args.seed, sample_index=index, class_id=args.class_id,
                    condition=args.condition, output_model_size=list(args.volume_size),
                    latent_size=list(resolution), downsample=list(ratios),
                    decode_mode=args.decode_mode, sampling_strategy=args.sampling_strategy,
                    decode_settings=dict(**decoding, effective_patch_size=effective_patch),
                    ddim_steps=args.ddim_steps, eta=args.eta,
                    ae_codebook_sha256=digest, ae_checkpoint=str(Path(args.AE_ckpt).resolve()),
                    diffusion_checkpoint=str(Path(args.model_ckpt).resolve()))
        Path(str(base) + '.generation.json').write_text(json.dumps(info, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Saved {base} (model D,H,W={args.volume_size})')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument('--AE-ckpt', required=True)
    parser.add_argument('--model-ckpt', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--volume-size', nargs=3, type=int, default=[128, 128, 128], help='Output voxels in model D H W order')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--class-id', type=int, default=0)
    parser.add_argument('--condition', nargs='+', type=float)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    add_sampling_arguments(parser)
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.03, 1.03, 1.03], help='Storage X Y Z spacing in micrometres')
    parser.add_argument('--sample-save-format', choices=['raw', 'nii.gz', 'both'], default='raw')
    parser.add_argument('--sample-raw-dtype', choices=['uint16', 'u16', 'float32', 'f32'], default='uint16')
    parser.add_argument('--sample-raw-byte-order', choices=['little', 'big'], default='little')
    parser.add_argument('--seed', type=int, default=19)
    return parser


def cli():
    main(build_parser().parse_args())


if __name__ == '__main__':
    cli()
