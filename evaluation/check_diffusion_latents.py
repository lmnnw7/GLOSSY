"""Evaluate checkpoint EMA noise loss on fixed latent samples and timesteps."""
import argparse
import csv
from pathlib import Path
import sys

# Resolve the project root from this evaluation entry point.
DEFAULT_PROJECT = Path(__file__).resolve().parents[1]


def main(args):
    project = Path(args.project_root).expanduser().resolve()
    if not (project / 'utils' / 'runtime.py').is_file():
        raise FileNotFoundError('Set --project-root to the GLOSSY repository root')
    sys.path.insert(0, str(project))
    import torch
    from utils.config_io import validate_latent_metadata
    from dataset.Singleres_dataset import Singleres_dataset
    from ddpm.factory import from_checkpoint, codebook_digest
    from utils.runtime import load_autoencoder, resolve_device

    if args.max_samples <= 0:
        raise ValueError('max-samples must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Report exists: {output}; choose a new filename')
    checkpoint = torch.load(args.model_ckpt, map_location='cpu', weights_only=False)
    if not checkpoint.get('model_config') or not checkpoint.get('diffusion_config'):
        raise ValueError('This workflow requires a new checkpoint with saved architecture settings')
    resolution = checkpoint.get('args', {}).get('resolution')
    if resolution is None:
        raise ValueError('Checkpoint does not record its training latent resolution')
    ae = load_autoencoder(args.ae_ckpt, 'cpu')
    digest = codebook_digest(ae)
    if checkpoint.get('ae_codebook_sha256') != digest:
        raise ValueError('Checkpoint and VAE codebooks differ')
    ratios = tuple(int(value) for value in ae.cfg.model.downsample)
    dataset = Singleres_dataset(args.data_path, resolution=resolution, class_id=args.class_id)
    records = [entry for entry in dataset.roots.values() if isinstance(entry, dict)]
    validate_latent_metadata(records + list(dataset.latent_metadata.values()), digest, ratios)
    del ae
    device = resolve_device(args.device)
    model, diffusion, config = from_checkpoint(checkpoint, argparse.Namespace(), device)
    if config.get('cond_dim') is not None:
        raise ValueError('This single-core workflow uses discrete class labels')
    classes = config.get('cond_classes')
    if classes is not None and any(label >= classes for label, _ in dataset.all_files):
        raise ValueError('Dataset class ID exceeds the checkpoint class count')
    del checkpoint
    rows = []
    timesteps = sorted({0, diffusion.num_timesteps // 2, diffusion.num_timesteps - 1})
    with torch.inference_mode():
        for index in range(min(len(dataset), args.max_samples)):
            latent, label, resolution_condition = dataset[index]
            if latent.shape[0] != config['channels']:
                raise ValueError('Latent channel count differs from the diffusion checkpoint')
            # Reset per sample for repeatability across checkpoint comparisons.
            torch.manual_seed(args.seed + index)
            latent = latent.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device) if classes is not None else None
            resolution_condition = resolution_condition.unsqueeze(0).to(device)
            for timestep in timesteps:
                t = torch.full((1,), timestep, dtype=torch.long, device=device)
                noise = torch.randn_like(latent)
                loss = diffusion.p_losses(model, latent, t, y=label, res=resolution_condition, noise=noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite held-out noise loss')
                rows.append(dict(sample=dataset.all_files[index][1], timestep=timestep, loss=float(loss)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample', 'timestep', 'loss'])
        writer.writeheader()
        writer.writerows(rows)
    mean = sum(row['loss'] for row in rows) / len(rows)
    print(f'EMA {diffusion.loss_type} noise loss: {mean:.6f}; samples={len(rows) // len(timesteps)}; '
          f'timesteps={timesteps}; report={output}')
    print('A finite loss checks execution; this is not a reconstruction or generation-quality score.')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default=str(DEFAULT_PROJECT))
    parser.add_argument('--ae-ckpt', required=True)
    parser.add_argument('--model-ckpt', required=True)
    parser.add_argument('--data-path', required=True, help='Held-out latent directory')
    parser.add_argument('--output', required=True, help='New CSV report')
    parser.add_argument('--class-id', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
