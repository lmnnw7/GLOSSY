"""Export REV-VQVAE latents using the original full-volume batching."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchio as tio
from torch.utils.data import DataLoader, Dataset
from utils.config_io import read_data_roots, add_raw_arguments, raw_options
from dataset.io_utils import collect_volume_files, load_scalar_image, to_model_tensor
from ddpm.factory import codebook_digest
from utils.runtime import project_path, resolve_device, load_autoencoder, normalize_latent


class ExportVolumeDataset(Dataset):
    """Load complete source volumes and keep their planned output paths aligned."""
    def __init__(self, planned, raw_meta):
        self.planned = planned
        self.raw_meta = raw_meta

    def __len__(self):
        return len(self.planned)

    def __getitem__(self, index):
        path, output = self.planned[index]
        image = load_scalar_image(path, raw_meta=self.raw_meta)
        return to_model_tensor(image), str(path), str(output)


@torch.inference_mode()
def encode_batch(samples, ae, patch_size, device, quantize):
    if samples.ndim != 5:
        raise ValueError('Expected complete source volumes with shape [B, C, D, H, W]')
    ratios = tuple(int(v) for v in ae.cfg.model.downsample)
    if patch_size <= 0 or any(patch_size % r for r in ratios):
        raise ValueError('patch-size must be a positive multiple of each VAE compression ratio')
    if any(s % patch_size for s in samples.shape[-3:]):
        raise ValueError('Source dimensions must be divisible by patch-size; crop explicitly before export')
    # Match key code: transfer complete volumes, then encode all their patches together.
    samples = samples.to(device)
    latent = ae.patch_encode(samples, patch_size=patch_size, quantize=quantize)
    return normalize_latent(latent, ae).float().cpu()


def generate(args):
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError('batch-size must be positive and num-workers must be nonnegative')
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
    dataset = ExportVolumeDataset(planned, raw_meta)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    for samples, paths, outputs in loader:
        latents = encode_batch(samples, ae, args.patch_size, device, args.quantize)
        for path, output, latent in zip(paths, outputs, latents):
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Latent axes intentionally use model D,H,W; the diffusion loader reads them directly.
            tio.ScalarImage(tensor=latent).save(str(output))
            info = dict(source=path, shape=list(latent.shape), ae_codebook_sha256=digest,
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', required=True, help='Source directory/file or optional legacy JSON mapping')
    parser.add_argument('--class-id', type=int, default=0, help='Class label for a direct source directory/file')
    add_raw_arguments(parser)
    parser.add_argument('--AE-ckpt', required=True)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=2, help='Complete source volumes per batch')
    parser.add_argument('--num-workers', type=int, default=16, help='DataLoader worker processes')
    parser.add_argument('--quantize', action=argparse.BooleanOptionalAction, default=False,
                        help='Continuous latents retain the original export; --quantize selects codebook embeddings')
    parser.add_argument('--output-root', help='Optional output directory, organized by class and source root')
    parser.add_argument('--manifest', help='Optional output JSON mapping; directory input does not need it')
    parser.add_argument('--overwrite', action='store_true')
    return parser


if __name__ == '__main__':
    generate(build_parser().parse_args())
