"""Compare two VAE checkpoints on identical, fixed held-out volumes or crops."""
import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dataset.io_utils import collect_volume_files, load_scalar_image, to_model_tensor
from utils.runtime import load_autoencoder, resolve_device
from utils.config_io import add_raw_arguments, raw_options


def reconstruction_metrics(source, reconstruction, pore_threshold=None):
    if source.shape != reconstruction.shape or not torch.isfinite(reconstruction).all():
        raise ValueError('Invalid reconstruction shape or non-finite values')
    source, reconstruction = source.float(), reconstruction.float()
    error = source - reconstruction
    x01, y01 = ((source + 1) / 2).clamp(0, 1), ((reconstruction + 1) / 2).clamp(0, 1)
    mse01 = (x01 - y01).square().mean()
    result = dict(mae=float(error.abs().mean()), rmse=float(error.square().mean().sqrt()),
                  psnr_db=float(-10 * torch.log10(mse01 + 1e-8)))
    if pore_threshold is not None:
        actual = float((x01 <= pore_threshold).float().mean())
        predicted = float((y01 <= pore_threshold).float().mean())
        result.update(porosity=predicted, target_porosity=actual,
                      porosity_absolute_error=abs(predicted - actual))
    return result


def evaluate_checkpoint(checkpoint, files, device, crop_size, pore_threshold, raw_meta=None):
    model = load_autoencoder(checkpoint, device)
    scores = []
    with torch.inference_mode():
        for filename in files:
            source = to_model_tensor(load_scalar_image(filename, raw_meta=raw_meta)).unsqueeze(0)
            if crop_size:
                if any(size < target for size, target in zip(source.shape[-3:], crop_size)):
                    raise ValueError(f'{filename}: volume is smaller than requested crop')
                slices = tuple(slice((size - target) // 2, (size - target) // 2 + target)
                               for size, target in zip(source.shape[-3:], crop_size))
                source = source[(slice(None), slice(None)) + slices]
            source = source.to(device)
            _, reconstruction = model(source, log_volume=True, val=True)
            scores.append(dict(path=filename, shape=list(source.shape[-3:]),
                               **reconstruction_metrics(source, reconstruction, pore_threshold)))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    mean = {key: sum(row[key] for row in scores) / len(scores)
            for key in scores[0] if key not in ('path', 'shape')}
    return dict(checkpoint=str(Path(checkpoint).resolve()), mean=mean, samples=scores)


def main(args):
    raw_meta = raw_options(args)
    if args.num_samples <= 0 or (args.crop_size and min(args.crop_size) <= 0):
        raise ValueError('Sample count and crop dimensions must be positive')
    if args.pore_threshold is not None and not 0 <= args.pore_threshold <= 1:
        raise ValueError('pore_threshold must be in [0,1]; pores use intensity <= threshold')
    if args.max_mae_increase < 0 or args.max_psnr_drop < 0:
        raise ValueError('Comparison tolerances must be nonnegative')
    files = collect_volume_files([args.data_dir])
    if not files:
        raise FileNotFoundError('No held-out RAW/NIfTI volumes found')
    random.Random(args.seed).shuffle(files)
    files = files[:args.num_samples]
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    baseline = evaluate_checkpoint(args.baseline_ckpt, files, device, args.crop_size, args.pore_threshold, raw_meta)
    candidate = evaluate_checkpoint(args.candidate_ckpt, files, device, args.crop_size, args.pore_threshold, raw_meta)
    mae_increase = candidate['mean']['mae'] - baseline['mean']['mae']
    psnr_drop = baseline['mean']['psnr_db'] - candidate['mean']['psnr_db']
    passed = mae_increase <= args.max_mae_increase and psnr_drop <= args.max_psnr_drop
    report = dict(seed=args.seed, crop_size=args.crop_size, pore_threshold=args.pore_threshold,
                  baseline=baseline, candidate=candidate, mean_mae_increase=mae_increase,
                  mean_psnr_drop_db=psnr_drop, tolerances=dict(mae=args.max_mae_increase, psnr_db=args.max_psnr_drop),
                  passed_mae_psnr=passed)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f'Report already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('baseline', 'candidate')}, indent=2))
    if args.check and not passed:
        raise SystemExit('Candidate exceeds the requested MAE/PSNR tolerances; see the paired report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--candidate-ckpt', required=True)
    parser.add_argument('--data-dir', required=True, help='Independent held-out volumes, outside training data')
    add_raw_arguments(parser)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-samples', type=int, default=16)
    parser.add_argument('--crop-size', type=int, nargs=3, help='Fixed center crop in model [D,H,W] axes')
    parser.add_argument('--pore-threshold', type=float, help='Optional pores: intensity <= threshold in [0,1]')
    parser.add_argument('--seed', type=int, default=216)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-mae-increase', type=float, default=0)
    parser.add_argument('--max-psnr-drop', type=float, default=0)
    parser.add_argument('--check', action='store_true', help='Exit with failure when mean MAE/PSNR exceeds tolerances')
    main(parser.parse_args())
