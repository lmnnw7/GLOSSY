import argparse
import os
import sys
from typing import List

import torch
import torchio as tio

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from runtime import load_autoencoder, resolve_device, decode_latent
from config_io import image_roots, add_raw_arguments, raw_options
from dataset.io_utils import to_model_tensor, to_storage_tensor
from dataset.io_utils import collect_volume_files, load_raw_meta, load_scalar_image, save_raw_volume, to_nii_gz_path
from sampling_config import add_sampling_arguments, resolve_sampling_options, decode_options


def _pcts(tensor, probs=(0.01, 0.5, 0.99), max_elements: int = 4_000_000):
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return [float("nan") for _ in probs]

    # torch.quantile may fail for very large tensors; sample uniformly when needed.
    if flat.numel() > max_elements:
        step = max(1, flat.numel() // max_elements)
        flat = flat[::step]
        if flat.numel() > max_elements:
            flat = flat[:max_elements]

    flat = flat.to(device="cpu", dtype=torch.float32)
    q = torch.tensor(probs, device=flat.device, dtype=flat.dtype)
    vals = torch.quantile(flat, q)
    return [float(v.item()) for v in vals]


def _collect_inputs(data_path: str) -> List[str]:
    lower = data_path.lower()
    if lower.endswith(".json"):
        return collect_volume_files(image_roots(data_path))

    if os.path.isdir(data_path):
        return collect_volume_files([data_path])

    if lower.endswith(".raw") or lower.endswith(".nii.gz"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input file not found: {data_path}")
        return [data_path]

    raise ValueError("--data-path must be a .json file, a directory, or a single .raw/.nii.gz file")


def _infer_spacing_from_affine(affine) -> List[float]:
    try:
        return [float(abs(affine[0, 0])), float(abs(affine[1, 1])), float(abs(affine[2, 2]))]
    except Exception:
        return [1.0, 1.0, 1.0]


def _build_raw_meta(input_path: str, input_img: tio.ScalarImage, dtype_name: str, byte_order: str,
                    raw_meta=None) -> dict:
    if input_path.lower().endswith(".raw"):
        try:
            meta = load_raw_meta(input_path, overrides=raw_meta)
        except Exception:
            meta = {}
        meta = dict(meta)
    else:
        meta = {}

    meta["dtype"] = dtype_name
    meta["byte_order"] = byte_order
    meta["spacing"] = _infer_spacing_from_affine(input_img.affine)
    meta['spacing_unit'] = 'mm'

    if dtype_name in ["uint16", "u16"]:
        meta["value_range"] = [0, 65535]
    else:
        meta["value_range"] = [0.0, 1.0]
    return meta


def _print_stats(prefix: str, tensor: torch.Tensor) -> None:
    t_min = float(tensor.min().detach().cpu())
    t_max = float(tensor.max().detach().cpu())
    t_mean = float(tensor.mean().detach().cpu())
    p01, p50, p99 = _pcts(tensor)
    print(
        f"{prefix}(min/max/mean)=({t_min:.4f}/{t_max:.4f}/{t_mean:.4f}), "
        f"p01/p50/p99=({p01:.4f}/{p50:.4f}/{p99:.4f})"
    )


def main(args):
    input_raw_meta = raw_options(args)
    resolve_sampling_options(args, include_sampling=False, allow_auto=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    ae = load_autoencoder(args.ae_ckpt, device)
    ae.eval()

    file_paths = _collect_inputs(args.data_path)
    if len(file_paths) == 0:
        raise FileNotFoundError("No .raw or .nii.gz files found for --data-path")

    if args.max_files > 0:
        file_paths = file_paths[: args.max_files]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(file_paths)} files")

    for i, file_path in enumerate(file_paths, start=1):
        img = load_scalar_image(file_path, raw_meta=input_raw_meta)
        image_in = to_model_tensor(img)
        sample = image_in.unsqueeze(0).to(device)

        with torch.no_grad():
            if args.patch_size_encode <= 0 or any(s % args.patch_size_encode for s in sample.shape[-3:]):
                raise ValueError('Input dimensions must be divisible by patch-size-encode')
            latent = ae.patch_encode_sliding(sample, quantize=False, patch_size=args.patch_size_encode)

            mode = args.decode_mode
            if mode == 'auto':
                voxels = int(sample.shape[-1] * sample.shape[-2] * sample.shape[-3])
                mode = 'full' if voxels <= args.auto_decode_threshold_voxels else 'sliding'
            recon = decode_latent(ae, latent, **decode_options(args, mode=mode))

        recon_cpu = recon.detach().squeeze(0).to(torch.float32).cpu()
        inp_cpu = sample.detach().squeeze(0).cpu()

        if args.print_stats:
            print(f"[{i}/{len(file_paths)}] {file_path}")
            _print_stats("  input_norm", inp_cpu)
            _print_stats("  latent", latent.detach())
            _print_stats("  recon", recon_cpu)
            mae = float(torch.mean(torch.abs(recon_cpu - inp_cpu)).item())
            print(f"  mae(input_norm vs recon)={mae:.6f}")

        base_name = os.path.basename(file_path)
        stem = base_name[:-7] if base_name.lower().endswith(".nii.gz") else os.path.splitext(base_name)[0]
        out_base = os.path.join(args.output_dir, f"{stem}_recon")
        # Both output formats use normalized grayscale storage intensities.
        recon_01 = torch.clamp((recon_cpu + 1.0) / 2.0, 0.0, 1.0)
        recon_storage = to_storage_tensor(recon_01)

        if args.save_format in ["nii.gz", "both"]:
            nii_path = to_nii_gz_path(out_base)
            tio.ScalarImage(tensor=recon_storage, affine=img.affine).save(nii_path)

        if args.save_format in ["raw", "both"]:
            raw_path = out_base + ".raw"
            raw_meta = _build_raw_meta(file_path, img, args.sample_raw_dtype, args.sample_raw_byte_order,
                                       raw_meta=input_raw_meta)
            save_raw_volume(recon_storage.numpy(), raw_path, raw_meta)

    print("Done: encode->decode reconstruction finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct AE encode/decode reconstruction for original volumes")
    parser.add_argument("--ae-ckpt", type=str, required=True, help="Path to AE checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Input .json/.raw/.nii.gz or directory")
    add_raw_arguments(parser)
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files; 0 means all")
    parser.add_argument("--patch-size-encode", type=int, default=64)

    add_sampling_arguments(parser, include_sampling=False, allow_auto=True)
    parser.add_argument("--auto-decode-threshold-voxels", type=int, default=64 * 64 * 64)

    parser.add_argument("--save-format", type=str, default="raw", choices=["nii.gz", "raw", "both"])
    parser.add_argument("--sample-raw-dtype", type=str, default="uint16", choices=["uint16", "u16", "float32", "f32"])
    parser.add_argument("--sample-raw-byte-order", type=str, default="little", choices=["little", "big"])
    parser.add_argument("--print-stats", action="store_true")

    main(parser.parse_args())
