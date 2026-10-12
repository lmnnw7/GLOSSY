import argparse
import os
import random
import sys
from typing import Dict
from typing import List

import torch
import torchio as tio

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from runtime import load_autoencoder
from config_io import add_raw_arguments, raw_options
from dataset.io_utils import to_model_tensor, to_storage_tensor
from dataset.io_utils import collect_volume_files, load_raw_meta, load_scalar_image, save_raw_volume






def psnr_01(x01: torch.Tensor, y01: torch.Tensor) -> float:
    mse = torch.mean((x01 - y01) ** 2)
    return float((-10.0 * torch.log10(mse + 1e-8)).item())


def pick_files(root_dir: str, num_samples: int, seed: int) -> List[str]:
    files = collect_volume_files([root_dir])
    if len(files) == 0:
        raise FileNotFoundError(f"No .raw or .nii.gz found in directory: {root_dir}")

    random.seed(seed)
    random.shuffle(files)
    return files[: min(num_samples, len(files))]


def _default_raw_meta(volume01: torch.Tensor) -> Dict:
    d, h, w = volume01.shape[-3], volume01.shape[-2], volume01.shape[-1]
    return {
        "shape": [int(d), int(h), int(w)],
        "dtype": "uint16",
        "byte_order": "little",
        "value_range": [0, 65535],
    }


def _error_raw_meta(volume01: torch.Tensor, base_meta: Dict) -> Dict:
    d, h, w = volume01.shape[-3], volume01.shape[-2], volume01.shape[-1]
    out = {
        "shape": [int(d), int(h), int(w)],
        "dtype": "float32",
        "byte_order": "little",
        "value_range": [0.0, 1.0],
    }
    if "spacing" in base_meta:
        out["spacing"] = base_meta["spacing"]
    return out


def main(args: argparse.Namespace) -> None:
    raw_meta = raw_options(args)
    if args.num_samples <= 0:
        raise ValueError('--num-samples must be positive')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_autoencoder(args.ae_ckpt, device)
    model.eval()

    selected_files = pick_files(args.data_dir, args.num_samples, args.seed)

    maes = []
    psnrs = []

    print(f"Selected {len(selected_files)} files for reconstruction check")
    for idx, file_path in enumerate(selected_files):
        img = load_scalar_image(file_path, raw_meta=raw_meta)
        x = to_model_tensor(img).unsqueeze(0).to(device)
        if any(s % model.patch_size for s in x.shape[-3:]):
            raise ValueError('Reconstruction input dimensions must be divisible by the VAE patch size')

        with torch.no_grad():
            _, x_recon = model(x, log_volume=True, val=True)

        x = x.squeeze(0).cpu()
        x_recon = x_recon.squeeze(0).cpu()

        x01 = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        recon01 = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)
        err01 = torch.abs(x01 - recon01)

        # Move volumes back to storage axis order before writing any outputs.
        x01_save = to_storage_tensor(x01)
        recon01_save = to_storage_tensor(recon01)
        err01_save = to_storage_tensor(err01)

        mae = float(torch.mean(torch.abs(x - x_recon)).item())
        psnr = psnr_01(x01, recon01)
        maes.append(mae)
        psnrs.append(psnr)

        base_name = os.path.basename(file_path)
        if base_name.lower().endswith(".nii.gz"):
            stem = base_name[:-7]
        elif base_name.lower().endswith(".raw"):
            stem = base_name[:-4]
        else:
            stem = base_name

        input_path = os.path.join(args.output_dir, f"{idx:03d}_{stem}_input.nii.gz")
        recon_path = os.path.join(args.output_dir, f"{idx:03d}_{stem}_recon.nii.gz")
        err_path = os.path.join(args.output_dir, f"{idx:03d}_{stem}_abs_err.nii.gz")

        if args.save_format in {"nii.gz", "both"}:
            tio.ScalarImage(tensor=x01_save, affine=img.affine).save(input_path)
            tio.ScalarImage(tensor=recon01_save, affine=img.affine).save(recon_path)
            tio.ScalarImage(tensor=err01_save, affine=img.affine).save(err_path)

        if args.save_format in {"raw", "both"}:
            if file_path.lower().endswith(".raw"):
                input_meta = load_raw_meta(file_path, overrides=raw_meta)
            else:
                input_meta = _default_raw_meta(x01)

            input_raw_path = os.path.join(args.output_dir, f"{idx:03d}_{stem}_input.raw")
            recon_raw_path = os.path.join(args.output_dir, f"{idx:03d}_{stem}_recon.raw")
            err_raw_path = os.path.join(args.output_dir, f"{idx:03d}_{stem}_abs_err.raw")

            save_raw_volume(x01_save.numpy(), input_raw_path, input_meta)
            save_raw_volume(recon01_save.numpy(), recon_raw_path, input_meta)
            save_raw_volume(err01_save.numpy(), err_raw_path, _error_raw_meta(err01_save, input_meta))

        print(
            f"[{idx + 1}/{len(selected_files)}] {file_path} | "
            f"MAE(-1~1)={mae:.6f}, PSNR(0~1)={psnr:.3f} dB"
        )

    print("\nSummary:")
    print(f"Average MAE(-1~1): {sum(maes) / len(maes):.6f}")
    print(f"Average PSNR(0~1): {sum(psnrs) / len(psnrs):.3f} dB")
    print(f"Saved recon comparison volumes to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-ckpt", type=str, required=True, help="Path to REV_VQVAE AE checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory that contains .nii.gz/.raw files")
    add_raw_arguments(parser)
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save reconstruction results")
    parser.add_argument("--num-samples", type=int, default=16, help="How many volumes to sample for quick check")
    parser.add_argument("--seed", type=int, default=216, help="Random seed for sampling")
    parser.add_argument(
        "--save-format",
        type=str,
        default="raw",
        choices=["nii.gz", "raw", "both"],
        help="Output format for reconstructed volumes",
    )
    args = parser.parse_args()
    main(args)
