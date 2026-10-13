import glob
import json
import os
from typing import Iterable, List

import numpy as np
import torch
import torchio as tio
from utils.config_io import load_json_object, validate_raw_meta, RAW_BYTE_ORDERS


SUPPORTED_VOLUME_PATTERNS = ("*.nii.gz", "*.raw")


def _collect_files_in_dir(root_dir: str) -> List[str]:
    files: List[str] = []
    for pattern in SUPPORTED_VOLUME_PATTERNS:
        files.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    files = sorted(files)
    return files


def collect_volume_files(paths: Iterable[str]) -> List[str]:
    files: List[str] = []
    for path in paths:
        path = os.fspath(path)
        if os.path.isfile(path) and path.lower().endswith(('.raw', '.nii.gz')):
            files.append(path)
        elif os.path.isdir(path):
            files.extend(_collect_files_in_dir(path))
        else:
            raise FileNotFoundError(f'Volume path does not exist: {path}')
    return sorted(set(files))


def _resolve_raw_meta_candidates(raw_path: str) -> List[str]:
    raw_dir = os.path.dirname(raw_path)
    candidates = [
        os.path.join(raw_dir, "raw_meta.json"),
        os.path.join(raw_dir, "_raw_meta.json"),
        f"{os.path.splitext(raw_path)[0]}.json",
        f"{raw_path}.json",
    ]
    return candidates


def _load_raw_meta(raw_path: str, overrides=None) -> dict:
    if overrides is not None and 'shape' in overrides:
        return validate_raw_meta(overrides)
    candidates = _resolve_raw_meta_candidates(raw_path)
    existing = [c for c in candidates if os.path.exists(c)]

    if len(existing) == 0 and overrides is None:
        raise FileNotFoundError(
            "RAW dimensions are required. Pass --raw-shape X Y Z, or create: "
            + " or ".join(candidates)
        )

    # Load defaults first, then apply per-file overrides.
    meta: dict = {}
    for meta_path in existing:
        meta.update(load_json_object(os.path.abspath(meta_path)))
    meta.update(overrides or {})
    return validate_raw_meta(meta)


def load_raw_meta(raw_path: str, overrides=None) -> dict:
    if not raw_path.lower().endswith(".raw"):
        raise ValueError(f"load_raw_meta expects a .raw path, got: {raw_path}")
    return _load_raw_meta(raw_path, overrides)


def _build_affine(spacing: Iterable[float] | None) -> np.ndarray:
    if spacing is None:
        return np.eye(4, dtype=np.float32)

    spacing = list(spacing)
    if len(spacing) != 3:
        raise ValueError(f"spacing must have 3 values, got {spacing}")

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = float(spacing[0])
    affine[1, 1] = float(spacing[1])
    affine[2, 2] = float(spacing[2])
    return affine


def _normalize_raw(raw_volume: np.ndarray, meta: dict) -> np.ndarray:
    raw_volume = raw_volume.astype(np.float32)
    if not np.isfinite(raw_volume).all():
        raise ValueError('RAW input contains non-finite values')

    if "value_range" in meta:
        min_v, max_v = meta["value_range"]
        min_v = float(min_v)
        max_v = float(max_v)
        if max_v <= min_v:
            raise ValueError(f"Invalid value_range: {meta['value_range']}")
        raw_volume = np.clip(raw_volume, min_v, max_v)
        return (raw_volume - min_v) / (max_v - min_v)

    if 'scale' in meta:
        scale = float(meta['scale'])
        if scale <= 0:
            raise ValueError('RAW scale must be positive')
        return raw_volume / scale
    dtype = _to_raw_dtype(str(meta.get('dtype', 'uint16')).lower(), 'little')
    if np.issubdtype(dtype, np.unsignedinteger):
        return raw_volume / np.iinfo(dtype).max
    if np.issubdtype(dtype, np.signedinteger):
        raise ValueError('Signed RAW inputs require an explicit value_range')
    if not np.isfinite(raw_volume).all() or raw_volume.min() < 0 or raw_volume.max() > 1:
        raise ValueError('Floating RAW inputs outside [0, 1] require value_range or scale')
    return raw_volume


def _to_raw_dtype(dtype_name: str, byte_order: str) -> np.dtype:
    dtype_map = {
        "uint16": "u2",
        "u16": "u2",
        "int16": "i2",
        "i16": "i2",
        "float32": "f4",
        "f32": "f4",
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported raw dtype '{dtype_name}'. Supported: uint16, int16, float32"
        )

    if byte_order not in RAW_BYTE_ORDERS:
        raise ValueError(f"Unsupported RAW byte_order: {byte_order!r}")
    byte_prefix = "<" if RAW_BYTE_ORDERS[byte_order] == "little" else ">"
    return np.dtype(byte_prefix + dtype_map[dtype_name])


def _denormalize_to_raw(volume01: np.ndarray, meta: dict) -> np.ndarray:
    volume01 = np.clip(volume01.astype(np.float32), 0.0, 1.0)

    dtype_name = str(meta.get("dtype", "uint16")).lower()
    byte_order = str(meta.get("byte_order", "little")).lower()
    np_dtype = _to_raw_dtype(dtype_name, byte_order)

    if "value_range" in meta:
        min_v, max_v = meta["value_range"]
        min_v = float(min_v)
        max_v = float(max_v)
        if max_v <= min_v:
            raise ValueError(f"Invalid value_range: {meta['value_range']}")
        raw = volume01 * (max_v - min_v) + min_v
    elif "scale" in meta:
        scale = float(meta["scale"])
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        raw = volume01 * scale
    elif dtype_name in {"uint16", "u16"}:
        raw = volume01 * 65535.0
    elif dtype_name in {"int16", "i16"}:
        raw = volume01 * float(np.iinfo(np.int16).max)
    else:
        raw = volume01

    if np.issubdtype(np_dtype, np.integer):
        raw = np.rint(raw)
        limits = np.iinfo(np_dtype)
        if raw.min() < limits.min or raw.max() > limits.max:
            raise ValueError(f'RAW normalization exceeds the range of {dtype_name}')

    return raw.astype(np_dtype, copy=False)


def save_raw_volume(volume01: np.ndarray, output_raw_path: str, meta: dict) -> None:
    if volume01.ndim == 4 and volume01.shape[0] == 1:
        volume01 = volume01[0]
    if volume01.ndim != 3:
        raise ValueError(f"volume01 must have shape [D, H, W] or [1, D, H, W], got {volume01.shape}")
    if not np.isfinite(volume01).all():
        raise ValueError('RAW input contains non-finite values')

    out_meta = dict(meta)
    out_meta["shape"] = [int(volume01.shape[0]), int(volume01.shape[1]), int(volume01.shape[2])]
    out_meta = validate_raw_meta(out_meta)

    raw = _denormalize_to_raw(volume01, out_meta)
    if not np.isfinite(raw).all():
        raise ValueError('RAW output contains non-finite values')
    os.makedirs(os.path.dirname(os.path.abspath(output_raw_path)), exist_ok=True)
    raw.tofile(output_raw_path)

    meta_path = f"{output_raw_path}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2, allow_nan=False)
        f.write('\n')


def _load_raw_scalar_image(file_path: str, raw_meta=None) -> tio.ScalarImage:
    meta = _load_raw_meta(file_path, raw_meta)
    shape = tuple(meta['shape'])

    dtype_name = str(meta.get("dtype", "uint16")).lower()
    byte_order = str(meta.get("byte_order", "little")).lower()
    dtype = _to_raw_dtype(dtype_name, byte_order)
    flat = np.fromfile(file_path, dtype=dtype)
    expected_count = int(np.prod(shape))
    if flat.size != expected_count:
        raise ValueError(
            f"raw size mismatch for {file_path}. "
            f"Expected {expected_count} values from shape {shape}, got {flat.size}."
        )

    volume = flat.reshape(shape)
    volume = _normalize_raw(volume, meta)
    volume = np.clip(volume, 0.0, 1.0)

    tensor = torch.from_numpy(volume).unsqueeze(0)
    spacing = meta.get('spacing')
    if spacing is not None and meta.get('spacing_unit', 'mm') == 'um':
        spacing = [float(s) / 1000 for s in spacing]
    affine = _build_affine(spacing)
    return tio.ScalarImage(tensor=tensor, affine=affine)


def load_scalar_image(file_path: str, raw_meta=None) -> tio.ScalarImage:
    file_path = os.fspath(file_path)
    lower = file_path.lower()
    if lower.endswith(".raw"):
        return _load_raw_scalar_image(file_path, raw_meta)
    return tio.ScalarImage(file_path)


def to_model_tensor(img: tio.ScalarImage) -> torch.Tensor:
    """Storage [C, X, Y, Z] -> model [C, Z, X, Y], preserving legacy axes."""
    data = img.data.to(torch.float32)
    if not torch.isfinite(data).all() or data.min() < 0 or data.max() > 1:
        raise ValueError('Image intensities must be normalized to [0, 1] before loading')
    return (data * 2 - 1).permute(0, 3, 1, 2).contiguous()


def to_storage_tensor(volume: torch.Tensor) -> torch.Tensor:
    return volume.permute(0, 2, 3, 1).contiguous()


def to_nii_gz_path(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".nii.gz"):
        return path
    if lower.endswith(".raw"):
        return f"{path[:-4]}.nii.gz"
    return f"{path}.nii.gz"
