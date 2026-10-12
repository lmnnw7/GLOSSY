"""Shared MPC computations; notebook cells retain plotting and input settings."""
from pathlib import Path
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def read_raw_volume(path, dimension, dtype):
    path = Path(path)
    expected = int(np.prod(dimension)) * np.dtype(dtype).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f'{path.name}: {actual} bytes, expected {expected} for shape={dimension}, dtype={np.dtype(dtype)}')
    return np.fromfile(path, dtype=dtype).reshape(tuple(dimension), order='C')


def make_pore_mask(volume, definition, pore_value, threshold):
    if definition == 'nonzero':
        return volume != 0
    if definition == 'equal':
        return volume == pore_value
    if definition == 'less_equal':
        return volume <= threshold
    raise ValueError('PORE_DEFINITION must be nonzero, equal, or less_equal')


def mpc_fixed_h_numpy(mask, n, h, axis):
    """MPC with fixed h and n sampled points, averaged over valid starts."""
    mask = np.asarray(mask, dtype=bool)
    n, h, axis = (int(n), int(h), int(axis))
    if n < 1 or h < 0:
        raise ValueError('n must be >= 1 and h must be >= 0')
    size = mask.shape[axis]
    span = (n - 1) * h
    if span >= size:
        return np.nan
    n_start = size - span
    views = []
    for j in range(n):
        sl = [slice(None)] * mask.ndim
        sl[axis] = slice(j * h, j * h + n_start)
        views.append(mask[tuple(sl)])
    return float(np.logical_and.reduce(views).mean())


def mpc_fixed_h_torch(mask, n, h, axis):
    """MPC with fixed h and n sampled points, averaged over valid starts."""
    n, h, axis = (int(n), int(h), int(axis))
    if n < 1 or h < 0:
        raise ValueError('n must be >= 1 and h must be >= 0')
    size = mask.shape[axis]
    span = (n - 1) * h
    if span >= size:
        return np.nan
    n_start = size - span
    first_slice = [slice(None)] * mask.ndim
    first_slice[axis] = slice(0, n_start)
    all_pore = mask[tuple(first_slice)].clone()
    for j in range(1, n):
        sl = [slice(None)] * mask.ndim
        sl[axis] = slice(j * h, j * h + n_start)
        all_pore.logical_and_(mask[tuple(sl)])
    return all_pore.float().mean().item()


def read_raw(path, dimension, dtype):
    path = Path(path)
    expected = int(np.prod(dimension))
    raw = np.fromfile(path, dtype=dtype)
    if raw.size != expected:
        raise ValueError(f'{path}: got {raw.size} voxels, expected {expected}. Check DIMENSION={dimension} and DATA_TYPE={np.dtype(dtype)}.')
    return raw.reshape(tuple(dimension), order='C')


def find_rev(distance, curve, beta, stable_tail):
    """
    REV is placed at distance[i+1] when the neighboring x-axis points
    satisfy abs(MPC[i+1] - MPC[i]) <= beta.
    """
    distance = np.asarray(distance, dtype=float)
    curve = np.asarray(curve, dtype=float)
    if curve.size < 2:
        return None
    delta = np.abs(np.diff(curve))
    if stable_tail:
        candidates = np.flatnonzero([np.all(delta[i:] <= beta) for i in range(delta.size)])
    else:
        candidates = np.flatnonzero(delta <= beta)
    return float(distance[candidates[0] + 1]) if candidates.size else None
