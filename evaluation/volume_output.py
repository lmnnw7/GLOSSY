"""Save model volumes using the established storage axis convention."""
from pathlib import Path
import numpy as np
import torch
import torchio as tio
from dataset.io_utils import save_raw_volume, to_storage_tensor


def save_volume(volume, base, save_format='raw', raw_dtype='uint16', byte_order='little', spacing=(1.03, 1.03, 1.03)):
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    volume = to_storage_tensor(volume.detach().float().cpu())
    volume01 = ((volume + 1) / 2).clamp(0, 1)
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError('Spacing must contain three positive numbers')
    if save_format in ('nii.gz', 'both'):
        # NIfTI spatial units are mm: input core spacing is provided in um.
        affine = np.diag([*(float(s) / 1000 for s in spacing), 1.])
        tio.ScalarImage(tensor=volume01, affine=affine).save(str(base) + '.nii.gz')
    if save_format in ('raw', 'both'):
        meta = dict(dtype=raw_dtype, byte_order=byte_order,
                    value_range=[0, 65535] if raw_dtype in ('uint16', 'u16') else [0., 1.],
                    spacing=list(spacing), spacing_unit='um', axis_order='XYZ')
        save_raw_volume(volume01.numpy(), str(base) + '.raw', meta)
