# FID for uint16 grayscale RAW volumes

**Purpose:** Provide uint16 RAW FID/MMD examples with a fixed intensity range shared by real and generated volumes.

This page specializes [general 3D FID usage](fid_3d_usage.md), which explains checkpoints, supported formats and result fields. Use the [experiment inventory](README.md) for setup and the [documentation guide](../../docs/README.md) for other topics.

Use [fid_3d_uint16_example.ipynb](fid_3d_uint16_example.ipynb) to compare real and generated 16-bit grayscale cores. Set the input paths, checkpoint, actual shape and dtype before running the cells.

```python
from fid_3d import compute_fid_3d

REAL_DIR = "../../data/real_grayscale"
FAKE_DIR = "../../data/generated_grayscale"
CHECKPOINT = "../../checkpoints/resnet_50_23dataset.pth"
SHAPE = (128, 128, 128)
DTYPE = "uint16"
INTENSITY_RANGE = (0, 65535)

result = compute_fid_3d(
    real=REAL_DIR,
    fake=FAKE_DIR,
    checkpoint=CHECKPOINT,
    depth=50,
    shape=SHAPE,
    dtype=DTYPE,
    resize=(128, 128, 128),
    normalize="zero_one",
    intensity_range=INTENSITY_RANGE,
    batch_size=2,
    device="auto",
    output="fid_grayscale_uint16.json",
)
result
```

`intensity_range=(0,65535)` applies a fixed normalization to `[0,1]`, preserving a common intensity scale across volumes rather than stretching each sample independently.

If `fid_3d` was already imported in the same notebook kernel, reload it before using updated source:

```python
import importlib
import fid_3d
importlib.reload(fid_3d)
compute_fid_3d = fid_3d.compute_fid_3d
```

From `GLOSSY/evaluation/digital_core`, the equivalent command is:

```bash
python fid_3d.py --real ../../data/real_grayscale --fake ../../data/generated_grayscale --checkpoint ../../checkpoints/resnet_50_23dataset.pth --depth 50 --shape 128 128 128 --dtype uint16 --resize 128 128 128 --normalize zero_one --intensity-range 0 65535 --batch-size 2 --output fid_grayscale_uint16.json
```

Use a checkpoint compatible with `--depth 50`. The example checkpoint is not included in this import.
