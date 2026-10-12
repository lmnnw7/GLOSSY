# Digital-core FID with a pretrained 3D medical model

**Purpose:** Explain general 3D FID/MMD inputs, pretrained feature-model requirements, commands and result fields.

See the [experiment inventory](README.md) for setup and [uint16 grayscale FID](fid_3d_uint16_usage.md) for fixed-range RAW normalization. The [documentation guide](../../docs/README.md) maps all project documents.

Use [fid_3d_example.ipynb](fid_3d_example.ipynb) interactively or [fid_3d.py](fid_3d.py) from the command line.

## Input formats

Supported volume formats are `.npy`, `.npz`, `.raw`, `.dat`, `.bin` and `.img`. Each file represents one 3D sample. RAW/DAT/BIN/IMG inputs require the actual volume shape and dtype. NPY/NPZ inputs normally use `shape=None`.

## Notebook example

Open the notebook from this directory and set the real-volume, generated-volume and pretrained checkpoint paths. For example:

```python
from fid_3d import compute_fid_3d

REAL_DIR = "../../data/real_binary"
FAKE_DIR = "../../data/generated_binary"
CHECKPOINT = "../../checkpoints/resnet_50_23dataset.pth"

result = compute_fid_3d(
    real=REAL_DIR,
    fake=FAKE_DIR,
    checkpoint=CHECKPOINT,
    depth=50,
    shape=(128, 128, 128),
    dtype="int8",
    resize=(128, 128, 128),
    batch_size=2,
    device="auto",
    output="fid_results.json",
    is_num_classes=0,
)
result
```

The shape and dtype must match your files. For NPY/NPZ inputs, use `shape=None` where appropriate. The checkpoint must match the selected 3D ResNet depth and architecture. The example ResNet50 checkpoint is not supplied by this import.

## Command line

Run from `GLOSSY/evaluation/digital_core`:

```bash
python fid_3d.py --real ../../data/real_binary --fake ../../data/generated_binary --checkpoint ../../checkpoints/resnet_50_23dataset.pth --depth 50 --shape 128 128 128 --dtype int8 --resize 128 128 128 --batch-size 2 --output fid_results.json
```

For NPY/NPZ files, omit `--shape` and `--dtype` when they are unnecessary. Evaluate separate model output directories by repeating the command with a different `--fake` path and output filename, or by looping over calls to `compute_fid_3d`.

## Output fields

- `fid`: feature-space Frechet distance between real and generated samples.
- `mmd_rbf`: maximum mean discrepancy with an RBF kernel.
- `num_real`, `num_fake`: numbers of evaluated samples.
- `feature_dim`: dimension of the extracted 3D ResNet features.
- `inception_score_fake_mean`, `inception_score_fake_std`: available when a compatible classification head and a positive `is_num_classes` are supplied.

Use the same pretrained feature extractor for real and generated volumes. Omitting the checkpoint uses a randomly initialized feature model, which is useful for checking the workflow but does not provide a comparable pretrained-feature FID result.

At least two real and two generated samples are required. Checkpoint-based feature extraction has not been executed in the editing environment; see [repository verification](../../docs/VALIDATION.md).

## Inception score

If the checkpoint includes a compatible classification head, set `is_num_classes` to its class count, for example 23. Inception score uses class probabilities. A medical classification head is not a digital-core classifier, so this optional result has a different interpretation from a rock-domain classification score. The default `is_num_classes=0` evaluates FID/MMD without this score.
