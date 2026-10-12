# Volumetric perceptual backbone

**Purpose:** Describe the bundled 3D ResNet feature extractor, available hub entry points and pretrained-weight loading.

This directory supports optional 3D perceptual features; it is separate from GLOSSY's trainable VAE and diffusion checkpoints. Use the [project README](../README.md) for installation and the [documentation guide](../docs/README.md) for other documents.

Local pretrained 3D ResNet feature extractors for `VolumetricPerceptualLoss`. Network layers, parameter names, checkpoint formats and numerical operations are retained. Module names, hub entry points and local directories use the GLOSSY naming convention.

The default loss uses `volumetric_resnet10_23datasets` and the supplied checkpoint at `weights/resnet_10_23dataset.pth`. Weights for other entry points are downloaded from their existing addresses when absent. Download addresses and checkpoint filenames select the same pretrained weights.

From the GLOSSY repository root:

```python
from pathlib import Path
import torch

backbone_dir = Path("perceptual_backbone").resolve()
model = torch.hub.load(
    repo_or_dir=str(backbone_dir),
    model="volumetric_resnet10_23datasets",
    source="local",
    trust_repo=True,
    model_dir=str(backbone_dir / "weights"),
)
```

Available entry points:

```text
volumetric_resnet10
volumetric_resnet10_23datasets
volumetric_resnet50
volumetric_resnet50_23datasets
volumetric_resnet101
volumetric_resnet152
volumetric_resnet200
```

See the repository's [acknowledgements](../ACKNOWLEDGEMENTS.md) for source credits and the pretrained feature-model citation. The supplied [license](LICENSE) is retained without changes.
