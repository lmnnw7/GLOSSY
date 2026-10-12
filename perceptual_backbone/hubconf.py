# Modified for GLOSSY: local module, cache and entry-point names.
# Optional list of dependencies required by the package
dependencies = ["torch", "gdown"]

from volumetric_models.models.resnet import (
    volumetric_resnet10,
    volumetric_resnet10_23datasets,
    volumetric_resnet50,
    volumetric_resnet50_23datasets,
    volumetric_resnet101,
    volumetric_resnet152,
    volumetric_resnet200
)
