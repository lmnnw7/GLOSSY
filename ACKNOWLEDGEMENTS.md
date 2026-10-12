# Acknowledgements

**Purpose:** Preserve upstream credits, adapted-source notices, the model citation and the bundled license reference.

Use [revision notes](docs/REVISION_NOTES.md) for implementation changes and the [documentation guide](docs/README.md) for all document roles.

We thank the authors and maintainers of the following projects for their open implementations and pretrained feature models. GLOSSY adapts these resources for digital-core reconstruction and evaluation. Renaming local modules, classes and entry points does not change the origin of the adapted code or pretrained weights.

| Project | Contribution acknowledged |
| --- | --- |
| [3D MedDiffusion](https://github.com/ShanghaiTech-IMPACT/3D-MedDiffusion) | Code base identified in the original supplied repository README. |
| [latent-diffusion](https://github.com/CompVis/latent-diffusion) | Autoencoder and perceptual-loss implementation references identified in the supplied repository. |
| [TATS](https://github.com/SongweiGe/TATS) | Source adaptation acknowledged by the retained slice-based perceptual-loss implementation. |
| [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) | Learned perceptual similarity implementation and pretrained perceptual weights. |
| [MedicalNet-models](https://github.com/warvito/MedicalNet-models) | Bundled 3D ResNet feature backbone, checkpoint-loading implementation and pretrained weights. |
| [MedicalNet / Med3D](https://github.com/Tencent/MedicalNet) | Pretrained 3D ResNet models used for volumetric perceptual features and optional FID evaluation. |

The bundled feature implementation is stored under [perceptual_backbone](perceptual_backbone/README.md). Its original [Apache 2.0 license](perceptual_backbone/LICENSE) is preserved. Modified source files carry a notice of the local naming changes. No new repository-wide license is assigned by this code revision.

## Retained source attribution

The following source notices were moved from `AutoEncoder/model/lpips.py` into this document:

> Adapted from https://github.com/SongweiGe/TATS

> Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models

The bundled feature-model README described an unofficial implementation for accessing MedicalNet models based on *Med3D: Transfer Learning for 3D Medical Image Analysis*. Its pretrained models remain third-party feature extractors after the local naming changes. The original README's model citation is retained below:

```bibtex
@article{chen2019med3d,
    title={Med3D: Transfer Learning for 3D Medical Image Analysis},
    author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
    journal={arXiv preprint arXiv:1904.00625},
    year={2019}
}
```

The diffusion implementation also retains the cosine noise schedule referenced by [Improved Denoising Diffusion Probabilistic Models](https://openreview.net/forum?id=-NEXDKk8gZ).
