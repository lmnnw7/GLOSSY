# Documentation guide

**Purpose:** Explain what every Markdown document contains and where to find each kind of information.

Start with the [project README](../README.md). Documentation is in English. Unless a page explicitly says otherwise, run its commands from the GLOSSY repository root. Example data and checkpoint paths must be replaced with your actual files.

## Document roles

| Document | What it is for | Read it when |
| --- | --- | --- |
| [Project README](../README.md) | Installation, project layout and the training-to-generation workflow. | Setting up or running GLOSSY for the first time. |
| [Documentation guide](README.md) | This index of all Markdown files and their responsibilities. | Finding the right instructions. |
| [Direct inputs](DIRECT_INPUTS.md) | Directory/file inputs and RAW metadata supplied on the command line. | Running without JSON input files. |
| [Optional JSON formats](JSON_CONFIG.md) | Dataset mappings, class labels, RAW sidecars and legacy compatibility. | Grouping multiple sources or using existing JSON inputs. |
| [Sampling and decoding](SAMPLING.md) | DDPM/DDIM settings and full/sliding/Gaussian VAE decode switches. | Selecting a decoder or changing sampling speed and memory use. |
| [VAE validation](VAE_VALIDATION.md) | VAE training fixes, real-data smoke checks and paired accuracy comparisons. | Validating a VAE before a long training run. |
| [Repository verification](VALIDATION.md) | What has actually been checked, what remains pending and how to rerun checks. | Assessing whether the code is ready for a particular environment. |
| [Revision notes](REVISION_NOTES.md) | Consolidated implementation changes and checkpoint/data compatibility. | Migrating older code, datasets or checkpoints. |
| [Digital-core experiments](../evaluation/digital_core/README.md) | Notebook/script inventory, experiment dependencies and input conventions. | Choosing a preprocessing or statistical evaluation experiment. |
| [General 3D FID](../evaluation/digital_core/fid_3d_usage.md) | Feature-model requirements, supported formats and FID/MMD commands. | Evaluating binary or array-based volumes with a 3D feature extractor. |
| [Uint16 grayscale FID](../evaluation/digital_core/fid_3d_uint16_usage.md) | Fixed-range normalization and FID/MMD examples for uint16 RAW cores. | Evaluating grayscale RAW volumes on a common intensity scale. |
| [Perceptual backbone](../perceptual_backbone/README.md) | Bundled 3D ResNet entry points, feature weights and local loading. | Configuring the optional 3D perceptual loss. |
| [Acknowledgements](../ACKNOWLEDGEMENTS.md) | Upstream credits, retained source notices, model citation and license location. | Reviewing attribution before redistribution. |

## Reading order

Executable preprocessing is in `preprocessing/`, model training in `train/`, reconstruction/generation/latent-loss evaluation in `evaluation/`, and checkpoint compatibility checks in `tools/check_checkpoints.py`. The experiment notebook uses the shared preprocessing implementation.

1. Install the environment using the project README.
2. Choose direct input commands; use the JSON format reference only if needed.
3. Validate the VAE, then train it, export latents and train diffusion.
4. Choose sampling/decoding settings and generate cores.
5. Use the experiment guide for independent digital-core evaluation.

## Documentation conventions

The project README is the workflow entry point. Topic pages hold the detailed rules for their subjects. Repository verification is the single place for current executed-check counts and environment limitations; VAE validation describes the additional experimental procedure. Revision notes describe changes and compatibility rather than repeating verification logs.

JSON **inputs** are optional. VAE model/training settings still use YAML. Generated JSON reports and RAW sidecars are output records, and `.meta.pt` files record latent provenance. Decoder defaults are defined in Python and selected with command-line flags; all decoding entry points default to `full`.
