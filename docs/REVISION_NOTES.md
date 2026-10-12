# Revision notes and compatibility

**Purpose:** Summarize implementation changes and explain how existing code, datasets and checkpoints interact with the current repository.

Use the [project README](../README.md) for current commands, [repository verification](VALIDATION.md) for executed-check results, and the [documentation guide](README.md) for other document roles.

## Current implementation

| Area | Current behavior | Detailed reference |
| --- | --- | --- |
| Project names | VAE class/module is `REV_VQVAE`; diffusion is `MSP_DiffNet`. Shared 3D features use `volumetric_resnet*` entry points. | [Acknowledgements](../ACKNOWLEDGEMENTS.md) retains source credits and model citations. |
| VAE training | Codebook initialization is resumable; quantization search is chunked; generator/discriminator optimizers are isolated; stage 2 freezes the encoder/codebook; validation uses deterministic splits/crops. | [VAE validation](VAE_VALIDATION.md) describes the fixes and experimental checks. |
| Diffusion | Local latent assembly uses actual patch counts; batched resolution conditions broadcast correctly; numeric continuous conditioning is supported; DDIM uses the clean endpoint and stochastic direction term. | [Sampling and decoding](SAMPLING.md) documents runtime choices. |
| Diffusion training | Single GPU or Linux torchrun/DDP, rank-specific sampling, EMA, atomic `last.pt` and complete training-state checkpoints. | [Project README](../README.md#msp-diffnet-training) gives training and resume commands. |
| Data inputs | Direct directories/files, CLI RAW metadata and optional legacy JSON mappings; storage/model axis conversions are shared. | [Direct inputs](DIRECT_INPUTS.md) and [optional JSON formats](JSON_CONFIG.md). |
| Latent export | Patch batches are encoded on GPU and assembled on CPU; quantized embeddings are the default; per-file provenance is recorded; existing files are protected. | [Direct inputs](DIRECT_INPUTS.md#export-latents-and-train-diffusion). |
| Generation | Saved checkpoint architecture settings, numeric class IDs, consistent grayscale RAW/NIfTI outputs and generation metadata. | [Project README](../README.md#generate-digital-cores). |
| Decoding | CLI selection of full/sliding/Gaussian decoding; default `full`; Gaussian blending retains the supplied weight formula with corrected boundaries and small-input handling. | [Sampling and decoding](SAMPLING.md). |
| Experiments | Shared S2/MPC calculations, cleared notebook outputs and portable example paths. | [Digital-core experiments](../evaluation/digital_core/README.md). |
| Repository checks | Standard-library source/notebook/configuration/documentation checks, synthetic CPU regressions and CI configuration. | [Repository verification](VALIDATION.md). |

Encoder, decoder, attention and residual network definitions retain their supplied computational structure and parameter names. Configured VAE learning rates/loss weights are retained. Training and quantizer fixes can change floating-point accumulation order and training trajectories; unchanged quality requires a paired experimental comparison.

## Consolidated duplicate code

| Earlier duplicate or wrapper | Active implementation |
| --- | --- |
| Copied latent-export script | [generate_training_latent.py](../train/generate_training_latent.py) |
| Copied forward-reconstruction script | [ae_reconstruction_check.py](../evaluation/ae_reconstruction_check.py) |
| Separate 4x/8x class-generation scripts | [generate.py](../evaluation/generate.py) |
| Separate 4x dataset module | Both dataset variants in [core_volume.py](../dataset/core_volume.py) |
| Repeated VAE loader/callback/trainer setup | [vae_training.py](../train/vae_training.py) |
| Copied notebook S2/MPC functions | [s2_pore_phase.py](../evaluation/digital_core/s2_pore_phase.py) and [multipoint_metrics.py](../evaluation/digital_core/multipoint_metrics.py) |

The two reconstruction scripts have distinct purposes: `ae_reconstruction_check.py` checks the model forward path and reports MAE/PSNR; `ae_encode_decode_recon.py` explicitly encodes patches and allows decoder-mode selection. The four YAML files cover two compression ratios and two training stages. The exports under `environment/` record the supplied historical environment; use the root environment/requirements files for installation.

## Checkpoint compatibility

Use `python tools/check_checkpoints.py --ae-ckpt <vae.ckpt> --diff-ckpt <diff.pt> --device cuda` in the training environment. It checks strict weight loading, VAE/latent channel and compression compatibility, and small encode/decode/denoiser forwards. `--skip-forward` limits the check to loading/metadata. These checks do not certify earlier accuracy.

Legacy diffusion `args` dictionaries and argparse namespaces restore recorded patch size, local window, class/channel counts and supported attention/conditioning settings. State dictionaries with `module.` prefixes are accepted. Missing architecture metadata still requires the original CLI settings; old 4x models may need an 8-voxel local window. The VAE used for old diffusion must retain the original codebook and preprocessing. Old checkpoints without a VAE digest cannot verify that pairing automatically.

VAE inference compatibility does not guarantee identical training continuation after optimizer/quantizer fixes. Old diffusion checkpoints may load for inference while omitting scaler, per-batch position or RNG state. The tool reports missing complete-resume fields; do not describe such files as proven exact training resumes. Whole pickled model objects referencing removed Python modules are distinct from supported state-dictionary checkpoints.

The codebook `_initialized` buffer is saved in new checkpoints. Older state dictionaries infer initialization from EMA counts. Network parameter keys are retained, but real historical checkpoints still require execution in the target environment. Entire pickled model objects referencing removed module paths may need conversion to state dictionaries in their original environment.

Stage 1 resumes a stage-1 Lightning checkpoint. Stage 2 initializes weights from stage 1 or fully resumes a stage-2 checkpoint. Missing or incompatible stage checkpoints fail explicitly. Normal completion writes `stage1-final.ckpt` or `stage2-final.ckpt` under the requested results directory's `checkpoints` folder.

New 4x diffusion training derives its local window from `64/4=16`; earlier 4x models with an 8-voxel latent window need `--sub-volume-size 8 8 8`. Positional embedding shapes depend on this setting. New diffusion checkpoints record architecture and diffusion settings for generation.

Resume diffusion with the same VAE, dataset, latent representation, resolution, world size, batch size and architecture settings. Checkpoint state includes optimizer/scaler, epoch/batch position, training step and per-rank RNG state. Resume reconstructs the sampler position; exact training equivalence still requires target-environment validation.

Explicit reconstruction now writes NIfTI in `[0,1]`, matching RAW normalization and the source image loader. Older explicit-reconstruction NIfTI outputs in `[-1,1]` require intensity conversion before reuse. Model outputs and reconstruction metrics retain their model-scale calculations.

## Dataset compatibility

Block and crop extraction now share the original notebook's implementation through `preprocessing/core_subvolume_extraction.py`; the notebook is updated to call it. `preprocessing/prepare_core_blocks.py` adds deterministic split folders/CSV records around that implementation. `evaluation/check_diffusion_latents.py` contains the latent noise-loss evaluation entry point. The Chinese documentation directory holds documentation only.

Direct diffusion paths identify the actual latent directory/file. Legacy flat JSON source mappings append `_latents`; explicit JSON mappings read their `latents` field directly. Relative VAE and dataset-mapping paths resolve against the GLOSSY root. Run documented commands from that root.

New latent `.meta.pt` records retain the VAE digest, compression, axis order and normalization convention. Quantized and continuous exports must not be mixed. Older directories without provenance remain readable, but automatic pairing can only check recorded metadata. Retraining or changing the VAE codebook requires exporting new latents and training diffusion with that VAE.

Module renaming does not change source attribution. Third-party checkpoint filenames, download addresses and established method names remain where needed for compatibility; see [Acknowledgements](../ACKNOWLEDGEMENTS.md).
