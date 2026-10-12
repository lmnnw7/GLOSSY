# GLOSSY

**Global-Local Synergy Latent Diffusion for Large Field of View 3D Digital Core Reconstruction**

**Purpose:** Introduce GLOSSY and provide installation and end-to-end training/generation instructions.

GLOSSY learns grayscale micro-CT core volumes with a REV-VQVAE autoencoder and an MSP-DiffNet latent diffusion model. MSP-DiffNet combines a local DiT branch with a global 3D U-Net branch. REV-VQVAE retains the supplied encoder/decoder network design and configured loss weights; training, codebook state and memory handling have been repaired. Reconstruction quality and convergence require measurements in the training environment.

## Documentation

The [documentation guide](docs/README.md) explains the purpose of **every Markdown file**. Use [direct inputs](docs/DIRECT_INPUTS.md) for commands without JSON inputs, [sampling and decoding](docs/SAMPLING.md) for decoder switches, [VAE validation](docs/VAE_VALIDATION.md) for model smoke checks, and [repository verification](docs/VALIDATION.md) for executed checks and remaining limitations. [Revision notes](docs/REVISION_NOTES.md) describe implementation and checkpoint compatibility changes.

## Environment

Use Python 3.11 and a CUDA-capable GPU for training. Diffusion multi-GPU training requires Linux/NCCL; VAE multi-device training selects NCCL on Linux and Gloo on Windows. CPU is supported for small regression tests and script-level inference fallback; large-volume generation requires substantial GPU and host memory.

From the repository root:

```bash
conda env create -f GLOSSY.yaml
conda activate GLOSSY
python -m pip check
```

Alternatively, create your own Python 3.11 environment and install:

```bash
python -m pip install -r requirements-cu121.txt
python -m pip check
```

For CPU checks:

```bash
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The CUDA 12.1 wheel pair follows the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/). NumPy is pinned to the supplied environment's 1.x version. The supplied dependency exports are retained under `environment/`, with the archived environment name updated and its machine-specific prefix removed. The active requirements list contains the dependencies used by this repository.

Fresh VAE training initializes genuine pretrained LPIPS/VGG features and may download the torchvision VGG16 backbone and LPIPS weights. Loading a complete VAE checkpoint for inference or resumed training restores these parameters from the checkpoint, without a fresh VGG download. `AutoEncoder/model/cache/vgg.pth` contains LPIPS linear weights, **not** the complete VGG16 backbone. Offline machines must populate the PyTorch hub checkpoint cache and the LPIPS cache beforehand. The optional `VolumetricPerceptualLoss` uses the bundled feature extractor under `perceptual_backbone/`; its existing checkpoint is under `perceptual_backbone/weights/`.

## Input data

Training uses grayscale volumes. Apply cropping and fixed intensity normalization; the manuscript does not apply denoising or pore segmentation before training. Pore extraction is a separate evaluation procedure.

For source images, supported formats are `.raw` with command-line metadata (or optional JSON sidecars) and normalized `.nii.gz`. Source intensities must be in `[0,1]` when loaded; scripts convert them to `[-1,1]` for the VAE. NIfTI intensities outside `[0,1]` must be normalized beforehand using a consistent dataset-level scale.

RAW inputs can run without JSON. Append these flags to a training, export or reconstruction command, using the actual storage dimensions and intensity range:

```text
--raw-shape 128 128 128 --raw-dtype uint16 --raw-byte-order little --raw-value-range 0 65535 --raw-spacing 1.03 1.03 1.03 --raw-spacing-unit um
```

If `--raw-shape` is provided, RAW metadata comes entirely from the command and optional sidecars are ignored. Without it, existing sidecars remain supported. For example, a RAW file `core.raw` can optionally use `core.raw.json` (see [example](config/raw_meta.example.json)):

```json
{
  "shape": [128, 128, 128],
  "dtype": "uint16",
  "byte_order": "little",
  "value_range": [0, 65535],
  "spacing": [1.03, 1.03, 1.03],
  "spacing_unit": "um",
  "axis_order": "XYZ"
}
```

`shape` specifies the actual array storage order, `[X,Y,Z]`, in C order (Z varies fastest). The existing VAE convention is retained: source tensors `[C,X,Y,Z]` become model tensors `[C,D,H,W]=[C,Z,X,Y]`. Output saving reverses this permutation. Latent files store `[C,D,H,W]` directly and are never transposed by the diffusion loader. Check axis order when importing RAW data from other software. NIfTI affine spacing is written in millimetres; `--spacing` and generated RAW metadata use micrometres.

RAW metadata may also be shared through `raw_meta.json` or `_raw_meta.json` in the directory. Per-file metadata overrides shared defaults. Unsigned integer RAW data with no `value_range` uses the full dtype range. Signed integers require an explicit range or scale.

JSON input files are optional. Use source/latent directories directly; see [commands without JSON inputs](docs/DIRECT_INPUTS.md). Optional legacy JSON mappings retain strict UTF-8/BOM-aware loading, duplicate-key checks and project-relative paths. RAW sizes, dtype, byte order, storage axes, spacing and normalization fields are validated before use. See [optional JSON formats](docs/JSON_CONFIG.md).

## RAW preprocessing entry points

The original block extraction and large-region crop from `evaluation/digital_core/core_subvolume_extraction.ipynb` now share [core_subvolume_extraction.py](preprocessing/core_subvolume_extraction.py). The notebook calls that implementation. For a 256-cubed uint16 source, extract complete 64-cubed blocks:

```bash
python preprocessing/core_subvolume_extraction.py --input data/core256.raw --output data/core64 --shape 256 256 256 --block-shape 64 64 64 --stride 64 64 64 --dtype uint16 --byte-order little --storage-order XYZ
```

For a fixed 48/8/8 train/validation/test split and the matching VAE pool:

```bash
python preprocessing/prepare_core_blocks.py --input data/core256.raw --output-root data/core64_split --shape 256 256 256 --block-size 64 --validation-count 8 --test-count 8 --seed 216
```

Source dimensions follow `--storage-order`. `ZYX` sources are explicitly converted to the loader's XYZ output convention; intensities and byte order are retained. Complete extraction windows use the original slab/slice method. Configure `dataset.validation_count: 8` and `dataset.split_seed: 216` for the 56-file `vae_pool` directory. A single parent core's subvolumes do not establish independent-core generalization.

## REV-VQVAE training

The YAML configs default to the source directory `data/core_images`. Override it with `--data-path`; use `--results-dir` and `--ckpt` for output and checkpoint paths. Run these commands **from the repository root**. Relative VAE configuration/data paths resolve against the repository root. Network layers and configured learning rates/loss weights are retained; training state, optimizer isolation, splitting and patch alignment are repaired.

```bash
python train/train_REV_VQVAE.py --config config/REV_VQVAE_8x.yaml --data-path data/core_images --results-dir results/vae8x_stage1
python train/train_REV_VQVAE_stage2.py --config config/REV_VQVAE_8x_s2.yaml --data-path data/core_images --results-dir results/vae8x_stage2 --ckpt results/vae8x_stage1/checkpoints/stage1-final.ckpt
```

For 4× compression use `REV_VQVAE_4x.yaml` and `REV_VQVAE_4x_s2.yaml`. Stage 2's `model.resume_from_checkpoint` accepts a compatible stage-1 checkpoint for initialization or a stage-2 checkpoint for complete training resume. Stage 1 accepts only stage-1 checkpoints. Checkpoint paths are checked before starting.

Validation crops are fixed and patch-aligned. Datasets with more than 40 volumes retain 40 validation files; smaller datasets use a nonempty holdout split. Set `dataset.validation_count` explicitly to override it. Stage-2 and validation crop sizes can be set with `dataset.stage2_crop_shape` and `dataset.validation_crop_shape`, in storage `[X,Y,Z]` axes. Set `model.log_max_volumes: 0` to disable reconstruction previews.

Before a long run, execute the [real-data smoke check and paired accuracy comparison](docs/VAE_VALIDATION.md). A few finite training steps cannot prove convergence or unchanged manuscript metrics.

## Export training latents

Pass the source directory directly. For RAW files, also append the RAW input flags shown above; normalized NIfTI files need no RAW metadata.

```bash
python train/generate_training_latent.py --data-path data/core_images --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --patch-size 64 --patch-batch-size 1
```

Export runs the unchanged encoder on patches and assembles the latent volume on CPU. It does not copy the whole core onto the GPU. The default exports quantized embeddings to match the manuscript's discrete representation. To retain the old continuous-embedding export convention use `--no-quantize`. Changing the convention requires re-exporting latents and retraining diffusion; do not mix these latent datasets.

Default latent folders are `<image_directory>_latents`. `--output-root` selects a separate destination, organized by class/source index. Each latent has a `.meta.pt` record containing its VAE codebook digest, compression and representation; directory-based diffusion training reads these records automatically. `--class-id` defaults to 0 for directory input. Optional JSON mappings remain supported through `--data-path config/SingleRes_dataset.json`; `--manifest` requests an optional JSON output mapping. Existing latent files are protected unless `--overwrite` is supplied.

Volumes must be divisible by the encoding patch size. Use a single resolution per training run. For a 128³ image, an 8× VAE produces a 16³ latent, while a 4× VAE produces a 32³ latent. Diffusion `--resolution` always means **latent** size; generation `--volume-size` means **output voxel** size.

## MSP-DiffNet training

Single GPU:

```bash
python train/train_MSP_DiffNet_SingleRes.py --data-path data/core_images_latents --results-dir results/core128_8x --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --resolution 16 16 16 --batch-size 1 --disable-ckpt-sampling
```

Four GPUs, as in the manuscript:

```bash
torchrun --standalone --nproc_per_node=4 train/train_MSP_DiffNet_SingleRes.py --data-path data/core_images_latents --results-dir results/core128_8x --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --resolution 16 16 16 --batch-size 1 --disable-ckpt-sampling
```

Defaults are 5,000 epochs, Adam at `1e-4`, 1,000 cosine diffusion steps, L1 noise loss, batch size 1 per GPU, and EMA decay 0.995 every 10 steps after a 2,000-step warm-up. DistributedSampler gives each rank its own data partition. `--no-enable-amp` disables AMP. For 4× compression, provide the 4× VAE and the actual latent dimensions.

The local latent sub-volume defaults to `--vq-size` (64 source voxels) divided by the VAE compression ratio: 8 for the 8× VAE and 16 for the 4× VAE. An explicit `--sub-volume-size 8 8 8` preserves the old 4× model's local window size. New checkpoints save the model and diffusion configuration, so generation reconstructs the trained architecture automatically.

Checkpoints include model/EMA/optimizer/scaler state, epoch, batch position, train step, per-rank RNG state and a codebook digest. `last.pt` is saved atomically and a final checkpoint is written at normal completion. Resume with the same world size, batch size, dataset, model arguments and VAE:

```bash
python train/train_MSP_DiffNet_SingleRes.py --data-path data/core_images_latents --results-dir results/core128_8x --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --resolution 16 16 16 --ckpt results/core128_8x/checkpoints/last.pt --disable-ckpt-sampling
```

Sampling and decoding defaults are defined directly in [sampling_config.py](sampling_config.py). Select DDPM/DDIM and `full`, `sliding` or `gaussian` VAE decoding through command-line options. Decoding defaults to `full`; use `--decode-mode` for generation/reconstruction and `--sample-decode-mode` for training previews. See [sampling options and examples](docs/SAMPLING.md).

Checkpoint sampling is optional. Full-volume decoding can require much more memory than diffusion training. `--disable-ckpt-sampling` lets training proceed without decoding samples. To enable it, omit that flag; DDIM at 100 steps is the default preview sampler. Final generation defaults to DDPM.

## Generate digital cores

```bash
python evaluation/generate.py --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --model-ckpt results/core128_8x/checkpoints/last.pt --output-dir outputs/core128 --volume-size 128 128 128 --num-samples 10 --class-id 0 --sample-save-format both --seed 19
```

Optional faster DDIM generation:

```bash
python evaluation/generate.py --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --model-ckpt results/core128_8x/checkpoints/last.pt --output-dir outputs/core128_ddim --volume-size 128 128 128 --sampling-strategy ddim --ddim-steps 100
```

Both 4× and 8× checkpoints use this entry point. A class ID is a dataset label. Existing CT/MRI anatomy names and fixed class-to-resolution mappings have been removed. Output files include a sample index and a `.generation.json` record. The same normalized grayscale intensities are saved to RAW and NIfTI.

Decoding defaults to `full`, the manuscript's global joint decoding. Pass `--decode-mode sliding` or `--decode-mode gaussian` to select another path. Ordinary sliding decoding requires each output dimension to be divisible by the voxel `--decode-patch-size`; use 64 or 128 for a 128-cubed output. Gaussian decoding supports overlap and shrinks its maximum patch size to fit small inputs. Both patch methods change decoder context and boundary behavior; Gaussian blending does not establish equivalence to full decoding. See [the mode options](docs/SAMPLING.md). Full 1024³ generation is not memory-validated in this revision; available memory and global decoder attention/output buffers can limit it, even with chunked quantization. Increasing a trained model's generation resolution is not proof of the manuscript's large-FOV quality claims.

Legacy diffusion checkpoints with `ema` or `model` are supported. Saved architecture arguments are used where available; for bare legacy checkpoints provide matching `--model-dim`, `--dim-mults`, `--num-classes`, etc. Direct `--data-path` uses the actual latent directory/file and never appends a suffix. Legacy flat JSON source-path mappings still append `_latents`; explicit JSON entries use their `latents` paths directly. Older latent folders without `.meta.pt` records remain readable, but their VAE pairing cannot be checked from the folder alone; re-export with the intended VAE to record provenance.

## Validation

Check actual old checkpoints before reusing them:

```bash
python tools/check_checkpoints.py --ae-ckpt checkpoints/old_vae.ckpt --diff-ckpt checkpoints/old_diff.pt --device cuda --volume-size 64 64 64 --output results/checkpoint_compatibility.json
```

This strictly loads weights and tests small forward paths, without updating the checkpoint. A missing old VAE digest leaves pairing unverified; a diffusion checkpoint missing training-state fields cannot establish exact resume. See [checkpoint compatibility](docs/REVISION_NOTES.md#checkpoint-compatibility). Use [check_diffusion_latents.py](evaluation/check_diffusion_latents.py) to evaluate a new diffusion checkpoint's EMA noise loss on held-out latent directories.

```bash
python tools/check_source.py
python -m unittest discover -s tests -v
```

The source checker needs only the standard library and checks Python, notebook cells, local module paths, optional JSON schemas and Markdown purposes/links. The full unittest command requires the supplied ML environment. See [repository verification](docs/VALIDATION.md) for dependency-free commands, the current test inventory and which checks were actually executed. Use [VAE validation](docs/VAE_VALIDATION.md) for real-data training and held-out accuracy checks.

## Digital core experiments

Additional experimental scripts and notebooks are available under [evaluation/digital_core](evaluation/digital_core/README.md). They cover core extraction, segmentation, porosity, axial porosity, two-point and multipoint correlations, diversity, and 3D FID/MMD. Their filenames, comments, notebook text and usage notes are in English. Shared S2/MPC calculations are implemented in helper modules, while notebook settings and plotting variants are retained. Saved outputs are cleared. Optional dependencies are listed in `requirements-evaluation.txt`. The experiments are standalone and their dataset-scale results remain unverified.

## Acknowledgements

We thank the authors and maintainers of the source implementations and pretrained feature models used in this repository. Project credits, retained source notices and the feature-model citation are collected in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). The bundled feature extractor's [license](perceptual_backbone/LICENSE) is retained without changes.

Checkpoints, data, binary weights and Python/notebook caches are excluded by `.gitignore`; acquire required pretrained feature weights before offline use. Publish the **contents of this GLOSSY directory** as the repository root so that the workflow, configuration paths and commands retain their expected locations.
