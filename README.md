# GLOSSY

**Global-Local Synergy Latent Diffusion for 3D Digital Core Reconstruction**

Data preparation scripts are in `dataset/`; runtime and sampling utilities are in `utils/`. Run the commands from the project root in Linux/Bash. The example uses 128³ uint16 RAW volumes and a VAE with 8× spatial compression.

**Workflow:** Data preparation → VAE training → Latent export → Diffusion training → Generation and evaluation.

Digital-core experiments are described in the [evaluation guide](evaluation/README.md).

## 1. Install the environment

```bash
conda env create -f GLOSSY.yaml

conda activate GLOSSY
```

The environment file includes training and evaluation dependencies. An installed environment can be updated with `conda env update -f GLOSSY.yaml`. VAE training requires pretrained VGG16/LPIPS weights, which must be downloaded or cached before offline execution.

## 2. Prepare VAE data

The example input is `data/core256.raw`; adjust relative paths to the dataset location. VAE preparation splits a 256³ core into eight non-overlapping 128³ volumes, partitioned as 6/1/1 for training/validation/test. Byte order, intensity range and voxel spacing must match the acquisition parameters.

```bash
python dataset/prepare_vae_blocks.py \
  --input data/core256.raw --output-root data/vae128 --shape 256 256 256 \
  --block-size 128 --validation-count 1 --test-count 1 --seed 216
```

RAW uses C-order XYZ storage. `vae_pool/` contains training and validation data. A 256³ → 64³ split uses `--block-size 64 --validation-count 8 --test-count 8` and a new output directory. Set RAW shape to 64³ and `dataset.validation_count: 8` in the VAE YAML for that split. Stage 2 uses volumes larger than the 64³ encoding patches to train cross-patch decoding.

## 3. Train the VAE

VAE training parameters are defined in the YAML files under `config/`. Encoding uses 64³ patches. The diffusion model uses a separate patch size of 1.

**Stage 1 — Train the autoencoder**

```bash
python train/train_REV_VQVAE.py \
  --config config/REV_VQVAE_8x.yaml --data-path data/vae128/vae_pool \
  --results-dir results/vae8x_stage1 \
  --raw-shape 128 128 128 --raw-dtype uint16 --raw-byte-order little \
  --raw-value-range 0 65535 --raw-spacing 1 1 1 --raw-spacing-unit um
```

**Stage 2 — Train cross-patch decoding**

```bash
python train/train_REV_VQVAE_stage2.py \
  --config config/REV_VQVAE_8x_s2.yaml --data-path data/vae128/vae_pool \
  --results-dir results/vae8x_stage2 \
  --ckpt results/vae8x_stage1/checkpoints/stage1-final.ckpt \
  --raw-shape 128 128 128 --raw-dtype uint16 --raw-byte-order little \
  --raw-value-range 0 65535 --raw-spacing 1 1 1 --raw-spacing-unit um
```

A pretrained VAE can be used by specifying its relative checkpoint path in the commands below, for example `checkpoints/your_vae.ckpt`. Latent export, diffusion training and generation must use the same VAE checkpoint.

## 4. Prepare diffusion data and train

Diffusion data consists of equal-sized blocks extracted with a stride of half the block size: 256³ → 128³ with stride 64 produces 27 blocks; 256³ → 64³ with stride 32 produces 343. Extraction saves complete blocks without changing uint16 values. Training, validation and test parent cores must be partitioned before extraction to avoid sharing voxels between datasets.

Block extraction is available through [core_subvolume_extraction.ipynb](evaluation/digital_core/core_subvolume_extraction.ipynb) or the command below. Open the notebook with `jupyter notebook evaluation/digital_core/core_subvolume_extraction.ipynb`, set `file_path`, `shape`, `block_size` and `output_dir`, and run the first section. The example writes 128³ blocks with stride 64 to `data/diffusion128`. Both entry points use the same extraction implementation and require a new output directory. After extraction, export latents and train the diffusion model:

**Extract overlapping blocks**

```bash
python dataset/prepare_diffusion_blocks.py \
  --input data/core256.raw --output-root data/diffusion128 \
  --shape 256 256 256 --block-size 128
```

**Export VAE latents**

```bash
python train/generate_training_latent.py \
  --data-path data/diffusion128 --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt \
  --batch-size 2 --num-workers 16 \
  --raw-shape 128 128 128 --raw-dtype uint16 --raw-byte-order little \
  --raw-value-range 0 65535 --raw-spacing 1 1 1 --raw-spacing-unit um
```

Export uses the original full-volume batching: `--batch-size 2` loads two complete, equally sized input volumes, transfers them to the selected device, and encodes all their 64³ patches together. `--num-workers` controls DataLoader workers. Each input volume produces its own complete latent file. The earlier `--patch-batch-size` option has been removed.

**Train the diffusion model**

```bash
python train/train_MSP_DiffNet_SingleRes.py \
  --data-path data/diffusion128_latents --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt \
  --results-dir results/diffusion --resolution 16 16 16 \
  --patch-size 1 --sub-volume-size 8 8 8 --disable-ckpt-sampling
```

The training resolution must match the exported latents. The example explicitly sets it to 16³. Use the `4x` YAML files for 4× VAE compression.

| Input volume | Latent size, 8× VAE | Latent size, 4× VAE |
| :--- | :--- | :--- |
| 128³ | 16³ | 32³ |
| 64³ | 8³ | 16³ |

Latent export uses continuous representations by default. `--quantize` selects quantized codebook embeddings. The latent representation must match the diffusion checkpoint. Resume training with `--ckpt results/diffusion/checkpoints/last.pt` and the architecture settings recorded in that checkpoint.

## 5. Generate and evaluate

```bash
python evaluation/generate.py \
  --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --model-ckpt results/diffusion/checkpoints/last.pt \
  --output-dir outputs/generated --volume-size 128 128 128 \
  --num-samples 16 --decode-mode full --sample-save-format both --seed 19 \
  --spacing 1 1 1
```

Generation loads the architecture configuration and weights from the checkpoint. Select the decoding mode using the following options:

| Decoding mode | Command-line options |
| :--- | :--- |
| Full-volume decoding (default) | `--decode-mode full` |
| Sliding-window decoding | `--decode-mode sliding --decode-patch-size 64` |
| Gaussian-weighted decoding | `--decode-mode gaussian --decode-patch-size 64` |

DDIM sampling is selected with `--sampling-strategy ddim --ddim-steps 100`.

Model evaluation utilities and the axial porosity distribution comparison, S2 and REV experiments are described in [evaluation/README.md](evaluation/README.md). Research comparisons use independent held-out cores. The bundled feature backbone uses the [Apache 2.0 license](perceptual_backbone/LICENSE).

## Acknowledgements

This repository builds upon [LDM](https://github.com/CompVis/latent-diffusion), [DiT](https://github.com/facebookresearch/DiT), and [pix2pixHD](https://github.com/NVIDIA/pix2pixHD). We thank their authors for sharing the code.
