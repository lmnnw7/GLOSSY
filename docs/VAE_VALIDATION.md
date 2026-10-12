# VAE training fixes and validation

**Purpose:** Describe VAE training fixes and the real-data procedures for checking training behavior and reconstruction accuracy.

Use [repository verification](VALIDATION.md) for local check results and pending environment checks. Use the [project README](../README.md) for full training commands and the [documentation guide](README.md) for other documents.

Encoder, decoder, attention and residual network definitions retain their supplied layers and parameters. Configured learning rates and loss weights are retained. Training/codebook repairs can change numerical accumulation and training trajectories; reconstruction quality and convergence require independent measurements. [Sampling and decoding](SAMPLING.md) separately describes decoder modes and Gaussian boundary repairs.

## Changes

- Codebook initialization is saved in `_initialized`. Older checkpoints infer this state from their EMA counts, so loaded trained codebooks are not initialized again.
- Quantization uses exhaustive squared-distance search in chunks of 1,024 latent voxels, rather than one full distance matrix. Counts and embedding sums replace a dense one-hot allocation. The distance, commitment, straight-through and EMA formulas retain their original objectives. FP32 is used for quantization and EMA under mixed precision. Floating-point accumulation order and random sampling differ from the old implementation; bitwise training equivalence is not claimed.
- Generator and discriminator steps toggle their own optimizer parameters. Discriminator reconstruction runs without a generator gradient graph or codebook updates. Alternation uses the saved global optimizer-step count and continues across epochs. GAN warm-up and generator losses use the same activation condition. Disabling GAN trains only the autoencoder.
- Stage 2 freezes encoder, pre-quantization projection and codebook intrinsically. Calls to `train()` and reconstruction preview callbacks retain these frozen modules in evaluation mode. Fixed perceptual features also stay in evaluation mode, preventing dropout and BatchNorm state changes.
- Zero losses are created on the input device. Non-finite losses and gradients stop before an invalid parameter update. Optional 3D feature normalization handles constant volumes; its channel-wise output uses the correct feature ranges and tensor device.
- Stage 1 resumes a complete stage-1 training checkpoint. Stage 2 loads only weights when initialized from stage 1, and resumes optimizers/training position when given a stage-2 checkpoint. Loading uses CPU first and strict parameter matching. A missing requested checkpoint raises an error rather than silently starting a new model.
- Normal completion saves a complete stage-specific checkpoint under `model.default_root_dir/checkpoints/stage1-final.ckpt` or `stage2-final.ckpt`, in addition to periodic and best-validation checkpoints.
- Complete checkpoint loading skips fresh pretrained LPIPS/VGG initialization and restores those parameters from the checkpoint. Fresh training still requires genuine pretrained features and the caches described in the root README. A checkpoint omitting perceptual parameters is rejected by strict loading.
- Single-device training uses the normal Lightning strategy. Multiple CUDA devices use NCCL on Linux and Gloo on Windows. Mixed precision uses the explicit Lightning precision names.
- File splitting uses a local random generator. Datasets with more than 40 volumes retain 40 validation volumes; smaller datasets reserve approximately one fifth, with at least one train and one validation volume. Set `dataset.validation_count` explicitly to choose the holdout size. Splits require at least two volumes.
- Validation uses fixed center crops, aligned to `patch_size`. Training stage-2 crops are aligned and bounded by each axis, preventing silent `unfold` truncation and out-of-range sampling on anisotropic inputs. Empty patch sampling has a finite retry count. Data augmentation remains training-only.

The manual optimizer implementation follows [Lightning 2.5 manual optimization](https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/2.5.0/docs/source-pytorch/model/manual_optimization.rst). The callback loss representation was checked against its [manual optimization loop](https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/2.5.0/src/lightning/pytorch/loops/optimization/manual.py).

## Regression checks in the training environment

Run from the GLOSSY repository root after installing its environment:

```bash
python -m pip check
python tools/check_source.py
python -m unittest discover -s tests -v
```

`tests/test_vae_training.py` adds small CPU checks for 4x/8x forward/backward training in both stages, optimizer isolation, frozen codebooks, GAN-disabled training, complete checkpoint resume, validation, quantizer formulas, mixed-precision quantization, zero-valued inputs and preview failures. These tests replace perceptual features with a small test fixture; they do not validate the real pretrained feature loss or real dataset quality. The GitHub workflow includes the tests, but has not been run during editing.

## Real-data training smoke check

Configure genuine input volumes and pretrained features first. Use a new output directory for each run. The examples below use normalized NIfTI input; append the [RAW flags](DIRECT_INPUTS.md#source-volumes-and-vae-training) for RAW data.

```bash
python tools/check_vae_training.py --config config/REV_VQVAE_8x.yaml --data-path data/core_images --output-dir results/smoke_stage1 --steps 4 --test-gan
python tools/check_vae_training.py --config config/REV_VQVAE_8x_s2.yaml --data-path data/core_images --ckpt results/vae8x_stage1/checkpoints/stage1-final.ckpt --output-dir results/smoke_stage2 --steps 4 --test-gan
```

The stage-2 configuration must reference a compatible checkpoint. The smoke tool runs one device, batch size 1 by default, real perceptual features and the real data loader. It checks gradients, validation metrics and frozen stage-2 codebooks; it saves an isolated `smoke.ckpt` and `smoke_report.json`. `--test-gan` enables the adversarial phase immediately only within the smoke configuration. Remove it to exercise the configured warm-up. `--cpu` selects FP32 CPU execution, which may be slow for real stage-2 volumes. Passing this check demonstrates only the steps and shapes tested.

The default batch sizes and stage-2 crop sizes are retained. Adjust batch size to available memory. Set `model.log_max_volumes: 0` to disable expensive volume previews. `model.codebook_chunk_size` controls temporary distance memory; it does not reduce global decoder attention or output-volume memory. A full 1024-cubed decode has not been memory-validated.

## Reconstruction accuracy comparison

Train a candidate checkpoint and compare it with the baseline on independent, identical held-out inputs using the same preprocessing:

```bash
python evaluation/compare_vae.py --baseline-ckpt checkpoints/baseline.ckpt --candidate-ckpt checkpoints/candidate.ckpt --data-dir data/held_out --output results/vae_comparison.json --crop-size 128 128 128 --check
```

This reports each sample and the mean MAE/RMSE in `[-1,1]`, and PSNR after clipping into `[0,1]`. The optional `--pore-threshold` reports porosity and absolute porosity error, using intensity less than or equal to the supplied threshold as pores. For RAW held-out data, append the same RAW flags to the comparison command. The crop dimensions are model `[D,H,W]` axes and must be multiples of the checkpoint's patch size; omit `--crop-size` to compare full volumes when memory permits.

`--check` exits with failure if mean MAE increases or mean PSNR decreases beyond `--max-mae-increase` and `--max-psnr-drop`. Both default to zero; choose experimental tolerances before evaluating results. The gate checks these two reconstruction metrics only. It does not establish full manuscript accuracy, pore connectivity, permeability, S2/MPC fidelity or diffusion-generation quality. Repeat the independent digital-core evaluation for those claims. Changing a codebook through new VAE training also requires re-exporting latents and training diffusion with that same VAE checkpoint.
