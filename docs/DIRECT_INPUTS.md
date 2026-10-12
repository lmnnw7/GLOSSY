# Run with directories and command-line metadata

**Purpose:** Explain training, latent export and reconstruction without JSON input files.

Use the [project README](../README.md) for installation and generation, [optional JSON formats](JSON_CONFIG.md) for grouped sources, and the [documentation guide](README.md) for other topics.

VAE training, latent export, diffusion training and reconstruction accept data directories or volume files directly. No JSON input files are required. Run commands from the GLOSSY root in the supplied ML environment. The VAE YAML files still select model/training hyperparameters; their default dataset root is `data/core_images`.

## Source volumes and VAE training

Normalized `.nii.gz` volumes include their shape and spacing. RAW files have no header, so their actual storage dimensions must be supplied. For a directory of 128-cubed uint16 RAW files, append:

```text
--raw-shape 128 128 128 --raw-dtype uint16 --raw-byte-order little --raw-value-range 0 65535 --raw-spacing 1.03 1.03 1.03 --raw-spacing-unit um
```

These are example values. Use the actual source intensity range, byte order and spacing. `--raw-shape` is storage `[X,Y,Z]`, with C-order bytes. Dtype defaults to uint16, byte order to little endian, and spacing units to mm. Unsigned integers use their dtype range when no intensity range is supplied; signed integers need explicit normalization. `--raw-scale` is an alternative to `--raw-value-range`.

Supplying `--raw-shape` bypasses all RAW JSON sidecars. A single command's RAW metadata applies to all RAW files in its source directory, so those files must share the declared shape/type. Use NIfTI files for inputs with different shapes or per-volume metadata.

For normalized NIfTI sources:

```bash
python train/train_REV_VQVAE.py --config config/REV_VQVAE_8x.yaml --data-path data/core_images --results-dir results/vae8x_stage1
python train/train_REV_VQVAE_stage2.py --config config/REV_VQVAE_8x_s2.yaml --data-path data/core_images --results-dir results/vae8x_stage2 --ckpt results/vae8x_stage1/checkpoints/stage1-final.ckpt
```

Append the RAW flags above to both commands when using RAW sources. The CLI data path, output directory and checkpoint override their YAML counterparts before validation. Stage 2 requires a compatible stage-1 or stage-2 checkpoint. VAE training requires at least two independent volumes for its train/validation split.

## Export latents and train diffusion

```bash
python train/generate_training_latent.py --data-path data/core_images --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --patch-size 64 --patch-batch-size 1
python train/train_MSP_DiffNet_SingleRes.py --data-path data/core_images_latents --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --results-dir results/core128_8x --resolution 16 16 16 --batch-size 1 --disable-ckpt-sampling
```

Append RAW flags to the export command for RAW sources. The diffusion loader reads multichannel NIfTI latents and needs no RAW input flags. A 128-cubed source compressed 8x produces a 16-cubed latent; set `--resolution` to the actual latent dimensions. Directory input defaults to class 0; `--class-id` selects another class and diffusion `--num-classes` must include that ID.

Default export directories are `<source_directory>_latents`. With `--output-root data/latents`, outputs are organized as `data/latents/<class_id>/<source_index>`; pass that actual directory to diffusion training. Direct diffusion input uses the directory exactly as given and never appends `_latents`.

Each exported `core.nii.gz` has a small `core.nii.gz.meta.pt` provenance record. The exporter saves primitive values with `torch.save`, and directory-based training reads them with `weights_only=True`. VAE digest, compression, latent axes and normalization are checked, and quantized/continuous exports cannot be mixed. JSON manifests are optional via `--manifest`; direct directory export does not generate or require a JSON manifest. Older latent directories without provenance remain readable, with the same unverified pairing limitation as earlier exports.

## Reconstruction and checks

```bash
python evaluation/ae_encode_decode_recon.py --ae-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --data-path data/core_images --output-dir outputs/reconstruction
python tools/check_vae_training.py --config config/REV_VQVAE_8x.yaml --data-path data/core_images --output-dir results/smoke8x --steps 4
```

Append RAW flags for RAW sources. The reconstruction decoder defaults to `full`; `--decode-mode sliding` or `--decode-mode gaussian` selects another path. Forward reconstruction checks and paired VAE comparisons accept the same RAW input flags.

Explicit reconstruction saves RAW and NIfTI as grayscale `[0,1]` storage intensities, with the source affine/spacing. MAE is still computed in model values `[-1,1]`. Older versions of this explicit tool saved NIfTI values in `[-1,1]`; convert those files before treating them as normalized source images.

Generation and evaluation may automatically write JSON output reports or RAW sidecars. These files are outputs for traceability and do not need to be prepared as inputs. Optional legacy dataset JSON files and metadata remain supported; see [their formats](JSON_CONFIG.md).

See [repository verification](VALIDATION.md) for executed checks and environment limitations, and [VAE validation](VAE_VALIDATION.md) for real-data training and accuracy procedures.
