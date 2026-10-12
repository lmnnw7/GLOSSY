# Sampling and VAE decoding command-line options

**Purpose:** Explain diffusion sampler settings, VAE decoder modes and their command-line switches.

Use the [project README](../README.md) for checkpoint creation, [direct inputs](DIRECT_INPUTS.md) for RAW input flags, and the [documentation guide](README.md) for other topics.

Generation, diffusion-training previews and direct AE encode/decode reconstruction use defaults defined directly in [sampling_config.py](../sampling_config.py). Choose the decoder on the command line; all three entry points default to `full`.

Generation defaults to DDPM, and diffusion-training previews default to DDIM at 100 steps. Diffusion sampling and VAE decoding are separate choices. Append one of these options to a generation or reconstruction command:

```text
--decode-mode full
--decode-mode sliding --decode-patch-size 128
--decode-mode gaussian --decode-patch-size 128 --decode-overlap-ratio 0.5
```

| Decode mode | Behavior | Relevant settings |
| --- | --- | --- |
| `full` | Quantize and decode the complete latent volume together. This retains the manuscript's global reconstruction path. | Patch and overlap settings are unused. |
| `sliding` | Decode independent non-overlapping patches and stitch them on CPU. | `--decode-patch-size`, `--decode-sliding-window` |
| `gaussian` | Decode overlapping patches and blend them with the supplied Gaussian weights on CPU. | `--decode-patch-size`, `--decode-overlap-ratio` |

`--decode-patch-size` defaults to 256 and specifies an output voxel side, not a latent side. The compression ratio comes from the loaded 4x/8x VAE checkpoint. Ordinary sliding decoding requires each output dimension to be divisible by this size and at least that large. For example, choose 64 or 128 for a 128-cubed output. Invalid dimensions fail before diffusion sampling begins.

For Gaussian decoding, the patch size is a maximum cube side. When an input is smaller, the decoder uses the largest fitting cube. A final window is placed at each axis boundary to cover non-divisible lengths. `--decode-overlap-ratio` defaults to 0.5 and must be in `[0,1)`; the latent stride is at least one voxel. Larger overlaps add more patches and computation. The Gaussian sigma stays at the supplied 0.4, and normalization retains `weighted_sum / (weight_sum + 1e-8)`. `--decode-sliding-window` defaults to 1 and affects only ordinary sliding patch batches.

Full, sliding and Gaussian decoding have different boundary behavior because patches change the decoder's spatial context. Gaussian blending smooths overlaps; it does not establish equal accuracy to full decoding. Global decoder attention and GPU/CPU buffers still depend on volume size.

## Command examples

Generate using the default full decoder:

```bash
python evaluation/generate.py --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --model-ckpt results/core128_8x/checkpoints/last.pt --output-dir outputs/core128 --volume-size 128 128 128
```

Choose Gaussian decoding for one generation run:

```bash
python evaluation/generate.py --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --model-ckpt results/core128_8x/checkpoints/last.pt --output-dir outputs/core128_gaussian --volume-size 128 128 128 --decode-mode gaussian --decode-patch-size 128 --decode-overlap-ratio 0.5 --sampling-strategy ddim --ddim-steps 100
```

Training previews also default to `full`. Use the prefixed command options to choose another decoder for previews:

```bash
python train/train_MSP_DiffNet_SingleRes.py --data-path data/core_images_latents --results-dir results/core128_8x_gaussian_preview --AE-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --resolution 16 16 16 --sample-decode-mode gaussian --sample-decode-patch-size 128 --sample-decode-overlap-ratio 0.5
```

`--disable-ckpt-sampling` continues to disable preview generation. Training uses the same model and objective whichever preview decoder is selected. Preview sampling settings are saved with the resolved training arguments in checkpoints.

Direct AE reconstruction accepts the same decode options as generation:

```bash
python evaluation/ae_encode_decode_recon.py --ae-ckpt results/vae8x_stage2/checkpoints/stage2-final.ckpt --data-path data/core_images --output-dir outputs/reconstruction_gaussian --decode-mode gaussian
```

The reconstruction tool also accepts `--decode-mode auto`: it chooses full decoding for inputs below `--auto-decode-threshold-voxels` and ordinary sliding otherwise. Generation and training previews accept `full`, `sliding` or `gaussian`.

Generation metadata records the resolved decode mode, requested/effective patch size, overlap, ordinary sliding batch size and diffusion sampling settings. DDIM's `eta` defaults to zero; it is unused by DDPM. Use `--eta` for generation and `--sample-eta` for previews.

## Verification and memory limits

The Gaussian decoder retains the supplied Gaussian window, sigma and normalization formula. Small-input patch sizing, final-boundary coverage and continuous-feature handling are repaired; output and blending buffers are on CPU. Ordinary sliding decoding also collects decoded patches on CPU. These methods can reduce the decoder's per-patch GPU workload, while requiring host memory for the assembled volume.

Measured peak memory and reconstruction equivalence require real checkpoints and the training environment. No measured GPU-memory reduction or full-volume accuracy result is available from local editing. See [repository verification](VALIDATION.md) for results and [VAE validation](VAE_VALIDATION.md#reconstruction-accuracy-comparison) for the paired comparison procedure.
