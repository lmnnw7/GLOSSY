# Optional JSON input formats

**Purpose:** Define optional dataset mappings and RAW sidecars, including validation rules and legacy compatibility.

Use [direct inputs](DIRECT_INPUTS.md) to run without JSON inputs. Use the [documentation guide](README.md) for training, decoding and validation topics.

The repository supplies three optional JSON examples. All main training, export and reconstruction paths accept direct inputs; see [commands without JSON inputs](DIRECT_INPUTS.md). Decoder mode remains a command-line option and defaults to `full`; see [sampling options](SAMPLING.md).

| File | Purpose | Fields to edit |
| --- | --- | --- |
| `config/REV_VQVAE_data.json` | VAE source volumes, grouped by dataset name | Each dataset's `images` paths |
| `config/SingleRes_dataset.json` | Diffusion class IDs, source images and default latent roots | Each numeric class's `images` and `latents` paths |
| `config/raw_meta.example.json` | Template for an individual RAW file or directory-wide metadata | Actual shape, dtype, byte order, intensity range and spacing |

## Dataset roots

VAE training and direct reconstruction accept named datasets:

```json
{
  "sandstone": {
    "images": ["data/core_images"]
  }
}
```

Diffusion export and training require nonnegative integer class IDs:

```json
{
  "0": {
    "images": ["data/core_images"],
    "latents": ["data/core_images_latents"]
  }
}
```

Both a single path and a nonempty path list are accepted. Each path may identify a directory or a supported volume file. Relative paths always resolve against the GLOSSY project root, including manifests stored elsewhere; absolute paths and `~` are also supported. Repeated equivalent paths within a list are removed while retaining their order.

Legacy entries such as `{"sandstone": "data/core_images"}` or `{"0": ["data/core_images"]}` remain supported. Diffusion training appends `_latents` for these legacy source paths. Explicit entries read `latents` directly and require that field for training. Empty roots, malformed entries and duplicate numeric class IDs are rejected. Directory-based training reads exported `.meta.pt` provenance directly. An optional exported `training_latents.json` can group multiple classes and roots. Exporting to a custom `--output-root` does not update the default paths in `SingleRes_dataset.json`.

## RAW metadata

Copy `raw_meta.example.json` to `core.raw.json` for a file named `core.raw`, then replace the example values with the actual storage information. Directory defaults in `raw_meta.json` or `_raw_meta.json` are merged before per-file sidecars. Sidecars may contain only their overrides; validation runs after merging.

`shape` requires three positive integers in storage `[X,Y,Z]` order. The supported `axis_order` is `XYZ` and the array uses C order (Z varies fastest). Other layouts require explicit conversion before use. The input bytes are not silently rearranged to match another axis declaration.

Supported dtypes are `uint16`, `int16` and `float32`, with `u16`, `i16` and `f32` aliases. Byte order accepts `little`/`big` and `le`/`be` aliases. Misspelled byte orders are rejected instead of being treated as big endian. Spacing requires three finite positive values, with units `um` or `mm`; the default unit is `mm`. Generated digital-core metadata uses the configured micrometre spacing.

`value_range` requires two finite numbers with the maximum greater than the minimum. An optional `scale` must be finite and positive. Normalization priority is `value_range`, then `scale`, then dtype defaults. Signed integers require an explicit `value_range` or `scale`. Float inputs without either must already be in `[0,1]`. Non-finite source values and output ranges that would overflow an integer dtype are rejected.

All JSON readers support UTF-8 with or without BOM. Duplicate keys, `NaN`, infinities and non-object roots are rejected. Latent manifest provenance fields are preserved. Generated JSON outputs use standard finite JSON. Directory export records provenance in `.meta.pt` files instead.

## Checks

Run from the project root:

```bash
python tools/check_source.py
python -m unittest discover -s tests -p test_json_config.py -v
```

These commands require only the Python standard library. See [repository verification](VALIDATION.md) for current results and the complete test inventory.

If using the optional JSON mappings, replace their example paths with real source/latent paths, then run:

```bash
python tools/check_source.py --check-data-paths
```

This path check requires supported volumes in the configured directories; it does not inspect NIfTI tensors or validate model execution. Supplying `--raw-shape` in a runtime command bypasses RAW sidecars entirely. Without it, CLI RAW fields override merged sidecar values.
