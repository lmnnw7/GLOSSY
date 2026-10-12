# Digital core experiments

**Purpose:** List the standalone digital-core experiments and explain their dependencies, inputs and execution order.

Use the [project README](../../README.md) for model training/generation, and the [documentation guide](../../docs/README.md) for all Markdown roles. Use the FID guides linked below for detailed feature-distribution evaluation commands.

These notebooks/scripts cover core preprocessing and statistical evaluation. Numerical experiment parameters are retained. Example paths now use repository data/output directories rather than personal drive paths. Saved notebook outputs and execution counts are cleared.

## Files

| File | Purpose |
| --- | --- |
| [core_subvolume_extraction.ipynb](core_subvolume_extraction.ipynb) | Extract core subvolumes, crop a large region and evaluate 3D multiscale sliced Wasserstein distance. |
| [automatic_segmentation.ipynb](automatic_segmentation.ipynb) | Apply NLM filtering and constrained Kapur threshold segmentation. |
| [porosity.ipynb](porosity.ipynb) | Calculate per-volume porosity and summary statistics; includes an Otsu segmentation example. |
| [axial_porosity.ipynb](axial_porosity.ipynb) | Compare slice-wise porosity profiles. |
| [two_point_correlation.ipynb](two_point_correlation.ipynb) | Calculate and plot directional and radial two-point correlation functions. |
| [directional_correlation_3d.ipynb](directional_correlation_3d.ipynb) | Visualize XYZ directional two-point correlation curves in 3D. |
| [multipoint_correlation.ipynb](multipoint_correlation.ipynb) | Calculate and plot multipoint correlation statistics. |
| [diversity.ipynb](diversity.ipynb) | Compare sample diversity using multiscale sliced Wasserstein distances and embeddings. |
| [fid_3d_example.ipynb](fid_3d_example.ipynb) | Compare real and generated cores using 3D ResNet features, FID and MMD. |
| [fid_3d_uint16_example.ipynb](fid_3d_uint16_example.ipynb) | FID/MMD example for uint16 grayscale RAW volumes. |
| [fid_3d.py](fid_3d.py) | Shared 3D feature extraction and distribution metrics. |
| [s2_pore_phase.py](s2_pore_phase.py) | Shared pore-phase two-point correlation implementation. |
| [multipoint_metrics.py](multipoint_metrics.py) | Shared NumPy/PyTorch multipoint correlation, pore masks, RAW reading and REV threshold calculation. |

Block/crop execution is shared with [preprocessing/core_subvolume_extraction.py](../../preprocessing/core_subvolume_extraction.py); the original notebook calls that module. [prepare_core_blocks.py](../../preprocessing/prepare_core_blocks.py) adds fixed train/validation/test directories for model runs. Source `ZYX` can be declared explicitly and becomes XYZ output; check preprocessing consistency when reusing older models.

Shared Python modules provide reusable calculations for the notebooks. Copied S2 and MPC function bodies have been consolidated into the shared modules; each notebook retains its input settings and plotting variants. Existing figure folders, result arrays, extracted paper text and Python/notebook caches are not included in this code import.

## Open the notebooks

From the GLOSSY repository root, install the optional experiment dependencies in your selected Python environment:

```bash
python -m pip install -r requirements-evaluation.txt
cd evaluation/digital_core
jupyter notebook
```

Open a notebook, set the example data and checkpoint paths, then run the cells for the selected experiment or plot variant. Keep these notebooks and their helper scripts in this directory. Individual notebooks retain their original input conventions, dimensions, thresholds, dtype and pore-phase values; these standalone calculations are not connected to the model training loop.

Cells may provide separate experiment or plotting variants; read their headings before running them. Some cells are independent examples rather than a single sequential pipeline.

| Example directory, relative to the GLOSSY root | Intended input |
| --- | --- |
| `data/real_binary`, `data/generated_binary` | Real/generated binary cores for pore statistics. |
| `data/real_grayscale`, `data/generated_grayscale` | Real/generated uint16 grayscale cores for fixed-range FID. |
| `data/baselines/dcgan`, `data/baselines/stylegan`, `data/baselines/stgan` | Externally supplied baseline outputs. Baseline model implementations are not bundled. |
| `data/ablations/*` | Outputs from the corresponding ablation experiments. |
| `data/*_256`, `data/core_images_512`, `data/source_large.raw` | Size-specific extraction/evaluation examples; match the notebook's declared dimensions. |

Notebook strings use `../../data/...` and `../../outputs/...` because the documented working directory is `evaluation/digital_core`. The standalone S2 script derives default data paths from the repository location. Data and experiment outputs are not bundled.

For example, several binary-volume experiments use int8 data with `1=pore` and `0=solid`, while automatic segmentation writes uint8 values of 0 and 255. Configure each evaluation to match its actual inputs. The shared GLOSSY image loader uses its existing storage/model axis convention; these standalone notebooks retain their own axis interpretation.

See [general 3D FID usage](fid_3d_usage.md) and [uint16 grayscale FID usage](fid_3d_uint16_usage.md). The supplied minimal FID dependencies remain in [requirements_fid_3d.txt](requirements_fid_3d.txt).

See [repository verification](../../docs/VALIDATION.md) for executed checks and pending experiments. Dataset-scale results and pretrained feature/GPU calculations require the target environment.
