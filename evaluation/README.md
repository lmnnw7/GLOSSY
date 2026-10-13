# Digital Core Evaluation

**Axial porosity distribution comparison · Two-point correlation (S2) · REV validation**

This document describes axial porosity distribution comparison, pore-phase two-point correlation (S2), and MPC-based representative elementary volume (REV) validation in `evaluation/digital_core/`. The experiments use prepared binary core data. Installation, model training and sample generation are described in the [project README](../README.md).

| Experiment | Notebook | Main result |
| :--- | :--- | :--- |
| Axial porosity distribution comparison | [axial_porosity.ipynb](digital_core/axial_porosity.ipynb) | Porosity across axial segments |
| Two-point correlation S2 | [two_point_correlation.ipynb](digital_core/two_point_correlation.ipynb) | Directional and radial S2 curves |
| MPC-based REV validation | [multipoint_correlation.ipynb](digital_core/multipoint_correlation.ipynb) | MPC curves and convergence distances |

## 1. Model evaluation utilities

| Script | Purpose |
| :--- | :--- |
| [ae_reconstruction_check.py](ae_reconstruction_check.py) | VAE forward reconstruction, per-volume and mean MAE/PSNR, reconstructed and error volumes. |
| [ae_encode_decode_recon.py](ae_encode_decode_recon.py) | VAE patch encoding and reconstruction with full, sliding-window or Gaussian-weighted decoding. |
| [compare_vae.py](compare_vae.py) | Paired comparison of two VAE checkpoints using MAE/RMSE/PSNR. |
| [check_diffusion_latents.py](check_diffusion_latents.py) | Diffusion noise-prediction loss on held-out latents using EMA weights, with a CSV report. |

## 2. Binary-core inputs

Provide one headerless binary RAW file per real, generated or baseline core. Volume dimensions, voxel spacing and axis orientation must be consistent across methods. Declare the data type and pore label, use the same pore definition, and record the preprocessing parameters.

Organize the input files as follows, using paths relative to the project root:

| Directory | Contents |
| :--- | :--- |
| `data/real_binary/` | Binary core data from real samples. |
| `data/generated_binary/` | Binary core data from GLOSSY outputs. |
| `data/baselines/` | Binary core data from the baseline methods included in the comparison. |

Each file contains one volume with consistent dimensions and a declared pore label. The examples use 128³ int8 RAW with `1=pore, 0=solid`. For uint8 binary data with `255=pore, 0=solid`, use `DATA_TYPE=np.uint8` and `PORE_VALUE=255`. A 128³ one-byte RAW volume contains 2 MiB. Store only binary RAW core files in these directories and match sample counts across methods.

Record the stored array order. S2 and axial porosity distribution calculations interpret `DIMENSION` as `(Z,Y,X)`, with array axes 0, 1 and 2 corresponding to Z, Y and X. For XYZ files, reshape using the stored `(X,Y,Z)` dimensions and apply `transpose(2,1,0)` in the loading function before S2 or axial porosity distribution calculation. Changing dimension labels alone does not reorder voxels. MPC axis titles must match its actual array order. The same physical directions must be compared across all methods.

## 3. Start the notebook experiments

From the project root:

```bash
conda activate GLOSSY

cd evaluation/digital_core

jupyter notebook
```

Notebook paths such as `../../data/real_binary` refer to this working directory. Configure parameters before calculation. Restart the kernel when switching experimental variants to avoid reusing earlier variable values. Include only methods with available data in group dictionaries; `TI` denotes real cores and `ours` denotes GLOSSY outputs.

Run axial porosity distribution comparisons, S2 comparisons and MPC-based REV validation on the same binary-core groups.

## 4. Axial porosity distribution comparison

### Configure inputs

Open [axial_porosity.ipynb](digital_core/axial_porosity.ipynb). The notebook reads binary core data and compares local porosity across axial segments. The example uses the Z direction, matching the manuscript experiment, and calculates the slice pore fractions and mean porosity of each layer range.

| Parameter | 128³ example |
| :--- | :--- |
| `FOLDERS` | `TI: ../../data/real_binary`, `ours: ../../data/generated_binary`; include available baseline groups |
| `DIMENSION` | `(128,128,128)`, interpreted as ZYX |
| `DATA_TYPE`, `PORE_VALUE` | `np.int8`, `1` |
| `LAYER_AXIS` | `0` for Z, `1` for Y, `2` for X |
| `LAYER_RANGES` | `[(0,32),(32,64),(64,96),(96,128)]` |
| `MAX_FILES_PER_GROUP` | `None`, using all files in each group |

For uint8 0/255 binary data, use `DATA_TYPE=np.uint8` and `PORE_VALUE=255`. Keep the physical direction and layer ranges consistent across methods, and update the ranges when changing volume size. For XYZ input files, apply the same axis conversion described in the input preparation section.

### Calculate and compare

Run the three code cells in order:

1. Configure input directories, volume dimensions and layer ranges.
2. Calculate each sample's slice porosity and the group mean and standard deviation.
3. Average the group mean sequence over each layer interval and plot the comparison.

### Outputs and interpretation

Outputs are written to the notebook working directory:

| Output | Contents |
| :--- | :--- |
| `porosity_layer_range_bar_values.csv` | Mean porosity for each layer interval and group |
| `porosity_layer_range_bar_chart.png` / `.pdf` | Layer-range comparison chart |

Dashed lines show the real-core TI reference, and hatched regions show deficits below that reference. Compare directional porosity variations using identical slice intervals. The calculated slice-wise mean and standard deviation remain in `results`.

## 5. Two-point correlation S2

### Configure and calculate

Open [two_point_correlation.ipynb](digital_core/two_point_correlation.ipynb). The first code cell performs the basic calculation and plotting; subsequent cells provide alternative comparison plots and curve-area visualizations. Select the required plotting variant.

| Parameter | 128³ example |
| :--- | :--- |
| `TESTB_FOLDER` | `../../data/real_binary` |
| `GEB_FOLDER` | `../../data/generated_binary` |
| `DIMENSION` | `(128,128,128)`, interpreted as ZYX |
| `DATA_TYPE`, `PORE_VALUE` | `np.int8`, `1` |
| `MAX_R` | `127` voxels |
| `DEVICE` | `"auto"` |
| `OUTPUT_DIR` | `S2_results` |

For uint8 0/255 binary data, change both dtype and pore value as described above. Configure `DCB_FOLDER`, `STYLE_FOLDER` and `ST_FOLDER` when those baselines are included. Keep only participating methods in the `groups` dictionary; a two-group comparison uses `TI` and `ours`.

Run the selected calculation cell. The implementation in [s2_pore_phase.py](digital_core/s2_pore_phase.py) averages valid voxel pairs without periodic wrapping. Directional lags cannot exceed the corresponding axis length minus one. Keep volume size, phase definition and lag range identical across methods.

### Outputs and interpretation

S2 is the probability that two locations separated by distance r both lie in the pore phase. The notebook calculates X/Y/Z directional curves and a radial average. `S2(0)` equals the pore fraction. The porosity-squared reference line assists interpretation of long-range behavior.

Compare generated and real curves over the same lag range, including their shapes, decay behavior and directional differences. Distances are in voxels; convert directional distances to physical lengths using the spacing along the corresponding axis.

Figures are saved under `OUTPUT_DIR`. The basic calculation does not automatically export a numerical table; preserve the notebook's curve arrays when tabulated results are needed. Curve-area visualizations summarize the computed S2 curves.

### Three-dimensional presentation

For directional S2 figures, use [directional_correlation_3d.ipynb](digital_core/directional_correlation_3d.ipynb). Set `DIMENSION`, `DATA_TYPE` and `PORE_VALUE` at the top of `s2_pore_phase.py` to the input binary format before importing the module, then restart the kernel and configure the notebook's `GROUPS`. The reader captures these settings as function defaults.

The notebook calculates directional S2 and writes PNG/PDF figures to `S2_results_3D`. It does not perform the radial FFT calculation. Use the same groups and pore-phase settings as the two-dimensional comparison.

## 6. MPC-based REV validation

### Select the experimental cell

Open [multipoint_correlation.ipynb](digital_core/multipoint_correlation.ipynb). Its first three code cells contain CPU, GPU 128³ and GPU 256³ variants. For MPC/REV figures, select the first code cell containing both `OUTPUT_DIR` and `REV_STABLE_TAIL` (the fourth code cell). The following cell provides an alternative figure layout.

| Parameter | 128³ example |
| :--- | :--- |
| `TESTB_FOLDER` | `../../data/real_binary`, then repeat for each generated/baseline group |
| `DIMENSION`, `DATA_TYPE` | `(128,128,128)`, `np.int8` |
| `PORE_DEFINITION`, `PORE_VALUE` | `"equal"`, `1` |
| `H_FIXED` | `1` voxel |
| `N_VALUES` | `np.arange(1,129)` |
| `MAX_R` | `127` voxels |
| `BETA`, `REV_STABLE_TAIL` | `0.0001`, `True` |
| `OUTPUT_DIR` | A separate directory for each group, such as `MPC_results_real` |

The dtype and pore value must match the input files. Match axis titles to the actual array order. Use the same point spacing, point-count range and convergence threshold for all groups. For 256³ data, update `DIMENSION`, `N_VALUES` and `MAX_R` together. Set `DEVICE="cpu"` when using the CPU fallback of the selected cell.

### Calculate and identify the convergence distance

MPC is the fraction of valid starting locations for which all n points spaced h voxels apart lie in the pore phase. The plotted span is `(n-1)h`; samples beyond the volume bounds are excluded. Run the selected cell separately for real, generated and baseline directories, using a separate output directory for each group.

The selected cell identifies a convergence distance when adjacent MPC changes are no greater than `BETA`. With `REV_STABLE_TAIL=True`, all subsequent changes must also satisfy the threshold. It reports the largest detected distance across samples in each direction and the overall maximum, and saves PNG/PDF/SVG plots. Common calculations are in [multipoint_metrics.py](digital_core/multipoint_metrics.py).

### Compare and report

Compare the MPC curves and detected convergence distances across methods using identical settings. This is the notebook's MPC-based REV estimate; report `H_FIXED`, the n range, `BETA`, the stable-tail rule, axis mapping and sample count alongside the result. If no qualifying distance is found, report it as undetected rather than zero. Convert the detected span to physical length using the relevant voxel spacing when needed.
