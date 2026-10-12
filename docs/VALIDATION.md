# Repository verification

**Purpose:** Record the scope and results of repository checks, and distinguish them from training and scientific validation.

For model smoke checks and paired reconstruction comparisons, use [VAE validation](VAE_VALIDATION.md). For the complete document map, use the [documentation guide](README.md).

## Latest local review

Review date: **2026-10-12**. The editing environment provides Python and NumPy, but no PyTorch, Lightning or TorchIO installation. It contains no supplied trained GLOSSY checkpoints or training volumes.

| Check | Local result | What it establishes |
| --- | --- | --- |
| Python source compilation and local module paths | Passed for 45 Python files. | Source parses and repository import paths resolve statically. |
| Notebook structure, code-cell syntax and local module paths | Passed for 10 notebooks and 37 code cells. | Notebook sources are structurally readable; the `%pip` cell is treated as Jupyter syntax. Saved outputs and execution counts are cleared. |
| Optional JSON schemas | Passed for 3 examples. | Supplied dataset mappings and RAW metadata satisfy their validators. |
| Markdown purposes and local links | Passed for 13 Markdown files. | Every page explains its role and its local file/heading links resolve. |
| Standard-library regression tests | 28 passed: 12 JSON, 8 sampling and 8 direct-input tests. | Configuration, CLI options, direct paths and provenance validation behave as tested. |
| Documented Python command arguments | Argument definitions are checked without ML imports; real command execution requires the appropriate environment. | Example commands use accepted flags and required arguments; model execution is not exercised. |
| Small NumPy digital-core calculations | Passed 120 S2/MPC/FFT reference cases and 5 simple FID/MMD/normalization/IS cases. | Directional S2, FFT correlations, MPC and simple feature-distribution metric cases agree on the tested arrays. |

The current review also replaces notebook machine-specific paths with repository examples, removes a user-directory module search, and standardizes documentation navigation and checkpoint examples. Explicit reconstruction now saves NIfTI using the same normalized intensity range as RAW. VAE/codebook/diffusion model sources are unchanged in this review, and notebook computational AST comparisons preserve their formulas and numerical parameters.

Local NumPy checks use version 2.3.5; the supplied environment pins 1.26.3. FFT comparisons use float32 tolerances of `rtol=1e-6`, `atol=1e-5`. Repeat numerical checks in the installed project environment alongside tensor tests.

An isolated check executes the actual reconstruction saving block with NumPy tensor/image wrappers, verifies real big-endian RAW bytes and sidecar spacing, and checks the values passed to the NIfTI writer. It does not exercise actual TorchIO serialization. Positive/negative checker fixtures also verify detection of broken links, missing document purposes, invalid notebook Python and missing local module paths.

The original notebook extraction/crop now shares a project preprocessing module. Seven NumPy preprocessing cases pass, covering slab slicing, ZYX/XYZ conversion, big-endian RAW, overlap, crop bounds, 48/8/8 split and output protection. The added legacy diffusion regression requires PyTorch and remains unexecuted locally. Actual trained VAE/diffusion checkpoints are not supplied; compatibility-tool results remain pending on the server. Legacy dictionary/Namespace argument restoration and strict-load dispatch pass with explicit constructor doubles; this does not execute real tensor loading.

## Repeat checks without the ML environment

```bash
python tools/check_source.py
python -m unittest discover -s tests -p test_json_config.py -v
python -m unittest discover -s tests -p test_sampling_config.py -v
python -m unittest discover -s tests -p test_direct_inputs.py -v
```

Run `python -m unittest discover -s tests -p test_preprocessing.py -v` with NumPy installed for the additional seven cases.

The source checker uses only the standard library. It checks Python, notebook Python cells, local module paths, optional JSON schemas, Markdown purpose statements and local documentation links. It does not import training libraries, resolve external web links, inspect real checkpoint tensors or execute notebook experiments.

If using the optional JSON dataset mappings, replace their example paths first, then run `python tools/check_source.py --check-data-paths`. This additional check requires existing supported volumes; no data is bundled. Directory inputs can be exercised with the [real-data smoke tool](VAE_VALIDATION.md#real-data-training-smoke-check).

## Checks requiring the supplied ML environment

```bash
python -m pip check
python tools/check_source.py
python -m unittest discover -s tests -v
```

There are **70 unittest cases** in total: the 28 standard-library cases above, 7 NumPy preprocessing cases, 20 pipeline cases and 15 VAE training cases. The latter 35 require the ML dependencies and have **not** been executed locally. They cover synthetic tensor forward/backward paths, shapes, I/O, decoding, codebook state, optimizer isolation and checkpoint resume. The added reconstruction case checks RAW/NIfTI intensity and spacing consistency, including clipping and big-endian RAW output. Their perceptual fixture is intentionally small; it does not validate genuine pretrained features.

The [GitHub workflow](../.github/workflows/checks.yml) installs the CPU environment, checks dependencies, runs all tests and checks public command help. It has not been executed during this local edit. Optional digital-core notebook dependencies are installed separately through [requirements-evaluation.txt](../requirements-evaluation.txt).

## Remaining experimental verification

Real PyTorch/TorchIO execution, real checkpoint compatibility, CUDA/DDP training, peak memory, long-run convergence and end-to-end generation remain pending in the target environment. Run real-data smoke checks, paired held-out reconstruction comparisons and independent digital-core experiments before claiming preserved accuracy or reproduced manuscript metrics.

Full 1024-cubed decoding has not been memory-validated. Patch decoding changes decoder context and must be compared experimentally with full decoding. The repository does not supply baseline GAN implementations or the manuscript's external Avizo/PNM and ImageJ/MorphoLibJ analyses.
