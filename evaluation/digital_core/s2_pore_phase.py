import argparse
import glob
import os
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


# Pore phase definition:
# raw data is binary int8, where 0 = solid/skeleton and 1 = pore.
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
TESTB_FOLDER = str(DATA_DIR / 'real_binary')
DCB_FOLDER = str(DATA_DIR / 'baselines' / 'dcgan')
STYLE_FOLDER = str(DATA_DIR / 'baselines' / 'stylegan')
ST_FOLDER = str(DATA_DIR / 'baselines' / 'stgan')
GEB_FOLDER = str(DATA_DIR / 'generated_binary')

DIMENSION = (128, 128, 128)  # interpreted as (Z, Y, X)
DATA_TYPE = np.int8
PORE_VALUE = 1
MAX_R = 150

OUTPUT_DIR = "S2_results"
DEVICE = "auto"
FIGURE_DPI = 600
INSET_START_R = 60
INSET_END_R = 120

CURVE_KEYS = {
    "x": "S2_x",
    "y": "S2_y",
    "z": "S2_z",
    "radial": "S2_radial",
}

GROUP_STYLES = {
    "TI": {
        "color": "#FF0000",
        "linewidth": 1.6,
        "linestyle": "-",
        "marker": "o",
        "markersize": 5.5,
        "markerfacecolor": "white",
        "markeredgecolor": "#FF0000",
        "markeredgewidth": 1.8,
        "markevery": "turning_points",
        "alpha": 1.0,
        "zorder": 6,
    },
    "DCGAN": {
        "color": "#00E51A",
        "linewidth": 1.6,
        "linestyle": "-",
        "marker": "^",
        "markersize": 7.2,
        "markerfacecolor": "#00E51A",
        "markeredgecolor": "#00E51A",
        "markeredgewidth": 1.2,
        "markevery": "turning_points",
        "alpha": 1.0,
        "zorder": 3,
    },
    "STGAN": {
        "color": "#000000",
        "linewidth": 1.4,
        "linestyle": "-",
        "marker": "d",
        "markersize": 6.4,
        "markerfacecolor": "white",
        "markeredgecolor": "#000000",
        "markeredgewidth": 1.8,
        "markevery": "turning_points",
        "alpha": 1.0,
        "zorder": 4,
    },
    "StyleGAN": {
        "color": "#9A9A9A",
        "linewidth": 1.6,
        "linestyle": "-",
        "marker": "*",
        "markersize": 8.2,
        "markerfacecolor": "#9A9A9A",
        "markeredgecolor": "#9A9A9A",
        "markeredgewidth": 1.0,
        "markevery": "turning_points",
        "alpha": 0.75,
        "zorder": 2,
    },
    "ours": {
        "color": "#0000FF",
        "linewidth": 4.0,
        "linestyle": "-",
        "marker": None,
        "solid_capstyle": "round",
        "alpha": 1.0,
        "zorder": 5,
    },
}

PLOT_RC = {
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.linewidth": 1.1,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def load_pore_indicator(filepath, shape=DIMENSION, dtype=DATA_TYPE, pore_value=PORE_VALUE):
    raw = np.fromfile(filepath, dtype=dtype)
    expected_size = int(np.prod(shape))
    if raw.size != expected_size:
        raise ValueError(
            f"{filepath}: voxel count mismatch, got {raw.size}, expected {expected_size}. "
            f"Check DIMENSION or DATA_TYPE."
        )
    return (raw.reshape(shape) == pore_value).astype(np.float32)


def s2_direction(image, axis, max_r):
    """S2(r) = P(x in pore, x+r in pore), estimated along one Cartesian axis."""
    n = image.shape[axis]
    max_r = min(max_r, n - 1)
    values = np.empty(max_r + 1, dtype=np.float64)

    for r in range(max_r + 1):
        if r == 0:
            values[r] = image.mean()
            continue

        a_slice = [slice(None)] * image.ndim
        b_slice = [slice(None)] * image.ndim
        a_slice[axis] = slice(0, n - r)
        b_slice[axis] = slice(r, n)
        values[r] = np.mean(image[tuple(a_slice)] * image[tuple(b_slice)])

    return values


def s2_direction_torch(image_gpu, axis, max_r):
    """PyTorch GPU version of directional S2. Returns a NumPy array."""
    n = image_gpu.shape[axis]
    max_r = min(max_r, n - 1)
    values = torch.empty(max_r + 1, dtype=torch.float64, device=image_gpu.device)

    for r in range(max_r + 1):
        if r == 0:
            values[r] = image_gpu.mean()
            continue

        a_slice = [slice(None)] * image_gpu.ndim
        b_slice = [slice(None)] * image_gpu.ndim
        a_slice[axis] = slice(0, n - r)
        b_slice[axis] = slice(r, n)
        values[r] = torch.mean(image_gpu[tuple(a_slice)] * image_gpu[tuple(b_slice)])

    return values.cpu().numpy()


def linear_autocorrelation_fft(image):
    """Return non-periodic autocorrelation for all lags dz, dy, dx."""
    z, y, x = image.shape
    fft_shape = (2 * z - 1, 2 * y - 1, 2 * x - 1)
    spectrum = np.fft.fftn(image, s=fft_shape)
    corr = np.fft.ifftn(spectrum * np.conj(spectrum)).real
    corr[corr < 0] = 0.0

    dz = np.r_[np.arange(z), -np.arange(z - 1, 0, -1)]
    dy = np.r_[np.arange(y), -np.arange(y - 1, 0, -1)]
    dx = np.r_[np.arange(x), -np.arange(x - 1, 0, -1)]
    return corr, dz, dy, dx


def linear_autocorrelation_fft_torch(image_gpu):
    """PyTorch GPU version of non-periodic autocorrelation for all lags dz, dy, dx."""
    z, y, x = image_gpu.shape
    fft_shape = (2 * z - 1, 2 * y - 1, 2 * x - 1)
    spectrum = torch.fft.fftn(image_gpu, s=fft_shape)
    corr = torch.fft.ifftn(spectrum * torch.conj(spectrum)).real
    corr = torch.clamp(corr, min=0.0)

    device = image_gpu.device
    dz = torch.cat((torch.arange(z, device=device), -torch.arange(z - 1, 0, -1, device=device)))
    dy = torch.cat((torch.arange(y, device=device), -torch.arange(y - 1, 0, -1, device=device)))
    dx = torch.cat((torch.arange(x, device=device), -torch.arange(x - 1, 0, -1, device=device)))
    return corr, dz, dy, dx


def overlap_counts(dz, dy, dx, shape):
    z, y, x = shape
    counts_z = z - np.abs(dz).astype(np.float64)
    counts_y = y - np.abs(dy).astype(np.float64)
    counts_x = x - np.abs(dx).astype(np.float64)
    return counts_z[:, None, None] * counts_y[None, :, None] * counts_x[None, None, :]


def overlap_counts_torch(dz, dy, dx, shape):
    z, y, x = shape
    counts_z = z - torch.abs(dz).to(torch.float64)
    counts_y = y - torch.abs(dy).to(torch.float64)
    counts_x = x - torch.abs(dx).to(torch.float64)
    return counts_z[:, None, None] * counts_y[None, :, None] * counts_x[None, None, :]


def s2_radial_average(image, max_r):
    """Radial average of S2 over displacement vectors with |r| in integer bins."""
    z, y, x = image.shape
    max_r = min(max_r, z - 1, y - 1, x - 1)

    corr, dz, dy, dx = linear_autocorrelation_fft(image)
    counts = overlap_counts(dz, dy, dx, image.shape)
    s2_grid = corr / counts

    zz = dz[:, None, None].astype(np.float32)
    yy = dy[None, :, None].astype(np.float32)
    xx = dx[None, None, :].astype(np.float32)
    radius = np.sqrt(zz * zz + yy * yy + xx * xx)
    bins = np.rint(radius).astype(np.int16)
    mask = bins <= max_r

    sums = np.bincount(bins[mask].ravel(), weights=s2_grid[mask].ravel(), minlength=max_r + 1)
    nums = np.bincount(bins[mask].ravel(), minlength=max_r + 1)
    radial = sums / np.maximum(nums, 1)
    return radial.astype(np.float64)


def s2_radial_average_torch(image_gpu, max_r):
    """PyTorch GPU radial average of S2. Returns a NumPy array."""
    z, y, x = image_gpu.shape
    max_r = min(max_r, z - 1, y - 1, x - 1)

    corr, dz, dy, dx = linear_autocorrelation_fft_torch(image_gpu)
    counts = overlap_counts_torch(dz, dy, dx, image_gpu.shape)
    s2_grid = corr / counts

    zz = dz[:, None, None].to(torch.float32)
    yy = dy[None, :, None].to(torch.float32)
    xx = dx[None, None, :].to(torch.float32)
    radius = torch.sqrt(zz * zz + yy * yy + xx * xx)
    bins = torch.round(radius).to(torch.int64)
    mask = bins <= max_r

    sums = torch.bincount(bins[mask].ravel(), weights=s2_grid[mask].ravel(), minlength=max_r + 1)
    nums = torch.bincount(bins[mask].ravel(), minlength=max_r + 1)
    radial = sums / torch.clamp(nums, min=1)
    return radial.to(torch.float64).cpu().numpy()


def calculate_one_file(filepath, max_r, do_radial=True, backend="numpy"):
    image = load_pore_indicator(filepath)

    if backend == "torch":
        device = torch.device("cuda")
        image_gpu = torch.as_tensor(image, dtype=torch.float32, device=device)
        sx = s2_direction_torch(image_gpu, axis=2, max_r=max_r)
        sy = s2_direction_torch(image_gpu, axis=1, max_r=max_r)
        sz = s2_direction_torch(image_gpu, axis=0, max_r=max_r)
        radial = s2_radial_average_torch(image_gpu, max_r=max_r) if do_radial else np.full(max_r + 1, np.nan)
        torch.cuda.empty_cache()
    else:
        sx = s2_direction(image, axis=2, max_r=max_r)
        sy = s2_direction(image, axis=1, max_r=max_r)
        sz = s2_direction(image, axis=0, max_r=max_r)
        radial = s2_radial_average(image, max_r=max_r) if do_radial else np.full(max_r + 1, np.nan)

    return {
        "file": os.path.basename(filepath),
        "porosity": float(image.mean()),
        "x": sx,
        "y": sy,
        "z": sz,
        "radial": radial,
    }


def save_figure(output_path, dpi=FIGURE_DPI):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi)
    plt.savefig(output_path.with_suffix(".pdf"), dpi=dpi)


def style_axes(ax, x_min, x_max, hide_top_right=True):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(direction="out", length=4.5, width=1.0, colors="#222222")
    ax.tick_params(which="minor", direction="out", length=2.5, width=0.8, colors="#444444")
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.1)
    ax.spines["top"].set_visible(not hide_top_right)
    ax.spines["right"].set_visible(not hide_top_right)


def add_legend(ax, ncol=2):
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="#d0d0d0",
        ncol=ncol,
        handlelength=3.0,
        columnspacing=1.2,
        borderpad=0.55,
        labelspacing=0.45,
    )
    legend.get_frame().set_linewidth(0.7)


def draw_curve(ax, r, group_name, curves, key, label=True):
    style = GROUP_STYLES.get(group_name, {}).copy()
    if style.get("marker") is not None and style.get("markevery") == "turning_points":
        style["markevery"] = turning_point_markevery(r, curves[key])
    ax.plot(r, curves[key], label=group_name if label else None, **style)


def turning_point_markevery(x, y, target_step=4):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= target_step:
        return list(range(n))

    fallback = list(range(0, n, target_step))
    if n < 3 or np.nanmax(y) == np.nanmin(y):
        return fallback

    x_span = np.nanmax(x) - np.nanmin(x)
    y_span = np.nanmax(y) - np.nanmin(y)
    if x_span == 0 or y_span == 0:
        return fallback

    xn = (x - np.nanmin(x)) / x_span
    yn = (y - np.nanmin(y)) / y_span
    v1x = xn[1:-1] - xn[:-2]
    v1y = yn[1:-1] - yn[:-2]
    v2x = xn[2:] - xn[1:-1]
    v2y = yn[2:] - yn[1:-1]
    cross = np.abs(v1x * v2y - v1y * v2x)
    dot = v1x * v2x + v1y * v2y
    bend = np.arctan2(cross, dot)

    target_count = max(8, n // target_step)
    min_gap = max(2, target_step // 2)
    chosen = []
    for idx in np.argsort(bend)[::-1] + 1:
        if len(chosen) >= target_count:
            break
        if all(abs(idx - existing) >= min_gap for existing in chosen):
            chosen.append(int(idx))

    if len(chosen) < max(4, target_count // 3):
        return fallback
    return sorted(chosen)


def y_axis_label(key):
    suffixes = {
        "x": "-x",
        "y": "-y",
        "z": "-z",
        "radial": "-radial",
    }
    return f"S2(r){suffixes.get(key, '')}"


def curve_ylim(group_curves, key, r, x_min, x_max, pad=1.12):
    mask = (r >= x_min) & (r <= x_max)
    values = []
    for curves in group_curves.values():
        curve = np.asarray(curves[key])
        values.append(curve[mask])
    top = float(np.nanmax(np.concatenate(values)))
    return max(top * pad, 1e-6)


def curve_yrange(group_curves, key, r, x_min, x_max, pad=0.12):
    mask = (r >= x_min) & (r <= x_max)
    values = []
    for curves in group_curves.values():
        curve = np.asarray(curves[key])
        values.append(curve[mask])
    values = np.concatenate(values)
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    span = max(y_max - y_min, 1e-6)
    return max(0, y_min - span * pad), y_max + span * pad


def draw_plot_with_inset(output_path, group_curves, max_r, key, title, figsize=(7.2, 7.2)):
    import matplotlib.pyplot as plt

    with plt.rc_context(PLOT_RC):
        r = np.arange(max_r + 1)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        for group_name, curves in group_curves.items():
            draw_curve(ax, r, group_name, curves, key)

        ax.set_xlabel("Lag distance r (voxel)")
        ax.set_ylabel(y_axis_label(key))
        style_axes(ax, 0, max_r)
        ax.set_ylim(*curve_yrange(group_curves, key, r, 0, max_r, pad=0.06))
        add_legend(ax, ncol=1)

        inset_start = min(INSET_START_R, max_r - 5)
        inset_end = min(INSET_END_R, max_r)
        if inset_end <= inset_start:
            inset_end = max_r
        inset = ax.inset_axes([0.29, 0.43, 0.50, 0.44])
        for group_name, curves in group_curves.items():
            draw_curve(inset, r, group_name, curves, key, label=False)
        style_axes(inset, inset_start, inset_end, hide_top_right=False)
        inset.set_ylim(*curve_yrange(group_curves, key, r, inset_start, inset_end, pad=0.12))
        inset.tick_params(labelsize=8, length=3)
        inset.tick_params(which="minor", length=1.8)
        zoom_indicator = ax.indicate_inset_zoom(inset, edgecolor="#777777", linewidth=0.8, alpha=0.75)
        zoom_artists = []
        if hasattr(zoom_indicator, "rectangle"):
            zoom_artists.append(zoom_indicator.rectangle)
        if hasattr(zoom_indicator, "connectors"):
            zoom_artists.extend(zoom_indicator.connectors)
        if not zoom_artists:
            for artist in zoom_indicator:
                if isinstance(artist, tuple):
                    zoom_artists.extend(artist)
                else:
                    zoom_artists.append(artist)
        for artist in zoom_artists:
            if artist is not None:
                artist.set_linestyle("--")
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.88)
    save_figure(output_path)
    plt.close(fig)


def save_radial_plot(output_path, group_curves, max_r):
    draw_plot_with_inset(
        output_path=output_path,
        group_curves=group_curves,
        max_r=max_r,
        key="radial",
        title="Pore-phase two-point correlation",
        figsize=(7.2, 7.2),
    )


def save_individual_curve_plots(output_dir, group_curves, max_r):
    for key, curve_label in CURVE_KEYS.items():
        if key == "radial" and any(np.all(np.isnan(curves[key])) for curves in group_curves.values()):
            continue
        draw_plot_with_inset(
            output_path=output_dir / f"{curve_label}_all_groups.png",
            group_curves=group_curves,
            max_r=max_r,
            key=key,
            title=curve_label,
            figsize=(7.2, 7.2),
        )


def calculate_group(group_name, folder, max_r, max_files=None, do_radial=True, backend="numpy"):
    files = sorted(glob.glob(os.path.join(folder, "*.raw")))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No .raw files found in {folder}")

    print(f"\n[{group_name}] folder: {folder}")
    print(f"[{group_name}] files: {len(files)}")

    results = []
    for i, filepath in enumerate(files, start=1):
        print(f"  ({i}/{len(files)}) {os.path.basename(filepath)}")
        result = calculate_one_file(filepath, max_r=max_r, do_radial=do_radial, backend=backend)
        results.append(result)

    mean_curves = {
        "x": np.mean([res["x"] for res in results], axis=0),
        "y": np.mean([res["y"] for res in results], axis=0),
        "z": np.mean([res["z"] for res in results], axis=0),
        "radial": np.mean([res["radial"] for res in results], axis=0),
        "porosity_mean": float(np.mean([res["porosity"] for res in results])),
        "porosity_std": float(np.std([res["porosity"] for res in results], ddof=1)) if len(results) > 1 else 0.0,
        "file_count": len(results),
    }
    print(
        f"[{group_name}] mean porosity = {mean_curves['porosity_mean']:.6f}, "
        f"std = {mean_curves['porosity_std']:.6f}, phi^2 = {mean_curves['porosity_mean'] ** 2:.6f}"
    )
    return mean_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate pore-phase two-point correlation S2(r) for binary raw cores.")
    parser.add_argument("--testb-folder", default=TESTB_FOLDER, help="Folder of original TESTB .raw files.")
    parser.add_argument("--dcb-folder", default=DCB_FOLDER, help="Folder of DCB .raw files.")
    parser.add_argument("--style-folder", default=STYLE_FOLDER, help="Folder of STYLE .raw files.")
    parser.add_argument("--st-folder", default=ST_FOLDER, help="Folder of ST .raw files.")
    parser.add_argument("--geb-folder", default=GEB_FOLDER, help="Folder of GEB .raw files.")
    parser.add_argument("--max-r", type=int, default=MAX_R, help="Maximum lag distance in voxels.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for quick testing.")
    parser.add_argument("--no-radial", action="store_true", help="Skip FFT radial average and calculate only x/y/z directions.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--device",
        default=DEVICE,
        choices=("auto", "cpu", "gpu"),
        help="Compute device. 'auto' uses PyTorch GPU when available, otherwise CPU.",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_device(device):
    if device == "cpu":
        return "numpy"
    if torch is not None and torch.cuda.is_available():
        return "torch"
    if device == "gpu":
        raise RuntimeError("GPU requested, but PyTorch CUDA is not available in this Python environment.")
    return "numpy"


def backend_label(backend):
    labels = {
        "numpy": "CPU (NumPy)",
        "torch": "GPU (PyTorch)",
    }
    return labels[backend]


def main():
    args = parse_args()
    args.max_r = min(args.max_r, min(DIMENSION) - 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    do_radial = not args.no_radial
    backend = resolve_device(args.device)
    print(f"Device: {backend_label(backend)}")

    groups = {
        "TI": args.testb_folder,
        "DCGAN": args.dcb_folder,
        "STGAN": args.st_folder,
        "StyleGAN": args.style_folder,
        "ours": args.geb_folder,
    }

    group_curves = {}
    for group_name, folder in groups.items():
        curves = calculate_group(
            group_name=group_name,
            folder=folder,
            max_r=args.max_r,
            max_files=args.max_files,
            do_radial=do_radial,
            backend=backend,
        )
        group_curves[group_name] = curves

    if do_radial:
        save_radial_plot(output_dir / "S2_radial_all_groups_with_phi2.png", group_curves, args.max_r)
    save_individual_curve_plots(output_dir, group_curves, args.max_r)

    print("\nDone.")
    print(f"Output directory: {output_dir.resolve()}")
    print("Saved plots only. No tabular files are written.")


if __name__ == "__main__":
    main()
