import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "This script needs PyTorch. Install it in the notebook/kernel environment first."
    ) from exc


SUPPORTED_EXTS = (".npy", ".npz", ".raw", ".dat", ".bin", ".img")


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        dtype=out.dtype,
        device=out.device,
    )
    return torch.cat([out, zero_pads], dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut_type="B"):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            if shortcut_type == "A":
                self.downsample = lambda x: downsample_basic_block(
                    x, planes * self.expansion, stride
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv3d(
                        in_planes,
                        planes * self.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * self.expansion),
                )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut_type="B"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            if shortcut_type == "A":
                self.downsample = lambda x: downsample_basic_block(
                    x, planes * self.expansion, stride
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv3d(
                        in_planes,
                        planes * self.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * self.expansion),
                )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet3D(nn.Module):
    def __init__(self, block, layers, shortcut_type="B", in_channels=1, num_classes=0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.feature_dim = 512 * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes) if num_classes else None

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        layers = [block(self.in_planes, planes, stride, shortcut_type)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1, shortcut_type))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward_logits(self, x):
        if self.fc is None:
            raise ValueError("This model has no classification head. Set is_num_classes > 0.")
        return self.fc(self.forward_features(x))

    def forward(self, x):
        return self.forward_features(x)


def build_resnet(depth, shortcut_type="B", in_channels=1, num_classes=0):
    specs = {
        10: (BasicBlock, [1, 1, 1, 1]),
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
        152: (Bottleneck, [3, 8, 36, 3]),
        200: (Bottleneck, [3, 24, 36, 3]),
    }
    if depth not in specs:
        raise ValueError(f"Unsupported ResNet depth {depth}. Choose from {sorted(specs)}.")
    block, layers = specs[depth]
    return ResNet3D(block, layers, shortcut_type=shortcut_type, in_channels=in_channels, num_classes=num_classes)


def strip_state_dict(state, keep_classifier=False):
    if isinstance(state, dict):
        for key in ("state_dict", "net", "model", "resnet"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    clean = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "resnet."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key.startswith("classifier."):
            new_key = "fc." + new_key[len("classifier.") :]
        if not keep_classifier and new_key.startswith("fc."):
            continue
        clean[new_key] = value
    return clean


def load_feature_model(args, device):
    model = build_resnet(
        args.depth,
        shortcut_type=args.shortcut,
        in_channels=1,
        num_classes=getattr(args, "is_num_classes", 0),
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            strip_state_dict(checkpoint, keep_classifier=getattr(args, "is_num_classes", 0) > 0),
            strict=False,
        )
        print(f"Loaded checkpoint: {args.checkpoint}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        if getattr(args, "is_num_classes", 0) > 0 and any(key.startswith("fc.") for key in missing):
            print("WARNING: classifier head was not loaded completely; IS may be meaningless.")
    else:
        print("WARNING: no checkpoint was provided; FID will use randomly initialized features.")
        print("Pass --checkpoint path/to/pretrained_3d_resnet.pth for pretrained-feature FID.")
    model.to(device)
    model.eval()
    return model


def list_volume_files(folder):
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(glob.glob(str(Path(folder) / f"*{ext}")))
    return sorted(files)


def load_volume(path, shape=None, dtype="int8", npz_key=None):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = npz_key or data.files[0]
        arr = data[key]
    elif suffix in (".raw", ".dat", ".bin", ".img"):
        if shape is None:
            raise ValueError(f"--shape is required for raw binary file: {path}")
        arr = np.fromfile(path, dtype=np.dtype(dtype)).reshape(tuple(shape))
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape} from {path}")
    return arr


def normalize_volume(arr, mode, intensity_range=None):
    arr = arr.astype(np.float32, copy=False)
    if mode == "none":
        return arr
    if intensity_range is not None:
        low, high = intensity_range
        if high <= low:
            raise ValueError("intensity_range must be (low, high) with high > low.")
        arr = (arr - float(low)) / (float(high) - float(low))
        arr = np.clip(arr, 0.0, 1.0)
        if mode == "zero_one":
            return arr
        if mode == "minus_one_one":
            return arr * 2.0 - 1.0
        raise ValueError(f"Unknown normalize mode: {mode}")
    if mode == "zero_one":
        amin = float(np.min(arr))
        amax = float(np.max(arr))
        if amax <= amin:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - amin) / (amax - amin)
    if mode == "minus_one_one":
        arr = normalize_volume(arr, "zero_one", intensity_range=intensity_range)
        return arr * 2.0 - 1.0
    raise ValueError(f"Unknown normalize mode: {mode}")


def prepare_tensor(arr, target_size, normalize, intensity_range=None):
    arr = normalize_volume(arr, normalize, intensity_range=intensity_range)
    tensor = torch.from_numpy(arr).float()[None, None]
    if target_size is not None and tuple(arr.shape) != tuple(target_size):
        tensor = F.interpolate(
            tensor,
            size=tuple(target_size),
            mode="trilinear",
            align_corners=False,
        )
    return tensor


@torch.no_grad()
def extract_features(folder, model, device, args):
    files = list_volume_files(folder)
    if not files:
        raise ValueError(f"No volume files found in {folder}. Supported: {SUPPORTED_EXTS}")

    features = []
    batch = []
    for idx, file_path in enumerate(files, start=1):
        arr = load_volume(file_path, shape=args.shape, dtype=args.dtype, npz_key=args.npz_key)
        batch.append(prepare_tensor(arr, args.resize, args.normalize, args.intensity_range))
        if len(batch) == args.batch_size or idx == len(files):
            x = torch.cat(batch, dim=0).to(device)
            feat = model(x).detach().cpu().numpy().astype(np.float64)
            features.append(feat)
            batch = []
            print(f"{folder}: extracted {idx}/{len(files)}", end="\r")
    print()
    return np.concatenate(features, axis=0), files


@torch.no_grad()
def extract_probabilities(folder, model, device, args):
    if model.fc is None:
        raise ValueError("IS requires a classification head. Set is_num_classes to the number of classes.")

    files = list_volume_files(folder)
    if not files:
        raise ValueError(f"No volume files found in {folder}. Supported: {SUPPORTED_EXTS}")

    probabilities = []
    batch = []
    for idx, file_path in enumerate(files, start=1):
        arr = load_volume(file_path, shape=args.shape, dtype=args.dtype, npz_key=args.npz_key)
        batch.append(prepare_tensor(arr, args.resize, args.normalize, args.intensity_range))
        if len(batch) == args.batch_size or idx == len(files):
            x = torch.cat(batch, dim=0).to(device)
            logits = model.forward_logits(x)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float64)
            probabilities.append(probs)
            batch = []
            print(f"{folder}: extracted IS probabilities {idx}/{len(files)}", end="\r")
    print()
    return np.concatenate(probabilities, axis=0)


def mean_and_cov(features, eps=1e-6):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]], dtype=np.float64)
    sigma = sigma + np.eye(sigma.shape[0], dtype=np.float64) * eps
    return mu, sigma


def calculate_fid(real_features, fake_features, eps=1e-6):
    mu1, sigma1 = mean_and_cov(real_features, eps)
    mu2, sigma2 = mean_and_cov(fake_features, eps)
    diff = mu1 - mu2
    product = sigma1.dot(sigma2)
    eigvals = np.linalg.eigvals(product)
    sqrt_trace = np.sum(np.sqrt(np.clip(np.real(eigvals), a_min=0.0, a_max=None)))
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * sqrt_trace
    return float(np.real(fid))


def calculate_mmd_rbf(real_features, fake_features, gamma=None):
    x = real_features
    y = fake_features
    if gamma is None:
        combined = np.concatenate([x, y], axis=0)
        sample = combined[: min(len(combined), 512)]
        distances = squared_distances(sample, sample)
        median = np.median(distances[distances > 0]) if np.any(distances > 0) else 1.0
        gamma = 1.0 / (2.0 * median)
    kxx = np.exp(-gamma * squared_distances(x, x))
    kyy = np.exp(-gamma * squared_distances(y, y))
    kxy = np.exp(-gamma * squared_distances(x, y))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean()), float(gamma)


def calculate_inception_score(probabilities, splits=10, eps=1e-16):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n = len(probabilities)
    if n < 2:
        raise ValueError("IS needs at least two generated volumes.")
    splits = max(1, min(int(splits), n))
    scores = []
    for part in np.array_split(probabilities, splits):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0


def squared_distances(a, b):
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * a.dot(b.T), 0.0)


def parse_size(values):
    if values is None:
        return None
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) == 3:
        return tuple(values)
    raise argparse.ArgumentTypeError("Use one value N or three values D H W.")


def normalize_size_arg(value):
    if value is None:
        return None
    if isinstance(value, int):
        return parse_size([value])
    return parse_size(list(value))


def normalize_range_arg(value):
    if value is None:
        return None
    values = list(value)
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Use two values LOW HIGH for intensity range.")
    return (float(values[0]), float(values[1]))


def save_features(path, real_features, fake_features, real_files, fake_files):
    np.savez_compressed(
        path,
        real_features=real_features,
        fake_features=fake_features,
        real_files=np.array(real_files, dtype=object),
        fake_files=np.array(fake_files, dtype=object),
    )


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_config(
    real,
    fake,
    checkpoint="",
    depth=50,
    shortcut="B",
    shape=None,
    dtype="int8",
    npz_key=None,
    resize=(128, 128, 128),
    normalize="zero_one",
    intensity_range=None,
    batch_size=2,
    device="auto",
    output=None,
    save_features_path="",
    is_num_classes=0,
    is_splits=10,
):
    if shape is not None:
        shape = normalize_size_arg(shape)
    if resize is not None:
        resize = normalize_size_arg(resize)
    intensity_range = normalize_range_arg(intensity_range)
    return SimpleNamespace(
        real=real,
        fake=fake,
        checkpoint=checkpoint,
        depth=depth,
        shortcut=shortcut,
        shape=shape,
        dtype=dtype,
        npz_key=npz_key,
        resize=resize,
        normalize=normalize,
        intensity_range=intensity_range,
        batch_size=batch_size,
        device=device,
        output=output,
        save_features=save_features_path,
        is_num_classes=is_num_classes,
        is_splits=is_splits,
    )


def compute_fid_3d(
    real,
    fake,
    checkpoint="",
    depth=50,
    shortcut="B",
    shape=None,
    dtype="int8",
    npz_key=None,
    resize=(128, 128, 128),
    normalize="zero_one",
    intensity_range=None,
    batch_size=2,
    device="auto",
    output=None,
    save_features_path="",
    is_num_classes=0,
    is_splits=10,
):
    """Compute 3D FID/MMD from a notebook or Python script.

    Parameters are the same as the command-line arguments. For raw binary
    volumes, pass shape=(128, 128, 128). For .npy/.npz volumes, shape can be None.
    """
    args = make_config(
        real=real,
        fake=fake,
        checkpoint=checkpoint,
        depth=depth,
        shortcut=shortcut,
        shape=shape,
        dtype=dtype,
        npz_key=npz_key,
        resize=resize,
        normalize=normalize,
        intensity_range=intensity_range,
        batch_size=batch_size,
        device=device,
        output=output,
        save_features_path=save_features_path,
        is_num_classes=is_num_classes,
        is_splits=is_splits,
    )
    torch_device = resolve_device(args.device)
    model = load_feature_model(args, torch_device)
    real_features, real_files = extract_features(args.real, model, torch_device, args)
    fake_features, fake_files = extract_features(args.fake, model, torch_device, args)

    if len(real_features) < 2 or len(fake_features) < 2:
        raise ValueError("FID needs at least two real and two generated volumes.")

    fid = calculate_fid(real_features, fake_features)
    mmd, gamma = calculate_mmd_rbf(real_features, fake_features)
    result = {
        "fid": fid,
        "mmd_rbf": mmd,
        "mmd_gamma": gamma,
        "num_real": int(len(real_features)),
        "num_fake": int(len(fake_features)),
        "feature_dim": int(real_features.shape[1]),
        "real_folder": str(Path(args.real).resolve()),
        "fake_folder": str(Path(args.fake).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "",
        "resnet_depth": args.depth,
        "resize": args.resize,
        "normalize": args.normalize,
        "intensity_range": args.intensity_range,
    }

    if args.is_num_classes:
        fake_probabilities = extract_probabilities(args.fake, model, torch_device, args)
        is_mean, is_std = calculate_inception_score(fake_probabilities, splits=args.is_splits)
        result.update(
            {
                "inception_score_fake_mean": is_mean,
                "inception_score_fake_std": is_std,
                "inception_score_splits": int(max(1, min(args.is_splits, len(fake_probabilities)))),
                "is_num_classes": int(args.is_num_classes),
                "is_note": "IS uses the checkpoint classification head. For digital rock, interpret cautiously unless the classifier is domain-appropriate.",
            }
        )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if args.save_features:
        save_features(args.save_features, real_features, fake_features, real_files, fake_files)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate 3D FID/MMD for digital rock volumes using a pretrained 3D ResNet feature extractor."
    )
    parser.add_argument("--real", required=True, help="Folder with real digital rock volumes.")
    parser.add_argument("--fake", required=True, help="Folder with generated digital rock volumes.")
    parser.add_argument("--checkpoint", default="", help="Pretrained 3D ResNet .pth/.pt file.")
    parser.add_argument("--depth", type=int, default=50, help="3D ResNet depth: 10, 18, 34, 50, 101, 152, 200.")
    parser.add_argument("--shortcut", choices=["A", "B"], default="B")
    parser.add_argument("--shape", type=int, nargs="+", default=None, help="Raw binary volume shape: D H W, e.g. 128 128 128.")
    parser.add_argument("--dtype", default="int8", help="Raw binary dtype, e.g. int8, uint8, float32.")
    parser.add_argument("--npz-key", default=None, help="Array key to read from .npz files. Default: first array.")
    parser.add_argument("--resize", type=int, nargs="+", default=[128, 128, 128], help="Resize to N or D H W. Use --resize 0 to disable.")
    parser.add_argument("--normalize", choices=["zero_one", "minus_one_one", "none"], default="zero_one")
    parser.add_argument("--intensity-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"), help="Fixed intensity range before normalization, e.g. 0 65535 for uint16 grayscale raw.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--output", default="fid_3d_results.json")
    parser.add_argument("--save-features", default="", help="Optional .npz path for extracted features.")
    parser.add_argument("--is-num-classes", type=int, default=0, help="Set >0 to calculate IS using the checkpoint classification head.")
    parser.add_argument("--is-splits", type=int, default=10, help="Number of splits for Inception Score.")
    args = parser.parse_args()

    args.shape = parse_size(args.shape)
    args.resize = None if args.resize == [0] else parse_size(args.resize)
    result = compute_fid_3d(
        real=args.real,
        fake=args.fake,
        checkpoint=args.checkpoint,
        depth=args.depth,
        shortcut=args.shortcut,
        shape=args.shape,
        dtype=args.dtype,
        npz_key=args.npz_key,
        resize=args.resize,
        normalize=args.normalize,
        intensity_range=args.intensity_range,
        batch_size=args.batch_size,
        device=args.device,
        output=args.output,
        save_features_path=args.save_features,
        is_num_classes=args.is_num_classes,
        is_splits=args.is_splits,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
