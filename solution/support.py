"""Данные BSD500, исходная QECNN и единый протокол сравнения по RGB-PSNR."""

import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 480, 320
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2


class YUVFile:
    """8-битный планарный YUV420, ровно один кадр в каждой строке memmap."""

    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        if size == 0 or size % FRAME_BYTES:
            raise ValueError(f"Invalid YUV420 file length ({size} bytes): {self.path}")
        self.frames = np.memmap(self.path, dtype=np.uint8, mode="r",
                                shape=(size // FRAME_BYTES, FRAME_BYTES))

    def __len__(self):
        return len(self.frames)

    def planes(self, index):
        n = WIDTH * HEIGHT
        a = self.frames[index]
        y = a[:n].reshape(HEIGHT, WIDTH)
        u = a[n:n + n // 4].reshape(HEIGHT // 2, WIDTH // 2)
        v = a[n + n // 4:].reshape(HEIGHT // 2, WIDTH // 2)
        return y, u, v

    def rgb(self, index):
        return rgb_from_planes(*self.planes(index))


def rgb255_from_planes(y, u, v):
    # As in the supplied lab, OpenCV interpolates the *uint8* chroma planes.
    size = (y.shape[1], y.shape[0])
    u = cv2.resize(u, size, interpolation=cv2.INTER_CUBIC) - 128.0
    v = cv2.resize(v, size, interpolation=cv2.INTER_CUBIC) - 128.0
    rgb = np.stack([y + 1.4075 * v,
                    y - 0.3455 * u - 0.7169 * v,
                    y + 1.7790 * u], axis=-1)
    return np.clip(rgb, 0, 255)


def rgb_from_planes(y, u, v):
    return np.ascontiguousarray(rgb255_from_planes(y, u, v).astype(np.float32) / 255.0)


def paired_frames(root, split):
    root = Path(root)
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    name = "BSD500train" if split == "train" else "BSD500val"
    comp = YUVFile(root / f"{split}compyuv" / f"{name}c.yuv")
    raw = YUVFile(root / f"{split}rawyuv" / f"{name}.yuv")
    if len(raw) != len(comp):
        raise ValueError(f"Unpaired {split} files: {len(raw)} raw / {len(comp)} compressed")
    return comp, raw


def psnr(reference, predicted):
    """RGB [0,1], все пиксели. Результат сети не округляется и не обрезается."""
    if reference.shape != predicted.shape:
        raise ValueError(f"PSNR shape mismatch: {reference.shape} vs {predicted.shape}")
    error = np.subtract(reference, predicted, dtype=np.float64)
    return float(-10.0 * np.log10(np.mean(error * error) + 1e-10))


def baseline_predictor(root, device="auto"):
    """Точная архитектура стартовой QECNN, имена модулей совместимы с .pth."""
    import torch
    from torch import nn

    class QECNN(nn.Module):
        def __init__(self):
            super().__init__()
            previous = 3
            for i, (channels, kernel) in enumerate([(128, 9), (64, 7), (64, 3), (32, 1)], 1):
                for suffix, inputs in [(str(i), previous), (str(i) * 2, previous if i == 1 else 2 * previous)]:
                    setattr(self, f"conv{suffix}", nn.Conv2d(inputs, channels, kernel, padding=kernel // 2))
                    setattr(self, f"prelu{suffix}", nn.PReLU(channels))
                previous = channels
            self.conv_out = nn.Conv2d(64, 3, 5, padding=2)

        def forward(self, x):
            single = both = x
            for i in range(1, 5):
                a = getattr(self, f"prelu{i}")(getattr(self, f"conv{i}")(single))
                b = getattr(self, f"prelu{i}{i}")(getattr(self, f"conv{i}{i}")(both))
                single, both = a, torch.cat([a, b], dim=1)
            return x + self.conv_out(both)

    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "mps" if torch.backends.mps.is_available() else "cpu")
    model = QECNN()
    model.load_state_dict(torch.load(Path(root) / "best_model2026.pth", weights_only=True,
                                     map_location="cpu"))
    model.to(device, memory_format=torch.channels_last).eval()

    @torch.inference_mode()
    def predict(rgb):
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device, memory_format=torch.channels_last)
        return model(x)[0].permute(1, 2, 0).cpu().numpy()

    return predict


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def _baseline_signature(root, comp):
    # Standard NumPy readers ignore this short trailer after the NPY payload.
    value = "qecnn-rgb-f32-v1:" + _file_digest(comp.path) + ":" + _file_digest(Path(root) / "best_model2026.pth")
    return value.encode("ascii")


def _write_baseline_cache(path, data, signature):
    # A process that stops mid-write leaves the previous cache available.
    # Unique staging paths also support simultaneous runs using the same cache.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        tmp = Path(directory) / "baseline.npy"
        with tmp.open("wb") as stream:
            np.save(stream, data, allow_pickle=False)
            stream.write(signature)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)


def build_baseline_cache(root, cache_path, device="auto"):
    """FP32 predictions with the source signature and data published in one atomic rename."""
    comp, _ = paired_frames(root, "test")
    path = Path(cache_path)
    signature = _baseline_signature(root, comp)
    shape = (len(comp), HEIGHT, WIDTH, 3)
    if path.is_file():
        try:
            # Keep one descriptor for verification and reading, so a concurrent
            # replacement can never mix predictions and another input's signature.
            with path.open("rb") as stream:
                stream.seek(-len(signature), os.SEEK_END)
                if stream.read() == signature:
                    stream.seek(0)
                    data = np.load(stream, allow_pickle=False)
                    if data.shape == shape and data.dtype == np.float32:
                        return data
        except (EOFError, OSError, ValueError):
            pass  # Truncated or invalid cache: regenerate it below.
    predict = baseline_predictor(root, device)
    data = np.empty(shape, dtype=np.float32)
    start = time.monotonic()
    for i in range(len(comp)):
        data[i] = predict(comp.rgb(i))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"QECNN {i+1}/{len(comp)} ({time.monotonic()-start:.1f} s)", flush=True)
    _write_baseline_cache(path, data, signature)
    return data


def _plot_curves(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx, decoder, baseline, proposed = np.asarray(rows).T
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True)
    axes[0].plot(idx, decoder, lw=1.6, color="#788694", label="x265")
    axes[0].plot(idx, baseline, lw=1.1, color="#d47032", label="QECNN")
    axes[0].plot(idx, proposed, lw=.9, color="#087c6a", label="Предложенная модель")
    axes[0].set_ylabel("RGB-PSNR, дБ")
    axes[0].legend(ncol=3, loc="upper right")
    axes[1].plot(idx, proposed - baseline, color="#087c6a", lw=1.2)
    axes[1].axhline(0, color="#666", lw=.6)
    axes[1].axhline(.1, color="#a0522d", ls="--", lw=.9, label="Целевой средний выигрыш")
    axes[1].axhline(np.mean(proposed - baseline), color="#087c6a", ls=":",
                    label="Фактический средний выигрыш")
    axes[1].legend(ncol=2, loc="upper right", fontsize=8)
    axes[1].set_ylabel("Выигрыш к QECNN, дБ")
    axes[1].set_xlabel("Номер изображения в тестовом файле")
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(1, len(rows))
    fig.savefig(output_dir / "psnr.png", dpi=160)
    plt.close(fig)


def _visual_example(example_dir, number, raw, decoded, base, proposed):
    from PIL import Image, ImageDraw
    # Crop around the largest mean squared difference between the two predictions.
    side = 96
    diff = np.mean((base - proposed) ** 2, axis=2)
    integral = cv2.integral(diff, sdepth=cv2.CV_64F)
    scores = integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side] + integral[:-side, :-side]
    y, x = (int(v) for v in np.unravel_index(np.argmax(scores), scores.shape))
    example_dir.mkdir(parents=True, exist_ok=True)
    frames = {"reference": raw, "x265": decoded, "qecnn": base, "proposed": proposed}
    # PNGs are only for display. PSNR was computed on the floating-point predictions.
    for tag, rgb in frames.items():
        img = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        img.save(example_dir / f"{tag}.png")
        img.crop((x, y, x + side, y + side)).save(example_dir / f"{tag}_crop.png")
        if tag == "x265":
            locator = img.copy()
            ImageDraw.Draw(locator).rectangle((x, y, x + side - 1, y + side - 1),
                                              outline=(255, 216, 70), width=2)
            locator.save(example_dir / "location.png")
    return {"image": number, "x": x, "y": y, "size": side}


def evaluate(root, predictor, output_dir, device="auto", baseline_cache=None):
    """Compare a predictor(planes, rgb) -> HWC float RGB on the fixed 100 test frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comp, raw = paired_frames(root, "test")
    if len(comp) != 100:
        raise ValueError(f"The lab evaluation requires 100 image pairs, got {len(comp)}")
    cached = None
    baseline = None
    if baseline_cache is None:
        baseline = baseline_predictor(root, device)
    else:
        cached = build_baseline_cache(root, baseline_cache, device)

    def predictions(i):
        decoded = comp.rgb(i)
        base = baseline(decoded) if cached is None else cached[i]
        proposed = np.asarray(predictor(comp.planes(i), decoded))
        if not np.all(np.isfinite(proposed)):
            raise ValueError(f"Non-finite prediction for image {i+1}")
        return raw.rgb(i), decoded, base, proposed

    rows = []
    for i in range(len(comp)):
        reference, decoded, base, proposed = predictions(i)
        rows.append((i + 1, psnr(reference, decoded), psnr(reference, base), psnr(reference, proposed)))
        if (i + 1) % 10 == 0:
            print(f"PSNR {i+1}/{len(comp)}", flush=True)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x265", "qecnn", "proposed", "gain_vs_qecnn", "gain_vs_x265"])
        for number, dec, base, proposed in rows:
            writer.writerow([number, dec, base, proposed, proposed - base, proposed - dec])
    _, decoder, base, proposed = np.asarray(rows).T
    delta = proposed - base
    delta_decoder = proposed - decoder
    summary = {
        "n_test": len(rows),
        "mean_x265": float(decoder.mean()),
        "mean_qecnn": float(base.mean()),
        "mean_proposed": float(proposed.mean()),
        "gain_vs_qecnn": float(delta.mean()),
        "gain_vs_x265": float(delta_decoder.mean()),
        "improved_vs_qecnn": int(np.sum(delta > 0)),
        "median_gain_vs_qecnn": float(np.median(delta)),
        "min_gain_vs_qecnn": float(np.min(delta)),
        "max_gain_vs_qecnn": float(np.max(delta)),
        "thresholds_met": bool(delta.mean() >= .10 and delta_decoder.mean() >= .32),
    }
    _plot_curves(rows, output_dir)
    order = np.argsort(delta, kind="stable")
    median = int(np.argmin(np.abs(delta - np.median(delta))))
    summary["examples"] = {}
    for tag, i in [("min", int(order[0])), ("typical", median), ("max", int(order[-1]))]:
        summary["examples"][tag] = _visual_example(output_dir / "examples" / tag, i + 1, *predictions(i))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8")
    return summary
