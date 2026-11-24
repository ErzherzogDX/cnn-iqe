"""Двухмасштабная остаточная сеть для восстановления изображений x265."""
import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import support


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.leaky_relu(self.conv1(x), 0.1)) * 0.5


class PhaseRestorer(nn.Module):
    """Shared two-scale residual estimator, averaged over the dihedral group D4."""
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(24, 32, 3, padding=1)
        self.fine = ResidualBlock(32)
        self.down = nn.Conv2d(32, 40, 3, stride=2, padding=1)
        self.coarse = nn.Sequential(ResidualBlock(40), ResidualBlock(40),
                                    ResidualBlock(40))
        self.up = nn.Conv2d(40, 32 * 4, 1)
        self.fuse = ResidualBlock(32)
        self.output = nn.Conv2d(32, 12, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def directional(self, x):
        packed = F.pixel_unshuffle(x, 2)
        local = F.avg_pool2d(packed, 3, stride=1, padding=1,
                             count_include_pad=False)
        features = torch.cat(((packed - 0.5) * 2, (packed - local) * 12), 1)
        fine = self.fine(F.leaky_relu(self.entry(features), 0.1))
        coarse = self.coarse(F.leaky_relu(self.down(fine), 0.1))
        fused = self.fuse(fine + F.pixel_shuffle(self.up(coarse), 2))
        residual = F.pixel_shuffle(self.output(fused), 2)
        return x + residual / 16

    def forward(self, x):
        # A group average is built into the estimator. A training pair has one
        # target and contributes a single loss after the branches are combined.
        batch = x.shape[0]
        viewsets = []
        for t in (False, True):
            base = x.transpose(2, 3) if t else x
            viewsets.append(torch.cat([base, base.flip((2,)), base.flip((3,)),
                                       base.flip((2, 3))], dim=0))
        if x.shape[2] == x.shape[3]:
            responses = self.directional(torch.cat(viewsets)).split(4 * batch)
        else:
            responses = [self.directional(views) for views in viewsets]
        combined = []
        for t, response in enumerate(responses):
            a, b, c, d = response.split(batch)
            restored = a + b.flip((2,)) + c.flip((3,)) + d.flip((2, 3))
            combined.append(restored.transpose(2, 3) if t else restored)
        return (combined[0] + combined[1]) / 8


def load_training_frames(root):
    comp, raw = support.paired_frames(root, 'train')
    if len(comp) <= 20:
        raise ValueError('At least 21 training pairs are required')
    arrays = []
    for clip in (comp, raw):
        frames = np.empty((len(clip), 3, support.HEIGHT, support.WIDTH), dtype=np.float32)
        for i in range(len(clip)):
            rgb = support.rgb255_from_planes(*clip.planes(i))
            frames[i] = (rgb / 255).transpose(2, 0, 1)
        arrays.append(frames)
    return arrays


def predict_function(weights, device):
    model = PhaseRestorer().to(device)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
    model.eval()

    @torch.inference_mode()
    def predict(_planes, rgb):
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        out = model(torch.from_numpy(x).unsqueeze(0).to(device))
        return out[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return predict


def validation(model, comp, raw, first, device):
    model.eval()
    quality = []
    base_quality = []
    with torch.inference_mode():
        for i in range(first, len(comp)):
            x = torch.from_numpy(comp[i:i+1]).to(device)
            y = raw[i]
            pred = model(x).clamp(0, 1).cpu().numpy()[0]
            quality.append(-10 * math.log10(np.square(pred - y).mean(dtype=np.float64) + 1e-10))
            base_quality.append(-10 * math.log10(np.square(comp[i] - y).mean(dtype=np.float64) + 1e-10))
    model.train()
    return float(np.mean(quality)), float(np.mean(base_quality))


def train_model(inputs, targets, device, weights, log):
    steps, seed, start_lr = 12000, 2091, .001
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit = len(inputs) - 20
    frame_mse = np.array([np.mean(np.square(a - b), dtype=np.float64)
                          for a, b in zip(inputs[:fit], targets[:fit])])
    model = PhaseRestorer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=start_lr)
    averaged = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(.95))
    weights.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    best = -math.inf
    start = time.monotonic()
    patch = 48
    batch = 12
    print(f'{fit} fitting frames + 20 holdout; {sum(p.numel() for p in model.parameters())} parameters on {device}', flush=True)
    with log.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['step', 'seconds', 'holdout_input_db', 'holdout_model_db', 'gain_db'])
        for step in range(1, steps + 1):
            # The four-pixel crop grid matches the two spatial reductions.
            # Each crop retains the original orientation of the training pair.
            ids = rng.integers(fit, size=batch)
            ys = rng.integers(0, (320 - patch) // 4 + 1, size=batch) * 4
            xs = rng.integers(0, (480 - patch) // 4 + 1, size=batch) * 4
            slices = [(i, slice(None), slice(y, y+patch), slice(x, x+patch))
                      for i, y, x in zip(ids, ys, xs)]
            src = torch.from_numpy(np.stack([inputs[s] for s in slices])).to(device)
            ref = torch.from_numpy(np.stack([targets[s] for s in slices])).to(device)
            scales = torch.from_numpy((1 / np.maximum(frame_mse[ids], 1e-12)).astype(np.float32)).to(device)
            progress = (step - 1) / steps
            lr = start_lr * (0.12 + 0.88 * (1 + math.cos(math.pi * progress)) / 2)
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            err = (model(src) - ref)[:, :, 4:-4, 4:-4]
            loss = (err.square().mean((1, 2, 3)) * scales).mean()
            loss.backward()
            opt.step()
            if step % 10 == 0:
                averaged.update_parameters(model)
            if step in (100, 500) or step % 1000 == 0 or step == steps:
                score, baseline = validation(averaged.module, inputs, targets, fit, device)
                elapsed = time.monotonic() - start
                writer.writerow([step, elapsed, baseline, score, score - baseline])
                file.flush()
                if score > best:
                    best = score
                    temporary = weights.with_suffix('.tmp')
                    torch.save({k: v.detach().cpu() for k, v in averaged.module.state_dict().items()}, temporary)
                    temporary.replace(weights)
                print(f'Step {step} | {elapsed:.1f}s | holdout {score:.4f} | gain {score-baseline:+.4f} dB', flush=True)


def auto_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    train = modes.add_parser('train', help='Train from scratch on the first 180 of 200 frames')
    evaluate = modes.add_parser('evaluate', help='Compare the supplied model and QECNN on the 100 test frames')
    for command in (train, evaluate):
        command.add_argument('--root', type=Path, default=Path('.'),
                             help='Original lab directory, containing the YUV files and QECNN weights')
        command.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    train.add_argument('--weights', type=Path, default=Path('training/phase_restorer.pth'))
    train.add_argument('--training-dir', type=Path, default=Path('training'))
    enhance = modes.add_parser('enhance', help='Restore a 480×320 YUV420p file and export RGB PNG frames')
    enhance.add_argument('input', type=Path)
    enhance.add_argument('--out', type=Path, default=Path('enhanced'))
    enhance.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    for command in (enhance, evaluate):
        command.add_argument('--weights', type=Path, default=Path(__file__).with_name('phase_restorer.pth'))
    evaluate.add_argument('--out', type=Path, default=Path('results'))
    evaluate.add_argument('--baseline-cache', type=Path, help='Optional FP32 cache of fixed QECNN predictions')
    args = parser.parse_args()
    device = auto_device() if args.device == 'auto' else args.device
    cv2.setNumThreads(1)
    if args.mode == 'enhance':
        clip = support.YUVFile(args.input)
        predictor = predict_function(args.weights, device)
        args.out.mkdir(parents=True, exist_ok=True)
        for i in range(len(clip)):
            planes = clip.planes(i)
            rgb = predictor(planes, support.rgb_from_planes(*planes))
            bgr = np.rint(rgb[:, :, ::-1] * 255).astype(np.uint8)
            path = args.out / f'{i + 1:04d}.png'
            if not cv2.imwrite(str(path), bgr):
                raise OSError(f'Cannot write {path}')
        print(f'Saved {len(clip)} PNG files to {args.out}')
        return
    if args.mode == 'train':
        inputs, targets = load_training_frames(args.root)
        train_model(inputs, targets, device, args.weights, args.training_dir / 'training.csv')
        return
    predictor = predict_function(args.weights, device)
    summary = support.evaluate(args.root, predictor, args.out, device, args.baseline_cache)
    for name in ('mean_x265', 'mean_qecnn', 'mean_proposed', 'gain_vs_qecnn', 'gain_vs_x265'):
        print(f'{name}: {summary[name]:.6f} dB')


if __name__ == '__main__':
    main()
