import os
import sys
import json
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, TensorDataset
import support

DEFAULT_PARAMS = {
    "patchsize": 44,
    "patchstep": 20,
    "f1": 128,
    "k1": 9,
    "f2": 64,
    "k2": 7,
    "f3": 64,
    "k3": 3,
    "f4": 32,
    "k4": 1,
    "out_kernel": 5,
    "batch_size": 64,
    "num_epochs": 10000,
    "max_training_seconds": 3000.0,
    "start_learning_rate": 0.0002,
    "end_learning_rate": 0.0001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-07
}

params = DEFAULT_PARAMS.copy()
if os.path.exists("params.json"):
    with open("params.json", "r", encoding="utf-8") as f:
        params.update(json.load(f))

patchsize     = int(params["patchsize"])
patchstep     = int(params["patchstep"])
f1            = int(params["f1"]);  k1 = int(params["k1"])
f2            = int(params["f2"]);  k2 = int(params["k2"])
f3            = int(params["f3"]);  k3 = int(params["k3"])
f4            = int(params["f4"]);  k4 = int(params["k4"])
out_kernel    = int(params["out_kernel"])

batch_size           = int(params["batch_size"])
num_epochs           = int(params["num_epochs"])
max_training_seconds = float(params["max_training_seconds"])
start_learning_rate  = float(params["start_learning_rate"])
end_learning_rate    = float(params["end_learning_rate"])
beta_1               = float(params["beta_1"])
beta_2               = float(params["beta_2"])
epsilon              = float(params["epsilon"])

w = 480
h = 320
max_samples_train = 20000
max_samples_validate = 2000
TrainMode = 0
PrepareDataSetFromYUV = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

testfolderRawYuv   = './testrawyuv/'
testfolderRawPng   = './testrawpng/'
testfolderCompYuv  = './testcompyuv/'
testfolderCompPng  = './testcomppng/'
trainfolderRawYuv  = './trainrawyuv/'
trainfolderRawPng  = './trainrawpng/'
trainfolderCompYuv = './traincompyuv/'
trainfolderCompPng = './traincomppng/'


def same_pad(k):
    return (k - 1) // 2

class EnhancerModel(nn.Module):
    def __init__(self, in_ch, out_ch,
                 f1, k1, f2, k2, f3, k3, f4, k4, out_kernel):
        super().__init__()

        self.conv1   = nn.Conv2d(in_ch, f1, k1, padding=same_pad(k1))
        self.prelu1  = nn.PReLU(f1)
        self.conv11  = nn.Conv2d(in_ch, f1, k1, padding=same_pad(k1))
        self.prelu11 = nn.PReLU(f1)

        self.conv2   = nn.Conv2d(f1, f2, k2, padding=same_pad(k2))
        self.prelu2  = nn.PReLU(f2)
        self.conv22  = nn.Conv2d(2 * f1, f2, k2, padding=same_pad(k2))
        self.prelu22 = nn.PReLU(f2)

        self.conv3   = nn.Conv2d(f2, f3, k3, padding=same_pad(k3))
        self.prelu3  = nn.PReLU(f3)
        self.conv33  = nn.Conv2d(2 * f2, f3, k3, padding=same_pad(k3))
        self.prelu33 = nn.PReLU(f3)

        self.conv4   = nn.Conv2d(f3, f4, k4, padding=same_pad(k4))
        self.prelu4  = nn.PReLU(f4)
        self.conv44  = nn.Conv2d(2 * f3, f4, k4, padding=same_pad(k4))
        self.prelu44 = nn.PReLU(f4)

        self.conv_out = nn.Conv2d(2 * f4, out_ch, out_kernel,
                                  padding=same_pad(out_kernel))

    def forward(self, x):
        c1  = self.prelu1(self.conv1(x))
        c11 = self.prelu11(self.conv11(x))
        feat1 = torch.cat([c1, c11], dim=1)

        c2  = self.prelu2(self.conv2(c1))
        c22 = self.prelu22(self.conv22(feat1))
        feat2 = torch.cat([c2, c22], dim=1)

        c3  = self.prelu3(self.conv3(c2))
        c33 = self.prelu33(self.conv33(feat2))
        feat3 = torch.cat([c3, c33], dim=1)

        c4  = self.prelu4(self.conv4(c3))
        c44 = self.prelu44(self.conv44(feat3))
        feat4 = torch.cat([c4, c44], dim=1)

        return x + self.conv_out(feat4)

    @torch.no_grad()
    def predict(self, x, verbose=0, batch_size=128):
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = x[None]
            x = np.transpose(x.astype(np.float32), (0, 3, 1, 2))
            x = torch.from_numpy(x).to(device)
        self.eval()
        outs = [self(x[i:i + batch_size]).cpu()
                for i in range(0, x.size(0), batch_size)]
        out = torch.cat(outs, dim=0).numpy()
        return np.transpose(out, (0, 2, 3, 1))


def build_enhancer():
    return EnhancerModel(
        3, 3,
        f1, k1, f2, k2, f3, k3, f4, k4, out_kernel
    ).to(device)


def psnr_batch(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))

def TrainImageEnhancementModel(folderRaw, folderComp, folderRawVal, folderCompVal):
    print('Loading raw train images...')
    Xraw  = support.LoadImagesFromFolderTorch(folderRaw,  max_samples_train, patchsize)
    print('Loading compressed train images...')
    Xcomp = support.LoadImagesFromFolderTorch(folderComp, max_samples_train, patchsize)

    print('Loading raw validation images...')
    XrawVal  = support.LoadImagesFromFolderTorch(folderRawVal,  max_samples_validate, patchsize)
    print('Loading compressed validation images...')
    XcompVal = support.LoadImagesFromFolderTorch(folderCompVal, max_samples_validate, patchsize)

    print(f"Train shape: {Xcomp.shape}, Val shape: {XcompVal.shape}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xcomp), torch.from_numpy(Xraw)),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(XcompVal), torch.from_numpy(XrawVal)),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_enhancer()
    optimizer = optim.Adam(model.parameters(),
                           lr=start_learning_rate,
                           betas=(beta_1, beta_2),
                           eps=epsilon)
    criterion = nn.MSELoss()

    if end_learning_rate != start_learning_rate:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_learning_rate / start_learning_rate,
            total_iters=num_epochs)
    else:
        scheduler = None

    best_val_loss = float('inf')
    train_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        for comp, raw in train_loader:
            comp = comp.to(device, non_blocking=True)
            raw  = raw.to(device,  non_blocking=True)
            optimizer.zero_grad()
            out  = model(comp)
            loss = criterion(out, raw)
            loss.backward()
            optimizer.step()

            bs = comp.size(0)
            train_loss += loss.item() * bs
            train_psnr += psnr_batch(raw, out).item() * bs
        train_loss /= len(train_loader.dataset)
        train_psnr /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for comp, raw in val_loader:
                comp = comp.to(device, non_blocking=True)
                raw  = raw.to(device,  non_blocking=True)
                out  = model(comp)
                bs   = comp.size(0)
                val_loss += criterion(out, raw).item() * bs
                val_psnr += psnr_batch(raw, out).item() * bs
        val_loss /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}, '
              f'Train Loss: {train_loss:.6f}, Train PSNR: {train_psnr:.2f} dB, '
              f'Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f} dB, '
              f'LR: {current_lr:.6g}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model2026.pth')
            print('  -> val_loss improved, saved best_model2026.pth')

        if scheduler is not None:
            scheduler.step()

        if time.time() - train_start_time > max_training_seconds:
            print(f'Training time budget exhausted after {epoch+1} epochs, stopping.')
            break

    torch.save(model.state_dict(), 'enhancer2026.pth')
    return model


def InferenceImageEnhancementModel(weights_path='best_model2026.pth'):
    model = build_enhancer()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def main():
    sys.stdout.reconfigure(line_buffering=True)   
    start_time = time.time()
    print("=== Image enhancing script started ===", flush=True)
       
    if PrepareDataSetFromYUV == 1:
        support.FromFolderYuvToFolderPNG(testfolderRawYuv,  testfolderRawPng,  w, h, patchsize, patchstep)
        support.FromFolderYuvToFolderPNG(testfolderCompYuv, testfolderCompPng, w, h, patchsize, patchstep)
        support.FromFolderYuvToFolderPNG(trainfolderRawYuv,  trainfolderRawPng,  w, h, patchsize, patchstep)
        support.FromFolderYuvToFolderPNG(trainfolderCompYuv, trainfolderCompPng, w, h, patchsize, patchstep)

    if TrainMode == 1:
        TrainImageEnhancementModel(trainfolderRawPng, trainfolderCompPng,testfolderRawPng,  testfolderCompPng)

    enhancer = InferenceImageEnhancementModel()
    psnr_enh = support.ShowFramePSNRPerformance(testfolderRawYuv, testfolderCompYuv, 0, 100, w, h, enhancer,TrainMode)
    
    end_time = time.time()
    time_spent = end_time - start_time
    
    peak_gpu_memory_gb = 0.0
    if torch.cuda.is_available():
        peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    results = {
        "success": True,
        "psnr_enh": float(psnr_enh),
        "time_spent": float(time_spent),
        "peak_gpu_memory_gb": float(peak_gpu_memory_gb)
    }
    
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"PSNR Enhancement: {psnr_enh:.3f} dB")
    print(f"Time spent: {time_spent:.2f} s")
    print(f"Peak GPU memory: {peak_gpu_memory_gb:.3f} GB")
    print("Results saved to results.json")


if __name__ == '__main__':
    main()