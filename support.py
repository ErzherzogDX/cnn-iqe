import os
import cv2
import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def _load_png_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return img[:, :, ::-1].copy()


def LoadImagesFromFolderTorch(folder, max_samples, patchsize,
                              n_workers=8, use_cache=True):
    files = sorted(f for f in os.listdir(folder) if f.endswith('.png'))[:max_samples]
    N = len(files)
    if N == 0:
        return np.zeros((0, 3, patchsize, patchsize), dtype=np.float32)

    cache_file = os.path.join(folder, f".cache_{N}_{patchsize}.npy")

    if use_cache and os.path.exists(cache_file):
        try:
            newest_png = max(os.path.getmtime(os.path.join(folder, f)) for f in files)
            if os.path.getmtime(cache_file) >= newest_png:
                print(f"Loading {N} images from cache {cache_file}")
                X = np.load(cache_file)
                if X.shape == (N, 3, patchsize, patchsize):
                    return X
        except Exception as e:
            print(f"Cache read failed ({e}), reloading from PNG")

    print(f"Loading {N} images from {folder} (parallel, {n_workers} workers)")
    X = np.empty((N, patchsize, patchsize, 3), dtype=np.uint8)
    paths = [os.path.join(folder, f) for f in files]

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for idx, img in enumerate(ex.map(_load_png_rgb, paths)):
            if img is not None:
                X[idx] = img

    X = X.astype(np.float32) / 255.0
    X = np.transpose(X, (0, 3, 1, 2))
    X = np.ascontiguousarray(X)

    if use_cache:
        try:
            np.save(cache_file, X)
        except Exception as e:
            print(f"Warning: could not save cache ({e})")

    return X


def cal_psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    return psnr

def yuv2rgb(Y, U, V, fw, fh):
    U_new = cv2.resize(U, (fw, fh), interpolation=cv2.INTER_CUBIC)
    V_new = cv2.resize(V, (fw, fh), interpolation=cv2.INTER_CUBIC)
    rf = Y + 1.4075 * (V_new - 128.0)
    gf = Y - 0.3455 * (U_new - 128.0) - 0.7169 * (V_new - 128.0)
    bf = Y + 1.7790 * (U_new - 128.0)
    rf = np.clip(rf, 0, 255)
    gf = np.clip(gf, 0, 255)
    bf = np.clip(bf, 0, 255)
    return rf, gf, bf

def FromFolderYuvToFolderPNG(folderyuv, folderpng, fw, fh,patchsize,patchstep):
    os.makedirs(folderpng, exist_ok=True)
    # Clear the PNG output folder, but preserve hidden files (e.g. the
    # .cache_*.npy files written by LoadImagesFromFolderTorch) so that
    # the training cache survives a data-prep re-run.
    for name in os.listdir(folderpng):
        if name.startswith('.'):
            continue
        p = os.path.join(folderpng, name)
        if os.path.isfile(p):
            os.remove(p)

    fwuv = fw // 2
    fhuv = fh // 2
    expected_frame_bytes = fw * fh + 2 * fwuv * fhuv  # YUV420

    # Number of tile positions along each axis. The +1 accounts for the
    # LAST valid tile whose top-left corner is at (fw - patchsize), which
    # the previous formula (without +1) was silently dropping.
    numdx = max(0, (fw - patchsize) // patchstep + 1)
    numdy = max(0, (fh - patchsize) // patchstep + 1)
    dx = (np.arange(numdx) * patchstep).astype(int)
    dy = (np.arange(numdy) * patchstep).astype(int)

    Im = np.zeros((patchsize, patchsize, 3), dtype=np.uint8)
    dir_list = os.listdir(folderyuv)
    pngframenum = 0

    for name in dir_list:
        fullname = os.path.join(folderyuv, name)
        if not fullname.endswith('.yuv'):
            continue

        file_size = os.path.getsize(fullname)
        frames = min(100, file_size // expected_frame_bytes)
        if frames <= 0:
            print(f"Skip {fullname}: not enough data for one frame")
            continue

        with open(fullname, 'rb') as fp:
            print(f"{fullname}, frames: {frames}")
            for f in range(frames):
                try:
                    Y = np.frombuffer(fp.read(fw * fh), dtype=np.uint8).reshape((fh, fw))
                    U = np.frombuffer(fp.read(fwuv * fhuv), dtype=np.uint8).reshape((fhuv, fwuv))
                    V = np.frombuffer(fp.read(fwuv * fhuv), dtype=np.uint8).reshape((fhuv, fwuv))
                except ValueError:
                    print(f"Corrupted frame {f} in {fullname}")
                    break

                r, g, b = yuv2rgb(Y, U, V, fw, fh)
                for i in dx:
                    for j in dy:
                        Im[:, :, 0] = b[j:j+patchsize, i:i+patchsize]
                        Im[:, :, 1] = g[j:j+patchsize, i:i+patchsize]
                        Im[:, :, 2] = r[j:j+patchsize, i:i+patchsize]
                        pngfilename = os.path.join(folderpng, f"{pngframenum}.png")
                        cv2.imwrite(pngfilename, Im)
                        pngframenum += 1
    return pngframenum - 1

def LoadImagesFromFolder(foldername, max_samples, patchsize):
    files = [f for f in os.listdir(foldername) if f.endswith('.png')]
    files = files[:max_samples]  # ← ограничение памяти!
    N = len(files)
    print(f"Loading {N} images from {foldername}")
    x = np.zeros((N, patchsize, patchsize, 3), dtype=np.float32)
    for idx, name in enumerate(files):
        img = cv2.imread(os.path.join(foldername, name))
        if img is None:
            continue
        x[idx, :, :, 0] = img[:, :, 2]  # R
        x[idx, :, :, 1] = img[:, :, 1]  # G
        x[idx, :, :, 2] = img[:, :, 0]  # B
    return x

def GetRGBFrame(folderyuv, VideoNumber, FrameNumber, fw, fh):
    # Enumerate only .yuv files, in a stable order, BEFORE applying the
    # VideoNumber index. The previous code advanced the counter on every
    # directory entry (including non-.yuv files), which shifted the index
    # whenever the folder contained anything else.
    yuv_names = sorted(f for f in os.listdir(folderyuv) if f.endswith('.yuv'))
    if VideoNumber < 0 or VideoNumber >= len(yuv_names):
        return None
    name = yuv_names[VideoNumber]
    fullname = os.path.join(folderyuv, name)

    file_size = os.path.getsize(fullname)
    expected_frame_bytes = fw * fh + 2 * (fw // 2) * (fh // 2)
    frames = file_size // expected_frame_bytes
    with open(fullname, 'rb') as fp:
        for f in range(frames):
            Y = np.frombuffer(fp.read(fw * fh), dtype=np.uint8).reshape((fh, fw))
            U = np.frombuffer(fp.read((fw//2)*(fh//2)), dtype=np.uint8).reshape((fh//2, fw//2))
            V = np.frombuffer(fp.read((fw//2)*(fh//2)), dtype=np.uint8).reshape((fh//2, fw//2))
            if f == FrameNumber:
                return yuv2rgb(Y, U, V, fw, fh)
    return None

def GetEngancedRGB(enhancer,RGBin, fw, fh):
    RGBin = np.expand_dims(RGBin, axis=0).astype(np.float32) / 255.0
    pred = enhancer.predict(RGBin, verbose=0)
    return np.squeeze(pred, axis=0) * 255.0

def ShowOneFrameEnhancement(RGBRAW, RGBCOMP,RGBENH):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(RGBRAW / 255.0)
    plt.title("Raw\nPSNR: —")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    psnr1 = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
    plt.imshow(RGBCOMP / 255.0)
    plt.title(f"Compressed\nPSNR: {psnr1:.2f} dB")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    psnr2 = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)
    plt.imshow(RGBENH / 255.0)
    plt.title(f"Enhanced\nPSNR: {psnr2:.2f} dB")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def ShowFramePSNRPerformance(folderyuv, foldercomp, VideoIndex, framesmax, fw, fh, enhancer, trainmode):
    yuv_names = sorted(f for f in os.listdir(folderyuv) if f.endswith('.yuv'))
    if VideoIndex < 0 or VideoIndex >= len(yuv_names):
        print(f"ShowFramePSNRPerformance: VideoIndex={VideoIndex} out of "
              f"range (only {len(yuv_names)} .yuv file(s) in {folderyuv})")
        return 0.0

    name = yuv_names[VideoIndex]
    fullname = os.path.join(folderyuv, name)

    file_size = os.path.getsize(fullname)
    expected_frame_bytes = fw * fh + 2 * (fw // 2) * (fh // 2)
    frames = min(framesmax, file_size // expected_frame_bytes)
    if frames <= 0:
        print(f"ShowFramePSNRPerformance: no complete frames in {fullname}")
        return 0.0

    PSNRCOMP = np.zeros(frames)
    PSNRENH = np.zeros(frames)
    delta = 0
    bestf = 0
    for f in range(frames):
        r1, g1, b1 = GetRGBFrame(folderyuv, VideoIndex, f, fw, fh)
        RGBRAW = np.stack([r1, g1, b1], axis=-1)
        r2, g2, b2 = GetRGBFrame(foldercomp, VideoIndex, f, fw, fh)
        RGBCOMP = np.stack([r2, g2, b2], axis=-1)
        PSNRCOMP[f] = cal_psnr(RGBRAW / 255.0, RGBCOMP / 255.0)
        RGBENH = GetEngancedRGB(enhancer, RGBCOMP, fw, fh)
        PSNRENH[f] = cal_psnr(RGBRAW / 255.0, RGBENH / 255.0)
        if (PSNRENH[f]-PSNRCOMP[f])>delta:
            delta = PSNRENH[f]-PSNRCOMP[f]
            bestf=f


    

    if trainmode==0:
        sorted_indices = np.argsort(PSNRCOMP)
        PSNRCOMP_sorted = PSNRCOMP[sorted_indices]
        PSNRENH_sorted = PSNRENH[sorted_indices]
        plt.plot(PSNRCOMP_sorted, label='Compressed')
        plt.plot(PSNRENH_sorted, label='Enhanced')
        plt.xlabel('Frame (sorted by Compressed PSNR)')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True)
        plt.title(f"{name}: Compressed = {PSNRCOMP.mean():.3f} dB, Enhanced = {PSNRENH.mean():.3f} dB")
        plt.show()
        r1, g1, b1 = GetRGBFrame(folderyuv, VideoIndex, bestf, fw, fh)
        RGBRAW = np.stack([r1, g1, b1], axis=-1)
        r2, g2, b2 = GetRGBFrame(foldercomp, VideoIndex, bestf, fw, fh)
        RGBCOMP = np.stack([r2, g2, b2], axis=-1)
        RGBENH = GetEngancedRGB(enhancer, RGBCOMP, fw, fh)
        ShowOneFrameEnhancement(RGBRAW,RGBCOMP,RGBENH)

    return PSNRENH.mean()

def yuv_import(filename, fw, fh, numfrm):
    # Bulk read: one fp.read() per frame instead of byte-by-byte reads.
    # Y is kept in the same shape (numfrm, fw, fh) as before to preserve
    # the file's existing contract with yuv_save().
    d00 = fw // 2
    d01 = fh // 2
    Y_size = fw * fh
    U_size = d00 * d01
    V_size = d00 * d01
    frame_size = Y_size + U_size + V_size

    Y = np.zeros((numfrm, fw, fh), np.uint8)
    U = np.zeros((numfrm, d00, d01), np.uint8)
    V = np.zeros((numfrm, d00, d01), np.uint8)

    with open(filename, 'rb') as fp:
        for i in range(numfrm):
            buf = fp.read(frame_size)
            if len(buf) < frame_size:
                break
            Y[i] = np.frombuffer(buf[:Y_size], dtype=np.uint8).reshape((fw, fh))
            U[i] = np.frombuffer(buf[Y_size:Y_size + U_size],
                                 dtype=np.uint8).reshape((d00, d01))
            V[i] = np.frombuffer(buf[Y_size + U_size:],
                                 dtype=np.uint8).reshape((d00, d01))
    return (Y, U, V)

def yuv_save(filename, fw, fh, numfrm,Y,U,V):
    fp = open(filename, 'wb')
    for i in range(numfrm):
        Y[i].astype('uint8').tofile(fp)
        U[i].astype('uint8').tofile(fp)
        V[i].astype('uint8').tofile(fp)
    fp.close()