# steg/metrics.py
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import os, json
from .chi_square import chi_square_stat, chi2_block_scores
from skimage.io import imread

# ---------- basic PSNR/SSIM and visual artifacts for main analysis ----------
def compute_psnr_ssim(cover_path, stego_path):
    orig = imread(cover_path)
    steg = imread(stego_path)
    psnr = float(peak_signal_noise_ratio(orig, steg, data_range=255))

    # SSIM with adaptive win_size
    h, w = orig.shape[0], orig.shape[1]
    min_side = min(h, w)
    if min_side < 7:
        win_size = min_side if (min_side % 2 == 1) else (min_side - 1)
        if win_size < 3:
            ssim = 1.0 if np.array_equal(orig, steg) else 0.0
            return psnr, float(ssim)
    else:
        win_size = 7

    ssim = structural_similarity(orig, steg, data_range=255, channel_axis=-1, win_size=win_size)
    return psnr, float(ssim)

def save_diff_map(cover_path, stego_path, outpath):
    orig = np.array(Image.open(cover_path).convert("RGB"), dtype=np.int16)
    steg = np.array(Image.open(stego_path).convert("RGB"), dtype=np.int16)
    diff = np.abs(orig - steg).astype(np.uint8)
    gray_diff = np.max(diff, axis=2)
    plt.figure(figsize=(6,6))
    plt.imshow(gray_diff, cmap='gray')
    plt.title("Difference map (max channel abs diff)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_histograms(cover_path, stego_path, outdir, base):
    os.makedirs(outdir, exist_ok=True)
    orig = np.array(Image.open(cover_path).convert("RGB"), dtype=np.uint8)
    steg = np.array(Image.open(stego_path).convert("RGB"), dtype=np.uint8)

    channels = ['R', 'G', 'B']

    for i, ch in enumerate(channels):
        fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        plt.suptitle(f"Histogram channel {ch}")

        # --- Cover ---
        axes[0].hist(orig[:, :, i].ravel(), bins=256)
        axes[0].set_title("Cover")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")

        # --- Stego ---
        axes[1].hist(steg[:, :, i].ravel(), bins=256)
        axes[1].set_title("Stego")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(outdir, f"{base}_hist_{ch}.png"), dpi=150)
        plt.close()

# AUC
def compute_auc_from_scores(pos_scores, neg_scores):
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return 0.0
    all_scores = np.concatenate([pos, neg])
    thresholds = np.unique(all_scores)
    tprs = []
    fprs = []
    P = pos.size
    N = neg.size
    for thr in thresholds:
        tp = np.sum(pos >= thr)
        fp = np.sum(neg >= thr)
        tpr = tp / P
        fpr = fp / N
        tprs.append(tpr)
        fprs.append(fpr)
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    order = np.argsort(fprs)
    fprs = fprs[order]
    tprs = tprs[order]
    auc = np.trapz(tprs, fprs)
    if auc < 0:
        auc = 0.0
    if auc > 1:
        auc = 1.0
    return float(auc)


def analyze_pair_for_payloads(cover_path, stego_path, payload, outdir, block_size_for_auc=16):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(stego_path))[0]

    # PSNR/SSIM
    psnr, ssim = compute_psnr_ssim(cover_path, stego_path)

    # save diff map and histograms
    diff_path = os.path.join(outdir, f"{base}_diff.png")
    save_diff_map(cover_path, stego_path, diff_path)
    save_histograms(cover_path, stego_path, outdir, base)

    # whole-image chi2 по всем каналам
    chi2res = chi_square_stat(stego_path)

    # block-level scores and AUC
    cover_block_scores = chi2_block_scores(cover_path, block_size=block_size_for_auc)
    stego_block_scores = chi2_block_scores(stego_path, block_size=block_size_for_auc)
    auc = compute_auc_from_scores(stego_block_scores, cover_block_scores)

    # prepare result
    res = {
        "cover": os.path.abspath(cover_path),
        "stego": os.path.abspath(stego_path),
        "payload": payload,
        "psnr": psnr,
        "ssim": ssim,
        "chi2": chi2res,  # Теперь содержит channels и overall
        "auc": auc,
        "cover_block_count": len(cover_block_scores),
        "stego_block_count": len(stego_block_scores)
    }

    # save json
    json_path = os.path.join(outdir, f"{base}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # visualization (table + simple barplot)
    from .visualization import save_metrics_table, plot_metrics
    table_path = os.path.join(outdir, f"{base}_metrics_table.png")
    plot_path = os.path.join(outdir, f"{base}_metrics_plot.png")
    save_metrics_table(res, table_path)
    plot_metrics(res, plot_path)

    return res

# ---------- lightweight analysis for experiments ----------
def analyze_pair_experiment(cover_path, stego_path, payload, outdir, block_size_for_auc=16):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(stego_path))[0]

    # whole-image chi2 on stego по всем каналам
    chi2res = chi_square_stat(stego_path)

    # block-level scores for ROC/AUC
    cover_block_scores = chi2_block_scores(cover_path, block_size=block_size_for_auc)
    stego_block_scores = chi2_block_scores(stego_path, block_size=block_size_for_auc)
    auc = compute_auc_from_scores(stego_block_scores, cover_block_scores)

    res = {
        "cover": os.path.abspath(cover_path),
        "stego": os.path.abspath(stego_path),
        "payload": payload,
        "chi2": chi2res,  # Теперь содержит channels и overall
        "auc": auc,
        "cover_block_count": len(cover_block_scores),
        "stego_block_count": len(stego_block_scores)
    }

    # save json only (no images)
    json_path = os.path.join(outdir, f"{base}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    return res