import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def SAM(gt, pr, eps=1e-8):
    dot = np.sum(gt * pr, axis=-1)
    norm_gt = np.linalg.norm(gt, axis=-1)
    norm_pr = np.linalg.norm(pr, axis=-1)
    cos_theta = np.clip(dot / (norm_gt * norm_pr + eps), -1, 1)
    return np.mean(np.degrees(np.arccos(cos_theta)))

def SID(gt, pr, eps=1e-8):
    gt_p = gt / (np.sum(gt, axis=-1, keepdims=True) + eps)
    pr_p = pr / (np.sum(pr, axis=-1, keepdims=True) + eps)
    return np.mean(np.sum(gt_p * np.log((gt_p + eps) / (pr_p + eps)) +
                          pr_p * np.log((pr_p + eps) / (gt_p + eps)), axis=-1))

def ERGAS(gt, pr, ratio=1.0, eps=1e-8):
    bands = gt.shape[-1]
    mean_gt = np.mean(gt, axis=(0,1))
    rmse = np.sqrt(np.mean((gt - pr)**2, axis=(0,1)))
    return 100/ratio * np.sqrt(np.mean((rmse / (mean_gt+eps))**2))

def evaluate_pair_ssc(gt_cube, pr_cube, wl_nm=None):
    """
    gt_cube, pr_cube: (H, W, C) numpy arrays
    wl_nm: wavelength array (optional, not used in these metrics)
    """
    scores = {}

    # Spectral metrics
    scores["SAM_deg"] = SAM(gt_cube, pr_cube)
    scores["SID"]     = SID(gt_cube, pr_cube)
    scores["ERGAS"]   = ERGAS(gt_cube, pr_cube)

    gt_rgb = np.mean(gt_cube, axis=-1) 
    pr_rgb = np.mean(pr_cube, axis=-1)

    scores["PSNR_dB"] = peak_signal_noise_ratio(gt_rgb, pr_rgb, data_range=gt_rgb.max() - gt_rgb.min())
    scores["SSIM"]    = structural_similarity(gt_rgb, pr_rgb, data_range=gt_rgb.max() - gt_rgb.min())

    scores["S_SAM"]   = scores["SAM_deg"]
    scores["S_SID"]   = scores["SID"]
    scores["S_ERGAS"] = scores["ERGAS"]
    scores["S_PSNR"]  = scores["PSNR_dB"]
    scores["S_SSIM"]  = scores["SSIM"] if "SSIM" in scores else 0.0

    # Dummy placeholders (replace with proper definitions if available)
    scores["S_SPEC"]  = scores["SAM_deg"]
    scores["S_SPAT"]  = scores["SSIM"]
    scores["S_COLOR"] = scores["PSNR_dB"]

    # Final SSC = average of sub-scores (toy version)
    scores["SSC"] = np.mean([
        scores["S_SAM"],
        scores["S_SID"],
        scores["S_ERGAS"],
        scores["S_PSNR"],
        scores["S_SSIM"],
    ])

    return scores

