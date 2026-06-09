import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import dotenv
import torch
import pyarrow.parquet as pq
import torch.nn.functional as F

# dotenv.load_dotenv()
base_url = "/kaggle/input/ariel-data-challenge-2025"

def show_figure(array, title=None, cmap="gray", aspect="auto"):
    image = array
    plt.figure(figsize=(8, 4))
    plt.imshow(image, cmap=cmap, aspect=aspect, vmin=-840, vmax=-660)
    if title:
        plt.title(title)
    plt.xlabel("Spectral direction — 356 px")
    plt.ylabel("Spatial direction — 32 px")
    plt.colorbar(label="Counts")
    plt.tight_layout()
    plt.show()


def reverse_adc(raw_signal: torch.Tensor):
    # NOTE: You will need to define AIRS_gain and AIRS_offset
    # For this competition, they are the same for all planets:
    AIRS_gain = 0.4369
    AIRS_offset = -1000.0
    return raw_signal / AIRS_gain + AIRS_offset  # Use division for reversal


def subtract_dark(raw_signal: torch.Tensor, dark_frame: torch.Tensor):
    """
    Subtract the dark frame from the raw signal.
    """
    # CORRECTED: Use a relative path to read axis_info
    axis_info_path = os.path.join(base_url, "axis_info.parquet")
    dt = pq.read_table(axis_info_path)["AIRS-CH0-integration_time"].drop_null().to_numpy().copy()

    # This logic for alternating integration times is specific to the competition
    dt[1::2] += 0.1

    dt = torch.tensor(dt, dtype=torch.float32, device="cuda")
    dt = dt.view(-1, 1, 1)
    return raw_signal - dark_frame * dt


def fix_dead_pixels_vectorized(signal, dead_pixels):
    """
    使用高效的卷积操作替换坏点。

    Args:
        signal (torch.Tensor): 输入信号，形状为 (B, H, W)，例如 (1250, 32, 356)。
        dead_pixels (torch.Tensor): 坏点掩码，形状为 (H, W)，例如 (32, 356)。

    Returns:
        torch.Tensor: 修复后的信号，形状与输入相同。
    """
    # 确保输入是 PyTorch Tensors
    # if not isinstance(signal, torch.Tensor):
    #     signal = torch.tensor(signal, dtype=torch.float32)
    # if not isinstance(dead_pixels, torch.Tensor):
    #     dead_pixels = torch.tensor(dead_pixels, dtype=torch.float32)

    # --- 1. 准备卷积 ---
    # a. 将2D掩码转换为布尔型，并适配3D信号的形状
    # mask 形状: (32, 356) -> (1, 32, 356)，以便广播到 (1250, 32, 356)
    mask = (dead_pixels == 1).unsqueeze(0)

    # b. 创建一个3x3的求和卷积核
    # conv2d需要4D输入: (out_channels, in_channels, kH, kW)
    kernel = torch.ones((1, 1, 3, 3), device=signal.device, dtype=signal.dtype)

    # c. 信号需要是4D: (B, C, H, W)。我们的信号是(B, H, W)，所以增加一个channel维度
    # signal_4d 形状: (1250, 32, 356) -> (1250, 1, 32, 356)
    signal_4d = signal.unsqueeze(1)

    # # --- 2. 计算邻居像素的和 ---
    # # 使用反射填充，与你原始代码的意图一致
    # # padding=1确保卷积后尺寸不变
    # sum_of_9_pixels = F.conv2d(signal_4d, kernel, padding='same', padding_mode='reflect')

    # --- 2. 【核心修正】分开执行填充和卷积 ---

    # 步骤 2a: 手动进行 'reflect' 填充
    # 我们需要在最后两个维度（H和W）的上下左右各填充1个像素
    # pad元组格式: (pad_left, pad_right, pad_top, pad_bottom)
    padded_signal_4d = F.pad(signal_4d, (1, 1, 1, 1), mode='reflect')

    # 步骤 2b: 在已填充的张量上执行卷积，此时 padding 设置为 0 或 'valid'
    # 'valid' 意味着不进行任何填充，这正是我们现在需要的
    sum_of_9_pixels = F.conv2d(padded_signal_4d, kernel, padding='valid')

    # ----------------------------------------------

    # 从4D转回3D，方便后续计算
    sum_of_9_pixels = sum_of_9_pixels.squeeze(1)

    # --- 3. 计算邻居像素的均值 ---
    # 9个像素的和 - 中心像素 = 8个邻居的和
    sum_of_8_neighbors = sum_of_9_pixels - signal
    mean_of_8_neighbors = sum_of_8_neighbors / 8.0

    # --- 4. 使用掩码替换坏点 ---
    # torch.where是最高效、最清晰的条件替换方法
    # torch.where(condition, value_if_true, value_if_false)
    # mask会被自动广播到signal的形状
    fixed_signal = torch.where(mask, mean_of_8_neighbors, signal)

    return fixed_signal


def linearity_correction(signal, linear_corr):
    """
    Perform linearity correction on the signal using polynomial coefficients and Horner's Method.
    """
    # Reshape linear_corr to match the signal shape and use Horner's method
    # Your implementation is good, but the one from the high-perf script is standard:
    c = linear_corr.view(6, signal.shape[1], signal.shape[2])  # Ensure correct shape

    # Using horner's method to evaluate the polynomial
    return ((((c[5] * signal + c[4]) * signal + c[3]) * signal + c[2]) * signal + c[1]) * signal + c[0]


# --- NEW: Add the CDS function ---
def get_cds(signal: torch.Tensor):
    """
    Step 5: Get Correlated Double Sampling (CDS)
    The science frames are alternating between the start of the exposure and the end of
    the exposure. The final CDS is the difference (End of exposure) - (Start of exposure).
    """
    if signal.ndim == 3:
        return signal[1::2, :, :] - signal[::2, :, :]
    else:
        return signal[1::2,:] - signal[::2,:]


def flat_correction(signal, flat_frame):
    """
    Perform flat field correction on the signal using the flat frame.
    """
    # Normalize the flat frame
    # A small epsilon is added to prevent division by zero
    epsilon = 1e-8
    normalized_flat = flat_frame / (torch.mean(flat_frame) + epsilon)

    # Apply flat field correction
    corrected_signal = signal / normalized_flat
    return corrected_signal


# --- REORDERED & UNCOMMENTED: The main processing pipeline ---
def process_data(raw_signal, calibration_data):
    """
    Process the raw signal with the complete calibration pipeline.
    """
    # Step 1: Reverse ADC
    if raw_signal.ndim == 3:  # For AIRS-CH0
        processed_signal = reverse_adc(raw_signal)

        # Step 2: Substitute dead pixels
        # Consider replacing with NaN masking for better results before interpolation
        dead_pixels = calibration_data['dead']
        processed_signal = fix_dead_pixels_vectorized(processed_signal, dead_pixels)

        # Step 3: Perform linearity correction
        linear_corr = calibration_data['linear_corr']
        processed_signal = linearity_correction(processed_signal, linear_corr)

        # Step 4: Subtract dark frame
        dark_frame = calibration_data['dark']
        processed_signal = subtract_dark(processed_signal, dark_frame)

        # Step 5: Get Correlated Double Sampling (CDS) - THE NEW STEP
        processed_signal = get_cds(processed_signal)

        # Step 6: Perform flat field correction
        flat_frame = calibration_data['flat']
        processed_signal = flat_correction(processed_signal, flat_frame)

        return processed_signal

    else:
        processed_signal = reverse_adc(raw_signal)
        # processed_signal = fix_dead_pixels_vectorized(processed_signal, dead_pixels)
        # processed_signal = linearity_correction(processed_signal, calibration_data['linear_corr'])
        # processed_signal = subtract_dark(processed_signal, calibration_data['dark'])
        # processed_signal = flat_correction(processed_signal, calibration_data['flat'])
        processed_signal = get_cds(processed_signal)
        return processed_signal

def read_signal(path):
    """
    Read data from a parquet file and return a PyTorch tensor.
    """
    # Read the parquet file using PyArrow
    arrow_table = pq.read_table(path)

    # Convert to Pandas DataFrame and then to NumPy array
    numpy_array = arrow_table.to_pandas(zero_copy_only=False).to_numpy()

    # Convert to PyTorch tensor
    if "AIRS-CH0" in path:
        tensor = torch.tensor(numpy_array, dtype=torch.float32, device="cuda").reshape(-1, 32, 356)
    else:
        tensor = torch.tensor(numpy_array, dtype=torch.float32, device="cuda")

    return tensor

def read_calibration(calibration_base_path):
    """
    Read calibration data from a parquet file and return a PyTorch tensor.
    """
    calibration_data=dict()
    calibration_name = ["dark", "dead", "linear_corr", "flat"]
    for name in calibration_name:
        path = os.path.join(calibration_base_path, f"{name}.parquet")
        # Read the parquet file using PyArrow
        arrow_table = pq.read_table(path)

        # Convert to Pandas DataFrame and then to NumPy array
        numpy_array = arrow_table.to_pandas(zero_copy_only=False).to_numpy()

        # Convert to PyTorch tensor
        tensor = torch.tensor(numpy_array, dtype=torch.float32, device="cuda")
        calibration_data[name] = tensor

    return calibration_data



import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# import preprocess

AIR_SLICE = (39,321)
# WIN_FGS = (23500, 44000)
# WIN_AIRS = (1958, 3666)
# WIN_FGS_OOT = 200
# WIN_AIRS_OOT = 170
BIN_LEN = 187 #时间分箱长度（作者使用）
CUT_BEGIN, CUT_END = 75, 115 # in-transit固定窗口 (作者使用）

def resample_1d(arr:torch.Tensor, target_len: int=187)->torch.Tensor:
    """
        将 1D 序列按均匀分箱重采样到 target_len。
        Parameters
        ----------
        arr : torch.Tensor, shape (T,)
            原始时间序列（例如白光）
        target_len : int
            目标分箱长度（默认 187）
        Returns
        -------
        torch.Tensor, shape (target_len,)
            分箱平均后的 1D 序列
        """
    T= arr.shape[0]
    m = T//target_len
    if m < 1:
        raise ValueError(f"Cannot resample {T} to {target_len} (m={m})")
    trimmed = arr[:m*target_len].view(target_len,m).mean(dim=1)
    return trimmed

def resample_2d_time(mat:torch.Tensor, bins:int=BIN_LEN)->torch.Tensor:
    """
        将 2D 矩阵按“时间维”均匀分箱到 bins。
        Parameters
        ----------
        mat : torch.Tensor, shape (T, W)
            例如 AIRS 的 (time, wavelength) 图
        bins : int
            目标时间分箱长度（默认 187）
        Returns
        -------
        torch.Tensor, shape (bins, W)
            分箱平均后的 2D 矩阵
        """
    T,m = mat.shape[0],mat.shape[0]//bins
    trim = mat[:m*bins].view(bins,m,-1).mean(dim=1)
    return trim

# Dataset class for CNN model
class ArielCNNDataset(Dataset):
    """
    读取并处理单个行星的观测，构造 1D/2D CNN 需要的输入与标签。
    输出字段（keys 不变）：
    - pid   : str
    - wc (white-light curve)   : torch.float32, shape (1, 187)
              * 对齐原作者：仅用 AIRS 构造白光
              * 步骤：AIRS 沿 y 求和 → (T,356) → 切 39..321 → (T,283)
                      → time-binning 到 187 → (187,283)
                      → white_curve = sum_over_lambda / mean_over_all_pixels
              * 不在 Dataset 内做 per-sample min-max。训练阶段用“训练集全局 min/max”。
    - map2d : torch.float32, shape (1, 40, 283)
              * 对齐原作者：做星光谱归一化（头 50 + 尾 50 帧），
                切 in-transit 固定窗口 [75:115) 40 帧，
                对整块做“去均值”（减去一个标量），最后加通道维。
              * [-1,1] 的归一化也放在训练阶段（用训练集统计量）。
    - （仅 train 时）
      y_mean  : torch.float32, shape (1,)   —— 光谱均值（1D-CNN 目标）
      y_shift : torch.float32, shape (283,) —— 去均值后的光谱（2D-CNN 目标）
    Notes
    -----
    * 快路径（disk_cache_dir）期望缓存中的 wc 为形状 (1,187)，map2d 为 (1,40,283)。
      如果你还在用旧缓存（wc 是 (1,374)），会抛出明确的错误提示。
    """
    def __init__(self,base_dir,planet_ids=None,split="train",return_labels=True,cache_calib=True,disk_cache_dir=None):
        self.base_dir = base_dir
        self.split = split
        self.return_labels = return_labels and (split == "train")
        self.cache_calib = cache_calib
        self._calib_cache = {}
        self.disk_cache_dir = disk_cache_dir

        # planet list
        if planet_ids is None:
            split_dir = os.path.join(base_dir,split)
            planet_ids = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir,d)))
        self.pids = planet_ids

        if self.return_labels:
            df = pd.read_csv(os.path.join(base_dir, "train.csv")).set_index("planet_id")
            df.index =df.index.map(str) #保证与目录名一致
            self._y_map = {str(idx):row.values.astype("float32") for idx, row in df.iterrows()}

    # # ---- 低级 I/O：读 parquet + 标定 → 返回已做 CDS 的张量 ----
    def _load_sensor(self, pid, band):
        """
                Parameters
                ----------
                pid : str
                band: str, in {"AIRS-CH0", "FGS1"}
                Returns
                -------
                torch.Tensor
                    AIRS : (5625, 32, 356)
                    FGS1 : (67500, 1024)
                """
        pdir = os.path.join(self.base_dir, self.split, pid)
        sig_path = os.path.join(pdir, f"{band}_signal_0.parquet")
        cal_dir = os.path.join(pdir, f"{band}_calibration_0")

        raw = read_signal(sig_path)

        if self.cache_calib and (pid, band) in self._calib_cache:
            calib = self._calib_cache[(pid, band)]
        else:
            calib = read_calibration(cal_dir)
            if self.cache_calib:
                self._calib_cache[(pid, band)] = calib

        processed_signal = process_data(raw, calib)# AIRS:(5625,32,356) / FGS1:(67500,1024)
        return processed_signal

    def __len__(self): return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]

        # 加个快速通道：如果有缓存（from .npz cache)直接读并返回：
        if self.disk_cache_dir is not None:
            npz_path = os.path.join(self.disk_cache_dir, f"{pid}.npz")
            if os.path.exists(npz_path):
                z = np.load(npz_path)
                sample = {
                    "pid": pid,
                    "wc": torch.tensor(z["wc"], dtype=torch.float32),  # (1, 187)
                    "map2d": torch.tensor(z["map2d"], dtype=torch.float32),  # (1, 40, 283)
                }

                # 形状检查
                if sample["wc"].shape!= (1,187):
                    raise ValueError(
                        f"[Cache shape mismatch] wc shape {tuple(sample['wc'].shape)} != (1,187). "
                        f"请切换到新缓存目录（如 cnn_cache_wc187/train），或重建缓存。"
                    )
                if sample["map2d"].shape != (1, 40, 283):
                    raise ValueError(
                        f"[Cache shape mismatch] map2d shape {tuple(sample['map2d'].shape)} != (1,40,283)."
                    )

                if self.return_labels:
                    sample.update({
                        "y_mean": torch.tensor(z["y_mean"], dtype=torch.float32),  # (1,)
                        "y_shift": torch.tensor(z["y_shift"], dtype=torch.float32)  # (283,)
                    })
                return sample
        # 慢路径：现算
        # -----FGS 白光：像素和-》（67500，）暂时用不着，先封起来
        # fgs = self._load_sensor(pid, "FGS1") #(67500,1024)
        # fgs_white = fgs.sum(dim=1).cpu()# (67500,)

        # --- AIRS: 沿 y 求和 -> (T,356)，再切 39–321 -> (T,283)
        airs = self._load_sensor(pid,"AIRS-CH0") #(5625,32,356)
        airs_2d = airs.sum(dim=1)[:,AIR_SLICE[0]:AIR_SLICE[1]+1].cpu() # (5625,283)

        # === 1D-CNN 的 wc：对齐原作者（仅 AIRS，(1,187)） ===
        # time-binning 到 187 帧
        airs_2d_bin = resample_2d_time(airs_2d) # (187,283)
        # 按作者：white_curve = sum_over_lambda/mean_over_all_pixels
        wc_mean = airs_2d_bin.mean()
        wc_187 = airs_2d_bin.sum(dim=1)/ (wc_mean+1e-8) # (187,)
        wc = wc_187.unsqueeze(0) # (1,187)，不做min-max 归一化per sample

        # === 2D-CNN 的 map2d：星光谱归一化 → in-transit 切片 → 整块去均值 ===
        # 星光谱：头 50 + 尾 50 帧（对齐作者 norm_star_spectrum）
        oot_left = airs_2d_bin[:50].mean(dim=0) # (283,)
        oot_right = airs_2d_bin[-50:].mean(dim=0) # (283,)
        star_spec = oot_left + oot_right # (283,)
        airs_2d_norm = airs_2d_bin/(star_spec.clamp_min(1e-8)) # (187,283)
        # in-transit 固定窗口 [75:115) 40 帧
        map_slice = airs_2d_norm[CUT_BEGIN:CUT_END] # (40,283)
        # 整块去均值（减去一个标量）
        map_centered = map_slice - map_slice.mean() # (40,283)
        # 加通道维度，得到 (1,40,283)
        map2d = map_centered.unsqueeze(0) # (1,40,283)

        sample = {"pid":pid, "wc":wc, "map2d":map2d}

        if self.return_labels:
            y_full = torch.from_numpy(self._y_map[pid]) # (283,)
            y_mean = y_full.mean().unsqueeze(0) # (1,)
            sample.update({
                "y_mean": y_mean,
                "y_shift": y_full - y_mean,
            })

        return sample


"""
One-Dimensional CNN for Ariel ADC 2025 (PyTorch, aligned with starter Keras)
============================================================================

输入  : 归一化白光曲线 wc — 形状 **(batch, 1, 187)**
         （仅 AIRS 构造：sum_λ / mean_all；不做 per-sample min-max。
          训练阶段用“训练集全局 min/max”统一缩放）
输出  : 光谱均值 μ̂ — 形状 **(batch, 1)**

MC Dropout : 在推理阶段保持 model.train()，多次前向取均值/方差。
"""

# from __future__ import annotations
from typing import Tuple
import torch
from torch import nn

# 模型本体
class Net1D(nn.Module):
    """
    Simple 1-D CNN aligned with the starter Keras model.

    Parameters
    ----------
    dropout_rate : float
        Dropout 概率（默认 0.2）

    Forward
    -------
    x : torch.Tensor, shape (B, 1, 187)
        White-light curve（已在训练阶段做统一归一化）
    Returns
    -------
    torch.Tensor, shape (B, 1)
        Predicted mean transit depth (linear)
    Notes
    -----
    - Conv1d(kernel=3, padding=0, 'valid') + MaxPool1d(2) × 4
    - 长度演化：187→92→45→21→9
    - Flatten 维度 = 256 * 9 = 2304（与 Keras 对齐）
    - 在第一层池化后加 BatchNorm1d(32)
    """
    def __init__(self, dropout_rate:float=0.2)->None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3), nn.ReLU(), # 187->185
            nn.MaxPool1d(2), # 185->92
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, 3), nn.ReLU(), # 92->90
            nn.MaxPool1d(2), # 90->45

            nn.Conv1d(64, 128, 3), nn.ReLU(), # 45->43
            nn.MaxPool1d(2), # 43->21

            nn.Conv1d(128, 256, 3), nn.ReLU(), # 21->`19
            nn.MaxPool1d(2), # 19->9
        )
        with torch.no_grad():
            L = self.features(torch.zeros(1,1,187)).shape[-1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * L, 500), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 100), nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(100, 1)  # 输出 μ̂
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass.
            Parameters
                ----------
                x : torch.Tensor, shape (B, 1, 187)
                    Normalised white-light curve.
            Returns
                -------
                torch.Tensor, shape (B, 1)
                    Mean transit depth.
        """
        z = self.features(x)
        return self.head(z)

# Training / inference helpers
def train_1d_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optim: torch.optim.Optimizer,
        device:torch.device
)->float:
    """
        单个 epoch 训练函数。
        Parameters
        ----------
        model      : Net1D (already on device)
        dataloader : yields dict with keys "wc", "y_mean"
        optim      : torch optimizer
        device     : torch.device
        Returns
        -------
        float : epoch mean loss (MSE)
    """
    model.train()
    mse = nn.MSELoss()
    running = 0.0
    for batch in dataloader:
        # 注意：在 Notebook/主程序里，先对 batch["wc"] 做“训练集全局 min/max”的统一归一化
        x = batch["wc"].to(device, non_blocking=True) #(B,1,187)
        y = batch["y_mean"].to(device, non_blocking=True)#(B,1)
        optim.zero_grad()
        y_hat = model(x)
        loss = mse(y_hat, y)
        loss.backward()
        optim.step()
        running += loss.item() * x.size(0)
    return running / len(dataloader.dataset)

def enable_dropout_only(model:nn.Module)->None:
    """
        Keep model in eval() but force ONLY Dropout layers into train() so MC dropout works,
        while BatchNorm stays in eval() (frozen running stats).
    """
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Dropout,torch.nn.AlphaDropout)):
            mod.train()
        elif isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            mod.eval()

@torch.no_grad()
def predict_mc_dropout(
        model:nn.Module,
        x:torch.Tensor,
        n_samples:int=200,
)->Tuple[torch.Tensor, torch.Tensor]:
    """
        多次前向推理以估计 (μ̂, σ_μ̂)。
        Parameters
        ----------
        model      : Net1D (keep in *train* mode to enable Dropout)
        x          : torch.Tensor, shape (B, 1, 374)  — already on same device
        n_samples  : int, default 200， 采样次数
        Returns
        -------
        μ_hat : torch.Tensor, shape (B, 1)
        σ_hat : torch.Tensor, shape (B, 1)
    """
    model.eval()
    enable_dropout_only(model)

    preds = []
    for _ in range(n_samples):
        preds.append(model(x))
    preds = torch.stack(preds, dim=0)
    mu = preds.mean(dim=0)
    sig = preds.std(dim=0)
    return mu, sig

@torch.no_grad()
def mc_1d_on_loader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 200,
):
    """
    返回:
      mu  : (N,)   —— 反归一化后的 y_mean 预测均值
      sig : (N,)   —— 反归一化后的 y_mean 预测标准差
      y   : (N,)   —— 真值（未归一化）
    依赖外部的 scale/unscale 函数: scale_wc_batch, unscale_y_batch 以及全局 ymin/ymax
    """
    model.eval()
    enable_dropout_only(model)

    mu_list, sig_list, y_true = [], [], []
    for batch in dataloader:
        x = scale_wc_batch(batch["wc"].to(device))          # (B,1,187) in [0,1]
        y = batch["y_mean"].to(device)                      # (B,1) original scale

        # 多次前向，收集在 CPU，避免显存积累
        outs = []
        for _ in range(n_samples):
            outs.append(model(x).detach().cpu())            # (B,1) in [0,1] scale
        outs = torch.stack(outs, dim=0)                     # (S,B,1)

        mu_norm = outs.mean(dim=0).squeeze(-1)              # (B,)
        sig_norm = outs.std(dim=0).squeeze(-1)              # (B,)

        # 反归一化：均值线性可直接整体反归一化；std 乘以尺度
        mu = unscale_y_batch(mu_norm.numpy().reshape(-1, 1)).squeeze(1)                 # (B,)
        sig = (sig_norm.numpy() * (ymax - ymin + 1e-8))                                   # (B,)

        mu_list.append(mu)
        sig_list.append(sig)
        y_true.append(y.cpu().numpy().squeeze(1))

    mu = np.concatenate(mu_list, axis=0)    # (N,)
    sig = np.concatenate(sig_list, axis=0)  # (N,)
    y  = np.concatenate(y_true, axis=0)     # (N,)
    return mu, sig, y


# net2d.py
# --------------------------------------------
# 2D CNN (PyTorch) for Ariel ADC 2025
# 输入: map2d ∈ R^{B, 1, 40, 283}  —— AIRS 2D切片，已做星光谱归一化 & 整块去均值
# 目标: y_shift ∈ R^{B, 283}       —— 去均值后的光谱（均值≈0）
# 规范化: 训练阶段用 train 统计量做 [-1,1] 归一化：
#        map2d / data_abs_max,  y_shift / targets_abs_max
# 输出: Δ̂ ∈ R^{B, 283}（线性）
# 不确定度: 用 MC-Dropout 多次前向（保持 model.train()）估计均值与方差
# --------------------------------------------
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


def _same_pad_3x1()->Tuple[int, int]:
    """
    返回 3x1 卷积的 padding='same'
    """
    return (1,0)

def _same_pad_1x3()->Tuple[int, int]:
    """
    返回 1x3 卷积的 padding='same'
    """
    return (0,1)

# 2D-CNN（预测去均值的 283 维形状 Δ̂）
# 简单复刻 Keras 的拓扑（时向和波向交替卷积/池化）。用动态推断 Flatten 维度，不纠结手算。

class Net2D(nn.Module):
    """
    2D CNN（贴近原作者 Keras 拓扑，语义化命名 + 动态展平）
    Parameters
    ----------
    time_len : int
        输入的时间维长度（默认 40，对应固定窗口 75~115）
    wave_len : int
        输入的波长维长度（默认 283，对应 39..321）
    dropout : float
        全连接头部的 dropout 概率
    Forward
    -------
    x : torch.Tensor, shape (B, 1, time_len, wave_len)
        2D 观测（map2d），训练时请先做全局 [-1,1] 规范化：x /= data_abs_max
    Returns
    -------
    torch.Tensor, shape (B, wave_len)
        预测的 Δ̂（零均值的光谱形状）
    """
    def __init__(self,time_len:int=40,wave_len:int=283,dropout:float=0.2):
        super().__init__()
        self.time_len = time_len
        self.wave_len = wave_len

        # 时向卷积块（与原作者的(3,1)卷积一致）
        self.block_t1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), padding=_same_pad_3x1()),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # 时间维/2
            nn.BatchNorm2d(32)
        )
        self.block_t2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=_same_pad_3x1()),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block_t3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=_same_pad_3x1()),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.block_t4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,1), padding=_same_pad_3x1()),
            nn.ReLU(),
        )

        # 波向卷积块（与原作者的(1,3)卷积一致）
        self.block_w1 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=(1,3), padding=_same_pad_1x3()),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)), # 波长维/2
            nn.BatchNorm2d(32)
        )
        self.block_w2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=_same_pad_1x3()),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.block_w3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=_same_pad_1x3()),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.block_w4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 3), padding=_same_pad_1x3()),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # 动态推断展平维度，避免Linear维度不匹配
        with torch.no_grad():
            dummy = torch.zeros(1,1,time_len,wave_len)
            z = self._forward_features(dummy)
            self.flat_dim = z.numel()

        # 全连接头（与原作者700-》283对齐）
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(700), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(700, wave_len),
        )

    def _forward_features(self, x: torch.Tensor)->torch.Tensor:
        """仅做卷积与池化，返回 (B, C, T', W')。"""
        z = self.block_t1(x)  # (B, 32, 20, 283)
        z = self.block_t2(z)  # (B, 64, 10, 283)
        z = self.block_t3(z)  # (B,128,  5, 283)
        z = self.block_t4(z)  # (B,256,  5, 283)
        z = self.block_w1(z)  # (B, 32,  5, ~141)
        z = self.block_w2(z)  # (B, 64,  5, ~70)
        z = self.block_w3(z)  # (B,128,  5, ~35)
        z = self.block_w4(z)  # (B,256,  5, ~17)
        return z

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
            Parameters
            ----------
                x : torch.Tensor, shape (B, 1, time_len, wave_len)
                    训练阶段请先做：x = x / data_abs_max
            Returns
            -------
                torch.Tensor, shape (B, wave_len)
                    预测 Δ̂
        """
        z = self._forward_features(x)
        y_hat = self.head(z) # (B, 283)
        return y_hat

# ---------------------------
# 训练 / 评估 / MC Dropout API
# ---------------------------

def train_2d_epoch(
        model:nn.Module, dataloader:torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device:torch.device,
        data_abs_max:float,
        targets_abs_max:float,
)->float:
    """
        训练单个 epoch（MSE 对 y_shift）
        Parameters
        ----------
        model : Net2D（已 to(device)）
        dataloader : 产出 dict，包括 "map2d", "y_shift"
            - batch["map2d"] : (B,1,40,283)
            - batch["y_shift"]: (B,283)
        optimizer : torch optimizer
        device : torch.device
        data_abs_max : float
            训练集上 map2d 的绝对最大值（全局），用于 map2d 归一化
        targets_abs_max : float
            训练集上 y_shift 的绝对最大值（全局），用于 y_shift 归一化
        Returns
        -------
        float : epoch 平均损失（MSE）
    """
    model.train()
    mse = nn.MSELoss()
    running = 0.0

    for batch in dataloader:
        # [-1,1] 归一化（按训练集统计量）
        x = (batch["map2d"]/(data_abs_max+1e-12)).to(device, non_blocking=True) # (B,1,40,283)
        y = (batch["y_shift"]/(targets_abs_max+1e-12)).to(device, non_blocking=True) # (B,283)

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x)
        loss = mse(y_hat,y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)

    return running/len(dataloader.dataset)

@torch.no_grad()
def eval_2d_rmse(
        model:nn.Module,
        dataloader:torch.utils.data.DataLoader,
        device:torch.device,
        data_abs_max:float,
        targets_abs_max:float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
        评估 RMSE（反归一化后，以物理尺度）
        Returns
        -------
        rmse : float
            预测 Δ̂ 与真值 Δ 的 RMSE
        preds : np.ndarray, shape (N, 283)
            反归一化后的预测
        trues : np.ndarray, shape (N, 283)
            真值（未归一化）
    """
    model.eval()
    preds, trues = [], []
    for batch in dataloader:
        x = (batch["map2d"]/(data_abs_max+1e-12)).to(device)
        y = batch["y_shift"].to(device)
        y_hat = model(x).cpu().numpy() * (targets_abs_max+1e-12) # 反归一化
        preds.append(y_hat)
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds, axis=0) # (N,283)
    trues = np.concatenate(trues, axis=0) # (N,283)
    rmse = float(((preds-trues)**2).mean()**0.5)

    return rmse, preds, trues

def enable_dropout_only(model:nn.Module)->None:
    """
        Keep model in eval() but force ONLY Dropout layers into train() so MC dropout works,
        while BatchNorm stays in eval() (frozen running stats).
    """
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Dropout,torch.nn.AlphaDropout)):
            mod.train()
        elif isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            mod.eval()

@torch.no_grad()
def mc_2d_on_loader(
        model:nn.Module,
        dataloader:torch.utils.data.DataLoader,
        device:torch.device,
        n_samples:int,
        data_abs_max:float,
        targets_abs_max:float,
):
    """
        MC Dropout：重复前向 n 次（保持 model.train() 使 Dropout 生效）估计 (μ, σ)
        Returns
        -------
        mu : np.ndarray, shape (N,283)
            预测均值（反归一化）
        sig: np.ndarray, shape (N,283)
            预测标准差（反归一化）
        y_true : np.ndarray, shape (N,283)
            真值（未归一化）
    """
    model.eval()
    enable_dropout_only(model)
    mu_list, sig_list, y_true = [],[],[]

    for batch in dataloader:
        x = (batch["map2d"]/(data_abs_max+1e-12)).to(device)
        y = batch["y_shift"].to(device)

        samples = []
        for _ in range(n_samples):
            samples.append(model(x).cpu().numpy()) #(B,283)
        samples =  np.stack(samples,axis=0) # (S,B,283)

        mu = samples.mean(axis=0)*(targets_abs_max+1e-12)
        sig = samples.std(axis=0)*(targets_abs_max+1e-12)

        mu_list.append(mu)
        sig_list.append(sig)
        y_true.append(y.cpu().numpy())

    mu = np.concatenate(mu_list,axis=0)
    sig = np.concatenate(sig_list,axis=0)
    y_true = np.concatenate(y_true,axis=0)
    return mu, sig, y_true


import os, glob, time, random, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# from dataset_cnn import ArielCNNDataset, resample_2d_time, AIR_SLICE, CUT_BEGIN, CUT_END
# from net1d import Net1D, train_1d_epoch, predict_mc_dropout

BASE = "/kaggle/input/ariel-data-challenge-2025"
OLD_CACHE = "cnn_cache/train"
NEW_CACHE = "cnn_cache_wc187/train"
os.makedirs(NEW_CACHE,exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def rebuild_wc_cache_to_187(base_dir, old_cache, new_cache):
    """
    只重算 wc -> (1,187),之前的是错误拼接导致的(1,374)，map2d/y_* 尽量从 old_cache 拷贝；old_cache 没有就现算。
    """
    ds = ArielCNNDataset(base_dir, split="train", return_labels=True, cache_calib=True, disk_cache_dir=None)
    
    def build_wc187_from_airs(airs_2d: torch.Tensor)->np.ndarray:
        # airs_2d: (5625,283)  -> time bin -> (187,283) -> white_curve
        airs_2d_bin = resample_2d_time(airs_2d)                  # (187,283)
        wc_mean = airs_2d_bin.mean()
        wc_187  = airs_2d_bin.sum(dim=1) / (wc_mean + 1e-8)     # (187,)
        return wc_187.unsqueeze(0).numpy().astype("float32")     # (1,187)
    
    print("Rebuilding wc cache to 187 points... ->", os.path.abspath(new_cache))
    for pid in tqdm(ds.pids):
        outp = os.path.join(new_cache, f"{pid}.npz")
        if os.path.exists(outp):
            continue  # 已存在，跳过
        
        # 读取旧缓存(拷贝map2d/y_*)
        oldp = os.path.join(old_cache, f"{pid}.npz")
        z_old = np.load(oldp) if os.path.exists(oldp) else None
        
        # 读AIRS->wc
        airs = ds._load_sensor(pid, "AIRS-CH0")  # (5625, 32,356)
        airs_2d = airs.sum(dim=1)[:, AIR_SLICE[0]:AIR_SLICE[1]+1].cpu()  # (5625,283)
        wc = build_wc187_from_airs(airs_2d)  # (1,187)
        
        if z_old is not None:
            np.savez_compressed(
                outp,
                wc=wc,
                map2d=z_old["map2d"],  # (1,40,283)
                y_mean=z_old["y_mean"],  # (283,)
                y_shift=z_old["y_shift"]  # (283,)
            )
        else:
            # 没有旧缓存，现算map2d & y
            airs_2d_bin = resample_2d_time(airs_2d)                      # (187,283)

            # 星光谱归一化 -> in-transit 切片 -> 整块去均值（对齐作者）
            oot_left  = airs_2d_bin[:50].mean(dim=0)
            oot_right = airs_2d_bin[-50:].mean(dim=0)
            star_spec = oot_left + oot_right
            airs_2d_norm = airs_2d_bin / (star_spec.clamp_min(1e-8))
            map_slice = airs_2d_norm[CUT_BEGIN:CUT_END]                  # (40,283)
            map2d = (map_slice - map_slice.mean()).unsqueeze(0).numpy().astype("float32")  # (1,40,283)

            # y 标签
            y_full = torch.from_numpy(ds._y_map[str(pid)]).float()
            y_mean = y_full.mean().unsqueeze(0).numpy().astype("float32")
            y_shift = (y_full - y_mean).numpy().astype("float32")
            np.savez_compressed(outp, wc=wc, map2d=map2d, y_mean=y_mean, y_shift=y_shift)


rebuild_wc_cache_to_187(BASE, OLD_CACHE, NEW_CACHE)


# 8:2 划分训练和验证集
def split_planet_ids(planet_ids, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    ids = list(map(str,planet_ids))
    rng.shuffle(ids)
    k = int(len(ids) * train_ratio)
    return ids[:k], ids[k:]  # 返回训练集和验证集的 planet_id 列表

#用 NEW_CACHE里已有的pid
pid_files = sorted(glob.glob(os.path.join(NEW_CACHE, "*.npz")))
all_ids = [os.path.splitext(os.path.basename(p))[0]for p in pid_files]
train_ids, valid_ids = split_planet_ids(all_ids, train_ratio=0.8, seed=42)
len(all_ids), len(train_ids), len(valid_ids)  # 检查划分结果


'''
统计全局归一化参数（对齐作者方式）
1D：wc 用训练集 全局 min/max；y_mean 用训练集 min/max。
2D：targets（y_shift）和 obs（map2d）都用训练集的绝对最大值进行 [-1,1] 规范化。
'''
# 扫描train缓存，统计scaler
W_list, Ymean_list, Yshift_list, MAP_list = [], [], [], []

for pid in tqdm(train_ids, desc="scan train scalers"):
    z = np.load(os.path.join(NEW_CACHE, f"{pid}.npz"))
    W_list.append(z["wc"].squeeze(0))  # (187,)
    Ymean_list.append(z["y_mean"].squeeze(0))  # ()
    Yshift_list.append(z["y_shift"])  # (283,)
    MAP_list.append(z["map2d"].squeeze(0))  # (40,283)
W = np.stack(W_list) # (N_train, 187)
Ymean = np.stack(Ymean_list)  # (N_train, 283)
Yshift = np.stack(Yshift_list)  # (N_train, 283)
MAP = np.stack(MAP_list)  # (N_train, 40, 283)

# 计算全局 min/max
# 1D scalers
wc_min, wc_max = W.min(), W.max()
y_min, y_max = Ymean.min(), Ymean.max()

# 2D scalers
targets_abs_max = max(abs(Yshift.min()), abs(Yshift.max()))
data_abs_max = max(abs(MAP.min()), abs(MAP.max()))

print("wc_min/max:", float(wc_min), float(wc_max))
print("y_mean min/max:", float(y_min), float(y_max))
print("targets_abs_max:", float(targets_abs_max))
print("data_abs_max:", float(data_abs_max))

# 保存scalers,训练/预测都用同一份
os.makedirs("models", exist_ok=True)
np.savez("models/scalers_v1.npz",
         wc_min=wc_min, wc_max=wc_max,
         y_min=y_min, y_max=y_max,
         targets_abs_max=targets_abs_max,
         data_abs_max=data_abs_max)


bs = 32
cuda = device.type == "cuda"
dl_kwargs = dict(batch_size=bs, num_workers=0, pin_memory=cuda, persistent_workers=False)
ds_train = ArielCNNDataset(BASE, planet_ids=train_ids, split="train",
                           return_labels=True, disk_cache_dir=NEW_CACHE)
ds_valid = ArielCNNDataset(BASE, planet_ids=valid_ids, split="train",
                           return_labels=True, disk_cache_dir=NEW_CACHE)
dl_train = DataLoader(ds_train, shuffle=True, **dl_kwargs)
dl_valid = DataLoader(ds_valid, shuffle=False, **dl_kwargs)

b = next(iter(dl_train))
print("Batch shapes:", b["wc"].shape, b["map2d"].shape, b["y_mean"].shape, b["y_shift"].shape) # (B,1,187), (B,1,40,283), (B,1), (B,283)


# 训练辅助：归一化/反归一化函数
# 载入scalers
sc = np.load("models/scalers_v1.npz")
wc_min = float(sc["wc_min"])
wc_max = float(sc["wc_max"])
ymin = float(sc["y_min"])
ymax = float(sc["y_max"])
targets_abs_max = float(sc["targets_abs_max"])
data_abs_max = float(sc["data_abs_max"])

# 1D helpers
def scale_wc_batch(x): # x: (B,1,187)
    return (x-wc_min) / (wc_max - wc_min + 1e-8)  # scale to [0,1]
def scale_y_batch(y): # y: (B,1)
    return (y - ymin) / (ymax - ymin + 1e-8)  # scale to [0,1]
def unscale_y_batch(y): # x: (B,1)
    return y * (ymax - ymin + 1e-8) + ymin  # unscale to original

# 2D helpers
def scale_obs2d_batch(m): # m: (B,1,40,283)
    return m / (data_abs_max + 1e-8)  # scale to [-1,1]
def scale_target2d_batch(t): # t: (B,283)
    return t / (targets_abs_max + 1e-8)  # scale to [-1,1]
def unscale_target2d_batch(t): # t: (B,283)
    return t * (targets_abs_max + 1e-8)  # unscale to original


# 1D CNN 模型
net1d = Net1D(dropout_rate=0.2).to(device)
opt = torch.optim.Adam(net1d.parameters(), lr=0.001, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.2)

def train_1d_epoch_scaled(model, dataloader, optim, device):
    model.train()
    mse = torch.nn.MSELoss()
    running = 0.0
    for batch in dataloader:
        x = scale_wc_batch(batch["wc"].to(device, non_blocking=True))
        y = scale_y_batch(batch["y_mean"].to(device, non_blocking=True))
        optim.zero_grad()
        y_hat = model(x)
        loss = mse(y_hat, y)
        loss.backward()
        optim.step()
        running += loss.item() * x.size(0)
    return running / len(dataloader.dataset)

@torch.no_grad()
def eval_1d_rmse(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    for batch in dataloader:
        x = scale_wc_batch(batch["wc"].to(device))
        y = batch["y_mean"].to(device)
        p = unscale_y_batch(model(x).cpu().numpy())
        preds.append(p[:,0])
        trues.append(y.cpu().numpy()[:,0])
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    return rmse, preds, trues


EPOCHS_1D = 300
best_rmse = 1e9; best_path = "models/net1d.pt"
t0 = time.time()
for ep in range(1, EPOCHS_1D+1):
    loss = train_1d_epoch_scaled(net1d, dl_train, opt,device)
    sch.step()
    if ep%10 == 0 or ep == 1:
        rmse_tr, _, _= eval_1d_rmse(net1d, dl_train, device)
        rmse_va, _, _ = eval_1d_rmse(net1d, dl_valid, device)
        print(f"[1D] ep {ep:4d} loss {loss:.6} RMSE(tr/va) {rmse_tr:.6e}/{rmse_va:.6e}")
        if rmse_va < best_rmse:
            best_rmse = rmse_va
            torch.save(net1d.state_dict(), best_path)
            print(f"New best RMSE: {best_rmse:.6e}, saved to {best_path}")
print("1D done in %.1f s"%(time.time()-t0))


# 载入最佳 1D 并画散点
net1d.load_state_dict(torch.load("models/net1d.pt", map_location=device))
rmse_va, p_va, y_va = eval_1d_rmse(net1d, dl_valid, device)
plt.figure(); plt.scatter(y_va, p_va, s=8, alpha=0.5)
m1, m2 = min(y_va.min(), p_va.min()), max(y_va.max(), p_va.max())
plt.plot([m1,m2],[m1,m2],'k--'); plt.title(f"1D valid RMSE={rmse_va:.6e}")
plt.xlabel("y_mean true"); plt.ylabel("y_mean pred"); plt.show()


@torch.no_grad()
def mc_1d_on_loader(model, dataloader, device, n_samples=500):
    model.eval()
    mu_list, sig_list, y_true = [], [], []
    for batch in dataloader:
        x = scale_wc_batch(batch["wc"].to(device))
        y = batch["y_mean"].to(device)
        mu, sig = predict_mc_dropout(model, x, n_samples=n_samples)
        mu = unscale_y_batch(mu).cpu().numpy()[:,0]
        sig = sig.cpu().numpy()[:,0] * (ymax-ymin+1e-8)
        mu_list.append(mu); sig_list.append(sig); y_true.append(y.cpu().numpy()[:,0])
    return np.concatenate(mu_list), np.concatenate(sig_list), np.concatenate(y_true)

mu_va, sig_va, y_va = mc_1d_on_loader(net1d, dl_valid, device, n_samples=200)
print("1D MC-Dropout: mu/std mean", mu_va.mean(), sig_va.mean())


mu_va, sig_va, y_va = mc_1d_on_loader(net1d, dl_valid, device, n_samples=200)
np.savez("models/one_d_valid.npz", mu=mu_va, sig=sig_va, y=y_va)


# from net2d import Net2D, train_2d_epoch, eval_2d_rmse, mc_2d_on_loader

net2d = Net2D(time_len=40, wave_len=283, dropout=0.2).to(device)
opt2 = torch.optim.Adam(net2d.parameters(),lr=1e-3,weight_decay=1e-6)

EPOCHS_2D = 120
best_rmse, best_path = 1e9, "models/net2d.pt"
for ep in range(1, EPOCHS_2D+1):
    loss = train_2d_epoch(net2d,dl_train,opt2,device,data_abs_max,targets_abs_max)
    if ep%10 == 0 or ep ==1:
        rmse_tr,_,_ = eval_2d_rmse(net2d, dl_train, device, data_abs_max, targets_abs_max)
        rmse_va, _, _ = eval_2d_rmse(net2d, dl_valid, device, data_abs_max, targets_abs_max)
        print(f"[2D] ep {ep:4d} loss {loss:.6} RMSE(tr/va) {rmse_tr:.6e}/{rmse_va:.6e}")
        if rmse_va < best_rmse:
            best_rmse = rmse_va
            os.makedirs("models", exist_ok=True)
            torch.save(net2d.state_dict(),best_path)
            print("  saved:", best_path)

# MC Dropout 20 次
net2d.load_state_dict(torch.load("models/net2d.pt", map_location=device))
mu2_va, sig2_va, y2_va =mc_2d_on_loader(net2d,dl_valid,device,n_samples=20,data_abs_max=data_abs_max,targets_abs_max=targets_abs_max)


# 加载一下训练完的结果
# from net2d import Net2D, eval_2d_rmse

# -- load best 2D --
net2d = Net2D(time_len=40, wave_len=283, dropout=0.2).to(device)
net2d.load_state_dict(torch.load("models/net2d.pt", map_location=device))

rmse_va, preds_va, trues_va = eval_2d_rmse(
    net2d, dl_valid, device, data_abs_max, targets_abs_max
)
print(f"[2D] valid RMSE (Δ̂): {rmse_va:.6e}  ({rmse_va*1e6:.1f} ppm)")


# wavelengths
wavelength = pd.read_csv(os.path.join(BASE, "wavelengths.csv")).values.squeeze()

per_wl_rmse = np.sqrt(((preds_va - trues_va)**2).mean(axis=0))  # (283,)
plt.figure(figsize=(8,3))
plt.plot(wavelength, per_wl_rmse*1e6)
plt.xlabel("Wavelength (μm)")
plt.ylabel("RMSE (ppm)")
plt.title("Per-wavelength RMSE on valid (2D Δ̂)")
plt.grid(alpha=0.3)
plt.show()

# 做 2D MC Dropout（例如 50 次）
mu2_va, sig2_va, y2_va = mc_2d_on_loader(
    net2d, dl_valid, device, n_samples=50,
    data_abs_max=data_abs_max, targets_abs_max=targets_abs_max
)

# 画几个样本
idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 自己改
for i in idxs:
    plt.figure(figsize=(7,3))
    plt.title(f"2D shape prediction (valid idx={i})")
    plt.plot(wavelength, y2_va[i], color="tomato", label="Δ true")
    plt.plot(wavelength, mu2_va[i], ".k", ms=3, label="Δ pred")
    plt.fill_between(wavelength, mu2_va[i]-sig2_va[i], mu2_va[i]+sig2_va[i],
                     color="silver", alpha=0.5, label="Δ ± σ")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("(Rp/Rs)^2 (zero-mean)")
    plt.legend()
    plt.tight_layout()
    plt.show()



def compose_final_spectrum(
        mu_1d:np.ndarray, # (N,),   1D y_mean predictions (original scale)
        sigma_1d:np.ndarray, # (N,),   1D std (original scale)
        mu_2d:np.ndarray, # (N,283), 2D delta predictions (original scale)
        sigma_2d: np.ndarray # (N,283), 2D std (original scale)
)->tuple[np.ndarray,np.ndarray]:
    """
    Combine μ̂ (scalar per sample) and Δ̂(λ) to get FINAL spectrum and its uncertainty.
    Returns
    -------
    y_final_pred : (N, 283)
        Final spectrum prediction for each sample.
    sigma_final  : (N, 283)
        Uncalibrated predictive std under independence assumption:
        Var_final = Var_mu + Var_delta.
    Notes
    -----
    - We assume independence between μ̂ and Δ̂ when combining uncertainties.
    """
    y_final_pred = mu_1d[:,None] + mu_2d
    sigma_final = np.sqrt(sigma_1d[:,None]**2 + sigma_2d**2)
    return y_final_pred, sigma_final

def calibrate_uncertainty_scalar(
        y_true: np.ndarray, # (N,283)
        y_pred: np.ndarray, # (N,283)
        sigma_pred: np.ndarray # (N,283)
)->tuple[float,float]:
    """
    Calibrate predictive std with a single scalar factor c to improve coverage.
    Returns
    -------
    c : float
        Scalar calibration factor. sigma <- c * sigma
    rmse : float
        RMSE on validation under original (physical) scale.
    Notes
    -----
    - c = RMSE / mean(sigma_pred). This is a simple, robust scalar calibration.
    """
    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = float(np.mean(sigma_pred)+1e-12)
    c = rmse/denom
    return c, rmse

def compute_coverage(
        y_true: np.ndarray, #(N,283)
        y_pred:np.ndarray, #(N,283)
        sigma_pred:np.ndarray, #(N,283)
        k: float = 1.0 # 1σ = 68%, 2σ = 95%
)->float:
    """
    Compute empirical coverage: fraction of points inside [y_pred ± k*sigma].
    Returns
    -------
    coverage : float
        Mean fraction of wavelengths covered over all samples.
    """
    lower = y_pred - k*sigma_pred
    upper = y_pred + k*sigma_pred
    inside = (y_true>=lower) & (y_true<=upper)
    return float(np.mean(inside))

def plot_final_sample(
    wavelengths: np.ndarray,      # (283,)
    y_true: np.ndarray,           # (283,)
    y_pred: np.ndarray,           # (283,)
    sigma: np.ndarray,            # (283,)
    sample_id: str | int = ""
) -> None:
    """
    Plot final spectrum for a single sample with ±1σ band.
    Notes
    -----
    - One figure per sample to keep visuals clean.
    """
    plt.figure(figsize=(8, 3))
    title = f"FINAL spectrum (sample {sample_id})" if sample_id != "" else "FINAL spectrum"
    plt.title(title)
    plt.plot(wavelengths, y_true, label="Target")
    plt.plot(wavelengths, y_pred, ".", ms=3, label="Prediction")
    plt.fill_between(wavelengths, y_pred - sigma, y_pred + sigma, alpha=0.35, label="±1σ")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel(r"$(R_p/R_s)^2$")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def save_valid_ouputs(
        path: str,
        y_true:np.ndarray, #(N,283)
        y_pred: np.ndarray, #(N,283)
        sigma_calibrated: np.ndarray, #(N,283)
        mu_1d: np.ndarray, #(N,)
        sigma_1d: np.ndarray, #(N,)
        mu_2d: np.ndarray, #(N,283)
        sigma_2d: np.ndarray
)->None:
    """
    Save arrays for later analysis / plotting.
    Outputs
    -------
    NPZ file with keys:
    - y_true, y_pred, sigma_calibrated, mu_1d, sigma_1d, mu_2d, sigma_2d
    """
    np.savez_compressed(
        path,
        y_true=y_true,
        y_pred=y_pred,
        sigma_calibrated=sigma_calibrated,
        mu_1d=mu_1d,
        sigma_1d=sigma_1d,
        mu_2d=mu_2d,
        sigma_2d=sigma_2d,
    )


# --------------------------
# 1) Compose FINAL on valid
#    (assumes you already have:
#     mu_va, sig_va, y_va from 1D;
#     mu2_va, sig2_va, y2_va from 2D;
#     wavelength from BASE/wavelengths.csv)
# --------------------------

# Sanity checks on shapes
assert mu_va.ndim == 1 and sig_va.ndim == 1, "1D arrays must be (N,)"
assert mu2_va.ndim == 2 and sig2_va.ndim == 2, "2D arrays must be (N,283)"
assert y_va.ndim == 1 and y2_va.ndim == 2, "Truth arrays must be (N,) and (N,283)"
assert mu2_va.shape[1] == sig2_va.shape[1] == y2_va.shape[1] == 283, "Wavelength dimension must be 283"
assert len(wavelength) == 283, "wavelength vector must have length 283"

# Compose prediction and uncalibrated sigma
y_final_pred, sigma_final = compose_final_spectrum(mu_va, sig_va, mu2_va, sig2_va)
# Compose ground-truth final spectra (mean + shift)
y_final_true = y_va[:, None] + y2_va

# Compute calibration factor and RMSE (before calibration)
calibration_factor_c, rmse_final_before = calibrate_uncertainty_scalar(y_final_true, y_final_pred, sigma_final)

# Calibrate sigma
sigma_final_calibrated = sigma_final * calibration_factor_c

# Coverage (after calibration; also report before)
coverage_68_before = compute_coverage(y_final_true, y_final_pred, sigma_final, k=1.0)
coverage_95_before = compute_coverage(y_final_true, y_final_pred, sigma_final, k=2.0)
coverage_68_after  = compute_coverage(y_final_true, y_final_pred, sigma_final_calibrated, k=1.0)
coverage_95_after  = compute_coverage(y_final_true, y_final_pred, sigma_final_calibrated, k=2.0)

# Per-wavelength RMSE on FINAL prediction
per_wl_rmse_final = np.sqrt(((y_final_pred - y_final_true) ** 2).mean(axis=0))  # (283,)

print(f"[FINAL] RMSE (before calib): {rmse_final_before:.6e}")
print(f"[FINAL] Scalar calibration c: {calibration_factor_c:.4f}")
print(f"[FINAL] Coverage 68%  before/after: {coverage_68_before:.3f} / {coverage_68_after:.3f}")
print(f"[FINAL] Coverage 95%  before/after: {coverage_95_before:.3f} / {coverage_95_after:.3f}")


# --------------------------
# 2) Plots
# --------------------------

# Per-wavelength RMSE curve
plt.figure(figsize=(8, 3))
plt.title("Per-wavelength RMSE on VALID (FINAL)")
plt.plot(wavelength, per_wl_rmse_final * 1e6)
plt.xlabel("Wavelength (μm)")
plt.ylabel("RMSE (ppm)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Show a few samples
sample_indices = [0, 1, 2, 3, 4]  # change as needed
for idx in sample_indices:
    plot_final_sample(
        wavelengths=wavelength,
        y_true=y_final_true[idx],
        y_pred=y_final_pred[idx],
        sigma=sigma_final_calibrated[idx],
        sample_id=idx
    )


# =========================
# FINAL: build submission.csv (ALIGNED: FGS1 + 282 AIRS)
# =========================
import os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ---- Base path & device ----
BASE = os.getenv("BASE_URL", "/kaggle/input/ariel-data-challenge-2025")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Load scalers (same as training) ----
sc = np.load("models/scalers_v1.npz")
wc_min = float(sc["wc_min"]); wc_max = float(sc["wc_max"])
ymin   = float(sc["y_min"]);  ymax   = float(sc["y_max"])
targets_abs_max = float(sc["targets_abs_max"])
data_abs_max    = float(sc["data_abs_max"])
print("Scalers loaded.")

def scale_wc_batch(x):        # x: (B,1,187)
    return (x - wc_min) / (wc_max - wc_min + 1e-8)

def unscale_y_batch(y):       # y: (B,1) or (B,)
    return y * (ymax - ymin + 1e-8) + ymin

# ---- Load best weights ----
net1d = Net1D(dropout_rate=0.2).to(device)
net1d.load_state_dict(torch.load("models/net1d.pt", map_location=device))
net1d.eval()

net2d = Net2D(time_len=40, wave_len=283, dropout=0.2).to(device)
net2d.load_state_dict(torch.load("models/net2d.pt", map_location=device))
net2d.eval()

def enable_dropout_only(model: torch.nn.Module)->None:
    """Enable MC Dropout while keeping BatchNorm frozen."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.AlphaDropout)):
            m.train()
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()

# ---- Test loader ----
ds_test  = ArielCNNDataset(BASE, split="test", return_labels=False, disk_cache_dir=None)
dl_test  = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=0,
                      pin_memory=(device.type=="cuda"), persistent_workers=False)
print("Test size:", len(ds_test))

# ---- MC-Dropout inference ----
S1D, S2D = 200, 50
enable_dropout_only(net1d)
enable_dropout_only(net2d)

all_pid = []
mu1d_list, sig1d_list = [], []
mu2d_list, sig2d_list = [], []

t0 = time.time()
for batch in dl_test:
    pids = batch["pid"]
    all_pid.extend(pids)

    # 1D: μ̂ and σ̂_μ   (FGS1 surrogate)
    x1 = scale_wc_batch(batch["wc"].to(device))          # (B,1,187)
    samples_1d = []
    for _ in range(S1D):
        yhat = net1d(x1)                                 # (B,1) in [0,1]
        samples_1d.append(yhat.detach().cpu().numpy())   # (B,1)
    s1 = np.stack(samples_1d, axis=0)                    # (S,B,1)
    mu1 = s1.mean(axis=0).squeeze(1)                     # (B,)
    sg1 = s1.std(axis=0).squeeze(1)                      # (B,)
    mu1 = unscale_y_batch(mu1)                           # (B,) original scale
    sg1 = sg1 * (ymax - ymin + 1e-8)                     # (B,)

    mu1d_list.append(mu1)
    sig1d_list.append(sg1)

    # 2D: Δ̂ and σ̂_Δ   (AIRS shape, 283 pix = 39..321)
    x2 = (batch["map2d"] / (data_abs_max + 1e-8)).to(device)  # (B,1,40,283)
    samples_2d = []
    for _ in range(S2D):
        yhat2 = net2d(x2)                                  # (B,283) in [-1,1]
        samples_2d.append(yhat2.detach().cpu().numpy())
    s2 = np.stack(samples_2d, axis=0)                      # (S,B,283)
    mu2 = s2.mean(axis=0) * (targets_abs_max + 1e-8)       # (B,283) -> original
    sg2 = s2.std(axis=0)  * (targets_abs_max + 1e-8)       # (B,283)

    mu2d_list.append(mu2)
    sig2d_list.append(sg2)

print(f"MC inference done in {time.time()-t0:.1f}s")

# ---- Concatenate over all batches ----
mu_1d = np.concatenate(mu1d_list, axis=0)            # (N,)
sig_1d = np.concatenate(sig1d_list, axis=0)          # (N,)
mu_2d  = np.concatenate(mu2d_list, axis=0)           # (N,283) AIRS 39..321
sig_2d = np.concatenate(sig2d_list, axis=0)          # (N,283)

# ---- ALIGNED assembly for Kaggle ----
# Kaggle expects 283 targets: [FGS1] + 282 AIRS points.
# We KEEP AIRS 39..320 (drop the last pixel 321 -> tail drop).
mu_2d_282  = mu_2d[:, :282]                          # (N,282) AIRS 39..320
sig_2d_282 = sig_2d[:, :282]                         # (N,282)

# wl_1 (FGS1) = μ̂_1d
# wl_2..wl_283 = μ̂_1d + Δ̂ (for 282 AIRS wavelengths)
y_pred_full = np.column_stack([mu_1d, mu_1d[:,None] + mu_2d_282])   # (N,283)

# σ_1 (FGS) = σ̂_μ ;  σ_AIRS = sqrt(σ̂_μ^2 + σ̂_Δ^2)
sigma_full  = np.column_stack([sig_1d, np.sqrt(sig_1d[:,None]**2 + sig_2d_282**2)])  # (N,283)

# ---- Per-channel σ floor → global scale c → clip  ----
# 1) compute per-column std from train.csv (wl_1..wl_283) and take 10% as a floor
train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
train_specs = train_df[[f"wl_{i}" for i in range(1, 284)]].to_numpy(dtype=np.float64)  # (Ntrain, 283)
per_col_std = train_specs.std(axis=0, ddof=1)                                          # (283,)
sigma_floor = np.clip(0.1 * per_col_std, 8e-4, None)                                   # floor >= 8e-4

# 2) global σ multiplier (RE-TUNE THIS AFTER ALIGNMENT; ~1.2–1.5 worked in your sweep)
c_aligned = float(globals().get("calibration_factor_c_aligned", 1.3))

sigma_full = np.maximum(sigma_full, sigma_floor[None, :]) * c_aligned
sigma_full = np.clip(sigma_full, 8e-4, 1e-2)  # final safety clip

# ---- Build DataFrame in exact Kaggle order ----
wl_cols = [f"wl_{i}"    for i in range(1, 284)]   # wl_1 = FGS1; wl_2..wl_283 = AIRS (39..320)
sg_cols = [f"sigma_{i}" for i in range(1, 284)]

df_wl = pd.DataFrame(y_pred_full, columns=wl_cols)
df_sg = pd.DataFrame(sigma_full, columns=sg_cols)
df    = pd.concat([df_wl, df_sg], axis=1)
df.insert(0, "planet_id", all_pid)

# ---- Strict checks ----
assert df.shape[1] == 567, f"Expect 567 columns, got {df.shape[1]}"
assert list(df.columns[:1]) == ["planet_id"]
assert df.columns[1:284].tolist() == wl_cols
assert df.columns[284:].tolist()   == sg_cols
vals = df.iloc[:, 1:].to_numpy()
assert np.isfinite(vals).all(), "Found NaN/Inf"
sig_vals = df.filter(like="sigma_").to_numpy()
assert (sig_vals > 0).all(), "All σ must be positive"
assert df["planet_id"].nunique() == len(df), "Duplicate planet_id rows"

# ---- Write file ----
out_path = "/kaggle/working/submission.csv"  # or your local path
df.to_csv(out_path, index=False)
print("Saved:", out_path, "shape:", df.shape)
print("NOTE: In the Kaggle submit dialog, select 'submission.csv' from Output files.")

