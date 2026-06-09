import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft
from scipy.stats import skew, kurtosis

# =========================
# Basic utils
# =========================

FS = 200          # sampling rate
WIN_SEC = 50
WIN_SAMPLES = FS * WIN_SEC

SAVE_DIR = "./figures"
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_eeg(df, title="", max_ch=20):
    fig, axs = plt.subplots(max_ch, 1, figsize=(25, 15), sharex=True)

    for i, ax in enumerate(axs):
        ax.plot(df.iloc[:, i], color="black", linewidth=0.6)
        ax.set_ylabel(df.columns[i], rotation=0, labelpad=25)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[:].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{title.replace(' ', '_')}.png", dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# Wavelet denoising
# =========================

def maddest(d):
    return np.mean(np.abs(d - np.mean(d)))

def visualize_denoise(df, wavelet="db8", level=1, mode="per"):
    for ch in df.columns:
        signal = df[ch].values
        # 小波分解
        coeffs = pywt.wavedec(signal, wavelet, mode=mode)
        sigma = (1 / 0.6745) * maddest(coeffs[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(df)))
        
        # Threshold 前的 coeffs
        coeffs_before = coeffs.copy()
        # Threshold 後
        coeffs_thresholded = [
            pywt.threshold(c, uthresh, mode='hard') if i>0 else c
            for i, c in enumerate(coeffs)
        ]
        reconstructed = pywt.waverec(coeffs_thresholded, wavelet, mode=mode)[:len(signal)]

        n_levels = len(coeffs)
        fig, axs = plt.subplots(n_levels + 1, 1, figsize=(20, 2*(n_levels+1)))
        fig.suptitle(f"Channel: {ch}", fontsize=16)

        for i in range(n_levels):
            axs[i].plot(coeffs_before[i], color='gray', alpha=0.5, label='original coeff')
            if i > 0:
                axs[i].plot(coeffs_thresholded[i], color='red', alpha=0.8, label='thresholded coeff')
            axs[i].set_ylabel(f"c{i}")
            axs[i].legend(loc='upper right')

        axs[-1].plot(signal, color='gray', alpha=0.5, label='original signal')
        axs[-1].plot(reconstructed, color='blue', alpha=0.8, label='denoised signal')
        axs[-1].set_ylabel("signal")
        axs[-1].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/denoise_{ch}.png", dpi=200, bbox_inches="tight")
        plt.show()

def denoise(df, wavelet="db8", level=1, mode="per"):
    out = {}

    for ch in df.columns:
        print(f'Processing channel {ch}')
        coeffs = pywt.wavedec(df[ch], wavelet, mode=mode)
        sigma = (1 / 0.6745) * maddest(coeffs[-level])
        print("len", len(df))
        uthresh = sigma * np.sqrt(2 * np.log(len(df)))

        coeffs[1:] = [
            pywt.threshold(c, uthresh, mode="hard")
            for c in coeffs[1:]
        ]

        out[ch] = pywt.waverec(coeffs, wavelet, mode=mode)[:len(df)]

    return pd.DataFrame(out)


# =========================
# Feature functions
# =========================

def energy(x):
    return np.sum(x ** 2)


def shannon_entropy(p):
    p = p / (np.sum(p) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))


def basic_stats(x):
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "skew": skew(x),
        "kurtosis": kurtosis(x),
        "rms": np.sqrt(np.mean(x ** 2))
    }


# =========================
# WPD features
# =========================

def wpd_features(x, wavelet="db4", level=4, subband_ratio=(0.5,1.0)):
    wp = pywt.WaveletPacket(x, wavelet, mode="per", maxlevel=level)
    nodes = wp.get_level(level, order="freq") #order according to frequency
    energies = np.array([energy(n.data) for n in nodes])
    rel_energy = energies / (energies.sum() + 1e-12)
    features = {
        "WPD_total_energy": energies.sum(),
        "WPD_shannon_entropy": shannon_entropy(rel_energy),
        "WPD_log_energy_entropy": np.sum(np.log(energies + 1e-12)),
    }
    # RSWE 
    for i, re in enumerate(rel_energy):
        features[f"WPD_p_{i}"] = re 
    # -----------------------------
    # RSWE entropy (only for middel/high frequency sub-band)
    # -----------------------------
    n_nodes = len(nodes)
    start_idx = int(n_nodes * subband_ratio[0])
    end_idx = int(n_nodes * subband_ratio[1])
    selected_rel_energy = rel_energy[start_idx:end_idx]
    selected_rel_energy /= selected_rel_energy.sum() + 1e-12  # normalize
    RSWE_entropy = -np.sum(selected_rel_energy * np.log(selected_rel_energy + 1e-12))
    features["WPD_RSWE_entropy_mid_high"] = RSWE_entropy
    
    return features, energies


# =========================
# DWT + STFT features
# =========================

def plot_dwt_spectrum(x, wavelet="db4", level=5, FS=200):
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="per")
    
    plt.figure(figsize=(14, 2.5 * (level + 1)))

    # Approximation
    A = coeffs[0]
    N = len(A)
    fft_A = np.abs(np.fft.rfft(A))**2
    f = np.linspace(0, FS / (2**(level+1)), len(fft_A))

    plt.subplot(level + 1, 1, 1)
    plt.plot(f, fft_A)
    plt.title(f"A{level}: 0–{FS/(2**(level+1)):.2f} Hz")
    plt.ylabel("Power")

    # Details
    for i, D in enumerate(coeffs[1:], start=1):
        band_low = FS / (2**(level - i + 2))
        band_high = FS / (2**(level - i + 1))

        fft_D = np.abs(np.fft.rfft(D))**2
        f = np.linspace(band_low, band_high, len(fft_D))

        plt.subplot(level + 1, 1, i + 1)
        plt.plot(f, fft_D)
        plt.title(f"D{level - i + 1}: {band_low:.2f}–{band_high:.2f} Hz")
        plt.ylabel("Power")

    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/dwt_fft.png", dpi=200, bbox_inches="tight")
    plt.show()

# =========================
# Main (single case)
# =========================

#if __name__ == "__main__":

# ---- Load label row
df_label = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
row = df_label.sample(n=1, random_state=60).iloc[0]
print("Selected row:\n", row, "\n")

# ---- Load EEG
eeg = pd.read_parquet(
    f"/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/{row['eeg_id']}.parquet"
)
sp = pd.read_parquet(
    f"/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/{row['spectrogram_id']}.parquet"   
)

start = int(row["eeg_label_offset_seconds"]) * FS
eeg = eeg.iloc[start:start + WIN_SAMPLES]


print("EEG shape:", eeg.shape)
print("Spectrogram shape:", sp.shape)
# ---- Plot raw
plot_eeg(eeg, title="Raw EEG")

# ---- Denoise
visualize_denoise(eeg) # For visualization
eeg_denoised = denoise(eeg, wavelet="db8")
plot_eeg(eeg_denoised, title="Denoised EEG")

eeg_denoised = eeg
# ---- Single channel analysis
#eeg_denoised = eeg
ch_name = eeg_denoised.columns[5]
signal = eeg_denoised[ch_name].values

print(f"\nUsing channel: {ch_name}")

# ---- WPD
wpd_feat, wpd_energy = wpd_features(signal)
print("\nWPD features:")
for k, v in wpd_feat.items():
    print(k, ":", v)

plt.figure(figsize=(8, 3))
plt.bar(range(len(wpd_energy)), wpd_energy)
plt.title("WPD Energy Distribution")
plt.xlabel("Subband")
plt.ylabel("Energy")
plt.show()

# ---- DWT + FFT
plot_dwt_spectrum(signal)
print("\nPipeline finished ✔")



import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Config
# =========================
FS = 200
WIN_SEC = 50
WIN_SAMPLES = FS * WIN_SEC
EEG_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"

# =========================
# Basic utils
# =========================
def energy(x):
    return np.sum(x ** 2)

def shannon_entropy(p):
    p = p / (np.sum(p) + 1e-12)
    return -np.sum(p * np.log(p + 1e-12))

def basic_stats(x):
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "skew": skew(x),
        "kurtosis": kurtosis(x),
        "rms": np.sqrt(np.mean(x ** 2))
    }

# =========================
# Wavelet + WPD features
# =========================
def wpd_features(x, wavelet="db4", level=3, subband_ratio=(0.5,1.0)):
    wp = pywt.WaveletPacket(x, wavelet, mode="per", maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    energies = np.array([energy(n.data) for n in nodes])
    rel_energy = energies / (energies.sum() + 1e-12)

    features = {
        "WPD_total_energy": energies.sum(),
        "WPD_shannon_entropy": shannon_entropy(rel_energy),
        "WPD_log_energy_entropy": np.sum(np.log(energies + 1e-12)),
    }

    # RSWE per subband
    for i, re in enumerate(rel_energy):
        features[f"WPD_p_{i}"] = re
    for i, n in enumerate(nodes):
        features[f"WPD_sig_{i}"] = n.data
    
    # RSWE entropy (中高頻 subband)
    n_nodes = len(nodes)
    start_idx = int(n_nodes * subband_ratio[0])
    end_idx = int(n_nodes * subband_ratio[1])
    selected_rel_energy = rel_energy[start_idx:end_idx]
    selected_rel_energy /= selected_rel_energy.sum() + 1e-12
    features["WPD_RSWE_entropy_mid_high"] = -np.sum(selected_rel_energy * np.log(selected_rel_energy + 1e-12))
    
    return features

def dwt_basic_features(x, wavelet="db4", level=5):
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="per")
    features = {}
    for i, coef in enumerate(coeffs):
        e = energy(coef)
        stats = basic_stats(coef)
        prefix = "A" if i==0 else f"D{i}"
        features[f"{prefix}_coef"] = coef
        features[f"{prefix}_energy"] = e
        for k,v in stats.items():
            features[f"{prefix}_{k}"] = v
    return features

def preprocess_eeg(eeg):
    eeg = eeg.copy()
    eeg = eeg.interpolate(limit_direction="both")
    eeg = eeg.fillna(0)
    return eeg

def denoise(df, wavelet="db8", level=1, mode="per"):
    out = {}

    for ch in df.columns:
        coeffs = pywt.wavedec(df[ch], wavelet, mode=mode)
        sigma = (1 / 0.6745) * maddest(coeffs[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(df)))

        coeffs[1:] = [
            pywt.threshold(c, uthresh, mode="hard")
            for c in coeffs[1:]
        ]

        out[ch] = pywt.waverec(coeffs, wavelet, mode=mode)[:len(df)]

    return pd.DataFrame(out)

def get_scalar_feature_columns(X_df):
    scalar_cols = []
    for col in X_df.columns:
        sample_val = X_df[col].iloc[0]
        if np.isscalar(sample_val):
            scalar_cols.append(col)
    return scalar_cols

def plot_feature_scatter(X, y, f1, f2, figsize=(6,5)):
    df_plot = pd.DataFrame({
        f1: X[f1],
        f2: X[f2],
        "label": y.values
    })

    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=df_plot,
        x=f1,
        y=f2,
        hue="label",
        palette="tab10",
        alpha=0.7
    )
    plt.title(f"{f1} vs {f2}")
    plt.tight_layout()
    plt.show()

def plot_feature_vs_class(X, y, feature, figsize=(6,4)):
    df_plot = pd.DataFrame({
        feature: X[feature],
        "label": y.values
    })

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=df_plot,
        x="label",
        y=feature,
        inner="quartile"
    )
    plt.title(f"{feature} vs Class")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# =========================
# Load dataset & extract features
# =========================
def prepare_dataset(df_label, eeg_dir=EEG_DIR, channels=None):
    """
    channels: list of channels to extract features, None = all
    """
    X = []
    y = []
    
    for _, row in tqdm(df_label.iterrows(), total=len(df_label)):
        eeg_path = os.path.join(eeg_dir, f"{row['eeg_id']}.parquet")
        if not os.path.exists(eeg_path):
            continue
        eeg = pd.read_parquet(eeg_path)
        start = int(row['eeg_label_offset_seconds'])*FS
        eeg = eeg.iloc[start:start+WIN_SAMPLES]
        eeg = preprocess_eeg(eeg)
        # denoise
        eeg = denoise(eeg, wavelet="db8")
        
        if channels is None:
            channels_use = eeg.columns.tolist()
        else:
            channels_use = [ch for ch in channels if ch in eeg.columns]
        
        feat = {}
        
        for ch in channels_use:
            sig = eeg[ch].values
            #feat.update({f"{ch}_sig":sig})
            # WPD features
            feat.update({f"{ch}_{k}":v for k,v in wpd_features(sig).items()})
            # DWT basic features
            feat.update({f"{ch}_{k}":v for k,v in dwt_basic_features(sig).items()})
        X.append(feat)
        y.append(row['expert_consensus'])
        
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name='label')
    return X_df, y_series

# =========================
# Train/validation split
# =========================
df_label = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")

# Only use 500 samples for fast run
df_label = df_label.sample(n=500, random_state=42)

channels = ['F3', 'F4', 'C3', 'C4', 'Cz', 'Pz']  # or None -> Choose all electrodes

X, y = prepare_dataset(df_label, channels=channels)

# Stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Class distribution (train):\n", y_train.value_counts())
print("Class distribution (val):\n", y_val.value_counts())
print(X_train.iloc[0])
print(y_train.iloc[0])

scalar_cols = get_scalar_feature_columns(X_train)
print("Number of scalar features:", len(scalar_cols))
print(scalar_cols[:10])

plot_feature_vs_class(X_train, y_train, "Cz_WPD_shannon_entropy")
plot_feature_vs_class(X_train, y_train, "Cz_WPD_RSWE_entropy_mid_high")
plot_feature_vs_class(X_train, y_train, "Cz_D3_energy")
plot_feature_scatter(
    X_train, y_train,
    "Cz_WPD_RSWE_entropy_mid_high",
    "Cz_WPD_shannon_entropy"
)
plot_feature_scatter(
    X_train, y_train,
    "Cz_D3_energy",
    "Cz_D4_energy"
)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)

y_train_enc = le.transform(y_train)
y_val_enc   = le.transform(y_val)

print(le.classes_)

def build_wpd_tensor(X_df, channels, level=3):
    """
    Returns:
        X_tensor: shape (N, C, H, W)
    """
    n_samples = len(X_df)
    n_bands = 2 ** level
    n_channels = len(channels)

    # 用第一筆 sample 決定時間長度
    first_key = f"{channels[0]}_WPD_sig_0"
    W = len(X_df.iloc[0][first_key])

    X_tensor = np.zeros((n_samples, n_bands, n_channels, W), dtype=np.float32)

    for i in tqdm(range(n_samples)):
        for h, ch in enumerate(channels):
            for c in range(n_bands):
                key = f"{ch}_WPD_sig_{c}"
                X_tensor[i, c, h, :] = X_df.iloc[i][key]

    return X_tensor

X_train = build_wpd_tensor(X_train, channels=channels, level=3)
X_val   = build_wpd_tensor(X_val,   channels=channels, level=3)

print(X_train.shape)
n_classes = y_train.nunique()
print(n_classes)

def one_hot(y, num_classes):
    y = y.values if isinstance(y, pd.Series) else y
    return np.eye(num_classes)[y]

print(y_train_enc)
y_train_oh = one_hot(y_train_enc, n_classes)
y_val_oh   = one_hot(y_val_enc,   n_classes)

print(y_train_oh.shape)

mean = X_train.mean(axis=(0,2,3), keepdims=True)
std  = X_train.std(axis=(0,2,3), keepdims=True) + 1e-6

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std

print(X_train[0])
print(y_train[0])


efficient=False
if(efficient):
    #Efficient RAM friendly dataset preparation
    import os
    import numpy as np
    import pandas as pd
    import pywt
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    FS = 200
    WIN_SEC = 50
    WIN_SAMPLES = FS * WIN_SEC
    EEG_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    
    channels = ['F3', 'F4', 'C3', 'C4', 'Cz', 'Pz']
    wavelet = "db4"
    level = 4
    n_bands = 2 ** level
    
    def preprocess_eeg(eeg):
        eeg = eeg.interpolate(limit_direction="both")
        eeg = eeg.fillna(0)
        return eeg
    
    def build_wpd_tensor_from_labels(
        df_label,
        channels,
        eeg_dir=EEG_DIR,
        level=3
    ):
        """
        Returns:
            X: (N, C_freq, H_ch, W_time)
            y: (N,)
        """
        n_samples = len(df_label)
        n_channels = len(channels)
        n_bands = 2 ** level
    
        # 先讀一筆決定 W
        for _, row in df_label.iterrows():
            eeg_path = os.path.join(eeg_dir, f"{row['eeg_id']}.parquet")
            if os.path.exists(eeg_path):
                eeg = pd.read_parquet(eeg_path)
                start = int(row['eeg_label_offset_seconds']) * FS
                eeg = eeg.iloc[start:start + WIN_SAMPLES]
                eeg = preprocess_eeg(eeg)
                wp = pywt.WaveletPacket(
                    eeg[channels[0]].values,
                    wavelet,
                    mode="per",
                    maxlevel=level
                )
                W = len(wp.get_level(level, order="freq")[0].data)
                break
    
        X = np.zeros(
            (n_samples, n_bands, n_channels, W),
            dtype=np.float32
        )
        y = np.zeros(n_samples, dtype=np.int64)
    
        valid_idx = 0
    
        for _, row in tqdm(df_label.iterrows(), total=n_samples):
            eeg_path = os.path.join(eeg_dir, f"{row['eeg_id']}.parquet")
            if not os.path.exists(eeg_path):
                continue
    
            eeg = pd.read_parquet(eeg_path)
            start = int(row['eeg_label_offset_seconds']) * FS
            eeg = eeg.iloc[start:start + WIN_SAMPLES]
            eeg = preprocess_eeg(eeg)
    
            for h, ch in enumerate(channels):
                sig = eeg[ch].values
                wp = pywt.WaveletPacket(
                    sig, wavelet, mode="per", maxlevel=level
                )
                nodes = wp.get_level(level, order="freq")
                for c, node in enumerate(nodes):
                    X[valid_idx, c, h, :] = node.data
    
            y[valid_idx] = row["expert_consensus"]
            valid_idx += 1
    
        return X[:valid_idx], y[:valid_idx]
    
    df_label = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    
    df_label = df_label.sample(n=10000, random_state=42)
    
    train_df, val_df = train_test_split(
        df_label,
        test_size=0.2,
        random_state=42,
        stratify=df_label["expert_consensus"]
    )
    X_train, y_train = build_wpd_tensor_from_labels(
        train_df, channels=channels, level=level
    )
    
    X_val, y_val = build_wpd_tensor_from_labels(
        val_df, channels=channels, level=level
    )
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    
    n_classes = len(le.classes_)
    print(le.classes_)
    mean = X_train.mean(axis=(0,2,3), keepdims=True)
    std  = X_train.std(axis=(0,2,3), keepdims=True) + 1e-6
    
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    print(X_train.shape)  # (N, n_bands, n_channels, W)
    print(X_val.shape)
    print(y_train_enc.shape)
    print(y_val_enc.shape)
    y_train_idx = y_train_enc
    y_val_idx = y_val_enc


import torch
from torch.utils.data import Dataset

class EEGTensorDataset(Dataset):
    def __init__(self, X, y):
        # numpy -> torch
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)

        # 如果 y 是 one-hot，就轉成 float
        if self.y.ndim == 2:
            self.y = self.y.float()
        else:
            self.y = self.y.long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



from torch.utils.data import DataLoader

batch_size = 16        # EEG 很吃記憶體，16 是穩的
num_workers = 2        # Kaggle 通常 2~4 最穩

classes = sorted(y_train.unique())
print(classes)
label2idx = {c: i for i, c in enumerate(classes)}
idx2label = {i: c for c, i in label2idx.items()}
y_train_idx = y_train.map(label2idx).values
y_val_idx   = y_val.map(label2idx).values
print(y_train.iloc[0], "->", y_train_idx[0])

print(y_train_idx)
train_ds = EEGTensorDataset(X_train, y_train_idx)
val_ds   = EEGTensorDataset(X_val,   y_val_idx)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# Model architecture
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

class EEGClassifier(nn.Module):
    def __init__(self, n_freq, n_ch, n_classes, W_time):
        super().__init__()

        # ===== Temporal convolution =====
        self.temporal = nn.Sequential(
            nn.Conv2d(
                in_channels=n_freq,
                out_channels=32,
                kernel_size=(1, 25),
                padding=(0, 12)
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )

        # ===== Spatial (channel) convolution =====
        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(n_ch, 1),
                groups=1
            ),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        # ===== Pooling + regularization =====
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.5)
        )

        # ===== Classifier =====
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (W_time // 4), 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (B, C_freq, H_ch, W_time)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler_state"])
    start_epoch = state["epoch"] + 1
    return model, optimizer, scheduler, start_epoch

def run_epoch(model, loader, train=True):
    model.train() if train else model.eval()

    total_loss = 0
    all_preds, all_targets = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X.size(0)

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc

# =========================
# Checkpoint config
# =========================
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEGClassifier(
    n_freq=X_train.shape[1],
    n_ch=X_train.shape[2],
    n_classes=n_classes,
    W_time=X_train.shape[3]
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

# =========================
# Training loop with checkpoint
# =========================
best_val_acc = 0.0
epochs = 100

for epoch in range(1, epochs + 1):
    train_loss, train_acc = run_epoch(model, train_loader, train=True)
    val_loss,   val_acc   = run_epoch(model, val_loader,   train=False)

    # Scheduler step
    scheduler.step(val_acc)

    print(
        f"[{epoch:02d}] "
        f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
    )

    # save checkpoint every epoch
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pt")
    save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)

    # save best model separately
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, best_model_path)
        print(f"  -> New best model saved at epoch {epoch} with val_acc {val_acc:.4f}")


import matplotlib.pyplot as plt

# Data extracted from logs
epochs = list(range(1, 101))

train_loss = [
1.9228,1.7640,1.7403,1.7290,1.6876,1.6700,1.6637,1.6386,1.6086,1.5706,
1.5355,1.4916,1.4606,1.4262,1.3211,1.2649,1.2177,1.1821,1.1436,1.1101,
1.0682,1.0166,0.9610,0.9593,0.9368,0.9237,0.9036,0.8793,0.8617,0.8591,
0.8386,0.8318,0.7829,0.7776,0.7711,0.7549,0.7468,0.7296,0.7237,0.7133,
0.7211,0.7296,0.7069,0.7141,0.6872,0.7094,0.7073,0.7006,0.6920,0.6974,
0.7083,0.7047,0.6930,0.6951,0.6899,0.6855,0.7004,0.6944,0.6878,0.7056,
0.6955,0.6904,0.6989,0.6958,0.6975,0.6926,0.6927,0.6897,0.6852,0.6919,
0.6914,0.6921,0.6917,0.6863,0.6942,0.7001,0.6899,0.6977,0.6846,0.6816,
0.6918,0.6849,0.6897,0.6869,0.6825,0.6954,0.6965,0.6918,0.6721,0.6874,
0.7148,0.6908,0.6846,0.6980,0.6991,0.6880,0.6904,0.6928,0.6905,0.6884
]

val_loss = [
1.7915,1.7934,1.8117,1.8031,1.8310,1.9200,1.7590,1.8456,1.7988,1.7966,
1.7892,1.8563,1.8348,1.8722,1.8153,1.8377,1.7980,2.0231,1.8836,1.9677,
1.8117,1.9504,1.9201,1.9386,1.9163,1.9971,1.8941,1.9224,1.9432,2.0163,
1.8921,2.0053,1.9272,1.9614,1.8844,2.0266,1.9555,1.9332,1.9198,1.9674,
1.9896,1.9988,1.9775,1.9828,1.9972,1.9414,1.8942,2.0668,1.9500,1.9436,
1.9108,2.0203,2.0462,1.9414,1.9585,2.0709,2.0175,1.9717,1.9623,2.0526,
2.0447,2.0873,1.9473,2.0305,1.9631,1.9650,2.0515,1.9071,2.0532,1.9957,
1.9288,2.0210,1.9319,2.0840,1.9919,1.9420,1.9145,1.8974,1.9520,1.9180,
2.0324,1.9604,1.9219,1.9701,1.9298,2.1080,2.0295,2.1302,2.0586,2.0240,
1.9118,1.8865,2.0362,1.9341,1.9887,2.1556,2.0294,1.9444,1.9619,1.9693
]

train_acc = [
0.2080,0.2401,0.2575,0.2715,0.2924,0.2965,0.3098,0.3207,0.3376,0.3543,
0.3787,0.4002,0.4220,0.4365,0.4784,0.5127,0.5255,0.5463,0.5617,0.5761,
0.5931,0.6170,0.6385,0.6401,0.6484,0.6498,0.6600,0.6698,0.6793,0.6851,
0.6834,0.6901,0.7080,0.7092,0.7173,0.7212,0.7229,0.7337,0.7364,0.7380,
0.7351,0.7312,0.7418,0.7358,0.7494,0.7362,0.7361,0.7432,0.7462,0.7408,
0.7392,0.7393,0.7459,0.7428,0.7436,0.7505,0.7430,0.7481,0.7469,0.7448,
0.7435,0.7448,0.7394,0.7402,0.7418,0.7432,0.7414,0.7419,0.7498,0.7416,
0.7473,0.7443,0.7460,0.7472,0.7457,0.7409,0.7441,0.7411,0.7468,0.7458,
0.7446,0.7492,0.7457,0.7452,0.7490,0.7462,0.7471,0.7451,0.7570,0.7441,
0.7350,0.7484,0.7515,0.7399,0.7451,0.7458,0.7481,0.7413,0.7435,0.7497
]

val_acc = [
0.1917,0.1867,0.2077,0.1930,0.1803,0.1913,0.2287,0.2087,0.2107,0.2390,
0.2193,0.2187,0.2027,0.2390,0.2240,0.2200,0.2583,0.2040,0.2553,0.2280,
0.1973,0.2553,0.2417,0.2400,0.2657,0.2110,0.2477,0.2673,0.2513,0.2433,
0.2280,0.2470,0.2110,0.2383,0.2410,0.2487,0.2287,0.2243,0.2293,0.2343,
0.2500,0.2650,0.2270,0.2277,0.2330,0.2410,0.1943,0.2620,0.2287,0.2240,
0.2197,0.2367,0.2697,0.2237,0.2307,0.2703,0.2463,0.2153,0.2433,0.2777,
0.2500,0.2683,0.2323,0.2487,0.2387,0.2283,0.2563,0.2150,0.2527,0.2390,
0.2030,0.2443,0.2370,0.2640,0.2207,0.2170,0.2160,0.2247,0.2340,0.2223,
0.2497,0.2167,0.2153,0.2243,0.2157,0.2837,0.2507,0.2813,0.2610,0.2417,
0.2050,0.2130,0.2527,0.2310,0.2267,0.2853,0.2393,0.2220,0.2330,0.2280
]

# Plot Loss
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()




