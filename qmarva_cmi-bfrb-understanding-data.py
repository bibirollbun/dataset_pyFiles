import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from scipy import stats
from collections import Counter
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


kaggle_root = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
train_csv = "train.csv"
test_csv = "test.csv"
test_demo = "test_demographics.csv"
sequence_percentile = 85


class GestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, augment_prob: float = 0.3):
        self.X = X
        self.y = y
        self.augment = augment
        self.augment_prob = augment_prob
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        
        if self.augment and random.random() < self.augment_prob:
            x = self._augment(x)
            
        x = x.transpose(0, 1)
        return x, y
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.3:
            noise = torch.randn_like(x) * 0.003
            x = x + noise
            
        if random.random() < 0.2:
            shift = random.randint(-2, 2)
            x = torch.roll(x, shift, dims=0)
            
        if random.random() < 0.2:
            scale = random.uniform(0.98, 1.02)
            x = x * scale
            
        return x


def preprocess_sequence(seq_df: pd.DataFrame, imu_cols: List[str]) -> np.ndarray:
    data = seq_df[imu_cols].copy()
    data = data.interpolate(method='linear').ffill().bfill().fillna(0)
    
    for col in imu_cols:
        q1, q99 = data[col].quantile([0.01, 0.99])
        data[col] = data[col].clip(q1, q99)
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data.astype(np.float32)


def pad_to_len(arr: np.ndarray, maxlen: int, dtype=np.float32) -> np.ndarray:
    t, c = arr.shape
    if t >= maxlen:
        return arr[:maxlen].astype(dtype, copy=False)
    out = np.zeros((maxlen, c), dtype=dtype)
    out[:t] = arr
    return out


def load_and_process_data(**args):
    print("Loading and processing data...")
    
    train_path = os.path.join(kaggle_root, train_csv)
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df):,} rows.")
    
    le = LabelEncoder()
    df["gesture"] = le.fit_transform(df["gesture"].astype(str))
    np.save("gesture_classes.npy", le.classes_)
    
    imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    
    print("Building sequences...")
    seq_groups = df.groupby("sequence_id")
    X_list, y_list, lengths = [], [], []
    
    for seq_id, seq_df in tqdm(seq_groups, desc="Processing sequences"):
        X_seq = preprocess_sequence(seq_df, imu_cols)
        X_list.append(X_seq)
        lengths.append(X_seq.shape[0])
        y_list.append(seq_df["gesture"].iloc[0])
    
    pad_len = int(np.percentile(lengths, sequence_percentile))
    print(f"Padding/truncating to length: {pad_len}")
    print(f"Length statistics: min={np.min(lengths)}, max={np.max(lengths)}, mean={np.mean(lengths):.1f}")
    np.save("sequence_maxlen.npy", pad_len)
    
    X = np.stack([pad_to_len(arr, pad_len) for arr in X_list])
    y = np.array(y_list, dtype=np.int64)
    
    print(f"Final data shape: X={X.shape}, y={y.shape}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, len(le.classes_), pad_len, le.classes_


def analyze_class_distribution(y, class_names=None, title="Class Distribution"):
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    
    class_counts = Counter(y)
    classes = sorted(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    
    if class_names is not None:
        labels = [class_names[c] for c in classes]
    else:
        labels = [f'Class {c}' for c in classes]
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    wedges, texts, autotexts = axes[0].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title(f'{title} - Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = axes[1].bar(range(len(classes)), counts, color=colors)
    axes[1].set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Imbalance statistics
    total = sum(counts)
    percentages = [c/total*100 for c in counts]
    imbalance_ratio = max(counts) / min(counts)
    
    plt.tight_layout()
    plt.show()
    
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print(f"\033[1;37m{title.upper()} ANALYSIS\033[0m")
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print(f"Total samples: \033[1;33m{total:,}\033[0m")
    print(f"Number of classes: \033[1;33m{len(classes)}\033[0m")
    print(f"Class imbalance ratio: \033[1;33m{imbalance_ratio:.2f}\033[0m (max/min)")
    
    if imbalance_ratio > 2:
        print("\033[1;31mHIGH CLASS IMBALANCE DETECTED!\033[0m")
        print("\033[0;31m   This could lead to overfitting on majority classes\033[0m")
    elif imbalance_ratio > 1.5:
        print("\033[1;33mModerate class imbalance detected\033[0m")
    else:
        print("\033[1;32mClasses are relatively balanced\033[0m")
    
    print("\n\033[1;37mClass breakdown:\033[0m")
    for i, (cls, count, pct) in enumerate(zip(classes, counts, percentages)):
        label = labels[i] if class_names else f'Class {cls}'
        print(f"  \033[1;34m{label}:\033[0m {count:,} samples (\033[0;33m{pct:.1f}%\033[0m)")
    
    return {
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio,
        'total_samples': total,
        'num_classes': len(classes)
    }


def create_train_val_pie_chart(total_train, total_val):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    sizes = [total_train, total_val]
    labels = ['Train', 'Validation']
    colors = ['#FF6B6B', '#4ECDC4']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Train/Val Split', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_class_distribution_chart(y_data, classes, class_names, title, color):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.6))
    counts = Counter(y_data)
    class_counts = [counts[c] for c in classes]
    
    if class_names is not None:
        class_labels = [class_names[c] for c in classes]
    else:
        class_labels = [f'Class {c}' for c in classes]
    
    bars = ax.bar(range(len(classes)), class_counts, color=color, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return class_counts


def calculate_stratification_quality(y_train, y_val, classes, val_split):
    total_train = len(y_train)
    total_val = len(y_val)
    
    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    
    train_class_counts = [train_counts[c] for c in classes]
    val_class_counts = [val_counts[c] for c in classes]
    
    train_proportions = np.array(train_class_counts) / total_train * 100
    val_proportions = np.array(val_class_counts) / total_val * 100
    
    proportion_diffs = []
    for i in range(len(classes)):
        train_pct = train_proportions[i]
        val_pct = val_proportions[i]
        diff = abs(train_pct - val_pct)
        proportion_diffs.append(diff)
    
    max_diff = max(proportion_diffs)
    mean_diff = np.mean(proportion_diffs)
    
    return max_diff, mean_diff


def print_split_analysis_summary(total_train, total_val, val_split, max_diff, mean_diff):
    print("\033[1;36m" + "=" * 70 + "\033[0m")
    print("\033[1;37mTRAIN/VALIDATION SPLIT ANALYSIS\033[0m")
    print("\033[1;36m" + "=" * 70 + "\033[0m")
    print(f"Train set: \033[1;33m{total_train:,}\033[0m samples (\033[0;33m{(1-val_split)*100:.1f}%\033[0m)")
    print(f"Validation set: \033[1;33m{total_val:,}\033[0m samples (\033[0;33m{val_split*100:.1f}%\033[0m)")
    print(f"Maximum class proportion difference: \033[1;33m{max_diff:.2f}%\033[0m")
    print(f"Average class proportion difference: \033[1;33m{mean_diff:.2f}%\033[0m")

    if max_diff > 5.0:
        print("\033[1;31mWARNING: Poor stratification detected!\033[0m")
        print("\033[0;31m   Large differences in class proportions between train/val\033[0m")
        print("\033[0;31m   This could contribute to overfitting issues\033[0m")

def analyze_train_val_split(X, y, val_split=0.2, random_state=42, class_names=None):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )
    
    total_train = len(y_train)
    total_val = len(y_val)
    
    create_train_val_pie_chart(total_train, total_val)
    
    train_counts = Counter(y_train)
    classes = sorted(train_counts.keys())
    
    create_class_distribution_chart(y_train, classes, class_names, 
                                  'Class Distribution - Train Set', '#FF6B6B')
    create_class_distribution_chart(y_val, classes, class_names, 
                                  'Class Distribution - Validation Set', '#4ECDC4')
    
    max_diff, mean_diff = calculate_stratification_quality(y_train, y_val, classes, val_split)
    print_split_analysis_summary(total_train, total_val, val_split, max_diff, mean_diff)
    
    return {
        'train_data': (X_train, y_train),
        'val_data': (X_val, y_val),
        'stratification_quality': {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'quality': 'excellent' if max_diff < 1.0 else 'good' if max_diff < 2.0 else 'fair' if max_diff < 5.0 else 'poor'
        }
    }



def analyze_train_val_split(X, y, val_split=0.2, random_state=42, class_names=None):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )
    
    total_train = len(y_train)
    total_val = len(y_val)
    
    create_train_val_pie_chart(total_train, total_val)
    
    train_counts = Counter(y_train)
    classes = sorted(train_counts.keys())
    
    create_class_distribution_chart(y_train, classes, class_names, 
                                  'Class Distribution - Train Set', '#FF6B6B')
    create_class_distribution_chart(y_val, classes, class_names, 
                                  'Class Distribution - Validation Set', '#4ECDC4')
    
    max_diff, mean_diff = calculate_stratification_quality(y_train, y_val, classes, val_split)
    print_split_analysis_summary(total_train, total_val, val_split, max_diff, mean_diff)
    
    return {
        'train_data': (X_train, y_train),
        'val_data': (X_val, y_val),
        'stratification_quality': {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'quality': 'excellent' if max_diff < 1.0 else 'good' if max_diff < 2.0 else 'fair' if max_diff < 5.0 else 'poor'
        }
    }


def calculate_sequence_stats(X, y, classes):
    sequence_stats = {}
    
    for cls in classes:
        cls_mask = (y == cls)
        cls_sequences = X[cls_mask]
        
        effective_lengths = []
        for seq in cls_sequences:
            nonzero_indices = np.where(np.any(seq != 0, axis=1))[0]
            if len(nonzero_indices) > 0:
                effective_lengths.append(nonzero_indices[-1] + 1)
            else:
                effective_lengths.append(0)
        
        sequence_stats[cls] = {
            'count': len(cls_sequences),
            'lengths': effective_lengths,
            'mean_length': np.mean(effective_lengths),
            'std_length': np.std(effective_lengths),
            'min_length': np.min(effective_lengths),
            'max_length': np.max(effective_lengths)
        }
    
    return sequence_stats


def create_sequence_length_boxplot(X, y, classes, class_names):
    sequence_stats = calculate_sequence_stats(X, y, classes)
    
    length_data = []
    length_labels = []
    
    for cls in classes:
        lengths = sequence_stats[cls]['lengths']
        length_data.append(lengths)
        label = class_names[cls] if class_names else f'Class {cls}'
        length_labels.append(label)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.boxplot(length_data, labels=length_labels)
    ax.set_title('Sequence Lengths by Class', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Effective Sequence Length', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    return sequence_stats


def create_signal_variability_boxplot(X, y, classes, class_names):
    class_variabilities = []
    length_labels = []
    
    for cls in classes:
        cls_mask = (y == cls)
        cls_sequences = X[cls_mask]
        
        variabilities = []
        for seq in cls_sequences:
            nonzero_mask = np.any(seq != 0, axis=1)
            if np.any(nonzero_mask):
                active_seq = seq[nonzero_mask]
                variabilities.append(np.std(active_seq.flatten()))
        
        class_variabilities.append(variabilities)
        label = class_names[cls] if class_names else f'Class {cls}'
        length_labels.append(label)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.boxplot(class_variabilities, labels=length_labels)
    ax.set_title('Signal Variability by Class', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def create_signal_magnitude_boxplot(X, y, classes, class_names):
    class_means = []
    length_labels = []
    
    for cls in classes:
        cls_mask = (y == cls)
        cls_sequences = X[cls_mask]
        
        means = []
        for seq in cls_sequences:
            nonzero_mask = np.any(seq != 0, axis=1)
            if np.any(nonzero_mask):
                active_seq = seq[nonzero_mask]
                means.append(np.mean(np.abs(active_seq.flatten())))
        
        class_means.append(means)
        label = class_names[cls] if class_names else f'Class {cls}'
        length_labels.append(label)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.boxplot(class_means, labels=length_labels)
    ax.set_title('Signal Magnitude by Class', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Mean Absolute Value', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def create_sample_sequence_plots(X, y, classes, class_names, sample_sequences):
    n_classes = len(classes)
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if n_classes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    sample_colors = plt.cm.tab10(np.linspace(0, 1, sample_sequences))
    
    for i, cls in enumerate(classes):
        row = i // cols
        col = i % cols
        
        cls_mask = (y == cls)
        cls_sequences = X[cls_mask]
        
        n_samples = min(sample_sequences, len(cls_sequences))
        
        if len(cls_sequences) > 0:
            sample_indices = np.random.choice(len(cls_sequences), n_samples, replace=False)
            
            ax = axes[row, col] if rows > 1 else axes[col]
            
            for j, idx in enumerate(sample_indices):
                seq = cls_sequences[idx]
                nonzero_mask = np.any(seq != 0, axis=1)
                if np.any(nonzero_mask):
                    active_seq = seq[nonzero_mask, 0]
                    color = sample_colors[j % len(sample_colors)]
                    ax.plot(active_seq, alpha=0.8, linewidth=1.5, color=color, label=f'Sample {j+1}')
            
            label = class_names[cls] if class_names else f'Class {cls}'
            ax.set_title(f'Sample Sequences - {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel('Feature Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.text(0.5, 0.5, 'No sequences\navailable', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Class {cls} - No Data', fontsize=12, fontweight='bold')
    
    total_plots = rows * cols
    for i in range(n_classes, total_plots):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def interpolate_sequence_for_profile(seq, common_length, n_features):
    from scipy import interpolate
    
    if len(seq) > 1:
        interp_seq = np.zeros((common_length, n_features))
        for feat_idx in range(n_features):
            x_orig = np.linspace(0, 1, len(seq))
            x_new = np.linspace(0, 1, common_length)
            f = interpolate.interp1d(x_orig, seq[:, feat_idx], 
                                   kind='linear', fill_value='extrapolate')
            interp_seq[:, feat_idx] = f(x_new)
        return interp_seq
    return None


def create_feature_specific_profiles(X, y, classes, class_names, n_features):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import interpolate
    
    n_classes = len(classes)
    
    feature_colors = [
        '#E74C3C',
        '#8E44AD',
        '#3498DB',
        '#2ECC71',
        '#F39C12',
        '#1ABC9C',
        '#E67E22',
        '#9B59B6',
        '#34495E',
        '#16A085',
        '#D35400',
        '#7F8C8D',
        '#C0392B',
        '#27AE60',
        '#2980B9',
        '#8F4A84',
        '#B7950B',
        '#17A2B8'
    ]
    
    def interpolate_sequences(sequences, common_length=100):
        interpolated = []
        for seq in sequences:
            if len(seq) > 1:
                x_orig = np.linspace(0, 1, len(seq))
                x_new = np.linspace(0, 1, common_length)
                f = interpolate.interp1d(x_orig, seq, kind='linear', fill_value='extrapolate')
                interpolated.append(f(x_new))
        return np.array(interpolated) if interpolated else None
    
    all_profiles = {}
    
    for feat_idx in range(n_features):
        print(f"\n\033[1;37mFeature {feat_idx + 1} profiles:\033[0m")
        
        feature_color = feature_colors[feat_idx % len(feature_colors)]
        all_profiles[feat_idx] = {}
        
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        
        if n_classes == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, cls in enumerate(classes):
            row = i // cols
            col = i % cols
            
            cls_mask = (y == cls)
            cls_sequences = X[cls_mask]
            
            if len(cls_sequences) > 0:
                ax = axes[row, col] if rows > 1 else axes[col]
                
                all_sequences = []
                for seq in cls_sequences:
                    nonzero_mask = np.any(seq != 0, axis=1)
                    if np.any(nonzero_mask):
                        active_seq = seq[nonzero_mask, feat_idx]
                        if len(active_seq) > 0:
                            all_sequences.append(active_seq)
                
                if all_sequences:
                    max_length = max([len(seq) for seq in all_sequences])
                    common_length = min(100, max_length)
                    
                    sequences_array = interpolate_sequences(all_sequences, common_length)
                    
                    if sequences_array is not None and len(sequences_array) > 0:
                        mean_profile = np.mean(sequences_array, axis=0)
                        std_profile = np.std(sequences_array, axis=0)
                        
                        all_profiles[feat_idx][cls] = {
                            'mean': mean_profile,
                            'std': std_profile,
                            'length': common_length
                        }
                        
                        x_axis = np.linspace(0, 100, common_length)
                        
                        ax.plot(x_axis, mean_profile, 
                               color=feature_color, linewidth=3, 
                               label=f'Feature {feat_idx + 1} Profile')
                        
                        ax.fill_between(x_axis, 
                                      mean_profile - std_profile,
                                      mean_profile + std_profile,
                                      color=feature_color, alpha=0.2,
                                      label='± 1 Std Dev')
                
                label = class_names[cls] if class_names else f'Class {cls}'
                ax.set_title(f'Feature {feat_idx + 1} Profile - {label}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Normalized Time (%)', fontsize=10)
                ax.set_ylabel(f'Feature {feat_idx + 1} Value', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            else:
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.text(0.5, 0.5, 'No sequences\navailable', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Class {cls} - No Data', fontsize=12, fontweight='bold')
        
        total_plots = rows * cols
        for i in range(n_classes, total_plots):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n\033[1;37mClass-wise feature comparison:\033[0m")
    
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
    
    if n_classes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, cls in enumerate(classes):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        has_data = False
        
        for feat_idx in range(n_features):
            if feat_idx in all_profiles and cls in all_profiles[feat_idx]:
                profile_data = all_profiles[feat_idx][cls]
                feature_color = feature_colors[feat_idx % len(feature_colors)]
                
                mean_profile = profile_data['mean']
                std_profile = profile_data['std']
                common_length = profile_data['length']
                x_axis = np.linspace(0, 100, common_length)
                
                ax.plot(x_axis, mean_profile, 
                       color=feature_color, linewidth=2.5, 
                       label=f'Feature {feat_idx + 1}')
                
                ax.fill_between(x_axis, 
                              mean_profile - std_profile,
                              mean_profile + std_profile,
                              color=feature_color, alpha=0.15)
                
                has_data = True
        
        if has_data:
            label = class_names[cls] if class_names else f'Class {cls}'
            ax.set_title(f'All Features - {label}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Normalized Time (%)', fontsize=11)
            ax.set_ylabel('Feature Values', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
        else:
            ax.text(0.5, 0.5, 'No data\navailable', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            label = class_names[cls] if class_names else f'Class {cls}'
            ax.set_title(f'All Features - {label}', fontsize=14, fontweight='bold')
    
    total_plots = rows * cols
    for i in range(n_classes, total_plots):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_overall_average_profiles(X, y, classes, class_names, n_features):
    n_classes = len(classes)
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if n_classes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, cls in enumerate(classes):
        row = i // cols
        col = i % cols
        
        cls_mask = (y == cls)
        cls_sequences = X[cls_mask]
        
        if len(cls_sequences) > 0:
            ax = axes[row, col] if rows > 1 else axes[col]
            
            max_length = max([len(seq[np.any(seq != 0, axis=1)]) for seq in cls_sequences 
                             if np.any(np.any(seq != 0, axis=1))])
            
            if max_length > 0:
                all_sequences = []
                for seq in cls_sequences:
                    nonzero_mask = np.any(seq != 0, axis=1)
                    if np.any(nonzero_mask):
                        active_seq = seq[nonzero_mask]
                        if len(active_seq) > 0:
                            all_sequences.append(active_seq)
                
                if all_sequences:
                    common_length = min(100, max_length)
                    interpolated_sequences = []
                    
                    for seq in all_sequences:
                        interp_seq = interpolate_sequence_for_profile(seq, common_length, n_features)
                        if interp_seq is not None:
                            interpolated_sequences.append(interp_seq)
                    
                    if interpolated_sequences:
                        sequences_array = np.array(interpolated_sequences)
                        mean_profile = np.mean(sequences_array, axis=0)
                        std_profile = np.std(sequences_array, axis=0)
                        
                        x_axis = np.linspace(0, 100, common_length)
                        
                        overall_mean = np.mean(mean_profile, axis=1)
                        overall_std = np.mean(std_profile, axis=1)
                        
                        ax.plot(x_axis, overall_mean, 
                               color='blue', linewidth=3, 
                               label='Average Profile')
                        
                        ax.fill_between(x_axis, 
                                      overall_mean - overall_std,
                                      overall_mean + overall_std,
                                      color='blue', alpha=0.2,
                                      label='± 1 Std Dev')
            
            label = class_names[cls] if class_names else f'Class {cls}'
            ax.set_title(f'Overall Average Profile - {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Normalized Time (%)', fontsize=10)
            ax.set_ylabel('Average Feature Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.text(0.5, 0.5, 'No sequences\navailable', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Class {cls} - No Data', fontsize=12, fontweight='bold')
    
    total_plots = rows * cols
    for i in range(n_classes, total_plots):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def interpolate_energy_profile(time_percent, energy, common_time):
    from scipy import interpolate
    
    if len(time_percent) > 1:
        f = interpolate.interp1d(time_percent, energy, 
                               kind='linear', fill_value='extrapolate')
        return f(common_time)
    return None


def create_signal_energy_distribution(X, y, classes, class_names):
    n_classes = len(classes)
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if n_classes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, cls in enumerate(classes):
        row = i // cols
        col = i % cols
        
        cls_mask = (y == cls)
        cls_sequences = X[cls_mask]
        
        if len(cls_sequences) > 0:
            ax = axes[row, col] if rows > 1 else axes[col]
            
            energy_distributions = []
            
            for seq in cls_sequences:
                nonzero_mask = np.any(seq != 0, axis=1)
                if np.any(nonzero_mask):
                    active_seq = seq[nonzero_mask]
                    if len(active_seq) > 1:
                        energy = np.sum(active_seq**2, axis=1)
                        time_percent = np.linspace(0, 100, len(energy))
                        energy_distributions.append((time_percent, energy))
            
            if energy_distributions:
                for time_percent, energy in energy_distributions:
                    ax.plot(time_percent, energy, alpha=0.3, color='blue', linewidth=0.5)
                
                common_time = np.linspace(0, 100, 50)
                interpolated_energies = []
                
                for time_percent, energy in energy_distributions:
                    interp_energy = interpolate_energy_profile(time_percent, energy, common_time)
                    if interp_energy is not None:
                        interpolated_energies.append(interp_energy)
                
                if interpolated_energies:
                    mean_energy = np.mean(interpolated_energies, axis=0)
                    std_energy = np.std(interpolated_energies, axis=0)
                    
                    ax.plot(common_time, mean_energy, color='red', linewidth=3, 
                           label='Average Energy', zorder=10)
                    
                    ax.fill_between(common_time, 
                                  mean_energy - std_energy,
                                  mean_energy + std_energy,
                                  color='red', alpha=0.2, zorder=5)
            
            label = class_names[cls] if class_names else f'Class {cls}'
            ax.set_title(f'Signal Energy Distribution - {label}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Normalized Time (%)', fontsize=10)
            ax.set_ylabel('Signal Energy', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        else:
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.text(0.5, 0.5, 'No sequences\navailable', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Class {cls} - No Data', fontsize=12, fontweight='bold')
    
    total_plots = rows * cols
    for i in range(n_classes, total_plots):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_sequence_characteristics(X, y, class_names=None, sample_sequences=5):
    print("\033[1;36m" + "=" * 70 + "\033[0m")
    print("\033[1;37mSEQUENCE CHARACTERISTICS ANALYSIS\033[0m")
    print("\033[1;36m" + "=" * 70 + "\033[0m")
    
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[2]
    
    sequence_stats = create_sequence_length_boxplot(X, y, classes, class_names)
    create_signal_variability_boxplot(X, y, classes, class_names)
    create_signal_magnitude_boxplot(X, y, classes, class_names)
    
    print(f"\n\033[1;37mCreating sample sequence plots for {n_classes} classes...\033[0m")
    create_sample_sequence_plots(X, y, classes, class_names, sample_sequences)
    
    print(f"\n\033[1;37mCreating average sequence profiles for {n_classes} classes...\033[0m")
    create_feature_specific_profiles(X, y, classes, class_names, n_features)
    
    print(f"\n\033[1;37mCreating overall average profiles for {n_classes} classes...\033[0m")
    create_overall_average_profiles(X, y, classes, class_names, n_features)
    
    print(f"\n\033[1;37mCreating signal energy distribution analysis...\033[0m")
    create_signal_energy_distribution(X, y, classes, class_names)
    
    max_length_diff = max([stats['mean_length'] for stats in sequence_stats.values()]) - \
                     min([stats['mean_length'] for stats in sequence_stats.values()])
    
    print(f"\n\033[1;32mAnalysis completed successfully!\033[0m")
    print(f"Maximum difference in mean lengths: \033[1;33m{max_length_diff:.1f}\033[0m")
    
    return {
        'sequence_stats': sequence_stats,
        'max_length_difference': max_length_diff
    }


X, y, num_classes, pad_len, class_names = load_and_process_data(
    kaggle_root=kaggle_root,
    train_csv=train_csv,
    test_csv=test_csv,
    test_demo=test_demo,
    sequence_percentile=sequence_percentile
)


print(f"\nDataset Summary:")
print(f"   • Data shape: {X.shape}")
print(f"   • Number of classes: {num_classes}")
print(f"   • Sequence length: {pad_len}")
print(f"   • Class names: {list(class_names)}")

if isinstance(class_names, np.ndarray):
    class_names_list = class_names.tolist()
else:
    class_names_list = class_names

print("\n" + "="*80)
print("1. ANALYZING CLASS DISTRIBUTION")
print("="*80)
class_stats = analyze_class_distribution(y, class_names_list, "Training Data")

print("\n" + "="*80)
print("2. ANALYZING TRAIN/VALIDATION SPLIT")
print("="*80)
split_results = analyze_train_val_split(X, y, 
                                       val_split=0.2, 
                                       random_state=42,
                                       class_names=class_names_list)

print("\n" + "="*80)
print("3. ANALYZING SEQUENCE CHARACTERISTICS")
print("="*80)
sequence_stats = analyze_sequence_characteristics(X, y, 
                                                class_names=class_names_list,
                                                sample_sequences=3)

print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("Class Balance Assessment:")
if class_stats['imbalance_ratio'] > 2:
    print("   HIGH IMBALANCE - Consider class weighting or resampling")
    print(f"   Imbalance ratio: {class_stats['imbalance_ratio']:.2f}")
elif class_stats['imbalance_ratio'] > 1.5:
    print("   MODERATE IMBALANCE - Monitor for overfitting")
    print(f"   Imbalance ratio: {class_stats['imbalance_ratio']:.2f}")
else:
    print("   WELL BALANCED - Good for training")
    print(f"   Imbalance ratio: {class_stats['imbalance_ratio']:.2f}")

print(f"\nTrain/Val Stratification:")
stratification = split_results['stratification_quality']
print(f"   Quality: {stratification['quality'].upper()}")
print(f"   Max difference: {stratification['max_diff']:.2f}%")

print(f"\nSequence Length Analysis:")
max_length_diff = sequence_stats['max_length_difference']
if max_length_diff > pad_len * 0.2:
    print("   SIGNIFICANT LENGTH VARIATION - May cause bias")
    print(f"  Max difference: {max_length_diff:.1f} steps")
else:
    print("   CONSISTENT LENGTHS - Good for training")
    print(f"  Max difference: {max_length_diff:.1f} steps")

print(f"\nAnalysis complete! Ready for model training.")
print("="*80)

