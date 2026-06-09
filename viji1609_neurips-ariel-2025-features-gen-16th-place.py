import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pickle


def load_fgs1_signals(dataset, planet_ids):
    """Return FGS1 net signals for given planet IDs (shape: n_planets × 67500)."""
    arr = np.full((len(planet_ids), 67500), np.nan, dtype=np.float32)
    for i, pid in tqdm(enumerate(planet_ids), total=len(planet_ids)):
        path = f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{pid}/FGS1_signal_0.parquet'
        sig = pl.read_parquet(path).cast(pl.Int32).sum_horizontal().to_numpy() / 1024
        arr[i] = sig[1::2] - sig[0::2]
    return arr

def load_airs_signals(dataset, planet_ids):
    """Return AIRS-CH0 net signals for given planet IDs (shape: n_planets × 5625)."""
    arr = np.full((len(planet_ids), 5625), np.nan, dtype=np.float32)
    for i, pid in tqdm(enumerate(planet_ids), total=len(planet_ids)):
        path = f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{pid}/AIRS-CH0_signal_0.parquet'
        sig = pl.read_parquet(path).cast(pl.Int32).sum_horizontal().to_numpy() / (32 * 356)
        arr[i] = sig[1::2] - sig[0::2]
    return arr



file_path = "/kaggle/input/nips-25-ariel-raw-data/nd_a_raw_train.pickle"

with open(file_path, "rb") as f:
    a_raw = pickle.load(f)

file_path = "/kaggle/input/nips-25-ariel-raw-data/nd_f_raw_train.pickle"

with open(file_path, "rb") as f:
    f_raw = pickle.load(f)


def winsorize_array(arr, lower=0.0001, upper=0.0001, axis=1):
    """
    Winsorize a 2D array along the specified axis.
    """
    q_low = np.quantile(arr, lower, axis=axis, keepdims=True)
    q_high = np.quantile(arr, 1 - upper, axis=axis, keepdims=True)
    return np.clip(arr, q_low, q_high)


f_data = winsorize_array(f_raw,lower=0.0001, upper=0.0001)
a_data = winsorize_array(a_raw,lower=0.01, upper=0.01)


def generate_rr_features_multi_segments(f_raw, a_raw, segment_list=[50]):
    import numpy as np
    import pandas as pd

    def compute_features(array, name_prefix, n_segments):
        n_planets, n_cols = array.shape
        segment_size = n_cols // n_segments  # integer segments; remainder naturally sits in last segment (which we skip)

        # Only build feature names for middle segments (skip 1st and last)
        features = {
            f"{name_prefix}_{stat}_{n_segments}_{i+1}": []
            for stat in ['rr_dip', 'rr_dome']
            for i in range(1, n_segments - 1)  # skip first (0) and last (n_segments-1)
        }

        for i in range(1, n_segments - 1):  # loop only middle segments
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else n_cols

            if end > start:
                segment = array[:, start:end]
                segment_mean = segment.mean(axis=1)

                if start > 0 and end < n_cols:
                    left = array[:, :start]
                    right = array[:, end:]
                    unobscured_mean = (left.mean(axis=1) + right.mean(axis=1)) / 2
                elif start == 0:
                    right = array[:, end:]
                    unobscured_mean = right.mean(axis=1)
                else:
                    left = array[:, :start]
                    unobscured_mean = left.mean(axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    rr_dip = np.where(unobscured_mean != 0,
                                  (segment_mean - unobscured_mean) / unobscured_mean, 0.0)
                    rr_dome = np.where(unobscured_mean != 0,
                                       (unobscured_mean - segment_mean) / unobscured_mean, 0.0)

                # vectorized append
                features[f"{name_prefix}_rr_dip_{n_segments}_{i+1}"].extend(rr_dip.tolist())
                features[f"{name_prefix}_rr_dome_{n_segments}_{i+1}"].extend(rr_dome.tolist())
            else:
                features[f"{name_prefix}_rr_dip_{n_segments}_{i+1}"].extend([0.0] * n_planets)
                features[f"{name_prefix}_rr_dome_{n_segments}_{i+1}"].extend([0.0] * n_planets)

        return pd.DataFrame(features)

    dfs = []
    for n_segments in segment_list:
        dfs.append(compute_features(f_raw, "f", n_segments))
        dfs.append(compute_features(a_raw, "a", n_segments))

    return pd.concat(dfs, axis=1)


train_labels = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv', index_col='planet_id')
segment_counts = [75]
train_seg = generate_rr_features_multi_segments(f_data, a_data, segment_list=segment_counts)
train_seg.index = train_labels.index
train_seg


segment_counts = [75]
train_seg = generate_rr_features_multi_segments(f_data, a_data, segment_list=segment_counts)
train_seg.index = train_labels.index
train_seg


planet_ids = [34983, 1873185,3849793,8456603,905997089,1338107575,1738121950,1783354373]



n_rows = 4
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 15), sharex=True, sharey=True)
axes = axes.flatten()

for i, pid in enumerate(planet_ids):
    ax = axes[i]
    if pid not in train_seg.index:
        continue
    row = train_seg.loc[pid]
    rr_cols = sorted(
        [col for col in row.index if col.startswith("a_rr_dip")],
        key=lambda x: int(x.split("_")[-1])
    )
    rr_vals = row[rr_cols].values
    x = range(1, len(rr_vals) + 1)
    ax.plot(x, rr_vals, marker='o', label=f"Planet {pid}")
    ax.set_title(f"Planet ID {pid}")
    ax.grid(True)

fig.suptitle("a_rr_dip Profiles", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


n_rows = 4
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 15), sharex=True, sharey=True)
axes = axes.flatten()

for i, pid in enumerate(planet_ids):
    ax = axes[i]
    if pid not in train_seg.index:
        continue
    row = train_seg.loc[pid]
    rr_cols = sorted(
        [col for col in row.index if col.startswith("f_rr_dip")],
        key=lambda x: int(x.split("_")[-1])
    )
    rr_vals = row[rr_cols].values
    x = range(1, len(rr_vals) + 1)
    ax.plot(x, rr_vals, marker='o', label=f"Planet {pid}")
    ax.set_title(f"Planet ID {pid}")
    ax.grid(True)

fig.suptitle("f_rr_dip Profiles", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


n_rows = 4
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 15), sharex=True, sharey=True)
axes = axes.flatten()

for i, pid in enumerate(planet_ids):
    ax = axes[i]
    if pid not in train_seg.index:
        continue
    
    row = train_seg.loc[pid]
    
    # dip features
    rr_dip_cols = sorted(
        [col for col in row.index if col.startswith("a_rr_dip")],
        key=lambda x: int(x.split("_")[-1])
    )
    rr_dip_vals = row[rr_dip_cols].values
    
    # dome features (same segment numbers)
    rr_dome_cols = sorted(
        [col for col in row.index if col.startswith("a_rr_dome")],
        key=lambda x: int(x.split("_")[-1])
    )
    rr_dome_vals = row[rr_dome_cols].values
    
    x = range(1, len(rr_dip_vals) + 1)
    
    ax.plot(x, rr_dip_vals, marker='o', label="Dip", color="tab:blue")
    ax.plot(x, rr_dome_vals, marker='s', label="Dome", color="tab:orange")
    
    ax.set_title(f"Planet ID {pid}")
    ax.grid(True)
    ax.legend()

fig.suptitle("a_rr_dip vs a_rr_dome Profiles", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



segment_counts = [130, 40, 50]
train_seg = generate_rr_features_multi_segments(f_data, a_data, segment_list=segment_counts)
train_seg.index = train_labels.index
train_seg


def plot_rr_multi_segments(train_seg, planet_ids, segment_list=(30,40,50), prefix="a", kind="both",
                           n_rows=4, n_cols=2):
    """
    kind: "dip", "dome", or "both"
    prefix: "a" for AIRS, "f" for FGS1
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    def _collect(rr_kind, nseg, row):       
        cols = [c for c in row.index if c.startswith(f"{prefix}_rr_{rr_kind}_{nseg}_")]
        # sort by the trailing segment index
        cols = sorted(cols, key=lambda x: int(x.split("_")[-1]))
        return row[cols].values if len(cols) else np.array([])

    for i, pid in enumerate(planet_ids[:n_rows*n_cols]):
        ax = axes[i]
        if pid not in train_seg.index:
            ax.set_title(f"Planet {pid} (missing)")
            ax.axis("off")
            continue

        row = train_seg.loc[pid]
        for nseg in segment_list:
            x = range(1, nseg-1)  
            if kind in ("dip", "both"):
                y_dip = _collect("dip", nseg, row)
                if y_dip.size:
                    ax.plot(x, y_dip, marker="o", linestyle="-", label=f"{nseg} seg · dip")
            if kind in ("dome", "both"):
                y_dome = _collect("dome", nseg, row)
                if y_dome.size:
                    ax.plot(x, y_dome, marker="s", linestyle="--", label=f"{nseg} seg · dome")

        ax.set_title(f"Planet {pid}")
        ax.grid(True)
        ax.set_xlabel("Segment index")
        ax.set_ylabel("RR value")
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f"{prefix.upper()} RR profiles across multiple segment counts", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


plot_rr_multi_segments(train_seg, planet_ids, segment_list=(30,40,50), prefix="a", kind="both")


plot_rr_multi_segments(train_seg, planet_ids, segment_list=(30,40,50), prefix="f", kind="both")

