import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
CONF = {
    'root_dir': "/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog",
    'window_size': 384,
    'step_size': 128,
    'batch_size': 64,
    'lr': 5e-4,
    'epochs': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': "/kaggle/working/",
}

# ==========================================
# 2. ROBUST DATASET CLASS
# ==========================================
class FoGDataset(Dataset):
    def __init__(self, root_dir, window_size=384, step_size=128):
        self.windows, self.labels, self.subjects = [], [], []
        csv_files = [f for f in sorted(os.listdir(root_dir)) if f.endswith('.csv')]
        
        print(f"Loading and Normalizing {len(csv_files[:100])} patient files...")
        for file in csv_files[:100]:
            df = pd.read_csv(os.path.join(root_dir, file))
            cols = ['AccV', 'AccML', 'AccAP']
            data = df[cols].values
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
            labels = df['StartHesitation'].values
            for start in range(0, len(df) - window_size + 1, step_size):
                self.windows.append(data[start:start+window_size].T)
                self.labels.append(labels[start:start+window_size].max())
                self.subjects.append(file.split('_')[0])
        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels  = np.array(self.labels,  dtype=np.float32)
        self.subjects = np.array(self.subjects)

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx]), torch.tensor(self.labels[idx])

# ==========================================
# 3. THE THREE MODELS (FOR ABLATION)
# ==========================================
class CNN_Only(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, x, return_feats=False):
        feat = self.cnn(x).squeeze(-1)
        if return_feats: return feat
        return self.classifier(feat)

class CNN_LSTM_NoAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x, return_feats=False):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        feat = x[:, -1, :]
        if return_feats: return feat
        return self.classifier(feat)

class FoG1DLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Linear(256, 1)

    def forward(self, x, return_feats=False):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        weights = F.softmax(self.attention(x), dim=1)
        feat = (x * weights).sum(dim=1)
        if return_feats: return feat
        return self.classifier(feat)

# ==========================================
# 4. METRICS & TRAINING LOGIC
# ==========================================
def get_metrics(y_true, y_probs):
    y_pred = (y_probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    mAP = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try: auc = roc_auc_score(y_true, y_probs)
    except: auc = 0.5
    return mAP, sens, spec, auc

# ==========================================
# 5. CONFUSION MATRIX PLOT (BLUE)
# ==========================================
def plot_confusion_matrices(cm_data: dict, output_dir: str):
    """
    cm_data: { model_name: (y_true np.array, y_pred np.array) }
    Saves one PNG per model confusion matrix.
    """
    blue_cmap = mcolors.LinearSegmentedColormap.from_list(
        'fog_blue', ['#eef4fc', '#b8d4ee', '#5b9bd5', '#1a5ea8', '#0a2d6e']
    )
    label_names = ['Normal', 'FoG']

    for name, (y_true, y_pred) in cm_data.items():
        fig, ax = plt.subplots(figsize=(6, 5))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

        annot = np.array([
            [f"{cm[r, c]:,}\n({cm_norm[r, c]*100:.1f}%)" for c in range(2)]
            for r in range(2)
        ])

        sns.heatmap(
            cm_norm,
            ax=ax,
            cmap=blue_cmap,
            annot=annot,
            fmt='',
            linewidths=2,
            linecolor='white',
            cbar=True,
            vmin=0, vmax=1,
            annot_kws={'size': 13, 'weight': 'bold'},
            xticklabels=label_names,
            yticklabels=label_names,
        )

        # Auto-contrast text
        for text_obj, (r, c) in zip(ax.texts, [(r, c) for r in range(2) for c in range(2)]):
            text_obj.set_color('white' if cm_norm[r, c] > 0.45 else '#0a2d6e')

        title_color = '#0a2d6e' if 'Proposed' in name else '#444444'
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=13, fontweight='bold',
                     pad=12, color=title_color)
        ax.set_xlabel('Predicted Label', fontsize=11, labelpad=8)
        ax.set_ylabel('True Label',      fontsize=11, labelpad=8)
        ax.tick_params(axis='both', labelsize=10)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8)
            spine.set_edgecolor('#1a5ea8')

        plt.tight_layout()

        clean = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        save_path = os.path.join(output_dir, f"{clean}_confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved confusion matrix → {save_path}")
        plt.show()
        plt.close()

# ==========================================
# 6. MAIN COMPARISON STUDY
# ==========================================
def run_comparison_study():
    os.makedirs(CONF['output_dir'], exist_ok=True)

    ds = FoGDataset(CONF['root_dir'])

    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(ds.windows, ds.labels, ds.subjects):
        if ds.labels[val_idx].sum() > 20: break

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx),
                              batch_size=CONF['batch_size'], shuffle=True)
    val_loader   = DataLoader(torch.utils.data.Subset(ds, val_idx),
                              batch_size=CONF['batch_size'], shuffle=False)

    models = {
        "CNN Only":              CNN_Only(),
        "CNN + LSTM":            CNN_LSTM_NoAttn(),
        "FoG1DLSTM (Proposed)":  FoG1DLSTM(),
    }

    results  = {}
    tsne_data = {}
    cm_data   = {}   # ← collects (y_true, y_pred) per model for conf matrices

    for name, model in models.items():
        print(f"\n{'='*50}\nTraining: {name}\n{'='*50}")
        model.to(CONF['device'])
        optimizer = optim.Adam(model.parameters(), lr=CONF['lr'])
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([35.0]).to(CONF['device'])
        )

        for epoch in range(CONF['epochs']):
            model.train()
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(CONF['device']), y.to(CONF['device'])
                optimizer.zero_grad()
                loss = criterion(model(X).squeeze(), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:02d}/{CONF['epochs']} | Loss: {epoch_loss/len(train_loader):.4f}")

        # ── Evaluation ──────────────────────────────────────────────────────
        model.eval()

        # Latency benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model(torch.randn(1, 3, 384).to(CONF['device']))
        latency = (time.time() - start_time) / 50 * 1000

        probs, targets, feats = [], [], []
        with torch.no_grad():
            for X, y in val_loader:
                f = model(X.to(CONF['device']), return_feats=True)
                p = torch.sigmoid(model.classifier(f).squeeze())
                probs.extend(p.cpu().numpy())
                targets.extend(y.numpy())
                feats.extend(f.cpu().numpy())

        probs_arr   = np.array(probs)
        targets_arr = np.array(targets)
        preds_arr   = (probs_arr > 0.5).astype(int)

        mAP, sens, spec, auc = get_metrics(targets_arr, probs_arr)
        results[name]  = [mAP, sens, spec, auc, latency]
        tsne_data[name] = (np.array(feats), targets_arr)
        cm_data[name]   = (targets_arr, preds_arr)   # ← store for conf matrix

        print(f"  AUC: {auc:.4f} | Sensitivity: {sens:.4f} | Specificity: {spec:.4f} | mAP: {mAP:.4f}")

    # ── Results Table ────────────────────────────────────────────────────────
    df_results = pd.DataFrame(
        results, index=['mAP', 'Sensitivity', 'Specificity', 'AUC-ROC', 'Latency (ms)']
    ).T
    print("\n" + "="*50 + "\nFINAL ABLATION TABLE\n" + "="*50)
    print(df_results.to_string())

    # ── t-SNE Plots ──────────────────────────────────────────────────────────
    for name, (f, t) in tsne_data.items():
        plt.figure(figsize=(8, 6))
        pos_idx = np.where(t == 1)[0]
        neg_idx = np.where(t == 0)[0][:len(pos_idx) * 2]
        plot_idx = np.concatenate([pos_idx, neg_idx])
        embed = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(f[plot_idx])
        sns.scatterplot(x=embed[:, 0], y=embed[:, 1], hue=t[plot_idx],
                        palette='rocket', alpha=0.7)
        plt.title(f"{name} — Feature Clustering (t-SNE)")
        clean = name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(CONF['output_dir'], f"{clean}_tSNE.svg")
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Saved t-SNE → {save_path}")
        plt.show()
        plt.close()

    # ── Confusion Matrices ───────────────────────────────────────────────────
    plot_confusion_matrices(cm_data, CONF['output_dir'])


if __name__ == "__main__":
    run_comparison_study()




