import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


print("ğŸ“‚ Veri yÃ¼kleniyor...")

labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']

zipdir = Path.cwd().parent / 'input' / 'grasp-and-lift-eeg-detection'
for zipfile in zipdir.glob('*.zip'):
    with ZipFile(zipfile, 'r') as zf:
        zf.extractall()

xs, ys = [], []
traindir = Path('train')

for datapath in tqdm(sorted(traindir.glob('*_data.csv')), desc="Train data loading"):
    eventpath = datapath.parent / (datapath.stem[:-5] + '_events.csv')
    
    x = pd.read_csv(datapath).iloc[:, 1:].values.astype(np.float32)
    y = pd.read_csv(eventpath).iloc[:, 1:].values.astype(np.float32)
    
    xs.append(x)
    ys.append(y)

xs_train, ys_train = xs[:-2], ys[:-2]
xs_valid, ys_valid = xs[-2:], ys[-2:]

print(f"âœ… Train: {len(xs_train)} subjects, Valid: {len(xs_valid)} subjects")


def normalize_eeg(x):
    
    return (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + 1e-8)

def add_features(x):
    
    diff = np.diff(x, axis=1)
    diff = np.concatenate([diff, diff[:, -1:]], axis=1)  
    
    
    window = 5
    ma = np.convolve(x.flatten(), np.ones(window)/window, 'same').reshape(x.shape)
    
    return np.concatenate([x, diff, ma], axis=1)


print("ğŸ”§ Veri Ã¶n iÅŸleme...")
for i in range(len(xs_train)):
    xs_train[i] = add_features(normalize_eeg(xs_train[i]))
for i in range(len(xs_valid)):
    xs_valid[i] = add_features(normalize_eeg(xs_valid[i]))



class EEGDataset(Dataset):
    def __init__(self, xs, ys=None, window_size=1000, stride=500, mode='train'):
        self.xs = xs
        self.ys = ys
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        
        self.samples = []
        for subj_idx, x in enumerate(xs):
            seq_len = x.shape[0]
            if mode == 'train':
                for start in range(0, seq_len - window_size + 1, stride):
                    self.samples.append((subj_idx, start, start + window_size))
            else:
                self.samples.append((subj_idx, 0, seq_len))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        subj_idx, start, end = self.samples[idx]
        x = self.xs[subj_idx][start:end]
        
        if self.mode == 'train':
            if np.random.random() > 0.5:
                x = x + np.random.normal(0, 0.01, x.shape)
            if np.random.random() > 0.5:
                shift = np.random.randint(-10, 11)
                x = np.roll(x, shift, axis=0)
            
            y = self.ys[subj_idx][start:end]
            return torch.tensor(x.T, dtype=torch.float32), torch.tensor(y.T, dtype=torch.float32)
        
        else:
            if self.ys is not None:
                y = self.ys[subj_idx]
                return torch.tensor(x.T, dtype=torch.float32), torch.tensor(y.T, dtype=torch.float32)
            else:
                return torch.tensor(x.T, dtype=torch.float32)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.scales = [1, 3] 
        self.attentions = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels//2, k, padding=k//2) 
            for k in self.scales
        ])
        self.combine = nn.Conv1d(in_channels, in_channels, 1)
        
    def forward(self, x):
        features = [att(x) for att in self.attentions]
        combined = torch.cat(features, dim=1)
        
        weights = torch.softmax(self.combine(combined), dim=2)
        return x * weights

class EEGNet(nn.Module):
    def __init__(self, n_channels=96, n_classes=6, seq_len=750):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            ResidualBlock1D(32, 64, kernel_size=5, stride=2),
            ResidualBlock1D(64, 128, kernel_size=3, stride=2),
        )
        
        self.attention = MultiScaleAttention(128, seq_len//4)
        
        self.lstm = nn.LSTM(128, 64, num_layers=1, batch_first=True, 
                           bidirectional=True, dropout=0.0)
        
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, n_classes, kernel_size=1)
        )
        
    def forward(self, x):
        original_seq_len = x.shape[2]
        
        x = self.features(x)
        
        
        x = self.attention(x)
        x_lstm = x.transpose(1, 2)  
        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm.transpose(1, 2)  
        
       
        output = self.classifier(x_lstm)
        
        
        if output.shape[2] != original_seq_len:
            output = F.interpolate(output, size=original_seq_len, mode='linear', align_corners=False)
        
        return output



class FocalLoss(nn.Module):
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()



def train_model(model, train_loader, valid_loader, device, epochs=20):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)  
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1  
    )
    
    criterion = FocalLoss(alpha=1, gamma=2)
    scaler = torch.cuda.amp.GradScaler()
    
    best_score = 0
    best_model = None
    
   
    train_losses_history = []
    val_aucs_history = []
    epochs_history = []
    
    
    for epoch in range(epochs):
       
        model.train()
        train_losses = []
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_losses.append(loss.item())
        
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            plot_roc = (epoch == epochs - 1)  
            val_score = validate_model(model, valid_loader, device, plot_roc=plot_roc)
            avg_train_loss = np.mean(train_losses)
            
            
            train_losses_history.append(avg_train_loss)
            val_aucs_history.append(val_score)
            epochs_history.append(epoch + 1)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val AUC: {val_score:.4f}")
            
            if val_score > best_score:
                best_score = val_score
                best_model = model.state_dict().copy()
                print(f"ğŸ’¯ New best validation AUC: {best_score:.4f}")
    
   
    plot_training_history(epochs_history, train_losses_history, val_aucs_history)
    
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, best_score

def plot_training_history(epochs, train_losses, val_aucs):
    
    plt.figure(figsize=(15, 5))
    
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log') 
    
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_aucs, 'r-o', linewidth=2, markersize=6)
    plt.title('Validation AUC Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])  
    
    
    plt.subplot(1, 3, 3)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6, label='Train Loss')
    line2 = ax2.plot(epochs, val_aucs, 'r-o', linewidth=2, markersize=6, label='Val AUC')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('AUC', color='r')
    ax1.set_yscale('log')
    ax2.set_ylim([0, 1])
    
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    print(f"\nğŸ“ˆ Training Summary:")
    print(f"   Initial Loss: {train_losses[0]:.4f}")
    print(f"   Final Loss: {train_losses[-1]:.4f}")
    print(f"   Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    print(f"   Initial AUC: {val_aucs[0]:.4f}")
    print(f"   Final AUC: {val_aucs[-1]:.4f}")
    print(f"   Best AUC: {max(val_aucs):.4f}")
    print(f"   AUC Improvement: {((max(val_aucs) - val_aucs[0]) / val_aucs[0] * 100 if val_aucs[0] > 0 else 0):.1f}%")

def validate_model(model, valid_loader, device, plot_roc=True):
    
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            
            
            seq_len = x.shape[2]
            window_size = 1000
            stride = 500
            
            if seq_len <= window_size:
                with torch.cuda.amp.autocast():
                    outputs = torch.sigmoid(model(x))
                preds = outputs.cpu().numpy()[0]  
                targets = y.numpy()[0]  
            else:
                
                pred_np = np.zeros((6, seq_len))
                counts = np.zeros(seq_len)
                
                for start in range(0, seq_len, stride):
                    end = min(start + window_size, seq_len)
                    actual_len = end - start
                    x_chunk = x[:, :, start:end]
                    
                    if x_chunk.shape[2] < window_size:
                        
                        pad_size = window_size - x_chunk.shape[2]
                        x_chunk = F.pad(x_chunk, (0, pad_size))
                    
                    with torch.cuda.amp.autocast():
                        chunk_pred = torch.sigmoid(model(x_chunk))
                    
                    
                    chunk_pred = chunk_pred[0, :, :actual_len].cpu().numpy()  # (n_classes, actual_len)
                    
                    
                    pred_np[:, start:end] += chunk_pred
                    counts[start:end] += 1
                
                
                counts[counts == 0] = 1  
                pred_np = pred_np / counts[np.newaxis, :]
                preds = pred_np
                targets = y.numpy()[0]  
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = np.concatenate(all_preds, axis=1) 
    all_targets = np.concatenate(all_targets, axis=1)
    
    if all_preds.shape != all_targets.shape:
        print(f"â�Œ Shape mismatch: preds {all_preds.shape} vs targets {all_targets.shape}")
        return 0.0
    
    try:
        auc_scores = []
        
        if plot_roc:
            plt.figure(figsize=(15, 10))
            
        for i in range(6):
            n_positives = np.sum(all_targets[i])
            n_negatives = len(all_targets[i]) - n_positives
            
            if n_positives > 0 and n_negatives > 0:
                pred_class = all_preds[i]
                target_class = all_targets[i]
                
                if len(pred_class) != len(target_class):
                    min_len = min(len(pred_class), len(target_class))
                    pred_class = pred_class[:min_len]
                    target_class = target_class[:min_len]
                
                auc = roc_auc_score(target_class, pred_class)
                auc_scores.append(auc)
                
                if plot_roc:
                    fpr, tpr, _ = roc_curve(target_class, pred_class)
                    plt.subplot(2, 3, i+1)
                    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
                    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'{labels[i]}')
                    plt.legend(loc="lower right")
                    plt.grid(True, alpha=0.3)
            else:
                auc_scores.append(0.0)
        
        if plot_roc and any(score > 0 for score in auc_scores):
            plt.tight_layout()
            plt.suptitle('ROC Curves for All Classes', fontsize=16, y=1.02)
            plt.show()
        
        mean_auc = np.mean(auc_scores) if auc_scores else 0.0
        print(f"ğŸ�† Mean AUC: {mean_auc:.4f}")
        print(f"ğŸ“Š Individual AUCs: {[f'{score:.4f}' for score in auc_scores]}")
        
        
        if plot_roc and any(score > 0 for score in auc_scores):
            plot_auc_comparison(auc_scores, labels)
        
        return mean_auc
    except Exception as e:
        print(f"â�Œ AUC calculation error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def plot_auc_comparison(auc_scores, class_labels):
    plt.figure(figsize=(12, 6))
    
   
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    bars = plt.bar(range(len(auc_scores)), auc_scores, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    
    for i, (bar, score) in enumerate(zip(bars, auc_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                label='Random Guess (0.5)')
    
    mean_auc = np.mean(auc_scores)
    plt.axhline(y=mean_auc, color='green', linestyle='-', alpha=0.7, 
                label=f'Mean AUC ({mean_auc:.3f})')
    plt.xlabel('EEG Event Classes', fontweight='bold')
    plt.ylabel('AUC Score', fontweight='bold')
    plt.title('AUC Scores by Event Class', fontsize=16, fontweight='bold')
    plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ğŸš€ Device: {device}")


window_sizes = [750]  
models = []

for i, window_size in enumerate(window_sizes):
    print(f"\nğŸ�¯ Training Stage {i+1}: Window Size = {window_size}")
    
    
    train_dataset = EEGDataset(xs_train, ys_train, window_size=window_size, stride=window_size//2, mode='train')
    valid_dataset = EEGDataset(xs_valid, ys_valid, window_size=window_size, mode='valid')
    
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=1)  # Daha bÃ¼yÃ¼k batch
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
     
    model = EEGNet(n_channels=96, n_classes=6, seq_len=window_size).to(device)
    
    
    if i > 0 and models:
        try:
            
            prev_state = models[-1].state_dict()
            current_state = model.state_dict()
            
            for key in current_state:
                if key in prev_state and current_state[key].shape == prev_state[key].shape:
                    current_state[key] = prev_state[key]
            model.load_state_dict(current_state)
            print("ğŸ“¦ Transferred weights from previous stage")
        except:
            print("âš ï¸� Could not transfer weights, training from scratch")
    
    epochs = 20  
    model, score = train_model(model, train_loader, valid_loader, device, epochs)
    models.append(model)
    
    print(f"âœ… Stage {i+1} completed with AUC: {score:.4f}")


print("\nğŸ”® Test predictions...")


xs_test = []
lengths = {}
testdir = Path('test')

for subj in range(1, 13):
    for series in [9, 10]:
        datapath = testdir / f'subj{subj}_series{series}_data.csv'
        x = pd.read_csv(datapath).iloc[:, 1:].values.astype(np.float32)
        x = add_features(normalize_eeg(x))
        xs_test.append(x)
        lengths[f'{subj}_{series}'] = x.shape[0]


test_dataset = EEGDataset(xs_test, mode='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)


print("ğŸ�ª Ensemble prediction...")
all_predictions = []

for model_idx, model in enumerate(models):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for test_idx, x in enumerate(tqdm(test_loader, desc=f"Model {model_idx+1} predicting")):
            x = x.to(device)
            
            seq_len = x.shape[2]
            window_size = 1000
            stride = 500
            
            if seq_len <= window_size:
                with torch.cuda.amp.autocast():
                    pred = torch.sigmoid(model(x))
                pred_np = pred.cpu().numpy()[0].T  
            else:
                preds_list = []
                positions = []
                
                for start in range(0, seq_len, stride):
                    end = min(start + window_size, seq_len)
                    actual_len = end - start
                    
                    x_chunk = x[:, :, start:end]
                    
                    if x_chunk.shape[2] < window_size:
                        pad_size = window_size - x_chunk.shape[2]
                        x_chunk = F.pad(x_chunk, (0, pad_size))
                    
                    with torch.cuda.amp.autocast():
                        chunk_pred = torch.sigmoid(model(x_chunk))
                    
                    chunk_pred = chunk_pred[:, :, :actual_len]
                    preds_list.append(chunk_pred.cpu().numpy()[0].T)  # (actual_len, 6)
                    positions.append((start, end))
                
                pred_np = np.zeros((seq_len, 6))
                counts = np.zeros(seq_len)
                
                for pred_chunk, (start, end) in zip(preds_list, positions):
                    pred_np[start:end] += pred_chunk
                    counts[start:end] += 1
                
                counts[counts == 0] = 1  # Division by zero'yu Ã¶nle
                pred_np = pred_np / counts[:, np.newaxis]
            
            predictions.append(pred_np)
    
    all_test_preds = np.concatenate(predictions, axis=0)
    all_predictions.append(all_test_preds)
    print(f"Model {model_idx+1} predictions shape: {all_test_preds.shape}")

final_predictions = np.mean(all_predictions, axis=0)
print(f"Final predictions shape: {final_predictions.shape}")


print("ğŸ“� Creating submission...")

indices = []
for sbj in range(1, 13):
    for series in [9, 10]:
        for t in range(lengths[f'{sbj}_{series}']):
            indices.append(f'subj{sbj}_series{series}_{t}')

print(f"Number of indices: {len(indices)}")
print(f"Final predictions shape: {final_predictions.shape}")

submission = pd.DataFrame(
    final_predictions,
    index=indices,
    columns=labels
)

submission.to_csv('submission.csv', index_label='id', float_format='%.6f')

print("âœ… Submission saved!")
print(f"ğŸ“Š Submission shape: {submission.shape}")
print(f"ğŸ“ˆ Prediction range: [{submission.values.min():.4f}, {submission.values.max():.4f}]")

print("\nğŸ“Š Final Prediction Statistics per Class:")
for i, label_name in enumerate(labels):
    class_preds = submission.iloc[:, i].values
    print(f"{label_name}:")
    print(f"  Mean: {np.mean(class_preds):.4f}")
    print(f"  Std: {np.std(class_preds):.4f}")
    print(f"  Min: {np.min(class_preds):.4f}")
    print(f"  Max: {np.max(class_preds):.4f}")
    print(f"  Median: {np.median(class_preds):.4f}")

plt.figure(figsize=(15, 10))
for i, label_name in enumerate(labels):
    plt.subplot(2, 3, i+1)
    plt.hist(submission.iloc[:, i].values, bins=50, alpha=0.7, density=True)
    plt.title(f'{label_name} Prediction Distribution')
    plt.xlabel('Prediction Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Final Prediction Distributions', fontsize=16, y=1.02)
plt.show()


submission.head()


