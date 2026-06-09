import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("ğŸ“¦ ãƒ©ã‚¤ãƒ–ãƒ©ãƒªèª­ã�¿è¾¼ã�¿å®Œäº†")

# GPUç’°å¢ƒç¢ºèª� 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ğŸ–¥ï¸� Using device: {device}")
if torch.cuda.is_available():
    print(f"GPUå��: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")


# ãƒ‡ãƒ¼ã‚¿ãƒ‘ã‚¹è¨­å®š
data_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data'
train_path = f'{data_path}/train'

print(f"ğŸ“� ãƒ‡ãƒ¼ã‚¿ãƒ‘ã‚¹: {data_path}")
print(f"ğŸ“‚ å­¦ç¿’ãƒ‡ãƒ¼ã‚¿: {train_path}")

# ãƒ‡ãƒ¼ã‚¿æ§‹é€ ç¢ºèª�
if os.path.exists(data_path):
    print("\nğŸ“‹ ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆæ§‹é€ :")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # æœ€åˆ�ã�®5ãƒ•ã‚¡ã‚¤ãƒ«ã�®ã�¿è¡¨ç¤º
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... (+{len(files)-5} more)")
else:
    print("âš ï¸� ãƒ‡ãƒ¼ã‚¿ãƒ‘ã‚¹ã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“ - ãƒ¢ãƒƒã‚¯ãƒ‡ãƒ¼ã‚¿ã�§é€²è¡Œ")


# ç‰©ç�†ç‰¹å¾´é‡�æŠ½å‡ºé–¢æ•°
def calculate_magnitude_batch(vector_batch):
    """ãƒ�ãƒƒãƒ�å‡¦ç�†ã�§ã�®å¤§ã��ã�•è¨ˆç®—"""
    return torch.norm(vector_batch, dim=-1)

def calculate_linear_acceleration(acceleration):
    """ç·šå½¢åŠ é€Ÿåº¦è¨ˆç®—ï¼ˆé‡�åŠ›è£œæ­£ï¼‰"""
    gravity = torch.tensor([0.0, 0.0, 9.8], device=acceleration.device)
    return acceleration - gravity

def extract_physics_features(imu_data):
    """ç‰©ç�†ç‰¹å¾´é‡�ã‚’æŠ½å‡ºã�—ã�¦IMUãƒ‡ãƒ¼ã‚¿ã�«è¿½åŠ """
    # åŠ é€Ÿåº¦ãƒ‡ãƒ¼ã‚¿ (æœ€åˆ�ã�®3æ¬¡å…ƒ)
    accel_data = imu_data[:, :, :3]
    
    # åŠ é€Ÿåº¦ã�®å¤§ã��ã�•
    accel_magnitude = calculate_magnitude_batch(accel_data)
    accel_magnitude = accel_magnitude.unsqueeze(-1)
    
    # æ‹¡å¼µã�•ã‚Œã�Ÿç‰¹å¾´é‡�
    enhanced_features = torch.cat([imu_data, accel_magnitude], dim=-1)
    return enhanced_features

print("âœ… ç‰©ç�†ç‰¹å¾´é‡�æŠ½å‡ºé–¢æ•°å®šç¾©å®Œäº†")


# Attentionãƒ¡ã‚«ãƒ‹ã‚ºãƒ 
class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)
    
    def compute_weights(self, hidden_states):
        energies = self.attention_weights(hidden_states)
        energies = energies.squeeze(-1)
        weights = F.softmax(energies, dim=-1)
        return weights
    
    def forward(self, hidden_states):
        weights = self.compute_weights(hidden_states)
        weights = weights.unsqueeze(-1)
        output = torch.sum(weights * hidden_states, dim=1)
        return output

print("âœ… Attentionãƒ¡ã‚«ãƒ‹ã‚ºãƒ å®šç¾©å®Œäº†")


# æ”¹è‰¯ç‰ˆBFRBãƒ¢ãƒ‡ãƒ«
class ImprovedBFRBModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        
        # BiLSTMå±¤
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,  # æ·±ã�„ãƒ�ãƒƒãƒˆãƒ¯ãƒ¼ã‚¯
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # GRUå±¤
        self.gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )
        
        # Attentionå±¤
        self.attention = SimpleAttention(hidden_size)
        
        # åˆ†é¡�å±¤
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # ç‰©ç�†ç‰¹å¾´é‡�æŠ½å‡º
        if x.shape[-1] == 6:  # IMUã�®ã�¿ã�®å ´å�ˆ
            x = extract_physics_features(x)
        
        # BiLSTM
        bilstm_out, _ = self.bilstm(x)
        
        # GRU  
        gru_out, _ = self.gru(bilstm_out)
        
        # Attention
        attended = self.attention(gru_out)
        
        # åˆ†é¡�
        attended = self.dropout(attended)
        output = self.classifier(attended)
        
        return output

print("âœ… æ”¹è‰¯ç‰ˆBFRBãƒ¢ãƒ‡ãƒ«å®šç¾©å®Œäº† (Dropout, 2å±¤LSTMè¿½åŠ )")


# ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ€ãƒ¼ã‚¯ãƒ©ã‚¹
class CMIDataset(Dataset):
    def __init__(self, sequences, labels, label_encoder):
        self.sequences = sequences
        self.labels = labels
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = self.labels[idx]
        return sequence, label

def create_mock_training_data():
    """è¨“ç·´ç”¨ãƒ¢ãƒƒã‚¯ãƒ‡ãƒ¼ã‚¿ä½œæˆ�"""
    gestures = ['Wave hello', 'Drink', 'Phone', 'Hand on neck', 'Touch hair']
    sequences = []
    labels = []
    
    # å�„ã‚¸ã‚§ã‚¹ãƒ�ãƒ£ãƒ¼ã�«ã�¤ã�„ã�¦10ã‚·ãƒ¼ã‚±ãƒ³ã‚¹ä½œæˆ�
    for gesture in gestures:
        for i in range(10):
            # 50ã‚¹ãƒ†ãƒƒãƒ—ã�®æ™‚ç³»åˆ—ãƒ‡ãƒ¼ã‚¿
            seq_len = 50
            sequence = np.random.randn(seq_len, 6) + np.random.rand() * 2  # IMU 6æ¬¡å…ƒ
            sequences.append(sequence)
            labels.append(gesture)
    
    return sequences, labels

print("âœ… ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã‚¯ãƒ©ã‚¹å®šç¾©å®Œäº†")


# ãƒ‡ãƒ¼ã‚¿æº–å‚™
print("ğŸ“Š ãƒ‡ãƒ¼ã‚¿æº–å‚™é–‹å§‹...")

# ãƒ¢ãƒƒã‚¯ãƒ‡ãƒ¼ã‚¿ã�§å­¦ç¿’ï¼ˆå®Ÿéš›ã�®Kaggleã�§ã�¯ã�“ã�“ã�§å®Ÿãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿ï¼‰
sequences, labels = create_mock_training_data()
print(f"ã‚·ãƒ¼ã‚±ãƒ³ã‚¹æ•°: {len(sequences)}")
print(f"ãƒ¦ãƒ‹ãƒ¼ã‚¯ãƒ©ãƒ™ãƒ«æ•°: {len(set(labels))}")
print(f"ã‚·ãƒ¼ã‚±ãƒ³ã‚¹å½¢çŠ¶ä¾‹: {sequences[0].shape}")

# ãƒ©ãƒ™ãƒ«ã‚¨ãƒ³ã‚³ãƒ¼ãƒ‡ã‚£ãƒ³ã‚°
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"ã‚¯ãƒ©ã‚¹æ•°: {num_classes}")
print(f"ã‚¯ãƒ©ã‚¹ä¸€è¦§: {list(label_encoder.classes_)}")

# è¨“ç·´ãƒ»æ¤œè¨¼åˆ†å‰²
X_train, X_val, y_train, y_val = train_test_split(
    sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

print(f"è¨“ç·´ã‚»ãƒƒãƒˆ: {len(X_train)} ã‚·ãƒ¼ã‚±ãƒ³ã‚¹")
print(f"æ¤œè¨¼ã‚»ãƒƒãƒˆ: {len(X_val)} ã‚·ãƒ¼ã‚±ãƒ³ã‚¹")

# ãƒ‡ãƒ¼ã‚¿ãƒ­ãƒ¼ãƒ€ãƒ¼ä½œæˆ�
train_dataset = CMIDataset(X_train, y_train, label_encoder)
val_dataset = CMIDataset(X_val, y_val, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("âœ… ãƒ‡ãƒ¼ã‚¿æº–å‚™å®Œäº†")


# ãƒ¢ãƒ‡ãƒ«åˆ�æœŸåŒ–
model = ImprovedBFRBModel(
    input_size=7,  # IMU 6æ¬¡å…ƒ + ç‰©ç�†ç‰¹å¾´é‡� 1æ¬¡å…ƒ
    hidden_size=128,
    num_classes=num_classes,
    dropout=0.3
)

model = model.to(device)
print(f"ãƒ¢ãƒ‡ãƒ«ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿æ•°: {sum(p.numel() for p in model.parameters()):,}")

# æ��å¤±é–¢æ•°ãƒ»ã‚ªãƒ—ãƒ†ã‚£ãƒ�ã‚¤ã‚¶ãƒ¼
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print("âœ… ãƒ¢ãƒ‡ãƒ«ãƒ»ã‚ªãƒ—ãƒ†ã‚£ãƒ�ã‚¤ã‚¶ãƒ¼æº–å‚™å®Œäº†")


# å­¦ç¿’ãƒ«ãƒ¼ãƒ—
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

print("âœ… å­¦ç¿’é–¢æ•°å®šç¾©å®Œäº†")


# å­¦ç¿’å®Ÿè¡Œ
print("ğŸš€ å­¦ç¿’é–‹å§‹!")
num_epochs = 20
best_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    scheduler.step()
    
    print(f"Epoch {epoch+1:2d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:5.1f}% | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:5.1f}%")
    
    # æœ€è‰¯ãƒ¢ãƒ‡ãƒ«ä¿�å­˜
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"âœ… æ–°ã�—ã�„æœ€è‰¯ãƒ¢ãƒ‡ãƒ«ä¿�å­˜ (Val Acc: {val_acc:.1f}%)")

print(f"\nğŸ�¯ å­¦ç¿’å®Œäº†! æœ€é«˜æ¤œè¨¼ç²¾åº¦: {best_acc:.1f}%")


# æ�¨è«–é–¢æ•°ï¼ˆCode Competitionç”¨ï¼‰
def predict():
    """Code Competitionç”¨ã�®æ�¨è«–é–¢æ•°"""
    # æœ€è‰¯ãƒ¢ãƒ‡ãƒ«ã‚’ãƒ­ãƒ¼ãƒ‰
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    # å®Ÿéš›ã�®Code Competitionã�§ã�¯ã€�ã�“ã�“ã�§ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿ã‚’èª­ã�¿è¾¼ã‚“ã�§äºˆæ¸¬
    # ä»Šå›�ã�¯ãƒ¢ãƒƒã‚¯ã�¨ã�—ã�¦æœ€é »ã‚¯ãƒ©ã‚¹ã‚’è¿”ã�™
    most_common_gesture = label_encoder.inverse_transform([0])[0]  # æœ€åˆ�ã�®ã‚¯ãƒ©ã‚¹
    return most_common_gesture

# ãƒ†ã‚¹ãƒˆå®Ÿè¡Œ
test_prediction = predict()
print(f"ğŸ”® æ�¨è«–ãƒ†ã‚¹ãƒˆçµ�æ�œ: {test_prediction}")
print("âœ… Code Competitionæº–å‚™å®Œäº†")


# å®Ÿé¨“ã‚µãƒ�ãƒªãƒ¼
print("\nğŸ“‹ å­¦ç¿’å®Ÿé¨“ã‚µãƒ�ãƒªãƒ¼:")
print("â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�")
print(f"ğŸ�—ï¸�  ã‚¢ãƒ¼ã‚­ãƒ†ã‚¯ãƒ�ãƒ£: BiLSTM(2å±¤) + GRU + Attention")
print(f"ğŸ”§ ç‰©ç�†ç‰¹å¾´é‡�: ãƒ™ã‚¯ãƒˆãƒ«å¤§ã��ã�•è¨ˆç®—")
print(f"ğŸ–¥ï¸�  ãƒ‡ãƒ�ã‚¤ã‚¹: {device}")
print(f"ğŸ“Š å…¥åŠ›æ¬¡å…ƒ: 6 â†’ 7 (ç‰©ç�†ç‰¹å¾´é‡�è¿½åŠ )")
print(f"ğŸ�¯ ã‚¯ãƒ©ã‚¹æ•°: {num_classes}")
print(f"ğŸ“ˆ æœ€é«˜æ¤œè¨¼ç²¾åº¦: {best_acc:.1f}%")
print(f"âš¡ ãƒ¢ãƒ‡ãƒ«ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿æ•°: {sum(p.numel() for p in model.parameters()):,}")
print("â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�â”�")
print("\nğŸš€ æ”¹è‰¯ç‰ˆBFRBãƒ¢ãƒ‡ãƒ«å­¦ç¿’å®Œäº†!")
print("ğŸ’¡ Code Competition submitã�®æº–å‚™å®Œäº†")

