import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os
import kornia.augmentation as K
import numpy as np

# Проверка CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


import math
import torch.nn as nn

class PositionalEncoding1d(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # (max_len, D)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)
    
class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model, max_h=64, max_w=256):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"

        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w

        d_half = d_model // 2

        # --- H positional encoding ---
        pe_h = torch.zeros(max_h, d_half)
        pos_h = torch.arange(max_h).unsqueeze(1)
        div_h = torch.exp(
            torch.arange(0, d_half, 2) * (-math.log(10000.0) / d_half)
        )
        pe_h[:, 0::2] = torch.sin(pos_h * div_h)
        pe_h[:, 1::2] = torch.cos(pos_h * div_h)

        # --- W positional encoding ---
        pe_w = torch.zeros(max_w, d_half)
        pos_w = torch.arange(max_w).unsqueeze(1)
        div_w = torch.exp(
            torch.arange(0, d_half, 2) * (-math.log(10000.0) / d_half)
        )
        pe_w[:, 0::2] = torch.sin(pos_w * div_w)
        pe_w[:, 1::2] = torch.cos(pos_w * div_w)

        self.register_buffer("pe_h", pe_h)  # (max_h, D/2)
        self.register_buffer("pe_w", pe_w)  # (max_w, D/2)

    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        """
        B, T, C = x.shape
        assert T == H * W, "T must be equal to H*W"
        assert H <= self.max_h and W <= self.max_w, "H or W exceeds max size"

        pe_h = self.pe_h[:H].unsqueeze(1).expand(H, W, C // 2)
        pe_w = self.pe_w[:W].unsqueeze(0).expand(H, W, C // 2)

        pe = torch.cat([pe_h, pe_w], dim=-1)   # (H, W, C)
        pe = pe.view(1, H * W, C)              # (1, HW, C)

        return x + pe

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        # x: (B, T, C)
        return self.encoder(x)
    
class PosTransformer1d(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_len=128):
        super().__init__()

        self.pos_enc = PositionalEncoding1d(embed_dim, max_len)

        self.encoder = TransformerBlock(embed_dim, num_heads, num_layers)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x
    
class PosTransformer2d(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.pos_enc = PositionalEncoding2d(embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, num_layers)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 1) flatten 2D → tokens
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # 2) positional encoding
        x = self.pos_enc(x, H, W)

        # 3) transformer
        x = self.transformer(x)

        # 4) restore 2D
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x

class SVTR_OCR(nn.Module):
    def __init__(
        self,
        num_classes
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), padding=(0, 0), stride=(2, 2), bias=False), # (3, 50, 50) -> (128, 24, 24)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        self.transformer2d = nn.Sequential(
            PosTransformer2d(256, 8, 1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False), # (256, 6, 6)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            PosTransformer2d(256, 8, 1),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2), bias=False), # (256, 3, 3)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            PosTransformer2d(256, 8, 1),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(0, 0), stride=(1, 1), bias=False), # (256, 1, 1)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.encoder(x)      # (B, C, H', W')
        x = self.transformer2d(x)  # (B, C, H', W')
        x = x.squeeze(2)          # (B, C, W')
        x = x.squeeze(2)          # (B, C)
        x = self.classifier(x)   # (B, T, num_classes)
        return x


image_size = 50  # Размер изображения

augmentation_pipeline = nn.Sequential(
    K.RandomAffine(degrees=11.0, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5),
)

# Функция для аугментации всего датасета
def augment_dataset(X, y, pipeline, device):
    X_tensor = torch.FloatTensor(X.values).reshape(-1, 1, image_size, image_size).to(device)
    X_aug = pipeline(X_tensor)
    X_aug = X_aug.cpu()
    return X_aug, torch.LongTensor(y.values)

# Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None, is_tensor=False):
        if is_tensor:
            self.X = X
            self.y = y
        else:
            self.X = torch.FloatTensor(X.values).reshape(-1, 1, image_size, image_size)
            self.y = torch.LongTensor(y.values)
        self.transform = transform
        
        # Validate labels
        assert self.y.min() >= 0, f"Labels contain negative values: {self.y.min()}"
        assert self.y.max() < len(y.unique()), f"Labels exceed num_classes: max={self.y.max()}, num_classes={len(y.unique())}"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def validation_training(
    model, train_X, train_y, val_X, val_y, criterion, optimizer, scheduler,
    num_epochs=10, device=device, clip_grad_norm=1.0, model_path='best_model.pth',
    augmentation_pipeline=None, freeze_list=None, patience=10
):
    """
    Дообучение с валидацией, возможностью аугментации и заморозки указанных слоёв.
    freeze_list: список имён слоёв для заморозки (например, ['conv1', 'conv2']). Если None — обучаются все.
    augmentation_pipeline: если не None, применяется к train данным перед каждой эпохой.
    Early stopping по val loss.
    """
    # Заморозить указанные слои, если freeze_list не None
    if freeze_list is not None:
        for name, module in model.named_children():
            if name in freeze_list:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
        print(f"Заморожены слои: {freeze_list}")

    # Загрузка лучшей модели после предыдущих этапов
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path} for validation training...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Training will start from current model weights.")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Аугментация train датасета перед эпохой, если задан pipeline
        if augmentation_pipeline is not None:
            X_aug, y_aug = augment_dataset(train_X, train_y, augmentation_pipeline, device)
            train_dataset = CustomDataset(X_aug, y_aug, is_tensor=True)
        else:
            train_dataset = CustomDataset(train_X, train_y)
        val_dataset = CustomDataset(val_X, val_y)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # --- Тренировка ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'ValTrain Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # --- Валидация ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'ValTrain Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(avg_val_loss)

        print(f'\nValTrain Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.7f}, Train Acc: {train_acc:.4f}%')
        print(f'Val Loss: {avg_val_loss:.7f}, Val Acc: {val_acc:.4f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Сохраняем лучшую модель по val loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'✓ Saved best model (Val Loss: {best_val_loss:.7f}, Val Acc: {best_val_acc:.4f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epoch(s)')

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break

        print('-' * 60)

    print(f'\nValidation training completed!')
    print(f'Best Val Loss: {best_val_loss:.7f}, Best Val Acc: {best_val_acc:.4f}%')

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def full_dataset_training(
    model, full_X, full_y, criterion, optimizer, scheduler,
    num_epochs=10, device=device, clip_grad_norm=1.0, model_path='best_model.pth',
    augmentation_pipeline=None, freeze_list=None,
    patience=10
):
    """
    Дотренировка на всём тренировочном датасете с возможностью аугментации и заморозки указанных слоёв.
    freeze_list: список имён слоёв для заморозки (например, ['conv1', 'conv2']). Если None — обучаются все.
    augmentation_pipeline: если не None, применяется к данным перед каждой эпохой.
    Early stopping по train loss.
    """
    # Заморозить указанные слои, если freeze_list не None
    if freeze_list is not None:
        for name, module in model.named_children():
            if name in freeze_list:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True

        print(f"Заморожены слои: {freeze_list}")

    # Загрузка лучшей модели после предыдущих этапов
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path} for full dataset training...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: {model_path} not found. Training will start from current model weights.")

    best_train_loss = float('inf')
    best_train_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Аугментация всего датасета перед эпохой, если задан pipeline
        if augmentation_pipeline is not None:
            X_aug, y_aug = augment_dataset(full_X, full_y, augmentation_pipeline, device)
            train_dataset = CustomDataset(X_aug, y_aug, is_tensor=True)
        else:
            train_dataset = CustomDataset(full_X, full_y)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f'Full Dataset Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        scheduler.step(avg_train_loss)

        print(f'\nFull Dataset Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.7f}, Train Acc: {train_acc:.4f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # Сохранение лучшей модели
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_train_acc = train_acc
            torch.save(model.state_dict(), model_path)
            print(f'✓ Saved best model (Train Loss: {best_train_loss:.7f}, Train Acc: {best_train_acc:.4f}%)')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epoch(s)')

        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break

        print('-' * 60)

    print('\nFull dataset training completed!')


def make_submission(model, test_data, test_ids, model_path='best_model.pth', output_path='submission.csv', device=device):
    """
    Создает submission файл с предсказаниями модели
    
    Args:
        model: PyTorch модель
        test_data: тестовые данные (DataFrame)
        test_ids: ID тестовых данных
        model_path: путь к сохраненной модели
        output_path: путь для сохранения submission файла
        device: устройство для вычислений (cuda/cpu)
    
    Returns:
        DataFrame с предсказаниями
    """
    # Проверка существования модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Подготовка тестовых данных
    test_tensor = torch.FloatTensor(test_data.values).reshape(-1, 1, 50, 50)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Получение предсказаний
    predictions = []
    print("Making predictions...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # Обратное преобразование меток (из mapped labels в original labels)
    reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
    original_predictions = [reverse_label_mapping[pred] for pred in predictions]
    
    # Создание submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': original_predictions
    })
    
    # Сохранение в CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Sample predictions:\n{submission_df.head()}")
    
    return submission_df

def ensemble_submission(model_class, model_paths, test_data, test_ids, output_path='submission_ensemble.csv', device=device):
    """
    Делает ансамбль из двух (или более) моделей по .pth-файлам.
    Аргументы:
        model_class: класс вашей модели (например, CNN)
        model_paths: список путей к .pth-файлам
        test_data: DataFrame тестовых данных
        test_ids: Series с id теста
        output_path: путь для сохранения submission
        device: cuda/cpu
    """
    models = []
    for path in model_paths:
        model = model_class(num_classes=Y_mapped.nunique()).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)

    test_tensor = torch.FloatTensor(test_data.values).reshape(-1, 1, 50, 50)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Ensemble predicting'):
            inputs = batch[0].to(device)
            logits_sum = None
            for model in models:
                logits = model(inputs)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits
            all_logits.append(logits_sum.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    _, predictions = torch.max(all_logits, 1)

    reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
    original_predictions = [reverse_label_mapping[pred.item()] for pred in predictions]

    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': original_predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"Ensemble submission saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Sample predictions:\n{submission_df.head()}")
    return submission_df

def random_choice_submission(submission_paths, output_path='submission_random_choice.csv'):
    """
    Создает submission, случайно выбирая предсказания из нескольких submission файлов.
    
    Args:
        submission_paths: список путей к submission CSV файлам
        output_path: путь для сохранения итогового submission файла
    """
    submissions = [pd.read_csv(path) for path in submission_paths]
    base_df = submissions[0][['id']].copy()
    
    # Добавляем столбцы с предсказаниями из каждого submission
    for i, sub in enumerate(submissions):
        base_df[f'label_{i}'] = sub['label']
    
    np.random.seed(7)  # для воспроизводимости

    # Случайный выбор между предсказаниями
    def random_label(row):
        choices = [row[f'label_{i}'] for i in range(len(submissions))]
        return np.random.choice(choices)

    base_df['label'] = base_df.apply(random_label, axis=1)
    final_submission = base_df[['id', 'label']]
    
    final_submission.to_csv(output_path, index=False)
    print(f"Random choice submission saved to {output_path}")
    return final_submission


def stratified_kfold_train(model_class, X, y, criterion, num_folds=5, num_epochs=50, device=device, random_state=42):
    """
    Performs stratified K-fold cross-validation training.
    
    Args:
        model_class: Class of the model to be trained (e.g., CNN).
        X: Features for training.
        y: Labels for training.
        criterion: Loss function.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        num_folds: Number of folds for cross-validation.
        num_epochs: Number of epochs for training each fold.
        device: Device for training (cuda/cpu).
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training on fold {fold + 1}/{num_folds}...")

        train_X, val_X = X.iloc[train_idx], X.iloc[val_idx]
        train_y, val_y = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(num_classes=y.nunique()).to(device)
        # Создаём новый оптимизатор и шедулер для каждого fold!
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

        # Training the model
        validation_training(
            model, train_X, train_y, val_X, val_y, criterion, optimizer, scheduler,
            num_epochs=num_epochs, device=device, clip_grad_norm=1.0, model_path=f'fold_{fold + 1}_{random_state}.pth',
            augmentation_pipeline=augmentation_pipeline, freeze_list=None, patience=10
        )

        print(f"Model for fold {fold + 1} saved as fold_{fold + 1}_{random_state}.pth")


df = pd.read_csv('/kaggle/input/kyrgyz-language-hand-written-letter-kyrgyz-mnist/train.csv')
TRAIN = df.drop(columns=['label'])
Y = df['label']

test_df = pd.read_csv('/kaggle/input/kyrgyz-language-hand-written-letter-kyrgyz-mnist/test.csv')
TEST = test_df.drop(columns=['id'])
TEST_id = test_df['id']

# Add label validation and mapping
print(f'Original labels: min={Y.min()}, max={Y.max()}, unique={Y.nunique()}')
print(f'Unique label values: {sorted(Y.unique())}')

# Map labels to consecutive integers starting from 0
label_mapping = {label: idx for idx, label in enumerate(sorted(Y.unique()))}
Y_mapped = Y.map(label_mapping)
print(f'Mapped labels: min={Y_mapped.min()}, max={Y_mapped.max()}, unique={Y_mapped.nunique()}')

num_classes = Y_mapped.nunique()
print(f'Number of classes: {num_classes}')
model = SVTR_OCR(num_classes=num_classes).to(device)

s = 0
for param in model.parameters():
    s += param.numel()
print(f'Number of parameters in the model: {s}')

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # label smoothing = 0.1

stratified_kfold_train(
    model_class=SVTR_OCR, X=TRAIN, y=Y_mapped, criterion=criterion,
    num_folds=5, num_epochs=50, device=device, random_state=69
)

ensemble_submission(SVTR_OCR, ['fold_1_69.pth', 'fold_2_69.pth', 'fold_3_69.pth', 'fold_4_69.pth', 'fold_5_69.pth'], TEST, TEST_id, output_path='submission_rs_69.csv', device=device)

stratified_kfold_train(
    model_class=SVTR_OCR, X=TRAIN, y=Y_mapped, criterion=criterion,
    num_folds=5, num_epochs=50, device=device, random_state=911
)

ensemble_submission(SVTR_OCR, ['fold_1_911.pth', 'fold_2_911.pth', 'fold_3_911.pth', 'fold_4_911.pth', 'fold_5_911.pth'], TEST, TEST_id, output_path='submission_rs_911.csv', device=device)
random_choice_submission(['submission_rs_69.csv', 'submission_rs_911.csv'], output_path='final_submission.csv')




