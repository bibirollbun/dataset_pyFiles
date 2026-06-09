import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 

import gc
gc.enable()

PATH = '../input/'


application_train = pd.read_csv(PATH+'application_train.csv')

y = application_train['TARGET']
X = application_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)


gc.collect()


for c in X.columns:
    col_type = X[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[c] = X[c].astype('category')


X.info()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)


import lightgbm as lgb
from lightgbm import early_stopping

# Параметры для обучения с ранней остановкой
fit_params = {
    "eval_set": [(X_test, y_test)],     # Валидационный набор
    "eval_names": ["valid"],
    "eval_metric": "auc",               # Метрика для оценки
    "callbacks": [early_stopping(10)],  # Ранняя остановка после 10 итераций без улучшения
}

# Инициализация классификатора
clf = lgb.LGBMClassifier(
    num_leaves=15,
    max_depth=-1,
    random_state=314,
    verbose=-1,                 # Вывод логов отключен
    n_jobs=4,
    n_estimators=1000,
    colsample_bytree=0.9,
    subsample=0.9,
    learning_rate=0.1
)

# Обучение модели
clf.fit(X_train, y_train, **fit_params)


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])


roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# --- 1. Data Loading and Preparation ---
# Assuming X is your features DataFrame and y is target Series
y = application_train['TARGET']
X = application_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# --- 2. Automatic Column Type Detection ---
def get_column_types(df):
    """Automatically detect numeric and categorical columns"""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Check for unprocessed columns
    used = set(num_cols + cat_cols)
    unused = [c for c in df.columns if c not in used]
    if unused:
        print(f"Warning: Unprocessed columns detected: {unused}")
    
    return num_cols, cat_cols

num_cols, cat_cols = get_column_types(X_train)

# --- 3. Data Preprocessing Pipeline ---
class DataPreprocessor:
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        # Numeric preprocessing
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        # Categorical preprocessing
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def fit_transform(self, X):
        # Numeric features
        X_num = self.num_imputer.fit_transform(X[self.num_cols])
        X_num = self.scaler.fit_transform(X_num)
        
        # Categorical features
        X_cat = self.cat_imputer.fit_transform(X[self.cat_cols])
        X_cat = self.ohe.fit_transform(X_cat)
        
        return np.hstack([X_num, X_cat])
    
    def transform(self, X):
        X_num = self.num_imputer.transform(X[self.num_cols])
        X_num = self.scaler.transform(X_num)
        
        X_cat = self.cat_imputer.transform(X[self.cat_cols])
        X_cat = self.ohe.transform(X_cat)
        
        return np.hstack([X_num, X_cat])

# Initialize and apply preprocessing
preprocessor = DataPreprocessor(num_cols, cat_cols)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- 4. BERT-style Encoder Model ---
class BertStyleEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, input_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructions = self.decoder(embeddings)
        return embeddings, reconstructions, torch.sigmoid(self.classifier(embeddings))

# --- 5. Masking Function ---
def mask_data(data, mask_prob=0.15):
    mask = torch.rand_like(data) < mask_prob
    masked_data = data.clone()
    masked_data[mask] = 0  # Simple masking, can be randomized
    return masked_data, mask

# --- 6. Training Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = X_train_processed.shape[1]
model = BertStyleEncoder(input_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
cls_criterion = nn.BCELoss()
rec_criterion = nn.MSELoss()

# Convert data to tensors
X_train_tensor = torch.FloatTensor(X_train_processed).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
X_test_tensor = torch.FloatTensor(X_test_processed).to(device)

# --- 7. Training Loop ---
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    # BERT-style masking
    masked_x, mask = mask_data(X_train_tensor)
    
    # Forward pass
    embeddings, reconstructions, predictions = model(masked_x)
    
    # Loss calculation
    cls_loss = cls_criterion(predictions, y_train_tensor)
    rec_loss = rec_criterion(reconstructions[mask], X_train_tensor[mask])
    total_loss = cls_loss + rec_loss * 2.0  # Weighted reconstruction loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    # Validation monitoring
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            _, _, val_preds = model(X_test_tensor)
            val_auc = roc_auc_score(y_test, val_preds.cpu().numpy())
        print(f'Epoch {epoch:3d} | Total Loss: {total_loss.item():.4f} | Val AUC: {val_auc:.4f}')




# --- 8. Generate Embeddings ---
model.eval()
with torch.no_grad():
    train_embeddings = model.encoder(X_train_tensor).cpu().numpy()
    test_embeddings = model.encoder(X_test_tensor).cpu().numpy()

# --- 9. Prepare Enhanced Dataset ---
X_train_enhanced = np.hstack([X_train_processed, train_embeddings])
X_test_enhanced = np.hstack([X_test_processed, test_embeddings])

# --- 10. LightGBM Training ---
lgb_model = lgb.LGBMClassifier(
    num_leaves=15,
    max_depth=-1,
    random_state=314,
    verbose=-1,                 # Вывод логов отключен
    n_jobs=4,
    n_estimators=1000,
    colsample_bytree=0.9,
    subsample=0.9,
    learning_rate=0.1
)

lgb_model.fit(
    X_train_enhanced, y_train,
    eval_set=[(X_test_enhanced, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100)]
)

# --- 11. Final Evaluation ---
test_probs = lgb_model.predict_proba(X_test_enhanced)[:, 1]
final_auc = roc_auc_score(y_test, test_probs)
print(f'\nFinal Test AUC with Enhanced Features: {final_auc:.4f}')


# --- 8. Generate Embeddings ---
model.eval()
with torch.no_grad():
    train_embeddings = model.encoder(X_train_tensor).cpu().numpy()
    test_embeddings = model.encoder(X_test_tensor).cpu().numpy()


# --- 10. LightGBM Training ---
lgb_model = lgb.LGBMClassifier(
    num_leaves=15,
    max_depth=-1,
    random_state=314,
    verbose=-1,                 # Вывод логов отключен
    n_jobs=4,
    n_estimators=1000,
    colsample_bytree=0.9,
    subsample=0.9,
    learning_rate=0.1
)

lgb_model.fit(
    train_embeddings, y_train,
    eval_set=[(test_embeddings, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100)]
)

# --- 11. Final Evaluation ---
test_probs = lgb_model.predict_proba(test_embeddings)[:, 1]
final_auc = roc_auc_score(y_test, test_probs)
print(f'\nFinal Test AUC with Enhanced Features: {final_auc:.4f}')





import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset

class DataPreprocessor:
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
    
    def fit_transform(self, X):
        X_num = self.num_imputer.fit_transform(X[self.num_cols])
        X_num = self.scaler.fit_transform(X_num)
        
        X_cat = self.cat_imputer.fit_transform(X[self.cat_cols])
        X_cat = self.ohe.fit_transform(X_cat)
        
        return np.hstack([X_num, X_cat])
    
    def transform(self, X):
        X_num = self.num_imputer.transform(X[self.num_cols])
        X_num = self.scaler.transform(X_num)
        
        X_cat = self.cat_imputer.transform(X[self.cat_cols])
        X_cat = self.ohe.transform(X_cat)
        
        return np.hstack([X_num, X_cat])

class LightweightAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Уменьшенные проекции
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Линейные проекции
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Масштабированное скалярное произведение
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Применение внимания
        out = torch.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        
        return self.norm(out + x)

class MemoryEfficientEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            
            LightweightAttention(64, num_heads=2),  # Уменьшенный attention
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            
            nn.Linear(128, input_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructions = self.decoder(embeddings)
        logits = self.classifier(embeddings)
        return embeddings, reconstructions, torch.sigmoid(logits)

def train_model(X_train, y_train, X_val, y_val, input_dim, device='cuda'):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                                 torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    model = AdvancedEncoder(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCELoss()
    
    best_auc = 0
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            embeddings, reconstructions, preds = model(batch_x)
            
            # Вычисляем только classification и reconstruction loss
            cls_loss = criterion(preds, batch_y)
            rec_loss = F.mse_loss(reconstructions, batch_x)
            loss = cls_loss + rec_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Валидация
        model.eval()
        with torch.no_grad():
            val_tensor = torch.FloatTensor(X_val).to(device)
            _, _, val_preds = model(val_tensor)
            val_auc = roc_auc_score(y_val, val_preds.cpu().numpy())
        
        print(f'Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}')
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def full_pipeline(X, y):
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = DataPreprocessor(num_cols, cat_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(X_train_processed, y_train.values, 
                       X_test_processed, y_test.values,
                       input_dim=X_train_processed.shape[1],
                       device=device)
    
    # Генерация эмбеддингов
    model.eval()
    with torch.no_grad():
        train_emb = model.encoder(torch.FloatTensor(X_train_processed).to(device)).cpu().numpy()
        test_emb = model.encoder(torch.FloatTensor(X_test_processed).to(device)).cpu().numpy()
    
    X_train_final = np.hstack([X_train_processed, train_emb])
    X_test_final = np.hstack([X_test_processed, test_emb])
    
    lgb_model = lgb.LGBMClassifier(
        num_leaves=127,
        learning_rate=0.05,
        n_estimators=2000,
        random_state=42
    )
    
    lgb_model.fit(
        X_train_final, y_train,
        eval_set=[(X_test_final, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(20)]
    )
    
    test_probs = lgb_model.predict_proba(X_test_final)[:, 1]
    final_auc = roc_auc_score(y_test, test_probs)
    print(f'Final Test AUC: {final_auc:.4f}')
    
    return model, lgb_model


nn_model, lgb_model = full_pipeline(X, y)





