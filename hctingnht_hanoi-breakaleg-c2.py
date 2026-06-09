# %%
# !pip install optuna sentence_transformers 
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
import optuna
from sentence_transformers import SentenceTransformer
import warnings
import torch
import gc  # Add garbage collector
warnings.filterwarnings('ignore')

# Load data (Assuming train.csv and test.csv are available, e.g., in /content/)
train_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')

# *** MÔ PHỎNG DỮ LIỆU ĐỂ CHẠY TRÊN COLAB (VÌ DỮ LIỆU GỐC KHÔNG CÓ) ***
# Nếu bạn đã tải dữ liệu lên Colab, hãy xóa đoạn mô phỏng này và bỏ comment 2 dòng trên
# data_size = 2000
# test_size = 250
# train_df = pd.DataFrame({
#     'text': ['prompt ' + str(i) + (' jailbreak' if i % 10 == 0 else ' safe') for i in range(data_size)],
#     'label': ['jailbreak' if i % 10 == 0 else 'safe' for i in range(data_size)]
# })
# test_df = pd.DataFrame({
#     'Id': range(test_size),
#     'text': ['test prompt ' + str(i) for i in range(test_size)]
# })
# *** KẾT THÚC MÔ PHỎNG DỮ LIỆU ***

# Prepare data
train_df['text'] = train_df['text'].fillna('').astype(str)
test_df['text'] = test_df['text'].fillna('').astype(str)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df['text'], 
    train_df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=train_df['label']
)

# Load Qwen3 embedding model with memory optimizations
print("Loading Qwen3 embedding model...")
# Use quantization to reduce memory usage
embedding_model = SentenceTransformer(
    "/kaggle/input/qwen-3-embedding/transformers/4b/1",
    device='cuda',
    model_kwargs={'torch_dtype': torch.float16}  # Use half precision
)

# Move model to float16 and GPU (if available)
embedding_model = embedding_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
embedding_model = embedding_model.half()

# Function to get embeddings with memory optimization
def get_embeddings(texts, batch_size=8):  # Reduced batch size to prevent OOM
    # Process in smaller batches to reduce memory usage
    all_embeddings = []
    texts_list = texts.values.tolist()
    
    for i in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[i:i+batch_size]
        batch_embeddings = embedding_model.encode(
            batch_texts, 
            batch_size=1,  # Process one at a time within the batch
            show_progress_bar=True,
            convert_to_numpy=True  # Keep as numpy to save memory
        )
        all_embeddings.append(batch_embeddings)
        
        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    return np.vstack(all_embeddings)

# Get embeddings for train, validation, and test sets
print("Getting embeddings for training data...")
X_train_embeddings = get_embeddings(X_train)
print("Getting embeddings for validation data...")
X_val_embeddings = get_embeddings(X_val)
print("Getting embeddings for test data...")
X_test_embeddings = get_embeddings(test_df['text'])

# Get the index of the 'jailbreak' class
jailbreak_class_index = list(np.unique(y_train)).index('jailbreak')

# Define objective function for Optuna with XGBoost
def objective_xgb(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 0.01, 10.0),
        'alpha': trial.suggest_float('alpha', 0.01, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',  # Use histogram-based algorithm which is more memory efficient
        'max_bin': 256,  # Limit bin count to reduce memory
    }
    
    # Convert labels to binary (1 for 'jailbreak', 0 for others)
    y_train_binary = (y_train == 'jailbreak').astype(int)
    y_val_binary = (y_val == 'jailbreak').astype(int)
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train_embeddings, y_train_binary)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val_embeddings)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_val_binary, y_pred_proba)
    
    # Clear memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return auc

# Define objective function for Optuna with SVC
def objective_svc(trial):
    param = {
        'C': trial.suggest_float('C', 0.1, 10.0),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        # Tham số này là CỐ ĐỊNH, nên nó không có trong study_svc.best_params
        'probability': True, 
        'random_state': 42,
        'cache_size': 200,  # Limit cache size to reduce memory usage
    }
    
    # Convert labels to binary (1 for 'jailbreak', 0 for others)
    y_train_binary = (y_train == 'jailbreak').astype(int)
    y_val_binary = (y_val == 'jailbreak').astype(int)
    
    # model có thể tính predict_proba() vì probability=True
    model = SVC(**param)
    model.fit(X_train_embeddings, y_train_binary)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val_embeddings)[:, 1]
    
    # Calculate AUC
    auc = roc_auc_score(y_val_binary, y_pred_proba)
    
    # Clear memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return auc

# Optimize XGBoost with Optuna
print("Optimizing XGBoost with Optuna...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=3, timeout=3600) # Giảm n_trials để chạy nhanh hơn

print("Best XGBoost parameters:", study_xgb.best_params)
print("Best XGBoost AUC:", study_xgb.best_value)

# Optimize SVC with Optuna
print("Optimizing SVC with Optuna...")
study_svc = optuna.create_study(direction='maximize')
study_svc.optimize(objective_svc, n_trials=3, timeout=3600) # Giảm n_trials để chạy nhanh hơn

print("Best SVC parameters:", study_svc.best_params)
print("Best SVC AUC:", study_svc.best_value)

# Select the best model based on validation performance
if study_xgb.best_value > study_svc.best_value:
    print("XGBoost performed better. Using XGBoost for final prediction.")
    best_params = study_xgb.best_params
    # Thêm các tham số cố định nếu cần
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['tree_method'] = 'hist'  # Use histogram-based algorithm
    best_params['max_bin'] = 256  # Limit bin count to reduce memory
    final_model = xgb.XGBClassifier(**best_params)
else:
    print("SVC performed better. Using SVC for final prediction.")
    best_params = study_svc.best_params.copy() # Sử dụng copy để tránh thay đổi dictionary của Optuna
    
    # ====================================================================================
    # *** BẮT ĐẦU PHẦN SỬA LỖI ĐỂ KHẮC PHỤC AttributeError: This 'SVC' has no attribute 'predict_proba' ***
    # Lý do: 'probability': True KHÔNG CÓ TRONG best_params của Optuna vì nó được hardcode trong objective.
    best_params['probability'] = True
    best_params['random_state'] = 42
    best_params['cache_size'] = 200  # Limit cache size to reduce memory usage
    # ====================================================================================
    
    final_model = SVC(**best_params)

# ====================================================================================
# PHẦN MÃ ĐƯỢC CUNG CẤP TRONG YÊU CẦU ĐÃ ĐƯỢC SỬA LỖI (VÌ final_model ĐÃ ĐƯỢC TẠO LẠI BÊN TRÊN)
# ====================================================================================

# Train the best model on the full training data
print("Training the best model on the full training data...")
y_train_full = (train_df['label'] == 'jailbreak').astype(int)
X_train_full_embeddings = get_embeddings(train_df['text'])
final_model.fit(X_train_full_embeddings, y_train_full)

# Make predictions on the test set
print("Making predictions on the test set...")
if isinstance(final_model, xgb.XGBClassifier):
    probabilities = final_model.predict_proba(X_test_embeddings)[:, 1]
else:
    # For SVC, we need to get the index of the positive class
    # Đoạn này hiện tại đã chạy đúng vì final_model là SVC(probability=True)
    positive_class_index = list(final_model.classes_).index(1)
    probabilities = final_model.predict_proba(X_test_embeddings)[:, positive_class_index]
    
# Clear memory after predictions
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Create submission file
submission_output = pd.DataFrame({'Id': test_df['Id'], 'TARGET': probabilities})
submission_output.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission_output.head())

