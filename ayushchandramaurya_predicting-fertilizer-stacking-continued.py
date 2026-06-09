import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)



# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom PyTorch MLP model
class TorchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TorchMLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.15)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        return self.model(x)
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            x_array = x.values if isinstance(x, pd.DataFrame) else x
            x_tensor = torch.FloatTensor(x_array).to(device)
            return self.softmax(self.forward(x_tensor)).cpu().numpy()


# Training function for TorchMLP
def train_torch_mlp(model, x_train, y_train, x_valid, y_valid, max_epochs=200, patience=10, batch_size=1024):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_dataset = TensorDataset(torch.FloatTensor(x_train.values).to(device), torch.LongTensor(y_train.values).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_tensor = torch.FloatTensor(x_valid.values).to(device)
    valid_labels = torch.LongTensor(y_valid.values).to(device)
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            valid_outputs = model(valid_tensor)
            valid_loss = criterion(valid_outputs, valid_labels).item()
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model


# Load existing stacking predictions
stacking_oof = np.load('/kaggle/input/predicting-fertilizer-name-stacking-ensemble/results/stacking_oof.npy')
stacking_test = np.load('/kaggle/input/predicting-fertilizer-name-stacking-ensemble/results/stacking_test.npy')

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# Augment original dataset
original_copy = original.copy()
for _ in range(6):
    original = pd.concat([original, original_copy], axis=0)

# Feature engineering
numerical_features = [col for col in train.select_dtypes(include=['int64', 'float64']).columns 
                      if col != 'id']
for df in [train, test, original]:
    for col in numerical_features:
        df[f'{col}_Binned'] = df[col].astype(str).astype('category')
    df = df.rename(columns={'Temparature': 'Temperature'})
    for col in df.columns:
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int8')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')

# Encode categorical variables
cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns 
            if col != "Fertilizer Name"]
for col in cat_cols:
    label_enc = LabelEncoder()
    train[f'{col}_encoded'] = label_enc.fit_transform(train[col])
    original[f'{col}_encoded'] = label_enc.transform(original[col])
    test[f'{col}_encoded'] = label_enc.transform(test[col])

target_enc = LabelEncoder()
train["Fertilizer Name"] = target_enc.fit_transform(train["Fertilizer Name"])
original["Fertilizer Name"] = target_enc.transform(original["Fertilizer Name"])

# Set encoded categorical columns as category type
encoded_cat_cols = [f'{col}_encoded' for col in cat_cols]
for col in encoded_cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')
    original[col] = original[col].astype('category')


# Prepare data
X = train.drop(columns=["id", "Fertilizer Name"] + cat_cols)
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"] + cat_cols)
X_original = original.drop(columns=["Fertilizer Name"] + cat_cols)
y_original = original["Fertilizer Name"]

# Scale features for TorchMLP and Logistic Regression
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
X_original_scaled = pd.DataFrame(scaler.transform(X_original), columns=X_original.columns, index=X_original.index)

# Create polynomial features for Logistic Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
numerical_features = [col for col in X.columns if col not in encoded_cat_cols]
X_poly = pd.DataFrame(poly.fit_transform(X_scaled[numerical_features]), 
                      columns=poly.get_feature_names_out(numerical_features), 
                      index=X_scaled.index)
X_scaled_poly = pd.concat([X_poly, X_scaled[encoded_cat_cols]], axis=1)
X_test_poly = pd.DataFrame(poly.transform(X_test_scaled[numerical_features]), 
                           columns=poly.get_feature_names_out(numerical_features), 
                           index=X_test_scaled.index)
X_test_scaled_poly = pd.concat([X_test_poly, X_test_scaled[encoded_cat_cols]], axis=1)
X_original_poly = pd.DataFrame(poly.transform(X_original_scaled[numerical_features]), 
                               columns=poly.get_feature_names_out(numerical_features), 
                               index=X_original_scaled.index)
X_original_scaled_poly = pd.concat([X_original_poly, X_original_scaled[encoded_cat_cols]], axis=1)



model_configs = {
    'lr': {
        'model': LogisticRegression,
        'params': {
            'C': 1e-2,
            'max_iter': 3000,
            'random_state': 0
        }
    },
    'cat': {
        'model': CatBoostClassifier,
        'params': {
            'loss_function': 'MultiClass',
            'depth': 8,
            'learning_rate': 0.027,
            'iterations': 6000,
            'early_stopping_rounds': 100,
            'cat_features': encoded_cat_cols,
            'task_type': 'GPU',
            'random_seed': 42,
            'verbose': 0
        }
    },
    'mlp': {
        'model': TorchMLP,
        'params': {
            'input_size': X_scaled.shape[1],
            'hidden_sizes': [100, 50],
            'output_size': len(np.unique(y))
        }
    }
}


# MAP@3 metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



# Train new base models
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_new = {name: np.zeros((len(X), y.nunique())) for name in model_configs}
test_preds_new = {name: np.zeros((len(X_test), y.nunique())) for name in model_configs}
map3_scores_new = {name: [] for name in model_configs}

for name, config in model_configs.items():
    print(f"\nTraining {name.upper()}...")
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/5")
        x_train = X.iloc[train_idx] if name == 'cat' else X_scaled_poly.iloc[train_idx] if name == 'lr' else X_scaled.iloc[train_idx]
        x_valid = X.iloc[valid_idx] if name == 'cat' else X_scaled_poly.iloc[valid_idx] if name == 'lr' else X_scaled.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Concatenate with original data
        x_train = pd.concat([x_train, X_original if name == 'cat' else X_original_scaled_poly if name == 'lr' else X_original_scaled], axis=0, ignore_index=True)
        y_train = pd.concat([y_train, y_original], axis=0, ignore_index=True)
        
        if name == 'cat':
            model = config['model'](**config['params'])
            model.fit(
                x_train, y_train,
                eval_set=(x_valid, y_valid)
            )
        elif name == 'mlp':
            model = config['model'](**config['params'])
            model = train_torch_mlp(model, x_train, y_train, x_valid, y_valid)
        else:  # lr
            model = config['model'](**config['params'])
            model.fit(x_train, y_train)
        
        # Predict with appropriate input
        pred_input = x_valid if name == 'cat' else x_valid.values
        oof_preds_new[name][valid_idx] = model.predict_proba(pred_input)
        test_pred_input = X_test if name == 'cat' else X_test_scaled_poly if name == 'lr' else X_test_scaled
        test_preds_new[name] += model.predict_proba(test_pred_input) / 5
        
        top_3_preds = np.argsort(oof_preds_new[name][valid_idx], axis=1)[:, -3:][:, ::-1]
        actual = [[label] for label in y_valid]
        map3_score = mapk(actual, top_3_preds)
        map3_scores_new[name].append(map3_score)
        print(f"âœ… {name.upper()} Fold {fold + 1}: MAP@3 Score: {map3_score:.5f}")
    
    print(f"ðŸŽ¯ Average {name.upper()} MAP@3 Score: {np.mean(map3_scores_new[name]):.5f}")

# Combine existing and new OOF/test predictions
stacking_train_new = np.hstack([stacking_oof] + [oof_preds_new[name] for name in oof_preds_new])
stacking_test_new = np.hstack([stacking_test] + [test_preds_new[name] for name in oof_preds_new])

# Meta-model (XGBoost)
meta_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y)),
    learning_rate=0.03,
    max_depth=8,
    n_estimators=10000,
    subsample=0.8,
    colsample_bytree=0.45,
    random_state=42,
    eval_metric='mlogloss',
    device='cuda',
    tree_method='hist',
    early_stopping_rounds=100
)

print("\nTraining Stacking Ensemble with XGBoost Meta-Model...")
final_oof_new = np.zeros((len(y), len(np.unique(y))))
final_test_new = np.zeros((len(X_test), len(np.unique(y))))
ensemble_scores_new = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(stacking_train_new, y)):
    x_train, x_valid = stacking_train_new[train_idx], stacking_train_new[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    meta_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False
    )
    
    final_oof_new[valid_idx] = meta_model.predict_proba(x_valid)
    final_test_new += meta_model.predict_proba(stacking_test_new) / 5
    
    top_3_preds = np.argsort(final_oof_new[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    ensemble_scores_new.append(map3_score)
    print(f"âœ… Ensemble Fold {fold + 1}: MAP@3 Score: {map3_score:.5f}")

print(f"ðŸŽ¯ Average Ensemble MAP@3 Score: {np.mean(ensemble_scores_new):.5f}")

# Optimize blending weights
weights_list = [(w, 1-w) for w in np.arange(0.05, 1.0, 0.05)]
best_weights = [0.6, 0.4]
best_map3 = np.mean(ensemble_scores_new)
print("\nOptimizing blending weights...")
for weights in weights_list:
    blended_oof = weights[0] * final_oof_new + weights[1] * stacking_oof
    top_3_preds = np.argsort(blended_oof, axis=1)[:, -3:][:, ::-1]
    map3 = mapk([[label] for label in y], top_3_preds)
    print(f"Weights {weights}: MAP@3 {map3:.5f}")
    if map3 > best_map3:
        best_map3 = map3
        best_weights = weights

print(f"Best weights: {best_weights}, MAP@3: {best_map3:.5f}")

# Blend predictions with best weights
final_test_blended = best_weights[0] * final_test_new + best_weights[1] * stacking_test

# Save new results
output_dir = 'results_new'
os.makedirs(output_dir, exist_ok=True)

np.save(f'{output_dir}/stacking_oof_new.npy', final_oof_new)
np.save(f'{output_dir}/stacking_test_new.npy', final_test_new)
np.save(f'{output_dir}/stacking_test_blended.npy', final_test_blended)
for name in oof_preds_new:
    np.save(f'{output_dir}/{name}_oof.npy', oof_preds_new[name])
    np.save(f'{output_dir}/{name}_test.npy', test_preds_new[name])

# Prepare submission with blended predictions
top_3_preds = np.argsort(final_test_blended, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission1 = pd.DataFrame({
    'id': submission['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission1.to_csv('my_submission.csv', index=False)

# Save scores
with open(f'{output_dir}/scores.txt', 'w') as f:
    for name, scores in map3_scores_new.items():
        f.write(f"{name.upper()} MAP@3 Scores: {scores}\n")
        f.write(f"{name.upper()} Average MAP@3: {np.mean(scores):.5f}\n")
    f.write(f"Ensemble MAP@3 Scores: {ensemble_scores_new}\n")
    f.write(f"Ensemble Average MAP@3: {np.mean(ensemble_scores_new):.5f}\n")
    f.write(f"Best Blending Weights: {best_weights}\n")
    f.write(f"Best Blended MAP@3: {best_map3:.5f}\n")


print(submission1.head())


df1 = pd.read_csv("/kaggle/input/ensemble-fertilizer/submission.csv")
all_preds = pd.DataFrame({
    'model1': df1['Fertilizer Name'],
    'model2': submission1['Fertilizer Name'],

})

model_weights = {
    'model1': 0.51,   
    'model2': 0.49
}

def weighted_mode(row, weights):
    """Calculate weighted mode for ensemble"""
    votes = {}
    for col, weight in weights.items():
        pred = row[col]
        if pred in votes:
            votes[pred] += weight
        else:
            votes[pred] = weight
    return max(votes, key=votes.get)


submission['Fertilizer Name'] = all_preds.apply(lambda row: weighted_mode(row, model_weights), axis=1)
submission.to_csv("submission.csv", index=False)
submission.head()




