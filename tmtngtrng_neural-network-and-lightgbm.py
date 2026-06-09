import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
import gc
import joblib
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb

from tqdm import tqdm
from itertools import combinations
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from IPython.display import clear_output
import warnings
warnings.simplefilter('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.dropna().drop_duplicates().drop(columns='id')


df.head()


df.info()


df.describe()


features = ['Age', 'Height', 'Weight', 'Duration','Heart_Rate','Body_Temp']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()


def add_advanced_features(df, features):
    df_new = df.copy()
    df_new['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df_new["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    
    # Lọc features hợp lệ
    features = [f for f in features if f in df_new.columns]

    for f in features:
        df_new[f"{f}_squared"] = df_new[f] ** 2
    for f1, f2 in combinations(features, 2):
        df_new[f"{f1}_x_{f2}"] = df_new[f1] * df_new[f2]
    for f1, f2 in combinations(features, 2):
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
    df_new["feature_mean"] = df_new[features].mean(axis=1)
    df_new["feature_std"] = df_new[features].std(axis=1)
    df_new["feature_min"] = df_new[features].min(axis=1)
    df_new["feature_max"] = df_new[features].max(axis=1)
    df_new["feature_range"] = df_new["feature_max"] - df_new["feature_min"]
    for f in features:
        if (df_new[f] > 0).all():
            df_new[f"{f}_log"] = np.log1p(df_new[f])
    
    return df_new
col = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


df = add_advanced_features(df, col)


df_inp = df.drop(columns='Calories')
df_out = df['Calories']
df_inp


def get_max_min(series: pd.Series) -> list:
    return [series.max(), series.min()]

def max_min_scalse(series: pd.Series) -> pd.Series:
    max_min = get_max_min(series)
    return series.apply(lambda x: (x - max_min[1]) / (max_min[0] - max_min[1]))

for col in df_inp.columns:
    if col != 'Sex':
        df_inp[col] = max_min_scalse(df_inp[col])


df_inp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def rmsle_loss(y_pred, y_true):
    y_pred = torch.clamp(y_pred, min=0)
    y_true = torch.clamp(y_true, min=0)
    return torch.sqrt(F.mse_loss(torch.log1p(y_pred), torch.log1p(y_true)))


class CalorieNet(nn.Module):
    def __init__(self, input_dim, dropout=0.1, use_residual=True):
            super(CalorieNet, self).__init__()
            
            self.use_residual = use_residual
            
            self.input_norm = nn.BatchNorm1d(input_dim)
            self.backbone = nn.ModuleList([
                self._make_block(input_dim, 512, dropout),
                self._make_block(512, 256, dropout),
                self._make_block(256, 128, dropout),
                self._make_block(128, 64, dropout),
            ])
            
            self.output_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),  
                nn.Linear(32, 1),
                nn.Softplus()
            )
            

            self._initialize_weights()
        
    def _make_block(self, in_features, out_features, dropout):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_norm(x)
        for block in self.backbone:
            if self.use_residual and x.shape[1] == block[0].in_features and x.shape[1] == block[0].out_features:
                x = x + block(x)
            else:
                x = block(x)
        x = self.output_head(x)
        return x


batch_size = 512

X_train, X_test, y_train, y_test = train_test_split(
    df_inp, df_out, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = CalorieNet(
        input_dim=55,  
        dropout=0.00,
        use_residual=True
    ).to(device)
no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm']
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if any(nd in name for nd in no_decay):
        no_decay_params.append(param)
    else:
        decay_params.append(param)
        
optimizer_params = [
    {'params': decay_params, 'weight_decay': 5e-4}, 
    {'params': no_decay_params, 'weight_decay': 0.0}
]
loss_fn_rmsle = rmsle_loss
best_val_loss = float("inf")
patience = 25
counter = 0
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
max_epochs = 500

optimizer = torch.optim.AdamW(
    optimizer_params,
    lr=3e-4, 
    betas=(0.9, 0.999),
    eps=1e-8,
    amsgrad=True
)

total_steps = max_epochs * len(train_loader)




scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    epochs=max_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0
)

for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0
    total_train_rmsle = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")

    for batch_X, batch_y in progress_bar:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if scaler:
            with torch.amp.autocast('cuda'):
                output = model(batch_X)
                loss = rmsle_loss(output, batch_y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch_X)
            loss = rmsle_loss(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        scheduler.step()  

        
        with torch.no_grad():
            rmsle = loss_fn_rmsle(output, batch_y)
            total_train_rmsle += rmsle.item()

        total_train_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item(),
            rmsle=rmsle.item(),
            lr=scheduler.get_last_lr()[0]
        )

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_rmsle = total_train_rmsle / len(train_loader)

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            output = model(batch_X)
            val_loss = loss_fn_rmsle(output, batch_y)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}/{max_epochs}")
    print(f"Train Loss: {avg_train_loss:.6f}, Train RMSLE: {avg_train_rmsle:.6f}")
    print(f"Val RMSLE: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.8f}")

    if avg_val_loss < best_val_loss:
        improvement = (best_val_loss - avg_val_loss) / best_val_loss * 100
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "calorie_model_best.pt")
        clear_output(wait=True)
        print(f"Saved best model (RMSLE: {best_val_loss:.6f}, improved by {improvement:.2f}%)")
        
    else:
        counter += 1
        print(f"No improvement for {counter}/{patience} epochs")
        if counter >= patience:
            
            print(f"Early stopping triggered after {patience} epochs without improvement")
            print(f"Best RMSLE: {best_val_loss:.6f}")
            break


def lgb_rmsle(preds, train_data):
    labels = train_data.get_label()
    preds = np.clip(preds, 0, None)
    rmsle = np.sqrt(np.mean(np.square(np.log1p(preds) - np.log1p(labels))))
    return 'rmsle', rmsle, False

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'None',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
        'n_estimators': 500,
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'verbosity': -1,
        'force_col_wise': True,
        'max_bin': 255,
        'feature_fraction_bynode': 0.8,
        'bagging_seed': 42,
        'feature_fraction_seed': 42,
    }

    X_temp, X_val, y_temp, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=trial.number
    )
    train_data = lgb.Dataset(X_temp, label=y_temp)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        feval=lgb_rmsle,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )

    y_pred = np.clip(model.predict(X_val), 0, None)
    rmsle = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_val))))
    return rmsle

print("Tuning LightGBM with Optuna...")

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

study.optimize(objective, n_trials=150)

clear_output(wait=True)
print(f"\nBest RMSLE: {study.best_value:.6f}")
print(f"Best Params: {study.best_params}")

final_params = study.best_params
final_params.update({
    'objective': 'regression',
    'metric': 'None',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'random_state': 42,
    'force_col_wise': True,
    'max_bin': 255,
    'feature_fraction_bynode': 0.8,
    'bagging_seed': 42,
    'feature_fraction_seed': 42,
})

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

final_model = lgb.train(
    final_params, train_data,
    valid_sets=[test_data],
    feval=lgb_rmsle,
    callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)]
)

y_pred = np.clip(final_model.predict(X_test), 0, None)
final_rmsle = np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_test))))
print(f"\nFinal RMSLE: {final_rmsle:.6f}")

joblib.dump(final_model, "lightgbm_optimized.pkl")
joblib.dump(study, "optuna_study.pkl")
print("Models saved!")


model1 = CalorieNet(
        input_dim=55,  
        dropout=0.00,
        use_residual=True
    ).to(device)
model1.load_state_dict(torch.load("calorie_model_best.pt"))
model1.eval()

model2 = joblib.load('lightgbm_optimized.pkl')

def multi_predict(model1, model2, X, alpha, device='cuda'):
    model1 = model1.to(device)
    model1.eval()
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds1 = model1(X_tensor).squeeze().cpu().numpy() 
    preds2 = np.clip(model2.predict(X), 0, None)
    blended = alpha * preds2 + (1 - alpha) * preds1
    return torch.tensor(blended, dtype=torch.float32).to(device)



best_alpha = 0.0
best_rmsle = float("inf")
preds1 = multi_predict(model1, model2, X_test, 0.0, 'cpu')
preds2 = multi_predict(model1, model2, X_test, 1.0, 'cpu')
out = torch.tensor(y_test.values, dtype=torch.float32)
for _ in np.arange(0, 100001):
    alpha = _/100000
    blended_preds = alpha * preds2 + (1 - alpha) * preds1
    rmsle = rmsle_loss(blended_preds, out).item()
    
    if rmsle < best_rmsle:
        best_rmsle = rmsle
        best_alpha = alpha
    if _ % 5000 == 0:
        print(f'Step: {_+1:06d} Best alpha: {best_alpha:.6f}, Best RMSLE: {best_rmsle:.6f}')
print(f'preds1: {rmsle_loss(preds1, out).item()}')
print(f'preds2: {rmsle_loss(preds2, out).item()}')
print(f"Best alpha: {best_alpha:.2f}, Best RMSLE: {best_rmsle:.6f}")


df_pred = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv').drop(columns='id')
col = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
df_pred = add_advanced_features(df_pred, col)
def scale_pred(name_col, series: pd.Series) -> pd.Series:
    max_min = get_max_min(df[name_col])
    return series.apply(lambda x: (x - max_min[1]) / (max_min[0] - max_min[1]))
for col in df_pred.columns:
    if col != 'Sex':
        df_pred[col] = scale_pred(col, df_pred[col])



df_pred


pred = multi_predict(model1, model2, df_pred, best_alpha, device)
pred = pred.cpu().numpy().flatten()
df_pred['Predicted_Calories'] = pred


id = range(750000,1000000)
df_submit = pd.DataFrame({
    'id': id,
    'Calories': df_pred['Predicted_Calories']
})
df_submit


df_submit.to_csv('submission.csv', index=False)

