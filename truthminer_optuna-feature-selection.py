# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    target = "label"
    n_folds = 5
    seed = 42

    run_optuna = True
    n_optuna_trials = 250


train = pd.read_parquet(CFG.train_path).reset_index()
sub  = pd.read_csv(CFG.sample_sub_path)


test  = pd.read_parquet(CFG.test_path) .reset_index()



def optimize_memory(df, verbose=True):
    """
    Optimize memory usage by downcasting numeric types where possible.
    """
    import numpy as np

    if verbose:
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f'Memory usage before optimization: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif pd.api.types.is_float_dtype(col_type):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    if verbose:
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f'Memory usage after optimization: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df


df = optimize_memory(train)


df.head()


df.shape


df = df.iloc[int(len(df) * 0.6):].reset_index(drop=True) ## only the last 40% rows



import optuna
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

df = df.drop(columns='index')
df = df.dropna()

target = "label" 
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]

# Low Variance Filter 
from sklearn.feature_selection import VarianceThreshold

var_thresh = VarianceThreshold(threshold=0.01)
X_reduced = var_thresh.fit_transform(X)
reduced_features = X.columns[var_thresh.get_support()].tolist()



X_reduced.shape



#  Remove Highly Correlated Features 
def remove_high_correlation(df, threshold=0.98):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=drop_cols), [c for c in df.columns if c not in drop_cols]


X_filtered, filtered_features = remove_high_correlation(X[reduced_features])
X = X_filtered
features = filtered_features  


X.shape


from sklearn.feature_selection import SelectKBest, f_regression

X = X_filtered
y = df["label"]

# Applying SelectKBest with ANOVA F-value
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)

# Displaying scores for each feature
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
print(feature_scores.sort_values(by='Score', ascending=False))


threshold = 1.0  
selected_features = feature_scores[feature_scores['Score'] > threshold]['Feature'].tolist()
X = X[selected_features]
X.shape


features = X.columns


import time
from lightgbm import LGBMRegressor

# === Config ===
SEED = 42
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(kf.split(X))
N_TRIALS = 60
MIN_FEATURES = 15
PENALTY = 1e-4

# === Loss Function ===
def negative_pearson(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return -corr

# === Optuna Objective ===
class FeatureSelectionOptuna:
    def __init__(self, model, features, X, y, splits, loss_fn, min_features, penalty):
        self.model = model
        self.features = features
        self.X = X
        self.y = y
        self.splits = splits
        self.loss_fn = loss_fn
        self.min_features = min_features
        self.penalty = penalty

    def __call__(self, trial):
        # Use a single binary vector instead of categorical sampling
        mask = [trial.suggest_int(f"f_{i}", 0, 1) for i in range(len(self.features))]
        selected = [f for f, m in zip(self.features, mask) if m]

        if len(selected) < self.min_features:
            return float("inf")

        total_loss = 0
        for train_idx, val_idx in self.splits:
            X_train = self.X.iloc[train_idx][selected]
            y_train = self.y.iloc[train_idx]
            X_val = self.X.iloc[val_idx][selected]
            y_val = self.y.iloc[val_idx]

            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_val)
            total_loss += self.loss_fn(y_val, preds)

        avg_loss = total_loss / len(self.splits)
        return avg_loss + self.penalty * len(selected)

# === Define Study ===
model = LGBMRegressor(n_jobs=-1, random_state=SEED)

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
start = time.time()

study.optimize(
    FeatureSelectionOptuna(
        model=model,
        features=features,
        X=X,
        y=y,
        splits=splits,
        loss_fn=negative_pearson,
        min_features=MIN_FEATURES,
        penalty=PENALTY,
    ),
    n_trials=N_TRIALS,
    show_progress_bar=True,
)

print(f"\n Total time: {time.time() - start:.2f} seconds")
best_features = [f for i, f in enumerate(features) if study.best_params[f"f_{i}"]]
print(" Best features selected:", best_features)


best_features = ['bid_qty', 'X2', 'X3', 'X9', 'X12', 'X18', 'X19', 'X21', 'X22', 'X23', 'X24', 'X25', 'X30', 'X34', 'X39', 'X40', 'X45',
                 'X51', 'X58', 'X60', 'X61', 'X62', 'X66', 'X67', 'X68', 'X69', 'X71', 'X73', 'X75', 'X77', 'X84', 'X86', 'X90', 'X92', 
                 'X93', 'X95', 'X99', 'X100', 'X103', 'X106', 'X108', 'X112', 'X117', 'X121', 'X123', 'X124', 'X127', 'X129', 'X132',
                 'X133', 'X135', 'X137', 'X138', 'X139', 'X144', 'X150', 'X151', 'X159', 'X163', 'X165', 'X167', 'X171', 'X173', 'X175',
                 'X177', 'X180', 'X187', 'X189', 'X197', 'X199', 'X201', 'X202', 'X203', 'X205', 'X206', 'X207', 'X209', 'X212', 'X214',
                 'X216', 'X219', 'X220', 'X223', 'X227', 'X230', 'X232', 'X235', 'X239', 'X250', 'X251', 'X253', 'X266', 'X267', 'X268',
                 'X271', 'X273', 'X274', 'X276', 'X278', 'X281', 'X287', 'X298', 'X304', 'X306', 'X310', 'X311', 'X317', 'X319', 'X321',
                 'X325', 'X327', 'X333', 'X334', 'X335', 'X338', 'X339', 'X340', 'X341', 'X342', 'X343', 'X346', 'X347', 'X353', 'X355',
                 'X358', 'X360', 'X361', 'X364', 'X368', 'X370', 'X376', 'X378', 'X382', 'X386', 'X389', 'X391', 'X395', 'X398', 'X400',
                 'X402', 'X406', 'X407', 'X408', 'X410', 'X415', 'X416', 'X419', 'X430', 'X431', 'X432', 'X436', 'X437', 'X443', 'X445',
                 'X446', 'X448', 'X454', 'X459', 'X460', 'X463', 'X468', 'X469', 'X471', 'X473', 'X474', 'X476', 'X477', 'X478', 'X480',
                 'X482', 'X485', 'X487', 'X495', 'X496', 'X501', 'X503', 'X504', 'X506', 'X507', 'X510', 'X511', 'X514', 'X519', 'X520',
                 'X526', 'X529', 'X532', 'X533', 'X535', 'X537', 'X540', 'X544', 'X545', 'X547', 'X549', 'X552', 'X560', 'X562', 'X564',
                 'X568', 'X569', 'X576', 'X577', 'X580', 'X581', 'X582', 'X584', 'X588', 'X589', 'X596', 'X601', 'X602', 'X603', 'X605',
                 'X606', 'X608', 'X610', 'X612', 'X614', 'X620', 'X626', 'X629', 'X630', 'X633', 'X640', 'X642', 'X646', 'X648', 'X657',
                 'X659', 'X666', 'X669', 'X670', 'X674', 'X677', 'X679', 'X684', 'X688', 'X692', 'X693', 'X694', 'X696', 'X697', 'X701',
                 'X709', 'X710', 'X713', 'X714', 'X716', 'X720','X725', 'X728', 'X733', 'X734', 'X738', 'X742', 'X744', 'X746', 'X750',
                 'X752', 'X753', 'X755', 'X756', 'X759', 'X760', 'X762', 'X768', 'X770', 'X772', 'X778']


df[best_features].shape


df = df.iloc[int(len(df) * 0.4):].reset_index(drop=True) ## only the last 40% rows


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor

# ------------------ Config ------------------ #
EARLY_PERCENTAGE = 0.4

class Config:
    FEATURES = best_features
    LABEL_COLUMN = "label"
    N_FOLDS = 5
    RANDOM_STATE = 42

# ------------------ Load and Prepare Data ------------------ #
# Ensure no 'label' in features
features = [f for f in Config.FEATURES if f != Config.LABEL_COLUMN]

# Defensive copies to avoid SettingWithCopyWarning
train_df = df[features + [Config.LABEL_COLUMN]].copy()
test_df = test[features].copy()
submission_df = sub.copy()

# Optional: optimize memory
train_df[features] = train_df[features].astype(np.float32)
train_df[Config.LABEL_COLUMN] = train_df[Config.LABEL_COLUMN].astype(np.float32)
test_df[features] = test_df[features].astype(np.float32)

X_train = train_df[features]
y_train = train_df[Config.LABEL_COLUMN]

# ------------------ Base Models ------------------ #
MODELS = [
    ("xgb", XGBRegressor(
        tree_method='hist',
        device='gpu',
        n_jobs=-1,
        learning_rate=0.015,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=10,
        reg_lambda=80,
        n_estimators=400,
        verbosity=0,
        random_state=Config.RANDOM_STATE
    )),
    ("lgb", LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        learning_rate=0.01,
        n_estimators=400,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=5,
        reg_lambda=80,
        random_state=Config.RANDOM_STATE
    )),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        n_jobs=-1,
        random_state=Config.RANDOM_STATE
    ))
]

# ------------------ Ridge Stacker ------------------ #
def negative_pearson(y_true, y_pred):
    return -pearsonr(y_true, y_pred)[0]

scorer = make_scorer(negative_pearson, greater_is_better=False)

kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

meta_train = np.zeros((len(X_train), len(MODELS)))
meta_test = np.zeros((len(test_df), len(MODELS)))

for i, (name, model) in enumerate(MODELS):
    print(f"\nTraining base model: {name}")
    fold_preds = np.zeros(len(X_train))
    fold_test = np.zeros((len(test_df), Config.N_FOLDS))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"  Fold {fold+1}/{Config.N_FOLDS}")
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        fold_preds[val_idx] = model.predict(X_train.iloc[val_idx])
        fold_test[:, fold] = model.predict(test_df)

    meta_train[:, i] = fold_preds
    meta_test[:, i] = fold_test.mean(axis=1)

# ------------------ RidgeCV Stacking ------------------ #
print("\nTraining RidgeCV stacker")
ridge = RidgeCV(alphas=np.logspace(-4, 2, 50), cv=Config.N_FOLDS, scoring=scorer)
ridge.fit(meta_train, y_train)

final_preds = ridge.predict(meta_test)
submission_df["prediction"] = final_preds
submission_df.to_csv("stacked_submission.csv", index=False)

print("\nSaved: stacked_submission.csv")
print("Ridge Coefficients:", ridge.coef_)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd

def smart_scaling(df: pd.DataFrame):
    """Scale each column with an optimal scaler based on distribution heuristics."""
    scaled_data = {}
    scalers = {}

    for col in df.columns:
        values = df[col].values.astype(np.float32)
        skewness = pd.Series(values).skew()

        if abs(skewness) > 1.5:
            scaler = RobustScaler()
        elif np.min(values) >= 0 and np.max(values) <= 1e3:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        reshaped = values.reshape(-1, 1)
        scaled = scaler.fit_transform(reshaped).ravel()
        scaled_data[col] = scaled
        scalers[col] = scaler  # Optional: if you want to transform test data later

    return pd.DataFrame(scaled_data, columns=df.columns, dtype=np.float32), scalers
    
X_top, _ = smart_scaling(df[best_features])  # Only keep the scaled DataFrame

y = df['label']


def train_autoencoder(X_top, bottleneck_dim=128, epochs=200, batch_size=2048, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_top.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = Autoencoder(input_dim=X_top.shape[1], bottleneck_dim=bottleneck_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    wait = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        epoch_loss = total_loss / len(loader.dataset)
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.6f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            wait = 0
            torch.save(model.state_dict(), "best_autoencoder.pt")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_autoencoder.pt"))
    return model.eval(), device


# Train and extract compressed features
autoencoder, device = train_autoencoder(X_top)

with torch.no_grad():
    compressed_features = autoencoder.encode(
        torch.tensor(X_top.values, dtype=torch.float32).to(device)
    ).cpu().numpy()

compressed_df = pd.DataFrame(compressed_features, columns=[f'ae_{i}' for i in range(compressed_features.shape[1])])


compressed_features = autoencoder.encode(
    torch.tensor(X_top.values, dtype=torch.float32).to(device)
).detach().cpu().numpy()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# ====================== Model ====================== #
class CryptoNetV3(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.norm_input = nn.LayerNorm(input_dim)

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3)
        )
        self.block4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Linear(64, 1)

        self._init_weights()

    def forward(self, x):
        x = self.norm_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output_layer(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# ====================== Pearson Correlation ====================== #
def pearson_corr(y_true, y_pred):
    vx = y_true - y_true.mean()
    vy = y_pred - y_pred.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()))
    return corr.item() if not torch.isnan(corr) else 0.0



from torch.utils.data import random_split, DataLoader, TensorDataset

# Make sure compressed_features is a numpy array
X_tensor = torch.tensor(compressed_features, dtype=torch.float32)

# If y_recent is a pandas Series, use `.values`; if it's already a numpy array, just use it directly
if isinstance(y, pd.Series):
    y_array = y.values
else:
    y_array = y

y_tensor = torch.tensor(y_array.reshape(-1, 1), dtype=torch.float32)

# Create dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split sizes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split dataset
train_set, val_set = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1024)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

# For reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# ====================== Improved CryptoNet ====================== #
class CryptoNetV4(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.ln0 = nn.LayerNorm(inp)
        def block(in_f, out_f, p):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.SiLU(),
                nn.Dropout(p),
                nn.LayerNorm(out_f),
            )
        self.b1 = block(inp, 512, 0.2)
        self.b2 = block(512, 256, 0.3)
        self.b3 = block(256, 128, 0.3)
        self.b4 = block(128, 64, 0.2)
        self.out = nn.Linear(64, 1)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.ln0(x)
        for b in (self.b1, self.b2, self.b3, self.b4):
            x = b(x)
        return self.out(x).squeeze(-1)


# ====================== Pearson Correlation ====================== #
def pearson_corr(y_true, y_pred):
    vx = y_true - y_true.mean()
    vy = y_pred - y_pred.mean()
    corr = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()))
    return corr.item() if not torch.isnan(corr) else 0.0

# ====================== Training Loop ====================== #
def train_model(X_train, y_train, X_val, y_val, input_dim, batch_size=1024, lr=5e-4, epochs=200, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, pin_memory=True)

    model = CryptoNetV3(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_corr = -np.inf
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).cpu()
                val_preds.append(preds)
                val_targets.append(yb)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_corr = pearson_corr(val_targets, val_preds)

        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | Val Corr: {val_corr:.5f}")

        if val_corr > best_corr:
            best_corr = val_corr
            wait = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return model



X_tensor = torch.tensor(compressed_df.values, dtype=torch.float32)
y_tensor = torch.tensor(df["label"].values, dtype=torch.float32)

# Split to train/val
val_split = 0.2
split_idx = int(len(X_tensor) * (1 - val_split))
X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

model = train_model(
    X_train, y_train, X_val, y_val,
    input_dim=X_tensor.shape[1],
    lr=3e-4,
    epochs=300,
    batch_size=1024,
    patience=10
)



## SUBMISSION
# 1. Encode the test features using the trained autoencoder
X_test  = test[best_features]
autoencoder.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    compressed_test = autoencoder.encode(test_tensor).cpu().numpy()

# 2. Create test DataLoader
test_dataset = TensorDataset(torch.tensor(compressed_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

# 3. Predict using CryptoNetV2
model.eval()
test_preds = []

with torch.no_grad():
    for (X_batch,) in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch).squeeze().cpu().numpy()
        test_preds.extend(preds)

# 4. Create submission
submission = sub
submission["prediction"] = test_preds
submission.to_csv("submission_autoencodernet.csv", index=False)

print("✅ Submission saved as: submission_autoencodernet.csv")














