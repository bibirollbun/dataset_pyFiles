import pandas as pd
import sklearn
import numpy as np
import torch
import plotly.express as px
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm


# Display all rows
pd.set_option('display.max_rows', None)

# Display all columns
pd.set_option('display.max_columns', None)

# Display full column width (disable cell truncation)
pd.set_option('display.max_colwidth', None)

# Optional: Adjust the overall display width
pd.set_option('display.width', 1000)


data_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
orig_df = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")

TARGET = 'diagnosed_diabetes'


ORIG_UNIQUE_COLS = list(set(orig_df.columns.tolist()) - set(data_df.columns.tolist())) 

SYNTH_COLS = data_df.columns.tolist()

CAT_COLS = data_df.select_dtypes(include='object').columns.tolist()

ORIG_UNIQUE_COLS.remove('diabetes_stage')

BINNING_TARGETS = [
        'age', 'bmi', 'triglycerides', 'physical_activity_minutes_per_week', 
        'cholesterol_total', 'ldl_cholesterol', 'hdl_cholesterol', 
        'systolic_bp', 'diastolic_bp', 'sleep_hours_per_day', 'diet_score'
    ]

orig_df[ORIG_UNIQUE_COLS].dtypes


from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


def fit_stat_binning(series, q=10):
    """Learn quantile bin boundaries from the training set."""
    _, bins = pd.qcut(series.dropna().rank(method='first'), q=q, retbins=True, duplicates='drop')
    return bins

def apply_stat_binning(series, bins):
    """Apply learned boundaries to any dataset."""
    return pd.cut(series, bins=bins, labels=False, include_lowest=True).astype(str)

def fit_ai_binning(X, y, col, max_depth=3):
    """Learn decision tree thresholds from the training set."""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    X_filled = X[[col]].fillna(X[col].mean())
    dt.fit(X_filled, y)
    return sorted(list(set([t for t in dt.tree_.threshold if t != -2])))

def apply_ai_binning(series, thresholds):
    """Apply learned AI thresholds to any dataset."""
    bins = [-np.inf] + thresholds + [np.inf]
    return pd.cut(series, bins=bins, labels=False).astype(str)


def domain_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6

    df_new= df.copy()

    df_new["trig_hdl_ratio"] = df_new["triglycerides"] / (df_new["hdl_cholesterol"] + eps)
    df_new["total_hdl_ratio"] = df_new["cholesterol_total"] / (df_new["hdl_cholesterol"] + eps)

    df_new["pulse_pressure"] = df_new["systolic_bp"] - df_new["diastolic_bp"]
    df_new["mean_arterial_pressure"] = (
        2 * df_new["diastolic_bp"] + df_new["systolic_bp"]
    ) / 3

    df_new["bmi_whr_interaction"] = df_new["bmi"] * df_new["waist_to_hip_ratio"]

    df_new["log_triglycerides"] = np.log1p(df_new["triglycerides"])

    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    df_new["smoking_status_int"] = df_new["smoking_status"].map(smoking_map).fillna(0)
    df_new.drop(columns="smoking_status", inplace=True)

    df_new.drop(columns=["triglycerides"], errors="ignore", inplace=True)

    return df_new

def feature_engineer_fast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast feature engineering for diabetes prediction
    """
    df_new = df.copy()

    # ----------------------------------
    # 1. PRECOMPUTE GROUPBY STATISTICS
    # ----------------------------------
    groupby_cache = {}

    for col in SYNTH_COLS:
        if col == TARGET:
            continue

        grp = orig_df.groupby(col)

        groupby_cache[col] = {
            "mean": grp[ORIG_UNIQUE_COLS + [TARGET]].mean(),
            "std": grp[ORIG_UNIQUE_COLS + [TARGET]].std(),
            "count": grp.size()
        }

    # Category groupby (multi-column)
    category_grp = orig_df.groupby(CAT_COLS)
    category_mean = category_grp[ORIG_UNIQUE_COLS + [TARGET]].mean()
    category_std = category_grp[ORIG_UNIQUE_COLS + [TARGET]].std()

    # ----------------------------------
    # 2. FEATURE CREATION
    # ----------------------------------
    for u_o in tqdm(ORIG_UNIQUE_COLS + [TARGET]):

        for col, stats in groupby_cache.items():
            # Mean
            df_new[f"orig_{col}_mean_{u_o}"] = df_new[col].map(stats["mean"][u_o])

            # Std
            df_new[f"orig_{col}_std_{u_o}"] = df_new[col].map(stats["std"][u_o])

            # Count
            df_new[f"orig_{col}_count_{u_o}"] = df_new[col].map(stats["count"])

        # Category stats (multi-column â†’ merge once per u_o)
        cat_stats = (
            category_mean[[u_o]]
            .rename(columns={u_o: f"category_col_mean_{u_o}"})
            .join(
                category_std[[u_o]].rename(columns={u_o: f"category_col_std_{u_o}"}),
                how="left"
            )
            .reset_index()
        )

        df_new = df_new.merge(cat_stats, on=CAT_COLS, how="left")

        # ----------------------------------
        # 3. FAST NA FILLING
        # ----------------------------------
        mean_fill = orig_df[u_o].mean()
        std_fill = orig_df[u_o].std()

        mean_cols = df_new.filter(like=f"mean_{u_o}").columns
        std_cols = df_new.filter(like=f"std_{u_o}").columns
        count_cols = df_new.filter(like=f"count_{u_o}").columns

        df_new.loc[:, mean_cols] = df_new.loc[:, mean_cols].fillna(mean_fill)
        df_new.loc[:, std_cols] = df_new.loc[:, std_cols].fillna(std_fill)
        df_new.loc[:, count_cols] = df_new.loc[:, count_cols].fillna(0)


    print("ðŸ”§ Engineering features (Strict Train-Fit Binning)...")

    for col in BINNING_TARGETS:
        # Statistical Binning: Fit on train, apply to all
        s_bins = fit_stat_binning(data_df[col], q=10)
        df_new[f'bin_{col}_stat'] = apply_stat_binning(df_new[col], s_bins)
        
        # AI Binning: Fit on train, apply to all
        a_thresholds = fit_ai_binning(data_df, data_df[TARGET], col)
        df_new[f'bin_{col}_ai'] = apply_ai_binning(df_new[col], a_thresholds)

    # Domain Features
    df_new = domain_features(df_new)

    return df_new

mlp_df = feature_engineer_fast(data_df)


mlp_df = mlp_df.drop(columns=mlp_df.select_dtypes(include='object').columns.tolist())
# mlp_df.dtypes


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

scaler = StandardScaler()

feature_cols = mlp_df.columns.tolist()
feature_cols.remove(TARGET)

mlp_df[feature_cols] = scaler.fit_transform(mlp_df[feature_cols])

# TEST_LIKE = 'test_like'

# mlp_df[TEST_LIKE] = 0
# mlp_df.loc[678260:, TEST_LIKE] = 1



import matplotlib.pyplot as plt

counts = mlp_df.groupby(TARGET).size()

plt.figure()
plt.bar(counts.index, counts.values)
plt.xlabel(TARGET)
plt.ylabel("Count")
plt.title("Counts per class")
plt.show()


from sklearn.utils import resample

# Separate majority and minority classes
df_majority = mlp_df[mlp_df[TARGET] == 1]
df_minority = mlp_df[mlp_df[TARGET] == 0]

# Upsample minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,                         # sample with replacement
    n_samples=len(df_majority),           # match majority class
    random_state=42
)

# Combine and shuffle
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
mlp_df = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)
# mlp_df.drop(columns=[TARGET], inplace = True)


counts = mlp_df.groupby(TARGET).size()

plt.figure()
plt.bar(counts.index, counts.values)
plt.xlabel(TARGET)
plt.ylabel("Count")
plt.title("Counts per class")
plt.show()


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array or pandas DataFrame of features
            y: numpy array or pandas Series of labels
            transform: optional callable to transform X
        """
        self.X = X.values if hasattr(X, 'values') else X
        self.y = y.values if hasattr(y, 'values') else y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]

        # Optional: apply transformation (e.g., normalization, augmentation)
        if self.transform:
            sample = self.transform(sample)

        # Convert to torch tensors
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return sample, label\
        
X = mlp_df.drop(columns=[TARGET])
y = mlp_df[TARGET]
    
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# for train_batch, val_batch in zip(train_loader, val_loader):
#     print(train_batch[0].shape)
#     break



import torch.nn as nn

class EnsembleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        # These are the shared weights the paper talks about
        self.W = nn.Parameter(torch.empty(output_dim, input_dim))

        self.R = nn.Parameter(  # R
            torch.empty(k, input_dim))
        

        self.S = nn.Parameter(  # S
            torch.empty(k, output_dim))

        self.B = nn.Parameter(torch.empty(k, output_dim))

        self.reset_parameters()

         

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.R)
        nn.init.xavier_uniform_(self.S)
        nn.init.xavier_uniform_(self.B)

    def forward(self, x):

        # >>> Equation (5) from the BatchEnsemble paper (arXiv v2).
        x = x * self.R
        x = x @ self.W.T
        x = x * self.S
        # <<<

        x = x + self.B
        
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super().__init__()
        self.W = nn.Parameter(torch.empty(k, input_dim, output_dim))
        self.B = nn.Parameter(torch.empty(k, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.B)


    def forward(self, x):
        x = x.transpose(0, 1) # (B, k, D) -> (k, B, D)
        x = x @ self.W # (k, B, D) x (k, B, O)
        x = x.transpose(0, 1) # (B, k, O)
        x = x + self.B
        return x


class EnsembleView(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        x = x.unsqueeze(-2).expand(-1, self.k, -1)
        return x


k = 16
d_in = X.shape[-1]
d_h = 128
d = 128
dropout = 0.2
d_out = 1

model = nn.Sequential(
    EnsembleView(k=k),

    # >>> MLPBackboneBatchEnsemble(n_blocks=2)
    EnsembleBlock(
        d_in, d_h, k=k,
    ),
    nn.ReLU(),
    nn.Dropout(dropout),

    EnsembleBlock(
        d_h, d, k=k
    ),
    nn.ReLU(),
    nn.Dropout(dropout),
    # # <<<

    LinearBlock(d, d_out, k=k),
)

class BasicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=1):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes
            output_dim: Number of outputs (1 for binary/regression)
        """
        super(BasicMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # Hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.3))  # optional dropout
            in_dim = h_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.my_layer = EnsembleBlock(3,2, 3)
    
    def forward(self, x):
        return self.model(x)



from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Use GPU if Metal backend available (Apple M1/M2), otherwise CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

input_dim = X_train.shape[1]  # number of features
# model = BasicMLP(input_dim=input_dim, hidden_dims=[128, 64], output_dim=1)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # includes sigmoid
optimizer = AdamW(model.parameters(), lr=0.0001)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',   # âœ… correct
    factor=0.5,
    patience=10000,
    min_lr=1e-7
)


def validate(model, val_loader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X).squeeze(1)
            probs = torch.sigmoid(logits)
            probs = probs.mean(dim= 1).squeeze(-1)

            all_probs.append(probs.cpu())
            all_labels.append(batch_y.cpu())
    model.train()
    return roc_auc_score(
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy()
    )


# Training loop
def train(model, train_loader, val_loader, optimizer, criterion, epoch = 40):
    step = 0
    model.train()
    for epoch in range(epoch):
        epoch_loss = 0
        for batch_X, batch_y in tqdm(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            # print(batch_y.shape)
            batch_y = batch_y.unsqueeze(-1).unsqueeze(-1).expand(-1, k, -1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        

        val_auc = validate(model, val_loader)
        scheduler.step(val_auc)

        train_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Val AUC: {val_auc:.4f}"
        )


train(model, train_loader=train_loader, val_loader=val_loader, optimizer= optimizer, criterion= criterion, epoch= 40)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
index = test_df.index

test_df = feature_engineer_fast(test_df)

test_df = test_df.drop(columns=test_df.select_dtypes(include='object').columns.tolist())
   
X_test = scaler.transform(test_df)


test_dataset = CustomDataset(X_test, y_train[:X_test.shape[0]])
test_loader = DataLoader(test_dataset, batch_size=64)


model.eval()
all_probs = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X).squeeze(1)
        probs = torch.sigmoid(logits)
        probs = probs.mean(dim= 1).squeeze(-1)

        all_probs.append(probs.cpu())

nn_probs = torch.cat(all_probs).numpy()
nn_probs


# import pickle
# with open('probs_weighted2.pkl', 'rb') as f:
#     xgb_cat_probs = pickle.load(f)

# probs = pd.read_csv("./data/submission.csv", index_col='id')
 
import seaborn as sns
import matplotlib.pyplot as plt

norm = 30

# xgb_probs = xgb_cat_probs['xgb'][:, 1] / norm
# cat_probs = xgb_cat_probs['cat'][:, 1] / norm

# diabetes_prob = ( xgb_probs + cat_probs +  nn_probs) / (3)

plt.figure(figsize=(10, 6))

# sns.kdeplot(xgb_probs, label='LGBM', linewidth=2)
# sns.kdeplot(cat_probs, label='CatBoost', linewidth=2)
sns.kdeplot(nn_probs, label='NN', linewidth=2)
# sns.kdeplot(old_probs, label='Blend', linewidth=2)

plt.xlabel('Predicted Probability (Class 1)')
plt.ylabel('Density')
plt.title('KDE of Individual Model Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.show()



predictions_df = pd.DataFrame({
    'id': index,
    'diagnosed_diabetes': nn_probs
})
predictions_df.to_csv('./submission.csv', index=False)




