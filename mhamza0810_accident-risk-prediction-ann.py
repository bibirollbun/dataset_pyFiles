import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
sns.set_style("dark"), plt.style.use('classic')
import sklearn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
!pip install catboost -q
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import scipy

import warnings
warnings.filterwarnings('ignore')


try:
  train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
  test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
  sample_df = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
  print("Data Files Loaded Sucessfully")
except FileNotFoundError:
  print("Error, File not Found")


cat_cols = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']
num_cols = ['num_lanes','curvature','speed_limit','num_reported_accidents']
target = 'accident_risk'
train_df['accident_risk_bins']=train_df[target]
#Binning the target to make it easier to visualize
train_df['accident_risk_bins'] = pd.cut(
    train_df[target],
    bins=[0, 0.25, 0.5, 0.75, 1],
    labels=["Q1", "Q2", "Q3", "Q4"],
    include_lowest=True
)


def quick_overview(df, name):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train_df, "train")
quick_overview(test_df , "test")

print(f"Duplicate rows (train): {train_df.duplicated().sum()}  |  (test): {test_df.duplicated().sum()}")
print("Number of missing values:")
train_df.isnull().sum()


def plot_kde(data, name, columns=None, figsize=(8, 4), fill=True, max_density=None):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    columns = data.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=figsize)
    for col in columns:
        sns.kdeplot(data[col], label=col, linewidth=2,clip=(0, None),linestyle="-.")

    if max_density is not None:
        plt.ylim(0, max_density)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("KDE PLOT")
plot_kde(train_df[target], "Accident Risk Distribution")

print("HISTOGRAM")
sns.histplot(train_df[target], kde=False)
plt.title(f"Accident Risk Distribution")
plt.xlabel("Accident Risk")
plt.ylabel("Count")

plt.show()


n_cols = 2
n_rows = math.ceil(len(cat_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    sns.countplot(data=train_df, x=col, ax=ax)
    ax.set_title(f"{col.capitalize()} Distribution", fontsize = 16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize = 14)


# Turn off any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

for col in cat_cols:
    print(train_df[col].value_counts(normalize=True).rename("proportion"))



for col in cat_cols:
    print(train_df.groupby(col)["accident_risk"].mean())


n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()


for i, col in enumerate(num_cols):
    if col == 'curvature':
        sns.histplot(train_df[col], ax=axes[i], kde=False)
    elif col == 'speed_limit':
        sns.histplot(train_df[col], ax=axes[i], kde=False, binwidth = 3)
    else:
        sns.histplot(train_df[col], ax=axes[i], kde=False, binwidth=0.3)
    axes[i].set_title(f"{col.capitalize()} Distribution")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.show()


outlier_summary = {}
for col in num_cols:
    z = np.abs(stats.zscore(train_df[col]))
    outlier_summary[col] = (z>3).sum()   # 3-Ïƒ rule

pd.Series(outlier_summary, name="#outliers (>3Ïƒ)").sort_values(ascending=False).to_frame().style.bar()


feat = train_df['num_reported_accidents']
z    = np.abs(stats.zscore(feat, nan_policy="omit"))
outlier_mask = (z > 3)

outliers= train_df.loc[outlier_mask, ['num_reported_accidents', "accident_risk"]]
print(f"Average Accidents Risk of dataset: {train_df['accident_risk'].mean()}")
print(f"Accident Risk of num_reported_accidents outliers: {outliers['accident_risk'].mean()}")


fig, axes = plt.subplots(math.ceil(len(num_cols)/3), 3, figsize=(15,4*math.ceil(len(num_cols)/3)))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x="accident_risk_bins", y=col, data=train_df, ax=axes[i], showfliers=False
)
    axes[i].set_title(f"{col} by Accident Risk Bins")


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



num_cols.append('accident_risk')
corr = train_df[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation")
plt.show()


target_corr = train_df[num_cols].corr()["accident_risk"].drop(
    "accident_risk").sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))
num_cols.remove('accident_risk')


train_df.head()


def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

train = clip(f)(train_df)
test = clip(f)(test_df)

train_df['score'] = train
test_df['score']  = test

train_df


# LOAD EXTERNAL DATA (OOF)

# LOAD ONLY BEST OOF ON PUBLIC LEADERBOARD
best_oof = pd.read_csv('/kaggle/input/autogluon-oof/AutoGluon OOF/OOF/oof_autogluon11_cv055869_lb05543.csv')
best_test = pd.read_csv('/kaggle/input/autogluon-oof/AutoGluon OOF/Test/autogluon11_cv055869_lb05543.csv')

train_df['oof_autogluon11'] = best_oof['oof_prediction']
test_df['oof_autogluon11']  = best_test['accident_risk']

train_df


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================
def create_features(df):
    # Copy dataframe
    df = df.copy()

    # Polynomial features
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2

    # Binned features
    df['curvature_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
    df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])

    # Interaction features
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']

    # Risk combinations
    df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int)
    df['weather_lighting_risk'] = (
        ((df['weather'] == 'foggy') | (df['weather'] == 'rainy')) &
        ((df['lighting'] == 'dim') | (df['lighting'] == 'night'))
    ).astype(int)

    # Derived categorical indicators
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)

    # Time-based and holiday proxies
    df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    df['is_weekend'] = df['holiday'].astype(int)

    # Safety and danger scores
    df['safety_score'] = (
        df['road_signs_present'].astype(int) * 2 +
        (df['lighting'] == 'daylight').astype(int) +
        (df['weather'] == 'clear').astype(int)
    )

    df['danger_score'] = (
        (df['curvature'] > 0.6).astype(int) +
        (df['speed_limit'] >= 60).astype(int) +
        df['is_bad_weather'] +
        df['is_night'] +
        (df['num_reported_accidents'] >= 2).astype(int)
    )

    # Ratio and intensity features
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50

    return df


create_features(train_df)
create_features(test_df)


#train_df.drop(columns='accident_risk_bins', inplace=True)
#for col in cat_cols:
    #oe = OrdinalEncoder()
    #train_df[col] = oe.fit_transform(train_df[[col]])
    #test_df[col] = oe.fit_transform(test_df[[col]])


train_df.columns


import numpy as np
import pandas as pd

#def create_domain_features(df):
    # 1. Curvature Ã— Speed
    #df["curvature_speed_interaction"] = df["curvature"] * df["speed_limit"]

    # 2. Lane Pressure
    #df["lane_pressure"] = df["speed_limit"] / (df["num_lanes"] + 1)

    # 3. Road Complexity
    #df["road_complexity"] = df["curvature"] * np.log1p(df["num_lanes"])

    # 4. Visibility Index (Lighting Ã— Weather)
    #lighting_map = {"daylight": 1.0, "dim": 0.6, "dark": 0.3}
    #weather_map = {"clear": 1.0, "rainy": 0.6, "foggy": 0.4, "snowy": 0.3, "windy": 0.8}
    #df["visibility_score"] = (
        #df["lighting"].map(lighting_map).fillna(0.5)
        #* df["weather"].map(weather_map).fillna(0.5)
    #)

    # 5. Time-of-Day Risk
    #risk_map = {"morning": 0.7, "afternoon": 0.9, "evening": 1.1, "night": 1.3}
    #df["time_risk_factor"] = df["time_of_day"].map(risk_map)

    # 6. Holiday Ã— School Interaction
    #df["holiday_school_combo"] = df["holiday"].astype(int) * df["school_season"].astype(int)

    # 7. Road Sign Ã— Public Road Interaction
    #df["signs_public_combo"] = df["road_signs_present"].astype(int) * df["public_road"].astype(int)

    # 8. Accident Density (only if column exists)
    #if "num_reported_accidents" in df.columns:
        #df["accident_density"] = df["num_reported_accidents"] / (df["num_lanes"] + 1)
    #else:
        #df["accident_density"] = 0  # placeholder for test data if missing

    # 9. Adjusted Curvature Risk
    #df["adj_curvature_risk"] = df["curvature"] / (df["visibility_score"] + 1e-5)

    # 10. Urbanization and Adjusted Speed Risk
    #road_type_map = {"urban": 1.0, "rural": 0.7, "highway": 0.9}
    #df["urban_risk"] = df["road_type"].map(road_type_map)
    #df["adjusted_speed_risk"] = df["speed_limit"] * df["urban_risk"]

    # 11. Combined Complexity Interaction
    #df["combined_complexity"] = (
        #df["curvature"]
        #* df["speed_limit"]
        #* df["lane_pressure"]
        #* df["visibility_score"]
    #)

    #return df



#train_df = create_domain_features(train_df)
#test_df = create_domain_features(test_df)



for col in cat_cols:
    oe = OrdinalEncoder()
    train_df[col] = oe.fit_transform(train_df[[col]])
    test_df[col] = oe.fit_transform(test_df[[col]])


features = [col for col in train_df.columns if col not in ['id', 'accident_risk','score_squared', 'log_score', 'score_norm_by_weather', 'accident_risk_bins']]



X = train_df[features]
y = train_df['accident_risk']
X_test_origional = test_df[features]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


train_df


y_binned = pd.qcut(y_train, q=10, labels=False, duplicates='drop')

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Convert to tensors ---
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_origional.values, dtype=torch.float32)

class TinyRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test_origional), kf.n_splits))

device = "cpu"  # Force CPU (safe for low-end laptops)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_binned)):
    print(f"Fold {fold + 1}")
    
    X_tr, y_tr = X_train_tensor[train_idx], y_train_tensor[train_idx]
    X_val, y_val = X_train_tensor[val_idx], y_train_tensor[val_idx]
    
    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    model = TinyRegressor(X_train.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, epochs=25, lr=1e-3, device=device)
    
    model.eval()
    with torch.no_grad():
        oof_preds[val_idx] = model(X_val.to(device)).cpu().numpy().ravel()
        test_preds[:, fold] = model(X_test_tensor.to(device)).cpu().numpy().ravel()

# --- Final test prediction ---
final_test_pred = test_preds.mean(axis=1)

# --- RMSE Evaluation ---
rmse = np.sqrt(np.mean((oof_preds - y_train.values) ** 2))
print(f"\nOOF RMSE: {rmse:.5f}")

oof_df = pd.DataFrame({
    'id': X_train.index,
    'oof_pred': oof_preds,
    'y_true': y_train.values
})

sub_pytorch = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': final_test_pred
})

oof_df.to_csv("oof_predictions_pytorch_light.csv", index=False)
sub_pytorch.to_csv("Submission_PyTorchRegression_wOOF.csv", index=False)

print("\nFiles saved: oof_predictions_pytorch_light.csv and Submission_PyTorch_Light.csv")

