import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split


TRAIN_DATASET_PATH = "/kaggle/input/playground-series-s5e4/train.csv"
TEST_DATASET_PATH = "/kaggle/input/playground-series-s5e4/test.csv"
SUBMISSION = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
)
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
)
warnings.filterwarnings(
    action='ignore',
    category=RuntimeWarning,
)


rmse = lambda y_preds, y_true: np.sqrt(mean_squared_error(y_true, y_preds))


dataset = pd.read_csv(TRAIN_DATASET_PATH)
test_dataset = pd.read_csv(TEST_DATASET_PATH)


dataset.head()


dataset.info()


dataset.describe()


dataset.describe(include=["O"])


print("Train Data:\n")
features_with_na = [feature for feature in dataset.columns if dataset[feature].isnull().sum() >= 1]
for i in features_with_na:
    print(f"{i} column has {dataset[i].isna().sum()} missing values or {dataset[i].isna().mean():.2%} values are missed\n")
print("Test Data:\n")
test_features_with_na = [feature for feature in test_dataset.columns if test_dataset[feature].isnull().sum() >= 1]
for i in test_features_with_na:
    print(f"{i} column has {test_dataset[i].isna().sum()} missing values or {test_dataset[i].isna().mean():.2%} values are missed\n")


train_imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
test_imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
dataset[features_with_na] = train_imputer.fit_transform(dataset[features_with_na])
test_dataset[test_features_with_na] = test_imputer.fit_transform(test_dataset[test_features_with_na])


num_features = dataset.select_dtypes([np.int64, np.float64]).columns
print(f"Numeric features amount: {len(num_features)}")
print(f"Numeric features names:")
print(*num_features, sep=", ")


fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=500, constrained_layout=True)
plt.suptitle("Features distribution plots", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.histplot(dataset, x=feature, kde=True, ax=ax)
    ax.set_title(feature)
fig.show()


fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=500, constrained_layout=True)
plt.suptitle("Relationship between features and target feature", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.scatterplot(dataset, x=feature, y="Listening_Time_minutes", ax=ax)
    ax.set_title(feature)
fig.show()


plt.figure(figsize=(10, 6))
sns.heatmap(dataset[num_features].corr(), annot=True, cmap="RdYlGn")


data = dataset.copy()
data["Ads_Density"] = data["Episode_Length_minutes"]/data["Number_of_Ads"]
sns.scatterplot(data, x="Ads_Density", y="Listening_Time_minutes")


data = dataset.copy()
data["Populaty_index"] = (  0.3 * data["Host_Popularity_percentage"]
                          + 0.1 * data["Guest_Popularity_percentage"]
                          + 0.6 * ((data["Number_of_Ads"] >= 2) * 100)
                         )
sns.scatterplot(data, x="Populaty_index", y="Listening_Time_minutes")


data = dataset.copy()
data["Has_Ads"] = data["Number_of_Ads"] >= 1
sns.barplot(data, x="Has_Ads", y="Listening_Time_minutes")


fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=500, constrained_layout=True)
plt.suptitle("Outliers", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.boxplot(dataset, y=feature, ax=ax)
    ax.set_title(feature)
fig.show()


lof = LocalOutlierFactor(
    n_neighbors=20,      # Number of neighbors to consider
    contamination=0.05   # Expected fraction of outliers (5%)
)

# 2. Detect outliers (returns -1 for outliers, 1 for inliers)
outliers_preds = lof.fit_predict(dataset[num_features]) == -1  # True = outlier

dataset = dataset[~outliers_preds].reset_index(drop=True)


fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=500, constrained_layout=True)
plt.suptitle("Outliers", fontsize=16, y=1.03)
for ax, feature in zip(axes.flat, num_features):
    sns.boxplot(dataset, y=feature, ax=ax)
    ax.set_title(feature)
fig.show()


dataset = dataset[dataset["Number_of_Ads"] != 12].reset_index(drop=True)


fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=500, constrained_layout=True)
plt.suptitle("Outliers")
for ax, feature in zip(axes.flat, num_features):
    sns.boxplot(dataset, y=feature, ax=ax)
    ax.set_title(feature)
fig.show()


print(f"Amout of removed features after outlier filtration {750_000-len(dataset)} or {(750_000-len(dataset))/750_000:.2%} of data are removed ", )


name_features = ["Podcast_Name", "Episode_Title"]
print(f"Name features amount: {len(name_features)}")
print(f"Name features names:")
print(*name_features, sep=", ")


for feature in name_features:
    print(f'{feature} feature has number of unique names {len(dataset[feature].unique())}')


for feature in name_features:
    print(f'{feature} feature unique names: {dataset[feature].unique()}')


dataset.groupby("Podcast_Name")["Listening_Time_minutes"].median().sort_values(ascending=False).plot.bar(figsize=(16, 6))


dataset.groupby("Episode_Title")["Listening_Time_minutes"].median().sort_values(ascending=False).plot.bar(figsize=(16, 6))


cat_features = [feature for feature in dataset.select_dtypes([object]).columns if feature not in name_features]
print(f"Categorical features amount: {len(cat_features)}")
print(f"Categorical features names:")
print(*cat_features, sep=", ")


for feature in cat_features:
    print(f'{feature} feature has number of unique categories {len(dataset[feature].unique())}')


fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=500, constrained_layout=True)
plt.suptitle("Categories distribution")
for ax, feature in zip(axes.flat, cat_features):
    dataset[feature].value_counts().sort_values(ascending=False).plot(kind="bar", ax=ax)
fig.show()


fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=500, constrained_layout=True)
plt.suptitle("Categories corelation with listening time")
for ax, feature in zip(axes.flat, cat_features):
    dataset.groupby(feature)["Listening_Time_minutes"].median().sort_values(ascending=False).plot(kind="bar", ax=ax)
fig.show()


dataset.groupby(["Publication_Day", "Publication_Time"])["Listening_Time_minutes"].median().sort_values(ascending=False).plot.bar(figsize=(10, 6))


dataset.groupby(["Genre", "Episode_Sentiment"])["Listening_Time_minutes"].median().sort_values(ascending=False).plot.bar(figsize=(10, 6))


dataset.groupby(["Genre", "Publication_Time"])["Listening_Time_minutes"].median().sort_values(ascending=False).plot.bar(figsize=(10, 6))


def feature_engeneering(df):
    # Features to create: Is_Weekend, Part_of_Day, Ads_Density, Popularity_Index, Is_Popular_Podcast, Has_Ads, Publication_Week_Day_Time, 
    # Genre_Episode_Sentiment, Genre_Publication_Time
    df = df.copy()
    df["Is_Weekend"] = (df["Publication_Day"] == "Saturday") | (df["Publication_Day"] == "Sunday")
    df["Part_of_Day"] = (df["Publication_Time"] == "Evening") | (df["Publication_Time"] == "Night")
    df["Ads_Density"] = df["Episode_Length_minutes"] / (df["Number_of_Ads"]+1e-3)
    df["Popularity_Index"] = (0.4 * df["Host_Popularity_percentage"] + 
                              0.2 * df["Guest_Popularity_percentage"] + 
                              0.4 * ((df["Number_of_Ads"] >= 2) * 100))
    df["Is_Popular_Podcast"] = df["Popularity_Index"] >= 50
    df["Has_Ads"] = df["Number_of_Ads"] >= 1
    
    df["Publication_Week_Day_Time"] = df['Publication_Day'].astype(str) + '_' + df['Publication_Time'].astype(str)
    df["Genre_Episode_Sentiment"] = df['Genre'].astype(str) + '_' + df['Episode_Sentiment'].astype(str)
    df["Genre_Publication_Time"] = df['Genre'].astype(str) + '_' + df['Publication_Time'].astype(str)
    return df
fe_transformer = FunctionTransformer(feature_engeneering)
new_num_features = ["Popularity_Index", "Ads_Density"]
new_cat_features = ["Is_Weekend", "Part_of_Day", "Is_Popular_Podcast", 
                    "Has_Ads", "Publication_Week_Day_Time", "Genre_Episode_Sentiment", "Genre_Publication_Time"]


numerical_transformer = Pipeline([
    ('std scaller', MinMaxScaler().set_output(transform="pandas"))
])
categorical_transformer = Pipeline([
    ('one hot', OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas"))
])
transformer = ColumnTransformer(transformers=[
    ("numerical", numerical_transformer, num_features.drop(["id", "Listening_Time_minutes"]).join(pd.Index(new_num_features), how="outer")),
    ("categorical", categorical_transformer, cat_features + new_cat_features)
], remainder='passthrough')
data_pipeline = Pipeline([
    ("column transformer", transformer.set_output(transform="pandas"))
])
X = dataset.drop(["id", "Listening_Time_minutes", *name_features], axis=1)
y = dataset["Listening_Time_minutes"]


fe_data_pipeline = Pipeline([
    ("fe", fe_transformer),
    ("data pipeline", data_pipeline),
])
X_processed_for_fs = fe_data_pipeline.fit_transform(X, y)
print(X_processed_for_fs.shape)


def feature_importance(model_type, X_train, y_train, plot=True):
    selector = None
    if model_type == "xgb":
        selector = XGBRegressor(
            objective="reg:squarederror", 
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            eval_metric="rmse",
            verbosity=0,
            gpu_id=0,
            early_stopping_rounds=100,
        )
    elif model_type == "lgbm":
        selector = LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            device="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
            max_bin=63,
            verbosity=-1,
            early_stopping_rounds=100,
        )
    elif model_type == "cat":
        selector = CatBoostRegressor(
            grow_policy='SymmetricTree',
            bootstrap_type='Bernoulli',
            od_type="Iter",
            eval_metric='RMSE',
            loss_function="RMSE",
            task_type="GPU",
            devices='0',
            verbose=False,
            early_stopping_rounds=100,
        )
    else:
        raise ValueError(f"Invalid model type. Use 'xgb', 'lgbm', or 'cat', not {model_type}")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    feature_importances_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        try:
            if model_type == "lgbm":
                selector.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                )
            else:
                selector.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False,
                )
                    
        except Exception as e:
            print(f"Error in fold {fold+1}: {str(e)}")
            continue
            
        y_pred = selector.predict(X_val_fold)
        fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        rmse_scores.append(fold_rmse)
        
        try:
            importances = selector.feature_importances_
        except AttributeError:
            importances = selector.get_feature_importance()
            
        feature_importances_list.append(importances)

    if not rmse_scores:
        raise ValueError("All folds failed - check model parameters or data")
        
    avg_rmse = np.mean(rmse_scores)
    feature_importances = np.mean(feature_importances_list, axis=0)
    
    # Create DataFrame with proper sorting
    feature_importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance Score": feature_importances
    }).sort_values("Importance Score", ascending=False)
    
    # Plotting improvements
    if plot:
        plt.figure(figsize=(10, 6))
        top_10 = feature_importance_df.head(10)
        ax = sns.barplot(x="Importance Score", y="Feature", data=top_10)
        
        # Add annotations
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(
                f'{width:.3f}',
                (width + 0.005, p.get_y() + p.get_height()/2.),
                ha='left', va='center', fontsize=10
            )
            
        plt.title(
            f"Top 10 Features ({model_type.upper()})\n"
            f"Average RMSE: {avg_rmse:.4f} ± {np.std(rmse_scores):.4f}"
        )
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.xlabel("Importance Score", fontsize=10)
        plt.ylabel("Feature", fontsize=10)
        plt.tight_layout()
        plt.show()
        
    return {
        'avg_rmse': avg_rmse,
        'std_rmse': np.std(rmse_scores),
        'feature_importances': feature_importance_df,
        'top_features': feature_importance_df.head(10)['Feature'].tolist()
    }


xgb_result = feature_importance("xgb", X_processed_for_fs, y)
lgbm_result = feature_importance("lgbm", X_processed_for_fs, y)
cat_result = feature_importance("cat", X_processed_for_fs, y)


fs_result_features = list(set(xgb_result['top_features']+lgbm_result['top_features']+cat_result['top_features']))
print("Final features count:", len(fs_result_features))
print("Final features:", end=" ")
print(*fs_result_features, sep=", ")

fs_data_pipeline = Pipeline([
    ("fe transformer", fe_data_pipeline),
    ("fs transformer", FunctionTransformer(lambda x: x[fs_result_features])),
])
X_processed = fs_data_pipeline.fit_transform(X, y)


X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if SUBMISSION:
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X_processed, y, cv=5, scoring='neg_mean_squared_error')
    lr.fit(X_processed, y)
    print(f"Model train performance: {np.mean(np.sqrt(-lr_scores))}")
else:
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict(X_test)
    print(f"Model train performance: {np.mean(np.sqrt(-lr_scores))}")
    print(f"Model test performance: {np.sqrt(mean_squared_error(y_test, lr_y_pred))}")


if SUBMISSION:
    lasso_lr = Lasso()
    lasso_lr_scores = cross_val_score(lasso_lr, X_processed, y, cv=5, scoring='neg_mean_squared_error')
    lasso_lr.fit(X_processed, y)
    print(f"Model train performance: {np.mean(np.sqrt(-lasso_lr_scores))}")
else:
    lasso_lr = Lasso()
    lasso_lr_scores = cross_val_score(lasso_lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    lasso_lr.fit(X_train, y_train)
    lasso_lr_y_pred = lasso_lr.predict(X_test)
    print(f"Model train performance: {np.mean(np.sqrt(-lasso_lr_scores))}")
    print(f"Model test performance: {np.sqrt(mean_squared_error(y_test, lasso_lr_y_pred))}")


if SUBMISSION:
    ridge_lr = Ridge()
    ridge_lr_scores = cross_val_score(ridge_lr, X_processed, y, cv=5, scoring='neg_mean_squared_error')
    ridge_lr.fit(X_processed, y)
    print(f"Model train performance: {np.mean(np.sqrt(-ridge_lr_scores))}")
else:
    ridge_lr = Ridge()
    ridge_lr_scores = cross_val_score(ridge_lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    ridge_lr.fit(X_train, y_train)
    ridge_lr_y_pred = ridge_lr.predict(X_test)
    print(f"Model train performance: {np.mean(np.sqrt(-ridge_lr_scores))}")
    print(f"Model test performance: {np.sqrt(mean_squared_error(y_test, ridge_lr_y_pred))}")


xgb_params = {
    'n_estimators': 400,
    'max_depth': 14,
    'learning_rate': 0.0345,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': -1,
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse',
}

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = []
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}/{n_splits}")  
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]  
    
    xgbr = XGBRegressor(**xgb_params)
    xgbr.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=100)    
    
    val_pred = xgbr.predict(X_val_fold)
    score = rmse(y_val_fold, val_pred)
    scores.append(score)
    test_preds += xgbr.predict(X_test) / n_splits      
    print(f"Fold {fold + 1} RMSE: {score:.4f}")
    
print(f'Optimized Cross-validated RMSE score: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')


lgbm_params = {
    'n_estimators': 400,
    'max_depth': 14,
    'learning_rate': 0.0345,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'metric': 'rmse',
    'verbosity': -1,
    'device': 'gpu',  # Correct parameter name
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_bin': 63
}


kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}/5...")
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    lgbm = LGBMRegressor(**lgbm_params)
    lgbm.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )
    
    val_pred = lgbm.predict(X_val_fold)
    score = rmse(y_val_fold, val_pred)
    scores.append(score)
    test_preds += lgbm.predict(X_test) / 5
    print(f"Fold {fold + 1} RMSE: {score:.4f}")

print(f'Optimized Cross-validated RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}')


cat_params = {
    'iterations': 400,
    'depth': 14,
    'learning_rate': 0.0345,
    'subsample': 0.8,
    'random_seed': 42,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'verbose': 100,
    'task_type': 'GPU',
    'devices': '0',
    'bootstrap_type': 'Bernoulli',
    'grow_policy': 'SymmetricTree', 
    'has_time': False,
    'used_ram_limit': '10gb',
    'gpu_ram_part': 0.95
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Training fold {fold + 1}/5...")
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    catboost = CatBoostRegressor(**cat_params)
    
    catboost.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=100,
        plot=False,
    )
    
    val_pred = catboost.predict(X_val_fold)
    score = rmse(y_val_fold, val_pred)
    scores.append(score)
    test_preds += catboost.predict(X_test) / 5
    print(f"Fold {fold + 1} RMSE: {score:.4f}")

print(f'Optimized Cross-validated RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}')
print(f'Max RMSE score: {np.max(scores):.3f}')
print(f'Min RMSE score: {np.min(scores):.3f}')


class PodcastDataset(Dataset):
    def __init__(self, features, targets):
        self.X = features.values if hasattr(features, 'values') else features
        self.y = targets.values if hasattr(targets, 'values') else targets
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32).to(DEVICE),
            torch.tensor(self.y[idx], dtype=torch.float32).to(DEVICE)
        )

class Model(nn.Module):
    def __init__(self, in_features: int, 
                 hidden_layer_features: int = 64, 
                 out_features: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_layer_features),
            nn.ReLU(),
            nn.Linear(hidden_layer_features, out_features)
        )
        
    def forward(self, x):
        return self.model(x).squeeze(-1)


EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 32
INPUT_FEATURES = X_train.shape[1]  # Make sure this matches your data

# Initialize model
model = Model(INPUT_FEATURES).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Create datasets
full_dataset = PodcastDataset(X_train, y_train)
train_dataset, val_dataset = random_split(full_dataset, [0.9, 0.1])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        print(f"\rEpoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}", end="")
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
            val_preds = model(X_val)
            val_loss += criterion(val_preds, y_val).item()
    
    # Calculate epoch metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\nEpoch: {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


X_submission = fs_data_pipeline.transform(test_dataset.drop(["id"], axis=1))


lasso_predictions = lasso_lr.predict(X_submission)
pd.DataFrame({"id": test_dataset["id"], "Listening_Time_minutes": lasso_predictions}).to_csv("lasso_submission.csv", index=False)


xgbr_preds = xgbr.predict(X_submission)
pd.DataFrame({"id": test_dataset["id"], "Listening_Time_minutes": xgbr_preds}).to_csv("xgb_submission.csv", index=False)


lgbm_preds = lgbm.predict(X_submission)
pd.DataFrame({"id": test_dataset["id"], "Listening_Time_minutes": lgbm_preds}).to_csv("lgbm_submission.csv", index=False)


catboost_preds = catboost.predict(X_submission)
pd.DataFrame({"id": test_dataset["id"], "Listening_Time_minutes": catboost_preds}).to_csv("catboost_submission.csv", index=False)


model.eval()
dl_preds = model(torch.from_numpy(X_submission.to_numpy()[None, ...].astype(np.float32)).to(DEVICE)).reshape(-1).cpu().detach().numpy()
pd.DataFrame({"id": test_dataset["id"], "Listening_Time_minutes": dl_preds}).to_csv("dl_submission.csv", index=False)

