import pandas as pd
import numpy as np
import seaborn as sns
import warnings

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import Pool

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# org = pd.read_csv('/kaggle/input/d/ruchikakumbhar/calories-burnt-prediction/calories.csv')
# org.rename(columns={'User_ID': 'id', 'Gender': 'Sex'}, inplace=True)

# org.head()


train.head(2)


plt.figure(figsize=(10, 6))
sns.histplot(train['Calories'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Calories', fontsize=16)
plt.xlabel('Calories')
plt.ylabel('Number of Observations')
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 5))

# 1. Histogram + KDE (both sexes)
plt.subplot(1, 2, 1)
sns.histplot(data=train, x='Calories', hue='Sex', kde=True, bins=30, palette='Set2', element='step')
plt.title('Calories Distribution by Sex')
plt.xlabel('Calories')
plt.ylabel('Number of Observations')
plt.grid(True)

# 2. Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(data=train, x='Sex', y='Calories', palette='Set2')
plt.title('Boxplot of Calories by Sex')
plt.xlabel('Sex')
plt.ylabel('Calories')
plt.grid(True)

plt.tight_layout()
plt.show()


# train_org = pd.concat([train, org], ignore_index=True)
# train = train_org


def val_loss_function(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    return np.sqrt(np.mean((log_pred - log_true) ** 2))

def cross_val_predict(model, X_train, y_train, X_test, val_loss_function, n_splits=5, random_state=42):
    print(f"Model: {model.__class__.__name__}")

    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_score = 0
    val_score_log = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # scaler = StandardScaler()
        # X_tr = scaler.fit_transform(X_tr) 
        # X_val = scaler.transform(X_val)
        
        model.fit(X_tr, y_tr)
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        cur_val_score = val_loss_function(y_val, val_preds)
        print(f"Current validation score: {cur_val_score}")
        
        cur_val_score_log = val_loss_function(np.expm1(y_val), np.expm1(val_preds))
        print(f"Current validation score np.expm1: {cur_val_score_log}")
        
        val_score += cur_val_score / n_splits
        val_score_log += cur_val_score_log / n_splits

        test_preds += model.predict(X_test) / n_splits
        # test_preds += model.predict(scaler.transform(X_test)) / n_splits

    print(f"Average validation score: {val_score}")
    print(f"Average validation score np.expm1: {val_score_log}")
    return oof_preds, test_preds, val_score


def plot_feature_importance(model, feature_names, top_n=50):
    """
    Plots the top `top_n` feature importances using both frequency (split) and gain.

    Parameters:
    - model: a fitted LGBMClassifier (or LGBMRegressor) from which to extract importances.
    - feature_names: list-like, feature names corresponding to the model's training data.
    - top_n: int, number of top features to display (default=50).

    Returns:
    - Displays a matplotlib figure with two horizontal bar plots.
    """
    # Frequency importance (default attribute)
    split_importance = model.feature_importances_
    # Gain importance (using the booster)
    gain_importance = model.booster_.feature_importance(importance_type="gain")
    
    # Create a DataFrame combining both importances
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "frequency": split_importance,
        "gain": gain_importance
    })
    
    # Sort features based on gain importance and select top_n
    importance_df = importance_df.sort_values("gain", ascending=False).head(top_n)
    
    # Create two side-by-side bar plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot frequency importance
    axes[0].barh(importance_df["feature"][::-1], importance_df["frequency"][::-1])
    axes[0].set_title("Feature Importance (Frequency)")
    axes[0].set_xlabel("Frequency")
    
    # Plot gain importance
    axes[1].barh(importance_df["feature"][::-1], importance_df["gain"][::-1])
    axes[1].set_title("Feature Importance (Gain)")
    axes[1].set_xlabel("Gain")
    
    plt.tight_layout()
    plt.show()

def plot_catboost_model_analysis(model, X_train, y_train, top_n=20):
    """
    Plots multiple visualizations for CatBoost model analysis, including:
    - Feature importance (split & gain)
    - SHAP values
    - Prediction distribution
    - Loss function evolution during training

    Parameters:
    - model: trained CatBoost model
    - X_train: Training data (features)
    - y_train: Training labels
    - top_n: Number of top features to display in importance plots
    """
    feature_names = X_train.columns if hasattr(X_train, "columns") else [f"Feature {i}" for i in range(X_train.shape[1])]

    split_importance = model.get_feature_importance(type="FeatureImportance")
    gain_importance = model.get_feature_importance(type="PredictionValuesChange")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Split Importance": split_importance,
        "Gain Importance": gain_importance
    }).sort_values("Gain Importance", ascending=False).head(top_n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.barplot(x="Split Importance", y="Feature", data=importance_df, ax=axes[0, 0], palette="Blues_r")
    axes[0, 0].set_title("Feature Importance (Split Count)")

    sns.barplot(x="Gain Importance", y="Feature", data=importance_df, ax=axes[0, 1], palette="Reds_r")
    axes[0, 1].set_title("Feature Importance (Gain)")

    if model.evals_result_:
        train_loss = model.evals_result_['learn']['Logloss' if 'Logloss' in model.evals_result_['learn'] else list(model.evals_result_['learn'].keys())[0]]
        axes[1, 0].plot(train_loss, label="Train Loss", color="blue")
        axes[1, 0].set_title("Loss Function Evolution")
        axes[1, 0].set_xlabel("Iterations")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()


    explainer = model.get_feature_importance(Pool(X_train, y_train), type="ShapValues")
    shap_values = np.array(explainer)[:, :-1]  # Ostatnia kolumna to wartości bazowe, więc usuwamy
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Importance": mean_shap
    }).sort_values("SHAP Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="SHAP Importance", y="Feature", data=shap_df, palette="coolwarm")
    plt.title("SHAP Feature Importance")
    plt.show()


def preprocess_data_male(df):

    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['Intensity'] = df['Heart_Rate'] * df['Duration']
    df['HR_per_min'] = df['Heart_Rate'] / df['Duration']

    df['Temp_Change'] = df['Body_Temp'] * df['Duration']

    df['Age_Weight'] = df['Age'] * df['Weight']
    df['Age_Heart'] = df['Age'] * df['Heart_Rate']
    df['Height_Weight'] = df['Height'] * df['Weight']
    df['Heart_Temp'] = df['Heart_Rate'] * df['Body_Temp']
    df['Weight_Temp'] = df['Weight'] * df['Body_Temp']  

    df['HeartRate_PercentMax'] = df['Heart_Rate'] / (220 - df['Age'])

    df['Duration_per_age'] = df['Duration'] / df['Age']

    max_age = df['Age'].max()
    
    # Tworzenie przedziałów co 5 lat od 20 wzwyż
    bins = list(range(20, int(max_age) + 6, 5))  # np. [20, 25, 30, ..., max+5]
    
    # Etkiety: używamy dolnej granicy jako liczby całkowite
    labels = bins[:-1]  # np. [20, 25, 30, ...]

    # Tworzymy kolumnę Age_Group jako int
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df['Age_Group'] = df['Age_Group'].astype(int)

    df['Calories_Estimated'] = (
    0.8098 * df['Age'] +
    0.0418 * df['Height'] +
    0.2655 * df['Weight'] +
    6.9455 * df['Duration'] +
    2.3353 * df['Heart_Rate'] +
    -20.8006 * df['Body_Temp'] +
    527.4488
)

    df['Calories_Estimated'] = np.clip(df['Calories_Estimated'], 0, None)
    
    df.drop(columns=['Sex'], inplace=True)
    df.drop(columns=['id'], inplace = True)
    df.drop(columns=['Age'], inplace = True)
    
    return df

def preprocess_data_female(df):

    # df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['Intensity'] = df['Heart_Rate'] * df['Duration']
    df['HR_per_min'] = df['Heart_Rate'] / df['Duration']

    df['Temp_Change'] = df['Body_Temp'] * df['Duration']

    df['Age_Weight'] = df['Age'] * df['Weight']
    df['Age_Heart'] = df['Age'] * df['Heart_Rate']
    df['Height_Weight'] = df['Height'] * df['Weight']
    df['Heart_Temp'] = df['Heart_Rate'] * df['Body_Temp']
    df['Weight_Temp'] = df['Weight'] * df['Body_Temp']  

    df['Duration_Squared'] = df['Duration'] ** 2

    df['HeartRate_PercentMax'] = df['Heart_Rate'] / (220 - df['Age'])

    df['High_HeartRate'] = (df['Heart_Rate'] > 160).astype(int)

    df['Duration_per_age'] = df['Duration'] / df['Age']

    df['Calories_Estimated'] = (
    0.2998 * df['Age'] +
    -0.0276 * df['Height'] +
    -0.1554 * df['Weight'] +
    6.5003 * df['Duration'] +
    1.6375 * df['Heart_Rate'] +
    -15.5978 * df['Body_Temp'] +
    457.3189 )
    df['Calories_Estimated'] = np.clip(df['Calories_Estimated'], 0, None) 
    
    df.drop(columns=['Sex'], inplace=True)
    df.drop(columns=['id'], inplace = True)
   
            
    return df


train_male = train[train['Sex'] == 'male']
train_female = train[train['Sex'] == 'female']


test_male = test[test['Sex'] == 'male']
test_female = test[test['Sex'] == 'female']


train_male = preprocess_data_male(train_male)
test_male = preprocess_data_male(test_male)

numerical_cols = train_male.columns.tolist()
target_column = 'Calories'
numerical_cols.remove(target_column)

X_train_male = train_male[numerical_cols]
X_test_male = test_male[numerical_cols]
y_train_male = train_male[target_column]
y_train_male = np.log1p(y_train_male)


train_female = preprocess_data_female(train_female)
test_female = preprocess_data_female(test_female)

numerical_cols = train_female.columns.tolist()
target_column = 'Calories'
numerical_cols.remove(target_column)

X_train_female = train_female[numerical_cols]
X_test_female = test_female[numerical_cols]
y_train_female = train_female[target_column]
y_train_female = np.log1p(y_train_female)


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# X = train_male[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
# y = train_male['Calories']

# model = LinearRegression()
# model.fit(X, y)

# y_pred = model.predict(X)

# y_pred = np.clip(y_pred, 1, 314)

# # R^2 score
# r2 = r2_score(y, y_pred)
# val_loss = val_loss_function(y, y_pred)

# print("R^2 score:", round(r2, 4))
# print("loss_function:", val_loss)
# print("coefs:")
# for feature, coef in zip(X.columns, model.coef_):
#     print(f"  {feature}: {coef:.4f}")
# print(f" (intercept): {model.intercept_:.4f}")


models = [
    # LGBMRegressor(boosting_type='gbdt'),
    # XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
    CatBoostRegressor(
    verbose=0)
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train_male, y_train_male, X_test_male, val_loss_function)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


# cur_model = models[0]
# feature_names = X_train_male.columns
# plot_feature_importance(cur_model, feature_names)


y_pred_male = results['CatBoostRegressor']['test']
#y_pred_male = (results['CatBoostRegressor']['test'] + results['XGBRegressor']['test'] + results['LGBMRegressor']['test']) / 3
y_pred_male = np.expm1(y_pred_male)
y_pred_male_series = pd.Series(y_pred_male, index=X_test_male.index)


# X = train_female[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
# y = train_female['Calories']

# model = LinearRegression()
# model.fit(X, y)

# y_pred = model.predict(X)

# y_pred = np.clip(y_pred, 1, 314)

# # R^2 score
# r2 = r2_score(y, y_pred)
# val_loss = val_loss_function(y, y_pred)

# print("R^2 score:", round(r2, 4))
# print("loss_function:", val_loss)
# print("coefs:")
# for feature, coef in zip(X.columns, model.coef_):
#     print(f"  {feature}: {coef:.4f}")
# print(f" (intercept): {model.intercept_:.4f}")


models = [
    # LGBMRegressor(boosting_type='gbdt'),
    # XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
    CatBoostRegressor(
    verbose=0)]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train_female, y_train_female, X_test_female, val_loss_function)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred_female = results['CatBoostRegressor']['test']

#y_pred_female = (results['CatBoostRegressor']['test'] + results['XGBRegressor']['test'] + results['LGBMRegressor']['test']) / 3

y_pred_female = np.expm1(y_pred_female)

y_pred_female_series = pd.Series(y_pred_female, index=X_test_female.index)


# models = [
#     # LGBMRegressor(boosting_type='gbdt'),
#     # XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6),
#     CatBoostRegressor(
#     verbose=0)]

# results = {}

# for model in models:
#     oof, test, score = cross_val_predict(model, X_train, y_train, X_test, val_loss_function)
#     results[model.__class__.__name__] = {
#         "oof": oof,
#         "test": test,
#         "score": score
#     }
#     print(f"Final validation score for {model.__class__.__name__}: {score}\n")


# cur_model = models[0]
# feature_names = X_train.columns
# plot_feature_importance(cur_model, feature_names)


# cur_model = models[-1]
# feature_names = X_train.columns
# plot_catboost_model_analysis(model, X_train, y_train, top_n=80)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

# def val_loss_function(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     y_pred = np.clip(y_pred, 0, None)  
#     log_true = np.log1p(y_true)
#     log_pred = np.log1p(y_pred)
#     return np.sqrt(np.mean((log_pred - log_true) ** 2))


# X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
#     X_train, y_train, test_size=0.2, random_state=42
# )


# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_split)
# X_val_scaled = scaler.transform(X_val_split)


# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor(y_train_split.values, dtype=torch.float32).view(-1, 1).to(device)
# X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
# y_val_tensor = torch.tensor(y_val_split.values, dtype=torch.float32).view(-1, 1).to(device)


# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64)


# class RegressionModel(nn.Module):
#     def __init__(self, input_dim):
#         super(RegressionModel, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 8),
#             nn.ReLU(),
#             nn.Linear(8, 4),
#             nn.ReLU(),
#             nn.Linear(4, 1),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.net(x)


# input_dim = X_train.shape[1]
# model = RegressionModel(input_dim).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# epochs = 50
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0
#     for xb, yb in train_loader:
#         optimizer.zero_grad()
#         preds = model(xb)
#         loss = criterion(preds, yb)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * xb.size(0)


#     model.eval()
#     val_preds = []
#     val_targets = []
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             preds = model(xb).squeeze().cpu().numpy()
#             targets = yb.squeeze().cpu().numpy()
#             val_preds.extend(preds)
#             val_targets.extend(targets)

#     val_rmsle = val_loss_function(val_targets, val_preds)
#     print(f"Epoch {epoch+1}: Train Loss = {train_loss / len(train_dataset):.4f}, "
#           f"Val RMSLE = {val_rmsle:.4f}")



# X_test_scaled = scaler.transform(X_test)

# X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# model.eval()
# with torch.no_grad():
#     test_preds = model(X_test_tensor).squeeze().cpu().numpy()
#     test_preds = np.clip(test_preds, 0, None) 


# test_preds


# y_pred = (results['CatBoostRegressor']['test'] + results['XGBRegressor']['test'] + results['LGBMRegressor']['test']) / 3
# y_pred = test_preds
# y_pred = results['CatBoostRegressor']['test']


y_pred_combined = pd.concat([y_pred_male_series, y_pred_female_series])
y_pred_combined = y_pred_combined.sort_index()
y_pred = y_pred_combined


min(y_pred)


max(y_pred)


# y_pred = np.clip(y_pred, 1, 314)


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub[target_column] = y_pred
sub.to_csv('submission.csv', index = False)

