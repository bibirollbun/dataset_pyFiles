import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
train_df.head()


print("Train Data Info: ")
train_df.info()


print("Test Data Info: ")
test_df.info()


train_df.describe().T


test_df.describe().T


plt.figure(figsize=(16,6))

colours = ["mediumaquamarine", "oldlace"] 
sns.heatmap(train_df.isnull(), cmap=sns.color_palette(colours))

plt.title("Train Missing Values", fontsize = 14, fontweight = "bold")
plt.xticks(rotation=45, ha="right")
plt.show()


nullies = train_df.isnull().sum()
null_df = pd.DataFrame(nullies[nullies > 0], columns=['Null Count'])

null_df['Percentage'] = (null_df['Null Count'] / len(train_df)) * 100

null_df = null_df.sort_values(by="Null Count", ascending=False)
print("Train Null Values DataFrame:")
null_df


nullies = test_df.isnull().sum()
null_df = pd.DataFrame(nullies[nullies > 0], columns=['Null Count'])

null_df['Percentage'] = (null_df['Null Count'] / len(train_df)) * 100

null_df = null_df.sort_values(by="Null Count", ascending=False)
print("Test Null Values DataFrame:")
null_df


print("Duplicates: ")
train_df.duplicated().sum()


print("Unique Values: ")
uniques = train_df.nunique()
unique_df = pd.DataFrame(uniques, columns=["Unique Count"])
unique_df = unique_df.sort_values(by="Unique Count", ascending=False)

unique_df


pal = ["darkcyan", "darkslategray", "olivedrab", "seagreen"]
plt.figure(figsize=(8, 4))
sns.countplot(x=train_df["Material"], palette=pal, order=train_df["Material"].value_counts().index)

plt.title(f"Distribution of {'Material'}", fontsize=14, fontweight="bold")
plt.xlabel("Material", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(x=train_df["Brand"], palette=pal, order=train_df["Brand"].value_counts().index)

plt.title(f"Distribution of {'Brand'}", fontsize=14, fontweight="bold")
plt.xlabel("Brand", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(x=train_df["Size"], palette=pal, order=train_df["Size"].value_counts().index)

plt.title(f"Distribution of {'Size'}", fontsize=14, fontweight="bold")
plt.xlabel("Size", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(x=train_df["Style"], palette=pal, order=train_df["Style"].value_counts().index)

plt.title(f"Distribution of {'Style'}", fontsize=14, fontweight="bold")
plt.xlabel("Style", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()


plt.figure(figsize=(8, 4))
sns.countplot(x=train_df["Color"], palette=pal, order=train_df["Color"].value_counts().index)

plt.title(f"Distribution of {'Color'}", fontsize=14, fontweight="bold")
plt.xlabel("Color", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()


# Feature engineering for train data
train_df["Storage_Efficiency"] = train_df["Compartments"] / train_df["Weight Capacity (kg)"]
train_df["Compartments_x_Size"] = train_df["Compartments"].astype(str) + "_" + train_df["Size"]
train_df["Material_x_Size"] = train_df["Material"].astype(str) + "_" + train_df["Size"]
train_df["Weight_Per_Compartment"] = train_df["Weight Capacity (kg)"] / train_df["Compartments"] # how much weight can a compartment hold
train_df["Compartments_Per_Size"] = train_df["Compartments"] / train_df["Size"].map({"Small": 1, "Medium": 2, "Large": 3})
train_df["Capacity_x_Compartments"] = train_df["Weight Capacity (kg)"] * train_df["Compartments"] #effect of weight capacity and N of compartments
train_df["Size_x_Compartments"] = train_df["Size"].map({"Small": 1, "Medium": 2, "Large": 3}) * train_df["Compartments"] #differentiate large bags with few compartments vs. small bags with many.
train_df["Weight_Diff_From_Mean"] = train_df["Weight Capacity (kg)"] - train_df["Weight Capacity (kg)"].mean()
train_df["Compartments_Diff_From_Mean"] = train_df["Compartments"] - train_df["Compartments"].mean()
train_df["Weight_Squared"] = train_df["Weight Capacity (kg)"] ** 2
train_df["Compartments_Squared"] = train_df["Compartments"] ** 2
train_df["Weight_Sqrt"] = np.sqrt(train_df["Weight Capacity (kg)"])

# feature engineering for test data
test_df["Storage_Efficiency"] = test_df["Compartments"] / test_df["Weight Capacity (kg)"]
test_df["Compartments_x_Size"] = test_df["Compartments"].astype(str) + "_" + test_df["Size"]
test_df["Material_x_Size"] = test_df["Material"].astype(str) + "_" + test_df["Size"]
test_df["Weight_Per_Compartment"] = test_df["Weight Capacity (kg)"] / test_df["Compartments"] # how much weight can a compartment hold
test_df["Compartments_Per_Size"] = test_df["Compartments"] / test_df["Size"].map({"Small": 1, "Medium": 2, "Large": 3})
test_df["Capacity_x_Compartments"] = test_df["Weight Capacity (kg)"] * test_df["Compartments"] #effect of weight capacity and N of compartments
test_df["Size_x_Compartments"] = test_df["Size"].map({"Small": 1, "Medium": 2, "Large": 3}) * test_df["Compartments"] #differentiate large bags with few compartments vs. small bags with many.
test_df["Weight_Diff_From_Mean"] = test_df["Weight Capacity (kg)"] - test_df["Weight Capacity (kg)"].mean()
test_df["Compartments_Diff_From_Mean"] = test_df["Compartments"] - test_df["Compartments"].mean()
test_df["Weight_Squared"] = test_df["Weight Capacity (kg)"] ** 2
test_df["Compartments_Squared"] = test_df["Compartments"] ** 2
test_df["Weight_Sqrt"] = np.sqrt(test_df["Weight Capacity (kg)"])


cats_col = train_df.select_dtypes(include=["object"]).columns
nums_col = train_df.select_dtypes(include=['float64']).columns.drop("Price")
for col in nums_col:
    if col in test_df.columns:
        median_value = train_df[col].median()
        train_df[col].fillna(median_value, inplace=True)
        test_df[col].fillna(median_value, inplace=True)

for col in cats_col:
    if col in test_df.columns:
        train_df[col].fillna("Unknown", inplace=True)
        test_df[col].fillna("Unknown", inplace=True)


# endoding categorical features
LE = LabelEncoder()
for col in cats_col:
    train_df[col] = LE.fit_transform(train_df[col])    
    test_df[col] = LE.transform(test_df[col])

# scaling numerical features 
SS = StandardScaler()
train_df[nums_col] = SS.fit_transform(train_df[nums_col])
test_df[nums_col] = SS.transform(test_df[nums_col])


corr_matrix = train_df.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right") 
plt.yticks(rotation=0)

plt.show()


X = train_df.copy()
y = X.pop("Price")

# Ensure discrete_features matches X.columns
discrete_features = [col in cats_col for col in X.columns]

# Mutual Information Function
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X.values, y, discrete_features=discrete_features)  # Use .values
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    return mi_scores.sort_values(ascending=False)

mi_scores = make_mi_scores(X, y, discrete_features)

mi_scores


plt.figure(figsize=(8, 5))

colors = sns.color_palette(pal, len(mi_scores))

plt.barh(np.arange(len(mi_scores)), mi_scores, color=colors)

plt.yticks(np.arange(len(mi_scores)), mi_scores.index, fontsize=12)

plt.xlabel("Mutual Information Score", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.title("Feature Importance (Mutual Information)", fontsize=16, fontweight="bold")

plt.show()


"""
def create_stratified_bins(y, bins=5):
    return pd.qcut(y, q=bins, labels=False)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
    }

    # Set up Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Convert `y` into bins for StratifiedKFold
    y_bins = create_stratified_bins(y)
    
    mse_scores = []
    
    for train_idx, val_idx in skf.split(X, y_bins):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    
    # Return the average MSE across folds
    return np.mean(mse_scores)

# ðŸ”¥ Run Optuna Optimization
study = optuna.create_study(direction="minimize")  # Minimize MSE
study.optimize(objective, n_trials=50)  # Run 50 trials
"""


#best_params = study.best_params


best_params = {'n_estimators': 500,
 'max_depth': 2,
 'learning_rate': 0.05821780823901427,
 'subsample': 0.7364377765045789,
 'colsample_bytree': 0.6593243245996594,
 'gamma': 3.8733288172824043,
 'reg_alpha': 0.5157184369475334,
 'reg_lambda': 1.0388849867243528}


X_train = train_df.drop(columns=["Price"])
y_train = train_df["Price"]

X_test = test_df.copy()


final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

test_predictions = final_model.predict(X_test)

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train on the new split to see RMSE
final_model.fit(X_train_split, y_train_split)

y_val_pred = final_model.predict(X_val)

# Compute RMSE
rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


sample_submission["Price"] =  test_predictions
sample_submission.to_csv("submission.csv",index=False)
sample_submission




