# ğŸ“¦ Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import os

import warnings
warnings.filterwarnings("ignore")

# ğŸ—‚ï¸� Check available files in input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ğŸ“¥ Load train and test datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

print("Train shape:", train.shape, "| Test shape:", test.shape)



#Checking data types
print("Train Dataset Summary (First Rows and Data Types)")

display(train.head().T,train.dtypes)


train.info()


#Checking null values
train.isnull().sum()


#Checking null values
test.isnull().sum()


# Basic cleaning
df_train = train.drop("id", axis=1)
df_test = test.drop("id", axis=1)

df_train.head()



#Check Duplicate Rows
df_train.duplicated().sum()


#Drop Duplicate Rows
df_train = df_train.drop_duplicates()
df_train.duplicated().sum()


#Scaling numeric columns using `StandardScaler` ensures that features have a consistent range.
from IPython.display import display, HTML


numeric_cols = df_train.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("accident_risk")

# 1ï¸�âƒ£ Smart Summary Table
desc = df_train[numeric_cols].describe().T
desc['missing'] = train[numeric_cols].isnull().sum()
desc['unique'] = [df_train[c].nunique() for c in numeric_cols]
display(HTML("<h3 style='color:#1f77b4'> Summary of Numerical Features</h3>"))
display(desc.style.background_gradient(cmap='Blues'))



fig, axes = plt.subplots(2, len(numeric_cols), figsize=(3*len(numeric_cols), 6))
fig.suptitle(" Outliers â€¢ Distribution â€¢ Skewness", fontsize=15, fontweight='bold')

for i, col in enumerate(numeric_cols):
    # Boxplot (Outliers)
    axes[0, i].boxplot(df_train[col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', color='red'),
                       medianprops=dict(color='black'))
    axes[0, i].set_title(col, fontsize=11, fontweight='bold')  
    axes[0, i].set_xticks([])

    # Histogram (Distribution + Skew)
    axes[1, i].hist(df_train[col].dropna(), bins=20, color="skyblue", edgecolor='gray')
    skewness = df_train[col].dropna().skew()
    axes[1, i].set_title(f"Skew={skewness:.2f}", fontsize=10)
    axes[1, i].set_xlabel(col, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



print("Inspecting Unique Values in Categorical Columns\n")

categorical_cols = df_train.select_dtypes(exclude="number").columns.tolist()

for col in categorical_cols:
    print(f"{col} : {train[col].unique()}")
    print("." * 50)


fig, axes = plt.subplots(2, 4, figsize=(16,8))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.barplot(data=df_train, x=col, y='accident_risk', ax=axes[i],
                estimator='mean', errorbar=None, palette='magma')
    axes[i].set_title(f'{col} vs Mean accident_risk')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#We can observe that accident risk tends to increase under poor lighting and rainy conditions.


# Distribution of Target Variable

y_train = df_train['accident_risk']

fig = plt.figure(figsize=(10, 5))
grid = plt.GridSpec(4, 1, hspace=0.1)
ax_hist = fig.add_subplot(grid[0:3, 0])
ax_box = fig.add_subplot(grid[3, 0], sharex=ax_hist)

sns.histplot(y_train, bins=30, kde=True, color='coral', ax=ax_hist, legend=False)
ax_hist.set_title("Distribution of accident_risk (Target Variable)")
ax_hist.set_xlabel("")

sns.boxplot(x=y_train, ax=ax_box, color='coral')
ax_box.set_xlabel("accident_risk")

plt.setp(ax_hist.get_xticklabels(), visible=False)
plt.tight_layout()
plt.show()


df_train.columns
bool_cols = ["road_signs_present", "public_road","holiday", "school_season"]

for col in bool_cols :
    df_train[col]= df_train[col].astype(int)
    df_test[col]=df_test[col].astype(int)


# List of columns that will be encoded with ordinal values
ordinal_columns = [
    'road_type',
    'lighting',
    'weather',
    'time_of_day',
]

# Specify the order of categories for each column
ordinal_categories = [
    ['rural', 'urban', 'highway'],         # road_type
    ['daylight', 'dim', 'night'],          # lighting
    ['clear', 'rainy', 'foggy'],           # weather
    ['morning', 'afternoon', 'evening'],   # time_of_day
]

# Create an instance of OrdinalEncoder with the specified categories
ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)

# Apply the OrdinalEncoder to the training data and test data
df_train[ordinal_columns] = ordinal_encoder.fit_transform(df_train[ordinal_columns])
df_test[ordinal_columns] = ordinal_encoder.transform(df_test[ordinal_columns])


print("Train Dataset Summary (First Rows,  Shape,  Data Types)")
df_train
display(df_train.head().T, df_train.shape, df_train.dtypes)


print("Test Dataset Summary (First Rows,  Shape,  Data Types)")

display(df_test.head().T, df_test.shape, df_test.dtypes)


# Data Normalization Using StandardScaler

columns=df_test.columns
scaler = StandardScaler()
df_train[columns] = scaler.fit_transform(df_train[columns])
df_test[columns] = scaler.transform(df_test[columns])


# Prepare input data
X_train = df_train.drop(['accident_risk'], axis=1)
y_train = df_train['accident_risk']

# Split the dataset into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)


# Correlation Heatmap of Train Dataset

plt.figure(figsize=(11, 7))
heatmap=sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size":9})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Train Dataset')
plt.show()


# Correlation Heatmap of Test Dataset

plt.figure(figsize=(11, 7))
heatmap=sns.heatmap(df_test.corr(), annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size":9})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Test DataSet')
plt.show()


# Target variable
target_column = 'accident_risk'

# Compute correlation matrix
correlation_matrix = df_train.corr()

# Extract and sort absolute correlations with the target
correlation_with_target = correlation_matrix[target_column].drop(target_column).abs()
correlation_with_target_sorted = correlation_with_target.sort_values(ascending=False)

# Plot horizontal bar chart of feature correlations
plt.figure(figsize=(10, 6))
ax = correlation_with_target_sorted.plot(kind='barh', color='green', edgecolor='black')

for index, value in enumerate(correlation_with_target_sorted):
    if not (pd.isna(value) or value == float('inf') or value == float('-inf')):
        plt.text(value + 0.002, index, f"{value:.4f}", va='center', fontsize=10)

ax.invert_yaxis()
plt.xlim(0, correlation_with_target_sorted.max() + 0.01)
plt.title("Feature Importance Relative to Target", fontsize=12)
plt.xlabel("Correlation Coefficient", fontsize=11)
plt.ylabel("Feature", fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout(pad=2)
plt.show()


def add_features(df):
    df = df.copy()
    if "speed_limit" in df.columns and "curvature" in df.columns:
        df["speed_curvature"] = df["speed_limit"] * df["curvature"]
    if "num_lanes" in df.columns and "speed_limit" in df.columns:
        df["lanes_speed"] = df["num_lanes"] * df["speed_limit"]
    if "num_reported_accidents" in df.columns and "speed_limit" in df.columns:
        df["accident_speed_risk"] = (
            df["num_reported_accidents"] * df["speed_limit"] / 100
        )
    if "curvature" in df.columns:
        df["curvature_squared"] = df["curvature"] ** 2
    if "speed_limit" in df.columns:
        df["speed_squared"] = df["speed_limit"] ** 2
    if "weather" in df.columns and "lighting" in df.columns:
        df["weather_lighting"] = df["weather"].astype(str) + "_" + df["lighting"].astype(str)
    return df


# Apply features to train and val data
X_train_fe = add_features(X_train)
X_val_fe = add_features(X_val)

# Label encoding for the combined column
if "weather_lighting" in X_train_fe.columns:
    lbl = LabelEncoder()
    X_train_fe["weather_lighting"] = lbl.fit_transform(X_train_fe["weather_lighting"])
    X_val_fe["weather_lighting"] = lbl.transform(X_val_fe["weather_lighting"])

# Definition of RMSE evaluation function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate(model, name):
    model.fit(X_train_fe, y_train)
    preds = model.predict(X_val_fe)
    score = rmse(y_val, preds)
    print(f"{name} RMSE: {score:.6f}")
    return score

# Models
models = {
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.03,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist"
    ),
    "CatBoost": CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        loss_function="RMSE",
        verbose=0,
        random_seed=42
    )
}

# Comparison of models
results = {name: evaluate(model, name) for name, model in models.items()}

best_model = min(results, key=results.get)
print("\nğŸ�† Best Model:", best_model, "| RMSE =", results[best_model])



# --- STEP 1: Full train data ---
X_full = train.drop(columns=["id", "accident_risk"])
y_full = train["accident_risk"]

# Apply feature engineering to train and test
X_full_fe = add_features(X_full)
test_fe = add_features(test.copy())

# --- STEP 2: Encode all categorical columns ---
categorical_cols = ["road_type", "lighting", "weather", "time_of_day", "weather_lighting"]
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fit on combined train + test to handle new categories
combined = pd.concat([X_full_fe[categorical_cols], test_fe[categorical_cols]], axis=0)
encoder.fit(combined)

# Transform
X_full_fe[categorical_cols] = encoder.transform(X_full_fe[categorical_cols])
test_fe[categorical_cols] = encoder.transform(test_fe[categorical_cols])

# --- STEP 3: Train CatBoost (no cat_features needed) ---
best_model = CatBoostRegressor(
    iterations=800,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3,
    loss_function="RMSE",
    verbose=0,
    random_seed=42
)

best_model.fit(X_full_fe, y_full)

# --- STEP 4: Predict and submit ---
test_preds = best_model.predict(test_fe)
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": test_preds
})
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv created! Download and submit it.")

