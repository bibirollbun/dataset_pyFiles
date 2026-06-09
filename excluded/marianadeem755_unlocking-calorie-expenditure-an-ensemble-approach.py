# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# ignore warnings
import warnings
warnings.filterwarnings("ignore")



# Function to style tables with grey and light pink palette
def style_table(df):
    styled_df = df.style.set_table_styles([
        {"selector": "th", "props": [("color", "#ffffff"), ("background-color", "#a6a6a6")]}  # Dark grey headers
    ]).set_properties(**{
        "text-align": "center",
        "font-size": "14px",
        "color": "#4b0082",  # Indigo text
        "background-color": "#f3f3f3",  # Light grey background
    }).hide(axis="index")
    return styled_df.to_html()

# Function to create styled heading with emojis using a grey gradient + pink border highlight
def styled_heading(text, background_color='linear-gradient(135deg, #e0e0e1, #cfcfcf)', text_color='#4b0082'):
    return f"""
    <div style="
        text-align: center;
        background: {background_color};
        color: {text_color};
        padding: 18px;
        font-size: 22px;
        font-weight: bold;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0px 5px 10px rgba(75, 0, 130, 0.2);
        border: 2px solid #f8c8dc;  /* Light pink highlight */
    ">
        {text}
    </div>
    """

# Function to display dataset analysis with updated grey theme
def print_dataset_analysis(dataset, dataset_name, n_top=5, heading_color='linear-gradient(135deg, #e0e0e1, #cfcfcf)', text_color='#4b0082'):
    heading = styled_heading(f"📊 {dataset_name} Overview", heading_color, text_color)
    display(HTML(heading))

    def subheader(text):
        return f"<h2 style='font-size: 18px; color: #4b0082; margin-top: 15px; text-decoration: underline;'>{text}</h2>"

    display(HTML(subheader("📏 Shape of the Dataset")))
    display(HTML(f"<p style='font-size: 16px; color: #4b0082;'>{dataset.shape[0]} rows and {dataset.shape[1]} columns</p>"))

    display(HTML(subheader("🔍 First 5 Rows")))
    display(HTML(style_table(dataset.head(n_top))))

    display(HTML(subheader("📊 Summary Statistics")))
    display(HTML(style_table(dataset.describe())))

    display(HTML(subheader("🔧 Null Values")))
    null_counts = dataset.isnull().sum()
    null_columns = null_counts[null_counts > 0]
    if null_columns.sum() == 0:
        display(HTML("<p style='font-size: 16px; color: #4b0082;'>✅ No null values found.</p>"))
    else:
        null_columns_df = pd.DataFrame({
            'Column Name': null_columns.index,
            'Null Values': null_columns.values
        })
        display(HTML(style_table(null_columns_df)))

    display(HTML(subheader("♻️ Duplicate Rows")))
    duplicate_count = dataset.duplicated().sum()
    display(HTML(f"<p style='font-size: 16px; color: #4b0082;'>{duplicate_count} duplicate rows found.</p>"))

    display(HTML(subheader("🗂️ Data Types")))
    dtypes_table = pd.DataFrame({
        'Column Name': dataset.columns,
        'Data Type': [dataset[col].dtype for col in dataset.columns]
    })
    display(HTML(style_table(dtypes_table)))

    display(HTML(subheader("📋 Column Names")))
    display(HTML(f"<p style='font-size: 16px; color: #4b0082;'>{', '.join(dataset.columns)}</p>"))

    display(HTML(subheader("🔢 Unique Values")))
    unique_values_table = pd.DataFrame({
        'Column Name': dataset.columns,
        'Unique Values': [', '.join(map(str, dataset[col].unique()[:7])) + (', ...' if len(dataset[col].unique()) > 7 else '') for col in dataset.columns]
    })
    display(HTML(style_table(unique_values_table)))

# Load datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Analyze datasets with elegant grey theme
print_dataset_analysis(df_train, "Training Data")
print_dataset_analysis(df_test, "Test Data")
print_dataset_analysis(sample_sub, "Sample Submission")



# Map Gender to numerical
for df in [df_train, df_test]:
    df["Gender_Num"] = df["Sex"].map({"male": 1, "female": 0})
    
# Remove duplicates from training if any
df_train = df_train.drop_duplicates()

# Feature: BMI (Body Mass Index)
for df in [df_train, df_test]:
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

# Feature: Effort ratio
for df in [df_train, df_test]:
    df["Effort"] = df["Heart_Rate"] / df["Duration"]

# Feature: Age buckets - will help capture nonlinearity
for df in [df_train, df_test]:
    df["Age_bin"] = pd.cut(df["Age"], bins=8, labels=False)

# Feature: Interaction terms
for df in [df_train, df_test]:
    df["HR_x_Dur"] = df["Heart_Rate"] * df["Duration"]
    df["Temp_x_Dur"] = df["Body_Temp"] * df["Duration"]
    df["BMI_x_Age"] = df["BMI"] * df["Age"]
    df["Effort_x_Age"] = df["Effort"] * df["Age"]
    
# Drop columns not used as features
drop_cols = ["Sex", "ID"]
df_train = df_train.drop(columns=drop_cols, errors='ignore')
df_test = df_test.drop(columns=drop_cols, errors='ignore')



features = [c for c in df_train.columns if c not in ["Calories"]]

X_train = df_train[features]
y_train = df_train["Calories"]

X_test = df_test[features]

# Log-transform target for RMSLE optimization
y_train_log = np.log1p(y_train)



# Use KBinsDiscretizer on Duration to create bins for stratified KFold
duration_vals = df_train["Duration"].values.reshape(-1, 1)
kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
kbins.fit(duration_vals)
y_bins = kbins.transform(duration_vals).astype(int).flatten()



cat_params = {
    "iterations": 3000,
    "learning_rate": 0.01,
    "depth": 10,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "l2_leaf_reg": 4,
    "early_stopping_rounds": 150,
    "verbose": 100,
    "task_type": "GPU",
    "random_seed": 42,
    "cat_features": ["Gender_Num", "Age_bin"]
}

cat_oof = np.zeros(len(X_train))
cat_preds = np.zeros(len(X_test))
cat_scores = []

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_bins)):
    model = CatBoostRegressor(**cat_params)
    model.fit(X_train.iloc[train_idx], y_train_log.iloc[train_idx],
              eval_set=(X_train.iloc[val_idx], y_train_log.iloc[val_idx]))
    
    cat_oof[val_idx] = model.predict(X_train.iloc[val_idx])
    cat_preds += model.predict(X_test) / folds.n_splits
    
    fold_score = np.sqrt(mean_squared_log_error(np.expm1(y_train_log.iloc[val_idx]), np.expm1(cat_oof[val_idx])))
    print(f"[Fold {fold+1}] CatBoost RMSLE: {fold_score:.6f}")
    cat_scores.append(fold_score)

print(f"\nCatBoost Average RMSLE: {np.mean(cat_scores):.6f}")



lgb_params = {
    "n_estimators": 3500,
    "learning_rate": 0.01,
    "num_leaves": 48,
    "colsample_bytree": 0.8,
    "subsample": 0.9,
    "max_depth": 14,
    "random_state": 42,
    "verbose": -1,
    "early_stopping_rounds": 150
}

lgb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    model = LGBMRegressor(**lgb_params)
    model.fit(X_train.iloc[train_idx], y_train_log.iloc[train_idx],
              eval_set=[(X_train.iloc[val_idx], y_train_log.iloc[val_idx])])
    
    lgb_oof[val_idx] = model.predict(X_train.iloc[val_idx])
    lgb_preds += model.predict(X_test) / kf.n_splits

    fold_score = np.sqrt(mean_squared_log_error(np.expm1(y_train_log.iloc[val_idx]), np.expm1(lgb_oof[val_idx])))
    print(f"[Fold {fold+1}] LightGBM RMSLE: {fold_score:.6f}")



xgb_params = {
    "max_depth": 10,
    "colsample_bytree": 0.7,
    "subsample": 0.9,
    "n_estimators": 3500,
    "learning_rate": 0.01,
    "gamma": 0.01,
    "tree_method": "gpu_hist",
    "eval_metric": "rmse",
    "random_state": 42,
    "early_stopping_rounds": 150
}

xgb_oof = np.zeros(len(X_train))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    model = XGBRegressor(**xgb_params)
    model.fit(X_train.iloc[train_idx], y_train_log.iloc[train_idx],
              eval_set=[(X_train.iloc[val_idx], y_train_log.iloc[val_idx])], verbose=False)
    
    xgb_oof[val_idx] = model.predict(X_train.iloc[val_idx])
    xgb_preds += model.predict(X_test) / kf.n_splits
    
    fold_score = np.sqrt(mean_squared_log_error(np.expm1(y_train_log.iloc[val_idx]), np.expm1(xgb_oof[val_idx])))
    print(f"[Fold {fold+1}] XGBoost RMSLE: {fold_score:.6f}")



# Weighted average based on validation performance
final_pred = (
    0.45 * np.expm1(cat_preds) +
    0.35 * np.expm1(lgb_preds) +
    0.20 * np.expm1(xgb_preds)
)

# Clip predictions to reasonable range (1 to max observed in train)
final_pred = np.clip(final_pred, 1, df_train["Calories"].max())



sample_sub["Calories"] = final_pred
sample_sub.to_csv("submission.csv", index=False)
print("Submission file created!")



#Display First Few rows of Submission File
sample_sub.head(10)

