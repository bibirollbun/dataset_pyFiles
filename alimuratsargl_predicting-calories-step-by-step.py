# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train.head()


train.info()


test.head()


test.info()


# Detect Missing Values
train.isnull().sum()


# Data Type Conversions
train['id'] = train['id'].astype('category')
test['id'] = test['id'].astype('category')


# Check with Boxplot

num_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

plt.figure(figsize=(20, 6))
for i, col in enumerate(num_features, 1):
    plt.subplot(1, len(num_features), i)
    sns.boxplot(data=train, y=col, color='lightblue')
    plt.title(col)
plt.tight_layout()
plt.show()


# Columns selected for outlier analysis (based on boxplots)
num_cols = ["Height", "Weight", "Heart_Rate", "Body_Temp"]

# Calculate IQR
Q1 = train[num_cols].quantile(0.25)
Q3 = train[num_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# 1. Winsorized training dataset (clipping applied, no rows removed)
train_winsorized = train.copy()
for col in num_cols:
    train_winsorized[col] = np.where(
        train_winsorized[col] < lower_bound[col], lower_bound[col],
        np.where(train_winsorized[col] > upper_bound[col], upper_bound[col], train_winsorized[col])
    )

# 2. Training dataset with outliers removed (rows outside bounds removed)
outlier_mask = (train[num_cols] < lower_bound).any(axis=1) | (train[num_cols] > upper_bound).any(axis=1)
train_removed = train[~outlier_mask].copy()

# Summary sizes
print(f"Original train size: {train.shape}")
print(f"Winsorized train size: {train_winsorized.shape}")
print(f"Outliers removed train size: {train_removed.shape}")


# Comparative Boxplot Visualization
for col in num_cols:
    plt.figure(figsize=(15, 5))

    # Original data
    plt.subplot(1, 3, 1)
    sns.boxplot(y=train[col], color='lightblue')
    plt.title(f'Original Data - {col}')

    # Outliers removed
    plt.subplot(1, 3, 2)
    sns.boxplot(y=train_removed[col], color='orange')
    plt.title(f'Outliers Removed - {col}')

    # Winsorized data
    plt.subplot(1, 3, 3)
    sns.boxplot(y=train_winsorized[col], color='green')
    plt.title(f'Winsorized - {col}')

    plt.tight_layout()
    plt.show()


def preprocess_features(df, is_test=False):
    import numpy as np
    import pandas as pd
    
    # 1. Age categories (between 20-79)
    bins = [20, 30, 40, 50, 60, 70, 80]
    labels = ['Young Adult (20-29)', 'Adult (30-39)', 'Middle Age (40-49)', 
              'Early Senior (50-59)', 'Senior (60-69)', 'Elderly (70-79)']
    df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    df['Age_Category'] = df['Age_Category'].astype('category')

    # 2. Calculate BMI and categorize (Check if Height and Weight exist)
    if all(col in df.columns for col in ["Weight", "Height"]):
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

        def categorize_bmi(bmi_value):
            if bmi_value < 18.5:
                return 'Underweight'
            elif 18.5 <= bmi_value < 25:
                return 'Normal weight'
            elif 25 <= bmi_value < 30:
                return 'Overweight'
            else:
                return 'Obese'
        df["BMI_category"] = df["BMI"].apply(categorize_bmi)
        bmi_categories = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
        df['BMI_category'] = pd.Categorical(df['BMI_category'], categories=bmi_categories, ordered=True)
    else:
        df["BMI"] = np.nan
        df["BMI_category"] = pd.Categorical([np.nan]*len(df))

    # 3. Calories / Duration ratio (Calories only present in train datasets)
    if not is_test and 'Calories' in df.columns and 'Duration' in df.columns:
        df["Calories_per_Duration"] = df["Calories"] / df["Duration"]
        df["Calories_per_Duration"] = df["Calories_per_Duration"].replace([np.inf, -np.inf], np.nan)
    else:
        df["Calories_per_Duration"] = np.nan

    # 4. Duration * Heart Rate (Exercise intensity indicator)
    if all(col in df.columns for col in ['Duration', 'Heart_Rate']):
        df["DurHR"] = df["Duration"] * df["Heart_Rate"]
    else:
        df["DurHR"] = np.nan

    # 5. Body Temperature Levels
    if 'Body_Temp' in df.columns:
        bins_temp = [35, 37.5, 39, 41.5, 45]
        labels_temp = ['Normal', 'Elevated', 'High', 'Very High']
        df['Body_Temp_Level'] = pd.cut(df['Body_Temp'], bins=bins_temp, labels=labels_temp, right=False)
        df['Body_Temp_Level'] = df['Body_Temp_Level'].astype('category')
    else:
        df['Body_Temp_Level'] = pd.Categorical([np.nan]*len(df))

    # 6. Heart Rate Levels
    if 'Heart_Rate' in df.columns:
        bins_hr = [0, 90, 110, 130]
        labels_hr = ['Low', 'Medium', 'High']
        df['Heart_Rate_Level'] = pd.cut(df['Heart_Rate'], bins=bins_hr, labels=labels_hr, right=False)
        df['Heart_Rate_Level'] = df['Heart_Rate_Level'].astype('category')
    else:
        df['Heart_Rate_Level'] = pd.Categorical([np.nan]*len(df))

    # 7. Max Heart Rate and Normalized Heart Rate
    if 'Age' in df.columns and 'Heart_Rate' in df.columns:
        df["Max_Heart_Rate"] = 220 - df["Age"]
        df["Normalized_Heart_Rate"] = df["Heart_Rate"] / df["Max_Heart_Rate"]
    else:
        df["Max_Heart_Rate"] = np.nan
        df["Normalized_Heart_Rate"] = np.nan

    # 8. New features based on squares and ratios
    if 'Duration' in df.columns:
        df["Duration_Squared"] = df["Duration"] ** 2
    else:
        df["Duration_Squared"] = np.nan

    if 'Heart_Rate' in df.columns and 'Weight' in df.columns:
        df["Heart_Rate_per_Weight"] = df["Heart_Rate"] / df["Weight"]
    else:
        df["Heart_Rate_per_Weight"] = np.nan

    if not is_test and 'Calories' in df.columns and 'Weight' in df.columns:
        df["Calories_per_Weight"] = df["Calories"] / df["Weight"]
    else:
        df["Calories_per_Weight"] = np.nan

    if 'BMI' in df.columns and 'Duration' in df.columns:
        df["BMI_Duration"] = df["BMI"] * df["Duration"]
    else:
        df["BMI_Duration"] = np.nan

    if 'Weight' in df.columns and 'Age' in df.columns:
        df["Weight_Age"] = df["Weight"] / df["Age"]
    else:
        df["Weight_Age"] = np.nan

    # 9. Replace infinite values with NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Categorical columns
    cat_cols = df.select_dtypes(['category']).columns
    non_cat_cols = df.columns.difference(cat_cols)

    # Fill NaNs in non-categorical columns with 0
    df[non_cat_cols] = df[non_cat_cols].fillna(0)

    # For categorical columns, add "Unknown" category if NaNs present and fill
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].cat.add_categories(['Unknown'])
            df[col] = df[col].fillna('Unknown')

    return df

# Usage:
train['Sex'] = train['Sex'].astype('category')
test['Sex'] = test['Sex'].astype('category')

train = preprocess_features(train, is_test=False)
train_removed = preprocess_features(train_removed, is_test=False)
train_winsorized = preprocess_features(train_winsorized, is_test=False)
test = preprocess_features(test, is_test=True)


# Post Feature Engineering Check
train.info()


train.isnull().sum()


# 1. Summary statistics of numerical variables
train.describe().T


# 2. Summary statistics of categorical variables
train.select_dtypes(include=['object', 'category']).describe().T


# 3. Distribution plots for numerical variables
num_cols = train.select_dtypes(include=["int64", "float64"]).columns
n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    train[col].hist(bins=30, ax=axes[i], color='skyblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# 4. Frequency distributions of categorical variables
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [col for col in cat_cols if col != 'id']

n_cols = 2
n_rows = (len(cat_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    order = train[col].value_counts().index
    sns.countplot(data=train, x=col, order=order, hue=col, palette="Set2", ax=ax)
    ax.set_title(f'Distribution of {col}')
    ax.tick_params(axis='x', rotation=45)
    
    # Annotate bars with counts
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# 5. Relationship between categorical variables and target variable (boxplots)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.boxplot(data=train, x=col, hue=col, y="Calories", palette="Set3", ax=axes[i])
    axes[i].set_title(f'{col} vs Calories')
    axes[i].tick_params(axis='x', rotation=45)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# --- 1) Correlation Matrix and Visualization ---

def plot_correlation_matrix(df, target='Calories', figsize=(15,10), cmap='coolwarm'):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap=cmap, fmt=".2f", square=True)
    plt.title('Correlation Matrix of Numerical Variables')
    plt.show()

    target_corr = corr[target].abs().sort_values(ascending=False)
    print(f"\nCorrelation strength ranking with {target}:\n", target_corr)
    return corr


# --- 2) Function to Remove Leakage and Highly Correlated Features ---

def drop_leakage_and_high_corr_features(df, target='Calories', corr_threshold=0.9, exclude_features=None):
    if exclude_features is None:
        exclude_features = []

    corr_matrix = df.corr(numeric_only=True)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    leakage_features = [col for col in df.columns if (target in col and col != target)]

    to_drop = set(leakage_features)

    for col in upper_tri.columns:
        for row in upper_tri.index:
            corr_value = upper_tri.loc[row, col]
            if pd.notnull(corr_value) and abs(corr_value) >= corr_threshold:
                if (col != target and row != target) and (col not in exclude_features and row not in exclude_features):
                    to_drop.add(row)

    print(f"\nRemoved {len(to_drop)} leakage or highly correlated features:\n", sorted(to_drop))
    df_reduced = df.drop(columns=list(to_drop))
    return df_reduced, list(to_drop)


# --- 3) Encoding Function ---

def encode_features(df, binary_cols=None, multi_cat_cols=None, label_encoders=None):
    df = df.copy()
    if binary_cols is None:
        binary_cols = ['Sex']

    if label_encoders is None:
        label_encoders = {}

    for col in binary_cols:
        if col in df.columns:
            if col not in label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
            else:
                le = label_encoders[col]
                df[col] = le.transform(df[col])

    if multi_cat_cols:
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    bool_cols = df.select_dtypes(include='bool').columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    return df, label_encoders


# --- 4) Scale Numerical Features ---

def scale_numeric_features(df, scaler=None, target='Calories', id_col='id'):
    df = df.copy()

    cols_to_exclude = [col for col in [target, id_col] if col in df.columns]
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(cols_to_exclude)

    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    return df, scaler


# --- 5) Preprocessing Pipeline Function ---

def preprocess_pipeline(df, to_drop, label_encoders, scaler, binary_cols, multi_cat_cols, target='Calories'):
    df_reduced = df.drop(columns=to_drop, errors='ignore')
    df_encoded, label_encoders = encode_features(df_reduced, binary_cols=binary_cols, multi_cat_cols=multi_cat_cols, label_encoders=label_encoders)
    df_scaled, scaler = scale_numeric_features(df_encoded, scaler=scaler, target=target)
    return df_scaled, label_encoders, scaler

# --- 6) Separate Features and Target ---

def get_features_and_target(df, target='Calories', id_col='id'):
    drop_cols = [col for col in [target, id_col] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target] if target in df.columns else None
    return X, y

# --- 7) RMSLE Calculation Function ---

def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# --- 8) Main Workflow ---

# Multi-category columns
multi_cat_cols = ['Age_Category', 'BMI_category', 'Body_Temp_Level', 'Heart_Rate_Level']
binary_cols = ['Sex']

# 8.1) Examine correlation matrix and target correlations
corr_matrix = plot_correlation_matrix(train, target='Calories')

# 8.2) Remove leakage and highly correlated features
exclude_features = []
train_reduced, dropped_features = drop_leakage_and_high_corr_features(train, target='Calories', corr_threshold=0.9, exclude_features=exclude_features)

# 8.3) Fit encoding and scaling on train set first
train_encoded, label_encoders = encode_features(train_reduced, binary_cols=binary_cols, multi_cat_cols=multi_cat_cols)
train_scaled, scaler = scale_numeric_features(train_encoded, target='Calories')

# 8.4) Apply preprocessing to other datasets (train_removed, train_winsorized, test)
train_removed_processed, _, _ = preprocess_pipeline(train_removed, dropped_features, label_encoders, scaler, binary_cols, multi_cat_cols)
train_winsorized_processed, _, _ = preprocess_pipeline(train_winsorized, dropped_features, label_encoders, scaler, binary_cols, multi_cat_cols)
test_processed, _, _ = preprocess_pipeline(test, dropped_features, label_encoders, scaler, binary_cols, multi_cat_cols)

# 8.5) Separate features and target variable
X_train, y_train = get_features_and_target(train_scaled, target='Calories')
X_removed, y_removed = get_features_and_target(train_removed_processed, target='Calories')
X_winsor, y_winsor = get_features_and_target(train_winsorized_processed, target='Calories')

if 'Calories' in test_processed.columns:
    X_test, y_test = get_features_and_target(test_processed, target='Calories')
else:
    X_test = test_processed.drop(columns=['id'], errors='ignore')
    y_test = None


# 9) Train-Validation Split Function

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

splits = {
    'train': split_data(X_train, y_train),
    'train_removed': split_data(X_removed, y_removed),
    'train_winsorized': split_data(X_winsor, y_winsor)
}


# 10) Models, Training, and Evaluation

model_constructors = {
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'LinearRegression': LinearRegression(),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
}

model_results = []

for set_name, (X_tr, X_val, y_tr, y_val) in splits.items():
    print(f"\nDataset: {set_name}")
    for model_name, model in model_constructors.items():
        model.fit(X_tr, y_tr)
        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)

        train_rmsle = rmsle(y_tr, y_tr_pred)
        val_rmsle = rmsle(y_val, y_val_pred)
        overfit_diff = train_rmsle - val_rmsle

        print(f"  Model: {model_name:15s} | Train RMSLE: {train_rmsle:.4f} | Validation RMSLE: {val_rmsle:.4f} | Overfit (Train-Val): {overfit_diff:.4f}")

        model_results.append({
            'Set': set_name,
            'Model': model_name,
            'Train RMSLE': train_rmsle,
            'Validation RMSLE': val_rmsle,
            'Overfitting (Train-Val)': overfit_diff,
            'Model_Obj': model,
            'X_val': X_val,
            'y_val': y_val,
            'y_val_pred': y_val_pred
        })

results_df = pd.DataFrame(model_results)


# 11) Plot results

plt.figure(figsize=(14, 7))
sns.barplot(data=results_df, x='Model', y='Validation RMSLE', hue='Set')
plt.title('Validation RMSLE by Model and Dataset')
plt.ylabel('Validation RMSLE (lower is better)')
plt.show()

plt.figure(figsize=(14, 7))
sns.barplot(data=results_df, x='Model', y='Overfitting (Train-Val)', hue='Set')
plt.title('Overfitting Difference (Train-Val) by Model and Dataset')
plt.ylabel('Overfitting Difference (smaller is better)')
plt.show()


# 12) Select best model and predict on test set

overfit_threshold = 0.05
filtered_models = results_df[results_df['Overfitting (Train-Val)'] <= overfit_threshold]

if filtered_models.empty:
    print("\nWarning: Overfitting difference > 0.05 in all models. Selecting model with lowest validation RMSLE.")
    best_model_row = results_df.loc[results_df['Validation RMSLE'].idxmin()]
else:
    best_model_row = filtered_models.loc[filtered_models['Validation RMSLE'].idxmin()]

print(f"\nSelected Best Model: {best_model_row['Model']} - Dataset: {best_model_row['Set']}")
best_model = best_model_row['Model_Obj']

# Predict on test set
y_test_pred = best_model.predict(X_test)


# 13) Save prediction results as DataFrame

test_results = pd.DataFrame({
    'id': test['id'],
    'Calories_Predicted': y_test_pred
})

# 13) Create submission file

submission = pd.DataFrame({
    'id': test['id'],
    'Calories': y_test_pred
})

# submission.to_csv('submission.csv', index=False)

print("\nPredictions successfully created and stored in the 'submission' DataFrame.")


# 14) Plot true vs predicted and residual diagnostics for selected model

def plot_model_diagnostics(y_true, y_pred, title_prefix='Model'):
    residuals = y_true - y_pred

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title_prefix} - True vs Predicted')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'{title_prefix} - Residual Plot')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7,5))
    sns.histplot(residuals, kde=True, bins=30, color='purple')
    plt.title(f'{title_prefix} - Residuals Distribution')
    plt.xlabel('Residual Values')
    plt.grid(True)
    plt.show()

# If validation true values exist, show validation performance and plots of selected model

if best_model_row['y_val'] is not None:
    print("\nValidation results for the best model:")
    plot_model_diagnostics(best_model_row['y_val'], best_model_row['y_val_pred'], title_prefix=f"{best_model_row['Model']} - {best_model_row['Set']}")
else:
    print("\nValidation true values are not available; cannot plot graphs.")

