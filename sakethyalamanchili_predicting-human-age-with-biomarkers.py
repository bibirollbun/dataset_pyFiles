import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import keras
import joblib
import random 
import os

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline 

# Linear Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# Ensemble Methods
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

# Boosting Methods
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Support Vector Regression
from sklearn.svm import SVR

# Deep Neural Network
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# Reproducibility setup
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

set_seed(42)


data = pd.read_csv("/kaggle/input/precision-health-predicting-human-age-with-biomark/Train.csv")
data


data.columns


data.describe()


data.info()


# Define X and y
X = data.drop(columns=['Age (years)', 'ID'])  
y = data['Age (years)']

# Splitting the Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_Val: {X_val.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_val: {y_val.shape}")


# Count missing values
missing_counts = data.isna().sum()

# Calculate percentage of missing values
missing_percentages = (missing_counts * 100) / len(data)

# Display
print("Missing Values (Count):")
print(missing_counts[missing_counts > 0])

print("\nMissing Values (Percentage):")
print(missing_percentages[missing_percentages > 0].round(2))


# Select columns with missing values
missing_columns = ['Alcohol Consumption', 'Chronic Diseases', 'Medication Use', 'Family History', 'Education Level']

# Set up the plotting grid
plt.figure(figsize=(16, 10), dpi=500)

# Plot distribution for each column
for i, col in enumerate(missing_columns, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=X_train, x=col, palette='pastel', order=X_train[col].value_counts().index)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()


X_train['Alcohol Consumption'] = X_train['Alcohol Consumption'].fillna('Occasional')
X_train['Chronic Diseases'] = X_train['Chronic Diseases'].fillna('Hypertension')
X_train['Medication Use'] = X_train['Medication Use'].fillna('Regular')
X_train['Family History'] = X_train['Family History'].fillna('Diabetes')
X_train['Education Level'] = X_train['Education Level'].fillna('High School')


X_train.isna().sum()


# Splitting Blood Pressure into Systolic and Diastolic
X_train[['Systolic BP', 'Diastolic BP']] = X_train['Blood Pressure (s/d)'].str.split('/', expand=True).astype(float)
X_train.drop(columns='Blood Pressure (s/d)', inplace=True)
X_train


# Binary Encode Gender
X_train['Gender'] = X_train['Gender'].map({'Male': 1, 'Female': 0})


numerical_cols = [
    'Height (cm)', 'Weight (kg)', 'Cholesterol Level (mg/dL)', 'BMI',
    'Blood Glucose Level (mg/dL)', 'Bone Density (g/cmÂ²)', 'Vision Sharpness',
    'Hearing Ability (dB)', 'Cognitive Function', 'Stress Levels',
    'Pollution Exposure', 'Sun Exposure', 'Systolic BP', 'Diastolic BP'
]

sns.set(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 18), dpi=500)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.boxplot(y=X_train[col], ax=axes[i], color=sns.color_palette("pastel")[i % 10])
    axes[i].set_title(col, fontsize=10, weight='bold')
    axes[i].set_ylabel("")
    axes[i].grid(True, linestyle="--", linewidth=0.5)

# Remove extra empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Overall layout
fig.suptitle("Outlier Detection in Training Data (Boxplots)", fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.show()


# Define leakage-free IQR-based outlier removal on training data

def get_iqr_bounds(series):
    """Calculate IQR bounds for a pandas Series."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

def filter_outliers_iqr(df, ref_df, cols):
    """Remove rows in df where values in cols are outliers based on reference df."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col in cols:
        lower, upper = get_iqr_bounds(ref_df[col])
        mask &= df[col].between(lower, upper)
    return df[mask]

# Step 2: Filter training data only using bounds from training data
X_train_clean = filter_outliers_iqr(X_train, X_train, numerical_cols)
y_train_clean = y_train.loc[X_train_clean.index]

# Step 3: Report change in shape
X_train.shape, X_train_clean.shape


y_train_clean = y_train.loc[X_train_clean.index]
y_train_clean.shape


sns.set(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 18), dpi=500)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.boxplot(y=X_train_clean[col], ax=axes[i], color=sns.color_palette("pastel")[i % 10])
    axes[i].set_title(col, fontsize=10, weight='bold')
    axes[i].set_ylabel("")
    axes[i].grid(True, linestyle="--", linewidth=0.5)

# Remove extra empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Overall layout
fig.suptitle("Outlier Detection in Training Data (Boxplots) after Outlier Removal.", fontsize=16, weight='bold', y=1.02)
plt.tight_layout()
plt.show()


numerical_cols = ['Height (cm)', 'Weight (kg)', 'Cholesterol Level (mg/dL)', 'BMI',
                  'Blood Glucose Level (mg/dL)', 'Bone Density (g/cmÂ²)', 'Vision Sharpness',
                  'Hearing Ability (dB)', 'Cognitive Function', 'Stress Levels',
                  'Pollution Exposure', 'Sun Exposure', 'Systolic BP', 'Diastolic BP']

categorical_cols = ['Smoking Status', 'Alcohol Consumption', 'Diet', 'Chronic Diseases',
                    'Medication Use', 'Family History', 'Mental Health Status',
                    'Sleep Patterns', 'Education Level', 'Income Level', 'Physical Activity Level']

# Create ColumnTransformer with scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
    ],
    remainder='passthrough'  # This will include Gender
)

# Fit and transform
X_preprocessed = preprocessor.fit_transform(X_train_clean)

# Get column names
scaled_features = numerical_cols
encoded_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
remainder_features = ['Gender']
all_feature_names = list(scaled_features) + list(encoded_features) + remainder_features

# Create final DataFrame
X_scaled_data = pd.DataFrame(X_preprocessed, columns=all_feature_names)

# For unscaled version, replace scaler with 'passthrough'
unscaled_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols)
    ],
    remainder='passthrough'
)

X_unscaled = unscaled_preprocessor.fit_transform(X_train_clean)
unscaled_feature_names = list(numerical_cols) + list(
    unscaled_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
) + remainder_features

X_unscaled_data = pd.DataFrame(X_unscaled, columns=unscaled_feature_names)


X_scaled_data


X_unscaled_data


# Save both preprocessors
joblib.dump(preprocessor, 'preprocessor_scaled.pkl')


joblib.dump(unscaled_preprocessor, 'preprocessor_unscaled.pkl')


# Preprocessing function to clean raw data
def preprocess_input_dataframe(df):
    df = df.copy()

    # Fill missing categorical values
    df['Alcohol Consumption'] = df['Alcohol Consumption'].fillna('Occasional')
    df['Chronic Diseases'] = df['Chronic Diseases'].fillna('Hypertension')
    df['Medication Use'] = df['Medication Use'].fillna('Regular')
    df['Family History'] = df['Family History'].fillna('Diabetes')
    df['Education Level'] = df['Education Level'].fillna('High School')

    # Encode gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Split Blood Pressure
    if 'Blood Pressure (s/d)' in df.columns:
        df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure (s/d)'].str.split('/', expand=True).astype(float)
        df.drop(columns='Blood Pressure (s/d)', inplace=True)

    # Drop ID if present
    if 'ID' in df.columns:
        df.drop(columns='ID', inplace=True)

    return df

def transform_with_preprocessor(df, preprocessor_path):
    import joblib
    import pandas as pd

    # Load preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # Transform input
    transformed_array = preprocessor.transform(df)

    # Attempt to retrieve feature names
    try:
        num_features = preprocessor.transformers_[0][2]  # numerical_cols
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(preprocessor.transformers_[1][2])  # get_feature_names_out(categorical_cols)
        passthrough_features = ['Gender']  # remainder='passthrough'
        all_feature_names = list(num_features) + list(cat_features) + passthrough_features
    except Exception as e:
        all_feature_names = [f"feature_{i}" for i in range(transformed_array.shape[1])]

    # Return as DataFrame
    return pd.DataFrame(transformed_array, columns=all_feature_names, index=df.index)


def preprocess_and_transform_scaled(df):
    """
    Preprocess and transform input data using the scaled pipeline.
    Used for models that require normalized features.
    """
    df_clean = preprocess_input_dataframe(df)
    return transform_with_preprocessor(df_clean, "preprocessor_scaled.pkl")

def preprocess_and_transform_unscaled(df):
    """
    Preprocess and transform input data using the unscaled pipeline.
    Used for tree-based models that do NOT need feature scaling.
    """
    df_clean = preprocess_input_dataframe(df)
    return transform_with_preprocessor(df_clean, "preprocessor_unscaled.pkl")


# For Validation

# For SVR, Linear, Ridge, etc.
X_val_scaled = preprocess_and_transform_scaled(X_val)

# For Random Forest, XGBoost, etc.
X_val_unscaled = preprocess_and_transform_unscaled(X_val)


X_val_scaled


X_val_unscaled


# Confirm shapes and column name matches
{
    "X_val_scaled_shape": X_val_scaled.shape,
    "X_val_unscaled_shape": X_val_unscaled.shape,
    "X_train_scaled_shape": X_scaled_data.shape,
    "X_train_unscaled_shape": X_unscaled_data.shape,
    "scaled_columns_match": list(X_scaled_data.columns) == list(X_val_scaled.columns),
    "unscaled_columns_match": list(X_unscaled_data.columns) == list(X_val_unscaled.columns)
}



# Define models that need scaling
scaled_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net": ElasticNet(),
    "Support Vector Regression": SVR()
}

# Define models that don't need scaling
unscaled_models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "AdaBoost": AdaBoostRegressor()
}


# Evaluate models
results = []

for name, model in scaled_models.items():
    model.fit(X_scaled_data, y_train_clean)
    y_pred = model.predict(X_val_scaled)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_val, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
        'RÂ² Score': r2_score(y_val, y_pred)
    })

for name, model in unscaled_models.items():
    model.fit(X_unscaled_data, y_train_clean)
    y_pred = model.predict(X_val_unscaled)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_val, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
        'RÂ² Score': r2_score(y_val, y_pred)
    })

# Convert to DataFrame
results_df = pd.DataFrame(results).sort_values(by='RMSE')
results_df


results_df_sorted = results_df.sort_values(by='RMSE')

# Set plot style
sns.set(style="whitegrid", context='notebook', font_scale=1.1)

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 12), dpi=500)

# Plot MAE
sns.barplot(x='MAE', y='Model', data=results_df_sorted, ax=axes[0], palette='Blues_d')
axes[0].set_title('Mean Absolute Error (MAE)', fontsize=14, weight='bold')
axes[0].set_xlabel('MAE', fontsize=12)
axes[0].set_ylabel('')
axes[0].bar_label(axes[0].containers[0], fmt='%.2f', label_type='edge', padding=3)

# Plot RMSE
sns.barplot(x='RMSE', y='Model', data=results_df_sorted, ax=axes[1], palette='Oranges_d')
axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=14, weight='bold')
axes[1].set_xlabel('RMSE', fontsize=12)
axes[1].set_ylabel('')
axes[1].bar_label(axes[1].containers[0], fmt='%.2f', label_type='edge', padding=3)

# Plot RÂ² Score
sns.barplot(x='RÂ² Score', y='Model', data=results_df_sorted, ax=axes[2], palette='Greens_d')
axes[2].set_title('RÂ² Score (Explained Variance)', fontsize=14, weight='bold')
axes[2].set_xlabel('RÂ² Score', fontsize=12)
axes[2].set_ylabel('')
axes[2].bar_label(axes[2].containers[0], fmt='%.2f', label_type='edge', padding=3)

# Overall layout adjustments
plt.suptitle("Model Performance Comparison", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


X_train_scaled_data = X_scaled_data
X_val_scaled_data = X_val_scaled
y_train_data = y_train_clean
y_val_data = y_val

# Define model configurations to experiment with
dnn_configs = [
    {"name": "Baseline", "layers": [128, 64, 32], "dropout": 0.2, "optimizer": Adam(0.001), "batch_norm": False},
    {"name": "Deep_Regularized", "layers": [256, 128, 64], "dropout": 0.3, "optimizer": Adam(0.001), "batch_norm": False},
    {"name": "BatchNorm_Model", "layers": [128, 64, 32], "dropout": 0.2, "optimizer": Adam(0.001), "batch_norm": True},
    {"name": "RMSProp_Model", "layers": [128, 64], "dropout": 0.2, "optimizer": RMSprop(0.001), "batch_norm": False},
    {"name": "Smaller_LR", "layers": [128, 64, 32], "dropout": 0.2, "optimizer": Adam(0.0005), "batch_norm": False}
]

# Function to build and train model
def run_dnn_experiment(config):
    model = Sequential()
    for i, units in enumerate(config['layers']):
        model.add(Dense(units, activation='relu', input_shape=(X_train_scaled_data.shape[1],) if i == 0 else None))
        if config['batch_norm']:
            model.add(BatchNormalization())
        model.add(Dropout(config['dropout']))
    model.add(Dense(1))

    model.compile(optimizer=config['optimizer'], loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train_scaled_data, y_train_data,
              validation_data=(X_val_scaled_data, y_val_data),
              epochs=100, batch_size=32, verbose=0,
              callbacks=[early_stop])

    preds = model.predict(X_val_scaled_data).flatten()
    mae = mean_absolute_error(y_val_data, preds)
    rmse = np.sqrt(mean_squared_error(y_val_data, preds))
    r2 = r2_score(y_val_data, preds)

    return {
        "Model": config['name'],
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "RÂ² Score": round(r2, 3)
    }

# Run all experiments and collect results
dnn_results = [run_dnn_experiment(cfg) for cfg in dnn_configs]

# Convert to DataFrame for display
dnn_results_df = pd.DataFrame(dnn_results).sort_values(by="RMSE")
dnn_results_df


# Extract best-performing Ridge Regression result
ridge_row = results_df[results_df['Model'] == 'Ridge Regression'].iloc[0]

# Extract best-performing DNN (RMSProp) result
rmsprop_row = dnn_results_df[dnn_results_df['Model'] == 'RMSProp_Model'].iloc[0]

comparison_dynamic = pd.DataFrame({
    'Model': ['Ridge Regression', 'RMSProp_Model'],
    'MAE': [ridge_row['MAE'], rmsprop_row['MAE']],
    'RMSE': [ridge_row['RMSE'], rmsprop_row['RMSE']],
    'RÂ² Score': [ridge_row['RÂ² Score'], rmsprop_row['RÂ² Score']]
})

sns.set(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), dpi=500)

# Custom palettes
palettes = [
    sns.color_palette("coolwarm", 2),
    sns.color_palette("YlOrBr", 2),
    sns.color_palette("viridis", 2)
]

metrics = ['MAE', 'RMSE', 'RÂ² Score']
titles = [
    'Mean Absolute Error (MAE)',
    'Root Mean Squared Error (RMSE)',
    'RÂ² Score (Explained Variance)'
]

# Iterate over metrics to create subplots
for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.barplot(x='Model', y=metric, data=comparison_dynamic, ax=ax, palette=palettes[i])
    ax.set_title(titles[i], fontsize=14, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    
    # Add value labels with spacing above the bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f' if metric == 'RÂ² Score' else '%.2f',
                     label_type='edge', padding=5)

# Adjust layout and title
plt.suptitle("Ridge Regression vs RMSProp DNN â€“ Final Model Performance", fontsize=18, weight='bold', y=1.05)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Save RMSProp DNN model

# Rebuild RMSProp model
model_rmsprop = Sequential([
    Dense(128, activation='relu', input_shape=(X_scaled_data.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Use explicit loss function object instead of string
model_rmsprop.compile(optimizer=RMSprop(0.001), loss=MeanSquaredError(), metrics=['mae'])

# Train the model
model_rmsprop.fit(X_scaled_data, y_train_clean,
                  validation_data=(X_val_scaled, y_val),
                  epochs=100,
                  batch_size=32,
                  verbose=0)

# Save model in a way that's safe to reload
model_rmsprop.save("rmsprop_dnn_best_model.h5")


# Load the best DNN model
rmsprop_dnn_model = load_model("rmsprop_dnn_best_model.h5")

# Get predictions for baseline (unshuffled)
baseline_preds = rmsprop_dnn_model.predict(X_val_scaled).flatten()
baseline_mse = mean_squared_error(y_val, baseline_preds)

# Store importances
importances = {}

for i, col in enumerate(X_val_scaled.columns):
    X_temp = X_val_scaled.copy()
    X_temp[col] = np.random.permutation(X_temp[col])  # Shuffle one feature
    shuffled_preds = rmsprop_dnn_model.predict(X_temp).flatten()
    shuffled_mse = mean_squared_error(y_val, shuffled_preds)

    # Importance: How much the MSE increased
    importances[col] = shuffled_mse - baseline_mse

# Convert to DataFrame
importance_df = pd.DataFrame({
    "Feature": list(importances.keys()),
    "Importance": list(importances.values())
}).sort_values(by="Importance", ascending=False)

# Get top 10
top_10_features = importance_df.head(10)
top_10_features


plt.figure(figsize=(10, 6), dpi=500)
sns.barplot(data=top_10_features, y='Feature', x='Importance', palette='viridis')
plt.title("Top 10 Important Features (Permutation Importance - RMSProp DNN)")
plt.xlabel("Increase in MSE After Shuffling")
plt.ylabel("Feature")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Load the model
rmsprop_dnn_model = load_model("rmsprop_dnn_best_model.h5")

# Preprocess test data
test_data = pd.read_csv('/kaggle/input/precision-health-predicting-human-age-with-biomark/Test.csv')
X_test_scaled = preprocess_and_transform_scaled(test_data)

# Predict
test_predictions = rmsprop_dnn_model.predict(X_test_scaled).flatten()
test_predictions = np.round(test_predictions).astype(int)

# Prepare submission
submission = pd.DataFrame({
    "ID": test_data["ID"],
    "Gender": test_data["Gender"],
    "Age(years)": test_predictions
})

# Save CSV
submission.to_csv("submission.csv", index=False)
submission

