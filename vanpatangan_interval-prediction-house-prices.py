import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
#sample_sub = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')


def check(df):
    
    # preallocate arrays for better performance 
    n_cols = len(df.columns)
    columns = df.columns.to_numpy()
    dtypes = np.array([df[col].dtype for col in columns])
    counts = df.count().to_numpy()
    nunique = df.nunique().to_numpy()
    nulls = df.isnull().sum().to_numpy()
    duplicates = np.array([df.duplicated().sum()] * n_cols)  
    
    # Create dataframe directly from numpy arrays
    df_check = pd.DataFrame({
        'column': columns,
        'dtype': dtypes,
        'instances': counts,
        'unique': nunique,
        'sum_null': nulls,
        'duplicates': duplicates
    })
    
    return df_check


print("Training Data Summary")
display(check(train))
#display(train.head())

print("Test Data Summary")
display(check(test))
#display(test.head())


# Numerical features 
numerical_features = train.select_dtypes(include=['float64', 'int64']).columns

# Determine number of rows and columns for the grid
num_features = len(numerical_features)
num_cols = 4
num_rows = (num_features + num_cols - 1) // num_cols

# Set a custom color palette
colors = sns.color_palette("husl", n_colors=num_features)

# Create subplots with a smaller figure size to reduce rendering time
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4*num_rows), facecolor='white')
axes = axes.flatten()

# Optimize histogram plotting
for i, feature in enumerate(numerical_features):
    # Use a subset of data 
    data = train[feature].dropna()  # Drop NaNs upfront
    sns.histplot(data=data, bins=30, ax=axes[i], kde=False, color=colors[i], 
                 edgecolor='black', linewidth=0.5, stat='count')
    axes[i].set_title(f'{feature}', fontsize=12, pad=8)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='both', labelsize=8)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Minimize layout adjustments
plt.suptitle('Histograms of Numerical Features', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


# Specify numerical columns in train
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns

# Calculate correlation matrix for numerical features only
corr_matrix = train[numerical_cols].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix for Numerical Features in train")
plt.show()


import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split


# Define feature types
numeric_features = [
    'sqft', 'sqft_lot', 'land_val', 'imp_val', 'year_built', 'year_reno', 
    'sqft_fbsmt', 'garb_sqft', 'gara_sqft', 'grade', 'fbsmt_grade', 
    'condition', 'stories', 'beds', 'bath_full', 'bath_3qtr', 'bath_half',
    'latitude', 'longitude'
]
categorical_features = ['city', 'zoning', 'join_status', 'sale_warning']
high_cardinality_features = ['subdivision', 'submarket']
binary_features = [
    'wfnt', 'golf', 'greenbelt', 'view_rainier', 'view_olympics', 
    'view_cascades', 'view_territorial', 'view_skyline', 'view_sound', 
    'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other', 
    'noise_traffic'  # Added noise_traffic to binary features
]

# Feature engineering function
def feature_engineering(df):
    df = df.copy()
    
    # Datetime features
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["sale_year"] = df["sale_date"].dt.year
    df["sale_month"] = df["sale_date"].dt.month
    df["sale_day"] = df["sale_date"].dt.day
    df["sale_dow"] = df["sale_date"].dt.weekday
    df["sale_quarter"] = df["sale_date"].dt.quarter
    df["season"] = df["sale_month"] % 12 // 3

    # Building age and renovation
    df["age_at_sale"] = df["sale_year"] - df["year_built"]
    df["reno_age_at_sale"] = df["sale_year"] - df["year_reno"]
    df["is_renovated"] = (df["year_reno"] > df["year_built"]).astype(int)

    # Size ratios with protection against division by zero
    df["lot_sqft_ratio"] = df["sqft"] / (df["sqft_lot"].clip(lower=1))
    df["garage_total"] = df["garb_sqft"] + df["gara_sqft"]
    df["bath_total"] = df["bath_full"] + df["bath_3qtr"] + 0.5 * df["bath_half"]
    df["sqft_per_bed"] = df["sqft"] / (df["beds"].clip(lower=1))
    df["sqft_per_bath"] = df["sqft"] / (df["bath_total"].clip(lower=1))

    # Logtransformed features
    df["log_sqft"] = np.log1p(df["sqft"])
    df["log_sqft_lot"] = np.log1p(df["sqft_lot"])

    # Aggregate view features
    view_cols = [
        'view_rainier', 'view_olympics', 'view_cascades', 'view_territorial',
        'view_skyline', 'view_sound', 'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other'
    ]
    df["total_views"] = df[view_cols].sum(axis=1)

    # Proximity interaction
    df["lat_lon"] = df["latitude"].astype(str) + "_" + df["longitude"].astype(str)

    # Binary features 
    df['has_waterfront'] = df['wfnt'].clip(0, 1)
    df['is_golf'] = df['golf'].clip(0, 1)
    df['is_greenbelt'] = df['greenbelt'].clip(0, 1)
    df['has_traffic_noise'] = df['noise_traffic'].clip(0, 1)
    df['has_rainier_view'] = df['view_rainier'].clip(0, 1)
    df['has_olympics_view'] = df['view_olympics'].clip(0, 1)
    df['has_cascades_view'] = df['view_cascades'].clip(0, 1)
    df['has_view'] = (df["total_views"] > 0).astype(int)

    return df

# Apply feature engineering
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

# Update feature lists with new features
numeric_features += [
    'sale_year', 'sale_month', 'sale_day', 'sale_dow', 'sale_quarter', 'season',
    'age_at_sale', 'reno_age_at_sale', 'lot_sqft_ratio', 'garage_total',
    'bath_total', 'sqft_per_bed', 'sqft_per_bath', 'log_sqft', 'log_sqft_lot', 'total_views'
]
binary_features += [
    'is_renovated', 'has_waterfront', 'is_golf', 'is_greenbelt', 
    'has_traffic_noise', 'has_rainier_view', 'has_olympics_view', 
    'has_cascades_view', 'has_view'
]

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

high_cardinality_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('target', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('high_card', high_cardinality_transformer, high_cardinality_features),
        ('binary', 'passthrough', binary_features)
    ])

# Apply preprocessing
y_train = np.log1p(train_fe['sale_price'])  # Log transform target
X_train = train_fe.drop(columns=['sale_price', 'id', 'sale_date', 'sale_nbr'])
X_test = test_fe.drop(columns=['id', 'sale_date', 'sale_nbr'])

# Fit and transform with TargetEncoder
X_train_transformed = preprocessor.fit_transform(X_train, y=y_train)
X_test_transformed = preprocessor.transform(X_test)

print("New Features and Encoding done! ðŸ¤–ðŸ˜Œ")


# Train Validation Split
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_transformed, y_train, test_size=0.2, random_state=42)


from sklearn.metrics import mean_pinball_loss
import optuna
import logging

# Set Optuna to only show warnings or errors
optuna.logging.set_verbosity(logging.WARNING)


# Train Quantile LGBM
def train_lgb_quantile(X_train, y_train, X_val, y_val, alpha, params):
    params = params.copy()
    params.update({
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "verbosity": -1,
        "random_state": 42
    })

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # No evaluation logs
        ]
    )
    return model


# Objective for Optuna
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "max_depth": trial.suggest_int("max_depth", -1, 10)
    }

    model = train_lgb_quantile(X_train_split, y_train_split, X_val, y_val, alpha=0.5, params=params)
    preds = model.predict(X_val)
    loss = mean_pinball_loss(y_val, preds, alpha=0.5)
    return loss

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
#print("Best Params:", best_params)

# Final model params
base_params = best_params.copy()
base_params.update({
    "bagging_freq": 1,
    "random_state": 42
})

# Train final models
model_lower = train_lgb_quantile(X_train_split, y_train_split, X_val, y_val, alpha=0.05, params=base_params)
model_median = train_lgb_quantile(X_train_split, y_train_split, X_val, y_val, alpha=0.5, params=base_params)
model_upper = train_lgb_quantile(X_train_split, y_train_split, X_val, y_val, alpha=0.95, params=base_params)

# Predict (log scale)
pred_lower = model_lower.predict(X_val)
pred_upper = model_upper.predict(X_val)
actual_y = np.expm1(y_val)
pred_lower_exp = np.expm1(pred_lower)
pred_upper_exp = np.expm1(pred_upper)
median_pred = np.expm1(model_median.predict(X_val))

# Winkler Score Function
def winkler_score(y_true, y_lower, y_upper, alpha):
    score = np.where(
        (y_true >= y_lower) & (y_true <= y_upper),
        y_upper - y_lower,
        (y_upper - y_lower) + (2 / alpha) * np.where(y_true < y_lower, y_lower - y_true, y_true - y_upper)
    )
    return np.mean(score)

# Calculate Winkler Score
winkler = winkler_score(actual_y, pred_lower_exp, pred_upper_exp, alpha=0.1)
coverage = ((actual_y >= pred_lower_exp) & (actual_y <= pred_upper_exp)).mean()
avg_width = np.mean(pred_upper_exp - pred_lower_exp)

print(f"[Winkler] Coverage: {coverage:.2%}")
print(f"[Winkler] Interval Width: {avg_width:.2f}")
print(f"[Winkler] Interval Score: {winkler:.2f}")


# Visualization
plt.figure(figsize=(14, 6))
plt.plot(actual_y.values[:100], label="Actual", marker="o")
plt.plot(median_pred[:100], label="Median Prediction", marker="x")
plt.fill_between(range(100), pred_lower_exp[:100], pred_upper_exp[:100], alpha=0.3, label="Prediction Interval")
plt.legend()
plt.title("Prediction Intervals vs Actuals (First 100)")
plt.xlabel("Sample Index")
plt.ylabel("Sale Price")
plt.show()


# predictions on the test set

pi_test_lower = np.expm1(model_lower.predict(X_test_transformed))  
pi_test_median = np.expm1(model_median.predict(X_test_transformed))  
pi_test_upper = np.expm1(model_upper.predict(X_test_transformed))  

# Export predictions
submission = pd.DataFrame({
    'id': test_fe['id'],  
    'pi_lower': pi_test_lower,
    'pi_upper': pi_test_upper
})

submission.to_csv("submission.csv", index=False, float_format="%.2f")
submission = pd.read_csv("submission.csv")
submission.head(5)

