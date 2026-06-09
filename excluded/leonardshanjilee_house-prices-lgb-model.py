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
    """
    Generates a concise summary of DataFrame columns.
    """
    # Use list comprehension to iterate over each column
    summary = [
        [col, df[col].dtype, df[col].count(), df[col].nunique(), df[col].isnull().sum(), df.duplicated().sum()]
        for col in df.columns
    ]

    # Create a DataFrame from the list of lists
    df_check = pd.DataFrame(summary, columns=["column", "dtype", "instances", "unique", "sum_null", "duplicates"])

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
num_cols = 6  
num_rows = (num_features + num_cols - 1) // num_cols  

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 4*num_rows)) 

# Flatten axes array for easy iteration
axes = axes.flatten()

# Plot histograms for each numerical feature
for i, feature in enumerate(numerical_features):
    sns.histplot(train[feature], ax=axes[i], kde=False)
    axes[i].set_title(feature)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Histograms of Numerical Features', y=1.02)  
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


from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# Features
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

    # Size ratios
    df["lot_sqft_ratio"] = df["sqft"] / (df["sqft_lot"] + 1)
    df["garage_total"] = df["garb_sqft"] + df["gara_sqft"]
    df["bath_total"] = df["bath_full"] + df["bath_3qtr"] + 0.5 * df["bath_half"]
    df["sqft_per_bed"] = df["sqft"] / (df["beds"] + 1)
    df["sqft_per_bath"] = df["sqft"] / (df["bath_total"] + 1)

    # Binary features 
    df['has_waterfront'] = (df['wfnt'] > 0).astype(int)
    df['is_golf'] = df['golf']
    df['is_greenbelt'] = df['greenbelt']
    df['has_traffic_noise'] = df['noise_traffic']
    df['has_rainier_view'] = df['view_rainier']
    df['has_olympics_view'] = df['view_olympics']
    df['has_cascades_view'] = df['view_cascades']

    # Polynomial features 
    df['sqft_squared'] = df['sqft']**2

    return df

# Impute numeric columns 
num_cols = train.select_dtypes(include=["float64", "int64"]).columns.drop("sale_price", errors="ignore")
imputer = SimpleImputer(strategy="median")
train[num_cols] = imputer.fit_transform(train[num_cols])
test[num_cols] = imputer.transform(test[num_cols])

# Apply feature engineering
data = pd.concat([train.drop(columns=["sale_price"], errors="ignore"), test], 
                 keys=["train", "test"]).reset_index(level=0).rename(columns={"level_0": "source"})
data_fe = feature_engineering(data)

# Resplit
train_fe = data_fe[data_fe["source"] == "train"].copy()
test_fe = data_fe[data_fe["source"] == "test"].copy()
train_fe["sale_price"] = train["sale_price"].values
test_fe["id"] = test_fe["id"].astype(int)

# Preprocessing
cat_cols = train_fe.select_dtypes("object").columns.drop("sale_date", errors="ignore")
for col in cat_cols:
    train_fe[col] = train_fe[col].astype("category")
    test_fe[col] = test_fe[col].astype("category")

# Encode categoricals
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_fe[cat_cols] = encoder.fit_transform(train_fe[cat_cols])
test_fe[cat_cols] = encoder.transform(test_fe[cat_cols])


import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Train Quantile LightGBM Models
features = [col for col in train_fe.columns if col not in ["id", "sale_price", "sale_date", "source"]]
X = train_fe[features]
y = train_fe["sale_price"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def train_quantile_model(X_train, y_train, alpha):
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": 42
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    return lgb.train(params, dtrain, num_boost_round=1000)

model_lower = train_quantile_model(X_train, y_train, alpha=0.05)
model_median = train_quantile_model(X_train, y_train, alpha=0.5)
model_upper = train_quantile_model(X_train, y_train, alpha=0.95)

# Prediction Intervals
pi_lower = model_lower.predict(X_val)
pi_median = model_median.predict(X_val)
pi_upper = model_upper.predict(X_val)

# Evaluation
interval_width = pi_upper - pi_lower
coverage = np.mean((y_val >= pi_lower) & (y_val <= pi_upper))

print(f"Interval Coverage: {coverage:.2%}")
print(f"Average Interval Width: {np.mean(interval_width):.2f}")

# Visualization
plt.figure(figsize=(14, 6))
plt.plot(y_val.values[:100], label="Actual", marker="o")
plt.plot(pi_median[:100], label="Predicted Median", marker="x")
plt.fill_between(range(100), pi_lower[:100], pi_upper[:100], alpha=0.3, label="Prediction Interval")
plt.legend()
plt.title("Prediction Intervals vs Actuals (first 100)" )
plt.show()


# Feature Importance Visualization
feature_importances = model_median.feature_importance(importance_type='gain')
feature_names = model_median.feature_name()
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
top_features = importance_df.sort_values(by='importance', ascending=False).head(10)


plt.figure(figsize=(10, 6)) 
sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
plt.title("Top 10 Feature Importances - Median Model (Gain)")
plt.xlabel("Feature Importance (Gain)")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# Predict on Test Data
test_X = test_fe[features]
test_fe["pi_lower"] = model_lower.predict(test_X)
test_fe["pi_median"] = model_median.predict(test_X)
test_fe["pi_upper"] = model_upper.predict(test_X)

# Export Predictions
submission = test_fe[["id", "pi_lower", "pi_upper"]]
submission.to_csv("submission.csv", index=False, float_format="%.2f")
submission = pd.read_csv("submission.csv")
submission.head(5)

