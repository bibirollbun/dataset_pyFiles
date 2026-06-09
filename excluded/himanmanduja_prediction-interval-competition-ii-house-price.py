%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_pinball_loss
import lightgbm as lgbm
import joblib


train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
print(f"Original training data shape: {train_df.shape}")
print(f"Original test data shape: {test_df.shape}")


train_na_percentages = train_df.isnull().sum() / len(train_df) * 100
test_na_percentages = test_df.isnull().sum() / len(test_df) * 100
train_cols_to_drop_auto_na = train_na_percentages[train_na_percentages > 50].index.tolist()
test_cols_to_drop_auto_na = test_na_percentages[test_na_percentages > 50].index.tolist()
combined_auto_na_drops = list(set(train_cols_to_drop_auto_na + test_cols_to_drop_auto_na))
explicit_drops = ['subdivision', 'sale_warning', 'sale_nbr']
columns_to_drop_list = list(set(combined_auto_na_drops + explicit_drops))
# Ensure explicit drops are in the list
for col in explicit_drops:
    if col not in columns_to_drop_list:
        columns_to_drop_list.append(col)
print(f"Final consolidated list of columns to be dropped: {sorted(columns_to_drop_list)}")

# Calculate median_year_built_global for imputation
if 'year_built' in train_df.columns:
    median_year_built_global = train_df['year_built'][train_df['year_built'] > 0].median()
    if pd.isna(median_year_built_global):
        median_year_built_global = train_df['year_built'][train_df['year_built'] > 0].mean()
        if pd.isna(median_year_built_global):
             median_year_built_global = 1980 # Further fallback
    print(f"Global median year built (for replacing 0s in 'year_built'): {median_year_built_global}")
else:
    median_year_built_global = 1980
    print(f"'year_built' not in train_df. Using default for median_year_built_global: {median_year_built_global}")


def extract_date_features(df, median_year_built):
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['sale_day'] = df['sale_date'].dt.day
    if 'year_built' in df.columns:
        df['year_built'] = df['year_built'].replace(0, median_year_built)
        df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce').fillna(median_year_built)
        df['age_of_property'] = df['sale_year'] - df['year_built']
    else:
        df['age_of_property'] = 0
    return df.drop('sale_date', axis=1)

train_df_fe = extract_date_features(train_df.copy(), median_year_built_global)
test_df_fe = extract_date_features(test_df.copy(), median_year_built_global)

# Assign to 'train' and 'test' for consistency with the rest of the notebook if needed
train = train_df_fe 
test = test_df_fe

print(f"Shapes after feature engineering: train: {train.shape}, test: {test.shape}")


plt.figure(figsize=(10, 6))
sns.histplot(train['sale_price'], kde=True, bins=50)
plt.title('Distribution of Sale Price (Target Variable)')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()


numeric_cols_for_corr = train.select_dtypes(include=np.number).columns.tolist()
corr_matrix = train[numeric_cols_for_corr].corr()
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Matrix of Numeric Features in Training Data')
plt.show()


if 'sale_price' in corr_matrix:
    corr_with_target = corr_matrix['sale_price'].sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=corr_with_target.values[1:11], y=corr_with_target.index[1:11])
    plt.title('Top 10 Features Most Positively Correlated with Sale Price')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature Name')
    plt.show()
else:
    print("Skipping 'Top Features Correlated with Sale Price' plot as 'sale_price' is not in the correlation matrix.")


# Drop columns from the 'train' DataFrame (which is train_df_fe)
cols_actually_in_train_to_drop = [col for col in columns_to_drop_list if col in train.columns]
train_processed = train.drop(columns=cols_actually_in_train_to_drop)

X = train_processed.drop(['id', 'sale_price'], axis=1, errors='ignore')
y = train['sale_price'] # Target from the original 'train' DataFrame (before dropping more columns for X)

print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of target vector y: {y.shape}")

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

if 'age_of_property' in X.columns and 'age_of_property' not in numeric_features:
    numeric_features.append('age_of_property')
    if 'age_of_property' in categorical_features:
        categorical_features.remove('age_of_property')

print(f"Identified {len(numeric_features)} numeric features: {numeric_features}")
print(f"Identified {len(categorical_features)} categorical features: {categorical_features}")

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')


# Original train-test split for validation (now commented out):
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f"Shape of X_train: {X_train.shape}")
# print(f"Shape of y_train: {y_train.shape}")
# print(f"Shape of X_val: {X_val.shape}")
# print(f"Shape of y_val: {y_val.shape}")

# Using full dataset for training:
X_train = X
y_train = y
print(f"Using full dataset for training. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# X_val, y_val are not created in this workflow as we are training on the full data.


model_lower = lgbm.LGBMRegressor(objective='quantile', alpha=0.05, random_state=42, n_estimators=2000, learning_rate=0.02, num_leaves=31, n_jobs=-1)
model_upper = lgbm.LGBMRegressor(objective='quantile', alpha=0.95, random_state=42, n_estimators=2000, learning_rate=0.02, num_leaves=31, n_jobs=-1)

pipeline_lower = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_lower)])
pipeline_upper = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_upper)])

# Preprocessor fitting (already done on full X_train if notebook cells are run sequentially, 
# but explicit fit here ensures it's based on the potentially reassigned X_train)
print(f"Fitting preprocessor on X_train (shape: {X_train.shape})...")
preprocessor.fit(X_train, y_train)
print("Preprocessor fitting complete.")

# Early stopping data preparation (now commented out):
# X_train_fit, X_eval_early_stopping, y_train_fit, y_eval_early_stopping = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# X_eval_early_stopping_transformed = preprocessor.transform(X_eval_early_stopping)
# fit_params_lower = {'regressor__eval_set': [(X_eval_early_stopping_transformed, y_eval_early_stopping)], 'regressor__callbacks': [lgbm.early_stopping(100, verbose=False)]}
# fit_params_upper = {'regressor__eval_set': [(X_eval_early_stopping_transformed, y_eval_early_stopping)], 'regressor__callbacks': [lgbm.early_stopping(100, verbose=False)]}

print(f"Training lower pipeline on full X_train (shape: {X_train.shape})...")
pipeline_lower.fit(X_train, y_train) # Train on full X_train, no fit_params for early stopping
print("Lower pipeline training complete.")

print(f"Training upper pipeline on full X_train (shape: {X_train.shape})...")
pipeline_upper.fit(X_train, y_train) # Train on full X_train, no fit_params for early stopping
print("Upper pipeline training complete.")


# All lines in this cell are commented out as X_val and y_val are not available.
# lower_preds = pipeline_lower.predict(X_val)
# upper_preds = pipeline_upper.predict(X_val)
# coverage = np.mean((y_val >= lower_preds) & (y_val <= upper_preds))
# interval_width = np.mean(upper_preds - lower_preds)
# print(f"Validation Set - Prediction Interval Coverage: {coverage:.2%}")
# print(f"Validation Set - Average Prediction Interval Width: ${interval_width:,.0f}")
# pinball_loss_lower = mean_pinball_loss(y_val, lower_preds, alpha=0.05)
# pinball_loss_upper = mean_pinball_loss(y_val, upper_preds, alpha=0.95)
# print(f"Validation Set - Pinball Loss (Lower Quantile, alpha=0.05): {pinball_loss_lower:.4f}")
# print(f"Validation Set - Pinball Loss (Upper Quantile, alpha=0.95): {pinball_loss_upper:.4f}")
# plt.figure(figsize=(12, 7))
# if len(y_val) >= 100:
#     sample_indices = np.random.choice(len(y_val), 100, replace=False)
# else:
#     sample_indices = np.arange(len(y_val))
# plt.scatter(range(len(sample_indices)), y_val.iloc[sample_indices], color='blue', label='Actual Price', alpha=0.7)
# plt.errorbar(
#     range(len(sample_indices)),
#     (lower_preds[sample_indices] + upper_preds[sample_indices]) / 2,
#     yerr=(upper_preds[sample_indices] - lower_preds[sample_indices]) / 2,
#     fmt='o', color='red', ecolor='lightcoral', elinewidth=2, capsize=4, label='Prediction Interval (5th-95th percentile)'
# )
# plt.title('Actual Prices vs. Prediction Intervals (Random Sample from Validation Set)')
# plt.xlabel('Sample Index (Randomly Selected)')
# plt.ylabel('Sale Price')
# plt.legend(loc='best')
# plt.show()
print("Model evaluation on a separate validation set is skipped in this workflow.")


joblib.dump(pipeline_lower, 'model_lower.joblib')
joblib.dump(pipeline_upper, 'model_upper.joblib')
print("Trained lower and upper quantile pipelines saved to 'model_lower.joblib' and 'model_upper.joblib'.")


# Drop columns from the 'test' DataFrame (which is test_df_fe)
cols_actually_in_test_to_drop = [col for col in columns_to_drop_list if col in test.columns]
test_processed = test.drop(columns=cols_actually_in_test_to_drop)

X_test_raw = test_processed.drop(['id'], axis=1, errors='ignore')

# Align X_test columns with X_train (which is the full X)
X_cols = X_train.columns # Use X_train.columns as it's the reference for fitted preprocessor
aligned_cols = [col for col in X_cols if col in X_test_raw.columns]
X_test = X_test_raw[aligned_cols]

missing_cols_in_X_test = [col for col in X_cols if col not in X_test.columns]
if missing_cols_in_X_test:
    print(f"Adding missing columns to X_test and filling with NaN: {missing_cols_in_X_test}")
    for col in missing_cols_in_X_test:
        X_test[col] = np.nan
    X_test = X_test[X_cols] # Ensure final order matches X_cols

print(f"Final X_test shape after column alignment: {X_test.shape}")
print(f"Final X_test columns: {X_test.columns.tolist()}")


# The pipelines (pipeline_lower, pipeline_upper) are already trained and include the fitted preprocessor.
print(f"Generating predictions for test set of shape {X_test.shape}...")
lower_test = pipeline_lower.predict(X_test)
upper_test = pipeline_upper.predict(X_test)
print(f"Generated {len(lower_test)} lower bound predictions and {len(upper_test)} upper bound predictions for the test set.")


submission = pd.DataFrame({'id': test['id'], 'pi_lower': lower_test, 'pi_upper': upper_test})
print(f"Initial submission DataFrame created with shape: {submission.shape}")

submission['pi_lower'] = submission['pi_lower'].clip(lower=0)
print("Lower predictions clipped at 0.")

original_upper = submission['pi_upper'].copy()
submission['pi_upper'] = np.maximum(submission['pi_upper'], submission['pi_lower'] * 1.01)
submission['pi_upper'] = np.maximum(submission['pi_upper'], submission['pi_lower'] + 1000)
changes_in_upper = (submission['pi_upper'] != original_upper).sum()
print(f"Upper predictions adjusted to be greater than lower predictions. {changes_in_upper} rows affected by this rule.")

print(f"Final shape of submission DataFrame before saving: {submission.shape}")
print(f"Columns in submission DataFrame: {submission.columns.tolist()}")
submission.to_csv('submission.csv', index=False)
print("Submission file saved to: submission.csv")
print(submission.head())




