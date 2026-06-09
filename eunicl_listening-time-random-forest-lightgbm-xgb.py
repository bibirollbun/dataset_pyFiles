# Imports and Data Loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
import warnings

# For imputation, scaling, and winsorization
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from scipy.stats.mstats import winsorize

# For modeling and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
nltk.download('vader_lexicon')

# List input files (Kaggle)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
print("Data loaded.")


# Missing Value Imputation

def impute_by_group(df, col, group_col, agg_func=np.median):
    """
    Impute missing values in `col` using the group-based aggregation (median by default)
    from the groups defined by `group_col`.
    """
    group_medians = df.groupby(group_col)[col].transform(agg_func)
    df[col] = df[col].fillna(group_medians)
    return df

# Impute for 'Episode_Length_minutes' and 'Guest_Popularity_percentage' using 'Podcast_Name'
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    train_df = impute_by_group(train_df, col, 'Podcast_Name')
    test_df  = impute_by_group(test_df, col, 'Podcast_Name')

# Use KNN imputation for selected numeric features.
num_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                'Guest_Popularity_percentage', 'Number_of_Ads']
combined = pd.concat([train_df[num_features], test_df[num_features]], axis=0)
knn_imputer = KNNImputer(n_neighbors=5)
combined_imputed = pd.DataFrame(knn_imputer.fit_transform(combined),
                                columns=num_features, index=combined.index)
train_df.loc[combined_imputed.index.intersection(train_df.index), num_features] = combined_imputed.loc[train_df.index, :]
test_df.loc[combined_imputed.index.intersection(test_df.index), num_features] = combined_imputed.loc[test_df.index, :]

print("Missing value imputation completed.")


# Outlier Handling and Scaling

def winsorize_series(s, lower_pct=0.01, upper_pct=0.99):
    """Winsorize a pandas Series by capping the values at the given quantiles."""
    return pd.Series(winsorize(s, limits=(lower_pct, 1 - upper_pct)), index=s.index)

for col in num_features:
    train_df[col] = winsorize_series(train_df[col])
    test_df[col]  = winsorize_series(test_df[col])

scaler = RobustScaler()
train_df[num_features] = scaler.fit_transform(train_df[num_features])
test_df[num_features]  = scaler.transform(test_df[num_features])

print("Outlier handling and scaling completed.")


# Target Transformation and EDA on Target

plt.figure(figsize=(8, 5))
sns.histplot(train_df['Listening_Time_minutes'], bins=50, kde=True)
plt.title("Target Distribution Before Transformation")
plt.xlabel("Listening_Time_minutes")
plt.ylabel("Frequency")
plt.show()

# Apply log1p transformation.
train_df['Log_Listening_Time'] = np.log1p(train_df['Listening_Time_minutes'])

plt.figure(figsize=(8, 5))
sns.histplot(train_df['Log_Listening_Time'], bins=50, kde=True, color='green')
plt.title("Target Distribution After log1p Transformation")
plt.xlabel("Log(Listening_Time_minutes + 1)")
plt.ylabel("Frequency")
plt.show()

print("Target transformation completed.")


# Feature Engineering Function

def feature_engineering(df, is_train=True):
    """
    Process the raw podcast dataset to create engineered features.
    """
    # Remove non-informative attributes.
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    
    # Extract textual features from Episode_Title.
    df['Title_Length'] = df['Episode_Title'].fillna('').apply(len)
    df['Title_WordCount'] = df['Episode_Title'].fillna('').apply(lambda x: len(x.split()))
    
    # Use VADER for sentiment analysis.
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    df['Title_Sentiment_Compound'] = df['Episode_Title'].fillna('').apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Process Episode_Length_minutes and Number_of_Ads.
    df['Episode_Length_minutes'] = pd.to_numeric(df['Episode_Length_minutes'], errors='coerce').replace(0, 0.1)
    df['Number_of_Ads'] = pd.to_numeric(df['Number_of_Ads'], errors='coerce')
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    
    # Process popularity features.
    df['Host_Popularity_percentage'] = pd.to_numeric(df['Host_Popularity_percentage'], errors='coerce')
    df['Guest_Popularity_percentage'] = pd.to_numeric(df['Guest_Popularity_percentage'], errors='coerce')
    df['Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Popularity_Product'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    
    # Process Genre and Episode_Sentiment.
    df['Genre'] = df['Genre'].astype('category')
    sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_mapping)
    df['Episode_Sentiment_cat'] = pd.Categorical(df['Episode_Sentiment'])
    
    # Engineer time features.
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
                   'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['Publication_Day_Num'] = df['Publication_Day'].map(day_mapping)
    df['Day_sin'] = np.sin(2 * np.pi * df['Publication_Day_Num'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Publication_Day_Num'] / 7)
    time_mapping = {'Morning': 540, 'Afternoon': 840, 'Evening': 1140, 'Night': 1380}
    df['Publication_Time_Minutes'] = df['Publication_Time'].map(time_mapping)
    df['Publication_Time_Minutes'] = df['Publication_Time_Minutes'].fillna(df['Publication_Time_Minutes'].median())
    df['Time_sin'] = np.sin(2 * np.pi * df['Publication_Time_Minutes'] / 1440)
    df['Time_cos'] = np.cos(2 * np.pi * df['Publication_Time_Minutes'] / 1440)
    
    # Create combined features.
    alpha = 5
    df['Effective_Episode_Duration'] = (df['Episode_Length_minutes'] - (alpha * df['Ad_Density'])).clip(lower=0)
    
    # Drop redundant columns.
    df = df.drop(columns=['Episode_Title', 'Publication_Day', 'Publication_Time'])
    
    # Adjust cyclic feature data types.
    cyclic_cols = ['Day_sin', 'Day_cos', 'Time_sin', 'Time_cos']
    df[cyclic_cols] = df[cyclic_cols].astype(np.float32)
    
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Episode_Sentiment_cat'] = df['Episode_Sentiment_cat'].astype('category')
    
    # Fill any remaining missing numeric values with medians.
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

print("Feature engineering function defined.")


# Apply Feature Engineering and perform EDA on Processed Data

processed_train = feature_engineering(train_df, is_train=True)
print("Head of Processed Training DataFrame:")
print(processed_train.head())
print("\nSummary Statistics:")
print(processed_train.describe(include='all'))
processed_train.info()

# Plot distributions for numeric features (excluding the target 'Listening_Time_minutes').
numeric_cols = processed_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Listening_Time_minutes' in numeric_cols:
    numeric_cols.remove('Listening_Time_minutes')
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(processed_train[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Correlation Heatmap.
plt.figure(figsize=(12, 10))
correlation_matrix = processed_train.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Categorical Features EDA.
categorical_cols = processed_train.select_dtypes(include=['category']).columns.tolist()
print("Categorical Columns:", categorical_cols)
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=processed_train, x=col)
    plt.title(f"Count Plot for {col}")
    plt.xticks(rotation=45)
    plt.show()

# Relationships with the target.
if 'Listening_Time_minutes' in processed_train.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=processed_train['Effective_Episode_Duration'], y=processed_train['Listening_Time_minutes'], alpha=0.4)
    plt.title("Effective_Episode_Duration vs. Listening_Time_minutes")
    plt.xlabel("Effective_Episode_Duration (minutes)")
    plt.ylabel("Listening Time (minutes)")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=processed_train['Ad_Density'], y=processed_train['Listening_Time_minutes'], alpha=0.4)
    plt.title("Ad Density vs. Listening_Time_minutes")
    plt.xlabel("Ad Density (ads per minute)")
    plt.ylabel("Listening Time (minutes)")
    plt.show()
    
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=processed_train['Title_Sentiment_Compound'], y=processed_train['Listening_Time_minutes'], alpha=0.4)
    plt.title("Title_Sentiment_Compound vs. Listening_Time_minutes")
    plt.xlabel("Title Sentiment (Compound Score)")
    plt.ylabel("Listening Time (minutes)")
    plt.show()

print("EDA on processed data completed.")


# Modeling: Data Preparation and Baseline Models

print("Starting modeling data preparation...")
target_col = 'Listening_Time_minutes'
if target_col not in processed_train.columns:
    raise ValueError(f"Target column '{target_col}' not found in processed_train.")
# IMPORTANT: Remove 'Log_Listening_Time' from predictors so the model doesn't cheat.
X = processed_train.drop(columns=[target_col, 'Log_Listening_Time'], errors='ignore')
y = processed_train[target_col]

# Convert categorical variables into dummy/indicator columns.
X_encoded = pd.get_dummies(X, drop_first=True)
print("Encoded Feature columns:")
print(X_encoded.columns)
print("Data preparation for modeling completed.\n")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Decision Tree.
print("Training Decision Tree baseline...")
dt_model = DecisionTreeRegressor(random_state=42)
dt_scores = cross_val_score(dt_model, X_encoded, y, cv=cv, scoring='neg_root_mean_squared_error')
dt_rmse = -np.mean(dt_scores)
print(f"Decision Tree Baseline RMSE: {dt_rmse:.4f}\n")

# Model 2: Random Forest.
print("Training Random Forest baseline...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf_model, X_encoded, y, cv=cv, scoring='neg_root_mean_squared_error')
rf_rmse = -np.mean(rf_scores)
print(f"Random Forest Baseline RMSE: {rf_rmse:.4f}\n")

# Model 3: LightGBM.
print("Training LightGBM model with early stopping (using GPU)...")
lgb_model = lgb.LGBMRegressor(
    device_type='gpu',
    n_estimators=1000,
    max_depth=13,
    learning_rate=0.03,
    num_leaves=512,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=8,
    reg_lambda=6,
    max_bin=225,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)
# Use callback-based early stopping.
X_train_lgb, X_valid_lgb, y_train_lgb, y_valid_lgb = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
lgb_model.fit(X_train_lgb, y_train_lgb,
              eval_set=[(X_valid_lgb, y_valid_lgb)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])
lgb_scores = cross_val_score(lgb_model, X_encoded, y, cv=cv, scoring='neg_root_mean_squared_error')
lgb_rmse = -np.mean(lgb_scores)
print(f"LightGBM Model Cross-validated RMSE: {lgb_rmse:.4f}\n")

# Model 4: XGBoost.
print("Training XGBoost model with early stopping...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=13,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
X_train_xgb, X_valid_xgb, y_train_xgb, y_valid_xgb = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb,
              eval_set=[(X_valid_xgb, y_valid_xgb)],
              eval_metric='rmse',
              early_stopping_rounds=100,
              verbose=False)
xgb_scores = cross_val_score(xgb_model, X_encoded, y, cv=cv, scoring='neg_root_mean_squared_error')
xgb_rmse = -np.mean(xgb_scores)
print(f"XGBoost Model Cross-validated RMSE: {xgb_rmse:.4f}\n")

# Compare models.
print("Model Comparison:")
print(f"Decision Tree RMSE: {dt_rmse:.4f}")
print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"LightGBM RMSE: {lgb_rmse:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

# Choose the best model (lowest RMSE).
if (rf_rmse < lgb_rmse) and (rf_rmse < dt_rmse) and (rf_rmse < xgb_rmse):
    chosen_model = 'RandomForest'
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("Random Forest selected as the final model.")
elif (lgb_rmse < dt_rmse) and (lgb_rmse < xgb_rmse):
    chosen_model = 'LightGBM'
    final_model = lgb.LGBMRegressor(
        device_type='gpu',
        n_estimators=1000,
        max_depth=13,
        learning_rate=0.03,
        num_leaves=512,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=8,
        reg_lambda=6,
        max_bin=225,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    print("LightGBM selected as the final model.")
elif xgb_rmse < dt_rmse:
    chosen_model = 'XGBoost'
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=13,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    print("XGBoost selected as the final model.")
else:
    chosen_model = 'DecisionTree'
    final_model = DecisionTreeRegressor(random_state=42)
    print("Decision Tree selected as the final model.")

print(f"Chosen final model: {chosen_model}")


# Final Model Training and Test Prediction

print("Training the chosen final model on the full training data...")
final_model.fit(X_encoded, y)
print("Final model training completed.")

# Process test data: Apply feature engineering.
processed_test = feature_engineering(test_df, is_train=False)
X_test_encoded = pd.get_dummies(processed_test, drop_first=True)
# Align test data columns with training features.
X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)
print("Test data prepared.")

# Generate predictions on test data.
predictions = final_model.predict(X_test_encoded)

# If you used a log transformation on the target during training, apply inverse transformation:
# predictions = np.expm1(predictions)

# Create submission dataframe in required format.
submission = pd.DataFrame({'id': X_test_encoded.index, 'Listening_Time_minutes': predictions})
submission = submission.set_index('id')
submission.to_csv('submission.csv')
print("Predictions saved to submission.csv.")

