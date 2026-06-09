# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train_df.head(20)


train_df.info()


missing_values = train_df.isnull().sum().sort_values(ascending = False)
missing_values[missing_values > 0]


# Impute missing numerical values with median
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), 
                                          inplace=True)
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), 
                                               inplace=True)
train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median(), 
                                 inplace=True)

# Confirm that no missing values remain
train_df.isnull().sum().sum()


import numpy as np

# Extract numeric part from Episode_Title (e.g., "Episode 98" -> 98)
train_df['Episode_Number'] = train_df['Episode_Title'].str.extract(r'(\d+)', expand=False).astype(int)

# Drop Episode_Title as we've extracted the useful part
train_df.drop(columns=['Episode_Title'], inplace=True)

# Encode categorical columns using one-hot encoding
categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
train_df_encoded = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)

# Drop Podcast_Name (assumed to be high cardinality and less useful without extra NLP)
train_df_encoded.drop(columns=['Podcast_Name'], inplace=True)

# Display the updated DataFrame shape and columns
train_df_encoded.shape, train_df_encoded.columns.tolist()


# Split into features and target
X = train_df_encoded.drop(columns=['Listening_Time_minutes', 'id'])
y = train_df_encoded['Listening_Time_minutes']

# Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Validation RMSE: {rmse:.4f}")


from sklearn.ensemble import RandomForestRegressor

# Retrain using a single thread
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
model.fit(X_train, y_train)

# Prepare the test set for prediction (similar preprocessing steps)
test_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
test_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].median(), inplace=True)
test_df['Episode_Number'] = test_df['Episode_Title'].str.extract(r'(\d+)', expand=False).astype(int)
test_df.drop(columns=['Episode_Title'], inplace=True)

# One-hot encode categorical columns and align with training data
test_df_encoded = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)
test_df_encoded.drop(columns=['Podcast_Name'], inplace=True)
test_df_encoded = test_df_encoded.reindex(columns=X.columns, fill_value=0)

# Make predictions
test_preds = model.predict(test_df_encoded)


# Create submission DataFrame
submission = pd.DataFrame({
    "id": test_df['id'],
    "Listening_Time_minutes": test_preds
})

submission


# Save the submission file
submission_path = '/kaggle/working/submission_1.csv'
submission.to_csv(submission_path, index=False)
submission.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Feature Engineering
def feature_engineering(df):
    # Extract episode number from title
    df['Episode_Number'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    
    # Create interaction features
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage'].fillna(0)
    df['Ads_Length_Interaction'] = df['Number_of_Ads'] * df['Episode_Length_minutes']
    
    # Time-based features
    df['Publication_Time_Encoded'] = df['Publication_Time'].map({
        'Morning': 1,
        'Afternoon': 2,
        'Evening': 3,
        'Night': 4
    })
    
    # Day of week encoding
    df['Publication_Day_Encoded'] = df['Publication_Day'].map({
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    })
    
    # Sentiment encoding
    df['Sentiment_Encoded'] = df['Episode_Sentiment'].map({
        'Negative': -1,
        'Neutral': 0,
        'Positive': 1
    })
    
    # Podcast name length (proxy for complexity)
    df['Podcast_Name_Length'] = df['Podcast_Name'].str.len()
    
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Handle missing values
median_values = {
    'Episode_Length_minutes': train_df['Episode_Length_minutes'].median(),
    'Guest_Popularity_percentage': train_df['Guest_Popularity_percentage'].median(),
    'Number_of_Ads': train_df['Number_of_Ads'].median()
}

for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
    train_df[col].fillna(median_values[col], inplace=True)
    test_df[col].fillna(median_values[col], inplace=True)

# Define features and target
features = [
    'Episode_Length_minutes', 'Host_Popularity_percentage', 
    'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Number',
    'Host_Guest_Interaction', 'Ads_Length_Interaction',
    'Publication_Time_Encoded', 'Publication_Day_Encoded',
    'Sentiment_Encoded', 'Podcast_Name_Length'
]

categorical_features = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
target = 'Listening_Time_minutes'

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data
X = train_df[features + categorical_features]
y = train_df[target]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Tuned Random Forest
rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_valid)
rmse_rf = np.sqrt(mean_squared_error(y_valid, y_pred_rf))
print(f"Random Forest RMSE: {rmse_rf:.4f}")

# Model 2: XGBoost
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_valid)
rmse_xgb = np.sqrt(mean_squared_error(y_valid, y_pred_xgb))
print(f"XGBoost RMSE: {rmse_xgb:.4f}")

# Model 3: LightGBM
lgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_valid)
rmse_lgb = np.sqrt(mean_squared_error(y_valid, y_pred_lgb))
print(f"LightGBM RMSE: {rmse_lgb:.4f}")

# Ensemble predictions (simple average)
ensemble_pred = (y_pred_rf + y_pred_xgb + y_pred_lgb) / 3
rmse_ensemble = np.sqrt(mean_squared_error(y_valid, ensemble_pred))
print(f"Ensemble RMSE: {rmse_ensemble:.4f}")

# Feature Importance Analysis (using XGBoost)
feature_importances = xgb_model.named_steps['regressor'].feature_importances_
cat_encoder = xgb_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_features = features + list(cat_features)

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Final Model Selection (use the best performing one)
if rmse_ensemble < min(rmse_rf, rmse_xgb, rmse_lgb):
    print("\nUsing ensemble model for final predictions")
    final_model = None  # We'll manually ensemble for predictions
else:
    best_model = min(
        (rmse_rf, rf, "Random Forest"),
        (rmse_xgb, xgb_model, "XGBoost"),
        (rmse_lgb, lgb_model, "LightGBM"),
        key=lambda x: x[0]
    )
    print(f"\nUsing {best_model[2]} model for final predictions")
    final_model = best_model[1]

# Prepare test data
X_test = test_df[features + categorical_features]

# Make predictions
if final_model is not None:
    test_preds = final_model.predict(X_test)
else:
    # Ensemble predictions
    test_preds_rf = rf.predict(X_test)
    test_preds_xgb = xgb_model.predict(X_test)
    test_preds_lgb = lgb_model.predict(X_test)
    test_preds = (test_preds_rf + test_preds_xgb + test_preds_lgb) / 3

# Create submission file
submission = pd.DataFrame({
    "id": test_df["id"],
    "Listening_Time_minutes": test_preds
})

submission.to_csv('/kaggle/working/submission_improved.csv', index=False)
print("\nSubmission file created successfully!")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


# Feature Engineering
def create_features(df):
    # Extract episode number
    df['Episode_Number'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    
    # Popularity interactions
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage'].fillna(0)
    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'].fillna(0)
    
    # Length and ads interactions
    df['Ads_Per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)  # +1 to avoid division by zero
    df['Ads_Length_Interaction'] = df['Number_of_Ads'] * df['Episode_Length_minutes']
    
    # Time features
    time_mapping = {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    df['Publication_Time_Encoded'] = df['Publication_Time'].map(time_mapping)
    
    # Day of week encoding
    day_mapping = {
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 
        'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
    }
    df['Publication_Day_Encoded'] = df['Publication_Day'].map(day_mapping)
    
    # Sentiment encoding
    sentiment_mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df['Sentiment_Encoded'] = df['Episode_Sentiment'].map(sentiment_mapping)
    
    # Podcast name features
    df['Podcast_Name_Length'] = df['Podcast_Name'].str.len()
    df['Podcast_Name_Words'] = df['Podcast_Name'].str.split().str.len()
    
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)


# Handle missing values explicitly
def handle_missing(df):
    # Numerical features - impute with median
    num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Number']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Interaction features - recalculate after imputation
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Ads_Per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['Ads_Length_Interaction'] = df['Number_of_Ads'] * df['Episode_Length_minutes']
    
    return df

train_df = handle_missing(train_df)
test_df = handle_missing(test_df)


# Verify no missing values remain
print("Missing values in train:", train_df.isnull().sum().sum())
print("Missing values in test:", test_df.isnull().sum().sum())

# Categorical encoding
cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# Feature selection with proper NaN handling
features = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Number',
    'Host_Guest_Interaction', 'Total_Popularity', 'Ads_Per_Minute',
    'Ads_Length_Interaction', 'Publication_Time_Encoded',
    'Publication_Day_Encoded', 'Sentiment_Encoded', 'Podcast_Name_Length',
    'Podcast_Name_Words', 'Genre', 'Publication_Day', 'Publication_Time',
    'Episode_Sentiment'
]

X = train_df[features]
y = train_df['Listening_Time_minutes']


# Create a pipeline that imputes before feature selection
from sklearn.pipeline import Pipeline

# Select top 15 features
selector = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle any remaining NaN values
    ('selector', SelectKBest(score_func=f_regression, k=15))
])

X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.named_steps['selector'].get_support()]
print("Selected Features:", list(selected_features))

# Update features to selected ones
X = train_df[selected_features]
X_test = test_df[selected_features]


# LightGBM parameters (tuned)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Cross-validation setup
folds = KFold(n_splits=7, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance = pd.DataFrame()

for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # LightGBM can handle NaN values internally, but we've already cleaned them
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(100)
        ]
    )
    
    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test) / folds.n_splits
    
    # Feature importance
    fold_importance = pd.DataFrame()
    fold_importance["Feature"] = selected_features
    fold_importance["Importance"] = model.feature_importance()
    fold_importance["Fold"] = fold + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
    # Fold RMSE
    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_idx]))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# Overall OOF RMSE
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall OOF RMSE: {oof_rmse:.4f}")


# Feature importance visualization
plt.figure(figsize=(12, 8))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance.groupby('Feature')['Importance']
                          .mean()
                          .reset_index()
                          .sort_values(by="Importance", ascending=False),
    palette="viridis"
)
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.show()

# Create submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "Listening_Time_minutes": test_preds
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission file created successfully!")







