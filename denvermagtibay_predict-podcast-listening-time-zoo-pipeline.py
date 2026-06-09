#!pip install lightgbm --upgrade
#!pip install xgboost --upgrade


# ğŸ“š Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ğŸ“¦ Load Data

# Update path if necessary
data_path = '/kaggle/input/tabular-playground-series-apr-2025'

train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

train_df.head()


# ğŸ”� Quick EDA (optional quick insights)

# Target Distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df['Listening_Time_minutes'], bins=40, kde=True)
plt.title('Distribution of Listening Time')
plt.show()

# Missing values
print("\nMissing values:")
print(train_df.isnull().sum())


# ğŸ§¹ Clean Missing Values
def clean_missing_values(df):
    df = df.copy()
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())
    return df

train_df = clean_missing_values(train_df)
test_df = clean_missing_values(test_df)


# ğŸ�— Feature Engineering
def preprocess(df, is_train=True):
    df = df.copy()

    if 'Publication_Hour' not in df.columns:
        time_map = {'Morning': 9, 'Afternoon': 14, 'Evening': 18, 'Night': 22}
        df['Publication_Hour'] = df['Publication_Time'].map(time_map)

    if 'Publication_Time' in df.columns:
        df = df.drop(columns=['Publication_Time'])

    if df['Genre'].dtype == object:
        df['Genre'] = df['Genre'].astype('category').cat.codes
    if df['Publication_Day'].dtype == object:
        df['Publication_Day'] = df['Publication_Day'].astype('category').cat.codes

    df = df.drop(columns=['id', 'Podcast_Name', 'Episode_Title'], errors='ignore')

    if is_train:
        X = df.drop(columns=['Listening_Time_minutes'])
        y = df['Listening_Time_minutes']
        return X, y
    else:
        return df


def add_features(df, is_train=True, genre_target_map=None):
    df = df.copy()

    # ğŸ›  Create 'Publication_Hour' if missing
    if 'Publication_Hour' not in df.columns:
        time_map = {'Morning': 9, 'Afternoon': 14, 'Evening': 18, 'Night': 22}
        df['Publication_Hour'] = df['Publication_Time'].map(time_map)

    # ğŸ›  Safe mapping for 'Episode_Sentiment'
    if 'Episode_Sentiment' in df.columns:
        if df['Episode_Sentiment'].dtype == object:
            sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)
        df = pd.get_dummies(df, columns=['Episode_Sentiment'], drop_first=True)

    # Feature engineering
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'] = df['Ad_Density'].replace([np.inf, -np.inf], 0).fillna(0)

    df['Is_Prime_Time'] = df['Publication_Hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)

    # Fix publication day if needed
    if df['Publication_Day'].dtype == object:
        df['Publication_Day'] = df['Publication_Day'].astype('category').cat.codes

    df['Is_Weekend'] = df['Publication_Day'].isin([5, 6]).astype(int)

    df['Host_Guest_Popularity'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Host_to_Guest'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)

    df['log_Episode_Length'] = np.log1p(df['Episode_Length_minutes'])

    df['Has_Guest'] = (df['Guest_Popularity_percentage'] > 0).astype(int)
    df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype(int)

    df['Host_Popularity_norm'] = df['Host_Popularity_percentage'] / 100
    df['Guest_Popularity_norm'] = df['Guest_Popularity_percentage'] / 100

    def length_category(mins):
        if mins < 15: return 0
        elif mins < 45: return 1
        else: return 2
    df['Episode_Length_Category'] = df['Episode_Length_minutes'].apply(length_category)

    df['Hour_sin'] = np.sin(2 * np.pi * df['Publication_Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Publication_Hour'] / 24)

    def time_bin(hour):
        if hour < 12: return 'Morning'
        elif hour < 17: return 'Afternoon'
        elif hour < 21: return 'Evening'
        else: return 'Night'

    df['Time_Slot'] = df['Publication_Hour'].apply(time_bin)
    df = pd.get_dummies(df, columns=['Time_Slot'], drop_first=True)

    # ğŸ›  Safe target encoding
    if 'Genre' in df.columns:
        if is_train:
            genre_target_map = df.groupby('Genre')['Listening_Time_minutes'].mean()
            df['Genre_Target'] = df['Genre'].map(genre_target_map)
            return df, genre_target_map
        else:
            df['Genre_Target'] = df['Genre'].map(genre_target_map).fillna(0)
            return df


# ğŸ›  Process Train and Test
# Clean missing values
train_df = clean_missing_values(train_df)
test_df = clean_missing_values(test_df)

# Feature engineering
train_fe, genre_target_map = add_features(train_df, is_train=True)
test_fe = add_features(test_df, is_train=False, genre_target_map=genre_target_map)

# Preprocessing
X_train, y_train = preprocess(train_fe, is_train=True)
X_test = preprocess(test_fe, is_train=False)


# ğŸ”¥ Model Zoo (Train, Evaluate, Pick Best)

# Train-Validation Split
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model Containers
models = {}
rmse_scores = {}

# Helper function to train and evaluate models
def evaluate(model, name):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    print(f"âœ… {name} RMSE: {rmse:.4f}")
    models[name] = model
    rmse_scores[name] = rmse



# âš¡ Try Different Models
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

# ğŸš€ Improved LightGBM setup with cleaner output
lgb_model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1  # ğŸ§¹ Suppress training warnings for cleaner logs
)

evaluate(lgb_model, "LightGBM")

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, tree_method="hist", n_jobs=-1
)
evaluate(xgb_model, "XGBoost")

# RandomForest
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=15,
    random_state=42, n_jobs=-1
)
evaluate(rf_model, "RandomForest")

# ExtraTrees
et_model = ExtraTreesRegressor(
    n_estimators=300, max_depth=15,
    random_state=42, n_jobs=-1
)
evaluate(et_model, "ExtraTrees")

# Ridge
ridge_model = Ridge(alpha=1.0)
evaluate(ridge_model, "Ridge")



#ğŸ�† Pick Best Model

# Print all RMSEs
print("\nğŸ“Š All RMSE Scores:")
for name, score in rmse_scores.items():
    print(f"{name}: {score:.5f}")

# Pick best model
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]

print(f"\nğŸ�† Best model: {best_model_name} with RMSE {rmse_scores[best_model_name]:.5f}")


# ğŸ“¤ Predict and Submit

# Final model predictions on test set
X_test = X_test[X_train.columns]  # Ensure same columns alignment
test_preds = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': test_preds
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()

