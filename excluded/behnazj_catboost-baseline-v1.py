import pandas as pd
import numpy as np
from textblob import TextBlob
from catboost import CatBoostRegressor, Pool

# Load the data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Combine train and test for feature engineering
combined = pd.concat([train_df, test_df], axis=0)
combined.drop("id", axis=1, inplace=True)

# 1. Feature: Publication_Day
combined['Publication_Day'] = combined['Publication_Day'].astype(str)
combined['Publication_Day'] = combined['Publication_Day'].apply(
    lambda x: x.split(',')[1].strip() if ',' in x else np.nan
)
combined['Publication_Day'] = pd.to_datetime(combined['Publication_Day'], errors='coerce')
combined['Publication_Day'] = combined['Publication_Day'].dt.dayofweek

# 2. Feature: Publication_Time
combined['Publication_Time'] = pd.to_datetime(combined['Publication_Time'], errors='coerce').dt.hour

# 3. Feature: Title Length
combined['Title_Length'] = combined['Episode_Title'].astype(str).apply(len)

# 4. Feature: Sentiment of Title using TextBlob
combined['Title_Sentiment'] = combined['Episode_Title'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

# 5. Handle missing values
categorical_cols = combined.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    combined[col].fillna(combined[col].mode()[0], inplace=True)

numerical_cols = combined.select_dtypes(include=np.number).columns.tolist()
for col in numerical_cols:
    combined[col].fillna(combined[col].median(), inplace=True)

# 6. Feature interaction: Sentiment * Title Length
combined['Episode_Sentiment_Length'] = combined['Title_Sentiment'] * combined['Title_Length']

# 7. Ensure categorical columns are string type
for col in categorical_cols:
    combined[col] = combined[col].astype(str)

# 8. Split back into train and test
X_train = combined.iloc[:len(train_df), :].copy()
X_test = combined.iloc[len(train_df):, :].copy()

# 9. Target variable
y = train_df["Listening_Time_minutes"]
X_train.drop("Listening_Time_minutes", axis=1, inplace=True)

# 10. Train the CatBoost model
train_pool = Pool(X_train, y, cat_features=categorical_cols)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=50
)
model.fit(train_pool)

# 11. Make predictions
test_pool = Pool(X_test, cat_features=categorical_cols)
preds = model.predict(test_pool)

# 12. Prepare submission
sample_submission["Listening_Time_minutes"] = preds
sample_submission.to_csv("submission_with_features.csv", index=False)

