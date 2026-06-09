import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# Remove outliers in target
train = train[train['Listening_Time_minutes'] < 120]


# Fill missing values
train.fillna(train.median(numeric_only=True), inplace=True)
test.fillna(test.median(numeric_only=True), inplace=True)


# Feature Engineering (a few new features and interaction)
for df in [train, test]:
    df['Popularity_Diff'] = abs(df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage'])
    df['Combined_Popularity'] = df['Host_Popularity_percentage'] * 0.7 + df['Guest_Popularity_percentage'] * 0.3
    df['Ad_Density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 0.1)
    df['Length_Category'] = pd.cut(df['Episode_Length_minutes'], bins=[0, 30, 60, 90, 120, 999],
                                   labels=['short', 'medium', 'long', 'vlong', 'extreme']).astype('category').cat.codes
    df['Prime_Time'] = df['Publication_Time'].isin(['Evening', 'Night']).astype(int)
    df['Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Sentiment_Score'] = df['Episode_Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    df['Ad_Sentiment_Impact'] = df['Number_of_Ads'] * df['Sentiment_Score']
    df['Episode_Seq'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    df['Host_Guest_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 0.1)
    df['Host_Length_Interaction'] = df['Host_Popularity_percentage'] * df['Episode_Length_minutes']
    df['Guest_Ad_Impact'] = df['Guest_Popularity_percentage'] * df['Number_of_Ads']
    # New features
    df['Popularity_Product'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Length_Sentiment_Interaction'] = df['Episode_Length_minutes'] * df['Sentiment_Score']



# Label Encoding
cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_cols:
    train[col] = train[col].astype('category').cat.codes
    test[col] = test[col].astype('category').cat.codes


# Features and target
X = train.drop(columns=['id', 'Listening_Time_minutes', 'Episode_Title'])
y = train['Listening_Time_minutes']
X_test = test.drop(columns=['id', 'Episode_Title'])



# Train model directly with slightly adjusted parameters
final_model = CatBoostRegressor(
    iterations=2100,      # Slightly increased
    learning_rate=0.019,  # Small adjustment
    depth=7,             # Slightly decreased
    l2_leaf_reg=3.5,      # Small adjustment
    bagging_temperature=0.45, # Small adjustment
    random_strength=0.25, # Small adjustment
    loss_function="RMSE",
    eval_metric="RMSE",
    task_type="CPU",
    verbose=100,
    random_seed=42)


# Train model
final_model.fit(X, y)


# Predict test set
final_preds = final_model.predict(X_test)


# Save submission
submission = pd.DataFrame({
    "id": test["id"],
    "Listening_Time_minutes": np.clip(final_preds, 0, None)})
submission.to_csv("/kaggle/working/catboost_submission.csv", index=False)
print("Submission saved.")

