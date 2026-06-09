import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler


train=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.head()


train.info()


train.isna().sum()


# Fill Episode_Length_minutes with median
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(), inplace=True)

# Create a new column to track whether a guest was present
train['Has_Guest'] = train['Guest_Popularity_percentage'].notnull().astype(int)

# Fill missing guest popularity with 0
train['Guest_Popularity_percentage'].fillna(0, inplace=True)

# Fill Number_of_Ads (only one missing) with median
train['Number_of_Ads'].fillna(train['Number_of_Ads'].median(), inplace=True)



train.nunique()


train.duplicated().value_counts()


train = train[train['Listening_Time_minutes'] <= train['Episode_Length_minutes']]


plt.figure(figsize=(8, 5))
sns.histplot(train['Listening_Time_minutes'], bins=50, kde=True)
plt.title("Distribution of Listening Time")
plt.xlabel("Listening Time (minutes)")
plt.show()


num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=train[col], y=train['Listening_Time_minutes'], alpha=0.3)
    plt.title(f"{col} vs Listening Time")
    plt.show()


cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=col, y='Listening_Time_minutes', data=train)
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Listening Time")
    plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(train[num_cols + ['Listening_Time_minutes']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Combined popularity interaction
train['Combined_Popularity'] = train['Host_Popularity_percentage'] * train['Guest_Popularity_percentage']
train['Combined_Popularity'] = train['Combined_Popularity'].fillna(0)

# Ads per minute
train['Ads_per_Minute'] = train['Number_of_Ads'] / train['Episode_Length_minutes']
train['Ads_per_Minute'] = train['Ads_per_Minute'].replace([np.inf, -np.inf], 0).fillna(0)

# Title word count
train['Title_Word_Count'] = train['Episode_Title'].apply(lambda x: len(str(x).split()))

# Weekend flag (if day name is available)
train['Is_Weekend'] = train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
train['Sentiment_Score'] = train['Episode_Sentiment'].map(sentiment_map)

# Ad_Impact: Ads * sentiment score (maybe ads affect listener mood)
train['Ad_Impact'] = train['Number_of_Ads'] * train['Sentiment_Score']

# Guest_Impact: Only if there's a guest
train['Guest_Impact'] = train['Has_Guest'] * train['Guest_Popularity_percentage'].fillna(0)


train.columns


test=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


test.isna().sum()


# Fill Episode_Length_minutes with median
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)

# Create a new column to track whether a guest was present
test['Has_Guest'] = test['Guest_Popularity_percentage'].notnull().astype(int)

# Fill missing guest popularity with 0
test['Guest_Popularity_percentage'].fillna(0, inplace=True)


# Combined popularity interaction
test['Combined_Popularity'] = test['Host_Popularity_percentage'] * test['Guest_Popularity_percentage']
test['Combined_Popularity'] = test['Combined_Popularity'].fillna(0)

# Ads per minute
test['Ads_per_Minute'] = test['Number_of_Ads'] / test['Episode_Length_minutes']
test['Ads_per_Minute'] = test['Ads_per_Minute'].replace([np.inf, -np.inf], 0).fillna(0)

# Title word count
test['Title_Word_Count'] = test['Episode_Title'].apply(lambda x: len(str(x).split()))

# Weekend flag (if day name is available)
test['Is_Weekend'] = test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
test['Sentiment_Score'] = test['Episode_Sentiment'].map(sentiment_map)

# Ad_Impact: Ads * sentiment score (maybe ads affect listener mood)
test['Ad_Impact'] = test['Number_of_Ads'] * test['Sentiment_Score']

# Guest_Impact: Only if there's a guest
test['Guest_Impact'] = test['Has_Guest'] * test['Guest_Popularity_percentage'].fillna(0)


drop_cols = ['id', 'Podcast_Name', 'Episode_Title', 'Listening_Time_minutes']  # if it's the target
X = train.drop(columns=drop_cols)
y = train['Listening_Time_minutes']

# List of numerical columns
num_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads',
    'Combined_Popularity', 'Ads_per_Minute', 'Title_Word_Count', 'Sentiment_Score', 'Ad_Impact', 'Guest_Impact',
]

X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
# Initialize RobustScaler
scaler = RobustScaler()

# Fit on train and transform
X[num_cols] = scaler.fit_transform(X[num_cols])

# List of categorical columns
cat_cols = [
    'Genre', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment', 'Has_Guest', 'Is_Weekend'
]

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# For test data later:
# X_test_processed = preprocessor.transform(X_test)
X.columns


drop_col = ['id', 'Podcast_Name', 'Episode_Title']  # if it's the target
X_test = test.drop(columns=drop_col)

# List of numerical columns
num_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads',
    'Combined_Popularity', 'Ads_per_Minute', 'Title_Word_Count', 'Sentiment_Score', 'Ad_Impact', 'Guest_Impact',
]

X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan)

# Fit on train and transform
X_test[num_cols] = scaler.transform(X_test[num_cols])

# List of categorical columns
cat_cols = [
    'Genre', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment', 'Has_Guest', 'Is_Weekend'
]

X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# For test data later:
# X_test_processed = preprocessor.transform(X_test)
X_test.columns


xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)


# Fit the model
xgb_model.fit(X, y)

# Get feature importance
importance = xgb_model.feature_importances_
features = X.columns if isinstance(X, pd.DataFrame) else feature_names_list  # Adjust if using NumPy

# Create a DataFrame for visualization
imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
imp_df = imp_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(12, 8))
plt.barh(imp_df['Feature'][:20], imp_df['Importance'][:20])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances from XGBoost")
plt.show()


top_features = [
    'Episode_Length_minutes',
    'Ads_per_Minute',
    'Number_of_Ads',
    'Ad_Impact',
    'Has_Guest_1',
    'Sentiment_Score',
    'Episode_Sentiment_Positive',
    'Host_Popularity_percentage',
    'Genre_Sports',
    'Episode_Sentiment_Neutral'
]


# Use only top features
X_top = X[top_features]

# Initialize XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    objective='reg:squarederror'
)

# Train model
xgb_model.fit(X, y)

# Predict
y_pred = xgb_model.predict(X_test)


# submission = pd.DataFrame({
#     "id": test["id"],       # or use test_df["id"] if you kept original test set
#     "target": y_pred          # your predicted labels
# })
# submission.to_csv("submission.csv", index=False)


# Fit the LightGBM model on all data
lgb_model_full = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
lgb_model_full.fit(X, y)

# Get feature importances
importances = pd.Series(lgb_model_full.feature_importances_, index=X.columns)
top_lgb_features = importances.sort_values(ascending=False).head(20)

# Plot
plt.figure(figsize=(10, 6))
top_lgb_features.plot(kind='barh')
plt.title("Top 20 Feature Importances from LightGBM")
plt.gca().invert_yaxis()
plt.show()


# Select top features
top_lgbm_features = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Ads_per_Minute',
    'Guest_Popularity_percentage',
    'Sentiment_Score',
    'Genre_Technology',
    'Genre_True Crime',
    'Combined_Popularity',
    'Number_of_Ads',
    'Publication_Day_Thursday'
]

X_top_lgb = X[top_lgbm_features]

model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X, y)

# Predictions
y_pred = model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],       # or use test_df["id"] if you kept original test set
    "target": y_pred          # your predicted labels
})
submission.to_csv("submission.csv", index=False)


# Initialize model
cat_model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=0  # suppress training output
)

# Fit model
cat_model.fit(X, y)

# Predict
y_pred = cat_model.predict(X_test)


feature_importances = cat_model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


feature_importances = cat_model.get_feature_importance()
feature_names = X.columns

# Create a DataFrame of features and their importance
feat_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

# Select top N features
top_n = 15
top_features = feat_imp_df['feature'].head(top_n).tolist()
print("Top Features:", top_features)


# Filter training and validation sets
X_train_top = X[top_features]
X_val_top = X_test[top_features]

# Train model on selected features
cat_model_top = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_state=42,
    verbose=0
)

cat_model_top.fit(X_train_top, y)

# Predict and calculate RMSE
y_pred_top = cat_model_top.predict(X_val_top)


# submission = pd.DataFrame({
#     "id": test["id"],       # or use test_df["id"] if you kept original test set
#     "target": y_pred_top          # your predicted labels
# })
# submission.to_csv("submission.csv", index=False)


# Define models
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
lgb = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
cat = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0)

models = [xgb, lgb, cat]
model_names = ['xgb', 'lgb', 'cat']
oof_preds = np.zeros((X.shape[0], len(models)))
test_preds = np.zeros((X_test.shape[0], len(models)))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, model in enumerate(models):
    fold_preds = np.zeros(X.shape[0])
    fold_test_preds = np.zeros((X_test.shape[0], 5))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        fold_preds[val_idx] = model.predict(X_val_fold)
        fold_test_preds[:, fold] = model.predict(X_test)

    oof_preds[:, i] = fold_preds
    test_preds[:, i] = fold_test_preds.mean(axis=1)

# Train meta-model
meta_model = Ridge()
meta_model.fit(oof_preds, y)
final_preds = meta_model.predict(test_preds)

# Optionally evaluate meta-model
oof_meta = meta_model.predict(oof_preds)
rmse = mean_squared_error(y, oof_meta, squared=False)
print("Stacked Model RMSE:", rmse)


# submission = pd.DataFrame({
#     "id": test["id"],
#     "target": final_preds
# })
# submission.to_csv("submission.csv", index=False)

