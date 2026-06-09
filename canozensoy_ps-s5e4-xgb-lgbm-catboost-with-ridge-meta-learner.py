# ğŸš€ XGBoost Ensemble Pipeline for Podcast Listening Time Prediction
import pandas as pd, numpy as np, gc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor


# === Load Data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


# === Remove Extreme Outliers ===
q1 = train['Listening_Time_minutes'].quantile(0.01)
q99 = train['Listening_Time_minutes'].quantile(0.99)
train = train[(train['Listening_Time_minutes'] > q1) & (train['Listening_Time_minutes'] < q99)]


# === Fill Missing Values ===
for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())


# === Feature Engineering ===
def enrich(df):
    df['Popularity_Diff'] = abs(df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage'])
    df['Combined_Popularity'] = 0.7 * df['Host_Popularity_percentage'] + 0.3 * df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 0.1)
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_Prime'] = df['Publication_Time'].isin(['Evening', 'Night']).astype(int)
    df['Sentiment_Score'] = df['Episode_Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    df['Ad_Sentiment'] = df['Sentiment_Score'] * df['Number_of_Ads']
    df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    df['Log_Episode_Length'] = np.log1p(df['Episode_Length_minutes'])
    df['Host_Impact'] = df['Host_Popularity_percentage'] * df['Episode_Length_minutes']
    return df

train = enrich(train)
test = enrich(test)


# === Encode Categorical ===
cat_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
for col in cat_cols:
    train[col] = train[col].astype("category").cat.codes
    test[col] = test[col].astype("category").cat.codes


# === Define Features and Target ===
TARGET = 'Listening_Time_minutes'
features = [col for col in train.columns if col not in ['id', TARGET, 'Episode_Title']]
X = train[features]
y = np.log1p(train[TARGET])
X_test = test[features]


# === Define Base Models ===
xgb_model = XGBRegressor(
    n_estimators=2500,
    learning_rate=0.01,
    max_depth=10,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.6,
    reg_lambda=2,
    tree_method='hist',
    enable_categorical=False,
    random_state=42,
    verbosity=0
)

lgbm_model = LGBMRegressor(
    n_estimators=2500,
    learning_rate=0.01,
    max_depth=10,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.6,
    reg_lambda=2,
    random_state=42,
    verbose=-1
)

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.01,
    depth=10,
    l2_leaf_reg=3,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0
)



# === Stacking ===
stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('cat', cat_model)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    passthrough=True,
    n_jobs=-1
)


# === K-Fold Training ===
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"ğŸ”� Fold {fold}")
    X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    stack_model.fit(X_train, y_train)
    oof[val_idx] = stack_model.predict(X_valid)
    preds += stack_model.predict(X_test) / kf.n_splits
    gc.collect()


# === Evaluation ===
rmse = mean_squared_error(np.expm1(y), np.expm1(oof), squared=False)
print(f"ğŸ�� Final OOF RMSE (Stacked): {rmse:.5f}")


# === Submission ===
preds = np.minimum(preds, 20)  # log1p(485M) â‰ˆ 20
final_preds = np.clip(np.expm1(preds), 0, 200)
sub = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': final_preds
})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("ğŸš€ Submission Saved!")

