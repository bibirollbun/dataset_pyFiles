import warnings
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)



train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)


train_df.columns


train_df.hist("Listening_Time_minutes")


test_df


target_col = "Listening_Time_minutes"
feature_cols = [col for col in train_df.columns if col != target_col]

X = train_df[feature_cols]
y = train_df[target_col]

# カテゴリ変数を指定
cat_features = [
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment"
    
]



X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(X_train, y_train, eval_set=(X_valid, y_valid))



y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred) ** 0.5
print(f"RMSE: {rmse}")



importance = model.get_feature_importance()
feature_names = X.columns

# DataFrameにまとめると見やすい
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# グラフ化
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()


# test_dfの特徴量をそのまま入れる（カテゴリ列はそのままでOK）
X_test = test_df.copy()

# 予測
pred_test = model.predict(X_test)

# 提出用DataFrame作成
submission = pd.DataFrame({
    'id': X_test.index,
    'Listening_Time_minutes': pred_test
})


# CSV出力
submission.to_csv('submission.csv', index=False)


submission.hist("Listening_Time_minutes")




