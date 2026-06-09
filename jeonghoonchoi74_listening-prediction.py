# Step 0. 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")



# Step 1. 데이터 불러오기 및 결합
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)
all_data = pd.concat([train, test], axis=0)



# Step 2. 변수 구분
numerical_features = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads'
]

categorical_features = [
    'Podcast_Name',
    'Episode_Title',
    'Genre',
    'Publication_Day',
    'Publication_Time',
    'Episode_Sentiment'
]

target = 'Listening_Time_minutes'



# Step 3. 이상치 처리 (IQR 기반 clipping)
for feature in numerical_features:
    Q1 = all_data[feature].quantile(0.25)
    Q3 = all_data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    all_data[feature] = all_data[feature].clip(lower=lower_bound, upper=upper_bound)



# Step 4. 결측치 처리
# 5% 미만: 바로 중위수로 보간
all_data['Number_of_Ads'].fillna(all_data['Number_of_Ads'].median(), inplace=True)

# 5~20%: 중위수 보간 + 결측 플래그 생성
for feature in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    flag = f"{feature}_imputed"
    all_data[flag] = all_data[feature].isnull().astype(int)
    all_data[feature].fillna(all_data[feature].median(), inplace=True)



# Step 5. 범주형 변수 Label Encoding
for feature in categorical_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature].astype(str))



# Step 6. 파생 피처 생성
all_data["Host_to_Guest_Popularity"] = all_data["Host_Popularity_percentage"] / (all_data["Guest_Popularity_percentage"] + 1e-5)
all_data["Popularity_per_Minute"] = (all_data["Host_Popularity_percentage"] + all_data["Guest_Popularity_percentage"]) / (all_data["Episode_Length_minutes"] + 1e-5)
all_data["Ads_per_Minute"] = all_data["Number_of_Ads"] / (all_data["Episode_Length_minutes"] + 1e-5)
all_data["Log_Episode_Length"] = np.log1p(all_data["Episode_Length_minutes"].clip(lower=0))

all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
all_data.fillna(0, inplace=True)



# Step 7. train/test 재분리
train_processed = all_data[all_data.index < 750000]
test_processed = all_data[all_data.index >= 750000]



# Step 8. 피처 및 타겟 설정
features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
    "Host_to_Guest_Popularity",
    "Popularity_per_Minute",
    "Ads_per_Minute",
    "Log_Episode_Length",
    "Episode_Length_minutes_imputed",
    "Guest_Popularity_percentage_imputed"
]

X = train_processed[features]
y = np.log1p(train_processed[target])  # 로그 변환된 타겟
X_test = test_processed[features]



# Step 9. 모델 구성 (LightGBM + XGBoost)
lgb_model = lgb.LGBMRegressor(
    max_depth=10,
    num_leaves=231,
    learning_rate=0.0443,
    n_estimators=722,
    min_child_samples=21,
    subsample=0.7364,
    colsample_bytree=0.7424,
    reg_alpha=0.002,
    reg_lambda=0.079,
    objective='regression',
    random_state=42
)

xgb_model = xgb.XGBRegressor(
    max_depth=7,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.5,
    reg_lambda=0.5,
    objective='reg:squarederror',
    random_state=42
)

ensemble_model = VotingRegressor(estimators=[('lgb', lgb_model), ('xgb', xgb_model)])



# Step 10. 학습 및 예측
ensemble_model.fit(X, y)
preds_log = ensemble_model.predict(X_test)
preds = np.expm1(preds_log)  # 역변환



# Step 11. 제출 파일 생성
submission = pd.DataFrame({
    "id": test_processed.index,
    "Listening_Time_minutes": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ 제출 파일 저장 완료: submission.csv")


