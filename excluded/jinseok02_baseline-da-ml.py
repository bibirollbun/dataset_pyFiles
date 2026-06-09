# ✅ 1. 라이브러리 불러오기
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ✅ 2. 데이터 불러오기 (kaggle dataset 경로에 맞게 조정)
train = pd.read_csv("/kaggle/input/spotify-da-ml/train.csv")
test = pd.read_csv("/kaggle/input/spotify-da-ml/test.csv")
submission = pd.read_csv("/kaggle/input/spotify-da-ml/sample_submission.csv")


# ✅ 3. 범주형 변수 인코딩 (track_genre만)
le = LabelEncoder()
train["track_genre"] = le.fit_transform(train["track_genre"])
test["track_genre"] = le.transform(test["track_genre"])

# ✅ 4. 학습/검증 분리
X = train.drop(columns=["id", "popularity"])
y = train["popularity"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ✅ 5. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# ✅ 6. 검증 RMSE 출력
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"Validation RMSE: {rmse:.4f}")


# ✅ 7. 테스트셋 예측 및 제출 파일 생성
X_test = test.drop(columns=["id"])
test_preds = model.predict(X_test)
submission["popularity"] = test_preds
submission.to_csv("baseline_submission.csv", index=False)

print("✅ baseline_submission.csv 파일 저장 완료!")




