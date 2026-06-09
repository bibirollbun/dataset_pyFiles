import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import optuna # 하이퍼파라미터 튜닝.
from sklearn.model_selection import train_test_split # 데이터 분할.
from sklearn.preprocessing import StandardScaler # 스케일링.
from sklearn.model_selection import StratifiedKFold # K-겹 교차검증.

from sklearn.linear_model import LogisticRegression # 로지스틱 회귀.
from sklearn.svm import SVC # 서포트 벡터 머신.
import lightgbm as lgb # LightGBM.
import xgboost as xgb # XGBoost.

import math # 수학 함수.

from sklearn.metrics import roc_auc_score, accuracy_score # 평가 지표.

import warnings # 경고 메시지 제어.
warnings.filterwarnings("ignore", category=FutureWarning) # FutureWarning 무시.


# 데이터 로드: 학습, 테스트, 제출 샘플.
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


# 학습 데이터 상위 5행 확인.
df_train.head()


# 학습 데이터 하위 5행 확인.
df_train.tail()


# 각 데이터프레임 형태(shape) 출력.
print(df_train.shape)
print(df_test.shape)
print(df_sub.shape)


# 학습 데이터 정보 요약 출력.
print(df_train.info(), '\n\n')
# 테스트 데이터 정보 요약 출력.
print(df_test.info())


# 학습 데이터 기술 통계량 출력.
print(df_train.describe(), '\n\n')
# 테스트 데이터 기술 통계량 출력.
print(df_test.describe())


# 분석 대상 컬럼 이름 리스트 정의.
cols = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
        'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
        'windspeed', 'rainfall']


# 각 컬럼 고유값 개수 출력. 반복.
for col in cols:
    print(col, df_train[col].nunique())


# 특정 컬럼('humidity', 'cloud', 'winddirection') 값 빈도수 출력. 반복.
for col in ['humidity', 'cloud', 'winddirection']:
    print(df_train[col].value_counts(), '\n\n')


# 수치형 특성 리스트 정의.
num_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
# 타겟 변수 정의.
target = 'rainfall'


# 타겟 변수('rainfall') 값 빈도수 계산.
counts = df_train['rainfall'].value_counts()
labels = counts.index
values = counts.values

# 그래프 크기 설정.
plt.figure(figsize=(15,5.5))

# 수평 막대 그래프 생성.
bars = plt.barh(labels, values)
plt.ylabel("Rainfall") # y축 라벨.
plt.xlabel("Frequency") # x축 라벨.
plt.title("The Distribution of the Target Column 'rainfall'") # 그래프 제목.

plt.yticks([1, 0]) # y축 눈금 설정.

total = values.sum()
# 각 막대에 값과 백분율 텍스트 추가. 반복.
for bar, count in zip(bars, values):
    width = bar.get_width()
    pct = count / total * 100
    plt.text(width, bar.get_y() + bar.get_height()/2,
             f"{count}\n({pct:.1f}%)",
             ha='left', va='center')
plt.show() # 그래프 표시.


# 수치형 특성 개수.
n_vars = len(num_cols)
# 서브플롯 생성: n_vars 행, 2 열.
fig, axes = plt.subplots(n_vars, 2, figsize=(12, n_vars * 3))

# 각 수치형 특성에 대해 히스토그램 및 박스플롯. 반복.
for i, col in enumerate(num_cols):
    # 히스토그램.
    axes[i, 0].hist(df_train[col], bins=60, edgecolor='black')
    axes[i, 0].set_title(f"{col}'s Histogram")

    # 박스플롯.
    axes[i, 1].boxplot(df_train[col], vert=False)
    axes[i, 1].set_title(f"{col}'s Boxplot")

plt.tight_layout() # 레이아웃 자동 조정.
plt.show() # 그래프 표시.


# 수치형 특성 개수.
n_vars = len(num_cols)
n_cols = 2  # 열 개수.
n_rows = (n_vars + n_cols -1) // n_cols # 필요한 행 개수.

# 서브플롯 생성.
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))

# 타겟 변수별 각 수치형 특성 박스플롯. 반복.
for i, col in enumerate(num_cols):
    row = i // n_cols
    col_idx = i % n_cols
    sns.boxplot(x='rainfall', y=col, data=df_train, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f"{col} by rainfall")

# 마지막 행의 빈 서브플롯 제거.
if n_vars % n_cols != 0:
    for j in range(n_vars % n_cols, n_cols):
        fig.delaxes(axes[n_rows-1, j])

plt.tight_layout() # 레이아웃 자동 조정.
plt.show() # 그래프 표시.


def remove_outliers(df, cols):
    
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df


df_train = remove_outliers(df_train, num_cols)


def create_features(df):
    
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    
    df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    
    df['humidity_temp_dew_ratio'] = df['humidity'] / (df['temp_dew_diff'].replace(0, 0.1))
    
    df['humidity_index'] = df['dewpoint'] * (df['humidity'] / 100)
    
    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'].replace(0, 0.1))

    df['expected_day'] = (df['id']) % 365 + 1
    
    df['day_mislabelled'] = df['day'] != df['expected_day']
    
    # df['wind_dir_sin'] = np.sin(np.deg2rad(df['winddirection']))
    
    return df


df_cleaned = create_features(df_train)
df_test = create_features(df_test)


num_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed',
            'temp_range', 'temp_dew_diff', 'humidity_temp_dew_ratio', 
            'humidity_index', 'cloud_sunshine_ratio', 'wind_dir_sin']


def prepare_data(df_train, target_col, num_cols):
        
    X = df_cleaned.drop(columns=[target_col])
    y = df_cleaned[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def create_models():
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    
    # 2. LightGBM
    lgb_params = {
        'n_estimators': 303,
        'learning_rate': 0.007220454649478007,
        'num_leaves': 85,
        'max_depth': 12,
        'subsample': 0.953589998504466,
        'colsample_bytree': 0.5526017783925301,
        'reg_alpha': 0.006950801637508764,
        'reg_lambda': 4.919943534568941,
        'random_state': 42,
        'verbose': -1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    # 3. XGBoost
    xgb_params = {
        'n_estimators': 183,
        'learning_rate': 0.001568933066732297,
        'max_depth': 11,
        'subsample': 0.6353249904132632,
        'colsample_bytree': 0.6794856890789495,
        'min_child_weight': 8,
        'gamma': 0.8227004044747224,
        'reg_alpha': 0.005817917041980371,
        'reg_lambda': 0.0014801406943793138,
        'random_state': 42
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    return lr_model, lgb_model, xgb_model


def ensemble_predict(models, X):
    lr_model, lgb_model, xgb_model = models
    
    lr_pred = lr_model.predict_proba(X)[:, 1]
    lgb_pred = lgb_model.predict_proba(X)[:, 1]
    xgb_pred = xgb_model.predict_proba(X)[:, 1]
    
    ensemble_pred_proba = np.mean([lr_pred, lgb_pred, xgb_pred], axis=0)
    ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
    
    return ensemble_pred, ensemble_pred_proba


def main(df_train, target_col, num_cols):
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(
        df_train, target_col, num_cols
    )
    
    lr_model, lgb_model, xgb_model = create_models()
    
    print("Training models...")
    lr_model.fit(X_train_scaled, y_train)
    lgb_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)
    
    models = {'Logistic Regression': lr_model, 
              'LightGBM': lgb_model, 
              'XGBoost': xgb_model}
    
    for name, model in models.items():
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)
        print(f"{name} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
    
    ensemble_pred, ensemble_pred_proba = ensemble_predict(
        [lr_model, lgb_model, xgb_model], 
        X_test_scaled
    )
    
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
    print(f"\nEnsemble - Accuracy: {ensemble_acc:.4f}, AUC: {ensemble_auc:.4f}")
    
    return scaler, lr_model, lgb_model, xgb_model

