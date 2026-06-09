import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import random
np.random.seed(1234)
random.seed(1234)


train_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
test_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')


# remove warning
import warnings
warnings.filterwarnings("ignore")


train_df.shape


test_df.shape


train_df.info()


train_df.head()


test_df.head()


train_df.describe().T


train_df.isnull().sum()


test_df.isnull().sum()


# 결측치 거
train_df = train_df.dropna(subset=["winPlacePerc"])


train_df["winPlacePerc"].plot.hist(bins=20)


train_df["matchId"].value_counts()


train_df["groupId"].value_counts()


# 제출용 Id
test_Id = test_df["Id"].copy()


sample_df = train_df.sample(200000, random_state=1234)


# pandas의 ydata_profiling을 사용해 피처들의 시각화 데이터 확인하기
!pip install ydata-profiling
from ydata_profiling import ProfileReport


profile = ProfileReport(sample_df, title="EDA Report", explorative=True)
profile = ProfileReport(sample_df, title="EDA Report (sampled)", minimal=True)
profile.to_notebook_iframe()


# 히트맵
num_df = train_df.select_dtypes(include=[np.number])
corr = num_df.corr(method="pearson")

plt.figure(figsize=(12, 10))         
sns.heatmap(
    corr,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap (Numeric Features)")
plt.xticks(rotation=90)            
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# 각 변수들의 상관관계 순위
K = 10
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values()

print(f"High Top {K} +corr"); print(pairs.tail(K)[::-1])
print(f"\nLow Top {K} -corr"); print(pairs.head(K))


# 타깃과의 상관관계 순위
TARGET, K = "winPlacePerc", 10
s = corr[TARGET].drop(TARGET).dropna().sort_values()

print(f"[{TARGET}] High Top {K} +corr"); print(s.tail(K)[::-1])
print(f"\n[{TARGET}] Low Top {K} -corr"); print(s.head(K))


TARGET = "winPlacePerc"

train_X = train_df.drop(columns=[TARGET]).copy()
test_X  = test_df.copy()

tmp = pd.concat([
    train_X.assign(__is_train__=1),
    test_X.assign(__is_train__=0)
], ignore_index=True)

tmp = pd.get_dummies(tmp, columns=["matchType"], drop_first=False)

train_enc = tmp[tmp["__is_train__"]==1].drop(columns="__is_train__")
test_enc = tmp[tmp["__is_train__"]==0].drop(columns="__is_train__")


# 베이스라인: LightGBM + GroupKFold(MAE) - 콜백 방식(버전 호환)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

TARGET = "winPlacePerc"

# 0) 준비: 타깃 결측 제거 & 그룹키 저장(피처에서는 드롭해도 groups로 사용)
groups = df["matchId"].copy()

# 1) 드롭 조합 정의
drop_map = {
    "train_Id_rm"   : ["Id","matchId","groupId"],
    "train_points_rm": ["Id","matchId","groupId","killPoints","rankPoints","winPoints"],
    "train_kP_rm"   : ["Id","matchId","groupId","killPlace"],
    "train_mPnG_rm" : ["Id","matchId","groupId","maxPlace","numGroups"],
    "train_all_rm"  : ["Id","matchId","groupId","killPoints","rankPoints","winPoints","killPlace","maxPlace","numGroups"],
}

def cv_mae_for(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    maes = []
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        model = LGBMRegressor(
            n_estimators=10000,           # 크게 주고 조기종료로 최적 반복수 선택
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            eval_metric="l1",             # MAE
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)  # 학습 로그 끔
            ],
        )
        pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration_)
        maes.append(mean_absolute_error(y.iloc[va_idx], pred))
    return float(np.mean(maes)), float(np.std(maes))

# 3) 각 버전 평가
y = [TARGET]
results = {}
for name, to_drop in drop_map.items():
    drop_cols = [c for c in to_drop if c in df.columns]  # 안전 드롭
    X = df.drop(columns=[TARGET] + drop_cols)
    mean_mae, std_mae = cv_mae_for(X, y, groups)
    results[name] = (mean_mae, std_mae)
    print(f"{name:15s}  CV MAE = {mean_mae:.5f} ± {std_mae:.5f}")

# 4) 최적 조합 표시
best_name = min(results, key=lambda k: results[k][0])
print("\n>>> Best by CV MAE:", best_name, "=", f"{results[best_name][0]:.5f} ± {results[best_name][1]:.5f}")

