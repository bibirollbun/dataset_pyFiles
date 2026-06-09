# 라이브러리 임포트
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 시드 고정
import random
np.random.seed(1234)
random.seed(1234)


# 데이터셋 불러오기
train_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
test_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
submission = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv')


# 원본 데이터 백업
train_df_original = train_df.copy()
test_df_original = test_df.copy()


# 상위 5행 확인
train_df.head()


# 데이터 타입 확인
train_df.dtypes


# 결측치 개수 확인
train_df.isnull().sum()


test_df.isnull().sum()


# 제출용 Id 보관
test_id = test_df["Id"].copy()


# 결측치가 있는 행 제거
train_df = train_df.dropna(subset=["winPlacePerc"])


# 전처리 -> Id,groupId,matchId drop하기 / matchType 원핫인코딩

# 불필요한 id 계열 컬럼 제거
train_df = train_df.drop(["Id", "groupId", "matchId"], axis=1)
test_df  = test_df.drop(["Id", "groupId", "matchId"], axis=1)

# matchType 원핫인코딩
train_df = pd.get_dummies(train_df, columns=["matchType"])
test_df  = pd.get_dummies(test_df, columns=["matchType"])

# train/test 컬럼 차이 맞추기 (혹시 모를 불일치 대비)
train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)


# X,y 분리
y = train_df["winPlacePerc"]
X = train_df.drop(columns=["winPlacePerc"])
X_test = test_df.copy()

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# KFold + LightGBM
kf = KFold(n_splits=3, shuffle=True, random_state=1234)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
rmses = []

# ---- 빠른 베이스라인 파라미터 ----
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.18,   # ↑ 수렴 가속
    "num_leaves": 31,        # ↓ 트리 복잡도
    "max_depth": 6,          # 깊이 제한
    "min_data_in_leaf": 300, # 리프 최소 샘플(속도/안정성↑)
    "feature_fraction": 0.7, # 컬럼 서브샘플
    "bagging_fraction": 0.7, # 행 서브샘플
    "bagging_freq": 5,       # 배깅 주기
    "max_bin": 31,           # 히스토그램 bin↓
    "extra_trees": True,     # 분할 무작위화(속도↑)
    "lambda_l2": 0.1,
    "num_threads": -1,
    "verbosity": -1,
    "random_state": 1234,
    # GPU 가능 시 (느리면 주석 처리)
    # "device": "gpu",
    # "gpu_platform_id": 0,
}

NUM_BOOST_ROUND = 700   # ↓ 1000 -> 700
EARLY_STOP = 60         # ↑ 더 민감한 조기종료
LOG_EVERY = 200

for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
    # DataFrame에서 행 인덱스로 분할 (X, y는 reset_index 된 DF 가정)
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    # fold마다 실제 사용하는 feature 목록 고정
    train_cols = X_trn.columns.tolist()

    dtrn = lgb.Dataset(X_trn[train_cols], label=y_trn, feature_name=train_cols)
    dval = lgb.Dataset(X_val[train_cols], label=y_val, feature_name=train_cols)

    model = lgb.train(
        params=lgb_params,
        train_set=dtrn,
        valid_sets=[dval],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP),
            lgb.log_evaluation(LOG_EVERY),
        ],
    )

    # 검증 예측
    oof[val_idx] = model.predict(X_val[train_cols], num_iteration=model.best_iteration)
    fold_rmse = mean_squared_error(y_val, oof[val_idx], squared=False)
    rmses.append(fold_rmse)
    print(f"[Fold {fold}] RMSE: {fold_rmse:.5f} | best_iter={model.best_iteration}")

    # 테스트 예측도 동일한 컬럼으로 맞춰줌
    test_preds += model.predict(X_test[train_cols], num_iteration=model.best_iteration) / kf.n_splits

print(f"\nOOF RMSE (mean): {np.mean(rmses):.5f} | (std): {np.std(rmses):.5f}")



'''
# 제출파일 만들기
sub = pd.DataFrame({
    "Id": test_id,
    "winPlacePerc": np.clip(test_preds, 0.0, 1.0)
})
sub.to_csv("submission_lgbm_kfold3.csv", index=False)
print("Saved: submission_lgbm_kfold3.csv")
'''


# 상관관계 히트맵 확인하기

#숫자형 컬럼 선택
num_feats = train_df.select_dtypes(include=["int64", "float64"])

# 상관계수 계산
corr = num_feats.corr()

# 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.show()

# winPlacePerc과만 비교버전
corr_target = corr["winPlacePerc"].sort_values(ascending=False)
print(corr_target.head(20))


# 준비 -> train / test 합치기
tr = train_df_original.copy(); tr["is_train"] = 1
te = test_df_original.copy();  te["is_train"] = 0
df = pd.concat([tr, te], axis=0, sort=False)


# 간단한 파생변수 만들기

# 이동 총합
df["totalDistance"] = df["walkDistance"] + df["rideDistance"] + df["swimDistance"]
# 아이템 묶기 (힐/부스트)
df["healItems"] = df["heals"] + df["boosts"]
# 지원 묶기 (어시/리바이브)
df["support"] = df["assists"] + df["revives"]


# 비율/효율 관련 피처 확장

# 1킬당 데미지
df["killEfficiency"] = df["damageDealt"] / (df["kills"] + 1)
# 헤드샷 비율
df["headshotRate"] = df["headshotKills"] / (df["kills"] + 1)
# 무기당 킬
df["killsPerWeapon"] = df["kills"] / (df["weaponsAcquired"] + 1)
# 부스트/힐 비율
df["boostHealRatio"] = df["boosts"] / (df["heals"] + 1)

# 이동성향 비율
den = df["totalDistance"].replace(0, np.nan) #총합
df["walkRatio"] = (df["walkDistance"]/den).fillna(0) #걷기
df["rideRatio"] = (df["rideDistance"]/den).fillna(0) #차
df["swimRatio"] = (df["swimDistance"]/den).fillna(0) #수영

# 순위 정규화 -> 분모 0 방지
df["killPlaceNorm"] = (df["killPlace"] / df["maxPlace"].replace(0, np.nan)).fillna(1.0)


# 팀단위 집계

# 팀 크기
df["group_size"] = df.groupby("groupId")["Id"].transform("count")
# 팀 킬 평균/합
df["group_kills_mean"] = df.groupby("groupId")["kills"].transform("mean")
df["group_kills_sum"] = df.groupby("groupId")["kills"].transform("sum")
# 팀 평균 이동거리/힐 아이템 사용
df["group_totalDist_mean"] = df.groupby("groupId")["totalDistance"].transform("mean")
df["group_healItems_mean"] = df.groupby("groupId")["healItems"].transform("mean")


# 매치단위 집계

# 매치 참가자수
df["match_players"] = df.groupby("matchId")["Id"].transform("count")
# 매치 평균 걷기/데미지/총이동/킬
df["match_walk_mean"] = df.groupby("matchId")["walkDistance"].transform("mean")
df["match_damage_mean"] = df.groupby("matchId")["damageDealt"].transform("mean")
df["match_totalDist_mean"] = df.groupby("matchId")["totalDistance"].transform("mean")
df["match_kills_mean"] = df.groupby("matchId")["kills"].transform("mean")


# 상대지표

eps = 1e-6 #0으로 나누는걸 방지
# 매치 평균 대비 비율
df["rel_walk"] = df["walkDistance"] / (df["match_walk_mean"] + eps)
df["rel_damage"] = df["damageDealt"] / (df["match_damage_mean"] + eps)
df["rel_totalDist"] = df["totalDistance"] / (df["match_totalDist_mean"] + eps)

# 팀 평균 대비 차이
df["diff_kills_vs_groupmean"] = df["kills"] - df["group_kills_mean"]


# matchType 원핫 (희귀 모드는 'other'로 묶어도 좋음)
df = pd.get_dummies(df, columns=["matchType"])

# 학습에 안 쓰는 식별자 제거
df = df.drop(columns=["Id","groupId","matchId"])

# 다시 train/test로 분리 + 칼럼 맞춤
train_fe = df[df["is_train"]==1].drop(columns=["is_train"])
test_fe  = df[df["is_train"]==0].drop(columns=["is_train"])
train_fe, test_fe = train_fe.align(test_fe, join="left", axis=1, fill_value=0)


# 데이터 셋 나누기
y = train_fe["winPlacePerc"]
X = train_fe.drop(columns=["winPlacePerc"])
X_test = test_fe.copy()

# NaN 제거
mask = y.notna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# KFold + LightGBM
kf = KFold(n_splits=3, shuffle=True, random_state=1234)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
rmses = []
models = []

# ---- 빠른 베이스라인 파라미터 ----
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.18,   # ↑ 수렴 가속
    "num_leaves": 31,        # ↓ 트리 복잡도
    "max_depth": 6,          # 깊이 제한
    "min_data_in_leaf": 300, # 리프 최소 샘플(속도/안정성↑)
    "feature_fraction": 0.7, # 컬럼 서브샘플
    "bagging_fraction": 0.7, # 행 서브샘플
    "bagging_freq": 5,       # 배깅 주기
    "max_bin": 31,           # 히스토그램 bin↓
    "extra_trees": True,     # 분할 무작위화(속도↑)
    "lambda_l2": 0.1,
    "num_threads": -1,
    "verbosity": -1,
    "random_state": 1234,
    # GPU 가능 시 (느리면 주석 처리)
    # "device": "gpu",
    # "gpu_platform_id": 0,
}

NUM_BOOST_ROUND = 700   # ↓ 1000 -> 700
EARLY_STOP = 60         # ↑ 더 민감한 조기종료
LOG_EVERY = 200

for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
    # DataFrame에서 행 인덱스로 분할 (X, y는 reset_index 된 DF 가정)
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    # fold마다 실제 사용하는 feature 목록 고정
    train_cols = X_trn.columns.tolist()

    dtrn = lgb.Dataset(X_trn[train_cols], label=y_trn, feature_name=train_cols)
    dval = lgb.Dataset(X_val[train_cols], label=y_val, feature_name=train_cols)

    model = lgb.train(
        params=lgb_params,
        train_set=dtrn,
        valid_sets=[dval],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP),
            lgb.log_evaluation(LOG_EVERY),
        ],
    )

    models.append(model) # 모델저장
    
    # 검증 예측
    oof[val_idx] = model.predict(X_val[train_cols], num_iteration=model.best_iteration)
    fold_rmse = mean_squared_error(y_val, oof[val_idx], squared=False)
    rmses.append(fold_rmse)
    print(f"[Fold {fold}] RMSE: {fold_rmse:.5f} | best_iter={model.best_iteration}")

    # 테스트 예측도 동일한 컬럼으로 맞춰줌
    test_preds += model.predict(X_test[train_cols], num_iteration=model.best_iteration) / kf.n_splits

print(f"\nOOF RMSE (mean): {np.mean(rmses):.5f} | (std): {np.std(rmses):.5f}")


# 피쳐중요도 출력
importances = pd.Series(0.0, index=X.columns)

for m in models:
    importances += pd.Series(m.feature_importance(importance_type="gain"), index=X.columns)

importances /= len(models)

# 상위 30 출력
print("\nTop 30 Feature Importances (avg gain):")
print(importances.sort_values(ascending=False).head(30))

# 시각화
plt.figure(figsize=(8,10))
importances.sort_values(ascending=False).head(30).plot(kind="barh")
plt.title("Top 30 Feature Importances (avg gain)")
plt.show()


# 평균 중요도 기준으로 정렬
imp_sorted = importances.sort_values(ascending=False)

print("=== Top 20 Features ===")
print(imp_sorted.head(20))

print("\n=== Bottom 20 Features ===")
print(imp_sorted.tail(20))

# 시각화 (상위/하위 나눠서)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

imp_sorted.head(20).plot(kind="barh", ax=axes[0], title="Top 20 Features")
imp_sorted.tail(20).plot(kind="barh", ax=axes[1], title="Bottom 20 Features")

plt.tight_layout()
plt.show()


"""
import optuna

def cv_rmse(params, X, y, n_splits=3, num_boost_round=500, early_stop=50):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)
    rmses = []
    for trn_idx, val_idx in kf.split(X):
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        cols = X_trn.columns.tolist()
        dtrn = lgb.Dataset(X_trn[cols], label=y_trn, feature_name=cols)
        dval = lgb.Dataset(X_val[cols], label=y_val, feature_name=cols)

        model = lgb.train(
            params=params,
            train_set=dtrn,
            valid_sets=[dval],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop),
                lgb.log_evaluation(100),
            ],
        )
        preds = model.predict(X_val[cols], num_iteration=model.best_iteration)
        rmses.append(mean_squared_error(y_val, preds, squared=False))
    return float(np.mean(rmses))

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": 1234,
        "num_threads": -1,
        
        # 학습속도
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.20),
        
        # 트리 복잡도
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 600),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        
        # 서브샘플링
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.9),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        
        # 정규화 / bin
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "max_bin": trial.suggest_int("max_bin", 31, 127),
        
        # 랜덤 분할
        "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
    }
    return cv_rmse(params, X, y, n_splits=3, num_boost_round=500, early_stop=50)
"""


"""
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # 10번 시도 (시간에 따라 조절)

print("Best params:", study.best_params)
print("Best RMSE:", study.best_value)
"""


# 최적 파라미터 정리
best_params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "random_state": 1234,
    "num_threads": -1,
    "learning_rate": 0.10756336449282501,
    "num_leaves": 23,
    "max_depth": 10,
    "min_data_in_leaf": 523,
    "min_gain_to_split": 0.3659405249910418,
    "feature_fraction": 0.8321379434774996,
    "bagging_fraction": 0.7505080416031427,
    "bagging_freq": 2,
    "lambda_l1": 0.7522585014749754,
    "lambda_l2": 0.710970529202487,
    "max_bin": 60,
    "extra_trees": False,
}
cols = X.columns.tolist()


# 최종학습
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

kf = KFold(n_splits=3, shuffle=True, random_state=1234)
FINAL_ROUNDS, FINAL_EARLY, LOG_EVERY = 700, 60, 200

oof = np.zeros(len(X)); test_preds = np.zeros(len(X_test))
rmses = []; models = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n===== Final (solid) Fold {fold} =====")
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    dtrn = lgb.Dataset(X_trn[cols], label=y_trn, feature_name=cols)
    dval = lgb.Dataset(X_val[cols], label=y_val, feature_name=cols)

    model = lgb.train(
        params=best_params,
        train_set=dtrn,
        valid_sets=[dval],
        num_boost_round=FINAL_ROUNDS,
        callbacks=[lgb.early_stopping(FINAL_EARLY), lgb.log_evaluation(LOG_EVERY)],
    )
    models.append(model)

    preds_val = model.predict(X_val[cols], num_iteration=model.best_iteration)
    oof[val_idx] = preds_val
    rmse = mean_squared_error(y_val, preds_val, squared=False)
    rmses.append(rmse)
    print(f"[Fold {fold}] RMSE: {rmse:.5f} | best_iter={model.best_iteration}")

    test_preds += model.predict(X_test[cols], num_iteration=model.best_iteration) / kf.n_splits

print(f"\nOOF RMSE mean={np.mean(rmses):.5f} | std={np.std(rmses):.5f}")



# 제출파일 생성
submission = pd.DataFrame({
    "Id": test_df_original["Id"],                      # test_df_original["Id"] 등
    "winPlacePerc": np.clip(test_preds, 0, 1)  # 안전 클리핑
})

submission.to_csv("submission.csv", index=False)
submission.head()

