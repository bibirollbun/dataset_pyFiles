# 위 사항을 전부 읽었다면 이제 시작!
!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# column을 전부 볼 수 있게 500까지 늘려줌
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print(f"Test shape: {test.shape}")

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
print(f"Test shape: {train.shape}")

# 실물 영접
train.head(20)


# efs, efs_time 두 Target 변수를 하나의 Target 변수로 바꿔주기 (KaplanMeier 활용)
# lifelines 라이브러리 import
from lifelines import KaplanMeierFitter


# event 발생 여부(efs)에 따른 생존 시간(efs_time) 분포 확인
plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")

plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


# KaplanMeierFitter 이용하여 두 타겟 칼럼을 하나의 칼럼으로 만들어주기
def transform_survival_probability(df, time_col = 'efs_time', event_col = 'efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    target = kmf.survival_function_at_times(df[time_col]).values
    return target

train["y"] = transform_survival_probability(train, time_col = 'efs_time', event_col = 'efs')

plt.hist(train.loc[train.efs==1, "y"], bins = 100, label = "percent of target alive until efs_time, Yes Event")
plt.hist(train.loc[train.efs==0, "y"], bins = 100, label = "efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# 개별적인 값 분포
plt.figure(figsize=(10, 6))

ax = plt.subplot(111)

kmf_event = KaplanMeierFitter()
kmf_event.fit(train.loc[train['efs'] == 1, 'efs_time'], event_observed=train.loc[train['efs'] == 1, 'efs'], label='Event Occurred (efs=1)')
kmf_event.plot_survival_function(ax=ax)

kmf_censored = KaplanMeierFitter()
kmf_censored.fit(train.loc[train['efs'] == 0, 'efs_time'], event_observed=train.loc[train['efs'] == 0, 'efs'], label='Censored (efs=0)')
kmf_censored.plot_survival_function(ax=ax)

plt.title('Kaplan-Meier Survival Function by EFS Group')
plt.xlabel('EFS Time')
plt.ylabel('Survival Probability')
plt.show()



from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# KaplanMeierFitter 적용하여 생존 곡선 도출 (efs_time에 따른 각 개체 생존확률)
kmf = KaplanMeierFitter()
kmf.fit(train['efs_time'], event_observed=train['efs'])

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Function')
plt.xlabel('EFS Time')
plt.ylabel('Survival Probability')
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# 사건 발생 그룹(efs=1)
plt.scatter(train.loc[train['efs'] == 1, 'efs_time'], train.loc[train['efs'] == 1, 'y'], alpha=0.5, label='Event Occurred (efs=1)', color='blue')

# 사건 미발생 그룹(efs=0)
plt.scatter(train.loc[train['efs'] == 0, 'efs_time'], train.loc[train['efs'] == 0, 'y'], alpha=0.5, label='Censored (efs=0)', color='orange')

plt.xlabel('EFS Time')
plt.ylabel('Transformed Target y')
plt.title('Scatter Plot of Transformed Target y vs EFS Time by EFS Group')
plt.legend()
plt.show()



# 1. 기본적인 통계량 확인
print("=== efs 값별 efs_time 기초 통계량 ===")
print(train.groupby('efs')['efs_time'].describe())

# 2. 각 시점별 위험군(risk set) 크기와 이벤트 발생 수 확인
event_times = sorted(train['efs_time'].unique())
risk_sets = []
for t in event_times[:10]:  # 처음 10개 시점만 확인
    at_risk = sum(train['efs_time'] >= t)
    events = sum((train['efs_time'] == t) & (train['efs'] == 1))
    risk_sets.append({
        'time': t,
        'at_risk': at_risk,
        'events': events,
    })
    
print("\n=== 처음 10개 시점의 위험군 분석 ===")
for rs in risk_sets:
    print(f"Time {rs['time']}: {rs['at_risk']} at risk, {rs['events']} events")


# 예측할 대상인 y 빼주고
train_y = train['y']

# 훈련데이터로 쓸 칼럼 다듬기
train_x = train.drop(['ID', 'efs', 'efs_time', 'y'], axis = 1)

# test 데이터셋에도 ID 칼럼 제외시켜주기
test_x = test.drop(['ID'], axis = 1)

# test 데이터와 합쳐주기
total = pd.concat([train_x, test_x], axis = 0, ignore_index = True)


# 문자열 데이터가 곧 범주형 데이터인지 확인
for column in total.columns:
    if total[column].dtype == "object":
        # 고유한 값의 개수를 확인하여 범주형 데이터인지 텍스트 데이터인지 판단
        unique_values = total[column].nunique()
        
        # 고유한 값이 50개 이하인 경우 범주형 데이터로 간주 (임의 판단)
        if unique_values <= 50:
            print(f"'{column}'은 범주형 데이터입니다. 고유 값 개수: {unique_values}")
        else:
            print(f"'{column}'은 텍스트 데이터입니다. 고유 값 개수: {unique_values}")


# 전체 칼럼 루프 돌면서
for column in total.columns:
    # 범주형 변수 > 결측치 메꿔주고 > 수치형으로 바꿔주기 (XGBoost가 결측치는 메꿔주는데 수치형으로는 들어가야함)
    if total[column].dtype == "object":
        total[column] = total[column].fillna("NaN")
        total[column], uniques = total[column].factorize() # 정수형으로 변환
        total[column] -= total[column].min() # 최솟값 빼줘서 출발점을 0으로 맞춤 일종의 정규화
        total[column] = total[column].astype("int32") # int32로 변환하여 메모리 사용량 아껴주고
        total[column] = total[column].astype("category") # 범주형 데이터로 바꿔서 순서, 빈도 등의 특성 유지

    # 수치형 변수 > 데이터 형변환하여 계산비용 줄여주기
    else:
        if total[column].dtype == "float64":
            total[column] = total[column].astype("float32")
        
        if total[column].dtype == "int64":
            total[column] = total[column].astype("int32")


train_x = total.iloc[:len(train)].copy()
# train_y
test_x = total.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb


print("Using XGBoost version",xgb.__version__)
# KFold 교차 검증 사용하여 학습
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
# Out-of-Fold 예측 결과를 저장하는 배열 초기화, 훈련 데이터에 대해 예측한 값들을 나중에 저장함.
oof_xgb = np.zeros(len(train))

# Test 데이터에 대한 예측값 저장하는 배열. 모든 폴드에서 예측한 결과의 평균을 낼 것.
pred_xgb = np.zeros(len(test))

# train 데이터셋에 대하여 KFold로 나누고, 각 폴드마다 train_index와 test_index를 생성 (train이랑 valid임.)
for i, (train_index, valid_index) in enumerate(kf.split(train_x)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train_x.loc[train_index].copy()
    y_train = train_y.loc[train_index].copy()
    x_valid = train_x.loc[valid_index].copy()
    y_valid = train_y.loc[valid_index].copy()
    x_test = test_x.copy()

    model_xgb = XGBRegressor(
        device="cpu",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        #early_stopping_rounds=25,
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb[valid_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.
 
It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.
 
The concordance index is a value between 0 and 1 where:
 
0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],  # efs_time: 사건 발생까지의 시간
                        -merged_df_race[prediction_label],  # prediction: 생존 확률(음수 처리, 높을수록 낮은 위험을 의미하기에)
                        merged_df_race[event_label])  # 사건 발생 여부
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb

# 모델이 예측한 생존 확률과 실제 생존 시간을 비교하여 순서 일관성을 평가
# 높은 C-index score는 예측된 생존 확률이 실제 생존 시간과 일치하는 정도를 의미함.
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# model_xgb.feature_importances_를 사용해 모델이 학습 과정에서 
# 각 피처를 얼마나 많이 사용했는지, 혹은 얼마나 기여했는지를 수치로 반환
feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": total.columns,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


submission = pd.DataFrame({
    'ID': test['ID'],
    'prediction': pred_xgb
})


submission.to_csv('submission.csv', index = False)




