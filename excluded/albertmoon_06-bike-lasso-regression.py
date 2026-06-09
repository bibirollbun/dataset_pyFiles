import pandas as pd

data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')


# 이상치 제거
train = train[train['weather'] != 4]

# 데이터 합기기
all_data_temp = pd.concat([train, test])

all_data = pd.concat([train, test], ignore_index=True)

# 파생 피처 추가하기
from datetime import datetime

all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])

# datetime 열을 datetime 형식으로 변환 (필요한 경우)
all_data['datetime'] = pd.to_datetime(all_data['datetime'])

# 각각의 구성 요소를 새로운 열로 추가
all_data['year'] = all_data['datetime'].dt.year
all_data['month'] = all_data['datetime'].dt.month
all_data['hour'] = all_data['datetime'].dt.hour

all_data['weekday'] = all_data['date'].apply(lambda x: 
                            datetime.strptime(x, "%Y-%m-%d").weekday())

# 필요없는 피처 제거기기
drop_features = ['casual', 'registered', 'datetime', 'date', 'month', 'windspeed']

all_data = all_data.drop(drop_features, axis=1)

# 데이터 나누기
X_train = all_data[~pd.isnull(all_data['count'])]
X_test = all_data[pd.isnull(all_data['count'])]

X_train = X_train.drop(['count'], axis=1)
X_test = X_test.drop(['count'], axis=1)

y = train['count']   # 타깃값 


import numpy as np

def rmsle(y_true, y_pred, convertExp=True):
    # 치우친 데이터를 로그적용으로 정규분포로 변환했으므로
    # 지수변환을 통해 복원해야 한다.
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))

    output = np.sqrt(np.mean((log_true - log_pred)**2))

    return output


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

lasso_model = Lasso()

lasso_alpha = 1/np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])
lasso_params = {'max_iter':[3000], 'alpha':lasso_alpha}

# 교차 검증용 평가 함수(RMSLE 점수 계산)
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

# 그리드 서치 객체 생성
gridsearch_lasso_model = GridSearchCV(estimator=lasso_model,
                                     param_grid=lasso_params,
                                     scoring=rmsle_scorer,
                                     cv=5)
# 그리드 서치 수행
log_y = np.log(y)
gridsearch_lasso_model.fit(X_train, log_y)

print('최적 하이퍼파라미터 :', gridsearch_lasso_model.best_params_)



preds = gridsearch_lasso_model.best_estimator_.predict(X_train)

print(f'라쏘 회귀 RMSLE 값 : {rmsle(log_y, preds, True):.4f}')




