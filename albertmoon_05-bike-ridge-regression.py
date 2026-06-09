import pandas as pd

data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')


# ì�´ìƒ�ì¹˜ ì œê±°
train = train[train['weather'] != 4]

# ë�°ì�´í„° í•©ê¸°ê¸°
all_data_temp = pd.concat([train, test])

all_data = pd.concat([train, test], ignore_index=True)

# íŒŒìƒ� í”¼ì²˜ ì¶”ê°€í•˜ê¸°
from datetime import datetime

all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])

# datetime ì—´ì�„ datetime í˜•ì‹�ìœ¼ë¡œ ë³€í™˜ (í•„ìš”í•œ ê²½ìš°)
all_data['datetime'] = pd.to_datetime(all_data['datetime'])

# ê°�ê°�ì�˜ êµ¬ì„± ìš”ì†Œë¥¼ ìƒˆë¡œìš´ ì—´ë¡œ ì¶”ê°€
all_data['year'] = all_data['datetime'].dt.year
all_data['month'] = all_data['datetime'].dt.month
all_data['hour'] = all_data['datetime'].dt.hour

all_data['weekday'] = all_data['date'].apply(lambda x: 
                            datetime.strptime(x, "%Y-%m-%d").weekday())

# í•„ìš”ì—†ëŠ” í”¼ì²˜ ì œê±°ê¸°ê¸°
drop_features = ['casual', 'registered', 'datetime', 'date', 'month', 'windspeed']

all_data = all_data.drop(drop_features, axis=1)

# ë�°ì�´í„° ë‚˜ëˆ„ê¸°
X_train = all_data[~pd.isnull(all_data['count'])]
X_test = all_data[pd.isnull(all_data['count'])]

X_train = X_train.drop(['count'], axis=1)
X_test = X_test.drop(['count'], axis=1)

y = train['count']   # íƒ€ê¹ƒê°’ 


import numpy as np

def rmsle(y_true, y_pred, convertExp=True):
    # ì¹˜ìš°ì¹œ ë�°ì�´í„°ë¥¼ ë¡œê·¸ì �ìš©ìœ¼ë¡œ ì •ê·œë¶„í�¬ë¡œ ë³€í™˜í–ˆìœ¼ë¯€ë¡œ
    # ì§€ìˆ˜ë³€í™˜ì�„ í†µí•´ ë³µì›�í•´ì•¼ í•œë‹¤.
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))

    output = np.sqrt(np.mean((log_true - log_pred)**2))

    return output


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

ridge_model = Ridge()


ridge_params = {'max_iter':[3000], 
                'alpha':[0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300 ,400, 800, 900, 1000]}

# êµ�ì°¨ ê²€ì¦�ìš© í�‰ê°€ í•¨ìˆ˜(RMSLE ì �ìˆ˜ ê³„ì‚°)
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

# ê·¸ë¦¬ë“œ ì„œì¹˜ ê°�ì²´ ìƒ�ì„±
gridsearch_ridge_model = GridSearchCV(estimator=ridge_model,   # ë¦¿ì§€ëª¨ë�¸
                                     param_grid=ridge_params,  # ê°’ ëª©ë¡�
                                     scoring=rmsle_scorer,     # í�‰ê°€ì§€í‘œ
                                     cv=5)                     # êµ�ì°¨ ê²€ì¦� ë¶„í•  ìˆ˜


log_y = np.log(y)   # íƒ€ê¹ƒê°’ ë¡œê·¸ë³€í™˜
gridsearch_ridge_model.fit(X_train, log_y)


print('ìµœì � í•˜ì�´í�¼íŒŒë�¼ë¯¸í„° :', gridsearch_ridge_model.best_params_)


# ì˜ˆì¸¡
preds = gridsearch_ridge_model.best_estimator_.predict(X_train)

# í�‰ê°€
print(f'ë¦¿ì§€ íšŒê·€ RMSLE ê°’ : {rmsle(log_y, preds, True):.4f}')




