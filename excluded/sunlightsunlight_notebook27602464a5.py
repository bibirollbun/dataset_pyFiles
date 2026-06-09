import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler , OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import time
import threading

def print_every_10_seconds():
    count = 0
    while True:
        count += 1
        print(f"定时 #{count}")
        time.sleep(10)

# 在后台线程中运行
timer_thread = threading.Thread(target=print_every_10_seconds, daemon=True)
timer_thread.start()

# 主程序可以继续做其他事情


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train


test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test


test['road_type'].value_counts()


test[['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']]=OrdinalEncoder().fit_transform(test[['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']])
test


train[['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']]=OrdinalEncoder().fit_transform(train[['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']])
train


x=train.drop(['id','accident_risk'], axis=1)
x


y=train['accident_risk']
y


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

model = XGBRegressor(random_state=42)

# 调整参数网格（使用新的GPU参数设置）
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [100, 200],
    'tree_method': ['hist'],      # 使用hist方法
    'device': ['cuda']            # 使用CUDA设备
}

grid_search = GridSearchCV(
    model, param_grid, cv=3, scoring='r2', 
    n_jobs=1,  # GPU训练时建议设置n_jobs=1
)

grid_search.fit(x, y)

print("最佳参数:", grid_search.best_params_)
print("最佳分数: {:.4f}".format(grid_search.best_score_))

best_model = grid_search.best_estimator_


an=test
test=test.drop('id',axis=1)
test


predictions = best_model.predict(test)
predictions


hh=pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
hh


output = pd.DataFrame({'id': an.id, 'accident_risk': predictions})
output.to_csv('submission.csv', index=False)
output

