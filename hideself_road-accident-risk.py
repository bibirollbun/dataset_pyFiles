# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
train_data.head()


test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_data.head()


y = train_data['accident_risk']

features = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
            'weather', 'road_signs_present', 'public_road', 'time_of_day',
            'holiday', 'school_season', 'num_reported_accidents']

# 数据独热编码
X = pd.get_dummies(train_data[features])
X_ans_test = pd.get_dummies(test_data[features])


# 查看缺失值
print(X.isnull().any())
print(X_ans_test.isnull().any())

# 查看独热编码后列数是否相同
print(X.columns == X_ans_test.columns)


import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV

# 网格搜索最优参数过程省略，最优参数在下面
# # 网格搜索寻找调优参数
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# base_model=xgb.XGBRegressor(
#     objective='reg:squarederror',
#     tree_method='hist',
#     device='cuda',
#     random_state=42
# )

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 5, 10, 15],
#     'learning_rate': [0.1, 0.05, 0.01],
# }

# # 创建GridSearch对象,寻找XGBoost模型的调优参数
# grid_search=GridSearchCV(
#     estimator=base_model,
#     param_grid=param_grid,
#     cv=5,  # 5折交叉验证
#     scoring='neg_mean_squared_error',
#     n_jobs=1
# )

# grid_search.fit(X_train,y_train)

# # 输出最佳参数和最佳分数
# print("最佳参数: ", grid_search.best_params_)
# print("最佳交叉验证分数: ", grid_search.best_score_)

# 最佳参数:  {'learning_rate': 0.1, 'max_depth': None, 'n_estimators': 200}
# 最佳交叉验证分数:  -0.0031396743096618594


xgb_model=xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    device='cuda',
    learning_rate=0.1,
    max_depth=None,
    n_estimators=200,
    random_state=42
)

xgb_model.fit(X,y)
predictions=xgb_model.predict(X_ans_test)

output=pd.DataFrame({'id':test_data.id,'accident_risk':predictions})
output.to_csv('/kaggle/working/submission.csv', index=False)
print('Your submission was successfully saved!')

