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


train_df= pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df= pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train_df.describe()


def data_conversion(df):  
    df= df.fillna(df[["Weight Capacity (kg)"]].mean())
    
    mode_values= df[["Brand","Material","Color"]].mode().iloc[0]
    df= df.fillna(mode_values)

    df= df.drop(["Size","id","Compartments","Style","Waterproof","Laptop Compartment"],axis=1)
    
    return df


from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error


d= data_conversion(train_df)
t= data_conversion(test_df)


categorical_features= ["Brand","Material","Color"]

X= d.drop('Price',axis=1)
y= d['Price']


model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, loss_function='RMSE', cat_features=categorical_features, verbose=200)

# 교차검증을 이용한 RMSE 평가
# cross_val_score는 기본적으로 R2 점수를 반환하므로, 'neg_mean_squared_error'를 사용하고 결과값의 부호를 반전시켜야 함
rmse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# RMSE 값을 계산 (부호를 반전시킨 후 제곱근을 구함)
rmse_scores = np.sqrt(-rmse_scores)

# 평균 RMSE 출력
print(f'교차검증을 통한 평균 RMSE: {rmse_scores.mean()}')


# 최종 오차값 확인: 학습 데이터와 테스트 데이터 분리 후 예측
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 최종 RMSE 계산
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'최종 RMSE: {final_rmse}')


d= data_conversion(test_df)

predictions= model.predict(d)


result_df = pd.DataFrame({'id': test_df["id"], 'Price': predictions})
result_df.to_csv('submission.csv', index=False)

