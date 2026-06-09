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


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

class AccidentRiskPredictor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model = None
        self.encoders = {}

    def load_data(self):
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

    def preprocess(self):
        # 特徴量の新規生成
        self.train['speed_curvature_ratio'] = self.train['speed_limit'] / (self.train['curvature'] + 1)
        self.test['speed_curvature_ratio'] = self.test['speed_limit'] / (self.test['curvature'] + 1)

        self.train['lane_speed_ratio'] = self.train['num_lanes'] / self.train['speed_limit']
        self.test['lane_speed_ratio'] = self.test['num_lanes'] / self.test['speed_limit']

        self.train['is_rush_hour'] = self.train['time_of_day'].isin(['morning', 'evening']).astype(int)
        self.test['is_rush_hour'] = self.test['time_of_day'].isin(['morning', 'evening']).astype(int)

        self.train['is_holiday_school_overlap'] = (
            self.train['holiday'].astype(int) & self.train['school_season'].astype(int)
        )
        self.test['is_holiday_school_overlap'] = (
            self.test['holiday'].astype(int) & self.test['school_season'].astype(int)
        )

        # ラベルエンコーディング
        categorical_cols = self.train.select_dtypes(include=['object', 'bool']).columns.tolist()
    
        for col in categorical_cols:
            le = LabelEncoder()
            self.train[col] = le.fit_transform(self.train[col])
            self.test[col] = le.transform(self.test[col])
            self.encoders[col] = le

        # 学習データとターゲットに分割
        self.X = self.train.drop(columns=['id', 'accident_risk'])
        self.y = self.train['accident_risk']
        self.X_test = self.test.drop(columns=['id'])

    def train_model(self):
        param_grid = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [15, 31, 63, 127],
        }
        grid = GridSearchCV(
            estimator=LGBMRegressor(random_state=42),
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',  # RMSEを最小化
            cv=3,  # 3分割クロスバリデーション
            verbose=1,
            n_jobs=-1  # 並列処理
        )

        grid.fit(self.X, self.y)
        self.model = grid.best_estimator_  # 最も良かったモデルを保存

        print("Best parameters found:")
        print(grid.best_params_)


    def evaluate(self):
        preds = self.model.predict(self.X)
        rmse = mean_squared_error(self.y, preds, squared=False)
        print(f'RMSE on training data: {rmse:.4f}')

    def predict_and_save(self, output_path='submission.csv'):
        preds = self.model.predict(self.X_test)
        submission = pd.DataFrame({
            'id': self.test['id'],
            'accident_risk': preds
        })
        submission.to_csv(output_path, index=False)
        print(f'Submission saved to {output_path}')



predictor = AccidentRiskPredictor("/kaggle/input/playground-series-s5e10/train.csv", 
                                  '/kaggle/input/playground-series-s5e10/test.csv')
predictor.load_data()
predictor.preprocess()
predictor.train_model()
predictor.evaluate()
predictor.predict_and_save()





