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


df = pd.read_csv("/kaggle/input/chydv-hackathon-2025/train.csv")
test_df = pd.read_csv("/kaggle/input/chydv-hackathon-2025/test.csv")
df.head()


df.info()


test_df.info()


Id = test_df['id']
df = df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


import matplotlib.pyplot as mp
import seaborn as sns

corr_matrix = df.corr()
mp.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, 
            annot=True,  
            cmap='coolwarm')
mp.show()


df['bound_so2'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
test_df['bound_so2'] = test_df['total sulfur dioxide'] - test_df['free sulfur dioxide']
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
test_df['total_acidity'] = test_df['fixed acidity'] + test_df['volatile acidity']


y = df['quality']
df_train = df.drop(columns=['quality'])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import cohen_kappa_score, make_scorer
optuna.logging.set_verbosity(optuna.logging.WARNING)


class PipelineClassifier:
    def __init__(self, trials=50):
        self.classifier = LGBMClassifier(
            verbose=-1,
            random_state=42
        )
        self.scaler = MinMaxScaler()
        self.n_classes = 11
        self.n_trials = trials
        
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        def _objective(trial):
            param = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_class': self.n_classes,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),  
                'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),  
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 200), 
                'max_depth': trial.suggest_int('max_depth', 5, 12),  
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),  
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),  
            }
            kappa_scorer = make_scorer(cohen_kappa_score, weights='quadratic')
            self.classifier = LGBMClassifier(
                **param,
                verbose=-1,
                random_state=42
            )
            cv_scores = cross_val_score(
                self.classifier, 
                X_scaled, 
                y, 
                scoring=kappa_scorer, 
                cv=5  
            )
            
            return cv_scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(_objective, 
                       n_trials=self.n_trials,
                       show_progress_bar=True)
                    
        best_params = study.best_params
        self.classifier = LGBMClassifier(
            **best_params,
            random_state=42,
            verbose=-1
        )
        
        return best_params
    
    def predict(self, X_test, X_full_train, y):
        self.classifier.fit(self.scaler.transform(X_full_train), y)
        X_scaled = self.scaler.transform(X_test)
        return self.classifier.predict(X_scaled)


clf = PipelineClassifier(trials=100)
clf.train(df_train, y)


result_y = clf.predict(test_df, df_train, y)
result_y


submission = pd.DataFrame({
    'id': Id,
    "quality":result_y
})
submission.to_csv("submission.csv", index=False)

