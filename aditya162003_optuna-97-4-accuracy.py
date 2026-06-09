# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train_df.head(5)


train_df.shape


x=train_df.drop('Personality',axis=1)
y=train_df['Personality']
y.head(4)
# x.head(4)


plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Personality Count')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.show()
y.value_counts()


train_df.isnull().sum()


binary_cols = ['Stage_fear', 'Drained_after_socializing']
x[binary_cols] = x[binary_cols].replace({'Yes': 1, 'No': 0})


x.head(5)


# Properly select multiple columns
x_id = x[['id', 'Stage_fear', 'Drained_after_socializing']]

# Drop those columns from x
x = x.drop(columns=['id'])

# Show the first 5 rows
x.head(5)


from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer  # Must be called first
from sklearn.impute import IterativeImputer
import pandas as pd

# Set fixed random state in both estimator and imputer
tree_regressor = DecisionTreeRegressor(random_state=42)
imp = IterativeImputer(estimator=tree_regressor, max_iter=200, random_state=42)

# Perform imputation
x_imputed = pd.DataFrame(imp.fit_transform(x), columns=x.columns)


x_imputed.isnull().sum()


x_imputed['id'] = x_id['id']


numeric_df = x_imputed.select_dtypes(include='number')
numeric_df.corr()


plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',fmt='.2f', vmin=-1, vmax=1)
plt.title('Heatmap')
plt.show()



y = y.replace({'Extrovert': 1, 'Introvert': 0})


y.head(5)


from sklearn.ensemble import RandomForestRegressor

X = numeric_df


model = RandomForestRegressor()
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False)


from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["LightGBM", "XGBoost", "CatBoost"])

    if classifier_name == "LightGBM":
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "class_weight": "balanced"  # ✅ LightGBM supports this
        }
        model = LGBMClassifier(**params)

    elif classifier_name == "XGBoost":
        # Manually compute scale_pos_weight
        scale_pos_weight = len(y[y == 0]) / len(y[y == 1])
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "scale_pos_weight": scale_pos_weight  # ✅ XGBoost's way to balance classes
        }
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)

    else:  # CatBoost
        # Compute class weights for CatBoost
        class_weights = [len(y) / (2 * sum(y == i)) for i in range(2)]
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 50, 300),
            "verbose": 0,
            "class_weights": class_weights  # ✅ CatBoost format
        }
        model = CatBoostClassifier(**params)

    # 5-fold cross-validated accuracy
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score





import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


best_trial = study.best_trial
print("Best trial parameters:", best_trial.params)
print("Best trial accuracy:", best_trial.value)


study.trials_dataframe()


study.trials_dataframe()['params_classifier'].value_counts()


study.trials_dataframe().groupby('params_classifier')['value'].mean()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


test_df.shape


test_df.head()


test_df.isnull().sum()


binary_cols = ['Stage_fear', 'Drained_after_socializing']
test_df[binary_cols] = test_df[binary_cols].replace({'Yes': 1, 'No': 0})


test_df.head()


from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer  # Must be called first
from sklearn.impute import IterativeImputer
import pandas as pd

# Set fixed random state in both estimator and imputer
tree_regressor = DecisionTreeRegressor(random_state=42)
imp = IterativeImputer(estimator=tree_regressor, max_iter=200, random_state=42)

# Perform imputation
x_imputed = pd.DataFrame(imp.fit_transform(test_df), columns=test_df.columns)


x_imputed.isnull().sum()



from catboost import CatBoostClassifier

# 1. Remove 'classifier' key
best_params = {
    'depth': 10,
    'learning_rate': 0.02954633095888415,
    'iterations': 98
}

# 2. Instantiate CatBoost
model = CatBoostClassifier(verbose=0, **best_params)

# 3. Train on full training data
model.fit(X, y)

# 4. Predict on test data
y_pred = model.predict(x_imputed)


y_pred


submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_pred
})
print(submission.head(15))
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)
print("Submitted successfully with XGBoost")

