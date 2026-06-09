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


train=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head()


train.describe()


test.describe()


display(train[train.columns.tolist()].describe(include = ['object']))


display(test[test.columns.tolist()].describe(include = ['object']))


import seaborn as sns
import matplotlib.pyplot as plt



plt.figure(figsize=(15, 5))
sns.heatmap(train.isnull(), cbar=False)
plt.title('SÃ¼tunlarÄ±n NaN olma durumu')
plt.show()


train.isnull().sum()


test.isnull().sum()


train.info()


train['Drained_after_socializing']=train['Drained_after_socializing'].map({'No':0,'Yes':1})
train['Stage_fear']=train['Stage_fear'].map({'No':0,'Yes':1})
train['Personality']=train['Personality'].map({'Introvert':0,'Extrovert':1})
test['Drained_after_socializing']=test['Drained_after_socializing'].map({'No':0,'Yes':1})
test['Stage_fear']=test['Stage_fear'].map({'No':0,'Yes':1})



train


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
train = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)



train


test = pd.DataFrame(imputer.fit_transform(test), columns=test.columns)



test.isnull().sum()


train.isnull().sum()


train['Time_spent_Alone']=np.log1p(train['Time_spent_Alone'])
test['Time_spent_Alone']=np.log1p(test['Time_spent_Alone'])


train.info()


test.info()


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



train = train.set_index('id')
test = test.set_index('id')


X = train.drop("Personality", axis=1)
y=train['Personality']


models = {
    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        reg_alpha=0.1,    # L1 cezasÄ±
        reg_lambda=1.0,   # L2 cezasÄ±
        eval_metric="logloss",
        use_label_encoder=False
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        reg_alpha=0.1,    # L1 cezasÄ±
        reg_lambda=1.0    # L2 cezasÄ±
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,    # CatBoostâ€™ta L2 regularizasyon
        verbose=0
    )
}



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import optuna


for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(name, "ROC AUC:", np.mean(scores))



# Optuna objective function
def objective(trial):
    params = {
        "iterations": 1000,
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "eval_metric": "AUC",
        "verbose": 0,
        "loss_function": "Logloss",
    }

    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X, y, cv=3, scoring="roc_auc")
    return np.mean(scores)

# Optuna Ã§alÄ±ÅŸtÄ±r
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best Params:", study.best_params)




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ðŸ”¹ En iyi model ile eÄŸitim
best_model = CatBoostClassifier(**study.best_params, iterations=1000, eval_metric="AUC", verbose=0)
best_model.fit(X_train,y_train)

# ðŸ”¹ Tahmin olasÄ±lÄ±klarÄ±
y_proba = best_model.predict_proba(X_test)[:,1]

# ðŸ”¹ Threshold optimizasyonu (F1 iÃ§in)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]

# ðŸ”¹ En iyi threshold ile sÄ±nÄ±flandÄ±rma
y_pred_opt = (y_proba >= best_threshold).astype(int)

print("\nðŸ“Š SonuÃ§lar")
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("Best Threshold:", round(best_threshold, 3))
print("F1 Score:", round(f1_score(y_test, y_pred_opt), 4))


y_final_proba = best_model.predict_proba(test)[:,1]
y_final_pred = (y_final_proba >= best_threshold).astype(int)


submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


submission['id']=test.index
submission['Personality']=y_final_pred


submission['Personality']=submission['Personality'].map({1:'Extrovert',0:'Introvert'})


importances = best_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()



submission.to_csv('submission.csv', index=False)

