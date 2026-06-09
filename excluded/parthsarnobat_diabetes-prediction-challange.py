import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e12'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

    
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv").set_index("id")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv").set_index("id")


x = train.drop(columns=['diagnosed_diabetes'])
y = train['diagnosed_diabetes']
x_test = test.copy()

print(f"train shape:{x.shape}, test shape:{x_test.shape}")


cat_features = [
    'gender', 'ethnicity', 'education_level', 'income_level', 
    'smoking_status', 'employment_status'
]

for col in cat_features:
    x[col] = x[col].astype(str).fillna('None')
    x_test[col] = x_test[col].astype(str).fillna('None')


oof_preds = np.zeros(len(x))
test_preds = np.zeros(len(x_test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(x,y)):
    print(f"Training fold {fold + 1}")

    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations = 3000,
        learning_rate = 0.03,
        depth = 6,
        task_type='GPU',      
        devices='0',
        eval_metric = 'AUC',
        early_stopping_rounds=200,
        random_seed=42,
        verbose=500
    )

    model.fit(
        x_train, y_train,
        eval_set=(x_val, y_val),
        cat_features=cat_features,
        use_best_model=True
    )

    oof_preds[val_idx] = model.predict_proba(x_val)[:, 1]
    test_preds += model.predict_proba(x_test)[:, 1] / 5

final_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall CV AUC Score: {final_auc:.5f}")


feature_importance = model.get_feature_importance()
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [x.columns[i] for i in sorted_idx])
plt.title('CatBoost Feature Importance')
plt.show()



submission = pd.DataFrame({
    'id': test.index,
    'diagnosed_diabetes': test_preds
})

submission.to_csv('submission.csv', index=False)
print("Submission file successfully created!")




