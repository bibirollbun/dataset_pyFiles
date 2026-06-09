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


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


sample = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train.head()


train['Soil Type'] = train['Soil Type'].astype('category')
train['Crop Type'] = train['Crop Type'].astype('category')

test['Soil Type'] = test['Soil Type'].astype('category')
test['Crop Type'] = test['Crop Type'].astype('category')


encode = LabelEncoder()
train['Fertilizer Name'] = encode.fit_transform(train['Fertilizer Name'])


x = train.drop(["id", "Fertilizer Name"], axis=1)
y = train['Fertilizer Name']
test = test.drop("id", axis=1)


xgb_oof = np.zeros((y.shape[0], 7))
xgb_pred = np.zeros((test.shape[0], 7))

cat_oof = np.zeros((y.shape[0], 7))
cat_pred = np.zeros((test.shape[0], 7))

light_oof = np.zeros((y.shape[0], 7))
light_pred = np.zeros((test.shape[0], 7))


cross = StratifiedKFold(n_splits=5, shuffle=True, random_state=2002)


for train_index, test_index in cross.split(x, y):
    x_train, x_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=7,
        device="cuda",
        max_depth=6,
        colsample_bynode=0.3,
        subsample=0.8,
        n_estimators=50_000,
        learning_rate=0.01,
        enable_categorical=True,
        min_child_weight=10,
        eval_metric='mlogloss', 
        early_stopping_rounds=500
    )
    
    cat = CatBoostClassifier(
        cat_features=["Soil Type", "Crop Type"],
        task_type="GPU",                
        loss_function="MultiClass",
        iterations=50_000,
        learning_rate=0.01,
        depth=6,
        random_strength=1,
        l2_leaf_reg=3,
        early_stopping_rounds=500,
        verbose=2000
    )
    
    light = lgb.LGBMClassifier(
        device='gpu',                  
        objective='multiclass',        
        num_class=7,
        n_estimators=50_000,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=31,
        colsample_bytree=0.8,
        min_child_samples=10,
    )

    xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)],verbose=2000)
    cat.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    light.fit(x_train, y_train, eval_set=[(x_test, y_test)], callbacks=[lgb.early_stopping(500, verbose=False), 
                                                                        lgb.log_evaluation(period=1000)])

    xgb_pred += xgb.predict_proba(test)
    cat_pred += cat.predict_proba(test)
    light_pred += light.predict_proba(test)

    xgb_oof[test_index] = xgb.predict_proba(x_test)
    cat_oof[test_index] = cat.predict_proba(x_test)
    light_oof[test_index] = light.predict_proba(x_test)


print(xgb_oof[0], light_oof[0], cat_oof[0], sep="\n")


meta_train = np.concatenate([xgb_oof, cat_oof, light_oof], axis=1)
meta_test = np.concatenate([xgb_pred, cat_pred, light_pred], axis=1)


meta_clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
meta_clf.fit(meta_train, y)

final_preds = meta_clf.predict_proba(meta_test)


print(final_preds[0])
print(sum(final_preds[0]))


final = []
for x in final_preds:
    y = list(enumerate(x))
    y.sort(key=lambda x: x[1], reverse=True)
    y = [i for i in y[:3]]
    final.append(encode.inverse_transform([j[0] for j in y]))


final_submission = [" ".join(x) for x in final]


final_submission[:5]


sample["Fertilizer Name"] = final_submission


sample


sample.to_csv("submission.csv", index=False)

