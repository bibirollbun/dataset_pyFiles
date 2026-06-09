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


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


from warnings import filterwarnings
filterwarnings("ignore")


train_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

train_data.head()


train_data.drop("id",axis=1,inplace=True)
test_data.drop("id",axis=1,inplace=True)
train_data.info()


print("\nSoil Type Values: \n",train_data["Soil Type"].value_counts())
print("\nCrop Type Values: \n",train_data["Crop Type"].value_counts())
print("\nFertilizer Name Values: \n",train_data["Fertilizer Name"].value_counts())


train_data.describe().T


categorical_features = ['Soil Type', 'Crop Type']

#Label Encoding
label_encoder = LabelEncoder()
train_data["Fertilizer Name"] = label_encoder.fit_transform(train_data["Fertilizer Name"])

X_train = train_data.drop("Fertilizer Name",axis=1)
y_train = train_data["Fertilizer Name"]


#Using OneHotEncoder with pipeline
preprocessor = ColumnTransformer(
    transformers= [
        ('cat',OneHotEncoder(drop="first"),categorical_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',XGBClassifier(eval_metric='mlogloss', random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


#Hyper-parameter tuning for XGBoost
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_distributions = {
    'model__n_estimators': [300, 500, 700],
    'model__max_depth': [3, 6, 8, 10],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.5, 0.7, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__gamma': [0, 1, 5],
    'model__tree_method':['hist'],
}

xgb_cv_model = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    cv=skf,
    verbose=2,
    n_jobs=-1,
    scoring='accuracy',
    random_state=42
)

#If you have time you can do this step but I won't
#xgb_cv_model.fit(X_train,y_train)
#print('\n - \n - \n Best parameters found: ',xgb_cv_model.best_params_)


pipeline.set_params(
    model__tree_method = "hist",
    model__subsample = 1.0,
    model__n_estimators = 700,
    model__max_depth = 10,
    model__learning_rate = 0.05,
    model__colsample_bytree = 0.4,
    model__gamma = 0
)

pipeline.fit(X_train,y_train)


#Scorer Function

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


y_pred_probs = pipeline.predict_proba(X_test)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y_test]

map3_score = mapk(actual, top_3_preds)
print(f"MAP@3 Score: {map3_score:.5f}")


test_probs = pipeline.predict_proba(test_data)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_encoder.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    'id': submission['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})


submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




