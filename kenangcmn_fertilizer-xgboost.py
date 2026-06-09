# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train_df.head()


train_df.info()


for col in train_df.select_dtypes(include='float64').columns:
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes(include='int64').columns:
    train_df[col] = train_df[col].astype('int32')


print("Missing Values in train data:")
print(train_df.isnull().sum())

print("\nMissing Values in test data:")
print(test_df.isnull().sum())


rmv = ["Fertilizer Name"]
features = [c for c in train_df.columns if c not in rmv]
cats = [c for c in features if train_df[c].dtype == "object"]

print(f"Features: {len(features)} (Categorical: {len(cats)})")


print(cats)


from sklearn.preprocessing import LabelEncoder

label_encoders = {col: LabelEncoder() for col in cats}

for col in cats:
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])

    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

le = LabelEncoder()
train_df['Fertilizer Name'] = le.fit_transform(train_df['Fertilizer Name'])


train_df.head()


def mapk(actual, predicted, k=3):
    def apk(actual_labels, predicted_labels):
        predicted_labels = predicted_labels[:k]
        score = 0.0
        correct = 0
        used = set()
        
        for i, label in enumerate(predicted_labels):
            if label in actual_labels and label not in used:
                correct += 1
                score += correct / (i + 1)
                used.add(label)
        
        return score / min(len(actual_labels), k)

    return np.mean([apk(a, p) for a, p in zip(actual, predicted)])


from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


FOLDS = 5

num_classes = len(np.unique(train_df[rmv]))
oof_xgb = np.zeros((len(train_df), num_classes))
pred_xgb = np.zeros((len(test_df), num_classes))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):

    print(f"### FOLD {fold+1} ###")

    X_train = train_df.iloc[train_idx][features]
    y_train = train_df.iloc[train_idx][rmv]
    X_test = train_df.iloc[val_idx][features]
    y_test =  train_df.iloc[val_idx][rmv]

    model = XGBClassifier(
        max_depth=12,
        colsample_bytree=0.467,
        subsample=0.86,
        n_estimators=1700,
        learning_rate=0.03,
        gamma=0.26,
        max_delta_step=4,
        reg_alpha=2.7,
        reg_lambda=1.4,
        early_stopping_rounds=100,
        objective='multi:softprob',
        random_state=42,
        enable_categorical=True,
        device='cuda'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=0
    )

    probs = model.predict_proba(X_test)
    oof_xgb[val_idx] = probs
    pred_xgb += model.predict_proba(test_df[features]) / FOLDS

    predicted_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    actual = [[int(label)] for label in y_test.values]
    map3_score = mapk(actual, predicted_3, k=3)

    print(f"MAP@3 Score: {map3_score:.4f}\n")


test_top_3_preds = np.argsort(pred_xgb, axis=1)[:, -3:][:, ::-1]
final_preds = le.inverse_transform(test_top_3_preds.ravel()).reshape(test_top_3_preds.shape)


submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv', index_col=0)
submission['Fertilizer Name'] = [' '.join(map(str, each)) for each in final_preds]
submission.reset_index(inplace=True)
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")


submission.head()

