


import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def load_params(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

params_lgb = load_params('/kaggle/input/introverts-lgbm-for-oof/Best_trial.yaml')
params_cat = load_params('/kaggle/input/introverts-catboost-for-oof/Best_trial.yaml')
params_xgb = load_params('/kaggle/input/introverts-xgboost-for-oof/Best_trial.yaml')

base_models = [
    LGBMClassifier(**params_lgb),
    CatBoostClassifier(**params_cat),
    XGBClassifier(**params_xgb)# use_label_encoder=False)
]

meta_model = LogisticRegression()


data0=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
print(data0.columns.tolist())
print(len(data0))

TEST0=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

class_names=sorted(data0['Personality'].unique().tolist())
print(class_names)

N=list(range(len(class_names)))
normal_mapping=dict(zip(class_names,N)) 
reverse_mapping=dict(zip(N,class_names))      


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df
    
data=labelencoder(data0)
TEST=labelencoder(TEST0)


X = data.drop('Personality',axis=1)
y = data['Personality']
X_test = TEST


n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros((X.shape[0], len(base_models)))
test_preds = np.zeros((X_test.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    oof = np.zeros(X.shape[0])
    test_fold_preds = np.zeros((n_folds, X_test.shape[0]))

    for j, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_fold_preds[j, :] = model.predict_proba(X_test)[:, 1]

    oof_preds[:, i] = oof
    test_preds[:, i] = test_fold_preds.mean(axis=0)


meta_model.fit(oof_preds, y)
weights=(meta_model.coef_/meta_model.coef_.sum())[0]
print(weights)

final_preds = meta_model.predict_proba(test_preds)[:, 1]
final_labels = (final_preds > 0.5).astype(int)

final_preds2 = (test_preds)[:, 1:].argmax(axis=1)
final_labels2 = (final_preds2 > 0.5).astype(int)


#meta_model
submit=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
print(len(submit))
submit['Personality']=final_labels
submit['Personality']=submit['Personality'].map(reverse_mapping)
display(submit)
submit.to_csv('submission.csv',index=False)


#argmax
submit['Personality']=final_labels2
submit['Personality']=submit['Personality'].map(reverse_mapping)
display(submit)
#submit.to_csv('submission_argmax.csv',index=False)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm1 = confusion_matrix(final_labels2, final_labels, labels=[0, 1])

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,
                               display_labels=["Extrovert", "Introvert"])
disp1.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix: Meta Model vs Argmax")
plt.ylabel("Argmax Prediction")
plt.xlabel("Meta Model Prediction")
plt.show()

