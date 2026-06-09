import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from lightgbm import LGBMClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv",index_col = 'id')


train.head()


cat_columns = [i for i in train.columns if train[i].dtype == np.object_]
num_columns = [i for i in train.columns if i not in cat_columns]


label_enc = LabelEncoder()
for i in cat_columns[:-1]:
    train[i] = label_enc.fit_transform(train[i])
    test[i] = label_enc.transform(test[i])
train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])


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


X = train.drop('Fertilizer Name',axis = 1)
y = train["Fertilizer Name"]


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape = (len(train) ,y.nunique()))
pred_prob = np.zeros(shape = (len(test),y.nunique()))

lgb_model = LGBMClassifier(
     n_estimators= 1214,
     learning_rate= 0.06408094783107429,
     num_leaves= 169,
     max_depth =10,
     min_child_samples= 19,
     subsample= 0.6420340301820501,
     colsample_bytree= 0.43403799235854973,
     reg_alpha=6.294093849568123,
     reg_lambda= 5.5559072866866455,
     random_state=42,
     verbosity =-1
)

for i, (train_idx, valid_idx) in enumerate(skf.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    x_train, x_valid = X.iloc[train_idx],X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    lgb_model.fit(x_train,y_train, eval_set=[(x_valid,y_valid)])
    oof[valid_idx] = lgb_model.predict_proba(x_valid)
    pred_prob +=lgb_model.predict_proba(test)

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]  
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"âœ… FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")


top_3_preds = np.argsort(oof, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y]
map3_score = mapk(actual, top_3_preds)
print(f'âœ… Final MAP@3 Score: {map3_score:.5f} ')


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


submission.head()

