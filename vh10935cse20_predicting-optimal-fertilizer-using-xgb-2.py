import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.shape


test.head(2)


train.head(2)


test.shape


train.isna().sum()


test.isna().sum()


train.dtypes


test.dtypes


from sklearn.preprocessing import OrdinalEncoder,LabelEncoder


cat_col = train.select_dtypes(include=['object']).columns
cat_col = cat_col[cat_col != 'Fertilizer Name']
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train[cat_col] = ord_enc.fit_transform(train[cat_col].astype(str))
test[cat_col] = ord_enc.transform(test[cat_col].astype(str))


train.head(4)


le=LabelEncoder()
train['Fertilizer Name']=le.fit_transform(train['Fertilizer Name'])


x=train.drop(['Fertilizer Name'],axis=1)
y=train['Fertilizer Name']


from sklearn.model_selection import train_test_split


train_X, test_X, train_y, test_y = train_test_split(x, y,test_size = 0.2, random_state =42,stratify=y)


from xgboost import XGBClassifier


model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(train_y)),
    n_estimators=3200,
    learning_rate=0.045,         
    max_depth=7,                
    colsample_bytree=0.6,       
    colsample_bylevel=0.8,      
    subsample=0.8,
)


model.fit(train_X, train_y)


y_pred_probs = model.predict_proba(test_X)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in test_y]


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
map3_score = mapk(actual, top_3_preds)
print(f"MAP@3 Score: {map3_score:.5f}")


test_probs = model.predict_proba(test)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('/kaggle/working/submission.csv',index=False)
print("✅ Submission file saved as 'submission.csv'")

