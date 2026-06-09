!pip install pytorch-tabnet


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


print(train.shape)
train.head()


print(test.shape)
test.head()


X = train.drop(['id','y'],axis=1)
Y = train['y']

#X_test = test.drop(['id'],axis=1)


for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        


scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


X.shape


X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size = 0.2)


from pytorch_tabnet.tab_model import TabNetClassifier
import torch


clf = TabNetClassifier(
    n_d=64, n_a=64, 
    n_steps=5,
    gamma=1.5,
    n_independent=2, n_shared=2,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-3),
    mask_type="sparsemax", 
    verbose=10
)


clf.fit(
    X_train, Y_train,
    eval_set=[(X_val, Y_val)],
    eval_name=["val"],
    eval_metric=["auc"],
    max_epochs=50,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=1,
    drop_last=False
)


from sklearn.metrics import roc_auc_score


y_val_pred = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(Y_val, y_val_pred)
print("Validation AUC:", auc)



y_test_pred = clf.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test['id'],
    'y': y_test_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file saved!")

















