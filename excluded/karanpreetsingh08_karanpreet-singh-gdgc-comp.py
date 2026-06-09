import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

df="/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26"
train=pd.read_csv(os.path.join(df,"train.csv"))
test=pd.read_csv(os.path.join(df,"test.csv"))
l_p=pd.read_csv(os.path.join(df,"feature_lookup.csv"))
s_s=pd.read_csv(os.path.join(df,"sample_submission.csv"))
print(train.shape)
print(test.shape)
train.head()


print(train.info())
missing_val=train.isnull().sum().sort_values(ascending=False)
missing_val.head(7)


df="/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26"
train=pd.read_csv(os.path.join(df,"train.csv"))
test=pd.read_csv(os.path.join(df,"test.csv"))
f_l=pd.read_csv(os.path.join(df,"feature_lookup.csv"))
s_s = pd.read_csv(os.path.join(df,"sample_submission.csv"))

print("train shape:",train.shape)
print("test shape:",test.shape)
print("feature lookup shape:",f_l.shape)
train.head()


plt.figure(figsize=(7,4))
sns.histplot(train["relationship_probability"],bins=64,kde=True)
plt.title("distribution:")
plt.show()


f_l.head()
num_feat=f_l[f_l["type"]=="numeric"]["feature_code"].tolist()
cat_feat=f_l[f_l["type"]=="categorical"]["feature_code"].tolist()
print("numeric:",num_feat)
print("categorical:",cat_feat)



train["missing_count_val"]=train[num_feat + cat_feat].isnull().sum(axis=1)
test["missing_count_val"]=test[num_feat + cat_feat].isnull().sum(axis=1)
train["social_points"]=train[["F14","F17","F18"]].mean(axis=1)
test["social_points"]=test[["F14","F17","F18"]].mean(axis=1)




for colm in num_feat:
    med=train[colm].median()
    train[colm]=train[colm].fillna(med)
    test[colm] =test[colm].fillna(med)
for colm in cat_feat:
    mode=train[colm].mode()[0]
    train[colm]=train[colm].fillna(mode)
    test[colm]=test[colm].fillna(mode)


from sklearn.preprocessing import LabelEncoder
encoders={}
for colm in cat_feat:
    le=LabelEncoder()
    train[colm]=le.fit_transform(train[colm].astype(str))
    mapping={cls: idx for idx, cls in enumerate(le.classes_)}
    test[colm]=test[colm].astype(str).map(mapping).fillna(-1).astype(int)
    encoders[colm]=le


from sklearn.model_selection import train_test_split
features=num_feat+cat_feat+["missing_count_val","social_points"]
x=train[features]
y=train["relationship_probability"]
x_test=test[features]
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape,x_val.shape



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
rf=RandomForestRegressor(n_estimators=400,random_state=42,n_jobs=-1)
rf.fit(x_train,y_train)
pred_val=rf.predict(x_val)
rmse=np.sqrt(mean_squared_error(y_val, pred_val))
mae= mean_absolute_error(y_val, pred_val)
r2=r2_score(y_val, pred_val)
rmse, mae, r2



import lightgbm as lgb
from sklearn.model_selection import KFold
kf=KFold(n_splits=5,shuffle=True,random_state=42)
oof=np.zeros(len(x))
test_preds=np.zeros(len(x_test))
params={
    "objective":"regression",
    "metric":"rmse",
    "learning_rate":0.05,
    "num_leaves":31,
    "feature_fraction":0.85,
    "bagging_fraction":0.8,
    "bagging_freq":1,
    "seed":42
}
for fold,(trn_idx,val_idx) in enumerate(kf.split(x,y)):
    print(f"Fold {fold+1}")
    x_tr,x_v=x.iloc[trn_idx],x.iloc[val_idx]
    y_tr,y_v=y.iloc[trn_idx],y.iloc[val_idx]
    train_data=lgb.Dataset(x_tr,label=y_tr)
    valid_data=lgb.Dataset(x_v,label=y_v)
    model=lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data,valid_data],
        valid_names=["train","valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )
    oof[val_idx]=model.predict(x_v,num_iteration=model.best_iteration)
    test_preds+=model.predict(x_test,num_iteration=model.best_iteration)/5




submission=pd.DataFrame({
    "ID":test["ID"],
    "relationship_probability":np.clip(test_preds,0,100)
})

submission.to_csv("submission.csv", index=False)

submission.head(),submission.shape,submission.columns





