!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
print("Test shape:", test.shape )

train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
#train = train.head(3)
print("Train shape:",train.shape)
train.head()


train.head()


plt.hist(train.loc[train.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
train["y"] = transform_survival_probability(train, time_col='efs_time', event_col='efs')

plt.hist(train.loc[train.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(train.loc[train.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time","y"]
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")


CONS = []
for c in FEATURES:
    if train[c].dtype!="object":
        CONS.append(c)
        #train[c] = train[c].fillna("NAN")
        #test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CONS)} CATEGORICAL FEATURES: {CONS}")


# age_at_hctに関しては、独自の方法でラベル付けするため、引きます
CONS.remove('age_at_hct') 
CONS.remove('donor_age') 


print(f"In these features, there are {len(CONS)} CATEGORICAL FEATURES: {CONS}")


combined = pd.concat([train,test],axis=0,ignore_index=True)
#print("Combined data shape:", combined.shape )

# LABEL ENCODE CATEGORICAL FEATURES
print("We LABEL ENCODE the CATEGORICAL FEATURES: ",end="")
for c in FEATURES:

    # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
    if c in CATS:
        print(f"{c}, ",end="")
        combined[c],_ = combined[c].factorize()
        combined[c] -= combined[c].min()
        combined[c] = combined[c].astype("int32")
        combined[c] = combined[c].astype("category")
        
    # REDUCE PRECISION OF NUMERICAL TO 32BIT TO SAVE MEMORY
    else:
        if combined[c].dtype=="float64":
            combined[c] = combined[c].astype("float32")
        if combined[c].dtype=="int64":
            combined[c] = combined[c].astype("int32")
    
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


train


test


def efs_time_average(df,train, column_name):
    col = column_name
    train_asia = train[train['race_group']== 1]
    df_mean = train_asia.groupby(col)["efs_time"].mean().to_frame().reset_index().rename(columns={"efs_time": str(col)+"average_efs_time"},
                                                                                    inplace=False)
    df= pd.merge(df, df_mean, on=col, how="left")
    return df


def dummie_efs (df,train,column_name):
    col = column_name
    # 各 year_hct ごとに 1 と 0 の数を集計
    efs_counts = train[[col,'efs']].value_counts().unstack(fill_value=0)

    # 1 の方が多ければ 1、0 の方が多ければ 0 を設定
    efs_counts['dum_efs'+str(column_name)] = (efs_counts[1] > efs_counts[0]).astype(int)

    # 元の df にマージ
    df = df.merge(efs_counts[['dum_efs'+str(column_name)]], on=col, how='left')
    return df 


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


# モデル回す時にwarning出てしまうので、消します
import warnings
warnings.filterwarnings('ignore')


train


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    x_train_0 = train.loc[train_index].copy() 
    x_valid_0 = train.loc[test_index].copy() 
    #print(x_valid_0 )
    x_test_0 = test.copy()
    # trainとvalidを分けた状態で平均を算出 validにはtrainのefs_timeの平均情報は入らないからリークを防げる
    x_train_0 = efs_time_average (x_train_0,x_train_0,'dri_score')
    x_train_0 = efs_time_average (x_train_0,x_train_0,'prim_disease_hct')
    x_train_0 = efs_time_average (x_train_0,x_train_0,'conditioning_intensity')
    x_train_0 = efs_time_average (x_train_0,x_train_0,'karnofsky_score')
    x_train_0 = efs_time_average (x_train_0,x_train_0,'sex_match')
    # valid data 
    x_valid_0 = efs_time_average (x_valid_0,x_train_0,'dri_score')
    x_valid_0 = efs_time_average (x_valid_0,x_train_0,'prim_disease_hct')
    x_valid_0 = efs_time_average (x_valid_0,x_train_0,'conditioning_intensity')
    x_valid_0 = efs_time_average (x_valid_0,x_train_0,'karnofsky_score')
    x_valid_0 = efs_time_average (x_valid_0,x_train_0,'sex_match')
    # test data
    x_test_0 = efs_time_average (x_test_0,x_train_0,'dri_score')
    x_test_0 = efs_time_average (x_test_0,x_train_0,'prim_disease_hct')
    x_test_0 = efs_time_average (x_test_0,x_train_0,'conditioning_intensity')
    x_test_0 = efs_time_average (x_test_0,x_train_0,'karnofsky_score')
    x_test_0 = efs_time_average (x_test_0,x_train_0,'sex_match')
    FEATURES = [c for c in x_train_0.columns if not c in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    x_train = x_train_0.loc[:,FEATURES].copy()
    y_train = x_train_0.loc[:,"y"]
    x_valid = x_valid_0.loc[:,FEATURES].copy()
    y_valid = x_valid_0.loc[:,"y"]
    x_test = x_test_0[FEATURES].copy()
    model_xgb = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        #early_stopping_rounds=25,
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)
    # foldごとに平均値をつけるので

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


feature_importance = model_xgb.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


# Featureを再設定
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    x_train_0 = train.loc[train_index].copy() 
    x_valid_0 = train.loc[test_index].copy() 
    #print(x_valid_0 )
    x_test_0 = test.copy()
    # trainとvalidを分けた状態で平均を算出 validにはtrainのefs_timeの平均情報は入らないからリークを防げる
    for j in range(len(CATS)):
        x_train_0 = dummie_efs (x_train_0,x_train_0,CATS[j])
        # valid data 
        x_valid_0 = dummie_efs (x_valid_0,x_train_0,CATS[j])
        # test data
        x_test_0 = dummie_efs (x_test_0,x_train_0,CATS[j])
        #print(CONS[j])
    # y,efs_time,efsを除いた、特徴量だけにする
    FEATURES = [c for c in x_train_0.columns if not c in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    x_train = x_train_0.loc[:,FEATURES].copy()
    y_train = x_train_0.loc[:,"y"]
    x_valid = x_valid_0.loc[:,FEATURES].copy()
    y_valid = x_valid_0.loc[:,"y"]
    x_test = x_test_0[FEATURES].copy()
    

    model_cat = CatBoostRegressor(
        task_type="GPU",  
        learning_rate=0.1,    
        grow_policy='Lossguide',
        #early_stopping_rounds=25,
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=250)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)


feature_importance = model_cat.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    x_train_0 = train.loc[train_index].copy() 
    x_valid_0 = train.loc[test_index].copy() 
    #print(x_valid_0 )
    x_test_0 = test.copy()
     # trainとvalidを分けた状態で平均を算出 validにはtrainのefs_timeの平均情報は入らないからリークを防げる
    for j in range(len(CATS)):
        #x_train_0 = dummie_efs (x_train_0,x_train_0,CONS[j])
        # valid data 
        #x_valid_0 = dummie_efs (x_valid_0,x_train_0,CONS[j])
        # test data
        #x_test_0 = dummie_efs (x_test_0,x_train_0,CONS[j])
        #print(CONS[j])
        #efs_time
        x_train_0 = efs_time_average (x_train_0,x_train_0,CATS[j])
        # valid data 
        x_valid_0 = efs_time_average (x_valid_0,x_train_0,CATS[j])
        # test data
        x_test_0 = efs_time_average (x_test_0,x_train_0,CATS[j])
        #print(CATS[j])
    # y,efs_time,efsを除いた、特徴量だけにする
    FEATURES = [c for c in x_train_0.columns if not c in RMV]
    #print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    x_train = x_train_0.loc[:,FEATURES].copy()
    y_train = x_train_0.loc[:,"y"]
    x_valid = x_valid_0.loc[:,FEATURES].copy()
    y_valid = x_valid_0.loc[:,"y"]
    x_test = x_test_0[FEATURES].copy()

    model_lgb = LGBMRegressor(
        device="gpu", 
        max_depth=3, 
        colsample_bytree=0.4,  
        #subsample=0.9, 
        n_estimators=2500, 
        learning_rate=0.02, 
        objective="regression", 
        verbose=-1, 
        #early_stopping_rounds=25,
    )
    model_lgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )
    
    # INFER OOF
    oof_lgb[test_index] = model_lgb.predict(x_valid)
    # INFER TEST
    pred_lgb += model_lgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for LightGBM KaplanMeier =",m)


feature_importance = model_lgb.feature_importances_ 
importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
plt.xlabel("Importance (Gain)")
plt.ylabel("Feature")
plt.title("LightGBM KaplanMeier Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


# SURVIVAL COX NEEDS THIS TARGET (TO DIGEST EFS AND EFS_TIME)
train["efs_time2"] = train.efs_time.copy()
train.loc[train.efs==0,"efs_time2"] *= -1


# age_at_hctに関しては、下記方法でないと精度が劣化するので注意
def label_age_at_hct (df):
    df['label_age_at_hct'] = df['age_at_hct'].apply(lambda x: 0 if x <25  else 1)
    return 


label_age_at_hct(train)
label_age_at_hct(test)


CATS3 =['graft_type',
'prod_type',
'dri_score',
'year_hct']


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb_cox = np.zeros(len(train))
pred_xgb_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    x_train_0 = train.loc[train_index].copy() 
    x_valid_0 = train.loc[test_index].copy() 
    #print(x_valid_0 )
    x_test_0 = test.copy()
    # trainとvalidを分けた状態で平均を算出 validにはtrainのefs_timeの平均情報は入らないからリークを防げる
    for j in range(len(CONS)):
        x_train_0 = dummie_efs (x_train_0,x_train_0,CONS[j])
        # valid data 
        x_valid_0 = dummie_efs (x_valid_0,x_train_0,CONS[j])
        # test data
        x_test_0 = dummie_efs (x_test_0,x_train_0,CONS[j])
    # y,efs_time,efsを除いた、
    x_train_columns = x_train_0.drop(columns=['efs_time2']) #cox比例ハザードの場合efs_time2を予測対象にするため、別で削除が必要
    FEATURES = [c for c in x_train_columns.columns if not c in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    x_train = x_train_0.loc[:,FEATURES].copy()
    y_train = x_train_0.loc[:,"efs_time2"]
    x_valid = x_valid_0.loc[:,FEATURES].copy()
    y_valid = x_valid_0.loc[:,"efs_time2"]
    x_test = x_test_0[FEATURES].copy()


    model_xgb_cox = XGBRegressor(
        device="cuda",
        max_depth=3,  
        colsample_bytree=0.5,  
        subsample=0.8,  
        n_estimators=2000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=80,
        objective='survival:cox',
        eval_metric='cox-nloglik',
    )
    model_xgb_cox.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500  
    )
    
    # INFER OOF
    oof_xgb_cox[test_index] = model_xgb_cox.predict(x_valid)
    # INFER TEST
    pred_xgb_cox += model_xgb_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost Survival:Cox =",m)


feature_importance = model_xgb_cox.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  # Replace FEATURES with your list of feature names
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Survival:Cox Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat_cox = np.zeros(len(train))
pred_cat_cox = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    x_train_0 = train.loc[train_index].copy() 
    x_valid_0 = train.loc[test_index].copy() 
    #print(x_valid_0 )
    x_test_0 = test.copy()
    # trainとvalidを分けた状態で平均を算出 validにはtrainのefs_timeの平均情報は入らないからリークを防げる
    #for j in range(len(CONS)):
        #x_train_0 = dummie_efs (x_train_0,x_train_0,CONS[j])
        # valid data 
        #x_valid_0 = dummie_efs (x_valid_0,x_train_0,CONS[j])
        # test data
        #x_test_0 = dummie_efs (x_test_0,x_train_0,CONS[j])
        #print(CONS[j])
    x_train_0 = dummie_efs (x_train_0,x_train_0,'year_hct')
    x_valid_0 = dummie_efs (x_valid_0,x_train_0,'year_hct')
    x_test_0 = dummie_efs (x_test_0,x_train_0,'year_hct')
    x_train_columns = x_train_0.drop(columns=['efs_time2']) #cox比例ハザードの場合efs_time2を予測対象にするため、別で削除が必要
    FEATURES = [c for c in x_train_columns.columns if not c in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    x_train = x_train_0.loc[:,FEATURES].copy()
    y_train = x_train_0.loc[:,"efs_time2"]
    x_valid = x_valid_0.loc[:,FEATURES].copy()
    y_valid = x_valid_0.loc[:,"efs_time2"]
    x_test = x_test_0[FEATURES].copy()


    model_cat_cox = CatBoostRegressor(
        loss_function="Cox",
        #task_type="GPU",   
        iterations=400,     
        learning_rate=0.1,  
        grow_policy='Lossguide',
        use_best_model=False,
    )
    model_cat_cox.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=100)
    
    # INFER OOF
    oof_cat_cox[test_index] = model_cat_cox.predict(x_valid)
    # INFER TEST
    pred_cat_cox += model_cat_cox.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat_cox /= FOLDS


y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost Survival:Cox =",m)


feature_importance = model_cat_cox.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": FEATURES, 
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 15))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Survival:Cox Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.show()


train = train.astype({col: 'object' for col in train.select_dtypes('category').columns})


test = train.astype({col: 'object' for col in test.select_dtypes('category').columns})


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

"""
クロスバリデーションを行う関数やで！
ディープニューラルネットワークでカプランマイヤー法使うねん！
""" 
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
# 全データ用のOOF予測値を初期化
oof_dnn = np.zeros(len(train))
pred_dnn = np.zeros(len(test))
    
for i, (train_index, val_index) in enumerate(kf.split(train)):
    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    x_train_columns = train.drop(columns=['efs_time2', 'label_age_at_hct']) #cox比例ハザードの場合efs_time2を予測対象にするため、別で削除が必要
    FEATURES = [c for c in x_train_columns.columns if not c in RMV]
    print(FEATURES)
    x_train = train.loc[train_index, FEATURES].copy()
    y_train = train.loc[train_index, "y"]
    x_valid = train.loc[val_index, FEATURES].copy()
    y_valid = train.loc[val_index, "y"]
    x_test = test[FEATURES].copy()
    
    # NaNを処理するで〜（エラー回避のため）
    x_train = x_train.fillna(x_train.mean())
    x_valid = x_valid.fillna(x_train.mean())
    x_test = x_test.fillna(x_train.mean())
        
    # DNNモデルの構築やで〜
    model_dnn = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
        
    # モデルのコンパイル
    model_dnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )
        
    # 早期停止の設定
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
        )
        
    # モデルの学習
    model_dnn.fit(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=250
    )
        
    # 検証データの予測
    oof_dnn[val_index] = model_dnn.predict(x_valid).flatten()
        
    # テストデータの予測
    pred_dnn += model_dnn.predict(x_test).flatten()
    
pred_dnn /= FOLDS


# 精度確認
y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_dnn
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for DNN_kp =",m)


from scipy.stats import rankdata 

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_xgb) + rankdata(oof_cat) + rankdata(oof_lgb)\
                     + rankdata(oof_xgb_cox) + rankdata(oof_cat_cox) + rankdata(oof_dnn)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(pred_xgb) + rankdata(pred_cat) + rankdata(pred_lgb)\
                     + rankdata(pred_xgb_cox) + rankdata(pred_cat_cox)  + rankdata(prd_dnn)
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

