import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")



train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.sample(3)


def fe(df):
    df['show_popularity']=df['Host_Popularity_percentage']+df['Guest_Popularity_percentage']
    df['no_buffering']=df['Episode_Length_minutes']/(df['Number_of_Ads'] + 0.0001)#lesser the ads, more likely we would keep streaming.
    return df
train=fe(train)
test=fe(test)


train = train.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)
test = test.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)


FEATURES = [col for col in train.columns if col != 'Listening_Time_minutes']


from sklearn.model_selection import KFold
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,'Listening_Time_minutes'].copy()
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,'Listening_Time_minutes'].copy()
    X_test = test[FEATURES].copy()
    
    model_lgb = LGBMRegressor(n_estimators=10000,
                             learning_rate=0.02,
                              max_depth=10,
                              num_leaves=50,
                             device='gpu', 
                             random_state=42,
                            )
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
    )

    # INFER OOF
    oof_lgb[test_index] = model_lgb.predict(X_val)
    # INFER TEST
    pred_lgb += model_lgb.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_lgb /= FOLDS


oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    X_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,'Listening_Time_minutes'].copy()
    X_val = train.loc[test_index,FEATURES].copy()
    y_val = train.loc[test_index,'Listening_Time_minutes'].copy()
    X_test = test[FEATURES].copy()
    
    model_cat = CatBoostRegressor(n_estimators=10000,
                             learning_rate=0.02,
                              max_depth=10,
                              cat_features=['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment'],
                             random_state=42,
                             task_type='GPU',
                              verbose=2000
                            )
    model_cat.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],  
    )

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(X_val)
    # INFER TEST
    pred_cat += model_cat.predict(X_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub.Listening_Time_minutes = (pred_lgb+pred_cat)/2
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()




