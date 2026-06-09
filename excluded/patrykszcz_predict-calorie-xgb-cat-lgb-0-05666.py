import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler,PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
from lightgbm import LGBMRegressor
import itertools
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.drop('id',axis = 1)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test = df_test.drop('id',axis=1)


def feature_engineering(df):

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 2})
    df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
    df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex']) + 1
    for col in ['Sex', 'Age', 'AgeSex']:
        df['CAT_' + col] = df[col].astype('category')
        
    features = ['Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Age', 'Sex', 'AgeSex']

    for comb in itertools.combinations(features, 2):
        df[" * ".join(comb)] = df[comb[0]] * df[comb[1]]
        df[" / ".join(comb)] = df[comb[0]] / df[comb[1]]
        df[" ** ".join(comb)] = df[comb[0]] * (df[comb[1]] ** 2)
        df[" *** ".join(comb)] = df[comb[1]] * (df[comb[0]] ** 2)
        
    
    return df


train = feature_engineering(df)
test = feature_engineering(df_test)
train['Duration_cat'] = pd.cut(train['Duration'],bins = 10 , labels=False, right=False)
test['Duration_cat'] = pd.cut(test['Duration'],bins = 10 , labels=False, right=False)


X = train.drop(['Calories'], axis = 1 )
y = np.log1p(train["Calories"])


cat_columns = [i for i in X.columns if X[i].dtype == 'category']
cat_columns


FOLDS = 20
KF = KFold(n_splits=FOLDS, shuffle = True, random_state = 42)
cat_features = cat_columns
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))
oof_lgb = np.zeros(len(train))
pred_lgb = np.zeros(len(test))


 # CATBOOST MODEL
cat_model = CatBoostRegressor(
    iterations= 3500,
    learning_rate= 0.02,
    depth= 12,
    loss_function= 'RMSE',
    l2_leaf_reg= 3,
    random_seed= 42,
    eval_metric= 'RMSE',
    early_stopping_rounds = 200,
    verbose= 1000,
    task_type= 'GPU')

 ## XGBOOST
xgb_model = XGBRegressor(
    max_depth=10,
    colsample_bytree=0.55,
    subsample=0.9,
    n_estimators=2000,
    learning_rate=0.01,
    gamma=0.01,
    max_delta_step=2,
    reg_alpha= 2,
    reg_lambda= 1,
    early_stopping_rounds=100,
    eval_metric="rmse",
    random_state = 13,
    enable_categorical=True,
    device = 'cuda')

lgb_model  = LGBMRegressor(
    objective= "regression",
    metric= "rmse",
    learning_rate=0.02,
    n_estimators= 3000, 
    num_leaves= 128,  
    max_depth= 10, 
    min_child_samples= 20, 
    min_split_gain= 0.01,
    subsample= 0.8,
    colsample_bytree= 0.8,
    early_stopping_rounds=100,
    reg_alpha= 3.0, 
    reg_lambda= 1.0,
    random_state= 42,
    verbosity= -1,
    feature_fraction= 0.7,
    force_col_wise=True
)

for i, (train_idx,valid_idx) in enumerate(KF.split(X,y)):
    print('#' * 15, i+1, '#' *15)
    ## SPLIT DS 
    X_train,y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
 
    ## CATBOOST FIT
    cat_model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],cat_features=cat_features,
            use_best_model=True,verbose=0)
    ## XGB FIt
    xgb_model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],verbose=0)
    ## LGB FIT
    lgb_model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)])

    ## PREDICTION CATBOOST
    oof_cat[valid_idx] = cat_model.predict(X_valid)
    pred_cat += cat_model.predict(test)
    ## PREDICTION XGB
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    pred_xgb += xgb_model.predict(test)
    ## PREDICTION LGB
    oof_lgb[valid_idx] = lgb_model.predict(X_valid)
    pred_lgb += lgb_model.predict(test)

    
    cat_rmse = mean_squared_error(y_valid,oof_cat[valid_idx]) ** 0.5
    xgb_rmse = mean_squared_error(y_valid, oof_xgb[valid_idx]) ** 0.5
    lgb_rmse = mean_squared_error(y_valid, oof_lgb[valid_idx]) ** 0.5
   
    
    print(f'FOLD {i+1} CATBOOST_RMSE = {cat_rmse:.4f} <=> XGB_RMSE = {xgb_rmse:.4f} <=> LGB_RMSE = {lgb_rmse:.4f} ')


# Average predictions from folds
pred_cat /= FOLDS
pred_xgb /= FOLDS
pred_lgb /= FOLDS


print(f'FINAL RMSE CATBOOST: {mean_squared_error(y,oof_cat) ** 0.5:.4f}')
print(f'FINAL RMSE XGBBOOST: {mean_squared_error(y,oof_xgb) ** 0.5:.4f}')
print(f'FINAL RMSE LGBBOOST: {mean_squared_error(y,oof_lgb) ** 0.5:.4f}')


y_preds = np.expm1(pred_cat) * 0.01 + np.expm1(pred_xgb)*0.98 + np.expm1(pred_lgb)*0.01
y_preds = np.clip(y_preds, 1, 314)

# Save submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
print('submission saved')
submission.head()




