import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.drop('id',axis = 1)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_test = df_test.drop('id',axis=1)
df = df.drop_duplicates()


df.describe()


numeric_cols = ['Age', 'Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Sex', 'AgeSex']
def feature_engineering(df : pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 2})
    df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
    df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex']) + 1
    for i in range(len(numeric_cols)):

        feature_1 = numeric_cols[i]
        for j in range(i+1,len(numeric_cols)):
            feature_2 = numeric_cols[j]
            df[f'{feature_1}_x_{feature_2}'] = df[feature_1] * df[feature_2]
           
    return df


train = feature_engineering(df,numeric_cols)
test = feature_engineering(df_test,numeric_cols)
train["Sex"] = train["Sex"].astype("category")
test["Sex"] = test["Sex"].astype("category")
train['Duration_cat'] = pd.cut(train['Duration'],bins = 10 , labels=False, right=False)
test['Duration_cat'] = pd.cut(test['Duration'],bins = 10 , labels=False, right=False)


plt.figure(figsize = (14,14))
sns.heatmap(train.corr(numeric_only = True))
plt.title('Pearson Correletion')
plt.show()


X = train.drop(['Calories'], axis = 1 )
y = np.log1p(train["Calories"])


FOLDS = 10
KF = KFold(n_splits=FOLDS, shuffle = True, random_state = 42)
cat_features = ['Sex']
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
    colsample_bytree=0.75,
    subsample=0.9,
    n_estimators=2000,
    learning_rate=0.01,
    gamma=0.01,
    max_delta_step=2,
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
 
    ## CATBOOST fit
    cat_model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],cat_features=cat_features,
            use_best_model=True,verbose=0)
    ## XGB FIR
    xgb_model.fit(X_train,y_train,eval_set=[(X_valid,y_valid)],verbose=0)
    ## LGB MODEL
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
    
    print(f'FOLD {i+1} CATBOOST_RMSE = {cat_rmse:.4f} <=> XGB_RMSE = {xgb_rmse:.4f} <=> LGB_RMSE = {lgb_rmse:.4f}')


pred_cat *= FOLDS
pred_xgb *= FOLDS
pred_lgb *= FOLDS


# Average predictions from folds
pred_cat /= FOLDS
pred_xgb /= FOLDS
pred_lgb /= FOLDS


y_preds = np.expm1(pred_cat) * 0.30 + np.expm1(pred_xgb)*0.60 + np.expm1(pred_lgb)*0.1
y_preds = np.clip(y_preds, 1, 314)

# Save submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = y_preds
submission.to_csv("submission.csv", index=False)
print('submission saved')
submission.head()




