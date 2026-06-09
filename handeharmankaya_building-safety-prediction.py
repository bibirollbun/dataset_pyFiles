import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import RobustScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')


train_df=pd.read_csv('/kaggle/input/predict-the-building-safety-under-the-earthquake/train.csv')
test_df=pd.read_csv('/kaggle/input/predict-the-building-safety-under-the-earthquake/test.csv')


train_df.head()


train_df.info()


train_df.shape


train_df.isnull().sum()


train_df.describe()


train_df.corr(numeric_only=True)


plt.figure(figsize=(20,20))
sns.heatmap(train_df.corr(numeric_only=True), annot=True,  fmt='.2f');


test_df.head()


test_df.info()


test_df.shape


test_df.isnull().sum()


test_df.describe()


test_df.corr(numeric_only=True)


plt.figure(figsize=(20,20))
sns.heatmap(test_df.corr(numeric_only=True), annot=True, fmt='.2f');


plt.figure(figsize=(8, 5))
sns.histplot(train_df['Max drift mm'], kde=True, bins=30, color='salmon');


#log(1 + x)
max_drift_log=np.log1p(train_df[['Max drift mm']])
plt.figure(figsize=(8, 5))
sns.histplot(max_drift_log['Max drift mm'], kde=True, bins=30, color='salmon');


sns.heatmap(train_df.corr()[['Max drift mm']].sort_values(by='Max drift mm', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1);


#target
y=train_df["Max drift mm"]
y=np.log1p(y)


train_df2 = train_df.drop(columns=['Max drift mm'])


all_data = pd.concat([train_df2, test_df], axis=0, ignore_index=True)


#Stiffness (E * I / L^3)
#Lower floors
all_data['stiffness_low'] = all_data['Columns 1-3 I mm4*10^6'] / (all_data['Floor height m']**3)
#Upper floors
all_data['stiffness_high'] = all_data['Columns 4-6 I mm4*10^6'] / (all_data['Floor height m']**3)
#Stiffness (Soft Story Risk)
all_data['stiffness_diff'] = all_data['stiffness_low'] - all_data['stiffness_high']
#Beam/Column Strength
all_data['strength_ratio'] = all_data['Column fy Mpa'] / all_data['Beam fy Mpa']


#Geometry
#Building Height
all_data['total_height'] = all_data['Number of floors'] * all_data['Floor height m']
#Building Width
all_data['total_width'] = all_data['Spans'] * all_data['Span width m']

#Slenderness Ratio
#Height / Width
all_data['slenderness'] = all_data['total_height'] / all_data['total_width']

#Total (Cross-sectional) Area
all_data['total_area_low'] = all_data['Columns 1-3 A mm2'] * (all_data['Spans'] + 1) 


#Earthquake
#Seismic Energy
all_data['seismic_power'] = all_data['PGA g'] * all_data['Magnitude']
#Fault Line Effect
all_data['fault_attenuation'] = all_data['Magnitude'] / np.log1p(all_data['Distance to fault km'])


#Mass and Force (F=ma)
#Total Building Mass
all_data['total_mass'] = all_data['Floor mass kg'] * all_data['Number of floors']
#Base Shear Proxy
all_data['base_shear'] = all_data['total_mass'] * all_data['PGA g']


#inf NaN values
all_data = all_data.replace([np.inf, -np.inf], np.nan)
all_data = all_data.fillna(0)


all_data.head()


scaler = RobustScaler()
all_data_scaled = pd.DataFrame(scaler.fit_transform(all_data), columns=all_data.columns)


train_idx = len(train_df)
train = all_data_scaled.iloc[:train_idx]
test = all_data_scaled.iloc[train_idx:]


xgb_params = {'n_estimators': 3000,'learning_rate': 0.01,'max_depth': 6,'subsample': 0.7,'colsample_bytree': 0.7,
              'reg_alpha': 0.5,'reg_lambda': 0.5,'n_jobs': -1,'random_state': 42,'objective': 'reg:absoluteerror'}

cat_params = {'iterations': 3000,'learning_rate': 0.01,'depth': 6,'l2_leaf_reg': 3,'loss_function': 'MAE',
              'verbose': 0,'random_state': 42}

lgbm_params = {'n_estimators': 3000,'learning_rate': 0.01,'max_depth': 6,'num_leaves': 31,'subsample': 0.7,
               'colsample_bytree': 0.7,'reg_alpha': 0.5,'reg_lambda': 0.5,'objective': 'mae','verbose': -1,
               'random_state': 42}

#K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train)) 
test_preds = np.zeros(len(test)) 
cv_scores = []

X_values = train.values
y_values = y.values  
X_test_values = test.values


for fold, (train_idx, val_idx) in enumerate(kf.split(X_values, y_values)):
    X_tr, X_val = X_values[train_idx], X_values[val_idx]
    y_tr, y_val = y_values[train_idx], y_values[val_idx]
    
    xgb = XGBRegressor(**xgb_params)
    cat = CatBoostRegressor(**cat_params)
    lgbm = LGBMRegressor(**lgbm_params)
    
    model = VotingRegressor(estimators=[('xgb', xgb), ('cat', cat), ('lgbm', lgbm)],
                            weights=[0.35, 0.45, 0.20])
    
    model.fit(X_tr, y_tr)
    
    val_pred_log = model.predict(X_val)
    oof_preds[val_idx] = val_pred_log 
    
    test_preds += model.predict(test) / 5 
    
    fold_score = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(val_pred_log)) * 100
    cv_scores.append(fold_score)
    print(f"Fold {fold+1} MAPE: %{fold_score:.4f}")

print(f"Mean CV MAPE: %{np.mean(cv_scores):.4f}")

final_real_preds = np.expm1(test_preds) 

min_drift = train_df['Max drift mm'].min()
final_real_preds = [max(x, min_drift) for x in final_real_preds]


df_test=pd.read_csv('/kaggle/input/predict-the-building-safety-under-the-earthquake/test.csv')
df_test['Index']=df_test.index
submission = pd.DataFrame({'Index': df_test.index,'Max drift mm': final_real_preds})
submission.to_csv('submission_kfold_tuned.csv', index=False)


xgb_final = XGBRegressor(**xgb_params)
cat_final = CatBoostRegressor(**cat_params)
lgbm_final = LGBMRegressor(**lgbm_params)

voting_model = VotingRegressor(estimators=[('xgb', xgb_final), ('cat', cat_final), ('lgbm', lgbm_final)],
                               weights=[0.35, 0.45, 0.20])

voting_model.fit(X_values, y_values)


import joblib
joblib.dump(voting_model, 'voting_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

