
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


import kagglehub

# Download latest version
path = kagglehub.dataset_download("ruchikakumbhar/calories-burnt-prediction")

print("Path to dataset files:", path)


df_orginal = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")


df_orginal


def plot_count(feature, title, df, size=1, ordered=True):
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    if ordered:
        g = sns.countplot(x=feature, data=df, order = df[feature].value_counts().index[:20], palette="Set3")
    else:
        g = sns.countplot(x=feature, data = df, palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show() 


plot_count("Sex", "Sex", train,4)


plot_count("Weight", "Weight", train,4)


labels = ['20s', '30s', '40s', '50s', '60s', '70s', '80s']
train['Age_binned'] = pd.cut(train['Age'],bins=range(20, 99, 10), labels=labels,right=False)
test['Age_binned'] = pd.cut(test['Age'],bins=range(20, 99, 10), labels=labels,right=False)


train_cat = pd.get_dummies(train[['Sex','Age_binned']],drop_first = False)
train_cat = train_cat.astype(int)
test_cat = pd.get_dummies(test[['Sex','Age_binned']],drop_first = False)
test_cat = test_cat.astype(int)



# cat_cols = train['Age']
# cat_cols = train['Age_binned']
cat_cols = train_cat[['Age_binned_20s',
       'Age_binned_30s', 'Age_binned_40s', 'Age_binned_50s', 'Age_binned_60s',
       'Age_binned_70s', 'Age_binned_80s']]

by_age_table = pd.DataFrame()
by_age_table['Calories'] = train['Calories']
by_age_table['Duration'] = train['Duration']
by_age_table['Heart_Rate'] = train['Heart_Rate']
by_age_table['Age_binned'] = train['Age_binned']

Calories_by_age = by_age_table.groupby('Age_binned', as_index=False)['Calories'].mean()
Duration_by_age = by_age_table.groupby('Age_binned', as_index=False)['Duration'].mean()
Heart_Rate_by_age = by_age_table.groupby('Age_binned', as_index=False)['Heart_Rate'].mean()


target_enc = pd.DataFrame()
target_enc['Age_binned'] = Duration_by_age['Age_binned']
target_enc['Duration_by_age'] = Duration_by_age['Duration']
target_enc['Calories_by_age'] = Calories_by_age['Calories']
target_enc['Heart_Rate_by_age'] = Heart_Rate_by_age['Heart_Rate']
train = train.merge(target_enc, on='Age_binned', how='left')
test = test.merge(target_enc, on='Age_binned', how='left')



train_num = train.drop(['Sex','Age_binned'],axis=1)


test_num = test.drop(['Sex','Age_binned'],axis=1)


train_data = pd.concat([train_num,train_cat],axis=1)



# filtered_df = train[(train["Genre"] == 'Business')] #& (train["Podcast_Name_Athlete's Arena"] == 1)]

# x=train_data['Body_Temp']
# y=train_data['Heart_Rate']

# # x=train['Episode_Title']
# # y=train['Listening_Time_minutes']

# plt.scatter(x,y,s=1)


h_m = train_data['Height']/100
h_m_s = h_m**2
train_data['BMI'] = train_data['Weight']/h_m_s


# train_data['Duration_by_age'] = train_data['Duration']/train_data['Age']
# train_data['Duration_Heart_Rate'] = train_data['Duration']*train_data['Heart_Rate']
train_data['Duration_Heart_Rate_Body_Temp'] = train_data['Duration']*train_data['Heart_Rate']*train_data['Body_Temp']
train_data['Duration_Heart_Rate_Body_Temp'] = np.log(train_data['Duration_Heart_Rate_Body_Temp'])
# train_data['Heart_Rate_by_Body_Temp'] = train_data['Heart_Rate']/train_data['Body_Temp']


train_data['Duration'].hist()


test_data = pd.concat([test_num,test_cat],axis=1)


h_m_test = test_data['Height']/100
h_m_s_test = h_m_test**2
test_data['BMI'] = test_data['Weight']/h_m_s_test


# test_data['Duration_by_age'] = test_data['Duration']/test_data['Age']
# test_data['Duration_Heart_Rate'] = test_data['Duration']*test_data['Heart_Rate']
test_data['Duration_Heart_Rate_Body_Temp'] = test_data['Duration']*test_data['Heart_Rate']*test_data['Body_Temp']
test_data['Duration_Heart_Rate_Body_Temp'] = np.log(test_data['Duration_Heart_Rate_Body_Temp'])
# test_data['Heart_Rate_by_Body_Temp'] = test_data['Heart_Rate']/test_data['Body_Temp']


train_data.info()


test_data.info()


plt.figure(figsize=(20, 16))
sns.heatmap(train_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()


corr_matrix = train_data.corr()

corr_matrix["Calories"].sort_values(ascending=False)


p_corr_matrix = train_data.corr(method = 'spearman')

p_corr_matrix["Calories"].sort_values(ascending=False)


X = train_data.drop(['id','Calories'],axis=1)
target = train_data['Calories']

test_data = test_data.drop(['id'],axis=1)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,target,test_size=0.2,random_state=42)





y_train = np.log(y_train + 1)





from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,RocCurveDisplay
from scipy.special import expit
from sklearn.model_selection import cross_val_score
import optuna


model = XGBRegressor(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.4f}")

importances = model.feature_importances_

if isinstance(X, pd.DataFrame):
    feature_names = X.columns
else:
    feature_names = [f'Feature {i}' for i in range(X.shape[1])]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()  
plt.title('Feature Importance (XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


def objective(trial):
    lgbm_params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }

    xgb_params = {
        'max_depth': trial.suggest_int('xgb_max_depth', 5, 20),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 10.0),
    }


    lgbm = LGBMRegressor(**lgbm_params)
    xgb = XGBRegressor(**xgb_params)


    voting_regressor = VotingRegressor(estimators=[('lgbm', lgbm), ('xgb', xgb)])

    score = cross_val_score(voting_regressor, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
    return np.mean(score) 

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, timeout=3600) 


print("Best Hyperparameters:", study.best_params)


best_params = study.best_params

lgbm_best = LGBMRegressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda']
)

xgb_best = XGBRegressor(
    n_estimators=best_params['xgb_n_estimators'],
    learning_rate=best_params['xgb_learning_rate'],
    max_depth=best_params['xgb_max_depth'],
    subsample=best_params['xgb_subsample'],
    colsample_bytree=best_params['xgb_colsample_bytree'],
    reg_alpha=best_params['xgb_reg_alpha'],
    reg_lambda=best_params['xgb_reg_lambda']
)


final_voting_regressor = VotingRegressor(estimators=[('lgbm', lgbm_best), ('xgb', xgb_best)])


final_voting_regressor.fit(X_train, y_train)


y_pred = final_voting_regressor.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(f"Final MSE: {mse:.4f}")


train_data


pred = final_voting_regressor.predict(test_data)
pred = np.exp(pred) - 1


submission = pd.DataFrame()
submission['id'] = sample_submission['id']

submission['Calories'] = pred

file_name = 'submission.csv'
submission.to_csv(file_name, index=False)
submission




