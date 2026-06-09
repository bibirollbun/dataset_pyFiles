import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import f1_score, matthews_corrcoef, ConfusionMatrixDisplay


train_df=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.head()


test_df.head()


train_df.info()


train_df.shape


test_df.shape


train_df.describe()


train_df.corr(numeric_only=True)


train_df.isnull().sum()


# Combine features to capture interaction effects.
train_df['temp_range'] = train_df['maxtemp'] - train_df['mintemp']
test_df['temp_range'] = test_df['maxtemp'] - test_df['mintemp']


# Convert continuous features into categorical bins.
train_df['humidity_bin'] = pd.cut(train_df['humidity'], bins=[0, 50, 75, 100], labels=[1, 2, 3]).astype(int)
test_df['humidity_bin'] = pd.cut(test_df['humidity'], bins=[0, 50, 75, 100], labels=[1, 2, 3]).astype(int)


def feature_engineering(train_df):
    
    train_df['hci'] = train_df['humidity'] * train_df['cloud']
    train_df['hsi'] = train_df['humidity'] * train_df['sunshine']
    train_df['csr'] = train_df['cloud'] / (train_df['sunshine'] + 1e-5)
    train_df['rd'] = 100 - train_df['humidity']
    train_df['sp'] = train_df['sunshine'] / (train_df['sunshine'] + train_df['cloud'] + 1e-5)
    train_df['wi'] = (0.4 * train_df['humidity']) + (0.3 * train_df['cloud']) - (0.3 * train_df['sunshine'])
    return train_df

train_df = feature_engineering(train_df)


def feature_engineering(test_df):
    
    test_df['hci'] = test_df['humidity'] * test_df['cloud']
    test_df['hsi'] = test_df['humidity'] * test_df['sunshine']
    test_df['csr'] = test_df['cloud'] / (test_df['sunshine'] + 1e-5)
    test_df['rd'] = 100 - test_df['humidity']
    test_df['sp'] = test_df['sunshine'] / (test_df['sunshine'] + test_df['cloud'] + 1e-5)
    test_df['wi'] = (0.4 * test_df['humidity']) + (0.3 * test_df['cloud']) - (0.3 * test_df['sunshine'])
    return test_df
test_df = feature_engineering(test_df)


# Drop column 'id'
train_df = train_df.drop(columns=['id', 'day', 'pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'winddirection'])
submission = pd.DataFrame({'id': test_df['id']})
test_df = test_df.drop(columns=['id', 'day', 'pressure', 'maxtemp', 'temparature', 'mintemp','dewpoint', 'winddirection'])


plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


sns.pairplot(train_df, diag_kind='kde', hue='rainfall')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='humidity_bin', y='rainfall', data=train_df)
plt.title('Rainfall vs Humidity Bins')
plt.show()


# Features and target variable
X = train_df.drop('rainfall', axis=1)
y = train_df['rainfall']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


xgb_params = {
    'n_estimators': 2407,
    'eta': 0.009462133032592785,
    'gamma': 0.2865859948765318,
    'max_depth': 31,
    'min_child_weight': 47,
    'subsample': 0.6956431754146083,
    'colsample_bytree': 0.3670732604094118,
    'grow_policy': 'lossguide',
    'max_leaves': 73,
    'enable_categorical': True,
    'n_jobs': -1,
    'device': 'cuda',
    'tree_method': 'hist'
}

lgbm_params = {
    'n_estimators': 2500,
    'random_state': 42,
    'max_bin': 1024,
    'colsample_bytree': 0.6,
    'reg_lambda': 80,
    'verbosity': -1
}


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train.shape


xgb_model = XGBClassifier(**xgb_params)
lgbm_model = LGBMClassifier(**lgbm_params)

def model_trainer(model, X, y, n_splits=5, random_state=42):

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    oof_probs, oof_mccs, oof_accuracies = [], [], []
    print("="*80)
    print(f"Training {model.__class__.__name__}")
    print("="*80, end="\n")
    
    for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_test_fold, y_test_fold = X[test_idx], y[test_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        
        accuracy = accuracy_score(y_test_fold, y_pred)
        mcc = matthews_corrcoef(y_test_fold, y_pred)
        oof_probs.append(model.predict_proba(X_test_fold))
        oof_mccs.append(mcc)
        oof_accuracies.append(accuracy)
        
        print(f"--- Fold {fold+1} MCC: {mcc:.6f}, Accuracy: {accuracy:.6f}")
        
    print(f"\n---> Mean MCC: {np.mean(oof_mccs):.6f} ± {np.std(oof_mccs):.6f}")
    print(f"---> Mean Accuracy: {np.mean(oof_accuracies):.6f} ± {np.std(oof_accuracies):.6f}")
    return oof_probs, oof_mccs, oof_accuracies

oof_probs_xgb, oof_mccs_xgb, oof_accuracies_xgb = model_trainer(xgb_model, X_train_scaled, y_train, random_state=42)
oof_probs_lgbm, oof_mccs_lgbm, oof_accuracies_lgbm = model_trainer(lgbm_model, X_train_scaled, y_train, random_state=42)

y_val_pred_xgb = xgb_model.predict(X_test_scaled)
y_val_pred_lgbm = lgbm_model.predict(X_test_scaled)

y_val_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
y_val_prob_lgbm = lgbm_model.predict_proba(X_test_scaled)[:, 1]


import joblib

# Save the models
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(lgbm_model, 'lgbm_model.pkl')

print("Models saved successfully!")


test_df.head()


final_ped=xgb_model.predict(test_df)


submission['rainfall']=final_ped


submission.head()


submission.to_csv("submission.csv", index=False)




