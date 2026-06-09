import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')


train_df.head()


train_df.shape


train_df.info()


train_df.isnull().sum()


test_df.head()


test_df.shape


test_df.info()


test_df.isnull().sum()


targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
features = [col for col in train_df.columns if col not in targets and col != 'id']

train_df[features].describe().T.style.background_gradient(cmap='Blues')


train_df[targets].sum()


target_data = train_df[targets].sum().sort_values(ascending=False)
sns.barplot(x=target_data.index, y=target_data.values, palette='pastel');
#Class Imbalance


cm = train_df[features].corr()
cmap = sns.diverging_palette(220, 20, as_cmap=True, s=60, l=90)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, linewidths=0.5, fmt=".2f", cmap=cmap)
plt.show()


corr_unstacked = cm.abs().unstack()


corr_unstacked[(corr_unstacked > 0.9) & (corr_unstacked < 1.0)].sort_values(ascending=False).drop_duplicates()


plot = ['Pixels_Areas', 'X_Perimeter', 'Sum_of_Luminosity', 'Steel_Plate_Thickness']

fig, axes = plt.subplots(2, 2,figsize=(15, 10))

for col, ax in zip(plot, axes.flatten()):
    sns.histplot(train_df[col], kde=True, ax=ax, color='purple')
    ax.set_title(col) 

plt.tight_layout()
plt.show()


train_df['is_train'] = 1
test_df['is_train'] = 0

for col in targets:
    test_df[col] = 0

df = pd.concat([train_df, test_df], ignore_index=True)


df['X_Range'] = df['X_Maximum'] - df['X_Minimum']
df['Y_Range'] = df['Y_Maximum'] - df['Y_Minimum']

df['Density'] = df['Pixels_Areas'] / (df['X_Range'] * df['Y_Range'] + 1)

df['Aspect_Ratio'] = df['X_Range'] / (df['Y_Range'] + 1)

df['Luminosity_Range'] = df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']


#log
skewed_features = ['Pixels_Areas', 'Sum_of_Luminosity', 'X_Perimeter', 'Y_Perimeter', 
                   'X_Range', 'Y_Range', 'Aspect_Ratio']

for feature in skewed_features:
    df[feature] = np.log1p(df[feature].abs())


drop = ['id', 'TypeOfSteel_A400']
df = df.drop(columns=drop)


train_new = df[df['is_train'] == 1].drop(columns=['is_train'])
test_new = df[df['is_train'] == 0].drop(columns=['is_train'] + targets)


train_new.head()


test_new.head()


features = [col for col in train_new.columns if col not in targets]
x = train_new[features].copy() 
y = train_new[targets].copy()


x = x.replace([np.inf, -np.inf], np.nan)
test_new = test_new.replace([np.inf, -np.inf], np.nan)

x = x.fillna(0)
test_new = test_new.fillna(0)


scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
test_scaled = pd.DataFrame(scaler.transform(test_new[features]), columns=features)


xgb_params = {'n_estimators': 1000,'learning_rate': 0.05,'max_depth': 6,'subsample': 0.8,
              'colsample_bytree': 0.8,'n_jobs': -1,'random_state': 42,'verbosity': 0, 'tree_method':'gpu_hist'}

lgbm_params = {'n_estimators': 1000,'learning_rate': 0.05,'max_depth': -1,'num_leaves': 31,'subsample': 0.8,
               'colsample_bytree': 0.8,'n_jobs': -1,'random_state': 42,'verbose': -1,'device':'gpu'}

def evaluate_multi_label_model(model_class, params, name):
    
    overall_auc = []
    for target in targets:
        clf = model_class(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, x_scaled, y[target], cv=skf, scoring='roc_auc')
        mean_score = scores.mean()
        overall_auc.append(mean_score)
        print(f"{target.ljust(15)} AUC: {mean_score:.4f}")
    print(f"\n>> {name} AUC: {np.mean(overall_auc):.5f}")
    return np.mean(overall_auc)

# XGBoost Performansı
xgb_score = evaluate_multi_label_model(XGBClassifier, xgb_params, "XGBoost")

# LightGBM Performansı
lgbm_score = evaluate_multi_label_model(LGBMClassifier, lgbm_params, "LightGBM")


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0,
        'tree_method': 'hist'}
    avg_auc = []
    for target in targets:
        clf = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Hız için 5 yerine 3 fold
        
        scores = cross_val_score(clf, x_scaled, y[target], cv=skf, scoring='roc_auc')
        avg_auc.append(scores.mean())
    return np.mean(avg_auc)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(study.best_params)
print(f"Best Score: {study.best_value:.5f}")


final_params = {
    'n_estimators': 535,
    'learning_rate': 0.011040911329582773,
    'max_depth': 6,
    'subsample': 0.7027249092143834,
    'colsample_bytree': 0.6023288247790183,
    'min_child_weight': 4,
    'gamma': 1.7303315704368445,
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
    'tree_method': 'hist'}

submission = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv')

for target in targets:
    clf = XGBClassifier(**final_params)
    clf.fit(x_scaled, y[target])
    preds = clf.predict_proba(test_scaled)[:, 1]
    submission[target] = preds
    
submission.to_csv('submission.csv', index=False)


import joblib

trained_models = {}
for target in targets:
    clf = XGBClassifier(**final_params)
    clf.fit(x_scaled, y[target])
    trained_models[target] = clf
model = {'models': trained_models,'scaler': scaler,'features': features,'targets': targets}

joblib.dump(model, 'steel_defect_model.pkl')

