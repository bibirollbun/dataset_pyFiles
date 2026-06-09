import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
import lightgbm
from mlxtend.regressor import StackingCVRegressor
from sklearn import clone
from sklearn.ensemble import VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import Lasso, LogisticRegression, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer, mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from catboost import Pool, CatBoostClassifier


def init():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    warnings.simplefilter('ignore')
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.max_columns", 1000)

def show_relation(data, colx, coly):
    if data[colx].dtype == 'object' or data[colx].dtype == 'category' or len(data[colx].unique()) < 20:
        seaborn.boxplot(x=colx, y=coly, data=data)
    else:
        plt.scatter(data[colx], data[coly])
    plt.xlabel(colx)
    plt.ylabel(coly)
    plt.show()


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break
        return score

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])      


init()

df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df_train_additional = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
pd.concat([df_train, df_train_additional], ignore_index=True)

print("Start Feature enggering" + "-" * 70 + "\n")
df_all = pd.concat([df_train.drop(['id', 'Fertilizer Name'], axis=1), df_test.drop(['id'], axis=1)], axis=0)

df_all['Temp_Humidity_Interaction'] = df_all['Temparature'] * df_all['Humidity']
df_all['N_P_Ratio'] = df_all['Nitrogen'] / (df_all['Phosphorous'].replace(0, 1e-6))
df_all['K_P_Ratio'] = df_all['Potassium'] / (df_all['Phosphorous'].replace(0, 1e-6))
df_all['Soil_Crop_Combination'] = df_all['Soil Type'].astype(str) + '_' + df_all['Crop Type'].astype(str)

df_all['P_to_K'] = df_all['Phosphorous'] / (df_all['Potassium'] + 1e-5)
df_all['Total_NPK'] = df_all['Nitrogen'] + df_all['Phosphorous'] + df_all['Potassium']
df_all['Climate_Index'] = (df_all['Temparature'] + df_all['Humidity']) / 2
df_all['Water_Stress'] = df_all['Humidity'] - df_all['Moisture']

original_numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in original_numerical_cols:
    df_all[f'{col}_Binned'] = df_all[col].astype(str)

numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
                      'Temp_Humidity_Interaction', 'N_P_Ratio', 'K_P_Ratio']
categorical_features = ['Soil Type', 'Crop Type', 'Soil_Crop_Combination']
categorical_features.extend([f'{col}_Binned' for col in original_numerical_cols])

poly_features_to_transform = original_numerical_cols
poly = PolynomialFeatures(degree=2, include_bias=False)
df_all_transformers = poly.fit_transform(df_all[poly_features_to_transform])

poly_feature_names = poly.get_feature_names_out(poly_features_to_transform)
df_all = df_all.drop(columns=poly_features_to_transform)
df_all = pd.concat([df_all, pd.DataFrame(df_all_transformers, columns=poly_feature_names,index=df_all.index)], axis=1)

numerical_features = df_all.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df_all.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

all_features_ordered = numerical_features + categorical_features
df_all = df_all[all_features_ordered]

all_categories_union = {}
for col in categorical_features:
    if col in df_all.columns:
        all_categories_union[col] = pd.concat([
            df_all[col],
        ], axis=0).astype(str).unique()
    else:
        print(f"Warning: Categorical column '{col}' not found after feature engineering. Skipping conversion.")

for col in categorical_features:
    if col in df_all.columns:
        df_all[col] = pd.Categorical(df_all[col], categories=all_categories_union[col])


le = LabelEncoder()
X_train = df_all[:df_train.shape[0]]
Y_train = df_train['Fertilizer Name']
Y_train = le.fit_transform(Y_train)
X_test = df_all[df_train.shape[0]:]

print("Training model" + "-" * 70 + "\n")
model_xgb = XGBClassifier(
    max_depth=8,
    colsample_bytree=0.5,
    subsample=0.7,
    n_estimators=3000,
    learning_rate=0.03,
    gamma=0.5,
    max_delta_step=2,
    reg_alpha=5,
    reg_lambda=3,
    early_stopping_rounds=100,
    objective='multi:softprob',
    random_state=13,
    enable_categorical=True,
    tree_method='hist',
    device='cuda'
)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
pred_xgb = np.zeros((X_test.shape[0], len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, Y_train)):
    print(f"\nFold {fold + 1}/{kfold.n_splits}")

    x_fold_train, x_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = Y_train[train_idx], Y_train[val_idx]

    model_xgb.fit(
        x_fold_train, y_fold_train,
        eval_set = [(x_fold_val, y_fold_val)],
        verbose = 100,
    )

    pred_xgb += model_xgb.predict_proba(X_test) / kfold.n_splits


pred_top3_xgb = np.argsort(pred_xgb, axis=1)[:, -3:][:, ::-1]
top3_label = []
for row in pred_top3_xgb:
    converted = [le.classes_[i] for i in row]
    top3_label.append(converted)

submission = pd.DataFrame({
    'id': df_test['id'],
    'Fertilizer Name': [' '.join(preds) for preds in top3_label],
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

