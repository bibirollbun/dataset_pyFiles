import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler, 
    OneHotEncoder, 
    FunctionTransformer,
    LabelEncoder,
    PolynomialFeatures,
    OrdinalEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

import os
import math
from itertools import product
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


import matplotlib.cm as cm                           # Some configuration
import matplotlib.colors as mcolors

cmap = cm.Blues  
norm = mcolors.Normalize(vmin=0, vmax=10)

plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)


root = '../input'

train = pd.read_csv(os.path.join(root, 'playground-series-s5e6/train.csv'))
test = pd.read_csv(os.path.join(root, 'playground-series-s5e6/test.csv'))
submission = pd.read_csv(os.path.join(root, 'playground-series-s5e6/sample_submission.csv'))

original = pd.read_csv(os.path.join(root, 'fertilizer-prediction/Fertilizer Prediction.csv'))


temp = pd.concat([train, original], ignore_index=True)
temp.loc[len(train):, 'id'] = range(len(train), len(train)+len(original))
temp['id'] = temp['id'].astype(int)

duplicates = temp[temp.duplicated(subset=temp.loc[:, 'Temparature':'Fertilizer Name'].columns)]
train = temp


pd.concat([
    train.describe().T,
    train.isna().sum().rename('null'),
    train.dtypes.rename('dtype'),
    train.nunique().rename('nunique')
], axis=1)


cat_columns = train.select_dtypes(include=['object']).columns.tolist()
num_columns = train.columns.difference(cat_columns).drop('id').tolist()


fig, ax = plt.subplots(len(num_columns), 2, figsize=(10, len(num_columns) * 3))
ax = ax.flatten()

legend_on = True

for i in range(0, 2 * len(num_columns), 2):
    idx = (i - int(i / 2))
    unique = len(train[num_columns[idx]].unique())
    bins = unique if unique < 50 else 50

    if bins < 50:
        g = train.groupby([num_columns[idx], 'Fertilizer Name']).size().reset_index().pivot(columns=num_columns[idx], index='Fertilizer Name', values=0)
        garr = g.to_numpy()
        bottom = np.zeros(garr.shape[1])
        for j in range(len(garr)):
            color = cmap(norm(j))
            ax[i].bar(g.columns, garr[j], bottom=bottom, label=g.index[j], color=color)
            bottom += garr[j]
        if legend_on:
            ax[i].legend(title='Fertilizer Name')
            legend_on = not legend_on
                
        ax[i].set_title(f'{num_columns[idx]} by Fertilizer Name')
    else:
        sns.histplot(train, x=num_columns[idx], ax=ax[i], bins=bins)
        ax[i].set_title(f'Hist: {num_columns[idx]}')

    ax2 = ax[i].twinx()
    sns.kdeplot(train[num_columns[idx]], ax=ax2)
    ax[i].grid()

    sns.boxplot(train, x=num_columns[idx], ax=ax[i+1])
    ax[i+1].set_title(f'Boxplot: {num_columns[idx]}')
    ax[i+1].grid()


plt.tight_layout()
plt.show()


ncols = 3
nrows = math.ceil(len(cat_columns) / ncols)

fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
ax = ax.flatten()

legend_on = True

for i, col in enumerate(cat_columns):
    if col != 'Fertilizer Name':
        g = train.groupby([col, 'Fertilizer Name']).size().unstack(fill_value=0)
        bottom = np.zeros(len(g))
        for j, cat in enumerate(g.columns):
            color = cmap(norm(j))
            values = g[cat].values
            ax[i].bar(g.index, values, bottom=bottom, label=cat, color=color)
            bottom += values
        
        if legend_on:
            ax[i].legend(title='Fertilizer Name')
            legend_on = not legend_on
        ax[i].set_title(f'{col} by Fertilizer Name')
        
    else:
        sns.countplot(data=train, x=col, ax=ax[i], color=color)
        ax[i].set_title(col)
        
    ax[i].tick_params(axis='x', labelrotation=60)
    ax[i].grid()


for j in range(i + 1, len(ax)):
    ax[j].set_visible(False)

plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 7))

corr = train[num_columns].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)


plt.figure(figsize=(8, 4))

cross = pd.crosstab(train['Soil Type'], train['Crop Type'], normalize='index')
sns.heatmap(cross, annot=True, cmap='Blues')

plt.title('Soil Type vs. Crop Type')
plt.tight_layout()
plt.show()


def stacked_bar(df, groupby, ax, legend):
    g = df.groupby(groupby).size().unstack()
    bottom = np.zeros(len(g))
    for i, cat in enumerate(g.columns):
        color = cmap(norm(i))
        values = g[cat].values
        ax.bar(g.index, values, bottom=bottom, label=cat, color=color)
        bottom += values
    ax.set_title(f'{groupby[0]} by {groupby[1]}')
    if legend:
        ax.legend(title=groupby[1], fontsize=6)  


cols = ['Nitrogen', 'Potassium', 'Phosphorous']
groupby = ['Soil Type', 'Crop Type']

nrows, ncols = len(cols), len(groupby)
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*3))
ax = ax.flatten()

groups = list(product(cols, groupby))
for i, (x, g) in enumerate(groups):
    legend = True if int(i // 2) < 1 else False
    stacked_bar(train, [x, g], ax[i], legend=legend)

plt.tight_layout()
plt.show()


# How do soil types differ in moisture, temperature or nutrient level?
cols = ['Moisture', 'Temparature', 'Humidity']

fig, ax = plt.subplots(1, len(cols), figsize=(len(cols) * 4, 4))

for i, col in enumerate(cols):
    sns.violinplot(train, x=col, y='Soil Type', ax=ax[i], palette='Blues')
    ax[i].set_title(f'{col} by Soil Type')

plt.tight_layout()
plt.show()


# Which combinations of soil and crop types domniate the dataset?

counts = train.groupby(['Soil Type', 'Crop Type']).size()
props = {key: {'color': cmap(norm(value))} for key, value in counts.items()}

mosaic(train, index=['Soil Type', 'Crop Type'], properties=lambda key: props.get(key, {}))
plt.title('Soil Type and Crop Type Ratio')
plt.show()


def average_precision_at_k(y_true, y_pred_topk):
    for i, pred in enumerate(y_pred_topk):
        if pred == y_true:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(probs, true_labels, k=3):
    top_k_preds = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    
    ap_scores = []
    for i in range(len(true_labels)):
        ap = average_precision_at_k(true_labels[i], top_k_preds[i])
        ap_scores.append(ap)
    return np.mean(ap_scores)


class DataFrameColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, handle_unknown='use_encoded_value', unknown_value=-1):
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoders_ = {}

    def fit(self, X, y=None):
        X = X[self.columns].copy()
        for col in X.columns:
            enc = OrdinalEncoder(
                handle_unknown=self.handle_unknown,
                unknown_value=self.unknown_value
            )
            enc.fit(X[[col]])
            self.encoders_[col] = enc
        return self

    def transform(self, X):
        check_is_fitted(self, 'encoders_')
        X = X.copy()
        for col in self.columns:
            enc = self.encoders_[col]
            encoded = enc.transform(X[[col]]).ravel()
            X[col] = pd.Series(encoded, index=X.index).astype('int').astype('category')
        return X.drop(['id'], axis=1)

        
basic_pipeline = Pipeline(steps=[
    ('column-encoder', DataFrameColumnEncoder(columns=['Soil Type', 'Crop Type']))
])


basic_pipeline.fit(train.drop('Fertilizer Name', axis=1))
X = basic_pipeline.transform(train.drop('Fertilizer Name', axis=1))

# Label-Encode y
le = LabelEncoder()
y = le.fit_transform(train['Fertilizer Name'])
y = pd.Series(y)

X_test = basic_pipeline.transform(test)


lgbm_params = {
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'learning_rate': 0.065,
    'num_leaves': 170,
    'max_depth': 10,
    'min_child_samples': 19,
    'subsample': 0.65,
    'colsample_bytree': 0.43,
    'reg_alpha': 6.3,
    'reg_lambda': 5.56,
    'random_state': 42,
    'verbosity': -1,
    'devicer': 'gpu'
}

xgb1_params = {
    'max_depth': 12,
    'colsample_bytree': 0.467,
    'subsample': 0.86,
    'n_estimators': 6000,
    'learning_rate': 0.015,
    'gamma': 0.25,
    'max_delta_step': 4,
    'reg_alpha': 2.7,
    'reg_lambda': 1.4,
    'early_stopping_rounds': 5,
    'objective': 'multi:softprob',
    'random_state': 13,
    'enable_categorical': True,
    'tree_method': 'hist',
    'device': 'cuda'
}

xgb2_params = {
    "max_depth": 13,
    "colsample_bytree": 0.43,
    "subsample": 0.73,
    "n_estimators": 2500,
    "learning_rate": 0.04,
    "gamma": 0.09,
    "max_delta_step": 4,
    "reg_alpha": 0.43,
    "reg_lambda": 2.47,
    "early_stopping_rounds": 100,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 13,
    "enable_categorical": True,
    "device": "cuda"
}

xgb3_params = {
    "max_depth": 9,
    "colsample_bytree": 0.5,
    "subsample": 0.97,
    "n_estimators": 1500,
    "learning_rate": 0.015,
    "gamma": 0.0035,
    "max_delta_step": 1,
    "reg_alpha": 0.6,
    "reg_lambda": 4.6,
    "early_stopping_rounds": 100,
    "eval_metric": "mlogloss",
    "objective": "multi:softprob",
    "random_state": 25,
    "enable_categorical": True,
    "device": "cuda"
}


def train_lgbm(params, X, y, test, folds=5, verbose=True, 
                  random_state=42):
    
    model = LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = np.zeros((len(test), len(np.unique(y))))
    map3_scores = np.zeros(folds)

    cat_features = X.select_dtypes(include='category').columns.tolist()

    models = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            categorical_feature=cat_features
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test)
        
        map3 = map_at_k(oof_preds[val_idx], np.array(y_val), k=3)
        map3_scores[i] = map3
        models.append(model)
        
        if verbose:
            print(f'Fold: {i+1} | MAP @ 3: {map3:.5f}')

    test_preds /= folds
    return models, oof_preds, map3_scores, test_preds


def train_xgboost(params, X, y, test, folds=5, verbose=True, 
                  random_state=42):
    
    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = np.zeros((len(test), len(np.unique(y))))
    map3_scores = np.zeros(folds)

    models = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test)
        
        map3 = map_at_k(oof_preds[val_idx], np.array(y_val), k=3)
        map3_scores[i] = map3
        models.append(model)
        
        if verbose:
            print(f'Fold: {i+1} | MAP @ 3: {map3:.5f}')

    test_preds /= folds
    return models, oof_preds, map3_scores, test_preds


# Train LGBM Model
lgbm_models, lgbm_oof_preds, lgbm_map3_scores, lgbm_test_preds = train_lgbm(lgbm_params, X, y, X_test)


# Train XGB 1 Model
xgb1_models, xgb1_oof_preds, xgb1_map3_scores, xgb1_test_preds = train_xgboost(xgb1_params, X, y, X_test)


# Train XGB 2 Model
xgb2_models, xgb2_oof_preds, xgb2_map3_scores, xgb2_test_preds = train_xgboost(xgb2_params, X, y, X_test)


# Train XGB Model
# xgb3_models, xgb3_oof_preds, xgb3_map3_scores, xgb3_test_preds = train_xgboost(xgb3_params, X, y, X_test)


stacked_oof = np.concatenate([lgbm_oof_preds, xgb1_oof_preds, xgb2_oof_preds], axis=1)
stacked_test_preds = np.concatenate([lgbm_test_preds, xgb1_test_preds, xgb2_test_preds], axis=1)

X_oof = pd.DataFrame(stacked_oof)
X_test_preds = pd.DataFrame(stacked_test_preds)


def train_logistic_regression(params, X, y, test, folds=5, verbose=True, random_state=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    n_classes = len(np.unique(y))
    oof_preds = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(test), n_classes))
    map3_scores = np.zeros(folds)
    
    models = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Optional scaler + model pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**params))
        ])

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test)
        
        map3 = map_at_k(oof_preds[val_idx], np.array(y_val), k=3)
        map3_scores[i] = map3
        models.append(model)
        
        if verbose:
            print(f'Fold: {i+1} | MAP @ 3: {map3:.5f}')

    test_preds /= folds
    return models, oof_preds, map3_scores, test_preds


lr_params = {
    'penalty': 'l2',
    'solver': 'lbfgs',
    'C': 0.00459,
    'max_iter': 5000
}

_, lr_oof_preds, lr_map3_scores, lr_test_preds = train_logistic_regression(lr_params, X_oof, y, X_test_preds)


def mean_feature_importance(models, name='lgbm', columns=None):
    i = 0 
    counters = Counter()
    for model in models:
        if name == 'lgbm':
            values = model.booster_.feature_importance(importance_type='gain')
            scores = {columns[i]: v for i, v in enumerate(values)}
        else: 
            scores = model.get_booster().get_score(importance_type='gain')
        counters.update(scores)
        i += 1
    counters = {k: (v / i) for k, v in counters.items()}
    factor=1.0/sum(counters.values())
    return {k: v*factor for k, v in counters.items()}



fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax = ax.flatten()

# (a) Training map3 scores
ax[0].scatter(np.arange(1, len(lgbm_map3_scores)+1), lgbm_map3_scores, 
              marker='d', s=100, color='steelblue', linewidth=2, alpha=0.75, label='LGBM')
ax[0].plot(np.arange(1, len(lgbm_map3_scores)+1), lgbm_map3_scores, color='blue', linewidth=0.5, alpha=0.5)

ax[0].scatter(np.arange(1, len(xgb1_map3_scores)+1), xgb1_map3_scores, 
              marker='d', s=100, color='orange', linewidth=2, alpha=0.75, label='XGB1')
ax[0].plot(np.arange(1, len(xgb1_map3_scores)+1), xgb1_map3_scores, color='orange', linewidth=0.5, alpha=0.5)

ax[0].scatter(np.arange(1, len(xgb2_map3_scores)+1), xgb2_map3_scores, 
              marker='d', s=100, color='green', linewidth=2, alpha=0.75, label='XGB2')
ax[0].plot(np.arange(1, len(xgb2_map3_scores)+1), xgb2_map3_scores, color='green', linewidth=0.5, alpha=0.5)

ax[0].scatter(np.arange(1, len(lr_map3_scores)+1), lr_map3_scores, 
              marker='d', s=100, color='red', linewidth=2, alpha=0.75, label='Logistic Regression')
ax[0].plot(np.arange(1, len(lr_map3_scores)+1), lr_map3_scores, color='red', linewidth=0.5, alpha=0.5)

ax[0].set_ylim(bottom=0.15, top=0.4)
ax[0].set_title('MAP@3 Scores')
ax[0].grid()
ax[0].legend()

# (b) Importance scores
lgbm_importance = mean_feature_importance(lgbm_models, name='lgbm', columns=X.columns)
xgb1_importance = mean_feature_importance(xgb1_models, name='xgb1')
xgb2_importance = mean_feature_importance(xgb2_models, name='xgb2')

features = list(xgb1_importance.keys())
x = np.arange(len(features))  

xgb1_vals = np.array([xgb1_importance[f] for f in features])
xgb2_vals = np.array([xgb2_importance[f] for f in features])
lgbm_vals = np.array([lgbm_importance[f] for f in features])

w = 0.25  # bar width

rects1 = ax[1].bar(x - w, xgb1_vals, width=w, label='XGBoost 1', color='orange')
rects2 = ax[1].bar(x,      xgb2_vals, width=w, label='XGBoost 2', color='green')
rects3 = ax[1].bar(x + w,  lgbm_vals, width=w, label='LightGBM',  color='steelblue')

ax[1].set_xticks(x)
ax[1].set_xticklabels(features, rotation=45, ha='right')
ax[1].set_ylabel('Importance (gain)')
ax[1].set_title('Feature Importances: XGBoost vs LightGBM')
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.flatten()

# (a) LGBM Confusion matrix
lgbm_cf_matrix = confusion_matrix(np.array(y, dtype=np.uint8), np.argmax(lgbm_oof_preds, axis=1))
sns.heatmap(lgbm_cf_matrix, cmap='Blues', ax=ax[0], annot=True, fmt='d')
ax[0].set_title('Confusion matrix (LGBM)')

# (b) XGB1 Confusion matrix
xgb1_cf_matrix = confusion_matrix(np.array(y, dtype=np.uint8), np.argmax(xgb1_oof_preds, axis=1))
sns.heatmap(xgb1_cf_matrix, cmap='Blues', ax=ax[1], annot=True, fmt='d')
ax[1].set_title('Confusion matrix (XGBoost 1)')

# (c) XGB2 Confusion matrix
xgb2_cf_matrix = confusion_matrix(np.array(y, dtype=np.uint8), np.argmax(xgb2_oof_preds, axis=1))
sns.heatmap(xgb2_cf_matrix, cmap='Blues', ax=ax[2], annot=True, fmt='d')
ax[2].set_title('Confusion matrix (XGBoost 2)')

# (d) Logistic Regression Confusion matrix
lr_cf_matrix = confusion_matrix(np.array(y, dtype=np.uint8), np.argmax(lr_oof_preds, axis=1))
sns.heatmap(lr_cf_matrix, cmap='Blues', ax=ax[3], annot=True, fmt='d')
ax[3].set_title('Confusion matrix (Logistic Regression)')

plt.tight_layout()
plt.show()


def encode_column(df, col):
    return pd.concat([
        df.drop(col, axis=1),
        df[col].map({k: i for i, k in enumerate(d[col].unique())})
    ], axis=1)


def encode_categorics(df):
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].map({k: i for i, k in enumerate(d[col].unique())})
    return df


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.poly_nutrient = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False) 
        self.poly_env = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.soil_agg = (
            X.groupby('Soil Type')[['Nitrogen', 'Potassium', 'Phosphorous']]
            .mean()
            .add_prefix('SoilMean_')
            .reset_index()
        )

        self.crop_agg = (
            X.groupby('Crop Type')[['Nitrogen', 'Potassium', 'Phosphorous']]
            .median()
            .add_prefix('CropMedian_')
            .reset_index()
        )

        self.poly_nutrient.fit(X[['Nitrogen', 'Potassium', 'Phosphorous']]) 
        self.poly_env.fit(X[['Temparature', 'Humidity', 'Moisture']])

        return self


    def transform(self, X):
        X = X.copy()
        
        # Add ratio-features
        X['N_to_P'] = X['Nitrogen'] / (X['Phosphorous'] + 1e-6)
        X['K_to_N'] = X['Potassium'] / (X['Nitrogen'] + 1e-6)
        X['P_to_K'] = X['Phosphorous'] / (X['Potassium'] + 1e-6) 

        X['Temp_to_Humidity'] = X['Temparature'] * X['Humidity']
        X['Temp_to_Moisture'] = X['Temparature'] * X['Moisture']
        X['Humidity_to_Moisture'] = X['Humidity'] * X['Moisture']
        
        # Add aggreations
        #X = X.merge(self.soil_agg, on='Soil Type', how='left')
        #X = X.merge(self.crop_agg, on='Crop Type', how='left')

        # Add nonlinearity: [Nitrogen, Potassium, Phosphorous]
        X_poly_nutrient = self.poly_nutrient.transform(X[['Nitrogen', 'Potassium', 'Phosphorous']])
        X_poly_env = self.poly_env.transform(X[['Temparature', 'Humidity', 'Moisture']])

        X = X.drop(['Nitrogen', 'Potassium', 'Phosphorous', 
                   'Temparature', 'Humidity', 'Moisture'], axis=1)

        X = pd.concat([
            X, 
            pd.DataFrame(X_poly_nutrient, 
                        columns=self.poly_nutrient.get_feature_names_out(['Nitrogen', 'Potassium', 'Phosphorous']),
                        index=X.index),
            pd.DataFrame(X_poly_env, 
                        columns=self.poly_env.get_feature_names_out(['Temparature', 'Humidity', 'Moisture']),
                        index=X.index)
        ], axis=1)
    
        self.feature_names_ = X.columns.tolist() 
        return X.drop('id', axis=1)

    def get_feature_names_out(self):
        return self.feature_names_


numeric_transformer = Pipeline(steps=[                        # Improved Pipeline
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categoric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one-hot-encoder', OneHotEncoder(handle_unknown='ignore'))
])

column_transformer = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('cat', categoric_transformer, make_column_selector(dtype_include=object))
    ]
)

pipeline = Pipeline(
    steps=[
        ('feature-engineering', FeatureEngineering()),
        ('column-transform', column_transformer)
    ]
)


top_3_preds = np.argsort(lr_test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)


submission['Fertilizer Name'] = [" ".join(label) for label in top_3_labels]
submission.to_csv("submission.csv", index=False)

