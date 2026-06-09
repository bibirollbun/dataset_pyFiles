#basics
import numpy as np
import pandas as pd 
import polars as pl
import seaborn as sns
import time
import matplotlib.pyplot as plt
import missingno as msno
pd.set_option('display.max_columns', 100)
# %load_ext cudf.pandas

import warnings
warnings.filterwarnings("ignore")

#preprocessing
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, LabelEncoder,OneHotEncoder, OrdinalEncoder

#feature engineering
from sklearn.feature_selection import mutual_info_classif


#transformers and pipeline
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from imblearn.over_sampling import SMOTE
from sklearn import set_config
from sklearn.base import clone

#algorithms
from xgboost import XGBClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from catboost import Pool
from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from lightgbm import log_evaluation
from cuml.svm import SVC, LinearSVC


#model evaluation
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold, GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, auc, accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer, RocCurveDisplay, confusion_matrix

# Optuna and visualization tools
import optuna
from optuna.samplers import TPESampler

random_state = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
print('train shape = ', train.shape)
orig_cols = train.columns[1:-1]
train.tail()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print('test shape = ', test.shape)
test.head()


rainfall = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
print("rainfall shape = ", rainfall.shape)
rainfall.head()


rainfall['rainfall'] = [1 if i == 'yes' else 0 for i in rainfall['rainfall']]
rainfall.head()


del train['id']
del test['id']
rainfall.columns = rainfall.columns.str.strip()
print(rainfall.columns)
print(train.columns)
orig_cols = [i for i in orig_cols if i !='day']
print(orig_cols)


train_df = pd.concat([train, rainfall], ignore_index=True, axis = 0)

plt.figure(figsize=(10, 10))
palette_color = sns.color_palette('pastel')
explode = [0.05, 0.05]

# Plotting
train_df.groupby('rainfall')['rainfall'].count().plot.pie(
    colors=palette_color,
    explode=explode,
    autopct="%1.1f%%",
    shadow=True,  # Adding shadow for better visibility
    startangle=140,  # Start angle for better alignment
    textprops={'fontsize': 14},  # Adjust text size
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}  # Adding edge color and width
)

# Adding a title
plt.title('Target Distribution', fontsize=18, weight='bold')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Displaying the plot
plt.show()


feature = [i for i in train_df.columns if i != 'rainfall']


train_df['winddirection'].fillna(train_df['winddirection'].median(), inplace=True)
train_df['windspeed'].fillna(train_df['windspeed'].median(), inplace=True)
test['winddirection'].fillna(test['winddirection'].median(), inplace=True)
test['windspeed'].fillna(test['windspeed'].median(), inplace=True)


mutual_info = mutual_info_classif(train_df[feature], train_df.rainfall, random_state=random_state)
mutual_info_series = pd.Series(mutual_info)
mutual_info_series.index = feature  
mutual_info_df = pd.DataFrame(mutual_info_series.sort_values(ascending=False), columns=["Numerical_Feature_MI"])
styled_mutual_info = mutual_info_df.style.background_gradient("cool")
styled_mutual_info


def data_engineering(df):    
    
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    
    df['tan_day'] = np.tan(2 * np.pi * df['day'] / 365)
    
    df['arcsin_day'] = np.arcsin(np.clip(df['day'] / 365, -1, 1))
    df['arccos_day'] = np.arccos(np.clip(df['day'] / 365, -1, 1))
    df['arctan_day'] = np.arctan(df['day'] / 365)
    def get_season(day):
        if day >= 335 or day <= 59:  
            return '0' #winter
        elif 60 <= day <= 151:  
            return '1' #spring
        elif 152 <= day <= 243:  
            return '2'#summer
        else:  
            return '3' #autumn
    df['season'] = df['day'].apply(get_season).astype('int8')
    return df


def temp_engineering(df):
    df['temp_diff'] = df['maxtemp'] - df['mintemp']
    df['dew_temp_diff'] = df['dewpoint'] - df['temparature']
    return df


def cloud(df):
    df['humidity_windspeed'] = df['humidity'] * df['windspeed']
    df['cloud + humidity'] = df['cloud'] + df['humidity']
    df['cloud + humidity + sunshine'] = df['cloud'] + df['humidity'] + df['sunshine']
    df['cloud * sunshine'] = df['cloud'] * df['sunshine']
    df['humidity * sunshine'] = df['humidity'] * df['sunshine']
    df['cloud^2'] = df['cloud'] ** 2
    return df


def mean_c(df):
    pass 
# this function will be written later, it will be used in kfold


train_df = cloud(train_df)
test = cloud(test)


train_df = temp_engineering(train_df)
test = temp_engineering(test)


train_df = data_engineering(train_df)
test = data_engineering(test)


for i,c1 in enumerate(orig_cols):
    for j,c2 in enumerate(orig_cols[i+1:]):
        n = f"{c1}_{c2}"
        m1 = train[c1].max()+1
        m2 = train[c2].max()+1
        train_df[n] = ((train_df[c1]+1 + (train_df[c2]+1)/(m2+1))*(m2+1))
        test[n] = ((test[c1]+1 + (test[c2]+1)/(m2+1))*(m2+1))


features = [i for i in train_df.columns if i != 'rainfall']
test[features].head()


smote = SMOTE(random_state=42)

print(f'shape before: {train_df.shape}')

X_resampled, y_resampled = smote.fit_resample(train_df[features], train_df['rainfall'])

resampled_df = pd.DataFrame(X_resampled, columns=features)
resampled_df['rainfall'] = y_resampled

train_df = resampled_df.copy()
print(f'shape after: {train_df.shape}')


FOLDS = 7

kf =KFold(n_splits=FOLDS, shuffle=True, random_state=42)

def kfold(model,data, test_data):
    
    x = data[features].copy()
    y = data['rainfall'].copy()
    
    oof_preds = np.zeros(len(x))
    test_preds = np.zeros(len(test_data))
    cv_in = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}")
    
        X_train = x.loc[train_idx].reset_index(drop=True).copy()
        y_train = y.iloc[train_idx].values
        
        X_valid = x.loc[valid_idx].reset_index(drop=True).copy()
        y_valid = y.iloc[valid_idx].values
        
        X_test = test.reset_index(drop=True).copy()

        model_cloned = clone(model)

        if isinstance(model_cloned, CatBoostClassifier):
            train_pool = Pool(X_train, y_train)
            valid_pool = Pool(X_valid, y_valid)
            X_test_pool = Pool(X_test)
            model_cloned.fit(X=train_pool, eval_set=valid_pool, verbose=50, early_stopping_rounds=100)

            oof_preds[valid_idx] = model_cloned.predict_proba(X_valid)[:, 1]
            test_preds += model_cloned.predict_proba(test)[:, 1] / FOLDS

            cv_in.append(roc_auc_score(data['rainfall'], oof_preds))
        if isinstance(model_cloned, XGBClassifier):
            model_cloned.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=50)
            best_iteration = model_cloned.best_iteration
        
            oof_preds[valid_idx] = model_cloned.predict_proba(X_valid, iteration_range=(0, best_iteration))[:, 1]
            test_preds += model_cloned.predict_proba(test, iteration_range=(0, best_iteration))[:, 1] / FOLDS
            
            cv_in.append(roc_auc_score(data['rainfall'], oof_preds))
        elif isinstance(model_cloned, LGBMClassifier):
             eval_set = [(X_valid, y_valid)]  
             model_cloned.fit(
                 X_train, y_train,
                 eval_set=eval_set,   
                 eval_metric='auc',  
                 callbacks=[early_stopping(50)]  
             )
             best_iteration = model_cloned.best_iteration_  
             oof_preds[valid_idx] = model_cloned.predict_proba(X_valid)[:, 1]
             test_preds += model_cloned.predict_proba(X_test)[:, 1] / FOLDS
             cv_in.append(roc_auc_score(data['rainfall'], oof_preds))
        elif isinstance(model_cloned, LinearSVC):
            meta_model.fit(Oof_preds[models].values, Oof_preds.iloc[:,-1].values)
            finall_preds = meta_model.predict_proba(test_preds.values)[:, -1]
        print("--" * 25)
    
    ras = roc_auc_score(data['rainfall'], oof_preds)
    print(f"Validation RAS: {ras}")
    return test_preds, oof_preds, cv_in


train_df = train_df.reset_index(drop=True)
test = test.reset_index(drop=True)


test_preds, Oof_preds, cv_info = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


catboost_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': 0.09613777604618812,
        'iterations': 200,
        'depth': 11,
        'random_strength':0,
        'l2_leaf_reg': 7.9815276045005765,
        'task_type':'GPU',
        'random_seed':42,
        'verbose':False    
    }


model_cbc = CatBoostClassifier(**catboost_params)
test_preds['cbc'], Oof_preds['cbc'], cv_info['cbc']= kfold(model_cbc, train_df, test)


# x_train, x_test, y_train, y_test = train_test_split(train_df[feature], train_df['rainfall'], random_state = 42)

# def objective(trial):
#     params = {
#         'n_estimators' : trial.suggest_int('n_estimators', 50, 500),
#         'eta' : trial.suggest_float('eta', 0.001, 0.5),
#         'alpha' : trial.suggest_float('alpha', 0.0, 1.0),
#         'subsample' : trial.suggest_float('subsample', 0.1, 1.0),
#         'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.1, 1.0),
#         'max_depth' : trial.suggest_int('max_depth', 1, 10),
#         'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
#         'gamma' : trial.suggest_float('gamma', 0.0, 5.0),
#         'max_bin' : trial.suggest_int('max_bin', 128, 25000),
#         'tree_method': 'gpu_hist',
#         'eval_metric': 'auc',
#         'objective': 'binary:logistic',
#         'verbose' : 0
#     }

#     model = XGBClassifier(**params)
#     model.fit(x_train, y_train)
#     preds = model.predict_proba(x_test)[:, 1]  
#     roc_auc = roc_auc_score(y_test, preds)

#     return roc_auc
# sampler = TPESampler()
# study = optuna.create_study(sampler=sampler, direction='maximize')

# study.optimize(objective, n_trials=200)
# print("Лучшие гиперпараметры:", study.best_params)
# print("Лучшая AUC:", study.best_value)


xgbc_par = {'n_estimators': 159,
            'eta': 0.022188417802067064,
            'alpha': 0.2292689574688371,
            'subsample': 0.32761784160414953,
            'colsample_bytree': 0.26576926056157896,
            'max_depth': 6,
            'min_child_weight': 2,
            'gamma': 0.6564905675241042,
            'max_bin': 17560,
            'tree_method': 'gpu_hist',
            'eval_metric': 'auc',
            'objective': 'binary:logistic'
}


model_xg = XGBClassifier(**xgbc_par, verbose = 50)
test_preds['xgbc'], Oof_preds['xgbc'], cv_info['xgbc'] = kfold(model_xg, train_df, test)


# x_train, x_test, y_train, y_test = train_test_split(train_df[feature], train_df['rainfall'], random_state = 42)
# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 500),
#         'max_depth': trial.suggest_int('max_depth', 1, 15),
#         'num_leaves': trial.suggest_int('num_leaves', 5, 400),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
#         'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
#         'random_state': random_state,
#         "eval_metric": "auc",
#         'verbose': -1,
#     }
#     model = LGBMClassifier(**params)
#     scores = cross_val_score(model, x_train, y_train, cv=5, scoring='roc_auc')
#     auc_score = scores.mean()
    
#     return auc_score

# sampler = TPESampler(seed=random_state)
# study = optuna.create_study(direction='maximize', sampler=sampler)
# study.optimize(objective, n_trials=100)
# print("Лучшие гиперпараметры:", study.best_params)
# print("Лучшая AUC:", study.best_value)


l_par = {'n_estimators': 232,
         'max_depth': 1,
         'num_leaves': 303,
         'learning_rate': 0.25101263781938554,
         'min_child_samples': 14,
         'subsample': 0.746858325807502,
         'colsample_bytree': 0.9082677247457509,
         'lambda_l1': 2.142983876868706e-06,
         'lambda_l2': 2.487359042523012,
         'random_state': random_state,
         "eval_metric": "auc",
          'verbose': -1,
        }


model_lgb = LGBMClassifier(**l_par)
test_preds['lgbmc'], Oof_preds['lgbmc'], cv_info['lqbmc'] = kfold(model_lgb, train_df, test)


# from tensorflow import keras
# from tensorflow.keras import layers


# def create_model(input_shape):
#     inputs = []
#     flat_embeddings = []

#     for feature, input_dim in input_shape.items():
#         output_dim = min(64, round(1.6 * (input_dim + 1) ** 0.56))  
#         input_layer = keras.Input(shape=(1,), name=feature)
#         embedding_layer = layers.Embedding(input_dim=input_dim + 1, output_dim=output_dim)(input_layer)  
#         embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
#         embedding_layer = layers.Flatten()(embedding_layer)
#         inputs.append(input_layer)
#         flat_embeddings.append(embedding_layer)
    
#     numerical_input = keras.Input(shape=(len(numerical_features),), name='numerical')
#     inputs.append(numerical_input)

#     concatenated_inputs = layers.Concatenate()(flat_embeddings + [numerical_input])
#     concatenated_inputs_bn = layers.BatchNormalization()(concatenated_inputs)

#     x = layers.Dense(256, activation='mish')(concatenated_inputs_bn)
#     x = layers.BatchNormalization()(x)
#     x = layers.Concatenate()([x, concatenated_inputs_bn])
#     x = layers.Dense(128, activation='mish')(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.BatchNormalization()(x)

#     outputs = layers.Dense(1, activation='sigmoid')(x)

#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model


# def nn_kfold(model,data, test_data):
   
#     x = data[features].copy()
#     y = data['rainfall'].copy()
    
#     oof_preds = np.zeros(len(x))
#     test_preds = np.zeros(len(test_data))
#     cv_in = []
#     for fold, (train_idx, valid_idx) in enumerate(kf.split(data)):
    
#         print(f"Fold {fold + 1}")
#         X_train = X.iloc[train_idx].reset_index(drop=True)
#         y_train = y.iloc[train_idx].reset_index(drop=True)
#         X_val = X.iloc[val_idx].reset_index(drop=True)
#         y_val = y.iloc[val_idx].reset_index(drop=True)
        
#         if include_original:
#             if original_data is None:
#                 raise ValueError("original_data must be provided when include_original is True")
#             X_train = pd.concat([original_data.drop(label, axis=1), X_train]).reset_index(drop=True)
#             y_train = pd.concat([original_data[label], y_train]).reset_index(drop=True)
        
#         model = model_builder(input_shape)
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#                       loss='binary_crossentropy',
#                       metrics=[keras.metrics.AUC(name='auc')])
        
#         X_train_inputs = {feature: X_train[feature].values for feature in categorical_features}
#         X_train_inputs['numerical'] = X_train[numerical_features].values
        
#         X_val_inputs = {feature: X_val[feature].values for feature in categorical_features}
#         X_val_inputs['numerical'] = X_val[numerical_features].values
        
#         model.fit(X_train_inputs, y_train, epochs=4, batch_size=1024, validation_data=(X_val_inputs, y_val), verbose=0)

#         val_preds_proba = model.predict(X_val_inputs).flatten()
#         val_predictions[val_idx] = val_preds_proba
#         val_scores.append(roc_auc_score(y_val, val_preds_proba))
        
#         # Predict on test data
#         test_inputs = {feature: test_data[feature].values for feature in categorical_features}
#         test_inputs['numerical'] = test_data[numerical_features].values
#         test_preds_proba = model.predict(test_inputs).flatten()
#         test_predictions += test_preds_proba / cv.get_n_splits()
        
#         print(f'Fold {fold}: {val_scores[-1]:.5f}')
    
#     print(f'Val Score: {np.mean(val_scores):.7f} ± {np.std(val_scores):.7f} | {label}')
    
#     return val_scores, val_predictions, test_predictions


# test_preds['nn'], Oof_preds['nn'], cv_info['nn']= nn_kfold(model_cbc, train_df, test)


transposed_df = cv_info.transpose()
# transposed_df.columns = ['fold1','fold2']
transposed_df.columns = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7']
transposed_df['Mean'] = transposed_df.mean(axis=1)
transposed_df['Std'] = transposed_df.std(axis=1)
transposed_df.sort_values(by = 'Mean', ascending=False).style.background_gradient('Dark2_r')
transposed_df


sns.set(font_scale=1.2, style="whitegrid")
correlation_train = Oof_preds.corr()
mask = np.triu(np.ones_like(correlation_train, dtype=bool))
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_train, 
            mask=mask, 
            annot=True, 
            fmt='.3f', 
            cmap='coolwarm', 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .75})
plt.title('Model Diversity Check - Correlation Heatmap', fontsize=20, pad=20)
plt.show()


Oof_preds['rainfall'] = train_df['rainfall']
Oof_preds


params = {'n_estimators': 205,
          'max_depth': 1,
          'num_leaves': 343,
          'learning_rate': 0.028636846317455077,
          'min_child_samples': 54,
          'subsample': 0.6519948792121483,
          'colsample_bytree': 0.9575417355687003,
          'lambda_l1': 1.6223574456155055e-07, 
          'lambda_l2': 2.7747351951775148,
         'random_state': random_state,
         "eval_metric": "auc",
          'verbose': -1,}


models = Oof_preds.columns[:-1]


finall_preds, finall_oof = pd.DataFrame(), pd.DataFrame()
Oof_preds[models].head()


def finall_kfold(model, data, target, test_data):
    oof_preds = np.zeros(len(data))
    test_preds = np.zeros(len(test_data))
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(data)):
        print(f"Fold {fold + 1}")
        
        X_train = data[train_idx]
        y_train = target[train_idx]
        
        X_valid = data[valid_idx]
        y_valid = target[valid_idx]
        
        model_cloned = clone(model)
        model_cloned.fit(X_train, y_train)
        
        oof_preds[valid_idx] = model_cloned.predict_proba(X_valid)[:, 1]
        test_preds += model_cloned.predict_proba(test_data)[:, 1] / FOLDS
    
    return test_preds, oof_preds


meta_model = LinearSVC(C=0.1, probability=True)
finall_preds, finall_oof = finall_kfold(
    meta_model,
    Oof_preds[models].values,  
    Oof_preds.iloc[:, -1].values,  
    test_preds.values  
)


x_train, x_test, y_train, y_test = train_test_split(Oof_preds[models].values, Oof_preds.iloc[:, -1].values, random_state = 42)
print(f'roc auc x_train: {roc_auc_score(y_train, x_train[:, 1])}')
print(f'roc auc x_test: {roc_auc_score(y_test, x_test[:, 1])}')


sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
sub.rainfall = finall_preds
sub.to_csv("submission.csv", index=False)
!head submission.csv




