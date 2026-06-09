import warnings
# отключаем вывод предупреждений в консоль (чтобы она не засорялась)
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd
# библиотека для визуализации данных
import seaborn as sns 
import matplotlib.pyplot as plt
# библиотека для статистического анализа данных (статистических тестов)
from scipy import stats

# библиотека для градиентного бустинга
from xgboost import XGBClassifier
# библиотека для градиентного бустинга
from lightgbm import LGBMClassifier
# библиотека для градиентного бустинга и работы с категориальными данными
from catboost import CatBoostClassifier
# матрика для измерения качества моделей
from sklearn.metrics import roc_auc_score
# метод make_pipline создает конвейеры (объединяет шаги предобработки (кодирование признаков) и модель в единый процесс)
from sklearn.pipeline import make_pipeline
# кодирование категориальных признаков
from category_encoders import TargetEncoder
# оптимизация гиперпараметров или ансамблей
from hillclimbers import climb_hill, partial
# нейронная сеть - модель для классификации
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# ансамблевый метод для классификации
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
# оценка калибровки вероятности модели
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# инструмент для кросс-валидации
from sklearn.model_selection import StratifiedKFold
# вычисление взаимной информации между признаками и целевой переменной
from sklearn.feature_selection import mutual_info_classif


!pip install -q hillclimbers


train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv', index_col='id')
original = pd.read_csv('/kaggle/input/loan-approval-prediction/credit_risk_dataset.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')


train.info()


test.info()


original.info()


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
original[['person_emp_length', 'loan_int_rate']] = imputer.fit_transform(original[['person_emp_length', 'loan_int_rate']])


original.info()


merged = pd.concat([train.drop(columns=['loan_status']), test], axis=0)
duplicates = merged[merged.duplicated(keep=False)]  

print(f"Полных дубликатов между train и test: {len(duplicates)}")
print(f"Длина train: {len(train.values)}")
print(f"Длина test: {len(test.values)}")
print(f"Длина original: {len(original.values)}")


merged = pd.concat([train, original], axis=0)
duplicates1 = merged[merged.duplicated(keep=False)]  

print(f"Полных дубликатов между train и original: {len(duplicates1)}")
print(f"Длина train: {len(train.values)}")
print(f"Длина test: {len(test.values)}")
print(f"Длина original: {len(original.values)}")


# удалим дкбликаты из original
original = pd.concat([original, duplicates1]).drop_duplicates(keep=False)
print(f"Длина train: {len(train.values)}")
print(f"Длина test: {len(test.values)}")
print(f"Длина original: {len(original.values)}")


grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
train['loan_int_rate_grade'] = train['loan_int_rate'] * train['loan_grade'].map(grade_mapping)
test['loan_int_rate_grade'] = test['loan_int_rate'] * test['loan_grade'].map(grade_mapping)
original['loan_int_rate_grade'] = original['loan_int_rate'] * original['loan_grade'].map(grade_mapping)
train['emp_length_age_ratio'] = train['person_emp_length'] / (train['person_age'] + 1e-6)  
test['emp_length_age_ratio'] = test['person_emp_length'] / (test['person_age'] + 1e-6)
original['emp_length_age_ratio'] = original['person_emp_length'] / (original['person_age'] + 1e-6)


# grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
# train['loan_grade'] = train['loan_grade'].map(grade_mapping)
# test['loan_grade'] = test['loan_grade'].map(grade_mapping)
# original['loan_grade'] = original['loan_grade'].map(grade_mapping)


# grade_mapping = {'Y': 1, 'N': 0}
# train['cb_person_default_on_file'] = train['cb_person_default_on_file'].map(grade_mapping)
# test['cb_person_default_on_file'] = test['cb_person_default_on_file'].map(grade_mapping)
# original['cb_person_default_on_file'] = original['cb_person_default_on_file'].map(grade_mapping)


original.info()


oof = {}
test_pred = {}
NUM_FOLD = 5
target = 'loan_status'


def cross_validation(model, label):
    
    train_copy, test_copy, original_copy = train.copy(), test.copy(), original.copy()
    
    if label in ['cb', 'et', 'rf', 'knn', 'mlp']:
        cat_cols = test_copy.columns.tolist()
        for df in [train_copy, test_copy, original_copy]:
            for col in cat_cols:  
                df[col] = df[col].astype('str').astype('category')
        
    elif label in ['xgb', 'lgbm', 'dart', 'goss', 'hgb']: 
        cat_cols = list(test_copy.select_dtypes(include=['object']).columns)
        for df in [train_copy, test_copy, original_copy]:
            for col in cat_cols:  
                df[col] = df[col].astype('str').astype('category')
                
    
    X = train_copy.drop([target], axis=1)
    y = train_copy[target]
    X_original = original_copy.drop([target], axis=1)
    y_original = original_copy[target]
        

    val_scores = []
    test_preds_model = []
    oof_model = np.zeros(len(train),)
    
    skf = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=1)

    for Fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train = pd.concat([X_train, X_original], axis=0)

        y_train = pd.concat([y_train, y_original]) 

    
        model.fit(X_train, y_train)
    
        y_pred = model.predict_proba(X_val)[:, 1]
    
        roc_auc_score_ = roc_auc_score(y_val, y_pred)
    
        print(f'Fold {Fold}: roc_auc_score= {roc_auc_score_:.5f}')
    
        val_scores.append(roc_auc_score_)
        
        oof_model[val_index] = y_pred
        
        test_preds_model.append(model.predict_proba(test_copy)[:, 1])
    # сохраняем предсказания конкретной модели (на валидационном наборе(на каждом фолде) - т.е. она будет полностью заполнена)
    oof[label] = oof_model
    # усредняем предсказания на тестовом наборе по всем фолдам
    test_preds_model = sum(test_preds_model)/len(test_preds_model)
    test_pred[label] = test_preds_model 

    print(f'mean validation roc_auc_score = {np.mean(val_scores):.5f}')
    print(f'std validation roc_auc_score = {np.std(val_scores):.5f}')
    
    plt.figure(figsize=(10, 4))
    plt.suptitle(label, y=1.0, fontsize=20)
    ax = plt.subplot(1, 2, 1)
    plt.title('Calibration')
    CalibrationDisplay.from_predictions(y, oof_model, n_bins=10, strategy='quantile', ax=ax)
    plt.subplot(1, 2, 2)
    plt.title('Histogram')
    plt.hist(oof_model, bins=10)
    plt.show()


%%time

params_lgbm = {
    
    'verbose': -1,
    'random_state': 1,
    'objective': 'binary',
    'n_estimators': 4100,
    'learning_rate': 0.01,
    'colsample_bytree': 0.6,
    'max_depth': 8,
    'max_bin': 5000,
}

model_1 = LGBMClassifier(**params_lgbm)

cross_validation(model_1, 'lgbm')


%%time

params_dart = {
    
    'verbose': -1,
    'random_state': 1,
    'boosting': 'dart',
    'n_estimators': 600,
    'learning_rate': 0.1,
    'colsample_bytree': 0.6,
    'num_leaves': 85,
    'min_data_in_leaf': 30,
    'max_bin': 1995,
    'objective': 'binary',
}

model_2 = LGBMClassifier(**params_dart)

cross_validation(model_2, 'dart')


%%time

params_goss_boosting = {
    
    'verbose': -1,
    'random_state': 1,
    'boosting': 'goss',
    'n_estimators': 600,
    'learning_rate': 0.1,
    'colsample_bytree': 0.6,
    'num_leaves': 85,
    'min_data_in_leaf': 30,
    'max_bin': 1995,
    'objective': 'binary',
}

model_11 = LGBMClassifier(**params_goss_boosting)

cross_validation(model_11, 'goss_b')


%%time

params_xgb = {
    
    'enable_categorical': True,
    'random_state': 1,
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'colsample_bytree': 0.6,
    'reg_lambda': 0.01,
    'max_depth': 4,
    'max_bin': 5000,
    'subsample': 0.95,
    'reg_alpha': 0.1,
 
}

model_3 = XGBClassifier(**params_xgb)

cross_validation(model_3, 'xgb')


%%time

params_et = {
    
    'random_state': 1,
    'n_estimators': 470,
    'min_samples_leaf': 1,
    'max_depth': 20,
    'criterion': 'log_loss',
}

model_4 = make_pipeline(TargetEncoder(), ExtraTreesClassifier(**params_et))

cross_validation(model_4, 'et')


%%time

params_rf = {
    
    'random_state': 1,
    'n_estimators': 450,
    'min_samples_leaf': 5,
    'max_leaf_nodes': 960,
    'criterion': 'entropy',
}

model_5 = make_pipeline(TargetEncoder(), RandomForestClassifier(**params_rf))

cross_validation(model_5, 'rf')


%%time

params_cb = {
    
    'verbose': False,
    'random_state': 1,
    'task_type': 'CPU',
    'cat_features' : test.columns.tolist(),
    'min_data_in_leaf': 5,
    'n_estimators': 1800,
    'random_strength': 0.79,
    'depth': 8,
    'bagging_temperature': 0.6,
    'l2_leaf_reg': 4,
    'rsm': 0.6,
}

model_6 = CatBoostClassifier(**params_cb)

cross_validation(model_6, 'cb')


%%time

model_7 = make_pipeline(TargetEncoder(), KNeighborsClassifier(n_neighbors=185, 
                                                              metric='manhattan',
                                                              weights='distance'))

cross_validation(model_7, 'knn')


%%time

params_mlp = {
    
    'random_state': 1,
    'hidden_layer_sizes': (32, 3),
    
}

model_8 = make_pipeline(TargetEncoder(), StandardScaler(), MLPClassifier(**params_mlp))

cross_validation(model_8, 'mlp')


%%time

params_goss = {
    
    'verbose': -1,
    'random_state': 1,
    'data_sample_strategy': 'goss',
    'n_estimators': 4000,
    'learning_rate': 0.01,
    'colsample_bytree': 0.6,
    'max_depth': 17,
    'max_bin': 4000,
}

model_9 = LGBMClassifier(**params_goss)

cross_validation(model_9, 'goss')


ensemble_fold_scores = []
ensemble_test_preds = []

y_ensemble = train[target]
X_ensemble = pd.DataFrame(oof)    
x_test_ensemble = pd.DataFrame(test_pred)

skf = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=1)

for i, (train_index_ens, val_index_ens) in enumerate(skf.split(X_ensemble, y_ensemble)):
    
    X_train_ens, X_val_ens = X_ensemble.iloc[train_index_ens], X_ensemble.iloc[val_index_ens]
    y_train_ens, y_val_ens = y_ensemble[train_index_ens], y_ensemble[val_index_ens]
    
    lr = LogisticRegression().fit(X_train_ens, y_train_ens)
    
    ensemble_val_pred = lr.predict_proba(X_val_ens)[:, 1] 
    
    ensemble_roc_auc_score = roc_auc_score(y_val_ens, ensemble_val_pred)
    ensemble_fold_scores.append(ensemble_roc_auc_score)
    
    print('Fold', i, '==> roc_auc_score (LR ensemble) is ==>', ensemble_roc_auc_score)
    
    ensemble_test_preds.append(lr.predict_proba(x_test_ensemble)[:, 1])
    
ensemble_test_preds = sum(ensemble_test_preds)/len(ensemble_test_preds)

print(f'\nCV roc_auc_score = {np.mean(ensemble_fold_scores):.5f}')
print(f'\nstd roc_auc_score = {np.std(ensemble_fold_scores):.5f}')


hc_test, hc_oof = climb_hill(train=train, target=target, objective='maximize', 
                             eval_metric=partial(roc_auc_score),oof_pred_df= X_ensemble, 
                             test_pred_df= x_test_ensemble,plot_hill=False,plot_hist=False, 
                             precision=0.001,negative_weights=True,return_oof_preds=True)


sample_submission[target] = hc_test
sample_submission.head()
sample_submission.to_csv('/kaggle/working/submission.csv', index=False) 

