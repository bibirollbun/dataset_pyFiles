# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from scipy import stats
from scipy.optimize import minimize
from scipy.stats import mstats

import catboost
from catboost import CatBoostClassifier
from catboost.utils import get_fnr_curve, get_fpr_curve, get_roc_curve

import lightgbm as lgb
import xgboost as xgb

from mlxtend.classifier import StackingCVClassifier

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier, VotingClassifier)
from sklearn.feature_selection import (SelectKBest, RFECV, chi2,
                                       VarianceThreshold, SequentialFeatureSelector)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             log_loss, roc_curve, roc_auc_score, f1_score)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import (KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import (LabelEncoder, QuantileTransformer, StandardScaler,
                                   PowerTransformer, MaxAbsScaler, MinMaxScaler,
                                   RobustScaler, PolynomialFeatures, OrdinalEncoder,
                                   OneHotEncoder, FunctionTransformer, KBinsDiscretizer)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from category_encoders import TargetEncoder, MEstimateEncoder
# from cuml.preprocessing import TargetEncoder

# from imblearn.over_sampling import (SMOTE, ADASYN,
#                                     BorderlineSMOTE, RandomOverSampler,
#                                     KMeansSMOTE)
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import make_pipeline, Pipeline

import optuna

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras

mpl.rcParams.update(mpl.rcParamsDefault)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

%matplotlib inline


def plot_numerical_features(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].set_ylabel('Frequency', fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)  

        mean_value = df[feature].mean()
        axes[i].axvline(mean_value, color='red', linestyle='--', label='Mean')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_numerical_boxplots(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        sns.boxplot(x=df[feature], ax=axes[i], color='lightgreen')
        axes[i].set_title(f'Boxplot of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=14)
        axes[i].grid(True, linestyle='--', alpha=0.7)  

        median_value = df[feature].median()
        axes[i].axvline(median_value, color='orange', linestyle='--', label='Median')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def plot_qq_plot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    ncols = 2
    nrows = (len(num_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(num_features):
        stats.probplot(df[feature], dist="norm", plot=axes[i])
        axes[i].set_title(f'QQ Plot of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel('Theoretical Quantiles', fontsize=14)
        axes[i].set_ylabel('Sample Quantiles', fontsize=14)
        axes[i].grid(True, linestyle='--', alpha= 0.7)  

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, method='spearman'):
    num_df = df.select_dtypes(include=[np.number])
    
    corr = num_df.corr(method=method)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, linewidths=.5)
    plt.title(f'Correlation Matrix ({method.capitalize()} Correlation)', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def plot_pairplot(df):
    num_features = df.select_dtypes(include=[np.number]).columns
    sns.pairplot(df[num_features], diag_kind='kde', plot_kws={'alpha': 0.6, 'edgecolor': 'k'}, height=2.5)
    plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=18, fontweight='bold')
    plt.show()

def plot_categorical_features(df, ncols=2, top_n=None):
    cat_features = df.select_dtypes(include=[object]).columns
    nrows = (len(cat_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(cat_features):
        if top_n is not None:
            top_categories = df[feature].value_counts().nlargest(top_n).index
            sns.countplot(data=df[df[feature].isin(top_categories)], y=feature, ax=axes[i], palette='viridis', order=top_categories)
        else:
            sns.countplot(data=df, y=feature, ax=axes[i], palette='viridis')
        
        axes[i].set_title(f'Count of {feature}', fontsize=18, fontweight='bold')
        axes[i].set_xlabel('Count', fontsize=14)
        axes[i].set_ylabel(feature, fontsize=14)
        axes[i].tick_params(axis='y', rotation=0)
        axes[i].grid(True, linestyle='--', alpha=0.7)  
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def PolynomialFeatures_labeled(input_df,power):
   
    poly = preprocessing.PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s+%d" % (variable,power)
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + "x" + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df

def variance_threshold(df,th):
    var_thres=VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:,new_cols]
   
def optimize_memory_usage(df, print_size=True):
    """
    Optimizes memory usage in a DataFrame by downcasting numeric columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to optimize.
        print_size (bool): If True, prints memory usage before and after optimization.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    # Types for optimization.
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Memory usage size before optimize (Mb).
    before_size = df.memory_usage().sum() / 1024**2
    
    for column in df.columns:
        column_type = df[column].dtype
        
        if column_type in numerics:
            try:
                if str(column_type).startswith('int'):
                    df[column] = pd.to_numeric(df[column], downcast='integer')
                else:
                    df[column] = pd.to_numeric(df[column], downcast='float')
                logger.info(f"Optimized column {column}: {column_type} -> {df[column].dtype}")
            except Exception as e:
                logger.error(f"Failed to optimize column {column}: {e}")
    
    # Memory usage size after optimize (Mb).
    after_size = df.memory_usage().sum() / 1024**2
    
    if print_size:
        print(
            'Memory usage size: before {:5.4f} Mb - after {:5.4f} Mb ({:.1f}%).'.format(
                before_size, after_size, 100 * (before_size - after_size) / before_size
            )
        )
    
    return df


train = pd.read_csv('/kaggle/input/iml2020-bank/train.csv')
test = pd.read_csv('/kaggle/input/iml2020-bank/test.csv')

display(train.shape, test.shape)
display(train.info(), test.info())

# test = test.drop(['id'], axis =1)
# train = train.drop(['id'], axis =1)

display(train.describe().T)
display(test.describe().T)

duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

duplicates = test.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

display(train.head(5))

for col in test.columns:
    pct_missing = np.mean(test[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


def preprocess_data(train_df, test_df, target_col='плохой_клиент'):
    """
    Обрабатывает train и test данные, сохраняя согласованность преобразований
    """
    # Создаем копии чтобы не изменять оригиналы
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # Сохраняем target если он есть
    if target_col in train_processed.columns:
        y_train = train_processed[target_col].copy()
        train_processed = train_processed.drop(columns=[target_col])
    else:
        y_train = None
    
    # Объединяем для согласованной обработки
    combined = pd.concat([train_processed, test_processed], axis=0)
    combined.reset_index(drop=True, inplace=True)
    
    # 1. Заполнение пропусков
    # Для семьи используем моду (наиболее частое значение)
    if 'семья' in combined.columns:
        family_mode = combined['семья'].mode()[0] if not combined['семья'].mode().empty else 0
        combined['семья'] = combined['семья'].fillna(family_mode)
    
    # Для дохода используем медиану (более стабильно чем среднее)
    if 'доход' in combined.columns:
        income_median = combined['доход'].median()
        combined['доход'] = combined['доход'].fillna(income_median)
    
    # 2. Обработка выбросов
    numeric_cols = ['линии', 'возраст', 'поведение_30-59_дней', 'Debt_Ratio', 
                   'доход', 'число_кредитов', 'поведение_90_дней', 
                   'недвижимость', 'поведение_60-89_дней', 'семья']
    
    numeric_cols = [col for col in numeric_cols if col in combined.columns]
    
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
        Q1 = combined[col].quantile(0.25)
        Q3 = combined[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Заменяем выбросы на граничные значения
        combined[col] = np.where(combined[col] < lower_bound, lower_bound, combined[col])
        combined[col] = np.where(combined[col] > upper_bound, upper_bound, combined[col])
    
    # 3. Создание новых признаков
    # Логарифмирование skewed features
    skewed_cols = ['линии', 'Debt_Ratio', 'доход']
    for col in skewed_cols:
        if col in combined.columns:
            combined[f'log_{col}'] = np.log1p(combined[col] + 1e-6)  # +1e-6 чтобы избежать log(0)
    
    # Возрастные группы
    if 'возраст' in combined.columns:
        combined['возрастная_группа'] = pd.cut(combined['возраст'], 
                                              bins=[0, 30, 45, 60, 100],
                                              labels=['молодой', 'средний', 'старший', 'пенсионер'])
    
    # Отношение долга к доходу
    if all(col in combined.columns for col in ['Debt_Ratio', 'доход']):
        combined['долг_к_доходу'] = combined['Debt_Ratio'] / (combined['доход'] + 1)
    
    # 4. Кодирование категориальных переменных
    if 'возрастная_группа' in combined.columns:
        age_dummies = pd.get_dummies(combined['возрастная_группа'], prefix='возраст')
        combined = pd.concat([combined, age_dummies], axis=1)
        combined = combined.drop(columns=['возрастная_группа'])
    
    # 5. Масштабирование числовых признаков
    scaler = StandardScaler()
    numeric_features = [col for col in numeric_cols if col in combined.columns]
    
    if numeric_features:
        combined[numeric_features] = scaler.fit_transform(combined[numeric_features])
    
    # 6. Разделяем обратно на train и test
    train_size = len(train_processed)
    train_final = combined.iloc[:train_size].copy()
    test_final = combined.iloc[train_size:].copy()
    
    # Возвращаем target
    if y_train is not None:
        train_final[target_col] = y_train.values
    
    # Удаляем временные колонки если они есть
    cols_to_drop = ['возрастная_группа'] if 'возрастная_группа' in train_final.columns else []
    train_final = train_final.drop(columns=cols_to_drop, errors='ignore')
    test_final = test_final.drop(columns=cols_to_drop, errors='ignore')
    
    return train_final, test_final, scaler

def remove_duplicates(df):
    """Удаляет дубликаты и возвращает информацию"""
    initial_size = len(df)
    df_clean = df.drop_duplicates()
    final_size = len(df_clean)
    duplicates_removed = initial_size - final_size
    print(f"Удалено дубликатов: {duplicates_removed}")
    return df_clean

print("\nОбработка train данных:")
train_data = remove_duplicates(train)
print("\nОбработка test данных:")
test_data = remove_duplicates(test)

# Обработка данных
train_processed, test_processed, scaler = preprocess_data(train, test)

print("\nПосле обработки:")
print(f"Train shape: {train_processed.shape}, Test shape: {test_processed.shape}")
print("Пропуски в train:")
print(train_processed.isnull().sum())
print("Пропуски в test:")
print(test_processed.isnull().sum())


X = train_processed.drop(columns=['плохой_клиент'])
y = train_processed['плохой_клиент']

# #not today
X = variance_threshold(X,0.03)
list_name = (X.columns)
test_processed = test_processed[list_name]

display(X.shape, y.shape, test_processed.shape)


def create_ensemble_classification(X, y, test_aligned, n_folds=7):
    """
    Ансамбль моделей для классификации с использованием F1 score
    
    Parameters:
    X (pd.DataFrame): Признаки train
    y (pd.Series): Целевая переменная train
    test_aligned (pd.DataFrame): Признаки test
    n_folds (int): Количество фолдов для кросс-валидации
    
    Returns:
    oof_df, predictions_df, model_info
    """
    FOLDS = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_oof = {}
    all_predictions = {}
    all_predictions_proba = {}  # Для вероятностей
    models = []

    # CatBoost модели 
    models.append(('cat_1', CatBoostClassifier(verbose=0, random_state=42)))
    models.append(('cat_2', CatBoostClassifier(verbose=0, random_state=42, iterations=200)))
    models.append(('cat_3', CatBoostClassifier(verbose=0, random_state=42, depth=6)))
    
    # XGBoost модели
    models.append(('xgb_1', xgb.XGBClassifier(random_state=42, n_jobs=1, use_label_encoder=False, eval_metric='logloss')))
    models.append(('xgb_2', xgb.XGBClassifier(random_state=42, n_jobs=1, n_estimators=200, use_label_encoder=False, eval_metric='logloss')))
    models.append(('xgb_3', xgb.XGBClassifier(random_state=42, n_jobs=1, max_depth=5, use_label_encoder=False, eval_metric='logloss')))
    
    # LightGBM модели
    models.append(('lgb_1', lgb.LGBMClassifier(random_state=42, n_jobs=1)))
    models.append(('lgb_2', lgb.LGBMClassifier(random_state=42, n_jobs=1, n_estimators=200)))
    models.append(('lgb_3', lgb.LGBMClassifier(random_state=42, n_jobs=1, num_leaves=31)))
    
    for name, model in models:
        try:
            print(f"\nTraining {name}...")
            oof = np.zeros(len(X))
            oof_proba = np.zeros(len(X))  # Для вероятностей
            pred = np.zeros(len(test_aligned))
            pred_proba = np.zeros(len(test_aligned))  # Для вероятностей
            
            for fold, (trn_idx, val_idx) in enumerate(FOLDS.split(X, y)):
                X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                # Обучение модели
                if name.startswith('cat'):
                    model.fit(X_train, y_train, verbose=0)
                else:
                    model.fit(X_train, y_train)
                
                # Предсказания для OOF
                oof_pred = model.predict(X_val)
                oof[val_idx] = oof_pred
                
                # Предсказания вероятностей для OOF
                if hasattr(model, 'predict_proba'):
                    oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
                
                # Предсказания для test
                fold_pred = model.predict(test_aligned)
                pred += fold_pred / FOLDS.n_splits
                
                # Предсказания вероятностей для test
                if hasattr(model, 'predict_proba'):
                    fold_proba = model.predict_proba(test_aligned)[:, 1]
                    pred_proba += fold_proba / FOLDS.n_splits
                
                # Расчет F1 score для фолда
                fold_f1 = f1_score(y_val, oof_pred)
                print(f'{name} - Fold {fold} F1: {fold_f1:.4f}')
            
            all_oof[name] = oof
            all_predictions[name] = pred
            
            # Сохраняем вероятности если доступны
            if hasattr(model, 'predict_proba'):
                all_predictions_proba[name] = pred_proba
            
            # Полный F1 score
            full_f1 = f1_score(y, oof)
            print(f'{name} - Full OOF F1: {full_f1:.4f}')
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Проверка, что хотя бы одна модель обучилась
    if not all_oof:
        print("Все модели завершились с ошибкой! Возвращаем None.")
        return None, None, None
    
    # Создаем DataFrame с результатами
    oof_df = pd.DataFrame(all_oof)
    predictions_df = pd.DataFrame(all_predictions)
    
    # Добавляем вероятности если есть
    if all_predictions_proba:
        predictions_proba_df = pd.DataFrame(all_predictions_proba)
        predictions_proba_df = predictions_proba_df.add_suffix('_proba')
        predictions_df = pd.concat([predictions_df, predictions_proba_df], axis=1)
    
    oof_df['target'] = y.values
    
    model_info = {
        'model_names': list(all_oof.keys()),
        'num_models': len(all_oof),
        'features_used': list(X.columns),
        'metric_used': 'f1_score'
    }
    
    return oof_df, predictions_df, model_info
    


def ensemble_predict(predictions_df, method='mean', threshold=0.5):
    """
    Усреднение предсказаний ансамбля
    
    Parameters:
    predictions_df (pd.DataFrame): DataFrame с предсказаниями моделей
    method (str): 'mean' или 'voting'
    threshold (float): Порог для бинаризации вероятностей
    
    Returns:
    final_predictions (np.array): Финальные предсказания
    """
    if method == 'mean':
        # Усреднение вероятностей
        proba_cols = [col for col in predictions_df.columns if '_proba' in col]
        if proba_cols:
            avg_proba = predictions_df[proba_cols].mean(axis=1)
            final_predictions = (avg_proba >= threshold).astype(int)
        else:
            # Если нет вероятностей, используем голосование
            model_cols = [col for col in predictions_df.columns if '_proba' not in col]
            avg_pred = predictions_df[model_cols].mean(axis=1)
            final_predictions = (avg_pred >= threshold).astype(int)
    
    elif method == 'voting':
        # Мажоритарное голосование
        model_cols = [col for col in predictions_df.columns if '_proba' not in col]
        votes = predictions_df[model_cols].apply(lambda x: x.round().astype(int))
        final_predictions = votes.mode(axis=1)[0].fillna(0).astype(int)
    
    return final_predictions


oof_results, test_predictions, info = create_ensemble_classification(X, y, test_processed)
    
if oof_results is not None:
    print("Ансамбль успешно обучен!")
        
    # Усреднение предсказаний
    final_predictions = ensemble_predict(test_predictions, method='mean', threshold=0.5)
    
    # Оценка качества на OOF
    oof_predictions = ensemble_predict(oof_results.drop(columns=['target']), method='mean', threshold=0.5)
    final_f1 = f1_score(y, oof_predictions)
    print(f"Final Ensemble OOF F1: {final_f1:.4f}")
        
else:
    print("Ансамбль не смог обучиться. Проверьте данные.")


pd.DataFrame({'AuthorId': np.arange(37500), 'DuplicateAuthorIds':final_predictions}).to_csv('solution.csv', index=False)

