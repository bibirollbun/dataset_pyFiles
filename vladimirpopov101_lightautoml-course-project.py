import logging
import random
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, cohen_kappa_score,
                              confusion_matrix, make_scorer)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore')


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_PATH = Path('/kaggle/input/prudential-life-insurance-assessment')

pd.set_option('display.max_columns', None)

plt.style.use('seaborn-v0_8-bright')
sns.set_palette("bright")


train_df = pd.read_csv(zipfile.ZipFile(DATA_PATH / 'train.csv.zip').open('train.csv'), 
                       index_col= 'Id')
test_df = pd.read_csv(zipfile.ZipFile(DATA_PATH / 'test.csv.zip').open('test.csv'), 
                      index_col= 'Id')
sample_submission = (pd.read_csv(zipfile.ZipFile(DATA_PATH / 'sample_submission.csv.zip')
                                 .open('sample_submission.csv')))


train_df.head()


train_df.shape


test_df.head()


test_df.shape


train_df.duplicated().sum()


target = train_df['Response']


target.describe()


target.value_counts().sort_index()


(target.value_counts(normalize=True).sort_values(ascending=False) * 100).round(2)


target.isnull().sum()


target.nunique()


plt.figure(figsize=(10, 6))

response_counts = target.value_counts().sort_index()
response_pct = (response_counts / len(target) * 100).values

ax = sns.barplot(x=response_counts.index, y=response_counts.values)

for i, (count, pct) in enumerate(zip(response_counts.values, response_pct)):
    ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)

plt.xlabel('Response', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Распределение целевой переменной', fontsize=14)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 6))

plt.scatter(train_df.index, train_df['Response'], s=5, alpha=0.4)

plt.title('Распределение целевой переменной по порядку наблюдений', fontsize=14)
plt.xlabel('Индекс наблюдения', fontsize=12)
plt.ylabel('Response', fontsize=12)
plt.yticks(range(1, 9))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


categorical_features = [
    'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 'Product_Info_6', 'Product_Info_7',
    'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5',
    'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
    'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 'Insurance_History_9',
    'Family_Hist_1',
    'Medical_History_2', 'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 'Medical_History_9',
    'Medical_History_11', 'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 'Medical_History_19',
    'Medical_History_20', 'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28',
    'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 'Medical_History_36', 'Medical_History_37',
    'Medical_History_38', 'Medical_History_39', 'Medical_History_40', 'Medical_History_41'
]

continuous_features = [
    'Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI',
    'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6',
    'Insurance_History_5',
    'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5'
]

discrete_features = [
    'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32'
]

binary_features = [f'Medical_Keyword_{i}' for i in range(1, 49)]


print(f"Категориальных: {len(categorical_features)}")
print(f"Непрерывных: {len(continuous_features)}")
print(f"Дискретных: {len(discrete_features)}")
print(f"Бинарных: {len(binary_features)}")

total_features_count = (
    len(categorical_features) 
    + len(continuous_features) 
    + len(discrete_features) 
    + len(binary_features)
)

(print(f"Всего: {total_features_count}"))


missing_pct = (train_df.isnull().sum() / len(train_df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]


plt.figure(figsize=(14, 10))
missing_pct.sort_values(ascending=True).plot(kind='barh', color='indianred')
plt.title('Пропущенные значения по признакам', fontsize=14)
plt.xlabel('Процент пропусков', fontsize=12)
plt.axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% порог')
plt.axvline(x=75, color='darkred', linestyle='--', linewidth=2, label='75% порог')
plt.legend()
plt.tight_layout()
plt.show()


numeric_features = continuous_features + discrete_features
numeric_df = train_df[numeric_features]
numeric_df.describe().T


fig, axes = plt.subplots(5, 4, figsize=(18, 20))
axes = axes.flatten()

for idx, col in enumerate(numeric_features):
    if idx < len(axes):
        axes[idx].hist(numeric_df[col].dropna(), bins=30, edgecolor='white', alpha=0.7)
        axes[idx].set_title(f'{col}', fontsize=11)
        axes[idx].set_xlabel('')
        axes[idx].grid(True, alpha=0.3)
        
        # добавляем информацию о пропусках
        missing_pct = numeric_df[col].isnull().sum() / len(numeric_df) * 100
        if missing_pct > 0:
            axes[idx].text(0.95, 0.95, f'Пропуски: {missing_pct:.1f}%', 
                          transform=axes[idx].transAxes, 
                          ha='right', va='top', 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=9)

for idx in range(len(continuous_features + discrete_features), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Распределения непрерывных и дискретных признаков', fontsize=16, y=1.00)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 5, figsize=(20, 14))
axes = axes.flatten()

for idx, col in enumerate(numeric_df.columns):
    if idx < len(axes):
        sns.boxplot(y=numeric_df[col], ax=axes[idx], color='lightblue')
        axes[idx].set_title(f'{col}', fontsize=11)
        axes[idx].set_ylabel('')

for idx in range(len(numeric_df.columns), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Boxplot численных признаков', fontsize=16, y=1.00)
plt.tight_layout()
plt.show()


binary_df = train_df[binary_features]
binary_df.describe().T


train_copy = train_df.copy()


train_copy['total_keywords'] = train_copy[binary_features].sum(axis=1)


train_copy['total_keywords'].describe()


plt.figure(figsize=(12, 6))

for response in sorted(train_copy['Response'].unique()):
    subset = train_copy[train_copy['Response'] == response]
    sns.kdeplot(data=subset, x='total_keywords', 
                label=f'Response {response}', fill=True, alpha=0.2, linewidth=1)

plt.title('Распределение количества медицинских ключевых слов по Response', fontsize=14)
plt.xlabel('Количество ключевых слов', fontsize=12)
plt.ylabel('Плотность', fontsize=12)
plt.legend(title='Response', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


keyword_freq = (train_copy[binary_features].sum() / len(train_df) * 100)


plt.figure(figsize=(14, 10))
keyword_freq.sort_values(ascending=True).plot(kind='barh', color='teal')
plt.title('Частота встречаемости медицинских ключевых слов', fontsize=14)
plt.xlabel('Процент наблюдений с ключевым словом')
plt.tight_layout()
plt.show()


cat_stats = pd.DataFrame({
    'Уникальных значений': train_df[categorical_features].nunique(),
    'Пропуски (%)': (train_df[categorical_features].isnull().sum() / len(train_df) * 100)
}).sort_values('Уникальных значений', ascending=False)


cat_stats.head(20)


cat_data = train_df[categorical_features].copy()

for col in categorical_features:
    cat_data[col] = cat_data[col]
    if cat_data[col].dtype == 'object':
        cat_data[col] = LabelEncoder().fit_transform(cat_data[col])

mi_scores_cat = mutual_info_classif(cat_data, train_df['Response'], random_state=RANDOM_STATE)

cat_importance = pd.DataFrame({
    'Признак': categorical_features,
    'MI Score': mi_scores_cat
}).sort_values('MI Score', ascending=False)


plt.figure(figsize=(12, 10))
top_20_cat = cat_importance.head(20)
plt.barh(range(len(top_20_cat)), top_20_cat['MI Score'], color='teal')
plt.yticks(range(len(top_20_cat)), top_20_cat['Признак'])
plt.xlabel('MI Score', fontsize=12)
plt.title('Топ-20 категориальных признаков по важности', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


top_12_cat = cat_importance.head(12)['Признак'].tolist()

fig, axes = plt.subplots(4, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, col in enumerate(top_12_cat):
    value_counts = train_df[col].value_counts().sort_index()
    value_counts.plot(kind='bar', ax=axes[idx], color='orchid', edgecolor='black')
    axes[idx].set_title(f'{col}', fontsize=11)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('Распределения топ-12 категориальных признаков', fontsize=16)
plt.tight_layout()
plt.show()


train_df['Medical_History_2'].describe()


all_data = train_df.drop('Response', axis=1).copy()


for col in all_data.columns:
    if all_data[col].dtype == 'object':
        all_data[col] = all_data[col].fillna('missing')
        all_data[col] = LabelEncoder().fit_transform(all_data[col])
    else:
        all_data[col] = all_data[col].fillna(-1)


mi_scores_all = mutual_info_classif(all_data, train_df['Response'], random_state=RANDOM_STATE)


all_importance = pd.DataFrame({
    'Признак': all_data.columns,
    'MI Score': mi_scores_all
}).sort_values('MI Score', ascending=False)


top_20_features = all_importance.head(20)['Признак'].tolist()


corr_matrix = all_data[top_20_features].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
            vmin=-1, vmax=1)

plt.title('Корреляционная матрица топ-20 признаков', fontsize=14)
plt.tight_layout()
plt.show()


corr_matrix = all_data.corr()


high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))


if high_corr_pairs:
    print(f"Высококоррелированные пары признаков (|r| > 0.7):")
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"  {feat1} - {feat2}: {corr_val:.3f}")
else:
    print("Высококоррелированных пар не обнаружено")


medical_keyword_cols = [col for col in train_copy.columns if col.startswith('Medical_Keyword_')]
train_copy['total_keywords'] = train_copy[medical_keyword_cols].sum(axis=1)


train_copy['total_keywords'].describe()


train_copy['age_group'] = pd.qcut(train_copy['Ins_Age'], 
                                q=4, labels=[1, 2, 3, 4], duplicates='drop').astype(int)


print(train_copy['age_group'].value_counts().sort_index())


train_copy['bmi_age_interaction'] = train_copy['BMI'] * train_copy['Ins_Age']


train_copy['bmi_age_interaction'].describe()


train_copy['bmi_category'] = pd.qcut(train_copy['BMI'], 
                                   q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)


print(train_copy['bmi_category'].value_counts().sort_index())


print("Корреляция с Response:")
print(f"   total_keywords: {train_copy['total_keywords'].corr(train_copy['Response']):.4f}")
print(f"   age_group: {train_copy['age_group'].corr(train_copy['Response']):.4f}")
print(f"   bmi_age_interaction: {train_copy['bmi_age_interaction'].corr(train_copy['Response']):.4f}")
print(f"   bmi_category: {train_copy['bmi_category'].corr(train_copy['Response']):.4f}")
print()


X_new = train_copy[['total_keywords', 'age_group', 'bmi_age_interaction', 'bmi_category']]
y = train_copy['Response']
mi_scores = mutual_info_classif(X_new, y, random_state=RANDOM_STATE)


print("Mutual Information:")
for feat, score in zip(X_new.columns, mi_scores):
    print(f"   {feat}: {score:.4f}")
print()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет Quadratic Weighted Kappa - метрику соревнования.
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# scorer для sklearn
qwk_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Трансформер для создания новых признаков.
    
    Создает следующие признаки:
    - total_keywords: сумма всех Medical_Keyword признаков
    - bmi_age_interaction: взаимодействие BMI и возраста
    - bmi_category: категоризация BMI по квантилям
    """
    
    def __init__(self):
        self.bmi_quantiles_ = None
        self.age_quantiles_ = None
        self.medical_keyword_cols_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Вычисляет квантили для категоризации.
        """
        # Определяем колонки ключевых слов
        self.medical_keyword_cols_ = [col for col in X.columns if col.startswith('Medical_Keyword_')]
        
        # Вычисляем квантили для BMI
        self.bmi_quantiles_ = np.percentile(
            X['BMI'].dropna(), 
            [20, 40, 60, 80]
        )
        
        
        logger.info(f"FeatureEngineer fitted. BMI quantiles: {self.bmi_quantiles_}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Создает новые признаки.
        """
        X_copy = X.copy()
        
        # Сумма ключевых слов
        if self.medical_keyword_cols_:
            X_copy['total_keywords'] = X_copy[self.medical_keyword_cols_].sum(axis=1)
        else:
            X_copy['total_keywords'] = 0
        
        # Произведение BMI и возраста
        X_copy['bmi_age_interaction'] = X_copy['BMI'] * X_copy['Ins_Age']
        
        # Категоризация BMI по квантилям
        X_copy['bmi_category'] = np.digitize(X_copy['BMI'], self.bmi_quantiles_) + 1
        
        logger.info(f"Created features. New shape: {X_copy.shape}")
        return X_copy


class DuplicateRemover(BaseEstimator, TransformerMixin):
    """
    Трансформер для удаления полных дубликатов из данных.
    """
    
    def __init__(self, subset: Optional[List[str]] = None):
        self.subset = subset
        self.n_duplicates_removed_ = 0
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет полные дубликаты из данных.
        """
        original_len = len(X)
        X_clean = X.drop_duplicates(subset=self.subset, keep='first')
        self.n_duplicates_removed_ = original_len - len(X_clean)
        
        logger.info(f"Removed {self.n_duplicates_removed_} duplicates. "
                   f"Shape: {original_len} -> {len(X_clean)}")
        return X_clean



def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные соревнования.
    """
    train_df = pd.read_csv(
        zipfile.ZipFile(DATA_PATH / 'train.csv.zip').open('train.csv'), 
        index_col='Id'
    )
    test_df = pd.read_csv(
        zipfile.ZipFile(DATA_PATH / 'test.csv.zip').open('test.csv'), 
        index_col='Id'
    )
    sample_submission = pd.read_csv(
        zipfile.ZipFile(DATA_PATH / 'sample_submission.csv.zip').open('sample_submission.csv')
    )
    
    logger.info(f"Data loaded. Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df, sample_submission


def prepare_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    remove_duplicates: bool = True,
    add_features: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    """
    Подготавливает данные для моделирования.
    """
    # Разделяем признаки и целевую переменную
    y_train = train_df['Response'].values
    X_train = train_df.drop('Response', axis=1)
    X_test = test_df.copy()
    
    # Удаление дубликатов
    if remove_duplicates:
        duplicates_mask = X_train.duplicated(keep='first')
        n_duplicates = duplicates_mask.sum()
        X_train = X_train[~duplicates_mask]
        y_train = y_train[~duplicates_mask]
        logger.info(f"Removed {n_duplicates} duplicates from train")
    
    # Добавление новых признаков
    if add_features:
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(X_train)
        X_train = feature_engineer.transform(X_train)
        X_test = feature_engineer.transform(X_test)
    
    feature_names = X_train.columns.tolist()
    logger.info(f"Prepared data. Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, feature_names



def run_lama_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    config_name: str = "basic",
    timeout: int = 600,
    n_folds: int = 5,
    preset: str = "default"
) -> Tuple[np.ndarray, float, TabularAutoML]:
    """
    Запускает LAMA с заданной конфигурацией.
    
    preset: 'default', 'lgb', 'lgb_tuned', 'cb', 'cb_tuned', 'linear', 'ensemble'
    """
    logger.info(f"Starting LAMA {config_name} configuration, preset={preset}")
    
    train_data = X_train.copy()
    train_data['Response'] = y_train
    
    task = Task('multiclass', metric='crossentropy')
    
    # Конфигурации
    if preset == "default":
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE}
        )
    
    elif preset == "lgb":
        # LightGBM
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE},
            general_params={'use_algos': [['lgb']]}
        )
    
    elif preset == "lgb_tuned":
        # LightGBM с тюнингом
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE},
            general_params={'use_algos': [['lgb']]},
            tuning_params={'max_tuning_iter': 100, 'max_tuning_time': timeout // 2}
        )
    
    elif preset == "cb":
        # CatBoost
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE},
            general_params={'use_algos': [['cb']]}
        )
    
    elif preset == "cb_tuned":
        # CatBoost с тюнингом
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE},
            general_params={'use_algos': [['cb']]},
            tuning_params={'max_tuning_iter': 100, 'max_tuning_time': timeout // 2}
        )
    
    elif preset == "linear":
        # Линейные модели
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE},
            general_params={'use_algos': [['linear_l2']]}
        )
    
    elif preset == "ensemble":
        # Ансамбль LGB + CB + Linear
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE},
            general_params={'use_algos': [['lgb', 'cb'], ['linear_l2']]}
        )
    
    elif preset == "ensemble_tuned":
        # Ансамбль с тюнингом
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=4,
            reader_params={'n_jobs': 4, 'cv': n_folds, 'random_state': RANDOM_STATE},
            general_params={'use_algos': [['lgb', 'cb'], ['lgb', 'cb']]},
            tuning_params={'max_tuning_iter': 50, 'max_tuning_time': timeout // 3}
        )
    
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    # Обучение
    oof_predictions = automl.fit_predict(
        train_data,
        roles={'target': 'Response'},
        verbose=1
    )

    class_mapping = automl.reader.class_mapping
    logger.info(f"LAMA class mapping: {class_mapping}")
        
    # Получаем OOF предсказания и конвертируем в классы
    oof_pred_indices = oof_predictions.data.argmax(axis=1)
        
    # Преобразуем индексы обратно в оригинальные метки классов
    index_to_class = {v: k for k, v in class_mapping.items()}
    oof_pred_classes = np.array([index_to_class[idx] for idx in oof_pred_indices])
    
    # Вычисляем QWK на OOF
    oof_qwk = quadratic_weighted_kappa(y_train, oof_pred_classes)
    logger.info(f"LAMA {config_name} OOF QWK: {oof_qwk:.5f}")
    
    # Предсказания на тесте
    test_predictions = automl.predict(X_test)
    test_pred_indices = test_predictions.data.argmax(axis=1)
    test_pred_classes = np.array([index_to_class[idx] for idx in test_pred_indices])    
    return test_pred_classes, oof_qwk, automl


# Загрузка данных
train_df, test_df, sample_submission = load_data()

# Подготовка данных без дополнительных признаков
X_train_basic, X_test_basic, y_train_basic, _ = prepare_data(
    train_df.copy(), 
    test_df.copy(),
    remove_duplicates=True,
    add_features=False
)

print(f"LAMA Basic - Train shape: {X_train_basic.shape}")
print(f"LAMA Basic - Test shape: {X_test_basic.shape}")


# Запуск LAMA Basic
lama_basic_preds, lama_basic_qwk, lama_basic_model = run_lama_baseline(
    X_train_basic,
    y_train_basic,
    X_test_basic,
    config_name="basic",
    timeout=1800
)

print(f"\n=== LAMA Basic Results ===")
print(f"OOF QWK: {lama_basic_qwk:.5f}")


def create_submission(
    test_ids: np.ndarray,
    predictions: np.ndarray,
    filename: str
) -> pd.DataFrame:
    """
    Создает файл submission для Kaggle.
    """
    submission = pd.DataFrame({
        'Id': test_ids,
        'Response': predictions.astype(int)
    })
    submission.to_csv(f'/kaggle/working/{filename}', index=False)
    logger.info(f"Submission saved to {filename}")
    
    # Проверка распределения предсказаний
    print(f"\nРаспределение предсказаний в {filename}:")
    print(submission['Response'].value_counts().sort_index())
    
    return submission




test_ids = test_df.index.values

submission_lama_basic = create_submission(
    test_ids, 
    lama_basic_preds, 
    'submission_lama_basic1800.csv'
)


# Подготовка данных
X_train_adv, X_test_adv, y_train_adv, feature_names = prepare_data(
    train_df.copy(), 
    test_df.copy(),
    remove_duplicates=True,
    add_features=True
)

print(f"LAMA Advanced - Train shape: {X_train_adv.shape}")
print(f"LAMA Advanced - Test shape: {X_test_adv.shape}")
print(f"New features: {[f for f in feature_names if f in ['total_keywords', 'bmi_age_interaction', 'bmi_category']]}")


# Запуск LAMA Advanced
lama_adv_preds, lama_adv_qwk, lama_adv_model = run_lama_baseline(
    X_train_adv,
    y_train_adv,
    X_test_adv,
    config_name="advanced",
    timeout=1800,
    preset="ensemble"
)

print(f"\n=== LAMA Advanced Results ===")
print(f"OOF QWK: {lama_adv_qwk:.5f}")


test_ids = test_df.index.values

submission_lama_adv = create_submission(
    test_ids, 
    lama_adv_preds, 
    'submission_lama_adv1800.csv'
)


from IPython.display import Image, display
display(Image(filename='/kaggle/input/img-subs/submissions_lama.png'))


class CatBoostPipeline:
    """
    Пайплайн для обучения и предсказания с CatBoost.
    """
    
    # Категориальные признаки из описания соревнования
    CATEGORICAL_FEATURES = [
        'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_5', 
        'Product_Info_6', 'Product_Info_7', 'Employment_Info_2', 'Employment_Info_3', 
        'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 
        'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 'InsuredInfo_7',
        'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 
        'Insurance_History_4', 'Insurance_History_7', 'Insurance_History_8', 
        'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 'Medical_History_3',
        'Medical_History_4', 'Medical_History_5', 'Medical_History_6', 'Medical_History_7',
        'Medical_History_8', 'Medical_History_9', 'Medical_History_11', 'Medical_History_12',
        'Medical_History_13', 'Medical_History_14', 'Medical_History_16', 'Medical_History_17',
        'Medical_History_18', 'Medical_History_19', 'Medical_History_20', 'Medical_History_21',
        'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
        'Medical_History_27', 'Medical_History_28', 'Medical_History_29', 'Medical_History_30',
        'Medical_History_31', 'Medical_History_33', 'Medical_History_34', 'Medical_History_35',
        'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
        'Medical_History_40', 'Medical_History_41'
    ]
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.best_params = None
        self.model = None
        self.categorical_features_ = None
        
    def _get_categorical_indices(self, X: pd.DataFrame) -> List[int]:
        """
        Возвращает индексы категориальных признаков.
        """
        cat_features = [col for col in self.CATEGORICAL_FEATURES if col in X.columns]
        if 'bmi_category' in X.columns:
            cat_features.append('bmi_category')
        if 'age_group' in X.columns:
            cat_features.append('age_group')
            
        cat_indices = [X.columns.get_loc(col) for col in cat_features]
        return cat_indices
    
    def _convert_predictions(self, preds: np.ndarray) -> np.ndarray:
        """
        Конвертирует регрессионные предсказания в классы 1-8.
        """
        return np.clip(np.round(preds), 1, 8).astype(int)
    
    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
        """
        Целевая функция для Optuna.
        """
        params = {
            'iterations': 2000,
            'loss_function': 'RMSE',
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 50.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'od_type': 'Iter',
            'od_wait': 50,
            'random_seed': self.random_state,
            'verbose': False,
            'task_type': 'GPU',
            'devices': '0'
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        
        cat_indices = self._get_categorical_indices(X)
        
        for train_idx, val_idx in cv.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_train_cv, y_train_cv,
                cat_features=cat_indices,
                eval_set=(X_val_cv, y_val_cv),
                early_stopping_rounds=50,
                verbose=False
            )
            
            preds = model.predict(X_val_cv)
            preds_classes = self._convert_predictions(preds)
            score = quadratic_weighted_kappa(y_val_cv, preds_classes)
            scores.append(score)
        
        return np.mean(scores)
    
    def optimize(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        n_trials: int = 50,
        timeout: int = 3600
    ) -> Dict:
        """
        Оптимизирует гиперпараметры с помощью Optuna.
        """
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        self.best_params['iterations'] = 2000
        self.best_params['od_type'] = 'Iter'
        self.best_params['od_wait'] = 50
        self.best_params['random_seed'] = self.random_state
        self.best_params['verbose'] = False
        self.best_params['task_type'] = 'GPU'
        self.best_params['devices'] = '0'
        
        logger.info(f"Best QWK: {study.best_value:.5f}")
        logger.info(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict] = None
    ) -> 'CatBoostPipeline':
        """
        Обучает финальную модель.
        """
        if params is None:
            if self.best_params is None:
                raise ValueError("No params provided and no best_params from optimization")
            params = self.best_params
        
        logger.info("Training final CatBoost Regressor model")
        
        cat_indices = self._get_categorical_indices(X_train)
        self.categorical_features_ = cat_indices
        
        self.model = CatBoostRegressor(**params)
        
        eval_set = (X_val, y_val) if X_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            cat_features=cat_indices,
            eval_set=eval_set,
            verbose=100
        )
        
        logger.info("Model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Делает предсказания.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        preds_raw = self.model.predict(X)
        return self._convert_predictions(preds_raw)
    
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Возвращает сырые регрессионные предсказания.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def fit_predict_cv(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        params: Optional[Dict] = None,
        n_folds: int = 5
    ) -> Tuple[np.ndarray, float]:
        """
        Обучает модель с кросс-валидацией и возвращает OOF предсказания.
        """
        if params is None:
            params = self.best_params
            
        logger.info(f"Running {n_folds}-fold cross-validation")
        
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        oof_preds_raw = np.zeros(len(y))
        scores = []
        
        cat_indices = self._get_categorical_indices(X)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            logger.info(f"Fold {fold + 1}/{n_folds}")
            
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_train_cv, y_train_cv,
                cat_features=cat_indices,
                eval_set=(X_val_cv, y_val_cv),
                early_stopping_rounds=50,
                verbose=False
            )
            
            preds_raw = model.predict(X_val_cv)
            oof_preds_raw[val_idx] = preds_raw
            
            preds_classes = self._convert_predictions(preds_raw)
            score = quadratic_weighted_kappa(y_val_cv, preds_classes)
            scores.append(score)
            logger.info(f"Fold {fold + 1} QWK: {score:.5f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        logger.info(f"CV QWK: {mean_score:.5f} (+/- {std_score:.5f})")
        
        # Возвращаем классы
        oof_preds = self._convert_predictions(oof_preds_raw)
        return oof_preds, mean_score
    
    def run_full_pipeline(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_trials: int = 50,
        optimize_timeout: int = 3600
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Запускает полный пайплайн.
        """
        logger.info("="*50)
        logger.info("STARTING CATBOOST REGRESSOR PIPELINE")
        logger.info("="*50)
        
        # 1. Подготовка данных
        logger.info("Step 1: Preparing data")
        y_train = train_df['Response'].values
        X_train = train_df.drop('Response', axis=1)
        X_test = test_df.copy()
        
        # Удаление дубликатов
        duplicates_mask = X_train.duplicated(keep='first')
        X_train = X_train[~duplicates_mask]
        y_train = y_train[~duplicates_mask]
        logger.info(f"Removed {duplicates_mask.sum()} duplicates")
        
        # Feature Engineering
        logger.info("Step 2: Feature engineering")
        self.feature_engineer.fit(X_train)
        X_train = self.feature_engineer.transform(X_train)
        X_test = self.feature_engineer.transform(X_test)
        
        # Optuna
        logger.info("Step 3: Hyperparameter optimization")
        self.optimize(X_train, y_train, n_trials=n_trials, timeout=optimize_timeout)
        
        # Кросс-валидация
        logger.info("Step 4: Cross-validation")
        oof_preds, cv_score = self.fit_predict_cv(X_train, y_train)
        
        # Обучение финальной модели
        logger.info("Step 5: Training final model")
        self.fit(X_train, y_train)
        
        # Предсказания
        logger.info("Step 6: Predicting on test")
        test_preds = self.predict(X_test)
        
        logger.info("="*50)
        logger.info(f"PIPELINE COMPLETED. CV QWK: {cv_score:.5f}")
        logger.info("="*50)
        
        return oof_preds, test_preds, cv_score


# Инициализация и запуск пайплайна
catboost_pipeline = CatBoostPipeline(random_state=RANDOM_STATE)

catboost_oof, catboost_preds, catboost_cv_qwk = catboost_pipeline.run_full_pipeline(
    train_df.copy(),
    test_df.copy(),
    n_trials=15,  # Количество испытаний Optuna
    optimize_timeout=300
)

print(f"\nCatBoost Pipeline Results")
print(f"CV QWK: {catboost_cv_qwk:.5f}")
print(f"Best params: {catboost_pipeline.best_params}")


submission_catboost = create_submission(
    test_ids, 
    catboost_preds, 
    'submission_catboost_reg.csv'
)


display(Image(filename='/kaggle/input/img-subs/submission_catboost.png'))

