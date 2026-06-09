%%bash --no-raise-error

# Обновление pip до последней версии
pip install -q --upgrade pip

# Установка необходимых пакетов. 
pip install -q --upgrade \
    catboost \
    dcor \
    unidecode \
    networkx \
    optuna-integration \
    optuna \
    langchain openai \
    langchain_community \
    sdv

# Перезапуск ядра для применения изменений.
echo "------------------------------------------------"
echo "Установка завершена. Перезапускаем среду выполнения..."
echo "------------------------------------------------"
python - <<'PY'
import os, signal
os.kill(os.getpid(), signal.SIGTERM)
PY


# ==============================================================================
# 1. СИСТЕМНЫЕ И СТАНДАРТНЫЕ БИБЛИОТЕКИ
# ==============================================================================
import logging
import warnings
import os
from typing import Dict, Any, List, Tuple, Optional, Callable

# ==============================================================================
# 2. СТОРОННИЕ БИБЛИОТЕКИ ДЛЯ АНАЛИЗА ДАННЫХ И ML
# ==============================================================================
import dcor
import lightgbm as lgb
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import pandas as pd
import torch
import plotly.io as pio
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from getpass import getpass
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier, Pool
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from optuna.integration import CatBoostPruningCallback
from langchain.agents import initialize_agent, AgentType
from optuna.visualization import plot_optimization_history, plot_param_importances
from scipy.signal import detrend
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve,
    roc_curve, auc
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# ==============================================================================
# 3. НАСТРОЙКА ОКРУЖЕНИЯ
# ==============================================================================
class Config:
    """Класс для хранения глобальных констант и настроек."""
    # --- Основные константы ---
    RANDOM_STATE = 42
    N_SPLITS = 5
    TARGET_COL = 'Exited'

    # --- Корпоративная палитра ---
    MTS_RED = "#E60012"
    HSE_DARKBLUE = "#1B365D"
    RUSSIAN_BLUE = "#0033A0"
    RUSSIAN_WHITE = "#FFFFFF"
    CORPORATE_PALETTE = [MTS_RED, HSE_DARKBLUE, RUSSIAN_BLUE]

    @staticmethod
    def setup_environment():
        """Настраивает глобальные параметры для логирования и визуализации."""
        # Настройка логирования
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Убираем предупреждения
        warnings.filterwarnings('ignore')

        # Фиксация random seed
        np.random.seed(Config.RANDOM_STATE)

        # Настройка визуализации
        Config._setup_visualization()

    @staticmethod
    def _setup_visualization():
        """Применяет корпоративный стиль к Seaborn и Matplotlib."""
        sns.set_theme(
            style="whitegrid",
            palette=Config.CORPORATE_PALETTE,
            rc={
                "figure.figsize": (12, 6), "figure.facecolor": Config.RUSSIAN_WHITE,
                "axes.facecolor": Config.RUSSIAN_WHITE, "axes.edgecolor": Config.HSE_DARKBLUE,
                "axes.titlesize": 18, "axes.labelsize": 14,
                "xtick.labelsize": 12, "ytick.labelsize": 12,
                "figure.dpi": 300, "savefig.dpi": 300,
                "axes.titlepad": 20, "axes.labelpad": 15,
                "legend.fontsize": 12, "legend.title_fontsize": 14,
                "grid.color": "#EEEEEE",
            }
        )
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=Config.CORPORATE_PALETTE)

# --- Глобальные объекты и вызов настроек ---
Config.setup_environment()
logger = logging.getLogger(__name__)
pio.renderers.default = "kaggle"

MTS_RED = "#E60012"
HSE_DARKBLUE = "#1B365D"
RUSSIAN_BLUE = "#0033A0"
RUSSIAN_WHITE = "#FFFFFF"
CORPORATE_PALETTE = [MTS_RED, HSE_DARKBLUE, RUSSIAN_BLUE]
MTS_HSE_CMAP = LinearSegmentedColormap.from_list("mts_hse_diverging", [Config.HSE_DARKBLUE, Config.RUSSIAN_WHITE, Config.MTS_RED])
MTS_RED_CMAP = LinearSegmentedColormap.from_list("mts_sequential_red", [Config.RUSSIAN_WHITE, Config.MTS_RED])


class FeatureEngineer:
    """
    Утилитарный класс для инженерии признаков. Все методы являются статическими
    и представляют собой независимые пайплайны предобработки.
    """

    @staticmethod
    def map_columns(df: pd.DataFrame, mappings: Dict[str, Dict[Any, Any]]) -> pd.DataFrame:
        df_copy = df.copy()
        for col, mapping in mappings.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].map(mapping)
        return df_copy

    @staticmethod
    def cast_columns(df: pd.DataFrame, int_cols: Optional[List[str]] = None,
                     cat_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df_copy = df.copy()
        if int_cols:
            for col in int_cols:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].astype(int)
        if cat_cols:
            for col in cat_cols:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].astype('category')
        return df_copy

    @staticmethod
    def run_v0_baseline(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Базовый пайплайн: только приведение типов, без создания признаков."""
        df_copy = df.copy()
        int_cols = ['HasCrCard', 'IsActiveMember']
        cat_cols = ['Geography', 'Gender', 'Surname']
        df_copy = FeatureEngineer.cast_columns(df_copy, int_cols=int_cols, cat_cols=cat_cols)

        cols_to_drop = ['CustomerId']
        if is_train:
            cols_to_drop.append('Exited')
        df_copy.drop(columns=[col for col in cols_to_drop if col in df_copy.columns], inplace=True)
        return df_copy

    @staticmethod
    def run_v1_preprocessing(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Версия 1: Базовые флаги и биннинг."""
        df_copy = df.copy()
        gender_map = {'Male': 0, 'Female': 1}
        df_copy = FeatureEngineer.map_columns(df_copy, {'Gender': gender_map})
        df_copy['Age_bin'] = pd.cut(df_copy['Age'], bins=[0, 25, 35, 45, 60, np.inf],
                                    labels=['very_young', 'young', 'mid', 'mature', 'senior'])
        df_copy['Is_two_products'] = (df_copy['NumOfProducts'] == 2)
        df_copy['Germany_Female'] = ((df_copy['Geography'] == 'Germany') & (df_copy['Gender'] == 1))
        df_copy['Germany_Inactive'] = ((df_copy['Geography'] == 'Germany') & (df_copy['IsActiveMember'] == 0))
        df_copy['Has_Zero_Balance'] = (df_copy['Balance'] == 0)
        df_copy['Tenure_log'] = np.log1p(df_copy['Tenure'])

        int_cols = ['HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Is_two_products', 'Has_Zero_Balance',
                    'Germany_Female', 'Germany_Inactive']

        cat_cols = ['Geography', 'Age_bin', 'Germany_Female', 'Germany_Inactive', 'Surname', 'Age_bin']
        df_copy = FeatureEngineer.cast_columns(df_copy, int_cols=int_cols, cat_cols=cat_cols)


        cols_to_drop = ['CustomerId', 'Tenure' ]
        if is_train:

            if 'Exited' in df_copy.columns:
                cols_to_drop.append('Exited')

        df_copy.drop(columns=[col for col in cols_to_drop if col in df_copy.columns], inplace=True, errors='ignore')
        return df_copy

    @staticmethod
    def run_v2_preprocessing(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Версия 2: V1 + новый флаг is_mature_inactive_transit."""
        df_copy = FeatureEngineer.run_v1_preprocessing(df, is_train=False)
        df_copy['is_mature_inactive_transit'] = (
                (df_copy['Has_Zero_Balance'] == 1) & (df_copy['IsActiveMember'] == 0) & (
                df_copy['Age'] > 40)).astype(int)

        if is_train and 'Exited' in df_copy.columns:
            df_copy.drop(columns=['Exited'], inplace=True, errors='ignore')
        return df_copy

    @staticmethod
    def run_v3_preprocessing(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Версия 3: V1 + полиномиальные/взаимодействующие признаки."""
        df_copy = FeatureEngineer.run_v1_preprocessing(df, is_train=False)
        df_copy['Balance_per_product'] = df_copy['Balance'] / (df_copy['NumOfProducts'] + 1e-9)
        df_copy['Age_x_Tenure'] = df_copy['Age'] * df_copy['Tenure_log']
        df_copy['CreditScore_x_Age'] = df_copy['CreditScore'] * df_copy['Age']

        if is_train and 'Exited' in df_copy.columns:
            df_copy.drop(columns=['Exited'], inplace=True, errors='ignore')
        return df_copy

    @staticmethod
    def target_encode_surname(train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """Out-of-Fold Target Encoding для признака 'Surname'."""
        train_df_copy, test_df_copy = train_df.copy(), test_df.copy()

        freq = train_df_copy['Surname'].value_counts()
        rare_thr = 5
        rare_surnames = freq[freq < rare_thr].index
        train_df_copy['Surname_mod'] = train_df_copy['Surname'].replace(rare_surnames, 'RARE')
        test_df_copy['Surname_mod'] = test_df_copy['Surname'].replace(rare_surnames, 'RARE')

        global_mean = y.mean()
        kf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
        smoothing = 30

        te_mean = pd.Series(index=train_df_copy.index, dtype=float)
        for tr_idx, val_idx in kf.split(train_df_copy, y):
            stats = y.iloc[tr_idx].groupby(train_df_copy.iloc[tr_idx]['Surname_mod']).agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
            te_mean.iloc[val_idx] = train_df_copy.iloc[val_idx]['Surname_mod'].map(smooth).fillna(global_mean)

        full_stats = y.groupby(train_df_copy['Surname_mod']).agg(['mean', 'count'])
        smooth_full = (full_stats['mean'] * full_stats['count'] + global_mean * smoothing) / (
                full_stats['count'] + smoothing)
        te_mean_test = test_df_copy['Surname_mod'].map(smooth_full).fillna(global_mean)

        tr = pd.DataFrame({'Surname_TE': te_mean}, index=train_df_copy.index)
        te = pd.DataFrame({'Surname_TE': te_mean_test}, index=test_df_copy.index)
        return tr, te


class EDAVisualizer:
    """
    Класс-хелпер для проведения визуального разведочного анализа данных (EDA).
    Содержит набор статических методов для построения стандартизированных графиков
    в соответствии с  корпоративным стилем.
    """
    @staticmethod
    def plot_numerical_distributions_with_target(df: pd.DataFrame, features: List[str], target: str):
        """Построение распределения числовых признаков в разрезе целевой переменной."""
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        axes = axes.flatten()
        fig.suptitle('Распределение числовых признаков по статусу оттока', fontsize=20, y=1.03)

        for i, feature in enumerate(features):
            ax = axes[i]
            sns.kdeplot(data=df, x=feature, hue=target, fill=True, ax=ax)
            ax.set_title(f'Распределение "{feature}"')

        for j in range(n_features, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    @staticmethod
    def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        gender_map = {'Male': 0, 'Female': 1}
        df = FeatureEngineer.map_columns(df, {'Gender': gender_map})

        df['Age_bin'] = pd.cut(
            df['Age'],
            bins=[0, 25, 35, 45, 60, np.inf],
            labels=['very_young', 'young', 'mid', 'mature', 'senior']
        )

        df['Is_two_products'] = (df['NumOfProducts'] == 2)
        df['Germany_Female'] = ((df['Geography'] == 'Germany') & (df['Gender'] == 1))
        df['Germany_Inactive'] = ((df['Geography'] == 'Germany') & (df['IsActiveMember'] == 0))
        df['Has_Zero_Balance'] = (df['Balance'] == 0)

        int_columns = ['HasCrCard', 'IsActiveMember', 'NumOfProducts', 'Is_two_products', 'Has_Zero_Balance']
        cat_columns = ['Geography', 'Age_bin', 'Germany_Female', 'Germany_Inactive', 'Surname', 'Age_bin']
        df = FeatureEngineer.cast_columns(df, int_cols=int_columns, cat_cols=cat_columns)

        df['Tenure_log'] = np.log1p(df['Tenure'])
        df.drop(columns=['CustomerId', 'Tenure'], inplace=True)

        return df

    @staticmethod
    def analyze_feature_dependencies(
        df: pd.DataFrame,
        target: str = 'Exited',
        drop_cols: list[str] = ['id', 'Surname'],
        random_state: int = 0,
        figsize: tuple[int, int] = (10, 8)  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Оценивает взаимозависимости признаков (MI, dCor, Partial Correlation)."""
        df_processed = EDAVisualizer.preprocess_df(df).drop(columns=drop_cols)

        df_enc = pd.get_dummies(df_processed, drop_first=True)
        X = df_enc.drop(columns=[target], errors='ignore')
        y = df_enc[target]
        X = X.fillna(X.median())
        df_xy = pd.concat([X, y], axis=1).dropna(how='any')
        X, y = df_xy.drop(columns=[target]), df_xy[target]

        mi_scores = mutual_info_classif(X, y, random_state=random_state)
        mi_df = pd.DataFrame({'feature': X.columns, 'MI': mi_scores}).sort_values('MI', ascending=False).reset_index(
            drop=True)

        dcor_scores = [dcor.distance_correlation(X[col].to_numpy(), y.to_numpy()) for col in X.columns]
        dcor_df = pd.DataFrame({'feature': X.columns, 'dCor': dcor_scores}).sort_values('dCor',
                                                                                        ascending=False).reset_index(
            drop=True)

        num_cols = df_enc.select_dtypes(include=[np.number]).columns
        detrended = pd.DataFrame({col: detrend(df_enc[col].to_numpy()) for col in num_cols if col in df_enc.columns},
                        index=df_enc.index).fillna(0)
        pcorr_df = detrended.corr()

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(pcorr_df, interpolation='nearest', aspect='auto', cmap=MTS_HSE_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(pcorr_df.columns)))
        ax.set_xticklabels(pcorr_df.columns, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(pcorr_df.index)))
        ax.set_yticklabels(pcorr_df.index)
        fig.colorbar(im, ax=ax, label='Коэффициент корреляции')
        ax.set_title('Матрица частичных корреляций (после удаления тренда)')
        plt.tight_layout()
        plt.show()

        return mi_df, dcor_df, pcorr_df
    @staticmethod
    def plot_target_distribution(
        data: pd.DataFrame,
        target_col: str,
        ax: plt.Axes,
        palette: Optional[List[str]] = None
    ) -> None:
        """Строит и стилизует график распределения целевой переменной."""
        sns.countplot(
            x=target_col,
            data=data,
            palette=palette if palette else CORPORATE_PALETTE,
            ax=ax
        )
        total = len(data)
        for p in ax.patches:
            count = p.get_height()
            percentage = f'({100 * count / total:.2f}%)'
            annotation_text = f'{int(count):,}\n{percentage}'
            ax.annotate(
                text=annotation_text,
                xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='center', xytext=(0, 15),
                textcoords='offset points', fontsize=12,
                fontweight='bold', color=HSE_DARKBLUE
            )
        ax.set_title(f'Дисбаланс классов в целевой переменной "{target_col}"')
        ax.set_ylabel('Количество клиентов')
        ax.set_xlabel(None)
        ax.set_xticklabels(['Остался (Класс 0)', 'Ушел (Класс 1)'])
        ax.tick_params(axis='y', which='both', length=0)
        ax.grid(axis='x', visible=False)

    @staticmethod
    def plot_categorical_analysis(df: pd.DataFrame, features: List[str], target: str = None):
        """Анализирует категориальные признаки: распределение или долю оттока."""
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        axes = axes.flatten()
        if target:
            fig.suptitle('Доля оттока по категориальным признакам', fontsize=20, y=1.03)
            global_mean = df[target].mean()
        else:
            fig.suptitle('Распределение категориальных признаков', fontsize=20, y=1.03)
        for i, feature in enumerate(features):
            ax = axes[i]
            if target:
                data = df.groupby(feature)[target].mean().sort_values(ascending=False).reset_index()
                bars = sns.barplot(x=feature, y=target, data=data, ax=ax, color=MTS_RED)
                ax.axhline(global_mean, ls='--', color=HSE_DARKBLUE, label=f'Средний отток ({global_mean:.2%})')
                ax.legend()
                ax.set_ylabel('Доля ушедших клиентов')
                for p in bars.patches:
                    ax.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
            else:
                counts = df[feature].value_counts()
                bars = sns.barplot(x=counts.index, y=counts.values, ax=ax, color=MTS_RED)
                ax.set_ylabel('Количество')
                for p in bars.patches:
                    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
            ax.set_title(f'Анализ признака "{feature}"')
            ax.set_xlabel(feature)
            ax.tick_params(axis='x', rotation=0)
        for j in range(n_features, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    @staticmethod
    def plot_numerical_summary(df: pd.DataFrame, features: List[str]):
        """Строит комбинированный график (гистограмма + boxplot) для каждого числового признака."""
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(n_cols * 9, n_rows * 7))
        fig.suptitle('Комплексный анализ числовых признаков', fontsize=20, y=1.0)
        main_gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.3)
        for i, feature in enumerate(features):
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[i], height_ratios=(4, 1), hspace=0.05)
            ax_hist = fig.add_subplot(inner_gs[0])
            sns.histplot(df[feature], ax=ax_hist, kde=True, color=MTS_RED, bins=30)
            mean_val = df[feature].mean()
            median_val = df[feature].median()
            ax_hist.axvline(mean_val, color=HSE_DARKBLUE, linestyle='--', linewidth=2, label=f'Среднее: {mean_val:.2f}')
            ax_hist.axvline(median_val, color=RUSSIAN_BLUE, linestyle='-', linewidth=2, label=f'Медиана: {median_val:.2f}')
            ax_hist.set_title(f'Анализ признака "{feature}"')
            ax_hist.set_xlabel('')
            ax_hist.set_ylabel('Количество')
            ax_hist.legend()
            plt.setp(ax_hist.get_xticklabels(), visible=False)
            ax_box = fig.add_subplot(inner_gs[1], sharex=ax_hist)
            sns.boxplot(x=df[feature], ax=ax_box, color=MTS_RED)
            ax_box.set_xlabel('Значение признака')
            ax_box.set_ylabel('')
            ax_box.tick_params(axis='y', left=False, labelleft=False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    @staticmethod
    def create_summary_report(df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает детальный сводный отчет по каждому признаку в DataFrame.

        Функция анализирует каждый столбец и собирает ключевые метрики:
        - Тип данных (Dtype)
        - Количество и процент пропущенных значений (N_Null, Percent_Null)
        - Количество уникальных значений (N_Unique)
        - Для числовых признаков: основные описательные статистики.
        - Для категориальных/объектных признаков: самое частое значение (Mode).

        Args:
            df (pd.DataFrame): Входной DataFrame для анализа.

        Returns:
            pd.DataFrame: Сводный отчет в виде DataFrame, где индекс - это
                          названия признаков, а столбцы - собранные метрики.
        """
        summary_data: List[Dict[str, Any]] = []

        for col in df.columns:

            col_summary = {
                "Dtype": df[col].dtype,
                "N_Null": df[col].isnull().sum(),
                "Percent_Null": round(100 * df[col].isnull().sum() / len(df), 2),
                "N_Unique": df[col].nunique(),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                stats = df[col].describe()
                col_summary.update({
                    "Mean": round(stats["mean"], 2),
                    "Std": round(stats["std"], 2),
                    "Min": stats["min"],
                    "Median": stats["50%"],
                    "Max": stats["max"],
                    "Mode": "—",
                })
            else:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                col_summary.update({
                    "Mean": "—",
                    "Std": "—",
                    "Min": "—",
                    "Median": "—",
                    "Max": "—",
                    "Mode": mode_value,
                })

            summary_data.append(col_summary)

        report_df = pd.DataFrame(summary_data, index=df.columns)

        column_order = [
            "Dtype", "N_Null", "Percent_Null", "N_Unique",
            "Mean", "Std", "Min", "Median", "Max", "Mode"
        ]
        report_df = report_df.reindex(columns=column_order)

        return report_df

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, numerical_features: List[str]):
        """Строит тепловую карту корреляций для числовых признаков."""
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numerical_features].corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap=MTS_HSE_CMAP, fmt='.2f',
            linewidths=0.5, linecolor=RUSSIAN_WHITE, annot_kws={"size": 10}
        )
        plt.title('Тепловая карта корреляций числовых признаков')
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.show()

    @staticmethod
    def plot_numerical_distributions_with_target(df: pd.DataFrame, features: List[str], target: str):
        """Строит распределения числовых признаков в разрезе целевой переменной."""
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        axes = axes.flatten()
        fig.suptitle('Распределение числовых признаков по статусу оттока', fontsize=20, y=1.03)
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.kdeplot(data=df, x=feature, hue=target, fill=True, ax=ax)
            ax.set_title(f'Распределение "{feature}"')
        for j in range(n_features, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    @staticmethod
    def plot_credit_score_analysis_by_country(
        df: pd.DataFrame, score_col: str, target_col: str, geo_col: str
    ):
        """Анализирует кредитный рейтинг в разрезе оттока по странам."""
        df_plot = df.copy()
        df_plot[target_col] = df_plot[target_col].map({0: 'Остался', 1: 'Ушел'})
        g = sns.catplot(
            data=df_plot, x=target_col, y=score_col, col=geo_col, kind='box',
            palette=[HSE_DARKBLUE, MTS_RED], height=6, aspect=0.85, order=['Остался', 'Ушел']
        )
        g.fig.suptitle(f'Анализ "{score_col}" в разрезе оттока по странам', y=1.03)
        g.set_axis_labels("Статус клиента", "Кредитный рейтинг")
        g.set_titles("Страна: {col_name}")
        g.despine()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    @staticmethod
    def plot_age_distribution_by_geo_and_churn(df: pd.DataFrame, age_col: str, geo_col: str, target_col: str):
        """Строит violin plots для анализа распределения возраста по стране и оттоку."""
        plt.figure(figsize=(16, 8))
        sns.violinplot(
            data=df, x=geo_col, y=age_col, hue=target_col,
            split=True, palette={0: HSE_DARKBLUE, 1: MTS_RED}, inner='quart'
        )
        plt.title(f'Распределение возраста ({age_col}) по странам и статусу оттока')
        plt.xlabel('Страна')
        plt.ylabel('Возраст')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, ['Остался', 'Ушел'], title='Статус клиента')
        plt.show()

    @staticmethod
    def plot_bivariate_categorical_churn_analysis(df: pd.DataFrame, cat_col1: str, cat_col2: str, target_col: str):
        """Анализирует долю оттока на пересечении двух категориальных признаков."""
        g = sns.catplot(
            data=df, x=cat_col1, y=target_col, col=cat_col2,
            kind='bar', palette=CORPORATE_PALETTE, height=6, aspect=1, errorbar=None
        )
        global_mean = df[target_col].mean()
        for ax in g.axes.ravel():
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2%}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')
            ax.set_ylabel('Доля ушедших клиентов')
            ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
            ax.axhline(global_mean, ls='--', color=HSE_DARKBLUE, label=f'Средний отток ({global_mean:.2%})')
            ax.legend()
        g.fig.suptitle(f'Доля оттока: "{cat_col1}" в разрезе "{cat_col2}"', y=1.03)
        g.set_axis_labels(f'{cat_col1}', 'Доля оттока')
        g.set_titles("Страна: {col_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    @staticmethod
    def analyze_salary_interaction(df: pd.DataFrame, salary_col: str, target_col: str):
        """Комплексно анализирует взаимодействие признака зарплаты с целевой переменной."""
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}
        )
        fig.suptitle(f'Анализ взаимодействия "{salary_col}" и "{target_col}"')
        palette_list = [HSE_DARKBLUE, MTS_RED]
        sns.kdeplot(data=df, x=salary_col, hue=target_col, fill=True, palette=palette_list, ax=ax1)
        ax1.set_title('Плотность распределения зарплаты')
        ax1.set_ylabel('Плотность')
        handles, _ = ax1.get_legend_handles_labels()
        ax1.legend(handles, ['Остался', 'Ушел'], title='Статус клиента')
        df_copy = df.copy()
        df_copy[target_col] = df_copy[target_col].astype(int)
        sns.boxplot(data=df_copy, x=salary_col, y=target_col, orient='h', palette=palette_list, ax=ax2)
        ax2.set_title('Сравнение распределений зарплаты (Boxplot)')
        ax2.set_xlabel('Предполагаемая зарплата')
        ax2.set_yticklabels(['Остался', 'Ушел'])
        ax2.set_ylabel('Статус')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_multidimensional_churn_analysis(
        df: pd.DataFrame, target_col: str, geo_col: str, gender_col: str, activity_col: str
    ):
        """Строит комплексный анализ оттока по географии, полу и активности."""
        df_plot = df.copy()
        df_plot[activity_col] = df_plot[activity_col].map({0: 'Неактивен', 1: 'Активен'})
        df_plot[gender_col] = df_plot[gender_col].map({'Male': 'Мужчины', 'Female': 'Женщины'})
        g = sns.catplot(
            data=df_plot, x=gender_col, y=target_col, hue=activity_col, col=geo_col,
            kind='bar', height=6, aspect=0.8, palette={'Неактивен': MTS_RED, 'Активен': HSE_DARKBLUE},
            legend=False, estimator=lambda x: sum(x) / len(x),
            order=['Женщины', 'Мужчины'], hue_order=['Активен', 'Неактивен'], errorbar=None
        )
        global_mean = df[target_col].mean()
        for ax in g.axes.ravel():
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
            ax.axhline(global_mean, ls='--', color='grey', alpha=0.9, zorder=0)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            ax.set_xlabel('')
        g.fig.suptitle('Зависимость оттока от пола, активности и страны', y=1.03)
        g.set_axis_labels("", "Доля ушедших клиентов")
        g.set_titles("Страна: {col_name}")
        legend_elements = [
            Patch(facecolor=HSE_DARKBLUE, label='Активен'), Patch(facecolor=MTS_RED, label='Неактивен'),
            Line2D([0], [0], color='grey', ls='--', label=f'Средний отток ({global_mean:.1%})')
        ]
        g.fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                     ncol=len(legend_elements), title='Статус клиента', frameon=False)
        g.fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_numeric_distribution_by_two_categories(
        df: pd.DataFrame, numeric_col: str, group_col_1: str, group_col_2: str,
        title_prefix: str, figsize: tuple = (15, 6)
    ):
        """Визуализирует распределение числового признака, сгруппированного по двум категориальным."""
        required_cols = [numeric_col, group_col_1, group_col_2]
        if any(col not in df.columns for col in required_cols):
            raise ValueError("Одна или несколько указанных колонок отсутствуют в DataFrame.")
        df_copy = df.copy()
        for col in [group_col_1, group_col_2]:
            df_copy[col] = df_copy[col].astype('category').cat.reorder_categories(
                sorted(df_copy[col].unique().tolist()), ordered=True
            )
        g = sns.FacetGrid(df_copy, col=group_col_1, hue=group_col_2,
                          height=figsize[1], aspect=figsize[0]/figsize[1]/df_copy[group_col_1].nunique(),
                          palette=CORPORATE_PALETTE, sharey=True)
        g.map_dataframe(sns.violinplot, x=group_col_2, y=numeric_col, inner='quartile')
        g.set_axis_labels(x_var=group_col_2, y_var=numeric_col)
        g.set_titles(col_template='{col_name}')
        g.add_legend(title=group_col_2)
        plt.suptitle(f"{title_prefix} {numeric_col} по {group_col_1} и {group_col_2}", y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    @staticmethod
    def plot_churn_by_age_groups_and_geo(
        df: pd.DataFrame, age_col: str = 'Age', target_col: str = 'Exited',
        geo_col: str = 'Geography', palette: Optional[list] = None
    ):
        """Создает point plot для анализа оттока по возрастным группам и странам."""
        df_plot = df.copy()
        bins = [18, 30, 40, 50, 60, 100]
        labels = ['18-30', '31-40', '41-50', '51-60', '60+']
        df_plot['AgeGroup'] = pd.cut(df_plot[age_col], bins=bins, labels=labels, right=False)
        plt.figure(figsize=(16, 9))
        sns.pointplot(
            x='AgeGroup', y=target_col, hue=geo_col, data=df_plot,
            dodge=True, palette=palette if palette else CORPORATE_PALETTE
        )
        plt.title('Уровень оттока по возрастным группам и странам')
        plt.xlabel('Возрастная группа')
        plt.ylabel('Доля ушедших клиентов')
        plt.legend(title='Страна')
        plt.xticks(rotation=0)
        plt.show()

    @staticmethod
    def plot_customer_quadrant_analysis(
        df: pd.DataFrame, balance_col: str = 'Balance', tenure_col: str = 'Tenure',
        is_active_col: str = 'IsActiveMember', target_col: str = 'Exited',
        palette: Optional[Dict[int, str]] = None, figsize: Tuple[int, int] = (14, 8)
    ) -> pd.DataFrame:
        """Проводит анализ оттока по квадрантам (баланс/срок обслуживания)."""
        df_analysis = df.copy()
        if palette is None:
            palette = {0: MTS_RED, 1: HSE_DARKBLUE}
        balance_median = df_analysis[balance_col].median()
        tenure_median = df_analysis[tenure_col].median()
        logger.info(f"Медианное значение '{balance_col}': {balance_median:.2f}")
        logger.info(f"Медианное значение '{tenure_col}': {tenure_median:.0f}")
        conditions = [
            (df_analysis[balance_col] <= balance_median) & (df_analysis[tenure_col] <= tenure_median),
            (df_analysis[balance_col] > balance_median) & (df_analysis[tenure_col] <= tenure_median),
            (df_analysis[balance_col] <= balance_median) & (df_analysis[tenure_col] > tenure_median)
        ]
        choices = ['Низкий баланс, Малый срок', 'Высокий баланс, Малый срок', 'Низкий баланс, Большой срок']
        df_analysis['CustomerSegment'] = np.select(conditions, choices, default='Высокий баланс, Большой срок')
        churn_analysis_df = df_analysis.groupby(['CustomerSegment', is_active_col])[target_col].mean().reset_index()
        churn_pivot = churn_analysis_df.pivot(index='CustomerSegment', columns=is_active_col, values=target_col)
        churn_pivot.columns = ['Churn (Неактивные)', 'Churn (Активные)']
        churn_pivot['Delta'] = churn_pivot['Churn (Неактивные)'] - churn_pivot['Churn (Активные)']
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=churn_analysis_df, x='CustomerSegment', y=target_col, hue=is_active_col, ax=ax, palette=palette)
        ax.set_title('Уровень оттока в зависимости от сегмента и активности клиента')
        ax.set_xlabel('Сегмент клиента (Баланс / Срок обслуживания)')
        ax.set_ylabel('Доля ушедших клиентов (Churn Rate)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['Неактивные', 'Активные'], title='Статус активности', loc='upper left')
        plt.xticks(rotation=10, ha='right')
        plt.tight_layout()
        plt.show()
        return churn_pivot

    @staticmethod
    def plot_churn_heatmap_by_details(
        df: pd.DataFrame, products_col: str = 'NumOfProducts', salary_col: str = 'EstimatedSalary',
        card_col: str = 'HasCrCard', target_col: str = 'Exited',
        figsize: Tuple[int, int] = (12, 8), cmap: Optional[str] = None
    ) -> pd.DataFrame:
        """Создает тепловую карту оттока: Продукты × Карта × Зарплата."""
        df_analysis = df.copy()
        conditions = [df_analysis[products_col] == 1, df_analysis[products_col] == 2]
        choices = ['1 продукт', '2 продукта']
        df_analysis['product_cat'] = np.select(conditions, choices, default='3+ продукта')
        salary_labels = ['Низкая ЗП', 'Средняя ЗП', 'Высокая ЗП']
        df_analysis['salary_tertile'] = pd.qcut(df_analysis[salary_col], q=3, labels=salary_labels)
        churn_heatmap_data = pd.pivot_table(df_analysis, values=target_col,
                                            index=['product_cat', card_col],
                                            columns=['salary_tertile'], aggfunc=np.mean)
        churn_heatmap_data.index = churn_heatmap_data.index.set_levels(['Нет карты', 'Есть карта'], level=card_col)
        plt.figure(figsize=figsize)
        sns.heatmap(churn_heatmap_data, annot=True, fmt=".2%",
                    cmap=cmap if cmap else MTS_HSE_CMAP,
                    linewidths=.5, linecolor='black')
        plt.title('Тепловая карта оттока: Продукты × Кредитная карта × Зарплата')
        plt.xlabel('Уровень зарплаты (по тертилям)')
        plt.ylabel('Кол-во продуктов и наличие карты')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        return churn_heatmap_data

    @staticmethod
    def plot_churn_by_geo_age_score(
        df: pd.DataFrame, age_col: str = 'Age', score_col: str = 'CreditScore',
        geo_col: str = 'Geography', target_col: str = 'Exited',
        height: float = 6, aspect: float = 0.8
    ):
        """Анализирует влияние кредитного рейтинга на отток в разных группах и странах."""
        df_analysis = df.copy()
        age_bins = [0, 35, 60, df_analysis[age_col].max() + 1]
        age_labels = ['Молодые (<35)', 'Средний возраст (35-60)', 'Пожилые (>60)']
        df_analysis['age_group'] = pd.cut(df_analysis[age_col], bins=age_bins, labels=age_labels, right=False)
        score_labels = ['Q1 (Низкий)', 'Q2', 'Q3', 'Q4 (Высокий)']
        df_analysis['score_quantile'] = pd.qcut(df_analysis[score_col], 4, labels=score_labels, duplicates='drop')
        aggregated_df = df_analysis.groupby([geo_col, 'age_group', 'score_quantile'])[target_col].mean().reset_index()
        markers = ['o', 's', '^']
        linestyles = ['-', '--', ':']
        g = sns.catplot(
            data=aggregated_df, x='score_quantile', y=target_col, hue='age_group', col=geo_col,
            kind='point', height=height, aspect=aspect, palette=CORPORATE_PALETTE,
            legend=False, markers=markers, linestyles=linestyles, errorbar=None
        )
        g.fig.suptitle('Влияние кредитного рейтинга на отток в разных группах и странах', y=1.03)
        g.set_axis_labels('Квартиль кредитного рейтинга', 'Уровень оттока')
        g.set_titles("Страна: {col_name}")
        g.despine(left=True)
        for ax in g.axes.flat:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        handles = [Line2D([0], [0], color=c, marker=m, linestyle=ls, label=l)
                   for c, m, ls, l in zip(CORPORATE_PALETTE, markers, linestyles, age_labels)]
        g.fig.legend(handles=handles, title='Возрастная группа', loc='lower center',
                     bbox_to_anchor=(0.5, -0.02), ncol=len(age_labels), frameon=False)
        g.fig.subplots_adjust(bottom=0.18, top=0.9)
        plt.show()

    @staticmethod
    def plot_inactive_churn_heatmap(
        df: pd.DataFrame, is_active_col: str = 'IsActiveMember', balance_col: str = 'Balance',
        salary_col: str = 'EstimatedSalary', target_col: str = 'Exited',
        figsize: Tuple[int, int] = (12, 8)
    ) -> pd.DataFrame:
        """Создает и визуализирует тепловую карту оттока среди неактивных клиентов."""
        df_inactive = df[df[is_active_col] == 0].copy()
        salary_labels = ['1. Низкая ЗП', '2. Средняя ЗП', '3. Высокая ЗП']
        df_inactive['salary_segment'] = pd.qcut(df_inactive[salary_col], q=3, labels=salary_labels)
        balance_labels = ['1. Низкий баланс', '2. Средний баланс', '3. Высокий баланс']
        zero_balance_label = '0. Нулевой баланс'
        non_zero_balance_mask = df_inactive[balance_col] > 0
        balance_tertiles = pd.qcut(df_inactive.loc[non_zero_balance_mask, balance_col], q=3, labels=balance_labels)
        df_inactive['balance_segment'] = balance_tertiles
        df_inactive['balance_segment'] = df_inactive['balance_segment'].cat.add_categories([zero_balance_label])
        df_inactive['balance_segment'].fillna(zero_balance_label, inplace=True)
        pivot_df = pd.pivot_table(df_inactive, values=target_col, index='balance_segment',
                                  columns='salary_segment', aggfunc='mean').sort_index(ascending=False)
        plt.figure(figsize=figsize)
        heatmap = sns.heatmap(pivot_df, annot=True, fmt='.1%', cmap=MTS_RED_CMAP,
                              linewidths=.5, linecolor='black', annot_kws={"size": 12})
        heatmap.set_title('Отток среди НЕАКТИВНЫХ клиентов\n(в разрезе баланса и зарплаты)')
        heatmap.set_xlabel('Сегмент по зарплате')
        heatmap.set_ylabel('Сегмент по балансу')
        plt.tight_layout()
        plt.show()
        return pivot_df




class HyperparameterTuner:
    """Класс-утилита для подбора гиперпараметров модели CatBoost с использованием Optuna."""

    @staticmethod
    def _objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cat_feature_names: List[str]) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 400, 2500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
            "random_seed": Config.RANDOM_STATE, "eval_metric": "AUC", "od_type": "Iter",
            "early_stopping_rounds": 50, "verbose": 0,
        }
        cv = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
        cv_aucs = [
            roc_auc_score(
                y.iloc[valid_idx],
                CatBoostClassifier(**params).fit(
                    X.iloc[train_idx], y.iloc[train_idx],
                    eval_set=(X.iloc[valid_idx], y.iloc[valid_idx]),
                    cat_features=cat_feature_names,
                    callbacks=[CatBoostPruningCallback(trial, "AUC")]
                ).predict_proba(X.iloc[valid_idx])[:, 1]
            ) for train_idx, valid_idx in cv.split(X, y)
        ]
        return np.mean(cv_aucs)

    @staticmethod
    def tune(X: pd.DataFrame, y: pd.Series, cat_feature_names: List[str], n_trials: int) -> Dict[str, Any]:
        logger.info(f"Запуск подбора гиперпараметров. Количество триалов: {n_trials}.")
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        objective_with_args = lambda trial: HyperparameterTuner._objective(trial, X, y, cat_feature_names)
        study.optimize(objective_with_args, n_trials=n_trials, show_progress_bar=True)
        logger.info(f"Подбор завершен. Лучший результат (ROC AUC): {study.best_value:.5f}")
        logger.info(f"Лучшие параметры: {study.best_params}")
        plot_optimization_history(study).show()
        plot_param_importances(study).show()
        logger.info(f"Лучшие парметры : {study.best_params}.")
        return study.best_params


class ErrorAnalyzer:
    """Класс-инструментарий для всестороннего анализа ошибок моделей МО."""

    @staticmethod
    def analyze_best_model(results: Dict, y_train: pd.Series, X_train_full: pd.DataFrame) -> Tuple[
        str, pd.DataFrame, plt.Figure]:
        """Определяет лучшую модель, готовит DataFrame с ошибками и создает дашборд."""
        mean_roc_aucs = {name: res['metrics_df']['ROC AUC'].mean() for name, res in results.items()}
        best_model_name = max(mean_roc_aucs, key=mean_roc_aucs.get)
        logger.info(
            f"Лучшая модель по среднему OOF ROC AUC: {best_model_name} (AUC = {mean_roc_aucs[best_model_name]:.4f})")

        best_model_results = results[best_model_name]
        oof_preds = best_model_results['oof_preds']
        precisions, recalls, thresholds = precision_recall_curve(y_train, oof_preds)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_threshold = thresholds[np.argmax(f1_scores)]
        logger.info(f"Оптимальный порог по F1-мере: {best_threshold:.4f} (F1 = {np.max(f1_scores):.4f})")

        oof_labels = (oof_preds >= best_threshold).astype(int)
        fp_mask = (y_train == 0) & (oof_labels == 1)
        fn_mask = (y_train == 1) & (oof_labels == 0)

        fp_df = X_train_full.loc[fp_mask].copy();
        fp_df['error_type'] = 'False Positive'
        fn_df = X_train_full.loc[fn_mask].copy();
        fn_df['error_type'] = 'False Negative'

        error_analysis_df = pd.concat([fp_df, fn_df])
        error_analysis_df['true_label'] = y_train.loc[error_analysis_df.index]
        error_analysis_df['predicted_proba'] = pd.Series(oof_preds, index=y_train.index).loc[error_analysis_df.index]

        feat_imp_series = pd.Series(best_model_results.get('feature_importances'),
                                    index=best_model_results.get('feature_names')) if best_model_results.get(
            'feature_importances') is not None else None

        dashboard_figure = ErrorAnalyzer._create_full_analysis_dashboard(y_train, oof_preds, best_threshold,
                                                                         best_model_name, feat_imp_series)

        return best_model_name, error_analysis_df, dashboard_figure

    @staticmethod
    def _create_full_analysis_dashboard(y_true: pd.Series, y_pred_proba: np.ndarray, threshold: float, model_name: str,
                                        feat_imp_series: Optional[pd.Series]) -> plt.Figure:
        fig = plt.figure(figsize=(22, 14), constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        y_pred_labels = (y_pred_proba >= threshold).astype(int)

        ax1 = fig.add_subplot(gs[0, 0]);
        ErrorAnalyzer._plot_roc_curve(y_true, y_pred_proba, ax1)
        ax2 = fig.add_subplot(gs[0, 1]);
        ErrorAnalyzer._plot_pr_curve(y_true, y_pred_proba, ax2)
        ax3 = fig.add_subplot(gs[0, 2]);
        ErrorAnalyzer._plot_confusion_matrix(y_true, y_pred_labels, threshold, ax3)
        ax4 = fig.add_subplot(gs[1, 0]);
        ErrorAnalyzer._plot_prediction_distribution(y_true, y_pred_proba, ax4)
        ax5 = fig.add_subplot(gs[1, 1]);
        ErrorAnalyzer._plot_calibration_curve(y_true, y_pred_proba, model_name, ax5)
        ax6 = fig.add_subplot(gs[1, 2]);
        ErrorAnalyzer._plot_feature_importance(feat_imp_series, ax6)

        fig.suptitle(f'Полный анализ модели: {model_name}', fontsize=24, y=1.03)
        return fig

    @staticmethod
    def _plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray, ax: plt.Axes) -> None:
        """Строит ROC-кривую на переданных осях (ax)."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MTS_RED, lw=2, label=f'AUC = {roc_auc:.4f}')
        ax.plot([0, 1], [0, 1], color=HSE_DARKBLUE, lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True)

    @staticmethod
    def _plot_pr_curve(y_true: pd.Series, y_pred_proba: np.ndarray, ax: plt.Axes) -> None:
        """Строит Precision-Recall кривую на переданных осях (ax)."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        ax.plot(recall, precision, color=HSE_DARKBLUE, lw=2, label=f'AP = {avg_precision:.4f}')
        random_level = y_true.mean()
        ax.axhline(y=random_level, color=MTS_RED, linestyle='--', lw=2, label=f'Random ({random_level:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="best")
        ax.grid(True)

    @staticmethod
    def _plot_prediction_distribution(y_true: pd.Series, y_pred_proba: np.ndarray, ax: plt.Axes) -> None:
        """Строит распределение вероятностей на переданных осях (ax)."""
        preds_df = pd.DataFrame({'true_label': y_true, 'probability': y_pred_proba})
        palette_map = {0: HSE_DARKBLUE, 1: MTS_RED}
        sns.histplot(data=preds_df, x='probability', hue='true_label',
                     kde=True, common_norm=False, stat='density', ax=ax, palette=palette_map)
        ax.set_title('Prediction Probabilities Distribution')
        ax.set_xlabel('Predicted Probability of class 1')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, title='True Label', labels=['Class 0 (Stayed)', 'Class 1 (Exited)'])

    @staticmethod
    def _plot_calibration_curve(y_true: pd.Series, y_pred_proba: np.ndarray, model_name: str, ax: plt.Axes) -> None:
        """Строит калибровочную кривую на переданных осях (ax)."""
        disp = CalibrationDisplay.from_predictions(y_true, y_pred_proba, n_bins=15,
                                                   name=model_name, ax=ax, strategy='uniform')
        disp.line_.set_color(MTS_RED)
        disp.ax_.get_lines()[1].set_color(HSE_DARKBLUE)
        disp.ax_.get_lines()[1].set_linestyle('--')
        ax.set_title('Calibration Curve')
        ax.grid(True)

    @staticmethod
    def _plot_confusion_matrix(y_true: pd.Series, y_pred_labels: np.ndarray, threshold: float, ax: plt.Axes) -> None:
        """Строит матрицу ошибок на переданных осях (ax)."""
        cm = confusion_matrix(y_true, y_pred_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Exited'])
        disp.plot(cmap=MTS_RED_CMAP, ax=ax, values_format='d')
        ax.set_title(f'Confusion Matrix (thr={threshold:.3f})')
        ax.grid(False)

    @staticmethod
    def _plot_feature_importance(feat_imp_series: Optional[pd.Series], ax: plt.Axes) -> None:
        """Строит график важности признаков на переданных осях (ax)."""
        ax.set_title('Top 20 Feature Importances')
        if feat_imp_series is not None and not feat_imp_series.empty:
            feat_imp_series.nlargest(20).sort_values().plot.barh(ax=ax, color=HSE_DARKBLUE)
            ax.grid(axis='x')
        else:
            ax.text(0.5, 0.5, 'Feature importances not available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_yticks([])
            ax.set_xticks([])

    @staticmethod
    def run_network_error_analysis(
            error_analysis_df: pd.DataFrame,
            similarity_threshold: float = 0.7,
            top_k_neighbors: int = 5
    ) -> None:
        """
        Главный метод для запуска и визуализации полного цикла сетевого анализа ошибок.
        Разделяет ошибки на FP и FN, строит для каждой группы графы, анализирует
        кластеры и выводит результаты.

        Args:
            error_analysis_df (pd.DataFrame): DataFrame с ошибками, полученный из analyze_best_model.
            similarity_threshold (float): Порог косинусного сходства для создания ребра в графе.
            top_k_neighbors (int): Количество ближайших соседей для рассмотрения.
        """
        logger.info("\n" + "=" * 25 + " ЗАПУСК СЕТЕВОГО АНАЛИЗА ОШИБОК " + "=" * 25)
        if 'error_type' not in error_analysis_df.columns:
            logger.error("DataFrame с ошибками должен содержать колонку 'error_type'.")
            return

        # --- Анализ для False Negatives ---
        fn_centrality, fn_clusters, fn_features = ErrorAnalyzer._run_single_type_analysis(
            error_analysis_df, 'False Negative', similarity_threshold, top_k_neighbors
        )

        # --- Анализ для False Positives ---
        fp_centrality, fp_clusters, fp_features = ErrorAnalyzer._run_single_type_analysis(
            error_analysis_df, 'False Positive', similarity_threshold, top_k_neighbors
        )


        # --- Анализ портретов крупнейших кластеров ---
        logger.info("\n" + "=" * 20 + " АНАЛИЗ ПОРТРЕТОВ КЛАСТЕРОВ FN " + "=" * 20)
        global_fn_portrait = fn_features.describe().T
        ErrorAnalyzer._analyze_cluster_portraits(fn_clusters, fn_features, global_fn_portrait, 'False Negative')

        logger.info("\n" + "=" * 20 + " АНАЛИЗ ПОРТРЕТОВ КЛАСТЕРОВ FP " + "=" * 20)
        global_fp_portrait = fp_features.describe().T
        ErrorAnalyzer._analyze_cluster_portraits(fp_clusters, fp_features, global_fp_portrait, 'False Positive')

    @staticmethod
    def _run_single_type_analysis(df: pd.DataFrame, error_type: str, threshold: float, top_k: int):
        """Вспомогательный пайплайн для анализа одного типа ошибок (FP или FN)."""
        logger.info(f"\n--- Анализ для типа: {error_type.upper()} ---")

        df_err = df[df['error_type'] == error_type].reset_index(drop=True)
        if df_err.empty:
            logger.warning(f"Ошибки типа '{error_type}' не найдены. Пропуск анализа.")
            return None, None, None


        drop_cols = ['id', 'Surname', 'true_label', 'predicted_proba', 'error_type']
        df_features = df_err.drop(columns=drop_cols, errors='ignore').select_dtypes(exclude=['category', 'object'])
        df_features_dummies = pd.get_dummies(df_err.drop(columns=drop_cols, errors='ignore'), drop_first=True)


        G = ErrorAnalyzer._build_similarity_graph(df_features_dummies, threshold=threshold, top_k=top_k)
        if G.number_of_nodes() == 0:
            logger.warning(
                f"Не удалось построить граф для {error_type}. Возможно, нет ошибок или порог сходства слишком высок.")
            return None, None, df_features

        centrality, clusters = ErrorAnalyzer._analyze_graph_communities(G)


        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        ErrorAnalyzer._plot_cluster_sizes(clusters, error_type=error_type, ax=ax)
        plt.tight_layout()
        plt.show()

        return centrality, clusters, df_features

    @staticmethod
    def _build_similarity_graph(df_features: pd.DataFrame, threshold: float, top_k: int) -> nx.Graph:
        """Строит граф на основе косинусного сходства между строками (ошибками)."""
        logger.info(f"Построение графа сходства: {df_features.shape[0]} узлов, threshold={threshold}, top_k={top_k}")
        sim_matrix = cosine_similarity(df_features.fillna(0))
        n = sim_matrix.shape[0]
        G = nx.Graph()

        for i in range(n):

            top_k_indices = np.argsort(sim_matrix[i])[-(top_k + 1):-1]
            for j in top_k_indices:
                if i != j and sim_matrix[i, j] >= threshold:
                    G.add_edge(i, j, weight=float(sim_matrix[i, j]))

        logger.info(f"Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} ребер.")
        return G

    @staticmethod
    def _analyze_graph_communities(G: nx.Graph) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        """Вычисляет центральности и находит сообщества (кластеры) в графе."""
        logger.info("Расчет метрик центральности и поиск сообществ...")
        centrality = pd.DataFrame({
            'degree': pd.Series(nx.degree_centrality(G)),
            'betweenness': pd.Series(nx.betweenness_centrality(G, seed=Config.RANDOM_STATE)),
        })
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        clusters = {i: sorted(list(c)) for i, c in enumerate(communities)}
        clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True))
        logger.info(f"Найдено {len(clusters)} сообществ.")
        return centrality, clusters

    @staticmethod
    def _plot_cluster_sizes(clusters: Dict[int, List[int]], error_type: str, ax: plt.Axes):
        """Строит гистограмму размеров кластеров в корпоративном стиле."""
        sizes = [len(nodes) for nodes in clusters.values()]
        color = Config.MTS_RED if "Negative" in error_type else Config.HSE_DARKBLUE
        sns.barplot(x=list(range(len(sizes))), y=sizes, color=color, ax=ax)
        ax.set_xlabel('ID Кластера (отсортированы по размеру)')
        ax.set_ylabel('Количество ошибок в кластере')
        ax.set_title(f'Размеры кластеров для ошибок типа: {error_type}')

    @staticmethod
    def _analyze_cluster_portraits(clusters: Dict, features_df: pd.DataFrame, global_portrait: pd.DataFrame,
                                   error_type: str, top_n_clusters: int = 4):
        """Анализирует и выводит портреты крупнейших кластеров."""
        if not clusters:
            return
        clusters_to_analyze = list(clusters.keys())[:top_n_clusters]
        for cid in clusters_to_analyze:
            node_indices = clusters.get(cid)
            if not node_indices: continue
            cluster_portrait = features_df.iloc[node_indices].describe().T
            ErrorAnalyzer._find_and_display_deviating_features(cluster_portrait, global_portrait, cid, error_type)

    @staticmethod
    def _find_and_display_deviating_features(cluster_portrait: pd.DataFrame, global_portrait: pd.DataFrame,
                                             cluster_id: int, error_type: str):
        """Сравнивает статистику кластера с глобальной и выводит отклонения в виде стилизованной таблицы."""
        comparison = cluster_portrait.join(global_portrait, lsuffix='_cluster', rsuffix='_global')
        comparison['mean_diff_std'] = (comparison['mean_cluster'] - comparison['mean_global']) / (
                    comparison['std_global'] + 1e-9)

        print(f"\n--- Анализ отклонений для кластера {cluster_id} ({error_type}) ---")
        display_df = comparison.sort_values(by='mean_diff_std', key=abs, ascending=False)[
            ['mean_cluster', 'mean_global', 'std_cluster', 'std_global', 'mean_diff_std']
        ].head(5)

        styled_df = display_df.style.format({
            'mean_cluster': '{:.2f}', 'mean_global': '{:.2f}',
            'std_cluster': '{:.2f}', 'std_global': '{:.2f}',
            'mean_diff_std': '{:+.2f}σ'
        }).set_caption(
            f"Топ-5 отклонений для кластера {cluster_id}"
        ).background_gradient(
            cmap=MTS_HSE_CMAP, subset=['mean_diff_std'], vmin=-3, vmax=3
        )
        display(styled_df)

class ModelTrainer:
    """Класс-оркестратор для унификации процесса обучения, оценки и предсказания."""

    def __init__(self, n_splits: int = Config.N_SPLITS, random_state: int = Config.RANDOM_STATE):
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def run_experiment_tune(self,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_engineering_pipeline: Callable,
                       models: Dict,
                       target_col: str = Config.TARGET_COL,
                       tune_hyperparams: bool = False,
                       tune_model_name: str = 'CatBoost',
                       n_trials: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
        """
        Запускает полный цикл эксперимента с опциональным подбором гиперпараметров.
        """
        self.logger.info(f"--- ЗАПУСК НОВОГО ЭКСПЕРИМЕНТА (FE: {feature_engineering_pipeline.__name__}) ---")
        if tune_hyperparams:
            self.logger.info(f"!!! РЕЖИМ ПОДБОРА ГИПЕРПАРАМЕТРОВ АКТИВИРОВАН для модели '{tune_model_name}' !!!")

        test_ids = test_df['id'].copy()
        original_train_for_analysis = train_df.copy()
        y_train = train_df[target_col].astype(int)


        self.logger.info("Шаг 1: Применение инженерии признаков...")
        X_train_processed = feature_engineering_pipeline(train_df, is_train=True)
        X_test_processed = feature_engineering_pipeline(test_df, is_train=False)


        train_cols = X_train_processed.columns
        test_cols = X_test_processed.columns
        if not train_cols.equals(test_cols):
            self.logger.warning("Колонки в train и test не совпадают! Выравнивание...")
            shared_cols = list(train_cols.intersection(test_cols))
            X_train_processed = X_train_processed[shared_cols]
            X_test_processed = X_test_processed[shared_cols]


        models_to_train = models.copy()

        if tune_hyperparams:
            if tune_model_name not in models:
                self.logger.error(
                    f"Модель '{tune_model_name}' для подбора параметров не найдена в словаре models. Тюнинг отменен.")
            else:
                self.logger.info(f"Шаг 1.5: Подбор гиперпараметров для '{tune_model_name}'...")

                cat_features = X_train_processed.select_dtypes(include=['category', 'object']).columns.tolist()


                best_params = HyperparameterTuner.tune(
                    X=X_train_processed,
                    y=y_train,
                    cat_feature_names=cat_features,
                    n_trials=n_trials
                )


                best_params['random_seed'] = self.random_state
                best_params['verbose'] = 0
                if 'early_stopping_rounds' not in best_params:
                    best_params['early_stopping_rounds'] = 50

                tuned_model = CatBoostClassifier(**best_params)


                tuned_model_name = f"{tune_model_name}_Tuned"
                models_to_train = {tuned_model_name: tuned_model}
                self.logger.info(f"Подбор завершен. Модель '{tuned_model_name}' будет использована для обучения.")

        # 2. Обучение и оценка моделей
        self.logger.info("Шаг 2: Обучение моделей на кросс-валидации...")
        all_results = self._evaluate_models(models_to_train, X_train_processed, y_train, X_test_processed)

        # 3. Анализ ошибок
        self.logger.info("Шаг 3: Анализ ошибок лучшей модели...")
        best_model_name, error_df, dashboard_figure = ErrorAnalyzer.analyze_best_model(
            all_results, y_train, original_train_for_analysis
        )

        # 4. Генерация сабмита (без изменений)
        self.logger.info("Шаг 4: Генерация файла для сабмита...")
        submission_df = self._generate_submission(
            f"submission_{best_model_name}_{feature_engineering_pipeline.__name__}.csv",
            test_ids,
            all_results[best_model_name]['test_preds']
        )

        self.logger.info("--- ТЮНИНГ УСПЕШНО ЗАВЕРШЕН ---")
        return submission_df, all_results, error_df, dashboard_figure

    def run_experiment(self,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_engineering_pipeline: Callable,
                       models: Dict,
                       target_col: str = Config.TARGET_COL) -> Tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
        """
        Запускает полный цикл эксперимента: FE, обучение, анализ ошибок, сабмит.

        Args:
            train_df: Исходный тренировочный DataFrame.
            test_df: Исходный тестовый DataFrame.
            feature_engineering_pipeline: Функция из класса FeatureEngineer (например, FeatureEngineer.run_v1_preprocessing).
            models: Словарь с моделями для обучения.
            target_col: Название целевой переменной.

        Returns:
            Кортеж (submission_df, error_analysis_df, dashboard_figure).
        """
        self.logger.info(f"--- ЗАПУСК НОВОГО ЭКСПЕРИМЕНТА (FE: {feature_engineering_pipeline.__name__}) ---")

        test_ids = test_df['id'].copy()
        original_train_for_analysis = train_df.copy()
        y_train = train_df[target_col].astype(int)

        # 1. Инженерия признаков
        self.logger.info("Шаг 1: Применение инженерии признаков...")
        X_train_processed = feature_engineering_pipeline(train_df, is_train=True)
        X_test_processed = feature_engineering_pipeline(test_df, is_train=False)

        train_cols = X_train_processed.columns
        test_cols = X_test_processed.columns
        if not train_cols.equals(test_cols):
            self.logger.warning("Колонки в train и test не совпадают! Выравнивание...")
            shared_cols = list(train_cols.intersection(test_cols))
            X_train_processed = X_train_processed[shared_cols]
            X_test_processed = X_test_processed[shared_cols]

        # 2. Обучение и оценка моделей
        self.logger.info("Шаг 2: Обучение моделей на кросс-валидации...")
        all_results = self._evaluate_models(models, X_train_processed, y_train, X_test_processed)

        # 3. Анализ ошибок
        self.logger.info("Шаг 3: Анализ ошибок лучшей модели...")
        best_model_name, error_df, dashboard_figure = ErrorAnalyzer.analyze_best_model(
            all_results, y_train, original_train_for_analysis
        )

        # 4. Генерация сабмита
        self.logger.info("Шаг 4: Генерация файла для сабмита...")
        file_name = f"submission_{best_model_name}_{feature_engineering_pipeline.__name__}.csv"
        submission_df = self._generate_submission(
            f"submission_{best_model_name}_{feature_engineering_pipeline.__name__}.csv",
            test_ids,
            all_results[best_model_name]['test_preds']
        )

        self.logger.info("--- ЭКСПЕРИМЕНТ УСПЕШНО ЗАВЕРШЕН ---")
        return submission_df, all_results, error_df, dashboard_figure


    def _evaluate_models(self, models: Dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Dict:
        """
        Обучает и валидирует модели с использованием кросс-валидации.
        Корректно обрабатывает CatBoost, XGBoost, LightGBM и sklearn-модели.
        """
        self.logger.info("Запуск кросс-валидации...")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        results = {}


        cat_feature_names = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
        if cat_feature_names:
            self.logger.info(f"Обнаружены категориальные признаки: {cat_feature_names}")

        for name, model in models.items():
            self.logger.info(f"Обучение модели: {name}")
            oof_preds = np.zeros(len(X_train))
            test_preds_folds, fold_metrics_list, importances_folds = [], [], []



            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                current_model = clone(model)
                fit_params = {}

                X_tr_fit, X_val_fit = X_tr, X_val

                if isinstance(current_model, (XGBClassifier, lgb.LGBMClassifier, CatBoostClassifier)):
                    fit_params['eval_set'] = [(X_val_fit, y_val)]

                    if isinstance(current_model, lgb.LGBMClassifier):

                        fit_params['callbacks'] = [lgb.early_stopping(50, verbose=False)]
                        fit_params['categorical_feature'] = 'auto'

                    elif isinstance(current_model, XGBClassifier):

                        fit_params['verbose'] = False

                    elif isinstance(current_model, CatBoostClassifier):
                        fit_params['cat_features'] = cat_feature_names


                elif cat_feature_names and not hasattr(current_model, 'cat_features'):
                    self.logger.info(f"  > Модель '{name}' требует ручного кодирования. Применяем .cat.codes.")
                    X_tr_fit, X_val_fit = X_tr.copy(), X_val.copy()
                    for col in cat_feature_names:

                        all_categories = pd.concat([X_train[col], X_test[col]]).astype('category').cat.categories
                        X_tr_fit[col] = pd.Categorical(X_tr_fit[col], categories=all_categories).codes
                        X_val_fit[col] = pd.Categorical(X_val_fit[col], categories=all_categories).codes

                # Обучение модели
                current_model.fit(X_tr_fit, y_tr, **fit_params)

                # Предсказания
                X_test_predict = X_test.copy()
                if not X_tr.equals(X_tr_fit):
                    for col in cat_feature_names:
                        all_categories = pd.concat([X_train[col], X_test[col]]).astype('category').cat.categories
                        X_test_predict[col] = pd.Categorical(X_test_predict[col], categories=all_categories).codes

                proba_val = current_model.predict_proba(X_val_fit)[:, 1]
                proba_test = current_model.predict_proba(X_test_predict)[:, 1]

                oof_preds[val_idx] = proba_val
                test_preds_folds.append(proba_test)

                # Сбор метрик и важности
                fold_metrics_list.append(
                    {'ROC AUC': roc_auc_score(y_val, proba_val), 'PR AUC': average_precision_score(y_val, proba_val)})
                if hasattr(current_model, 'feature_importances_'):
                    importances_folds.append(current_model.feature_importances_)
                elif hasattr(current_model, 'coef_'):
                    importances_folds.append(np.abs(current_model.coef_[0]))

            results[name] = {
                'oof_preds': oof_preds,
                'test_preds': np.mean(test_preds_folds, axis=0),
                'metrics_df': pd.DataFrame(fold_metrics_list),
                'feature_importances': np.mean(importances_folds, axis=0) if importances_folds else None,
                'feature_names': X_train.columns
            }
            self.logger.info(
                f"  Модель {name} | CV ROC AUC: {results[name]['metrics_df']['ROC AUC'].mean():.4f} ± {results[name]['metrics_df']['ROC AUC'].std():.4f}")
        return results

    def _generate_submission(self, filename: str, df_test_id: pd.Series, test_preds: np.ndarray) -> pd.DataFrame:
        print(f'filename = {filename}')
        if filename == 'submission_CatBoost_final_run_v3_preprocessing.csv':
            filename = 'submission.csv'
        print(f'filename1 = {filename}')
        submission_df = pd.DataFrame({'id': df_test_id, 'Exited': test_preds})
        submission_df.to_csv(filename, index=False)
        self.logger.info(f"Файл для сабмита успешно сохранен: {filename}")
        return submission_df



# Загружаем данные
df_train = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
df_test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')

# Создаем экземпляры наших классов-помощников
visualizer = EDAVisualizer()
feature_engineer = FeatureEngineer()



# --- 1. Анализ тренировочного набора данных ---
print("=" * 40)
print("     Сводный отчет по TRAIN данным      ")
print("=" * 40)
train_summary = visualizer.create_summary_report(df_train)
display(train_summary)

# --- 2. Анализ тестового набора данных ---
print("\n" * 2 + "=" * 40)
print("      Сводный отчет по TEST данным      ")
print("=" * 40)
test_summary = visualizer.create_summary_report(df_test)
display(test_summary)


fig, ax = plt.subplots(figsize=(10, 7))
visualizer.plot_target_distribution(data=df_train, target_col='Exited', ax=ax)
plt.tight_layout()
plt.show()


numerical_features = df_train.select_dtypes(include=np.number).columns.drop('Exited').tolist()
categorical_features = df_train.select_dtypes(include='object').columns.tolist()
binary_features = ['HasCrCard', 'IsActiveMember']

visualizer.plot_categorical_analysis(df_train, features=['Geography', 'Gender'] + binary_features, target=Config.TARGET_COL)


numerical_features = df_train.select_dtypes(include=np.number).columns.drop([Config.TARGET_COL, 'id', 'CustomerId']).tolist()
visualizer.plot_numerical_summary(df_train, numerical_features)


features_for_corr = numerical_features + [Config.TARGET_COL]
visualizer.plot_correlation_heatmap(df_train, features_for_corr)


visualizer.plot_categorical_analysis(df_train, features=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'], target='Exited')


visualizer.plot_numerical_distributions_with_target(df_train, numerical_features, 'Exited')



visualizer.plot_credit_score_analysis_by_country(
    df=df_train,
    score_col='CreditScore',
    target_col='Exited',
    geo_col='Geography'
)


visualizer.plot_age_distribution_by_geo_and_churn(
    df=df_train,
    age_col='Age',
    geo_col='Geography',
    target_col='Exited'
)


visualizer.plot_bivariate_categorical_churn_analysis(
    df=df_train,
    cat_col1='NumOfProducts',
    cat_col2='Geography',
    target_col='Exited'
)


visualizer.analyze_salary_interaction(
    df=df_train,
    salary_col='EstimatedSalary',
    target_col='Exited'
)


## Активен или неактивен
visualizer.plot_bivariate_categorical_churn_analysis(
    df=df_train,
    cat_col1='IsActiveMember',
    cat_col2='Geography',
    target_col='Exited'
)





visualizer.plot_bivariate_categorical_churn_analysis(
    df=df_train,
    cat_col1='Gender',
    cat_col2='Geography',
    target_col='Exited'
)


visualizer.plot_multidimensional_churn_analysis(
    df=df_train,
    target_col='Exited',
    geo_col='Geography',
    gender_col='Gender',
    activity_col='IsActiveMember'
)



visualizer.plot_numeric_distribution_by_two_categories(
        df=df_train,
        numeric_col='Balance',
        group_col_1='Geography',
        group_col_2='Exited',
        title_prefix='Распределение переменных '
    )


visualizer.plot_churn_by_age_groups_and_geo(df_train)


churn_report_table = visualizer.plot_customer_quadrant_analysis(df_train)

display(
    churn_report_table
        .style
        .format("{:.2%}")
        .background_gradient(cmap=MTS_HSE_CMAP)
)



final_churn_report = visualizer.plot_churn_heatmap_by_details(df_train)

display(final_churn_report.style.format("{:.2%}").background_gradient(cmap=MTS_HSE_CMAP))


visualizer.plot_churn_by_geo_age_score(df_train)


visualizer.plot_inactive_churn_heatmap(df_train)


trainer = ModelTrainer()
models = {
    'CatBoost': CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6,
        eval_metric='AUC', random_seed=42,
        od_type='Iter', early_stopping_rounds=50, verbose=0
    ),
    'XGBoost': XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=6,
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=-1,
        early_stopping_rounds=50,
        enable_categorical=True
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=31,
        random_state=42, n_jobs=-1,
        colsample_bytree=0.7, subsample=0.7, reg_alpha=0.1, reg_lambda=0.1, verbose=-1

    ),
    'LogisticRegression': LogisticRegression(
        solver='liblinear', random_state=42, C=0.1, penalty='l1'
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
    )
}



submission_v0, results_v0, errors_v0, dashboard_v0 = trainer.run_experiment(
    train_df=df_train,
    test_df=df_test,
    feature_engineering_pipeline=FeatureEngineer.run_v0_baseline,
    models=models
)



dashboard_v0.show()




submission_v1, results_v1, errors_v1, dashboard_v1 = trainer.run_experiment(
    train_df=df_train,
    test_df=df_test,
    feature_engineering_pipeline=FeatureEngineer.run_v1_preprocessing,
    models={'CatBoost_v1': models['CatBoost']}
)



dashboard_v1.show()




mi_df, dcor_df, pcorr_df = visualizer.analyze_feature_dependencies(df_train)


ErrorAnalyzer.run_network_error_analysis(
        error_analysis_df=errors_v1,
        similarity_threshold=0.8,
        top_k_neighbors=7
    )





submission_v2, results_v2, errors_v2, dashboard_v2 = trainer.run_experiment(
    train_df=df_train,
    test_df=df_test,
    feature_engineering_pipeline=FeatureEngineer.run_v2_preprocessing,
    models={'CatBoost_v2': models['CatBoost']}
)



dashboard_v2.show()





submission_v3, results_v3, errors_v3, dashboard_v3 = trainer.run_experiment(
    train_df=df_train,
    test_df=df_test,
    feature_engineering_pipeline=FeatureEngineer.run_v3_preprocessing,
    models={'CatBoost_v3': models['CatBoost']}
)

dashboard_v3.show()




submission_tuned, results_tuned, errors_tuned, dashboard_tuned = trainer.run_experiment_tune(
    train_df=df_train,
    test_df=df_test,
    feature_engineering_pipeline=FeatureEngineer.run_v3_preprocessing,
    models=models,
    tune_hyperparams=True,
    tune_model_name='CatBoost',
    n_trials=2
)


dashboard_tuned.show


trainer = ModelTrainer(n_splits=10)
models = {
    'CatBoost': CatBoostClassifier(
            iterations=1000, learning_rate=0.08943219937971596, depth=6,
            eval_metric='AUC', random_seed=42,
            od_type='Iter', early_stopping_rounds=50, verbose=0
        )
}


submission_final, results_final, errors_final, dashboard_final = trainer.run_experiment(
    train_df=df_train,
    test_df=df_test,
    feature_engineering_pipeline=FeatureEngineer.run_v3_preprocessing,
    models={'CatBoost_final': models['CatBoost']}
)

dashboard_final.show()





evolution_data = [
    {
        "Этап": "v0: Baseline",
        "Описание": "Обучение на 'сырых' данных, только приведение типов.",
        "ROC AUC (CV)": results_v0['CatBoost']['metrics_df']['ROC AUC'].max()
    },
    {
        "Этап": "v1: Базовый FE",
        "Описание": "Добавлены базовые флаги (Is_two_products, Germany_Female) и биннинг.",
        "ROC AUC (CV)": results_v1['CatBoost_v1']['metrics_df']['ROC AUC'].max()
    },
    {
        "Этап": "v2: Продвинутый FE",
        "Описание": "Добавлен признак 'is_mature_inactive_transit' на основе анализа ошибок.",
        "ROC AUC (CV)": results_v2['CatBoost_v2']['metrics_df']['ROC AUC'].max()
    },
    {
        "Этап": "v3: Кодирование Surname",
        "Описание": "Удалены ID, применен Out-of-Fold Target Encoding для Surname.",
        "ROC AUC (CV)": results_v3['CatBoost_v3']['metrics_df']['ROC AUC'].max()
    },
    {
        "Этап": "v5: Финальная модель",
        "Описание": "Обучение на всех данных с лучшими параметрами и 10 фолдами.",
        "ROC AUC (CV)": results_final['CatBoost_final']['metrics_df']['ROC AUC'].max()
    }
]

evolution_df = pd.DataFrame(evolution_data)


from IPython.display import display, HTML
display(HTML(evolution_df.to_html(index=False)))



metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_train)
metadata.set_primary_key(column_name='id')

print('CUDA available:', torch.cuda.is_available())

minority_class_value = 1
minority_data = df_train[df_train['Exited'] == minority_class_value]

ctgan = CTGANSynthesizer(
    metadata,
    epochs=500,
    cuda=True,
    verbose=True
)
ctgan.fit(minority_data)

# 3. Генерация синтетических примеров
class_dist = df_train['Exited'].value_counts(normalize=True)
majority_count = class_dist[0] * len(df_train)
minority_count = class_dist[1] * len(df_train)
num_synthetic = int(majority_count - minority_count)

synthetic_data = ctgan.sample(num_rows=num_synthetic)
print(synthetic_data.head())

# Объединяем исходные данные с новыми синтетическими
df_train_augmented = pd.concat([df_train, synthetic_data], ignore_index=True)

# Перемешиваем датасет, чтобы синтетические данные не шли в конце
df_train_augmented = df_train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- Анализ дисбаланса в новом, аугментированном датасете ---")
print(df_train_augmented['Exited'].value_counts(normalize=True))


# ==============================================================================
# СРАВНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ
# ==============================================================================
trainer = ModelTrainer(n_splits=10)
models = {
    'CatBoost': CatBoostClassifier(
            iterations=1000, learning_rate=0.08943219937971596, depth=6,
            eval_metric='AUC', random_seed=42,
            od_type='Iter', early_stopping_rounds=50, verbose=0
        )
}

print("\n" + "="*30)
print("ЗАПУСК ЭКСПЕРИМЕНТА НА ИСХОДНЫХ ДАННЫХ")
print("="*30)

submission_base, results_base, _, _ = trainer.run_experiment(
    train_df=df_train.copy(),
    test_df=df_test.copy(),
    feature_engineering_pipeline=FeatureEngineer.run_v3_preprocessing,
    models={'CatBoost_base': models['CatBoost']}
)
base_metrics = results_base['CatBoost_base']['metrics_df'].mean()


print("\n" + "="*30)
print("ЗАПУСК ЭКСПЕРИМЕНТА НА АУГМЕНТИРОВАННЫХ ДАННЫХ")
print("="*30)
submission_aug, results_aug, _, _ = trainer.run_experiment(
    train_df=df_train_augmented.copy(),
    test_df=df_test.copy(),
    feature_engineering_pipeline=FeatureEngineer.run_v3_preprocessing,
    models={'CatBoost_augmented': models['CatBoost']}
)
aug_metrics = results_aug['CatBoost_augmented']['metrics_df'].mean()







### Токен уже не валидный, так как был удален из целей безопасноти - для тестирвоание введите свой токен вручную
if 'OPENAI_API_KEY' not in os.environ:
    try:
        os.environ['OPENAI_API_KEY'] =  "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"
    except Exception as e:
        print("Не удалось запустить getpass. Установите переменную окружения OPENAI_API_KEY вручную.")

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OpenAI API ключ не найден. Пожалуйста, установите его для продолжения.")


# ==============================================================================
# 2. ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ
# ==============================================================================

class SampleArgs(BaseModel):
    num_samples: int = Field(description="Количество примеров для извлечения", default=2)

def get_false_positive_samples(num_samples: int) -> str:
    """
    Извлекает случайные примеры ошибок типа 'False Positive'.
    Используй этот инструмент, чтобы получить данные для анализа ложноположительных срабатываний.
    """
    # Теперь num_samples гарантированно будет int!
    samples = errors_final[errors_final['error_type'] == 'False Positive'].sample(
        n=min(num_samples, len(errors_final[errors_final['error_type'] == 'False Positive'])),
        random_state=42
    )
    if samples.empty:
        return "Не найдено ошибок типа 'False Positive'."
    return samples.to_string()

def get_false_negative_samples(num_samples: int) -> str:
    """
    Извлекает случайные примеры ошибок типа 'False Negative'.
    Используй этот инструмент, чтобы получить данные для анализа ложноотрицательных срабатываний.
    """
    samples = errors_final[errors_final['error_type'] == 'False Negative'].sample(
        n=min(num_samples, len(errors_final[errors_final['error_type'] == 'False Negative'])),
        random_state=42
    )
    if samples.empty:
        return "Не найдено ошибок типа 'False Negative'."
    return samples.to_string()


tools_for_analyst = [
    StructuredTool.from_function(
        func=get_false_positive_samples,
        name="GetFalsePositiveSamples",
        description="Используй, чтобы получить случайные примеры ошибок типа False Positive для анализа.",
        args_schema=SampleArgs
    ),
    StructuredTool.from_function(
        func=get_false_negative_samples,
        name="GetFalseNegativeSamples",
        description="Используй, чтобы получить случайные примеры ошибок типа False Negative для анализа.",
        args_schema=SampleArgs
    ),
]

print("--- Инструменты созданы с использованием StructuredTool для надежности. ---")

# ==============================================================================
# 3. ИНИЦИАЛИЗАЦИЯ "КОМАНДЫ" АГЕНТОВ
# ==============================================================================


llm = ChatOpenAI(temperature=0.2, model_name="gpt-4-turbo-preview")

analyst_agent = initialize_agent(
    tools_for_analyst,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


prompt_template_summary = """
Ты — руководитель отдела Data Science. Ты получил детальный анализ нескольких ошибок от своего ведущего аналитика.
**ОТЧЕТ АНАЛИТИКА:**
{analyst_report}
**ТВОЯ ЗАДАЧА:**
Прочитай отчет и обобщи его. Напиши краткое, стратегическое заключение для команды.
1.  **Общие паттерны:** Выдели 1-2 общих паттерна или типа клиентов, на которых модель чаще всего ошибается.
2.  **Приоритетные направления для улучшения:** На основе всех рекомендаций, предложи 3 самых приоритетных направления для дальнейшей работы.
3.  **Итоговый вывод:** Сделай общий вывод о текущем состоянии модели и ее потенциале.
Твое стратегическое заключение:
"""
PROMPT_SUMMARY = PromptTemplate.from_template(prompt_template_summary)
output_parser = StrOutputParser()
strategist_chain = PROMPT_SUMMARY | llm | output_parser

print("--- Команда из двух агентов (обновленная) готова к работе. ---\n")

# ==============================================================================
# 4. ЗАПУСК РАБОТЫ КОМАНДЫ АГЕНТОВ
# ==============================================================================
task_for_fp = """
Проанализируй ошибки типа False Positive.
Используй свой инструмент GetFalsePositiveSamples, чтобы получить 2 примера.
Для каждого примера детально опиши:
1. Анализ признаков: есть ли аномальные или пограничные значения?
2. Гипотезы: почему модель могла ошибиться?
3. Рекомендации по Feature Engineering.
"""

task_for_fn = """
Теперь проанализируй ошибки типа False Negative.
Используй свой инструмент GetFalseNegativeSamples, чтобы получить 2 примера.
Для каждого примера детально опиши:
1. Анализ признаков.
2. Гипотезы.
3. Рекомендации.
"""

print(f"\n{'#'*25} ЗАДАЧА ДЛЯ АГЕНТА-АНАЛИТИКА: АНАЛИЗ FALSE POSITIVES {'#'*25}")
fp_analysis_report = analyst_agent.run(task_for_fp)
print(f"\n--- Отчет Аналитика по FP готов ---")

print(f"\n{'#'*25} ЗАДАЧА ДЛЯ АГЕНТА-АНАЛИТИКА: АНАЛИЗ FALSE NEGATIVES {'#'*25}")
fn_analysis_report = analyst_agent.run(task_for_fn)
print(f"\n--- Отчет Аналитика по FN готов ---")

# ==============================================================================
# 5. ФИНАЛЬНЫЙ СИНТЕЗ ОТ АГЕНТА-СТРАТЕГА
# ==============================================================================
full_analyst_report = f"ОТЧЕТ ПО FALSE POSITIVES:\n{fp_analysis_report}\n\nОТЧЕТ ПО FALSE NEGATIVES:\n{fn_analysis_report}"
final_recommendations = strategist_chain.invoke({"analyst_report": full_analyst_report})
print(f"\n\n{'#'*30} СТРАТЕГИЧЕСКИЕ ВЫВОДЫ ОТ АГЕНТА-СТРАТЕГА {'#'*30}")
print(final_recommendations)

