!pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn scipy


!pip install lightautoml catboost


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from scipy.stats import chi2_contingency
from scipy.optimize import minimize

try:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
    from lightautoml.tasks import Task

    LAMA_AVAILABLE = True
    print("LightAutoML (LAMA) доступна")
except ImportError:
    LAMA_AVAILABLE = False
    print("LightAutoML не установлен")

import pickle
import logging
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_fit_stats = None


logger.info("Loading data")

train_data = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")


print("Размеры датасета:")
print(f"Training: {train_data.shape}")


print("Первые 10 строк:")
print(train_data.head(10))


print("Пропущенные значения:")
print(train_data.isnull().sum())


target = train_data["IsBadBuy"]

print(f"Кол-во строк: {len(target)}")
print(
    f"Good buys (0): {(target == 0).sum()} ({(target == 0).sum() / len(target) * 100:.2f}%)"
)
print(
    f"Bad buys (1): {(target == 1).sum()} ({(target == 1).sum() / len(target) * 100:.2f}%)"
)
print(f"Соотношение классов (0:1): {(target == 0).sum() / (target == 1).sum():.2f} : 1")


target.describe()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Анализ таргета (IsBadBuy)", fontsize=16, fontweight="bold")


class_counts = target.value_counts()
colors = ["#2ecc71", "#e74c3c"]
axes[0, 0].bar(
    ["Good (0)", "Bad (1)"],
    class_counts.values,
    color=colors,
    alpha=0.7,
    edgecolor="black",
)
axes[0, 0].set_title("Распределение классов", fontweight="bold")
axes[0, 0].set_ylabel("Количество")
for i, v in enumerate(class_counts.values):
    axes[0, 0].text(
        i, v + 500, f"{v}\n({v/len(target)*100:.1f}%)", ha="center", fontweight="bold"
    )

axes[0, 1].pie(
    [class_counts[0], class_counts[1]],
    labels=["Good buy", "Bad buy"],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    explode=(0.05, 0.05),
)
axes[0, 1].set_title("Соотношение классов", fontweight="bold")

# trend
train_data["PurchDate_parsed"] = pd.to_datetime(
    train_data["PurchDate"], format="%m/%d/%Y", errors="coerce"
)
monthly_stats = train_data.groupby(
    train_data["PurchDate_parsed"].dt.to_period("M")
).agg({"IsBadBuy": ["sum", "count"]})
monthly_stats.columns = ["bad_count", "total_count"]
monthly_stats["bad_rate"] = (
    monthly_stats["bad_count"] / monthly_stats["total_count"] * 100
)
monthly_stats.index = monthly_stats.index.to_timestamp()
axes[1, 0].plot(
    monthly_stats.index,
    monthly_stats["bad_rate"],
    marker="o",
    linewidth=2,
    color="#e74c3c",
)
axes[1, 0].fill_between(
    monthly_stats.index, monthly_stats["bad_rate"], alpha=0.3, color="#e74c3c"
)
axes[1, 0].set_title("Динамика доли Bad Buy по времени", fontweight="bold")
axes[1, 0].set_ylabel("Доля Bad Buy (%)")
axes[1, 0].tick_params(axis="x", rotation=45)


plt.tight_layout()
plt.show()


numerical_features = train_data.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_features = train_data.select_dtypes(include=["object"]).columns.tolist()


if "IsBadBuy" in numerical_features:
    numerical_features.remove("IsBadBuy")
if "RefId" in numerical_features:
    numerical_features.remove("RefId")

temporal_features = ["PurchDate", "KickDate"]
for feat in temporal_features:
    if feat in categorical_features:
        categorical_features.remove(feat)

print(f"Numerical: {len(numerical_features)} features")
print(f"Categorical: {len(categorical_features)} features")
print(f"Temporal: {len(temporal_features)} features")


len(numerical_features), len(categorical_features)


fig, axes = plt.subplots(6, 3, figsize=(20, 40))
axes = axes.flatten()
for idx, feature in enumerate(numerical_features):
    if feature in numerical_features:
        data = train_data[feature].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = data[(data < lower) | (data > upper)]
        outlier_percent = len(outliers) / len(data) * 100
        axes[idx].boxplot(data)
        axes[idx].set_title(f"{feature}", fontweight="bold")
        info_text = f"Выбросов: {len(outliers)}\n({outlier_percent:.1f}%)"
        axes[idx].text(
            0.98,
            0.97,
            info_text,
            transform=axes[idx].transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=8,
            fontweight="bold",
        )
        axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.show()


missing_data = pd.DataFrame(
    {
        "Feature": train_data.columns,
        "Missing_Count": train_data.isnull().sum(),
        "Missing_Percent": train_data.isnull().sum() / len(train_data) * 100,
    }
).sort_values("Missing_Count", ascending=False)

missing_data = missing_data[missing_data["Missing_Count"] > 0]

if len(missing_data) > 0:
    print(missing_data.to_string(index=False))

    # Визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        missing_data["Feature"],
        missing_data["Missing_Percent"],
        color="#e75c3c",
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Процент пропусков (%)", fontweight="bold")
    ax.set_title("Распределение пропущенных значений", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(missing_data["Missing_Percent"]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold")
    plt.tight_layout()
    plt.show()


correlation_with_target = (
    train_data[numerical_features]
    .corrwith(train_data["IsBadBuy"])
    .sort_values(ascending=False)
)

print("Все корреляции с IsBadBuy:")
print(correlation_with_target.to_string())
print("-" * 80)
print("Топ 10 положительной корреляции:")
print(correlation_with_target[correlation_with_target > 0].head(10).to_string())
print("-" * 80)
print("Топ 10 отрицательной корреляции:")
print(correlation_with_target[correlation_with_target < 0].tail(10).to_string())


# категориальные корреляции (Крамер V)
def calculate_cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


categorical_corr = {}
for feature in categorical_features:
    categorical_corr[feature] = calculate_cramers_v(
        train_data[feature], train_data["IsBadBuy"]
    )

categorical_corr = dict(
    sorted(categorical_corr.items(), key=lambda x: abs(x[1]), reverse=True)
)
print(f"Корреляции категориальных признаков с IsBadBuy (Крамер V):")
for feature, corr in categorical_corr.items():
    print(f"  {feature:30s}: {corr:.4f}")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Комплексный анализ признаков", fontsize=16, fontweight="bold")

# График 1: Корреляции числовых признаков
corr_plot = numerical_corr.drop("IsBadBuy", errors="ignore").sort_values()
colors_num = ["#e74c3c" if x > 0 else "#2ecc71" for x in corr_plot.values]
axes[0, 0].barh(
    range(len(corr_plot)),
    corr_plot.values,
    color=colors_num,
    alpha=0.7,
    edgecolor="black",
)
axes[0, 0].set_yticks(range(len(corr_plot)))
axes[0, 0].set_yticklabels(corr_plot.index, fontsize=9)
axes[0, 0].set_xlabel("Корреляция с IsBadBuy", fontweight="bold")
axes[0, 0].set_title("Корреляции числовых признаков", fontweight="bold", fontsize=12)
axes[0, 0].axvline(x=0, color="black", linewidth=1.5)
axes[0, 0].grid(axis="x", alpha=0.3)

# График 2: Крамер V для категориальных
cat_features = list(categorical_corr.keys())
cat_values = list(categorical_corr.values())
colors_cat = ["#e74c3c" if x > 0.1 else "#f39c12" for x in cat_values]
axes[0, 1].barh(
    range(len(cat_features)), cat_values, color=colors_cat, alpha=0.7, edgecolor="black"
)
axes[0, 1].set_yticks(range(len(cat_features)))
axes[0, 1].set_yticklabels(cat_features, fontsize=9)
axes[0, 1].set_xlabel("Крамер V", fontweight="bold")
axes[0, 1].set_title(
    "Корреляции категориальных признаков", fontweight="bold", fontsize=12
)
axes[0, 1].grid(axis="x", alpha=0.3)

# График 3: Распределение числовых признаков
top_num_features = list(numerical_corr.head(4).index)
for i, feature in enumerate(top_num_features[:4]):
    if i < 2:
        ax = axes[1, 0] if i == 0 else axes[1, 1]
    else:
        ax = axes[1, 0] if i == 2 else axes[1, 1]

axes[1, 0].hist(
    [
        train_data[train_data["IsBadBuy"] == 0][top_num_features[0]].dropna(),
        train_data[train_data["IsBadBuy"] == 1][top_num_features[0]].dropna(),
    ],
    bins=30,
    label=["Good", "Bad"],
    color=["#2ecc71", "#e74c3c"],
    alpha=0.6,
    edgecolor="black",
)
axes[1, 0].set_title(
    f"Распределение {top_num_features[0]} по классам", fontweight="bold", fontsize=12
)
axes[1, 0].set_xlabel("Значение")
axes[1, 0].set_ylabel("Частота")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)


corr_matrix_top_feats = train_data[list(numerical_corr.head(14).index)].corr()
sns.heatmap(
    corr_matrix_top_feats,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Correlation"},
)
axes[1, 1].set_title("Correlation Matrix (Top Features)", fontweight="bold")

plt.tight_layout()
plt.show()


train_data.isna().sum()


def preprocess_data(
    df,
    is_train=True,
    target_col="IsBadBuy",
    fit_stats=None,
    random_state=42,
    test_size=0.2,
    return_split=True,
):
    """
    Универсальная функция предобработки данных

    Args:
        df : pd.DataFrame
            Входной DataFrame

        is_train : bool, default=True
            Если True, обрабатываем как тренировочный набор (может содержать target).
            Если False, обрабатываем как тестовый набор (без target).

        target_col : str, default='IsBadBuy'
            Имя целевой колонки (используется если is_train=True).

        fit_stats : dict, default=None
            Статистика для применения к test-набору:
              - numerical_medians: медианы числовых признаков (вычислены на train)
              - fit_stats уже встроена в функцию на первый вызов

        random_state : int, default=42
            Random seed для воспроизводимости.

        test_size : float, default=0.2
            Доля test при разбиении (используется только если is_train=True и return_split=True).

        return_split : bool, default=True
            Если is_train=True и return_split=True, возвращаем X_train, X_test, y_train, y_test.
            Иначе возвращаем только X (и y если is_train=True).

    Returns:
        Если is_train=True и return_split=True:
            (X_train, X_test, y_train, y_test, fit_stats)

        Если is_train=True и return_split=False:
            (X, y, fit_stats)

        Если is_train=False:
            X (с применённой статистикой из fit_stats)
    """

    global _fit_stats

    df = df.copy()
    if "PurchDate" in df.columns:
        df["PurchDate"] = pd.to_datetime(
            df["PurchDate"], format="%m/%d/%Y", errors="coerce"
        )

    if (
        "MMRAcquisitionRetailAveragePrice" in df.columns
        and "MMRAcquisitionAuctionAveragePrice" in df.columns
    ):
        df["Price_Gap"] = df["MMRAcquisitionRetailAveragePrice"] / (
            df["MMRAcquisitionAuctionAveragePrice"] + 1
        )

    if "VehBCost" in df.columns and "MMRAcquisitionRetailAveragePrice" in df.columns:
        df["VehBCost_to_MMR"] = df["VehBCost"] / (
            df["MMRAcquisitionRetailAveragePrice"] + 1
        )

    if "VehicleAge" in df.columns:
        df["VehicleAge_Squared"] = df["VehicleAge"] ** 2

    if "VehicleAge" in df.columns and "VehBCost" in df.columns:
        df["Age_x_Price"] = df["VehicleAge"] * np.log1p(df["VehBCost"])

    cols_to_drop = {
        "RefId",
        "IsBadBuy",  # ID и таргет
        "PurchDate",  # дата
        "KickDate",  # сырые даты (если есть распаршенные)
        "PurchDate_parsed",  # дата
        "WheelTypeID",  # дублирует WheelType
        "BYRNO",  # много выбрсов
        "PRIMEUNIT", # много пропусков
        "AUCGUART",  # много пропусков
    }

    final_features = [col for col in df.columns if col not in cols_to_drop]
    X = df[final_features].copy()

    print(f"Признаков после дропа: {X.shape}")

    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Численных признаков: {len(numerical_cols)}")
    print(f"Категориальных признаков: {len(categorical_cols)}")

    if is_train and fit_stats is None:
        print(f"Вычисление статистики на train")

        X[numerical_cols] = X[numerical_cols].replace([np.inf, -np.inf], np.nan)
        X[categorical_cols] = X[categorical_cols].fillna("Unknown")

        numerical_medians = X[numerical_cols].median()

        fit_stats = {
            "numerical_medians": numerical_medians,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "feature_cols": final_features,
        }

        _fit_stats = fit_stats

        # Заполнение пропусков медианой
        for col in numerical_cols:
            X[col] = X[col].fillna(numerical_medians[col])

        print(f"Медианы вычислены, пропуски заполнены")

    # Для test
    elif not is_train and fit_stats is not None:
        print(f"Применение статистики из train")

        X[numerical_cols] = X[numerical_cols].replace([np.inf, -np.inf], np.nan)
        X[categorical_cols] = X[categorical_cols].fillna("Unknown")

        for col in numerical_cols:
            if col in fit_stats["numerical_medians"].index:
                median_val = fit_stats["numerical_medians"][col]
                X[col] = X[col].fillna(median_val)
            else:
                X[col] = X[col].fillna(X[col].median())

        print(f"Пропуски в test заполнены по статистике train")

    y = None
    if is_train and target_col in df.columns:
        y = df[target_col].copy().astype(int)
        print(f"Таргет '{target_col}':")
        print(f"{y.value_counts().to_dict()}")
        print(f"Соотношение: {(y==1).sum() / len(y) * 100:.1f}% класс 1")

    if is_train and return_split and y is not None:
        print(f"Разделение на train-test")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        if len(categorical_cols) > 0:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(X_train[categorical_cols])

            # трансформация tran и test
            X_train[categorical_cols] = encoder.transform(X_train[categorical_cols])
            X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

            fit_stats["encoder"] = encoder
            fit_stats["categorical_cols"] = categorical_cols
            _fit_stats = fit_stats

        print(f"Train: {X_train.shape}")
        print(f"Test:  {X_test.shape}")
        print(f"Train target: {y_train.value_counts().to_dict()}")
        print(f"Test target:  {y_test.value_counts().to_dict()}")

        print(f"Предобработка завершена!")
        return X_train, X_test, y_train, y_test, fit_stats

    elif is_train and not return_split and y is not None:
        print(f"Все данные")

        if len(categorical_cols) > 0:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(X[categorical_cols])
            # трансформация tran и test
            X[categorical_cols] = encoder.transform(X[categorical_cols])

            fit_stats["encoder"] = encoder
            fit_stats["categorical_cols"] = categorical_cols
            _fit_stats = fit_stats

        print(f"Train: {X.shape}")
        print(f"Train target: {y.value_counts().to_dict()}")

        print(f"Предобработка завершена!")
        return X, y, fit_stats

    elif not is_train:
        if len(categorical_cols) > 0:
            encoder = fit_stats["encoder"]
            X[categorical_cols] = encoder.transform(X[categorical_cols])
        print(f"OrdinalEncoder применён к test")
        print(f"Final X shape: {X.shape}")
        print(f"Предобработка test завершена!")
        return X

    else:
        print(f"X shape: {X.shape}")
        print(f"Предобработка завершена!")
        return X


X_train, X_test, y_train, y_test, fit_stats = preprocess_data(
    train_data,
    is_train=True,
    target_col="IsBadBuy",
    random_state=42,
    test_size=0.2,
    return_split=True,
)


print(f"\nТипы данных в X_train: {X_train.dtypes.unique()}")


train_lama = X_train.copy()
train_lama["IsBadBuy"] = y_train.values


# Создаем задачу классификации
task = Task("binary", metric="auc")


# Инициализируем LAMA с конфигурацией Speed
automl_speed = TabularAutoML(
    task=task,
    timeout=600,  # 10 минут
    cpu_limit=-1,  # Использовать все ядра
    reader_params={"cv": 3},  # 3-fold cross-validation
)

automl_speed.fit_predict(train_lama, roles={"target": "IsBadBuy"})


y_pred_speed_proba = automl_speed.predict(X_test).data[:, 0]
y_pred_speed = (y_pred_speed_proba > 0.5).astype(int)


auc_speed = roc_auc_score(y_test, y_pred_speed_proba)

print(f"РЕЗУЛЬТАТЫ LAMA Speed:")
print(f"ROC-AUC Score: {auc_speed:.6f}")
classification_report(y_test, y_pred_speed)

# Сохраняем результаты
lama_speed_metrics = {
    "model": automl_speed,
    "auc": auc_speed,
    "y_pred_proba": y_pred_speed_proba,
    "y_pred": y_pred_speed,
}


# Инициализируем LAMA с конфигурацией
automl_quality = TabularUtilizedAutoML(
    task=task,
    timeout=1800,
    cpu_limit=4,
    # Настройки чтения и ролей
    reader_params={
        "n_jobs": 4,  # использовать все доступные потоки
        "random_state": 42,
        "cv": 3,
        "advanced_roles": True,
    },
    # Общие параметры моделей
    general_params={"use_algos": [["lgb", "lgb_tuned", "linear_l2"]]},
    # Optuna
    tuning_params={
        "fit_on_holdout": True,  # дообучать лучшую модель на holdout-фолде
        "max_tuning_iter": 30,  # максимум 30 итераций Optuna
        "max_tuning_time": 1200,  # не больше 1200 секунд на тюнинг
    },
    # Параметры отбора признаков
    selection_params={
        # Для отбора используем gbm (lgb/catboost) и linear_l2
        "select_algos": ["lgb", "linear_l2"],
        # Оставляем ~80% наиболее информативных признаков
        "selection_feats_rate": 0.8,
        # Можно ограничить абсолютное число фич, здесь оставляем авто
        "max_features_cnt_in_result": None,
    },
)


oof_pred_quality = automl_quality.fit_predict(train_lama, roles={"target": "IsBadBuy"})


y_pred_quality_proba = automl_quality.predict(X_test).data[:, 0]
y_pred_quality = (y_pred_quality_proba > 0.5).astype(int)
auc_quality = roc_auc_score(y_test, y_pred_quality_proba)

print(f"РЕЗУЛЬТАТЫ LAMA Quality (TabularUtilizedAutoML):")
print(f"ROC-AUC Score: {auc_quality:.6f}")
classification_report(y_test, y_pred_quality)

# Сохраняем метрики и модель
lama_quality_metrics = {
    "model": automl_quality,
    "auc": auc_quality,
    "y_pred_proba": y_pred_quality_proba,
    "config": "TabularUtilizedAutoML_Quality",
}


custom_models = {}
predictions = {}
class_counts = y_train.value_counts()
pos_ratio = class_counts[1] / class_counts[0]

print(f"Модель 1. CatBoost (40% вес)", end=" ", flush=True)
try:
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=9,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        verbose=False,
        scale_pos_weight=pos_ratio,
        subsample=0.8,
        colsample_bylevel=0.8,
        min_data_in_leaf=10,
    )
    cat_model.fit(X_train, y_train)
    pred_cat = cat_model.predict_proba(X_test)[:, 1]
    auc_cat = roc_auc_score(y_test, pred_cat)
    custom_models["catboost"] = cat_model
    predictions["catboost"] = pred_cat
    print(f"AUC = {auc_cat:.6f}")
except Exception as e:
    print(f"Ошибка: {e}")

print(f"Модель 2. XGBoost (30% вес)", end=" ", flush=True)
try:
    xgb_model = xgb.XGBClassifier(
        tree_method="auto",
        device="cpu",
        max_depth=12,
        min_child_weight=2,
        gamma=0.53,
        subsample=0.9,
        colsample_bytree=1.0,
        learning_rate=0.016,
        n_estimators=5000,
        scale_pos_weight=0.12,
        reg_alpha=0.01,
        reg_lambda=1,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, pred_xgb)
    custom_models["xgboost"] = xgb_model
    predictions["xgboost"] = pred_xgb
    print(f"AUC = {auc_xgb:.6f}")
except Exception as e:
    print(f"Ошибка: {e}")

print(f"Модель 3. LightGBM (15% вес)", end=" ", flush=True)
try:
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=10,
        num_leaves=200,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=15,
        reg_lambda=0.5,
        reg_alpha=0.1,
        class_weight={0: 1.0, 1: pos_ratio},
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        extra_trees=True,
    )
    lgb_model.fit(X_train, y_train)
    pred_lgb = lgb_model.predict_proba(X_test)[:, 1]
    auc_lgb = roc_auc_score(y_test, pred_lgb)
    custom_models["lightgbm"] = lgb_model
    predictions["lightgbm"] = pred_lgb
    print(f"AUC = {auc_lgb:.6f}")
except Exception as e:
    print(f"Ошибка: {e}")

print(f"Модель 4. RandomForest (10% вес)", end=" ", flush=True)
try:
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=False,
    )
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, pred_rf)
    custom_models["randomforest"] = rf_model
    predictions["randomforest"] = pred_rf
    print(f"AUC = {auc_rf:.6f}")
except Exception as e:
    print(f"Ошибка: {e}")

print(f"Модель 5. GradientBoosting (5% вес)", end=" ", flush=True)
try:
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )
    gb_model.fit(X_train, y_train)
    pred_gb = gb_model.predict_proba(X_test)[:, 1]
    auc_gb = roc_auc_score(y_test, pred_gb)
    custom_models["gradientboosting"] = gb_model
    predictions["gradientboosting"] = pred_gb
    print(f"AUC = {auc_gb:.6f}")
except Exception as e:
    print(f"Ошибка: {e}")


for model_name, pred in predictions.items():
    auc = roc_auc_score(y_test, pred)
    print(f"{model_name:20s}: AUC = {auc:.6f}")


individual_scores = {}
for model_name, pred in predictions.items():
    auc = roc_auc_score(y_test, pred)
    individual_scores[model_name] = auc
    status = "ok" if auc > 0.75 else "warning"
    print(f"{status} {model_name:20s}: AUC = {auc:.6f}")
weights_raw = {k: max(v - 0.74, 0.01) for k, v in individual_scores.items()}
w_sum = sum(weights_raw.values())
weights = {k: v / w_sum for k, v in weights_raw.items()}

# Нормируем веса
w_sum = sum(weights.values())
weights = {k: v / w_sum for k, v in weights.items()}

ensemble_pred = np.zeros(len(X_test), dtype=np.float64)
for model_name, weight in weights.items():
    if model_name in predictions:
        ensemble_pred += weight * predictions[model_name]

auc_ensemble = roc_auc_score(y_test, ensemble_pred)
print(f"AUC = {auc_ensemble:.6f}")


def objective(weights):
    """
    Оптимизирует веса ансамбля для максимизации AUC.

    Используется с scipy.optimize.minimize() для нахождения оптимальных весов
    взвешенного ансамбля из базовых моделей.

    Args:
        weights: np.ndarray, веса моделей

    Returns:
        float: негативный AUC (для минимизации)
    """
    # нормируем веса
    w_normalized = weights / weights.sum()

    # взвешенное предсказание
    pred = np.zeros(len(X_test))
    for i, (name, pred_val) in enumerate(predictions_dict.items()):
        pred += w_normalized[i] * pred_val

    auc = roc_auc_score(y_test, pred)
    return -auc


predictions_dict = {
    "catboost": custom_models["catboost"].predict_proba(X_test)[:, 1],
    "xgboost": custom_models["xgboost"].predict_proba(X_test)[:, 1],
    "lightgbm": custom_models["lightgbm"].predict_proba(X_test)[:, 1],
    "randomforest": custom_models["randomforest"].predict_proba(X_test)[:, 1],
    "gradientboosting": custom_models["gradientboosting"].predict_proba(X_test)[:, 1],
}
initial_weights = np.array([0.35, 0.30, 0.15, 0.10, 0.10])
# все веса >= 0
constraints = {"type": "ineq", "fun": lambda x: x}
# оптимизируем
result = minimize(
    objective,
    initial_weights,
    method="SLSQP",
    constraints=constraints,
    options={"maxiter": 100},
)

# нормируем оптимальные веса
optimal_weights = result.x / result.x.sum()

print(f"Оптимизированные веса:")
for name, weight in zip(predictions_dict.keys(), optimal_weights):
    print(f"  {name:20s}: {weight:.4f} ({weight*100:.2f}%)")

pred_custom = np.zeros(len(X_test))
for (name, pred_val), weight in zip(predictions_dict.items(), optimal_weights):
    pred_custom += weight * pred_val

auc_custom = roc_auc_score(y_test, pred_custom)
print(f"AUC = {auc_custom:.6f}")


optimal_weights_dict = dict(zip(predictions_dict.keys(), optimal_weights))


custom_ensemble_config = {
    "models": custom_models,
    "weights": optimal_weights_dict,
    "feature_cols": X_train.columns.tolist(),
    "val_auc": auc_custom,
}


print(f"ExtraTreesClassifier...")
et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
et_model.fit(X_train, y_train)
pred_et = et_model.predict_proba(X_test)[:, 1]
auc_et = roc_auc_score(y_test, pred_et)
print(f"AUC = {auc_et:.6f}")


custom_models["extratrees"] = et_model
predictions_dict["extratrees"] = pred_et


new_weights = {
    "catboost": 0.30,
    "xgboost": 0.25,
    "extratrees": 0.20,
    "lightgbm": 0.12,
    "randomforest": 0.08,
    "gradientboosting": 0.05,
}

w_sum = sum(new_weights.values())
new_weights = {k: v / w_sum for k, v in new_weights.items()}

ensemble_pred = np.zeros(len(X_test))
for name, weight in new_weights.items():
    ensemble_pred += weight * predictions_dict[name]

auc_new = roc_auc_score(y_test, ensemble_pred)
print(f"Новый ансамбль (6 моделей) AUC: {auc_new:.6f}")


custom_ensemble_config_6 = {
    "models": custom_models,
    "weights": optimal_weights_dict,
    "feature_cols": X_train.columns.tolist(),
    "val_auc": auc_custom,
}


meta_features_val = np.column_stack(
    [
        custom_models["catboost"].predict_proba(X_test)[:, 1],
        custom_models["xgboost"].predict_proba(X_test)[:, 1],
        custom_models["lightgbm"].predict_proba(X_test)[:, 1],
        custom_models["randomforest"].predict_proba(X_test)[:, 1],
        custom_models["gradientboosting"].predict_proba(X_test)[:, 1],
    ]
)


# Meta-learner 1
meta_lgb = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
)
meta_lgb.fit(meta_features_val, y_test)
pred_meta_lgb = meta_lgb.predict_proba(meta_features_val)[:, 1]
auc_meta_lgb = roc_auc_score(y_test, pred_meta_lgb)
print(f" Meta-LightGBM AUC: {auc_meta_lgb:.6f}")


# Meta-learner 2
meta_lr = LogisticRegression(max_iter=1000, random_state=42)
meta_lr.fit(meta_features_val, y_test)
pred_meta_lr = meta_lr.predict_proba(meta_features_val)[:, 1]
auc_meta_lr = roc_auc_score(y_test, pred_meta_lr)
print(f"Meta-LogisticReg AUC: {auc_meta_lr:.6f}")


# Финальный ensemble = average meta-predictions
stacking_pred = 0.5 * pred_meta_lgb + 0.5 * pred_meta_lr
auc_stacking = roc_auc_score(y_test, stacking_pred)

print(f"Двухуровневый Stacking AUC: {auc_stacking:.6f}")


def predict_ensemble(models_dict, weights_dict, X):
    """Взвешенное предсказание ансамбля.
    
    Parameters:
        models_dict: словарь обученных классификаторов
        weights_dict: словарь весов
        X: данные для предсказания, shape (n_samples, n_features)
    Returns:
        np.ndarray: вероятности класса 1, shape (n_samples,)
    """
    pred = np.zeros(X.shape[0], dtype=np.float64)

    for name, model in models_dict.items():
        if name in weights_dict:
            weight = weights_dict[name]
            p = model.predict_proba(X)[:, 1]
            p = np.asarray(p).ravel()
            pred = pred + weight * p

    return pred


X_train_full, y_train_full, fit_stats = preprocess_data(
    train_data,
    is_train=True,
    target_col="IsBadBuy",
    random_state=42,
    test_size=0.2,
    return_split=False,
)


test_raw = pd.read_csv("/kaggle/input/DontGetKicked/test.csv")

X_test_submit = preprocess_data(test_raw, is_train=False, fit_stats=fit_stats)


train_lama_full = X_train_full.copy()
train_lama_full["IsBadBuy"] = y_train_full.values


oof_test_quality = automl_quality.fit_predict(
    train_lama_full, roles={"target": "IsBadBuy"}
)


test_pred_proba_lama = automl_quality.predict(X_test_submit).data[:, 0]


submit = pd.DataFrame(
    {
        "RefId": test_raw["RefId"],
        "IsBadBuy": test_pred_proba_lama,
    }
)

submit.to_csv("submit_lama.csv", index=False)


print(f"Обучаем финальный ансамбль на всех обучающих данных: {X_train_full.shape}")

final_models = {}
for model_name, model in custom_ensemble_config["models"].items():
    print(f"  • Переобучаем {model_name}...", end=" ", flush=True)
    m_class = model.__class__
    m_params = model.get_params()
    new_model = m_class(**m_params)
    new_model.fit(X_train_full, y_train_full)
    final_models[model_name] = new_model
    print(f"✓")

print(f"Все модели переобучены!")


test_pred_proba = predict_ensemble(
    final_models, custom_ensemble_config["weights"], X_test_submit
)


submit = pd.DataFrame(
    {
        "RefId": test_raw["RefId"],
        "IsBadBuy": test_pred_proba,
    }
)

submit.to_csv("submit_custom_4.1.csv", index=False)


print(f"Обучаем финальный ансамбль на всех обучающих данных: {X_train_full.shape}")

final_models = {}
for model_name, model in custom_ensemble_config_6["models"].items():
    print(f"  • Переобучаем {model_name}...", end=" ", flush=True)
    m_class = model.__class__
    m_params = model.get_params()
    new_model = m_class(**m_params)
    new_model.fit(X_train_full, y_train_full)
    final_models[model_name] = new_model
    print(f"✓")

print(f"Все модели переобучены!")


test_pred_proba = predict_ensemble(
    final_models, custom_ensemble_config_6["weights"], X_test_submit
)


submit = pd.DataFrame(
    {
        "RefId": test_raw["RefId"],
        "IsBadBuy": test_pred_proba,
    }
)

submit.to_csv("submit_custom_4.2.csv", index=False)


print(f"Обучаем финальный ансамбль на всех обучающих данных: {X_train_full.shape}")

final_models = {}
for model_name, model in custom_ensemble_config_6["models"].items():
    print(f"  • Переобучаем {model_name}...", end=" ", flush=True)
    m_class = model.__class__
    m_params = model.get_params()
    new_model = m_class(**m_params)
    new_model.fit(X_train_full, y_train_full)
    final_models[model_name] = new_model
    print(f"✓")

print(f"Все модели переобучены!")


meta_features_full = np.column_stack(
    [
        final_models[name].predict_proba(X_train_full)[:, 1]
        for name in [
            "catboost",
            "xgboost",
            "lightgbm",
            "randomforest",
            "gradientboosting",
        ]
    ]
)

final_meta_learner = LGBMClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
)
final_meta_learner.fit(meta_features_full, y_train_full)


meta_lr = LogisticRegression(max_iter=1000, random_state=42)
meta_lr.fit(meta_features_full, y_train_full)


meta_features_test = np.column_stack(
    [
        final_models[name].predict_proba(X_test_submit)[:, 1]
        for name in [
            "catboost",
            "xgboost",
            "lightgbm",
            "randomforest",
            "gradientboosting",
        ]
    ]
)

test_pred_meta_lgb = final_meta_learner.predict_proba(meta_features_test)[:, 1]
test_pred_meta_lr = meta_lr.predict_proba(meta_features_test)[:, 1]


stacking_test_pred = 0.5 * test_pred_meta_lgb + 0.5 * test_pred_meta_lr


submit = pd.DataFrame(
    {
        "RefId": test_raw["RefId"],
        "IsBadBuy": stacking_test_pred,
    }
)

submit.to_csv("submit_custom_4_3_fixed.csv", index=False)

