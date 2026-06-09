class CFG:
    n_splits = 5 # Bayesian CV splits
    n_folds = 5 # Stratified CV Folds
    n_iter = 15 # Bayesian runs
    n_jobs = -1
    seed = 42 # Random seed
    frac = 1 # Fraction of data to be used
    task_type="GPU"
    bayesian_tune_cat = True


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, log_loss, classification_report
from sklearn.metrics import classification_report
import shap
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from contextlib import contextmanager
import sys, os
import json
from pandas.api.types import CategoricalDtype
from catboost import CatBoostClassifier, Pool
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize


train = pd.read_csv('../input/playground-series-s5e8/train.csv', low_memory=False)
test = pd.read_csv('../input/playground-series-s5e8/test.csv', low_memory=False)
submission = pd.read_csv('../input/playground-series-s5e8/sample_submission.csv')
train = train.sample(frac=CFG.frac)


train.head(7)


train.info()


def plot_numeric_distributions_by_target(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list = None,
    n_cols: int = 3,
    figsize_per_plot: tuple = (5, 4),
    plot_type: str = "violin" # "hist"  # or "violin"
) -> None:
    """
    Plots the distribution of numeric features in the DataFrame, broken down by the target column.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame.
    - target_col: str
        The name of the target column to split by.
    - exclude_cols: list
        List of columns to exclude from plotting (e.g., ID).
    - n_cols: int
        Number of columns in the plot grid.
    - figsize_per_plot: tuple
        Size of each subplot (width, height).
    - plot_type: str
        Type of plot: "hist" or "violin"
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.difference([target_col] + exclude_cols).tolist()
    n_features = len(numeric_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(n_rows, n_cols, i)
        try:
            if plot_type == "hist":
                sns.histplot(data=df, x=feature, hue=target_col, kde=False, multiple="stack", palette="Set1")
            elif plot_type == "violin":
                sns.violinplot(data=df, x=target_col, y=feature, palette="Set3", inner="quartile")
            else:
                raise ValueError("Unsupported plot_type. Use 'hist' or 'violin'.")
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature if plot_type == "hist" else target_col)
            plt.ylabel('Count' if plot_type == "hist" else feature)
        except Exception as e:
            plt.text(0.5, 0.5, f"Error plotting {feature}\n{str(e)}", ha='center')

    plt.tight_layout()
    plt.show()

plot_numeric_distributions_by_target(train, target_col='y', exclude_cols=['id', 'day'])


def print_unique_categories(df: pd.DataFrame) -> None:
    """
    Prints the unique values for each categorical (object or category dtype) column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    print(f"Found {len(categorical_cols)} categorical columns.\n")
    
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"{col} ({len(unique_vals)} unique): {unique_vals}\n")

print_unique_categories(train)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["has_loan_or_housing"] = ((df["housing"] == "yes") | (df["loan"] == "yes")).astype(int)

    df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)
    df["previous_successful_contact"] = (df["poutcome"] == "success").astype(int)
    df["days_since_last_contact"] = df["pdays"].apply(lambda x: 999 if x == -1 else x)

    df["week_of_month"] = (df["day"] - 1) // 7 + 1
    df["is_end_of_month_contact"] = (df["day"] > 25).astype(int)

    df["is_balance_positive"] = (df["balance"] > 0).astype(int)
    df["log_balance"] = np.log1p(df["balance"])

    # quantile binning
    df["duration_quantile"] = pd.qcut(df["duration"], q=10, labels=False, duplicates="drop")
    df["pdays_quantile"] = pd.qcut(df["pdays"], q=10, labels=False, duplicates="drop")

    df["contact_efficiency_ratio"] = df["duration"] / (df["campaign"] + 1)

    df["contacted_multiple_times"] = (df["campaign"] > 1).astype(int)

    df["is_blue_collar"] = (df["job"] == "blue-collar").astype(int)
    df["is_at_risk_client"] = (
        ((df["loan"] == "yes") | (df["housing"] == "yes")) &
        (df["balance"] < 100) &
        (df["education"].isin(["primary", "unknown"]))
    ).astype(int)

    binary_flag_cols = ["is_at_risk_client", "was_previously_contacted", "has_loan_or_housing", 
                       "is_end_of_month_contact", "is_balance_positive", "contacted_multiple_times",
                       "previous_successful_contact", "is_blue_collar", "is_at_risk_client"]

    for col in binary_flag_cols:
        df[col] = df[col].astype("category")

    df.drop(["duration", "pdays"], axis=1, inplace=True)

    if 'y' in df.columns.tolist():
        df['y'] = df['y'].astype("category")

    return df

train = add_engineered_features(train)
test = add_engineered_features(test)


def downcast_dtype(df):
    """
    Downcasts numeric columns and converts object columns to category
    to reduce memory usage.

    Parameters:
    df: pd.DataFrame

    Returns:
    df: downcasted DataFrame (inplace)
    """
    previous_memory = df.memory_usage(deep=True).sum() / 1024**2  # in MB

    for col in df.columns:
        col_dtype = df[col].dtypes

        if pd.api.types.is_numeric_dtype(df[col]):
            xmin = df[col].min()
            xmax = df[col].max()

            if pd.api.types.is_integer_dtype(df[col]):
                if np.iinfo(np.int8).min <= xmin <= np.iinfo(np.int8).max and xmax <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif np.iinfo(np.int16).min <= xmin <= np.iinfo(np.int16).max and xmax <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif np.iinfo(np.int32).min <= xmin <= np.iinfo(np.int32).max and xmax <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            elif pd.api.types.is_float_dtype(df[col]):
                if np.finfo(np.float16).min <= xmin <= np.finfo(np.float16).max and xmax <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif np.finfo(np.float32).min <= xmin <= np.finfo(np.float32).max and xmax <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif pd.api.types.is_object_dtype(df[col]):
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Heuristic: low cardinality
                df[col] = df[col].astype('category')

    after_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (previous_memory - after_memory) / previous_memory

    print(f"Memory usage before downcasting: {previous_memory:.2f} MB")
    print(f"Memory usage after downcasting: {after_memory:.2f} MB")
    print(f"Reduced by: {reduction:.2f}%")

    return df

train = downcast_dtype(train)
test = downcast_dtype(test)


def conditional_entropy(x, y):
    """
    Calculate the conditional entropy H(X|Y).
    """
    y_counter = Counter(y)
    xy_counter = Counter(zip(x, y))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0

    for (x_i, y_i), xy_count in xy_counter.items():
        p_xy = xy_count / total_occurrences
        p_y = y_counter[y_i] / total_occurrences
        entropy += p_xy * np.log(p_y / p_xy)
    return entropy


def theils_u(x, y):
    """
    Compute Theil’s U (Uncertainty Coefficient) for two categorical variables.
    U(X|Y) — how much knowing Y reduces uncertainty in X.
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = [n / total_occurrences for n in x_counter.values()]
    s_x = ss.entropy(p_x)
    return 1.0 if s_x == 0 else (s_x - s_xy) / s_x

def categorical_correlation(df: pd.DataFrame, method: str = 'theils_u') -> pd.DataFrame:
    """
    Compute the pairwise correlation matrix for categorical variables using Theil’s U.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with categorical or object columns.

    method : str
        {'theils_u'}

    Returns
    -------
    pd.DataFrame
        Square correlation matrix with values in [0, 1].
    """
    if df.empty:
        return pd.DataFrame()

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) == 0:
        return pd.DataFrame()

    corr_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

    for col1 in categorical_cols:
        for col2 in categorical_cols:
            x = df[col1].dropna().astype(str)
            y = df[col2].dropna().astype(str)

            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
            elif method == 'theils_u':
                corr_matrix.loc[col1, col2] = theils_u(x.tolist(), y.tolist())
            else:
                raise NotImplementedError(f"Method '{method}' is not supported.")

    return corr_matrix.astype(float)

def plot_upper_triangle_heatmap(df: pd.DataFrame, method: str = 'theils_u', figsize=(14, 14)):
    corr_matrix = categorical_correlation(df, method)

    # Create mask for upper triangle
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='YlGnBu',
        fmt='.2f',
        square=True,
        cbar_kws={'shrink': .75}
    )
    plt.title(f"Theil’s U Correlation (U(X|Y)) - Association Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_upper_triangle_heatmap_with_binned_numerics(df: pd.DataFrame, method: str = 'theils_u', quantiles: int = 20, figsize=(14, 14)):
    df_copy = df.copy()

    if 'y' in df_copy.columns.tolist():
        df_copy['y'] = df_copy['y'].astype('category')

    # Quantile-bin all numeric columns
    num_cols = df_copy.select_dtypes(include=['number']).columns
    for col in num_cols:
        try:
            df_copy[col] = pd.qcut(df_copy[col], q=quantiles, duplicates='drop')
        except ValueError:
            # Skip columns that can't be binned (e.g., constant)
            df_copy.drop(columns=col, inplace=True)

    # Cast to categorical
    for col in df_copy.columns:
        if pd.api.types.is_object_dtype(df_copy[col]) or isinstance(df_copy[col].dtype, CategoricalDtype):
            df_copy[col] = df_copy[col].astype('category')

    plot_upper_triangle_heatmap(df_copy, method=method, figsize=figsize)

categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
if 'y' not in categorical_cols:
    categorical_cols.append('y')

plot_upper_triangle_heatmap_with_binned_numerics(train.drop('id', axis=1)[categorical_cols].sample(frac=0.1))


def plot_target_distribution(df: pd.DataFrame, target_col: str, title: str = "Target Distribution") -> None:
    """
    Plots the distribution of a binary target variable and annotates the percentage of each class.

    Parameters:
    - df: pd.DataFrame: The input DataFrame
    - target_col: str: The name of the target column
    - title: str: Title for the plot
    """
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=target_col, data=df, palette="Set2")

    total = len(df)
    for p in ax.patches:
        count = p.get_height()
        percentage = 100 * count / total
        ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2, count),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.title(title)
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

plot_target_distribution(train, target_col='y') # Imbalanced


def plot_radar_chart_by_target(df: pd.DataFrame, target_col: str) -> None:
    """
    Plots a radar chart comparing the normalized mean of numeric features for each class in the target column.
    """
    # Select numeric columns and exclude the target
    numeric_cols = df.select_dtypes(include=["number"]).columns.difference([target_col])

    # Compute class-wise means
    mean_per_class = df.groupby(target_col)[numeric_cols].mean()

    # Handle infs and NaNs
    mean_per_class.replace([np.inf, -np.inf], np.nan, inplace=True)
    mean_per_class.dropna(axis=1, inplace=True)

    # Normalize
    scaler = MinMaxScaler()
    normalized_means = pd.DataFrame(
        scaler.fit_transform(mean_per_class),
        index=mean_per_class.index,
        columns=mean_per_class.columns
    )

    # Radar setup
    labels = normalized_means.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # loop closure

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    for cls in normalized_means.index:
        values = normalized_means.loc[cls].tolist()
        values += values[:1]  # match loop closure
        ax.plot(angles, values, label=f"{target_col} = {cls}")
        ax.fill(angles, values, alpha=0.1)

    ax.set_title("Radar Chart of Normalized Numeric Features by Target")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()

plot_radar_chart_by_target(train.drop('id', axis=1), target_col='y')


def prepare_data(df: pd.DataFrame, target_col: str, id_col: str = "id"):
    """
    Splits the dataframe into features and target, excluding the ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features, ID, and target.

    target_col : str
        Name of the target column.

    id_col : str
        Name of the ID column to exclude from features.

    Returns
    -------
    X : pd.DataFrame
        Features dataframe excluding ID and target.

    y : pd.Series
        Target series.
    """
    X = df.drop(columns=[target_col, id_col], errors='ignore')
    y = df[target_col]
    return X, y

def prepare_submission(test_df: pd.DataFrame, id_col: str, target_col: str):
    submission = pd.DataFrame({id_col: test_df[id_col]})
    submission[target_col] = ""
    return submission


def convert_to_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df

cat_cols = train.select_dtypes(include=['object']).columns.tolist()
train = convert_to_categorical(train, cat_cols)
test = convert_to_categorical(test, cat_cols)

X, y = prepare_data(train, target_col="y")
print("\nOriginal X_train shape:", X.shape)
X_test = test.drop(columns=['id'], errors='ignore')


def get_categorical_feature_indices(df: pd.DataFrame, include_target: bool = False, target_col: str = 'y') -> list:
    """
    Returns the indices of categorical features in the DataFrame.

    Parameters:
    - df: pd.DataFrame: The input DataFrame
    - include_target: bool: Whether to include the target column if it's categorical
    - target_col: str: Name of the target column

    Returns:
    - List of indices of categorical columns
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if include_target and target_col not in cat_cols and df[target_col].dtype == 'object':
        cat_cols.append(target_col)
    elif not include_target and target_col in cat_cols:
        cat_cols.remove(target_col)

    return [df.columns.get_loc(col) for col in cat_cols]



from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score, precision_recall_curve, recall_score
ap_scorer = make_scorer(average_precision_score, needs_proba=True, greater_is_better=True)

def best_f1_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return np.max(f1_scores)

f1_from_proba = make_scorer(best_f1_threshold, needs_proba=True, greater_is_better=True)
recall_minority = make_scorer(lambda y_true, y_pred: recall_score(y_true, y_pred, pos_label=1), greater_is_better=True)

def tune_catboost_hyperparams(X, y, cat_features, task_type="CPU", seed=42, n_iter=25, n_splits=5, n_jobs=-1):
    """
    Performs Bayesian hyperparameter tuning for CatBoost using skopt.

    Parameters:
    - X: pd.DataFrame: Training features
    - y: pd.Series: Target variable
    - cat_features: list of categorical feature indices or names
    - task_type: str: "CPU" or "GPU"
    - seed: int: Random seed for reproducibility
    - n_iter: int: Number of iterations for Bayesian search
    - n_splits: int: Number of Stratified K-Folds
    - n_jobs: int: Parallel jobs for BayesSearchCV

    Returns:
    - model_params: dict of best parameters ready to use in CatBoostClassifier
    """
    print("Starting Bayesian hyperparameter tuning for CatBoost...")

    search_space_cb = {
        "iterations": Integer(300, 1200),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "depth": Integer(4, 8),
        "l2_leaf_reg": Real(1.0, 32.0),
        "bagging_temperature": Real(0, 1.0),
        "random_strength": Real(0, 1.0),
        "border_count": Integer(32, 255),
        "boosting_type": Categorical(["Plain", "Ordered"]),
        "scale_pos_weight": Real(0.5, 10.0)
    }

    base_cb = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type=task_type,
        cat_features=cat_features,
        verbose=0,
        od_type='Iter',
        od_wait=100,
    )

    bayes_search_cb = BayesSearchCV(
        estimator=base_cb,
        search_spaces=search_space_cb,
        scoring= ap_scorer, # f1_from_proba, #ap_scorer, # "roc_auc",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        n_iter=n_iter,
        n_jobs=1,
        n_points=1,
        verbose=1
    )

    bayes_search_cb.fit(X, y)

    print("Best Parameters for CatBoost:")
    print(json.dumps(bayes_search_cb.best_params_, indent=4))

    best_params = bayes_search_cb.best_params_
    best_params.update({
                        "random_seed": seed,
                        "eval_metric": "Logloss",
                        "verbose": 0,
                        "task_type": task_type,
                        "use_best_model": True,
                        "od_type": 'Iter',
                        "od_wait": 50,
                        })

    return best_params


cat_features = get_categorical_feature_indices(X)

if CFG.bayesian_tune_cat:
    model_params = tune_catboost_hyperparams(
        X=X,
        y=y,
        cat_features=cat_features,
        task_type=CFG.task_type,
        seed=CFG.seed,
        n_iter=CFG.n_iter,
        n_splits=CFG.n_splits,
        n_jobs=CFG.n_jobs
    )
else:
    print("Using best CatBoost model params learnt earlier.")
    model_params = {
        "bagging_temperature": 0.9,
        "boosting_type": "Ordered",
        "border_count": 256,
        "depth": 8,
        "iterations": 800,
        "l2_leaf_reg": 6.0,
        "learning_rate": 0.05,
        "random_strength": 0.5,
        "scale_pos_weight": 2,
        "verbose": 0,
        "od_type": 'Iter',
        "od_wait": 50
    }


oof_preds_cat = np.zeros((len(X), len(np.unique(y))))
test_preds_cat = np.zeros((len(X_test), len(np.unique(y))))

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training CatBoost fold {fold+1}/{CFG.n_folds}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    base_model = CatBoostClassifier(**model_params)
    base_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)

    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)

    oof_preds_cat[val_idx] = calibrated_model.predict_proba(X_val)
    test_preds_cat += calibrated_model.predict_proba(X_test) / CFG.n_splits


true_labels = pd.Categorical(y, categories=base_model.classes_).codes
probs_class1 = oof_preds_cat[:, 1]  # Assuming this is probability for class 1

# Compute FPR, TPR and thresholds
fpr, tpr, thresholds = roc_curve(true_labels, probs_class1)
roc_auc_scores = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_scores:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


class_names = [str(cls) for cls in base_model.classes_]

# Map true labels accordingly
true_labels = pd.Categorical(y, categories=base_model.classes_).codes
oof_pred_labels = np.argmax(oof_preds_cat, axis=1)
final_preds = np.argmax(test_preds_cat, axis=1)
final_probs = test_preds_cat[:, 1]

report = classification_report(true_labels, oof_pred_labels, target_names=class_names)
print("\nClassification Report:\n", report)
oof_logloss = log_loss(true_labels, oof_preds_cat)
print(f"\nOut-of-Fold LogLoss: {oof_logloss:.5f}")

submission = prepare_submission(test, 'id', 'y')
submission['y'] = final_probs
submission.to_csv("submission.csv", index=False)
print(f"Submission saved")
print(submission.head(7))


def plot_shap_summary_catboost(model, X: pd.DataFrame, max_display: int = 15):
    """
    Generates SHAP summary plots for a CatBoost model using TreeExplainer.

    Parameters
    ----------
    model : CatBoostClassifier
        A trained CatBoost model (CPU mode).

    X : pd.DataFrame
        Training features used with the model.

    max_display : int
        Maximum number of features to display in summary plots.
    """
    # SHAP expects raw data (no NaNs for numeric, strings for categorical)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("Generating SHAP bar summary plot...")
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)
    
    print("Generating SHAP beeswarm plot...")
    shap.summary_plot(shap_values, X, plot_type="dot", max_display=max_display)

plot_shap_summary_catboost(base_model, X.sample(frac=0.2))


from sklearn import metrics

def permutation_importances(model, X, y, metric):
    baseline = metric(model, X, y)
    imp = []
    for col in X.columns:
        save = X[col].copy()
        X[col] = np.random.permutation(X[col])
        m = metric(model, X, y)
        X[col] = save
        imp.append(m-baseline)
    return np.array(imp)

def log_loss(m, X, y): 
    return metrics.log_loss(y, m.predict_proba(X)[:,1])
    
def get_feature_imp_plot(method):
    
    if method == "Permutation":
        fi =  permutation_importances(base_model, X, y, log_loss)
        
    else:
        fi = model.get_feature_importance(Pool(X_test, label=y_test, cat_features=categorical_features_indices), 
                                                                     type=method)
        
    feature_score = pd.DataFrame(list(zip(X_test.dtypes.index, fi )),
                                    columns=['Feature','Score'])

    feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

    plt.rcParams["figure.figsize"] = (12,7)
    ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
    ax.set_title("Feature Importance using {}".format(method), fontsize = 14)
    ax.set_xlabel("features")
    plt.show()

%time get_feature_imp_plot(method="Permutation")

