import os
import sys
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print(f"Python Version: {sys.version}")


from pathlib import Path
from typing import Optional, Union

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


class CFG:
    # Data Configuration
    DATA_DIR = Path('../input/playground-series-s5e12')
    TRAIN_PATH = DATA_DIR / Path('train.csv')
    TEST_PATH = DATA_DIR / Path('test.csv')
    SUB_SAMPLE_PATH = DATA_DIR / Path('sample_submission.csv')
    
    # Hyperparameter Configuration
    SEED = 15
    VALID_SIZE = 0.1


train = pd.read_csv(CFG.TRAIN_PATH, index_col='id')
test = pd.read_csv(CFG.TEST_PATH, index_col='id')
sample = pd.read_csv(CFG.SUB_SAMPLE_PATH)

print("*" * 80)
print(f"Train shape: {train.shape}")
print(train.info(), "\n")
print("*" * 80)
print(f"Test shape: {test.shape}")
print(test.info(), "\n")
print("*" * 80)
print(f"Sample shape: {sample.shape}")
print(sample.info(), "\n")


print("Features and their unique value counts:")
print(train.nunique())


CFG.TARGET = "diagnosed_diabetes"
CFG.NOMINAL_COLS = ["gender", "ethnicity", "employment_status", "family_history_diabetes",
                    "hypertension_history", "cardiovascular_history"]
CFG.ORDINAL_COLS = ["education_level", "income_level", "smoking_status"]
CFG.CONTINUOUS_COLS = ["bmi", "waist_to_hip_ratio", "diet_score", "sleep_hours_per_day",
                       "screen_time_hours_per_day"]
CFG.DISCRETE_HIGH_COLS = ["age", "physical_activity_minutes_per_week", "systolic_bp", "diastolic_bp",
                          "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol",
                          "triglycerides"]
CFG.DISCRETE_LOW_COLS = ["alcohol_consumption_per_week"]

ALL_NUM_COLS = CFG.CONTINUOUS_COLS + CFG.DISCRETE_HIGH_COLS + CFG.DISCRETE_LOW_COLS
ALL_CAT_COLS = CFG.NOMINAL_COLS + CFG.ORDINAL_COLS

sns.set_style('whitegrid')


def plot_binary_target_counts(
    df: pd.DataFrame,
    target: str,
    title: str = "Target Class Distribution",
    xlabel: str = "Class",
    ylabel: str = "Count",
    figsize: tuple = (6, 6)
) -> None:
    """
    Plots a target class distribution countplot.
    
    Displays a seaborn count plot with optional x and y labels, and a title.
    Also displays the percentage of the sum of values for each binary class
    on the top of the bars in the countplot.
    
    Args:
        df: The Pandas DataFrame containing the features and target.
        target: The name of the target feature column.
        title: An optional title for the displayed plot.
        xlabel: An optional label for the x-axis (target values).
        ylabel: An optional label for the y-axis. Defaults to "Count".capitalize
        figsize: Confgures the size of the displayed plot. Defaults to `(6, 6)`.
    Returns:
        None
    """
    # Initialize and configure
    total = len(df)
    plt.figure(figsize=figsize)
    ax = sns.countplot(data=df, x=target)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Loop through each bar in the blot
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        ax.annotate(text=percentage,
                    xy=(p.get_x() + p.get_width() / 2., height),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="black")
    plt.show()

plot_binary_target_counts(train, CFG.TARGET)


def plot_histograms_with_kde(
    df: pd.DataFrame,
    numeric_cols: list[str],
    n_cols: int = 3,
    bins: int = 30,
    color: str = "skyblue"
) -> None:
    """
    Generates a grid of histograms with Kernel Density Estimates (KDE) for specified columns.

    This function dynamically calculates the grid layout based on the number of 
    provided columns and generates a Seaborn histplot for each.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_cols (list[str]): A list of column names within `df` to be plotted. 
            These should correspond to numeric data types.
        n_cols (int, optional): The number of subplots to arrange horizontally 
            in one row. Defaults to 3.
        bins (int, optional): The number of bins to divide the data into for the 
            histogram. Defaults to 30.
        color (str, optional): The color of the histogram bars and KDE line. 
            Defaults to "skyblue".

    Returns:
        None: The function displays a matplotlib figure via `plt.show()` and 
        does not return a value.
    """
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols 
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(data=df, x=col, kde=True, bins=bins, color=color)
        plt.title(f"Distribution: {col}")
        plt.xlabel("")
    plt.tight_layout()
    plt.show()
    
plot_histograms_with_kde(train, ALL_NUM_COLS)


def plot_categorical_countplots(
    df: pd.DataFrame,
    categorical_cols: list[str],
    ordinal_cols: list[str],
    n_cols: int = 3,
) -> None:
    """
    Generates a grid of Seaborn countplots for specified categorical columns.

    This function distinguishes between ordinal and nominal variables when ordering 
    the x-axis:
    - **Ordinal columns** (listed in `ordinal_cols`) are sorted alphanumerically.
    - **Nominal columns** (all others) are sorted by frequency (descending).

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        categorical_cols (list[str]): A list of column names to be plotted.
        ordinal_cols (list[str]): A subset of column names that should be treated 
            as ordinal. These will be plotted in sorted order (e.g., A, B, C) 
            rather than by count.
        n_cols (int, optional): The number of subplots to arrange horizontally 
            in one row. Defaults to 3.

    Returns:
        None: The function displays a matplotlib figure via `plt.show()` and 
        does not return a value.
    """
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(categorical_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Logic to determine ordering
        if col in ordinal_cols:
            order = sorted(df[col].dropna().unique())
        else:
            order = df[col].value_counts().index 
            
        sns.countplot(data=df, x=col, order=order)
        plt.title(f"Counts: {col}")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("")
        
    plt.tight_layout()
    plt.show()

plot_categorical_countplots(train, ALL_CAT_COLS, CFG.ORDINAL_COLS)


def plot_boxplots(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target: str,
    n_cols: int = 3,
    outliers: bool = False,
    xlabel: str = ""
) -> None:
    """
    Generates a grid of boxplots comparing numeric columns against a target variable.
    
    This function is useful for analyzing the relationship between continuous features 
    and a categorical target, visualizing distribution spread and central tendency.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_cols (list[str]): A list of numeric column names (y-axis) to analyze.
        target (str): The name of the categorical column (x-axis) used for grouping.
        n_cols (int, optional): The number of subplots to arrange horizontally. 
            Defaults to 3.
        outliers (bool, optional): If True, displays outlier points (fliers) beyond 
            the whiskers. Defaults to False.
        xlabel (str, optional): A custom label for the x-axis applied to all subplots. 
            Defaults to an empty string.

    Returns:
        None: The function displays a matplotlib figure via `plt.show()` and 
        does not return a value.
    """
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols 
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(data=df, x=target, y=col, showfliers=outliers)
        plt.title(f"{col} vs Target")
        plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()
    
plot_boxplots(train, ALL_NUM_COLS, CFG.TARGET)


def plot_barplots(
    df: pd.DataFrame,
    categorical_cols: list[str],
    ordinal_cols: list[str],
    target: str,
    n_cols: int = 3
) -> None:
    """
    Generates a grid of bar plots showing the mean target value for categorical columns.

    This function visualizes the relationship between categorical features and a 
    numeric target (usually a binary probability). It handles ordering differently 
    based on the column type:
    - **Ordinal columns:** Sorted alphanumerically.
    - **Nominal columns:** Sorted by the mean target value (ascending).

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        categorical_cols (list[str]): A list of categorical column names (x-axis).
        ordinal_cols (list[str]): A subset of columns to treat as ordinal. These 
            are sorted by label rather than by target mean.
        target (str): The numeric target column (y-axis). The bar height represents 
            the mean of this column (e.g., survival probability).
        n_cols (int, optional): The number of subplots to arrange horizontally. 
            Defaults to 3.

    Returns:
        None: The function displays a matplotlib figure via `plt.show()` and 
        does not return a value.
    """
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(categorical_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Logic to determine ordering
        if col in ordinal_cols:
            order = sorted(df[col].dropna().unique())
        else:
            order = df.groupby(col)[target].mean().sort_values().index
            
        sns.barplot(data=df, x=col, y=target, order=order, errorbar=None)
        plt.title(f"Target Probability by {col}")
        plt.ylim(0, 1)
        plt.ylabel("Probability")
        plt.xticks(rotation=45, ha="right")
        
    plt.tight_layout()
    plt.show()
    
plot_barplots(train, ALL_CAT_COLS, CFG.ORDINAL_COLS, CFG.TARGET)


def create_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target: str,
    figsize: tuple[int, int] = (14, 12),
    cmap: str = "RdBu_r",
    title: str = "Feature Correlation Matrix"
) -> None:
    """
    Calculates and displays a correlation matrix heatmap for numeric features.

    This function computes the pairwise correlation (Pearson) between specified 
    numeric columns and the target variable. It applies a mask to hide the upper 
    triangle of the heatmap to reduce visual clutter.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_cols (list[str]): A list of numeric feature names to include.
        target (str): The name of the target variable to include in the matrix.
        figsize (tuple[int, int], optional): The width and height of the figure 
            in inches. Defaults to (14, 12).
        cmap (str, optional): The mapping from data values to color space. 
            Defaults to "RdBu_r" (Red-Blue, diverging).
        title (str, optional): The title of the plot. Defaults to "Feature 
            Correlation Matrix".

    Returns:
        None: The function displays a matplotlib figure via `plt.show()` and 
        does not return a value.
    """
    # Ensure target isn't duplicated if it's already in numeric_cols
    cols_to_plot = numeric_cols + [target] if target not in numeric_cols else numeric_cols
    
    numeric_df = df[cols_to_plot]
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) 
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        center=0, 
        square=True, 
        linewidths=.5
    )
    plt.title(title)
    plt.show()
    
create_correlation_matrix(train, ALL_NUM_COLS, CFG.TARGET)


def plot_pairplot(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
    sample_size: int = 50000,
    seed: int = 15,
    palette: str = "bright"
) -> None:
    """
    Generates a pairwise scatter plot matrix for selected features.

    This function visualizes the joint distribution of every pair of specified 
    features, colored by the target variable. 
    
    Since pairplots are computationally expensive, this function automatically 
    downsamples the data if the input DataFrame exceeds the `sample_size`.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        cols (list[str]): A list of feature names to include in the grid.
        target (str): The name of the categorical column used for color encoding 
            (hue).
        sample_size (int, optional): The maximum number of rows to include in the 
            plot. If `df` is smaller, all rows are used. Defaults to 50,000.
        seed (int, optional): Random seed for reproducibility during sampling. 
            Defaults to 15.
        palette (str, optional): The color palette to use for the target classes. 
            Defaults to "bright".

    Returns:
        None: The function displays a matplotlib figure via `plt.show()` and 
        does not return a value.
    """
    # Avoid duplicate columns if target is already in cols
    plot_cols = cols + [target] if target not in cols else cols
    
    # Handle sampling safely
    n = min(sample_size, len(df))
    sample = df[plot_cols].sample(n=n, random_state=seed)
    
    sns.pairplot(
        sample, 
        hue=target, 
        palette=palette, 
        corner=True, 
        plot_kws={"alpha": 0.6, "s": 15}
    )
    plt.suptitle(f"Interactions: {', '.join(cols)}", y=1.02)
    plt.show()

plot_pairplot(train, ['age', 'bmi', 'systolic_bp', 'waist_to_hip_ratio'], CFG.TARGET)


plot_pairplot(train, ['triglycerides', 'hdl_cholesterol', 'waist_to_hip_ratio', 'bmi'], CFG.TARGET)


plot_pairplot(train, ['physical_activity_minutes_per_week', 'screen_time_hours_per_day', 'bmi'], CFG.TARGET)


def create_violin_plots(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    ordinal_cols: list[str],
    figsize: tuple[int, int] = (20, 16),
    title: str = ""
) -> None:
    """
    Generates a matrix of violin plots crossing numeric vs. categorical features.

    This function creates a grid layout where:
    - **Rows** represent categorical variables.
    - **Columns** represent numeric variables.
    
    It automatically handles the ordering of categorical groups:
    - **Ordinal columns:** Sorted alphanumerically.
    - **Nominal columns:** Sorted by frequency (descending), truncated to the 
      top 10 most frequent categories to maintain readability.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numeric_cols (list[str]): List of numeric column names (y-axis).
        categorical_cols (list[str]): List of categorical column names (x-axis).
        ordinal_cols (list[str]): Subset of columns to treat as ordinal.
        figsize (tuple[int, int], optional): Dimensions of the figure. 
            Defaults to (20, 16).
        title (str, optional): Overall figure title. Defaults to an empty string.

    Returns:
        None: The function displays a matplotlib figure via `plt.show()` and 
        does not return a value.
    """
    fig, axes = plt.subplots(
        len(categorical_cols), 
        len(numeric_cols), 
        figsize=figsize, 
        squeeze=False
    )
    
    for i, cat_col in enumerate(categorical_cols):
        # Determine ordering logic
        if cat_col in ordinal_cols:
            order = sorted(df[cat_col].dropna().unique())
        else:
            # Limit to top 10 for nominal columns to avoid overcrowding
            order = df[cat_col].value_counts().index[:10]
        
        for j, num_col in enumerate(numeric_cols):
            ax = axes[i, j]
            sns.violinplot(
                data=df, 
                x=cat_col, 
                y=num_col, 
                ax=ax, 
                order=order, 
                linewidth=1
            )
            
            ax.set_title(f'{num_col} by {cat_col}')
            ax.set_xlabel('')
            ax.set_ylabel(num_col if j == 0 else '') 
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()
    
title = "How Context Changes Biology (Cat vs Num)"
target_numericals = ['bmi', 'age', 'systolic_bp', 'physical_activity_minutes_per_week']
target_categoricals = ['education_level', 'income_level', 'employment_status', 'smoking_status']

create_violin_plots(train, target_numericals, target_categoricals, CFG.ORDINAL_COLS, title=title)


def plot_mi_scores(
    df: pd.DataFrame,
    target: str,
    categorical_cols: list[str],
    sample_size: int = 100_000,
    seed: int = 15,
    title: str = None,
    figsize: tuple[int, int] = (12, 10)
) -> pd.Series:
    """Computes and visualizes Mutual Information (MI) scores for feature selection.

    This function calculates the dependency between features and a categorical target 
    using `mutual_info_classif`. It automatically handles categorical feature encoding 
    (via OrdinalEncoder) and performs the calculation on a random sample of data 
    to manage computational cost.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target (str): The name of the target column (classification label).
        categorical_cols (list[str]): List of categorical feature names. These 
            will be encoded ordinally before MI calculation.
        sample_size (int, optional): The maximum number of rows to use for 
            calculation. Defaults to 100,000.
        seed (int, optional): Random state for sampling and MI calculation. 
            Defaults to 15.
        title (str, optional): The title of the plot. If None, defaults to 
            "Mutual Information Scores".
        figsize (tuple[int, int], optional): Figure dimensions. Defaults to (12, 10).

    Returns:
        pd.Series: A Series containing the MI scores, indexed by feature name 
        and sorted in descending order.
    """
    df_sample = df.sample(min(sample_size, len(df)), random_state=seed).copy()
    
    X = df_sample.drop(columns=[target])
    y = df_sample[target]
    
    if categorical_cols:
        oe = OrdinalEncoder()
        X[categorical_cols] = oe.fit_transform(X[categorical_cols].astype(str))
    
    discrete_mask = [col in categorical_cols for col in X.columns]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=seed)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=figsize)
    sns.barplot(x=mi_series.values, y=mi_series.index)
    plt.title(title if title else "Mutual Information Scores")
    plt.xlabel("Mutual Information")
    plt.show()
    
    return mi_series

plot_mi_scores(train, CFG.TARGET, ALL_CAT_COLS)


def plot_roc_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_proba: Union[np.ndarray, pd.Series],
    label_name: str = "Model",
    ax: Optional[Axes] = None,
    figsize: tuple[int, int] = (8, 6)
) -> float:
    """
    Calculates and plots the Receiver Operating Characteristic (ROC) curve.

    This function computes the False Positive Rate (FPR) and True Positive Rate (TPR)
    to visualize classifier performance. It can either create a new figure or plot
    onto an existing Matplotlib axes object to compare multiple models.

    Args:
        y_true (Union[np.ndarray, pd.Series]): Ground truth binary labels (0 or 1).
        y_pred_proba (Union[np.ndarray, pd.Series]): Predicted probabilities or
            decision function scores for the positive class.
        label_name (str, optional): The legend label for this specific curve.
            Defaults to "Model".
        ax (Optional[Axes], optional): Existing matplotlib axes to plot on. If None,
            a new figure and axes are created. Defaults to None.
        figsize (tuple[int, int], optional): Size of the figure if a new one is
            created. Defaults to (8, 6).

    Returns:
        float: The Area Under the Curve (AUC) score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.plot(fpr, tpr, lw=2, label=f"{label_name} (AUC = {roc_auc:.4f})")

    if not any(line.get_label() == "Random Guess" for line in ax.lines):
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    return roc_auc

# Single Feature Baseline
X = train.drop(columns=[CFG.TARGET])
y = train[CFG.TARGET]
plt.figure(figsize=(10, 8))
ax = plt.gca()
activity_col = 'physical_activity_minutes_per_week'
y_pred_single = -X[activity_col] 
plot_roc_curve(y, y_pred_single, label_name='Single Feature (Activity)', ax=ax)
plt.show()


X = train.drop(columns=[CFG.TARGET])
y = train[CFG.TARGET]

# We must used a stratified split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=CFG.VALID_SIZE, random_state=CFG.SEED, stratify=y)

print(f"X train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"y_valid shape: {y_valid.shape}")


log_features = ["physical_activity_minutes_per_week"]
log_transformer = Pipeline(steps=[
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("scaler", StandardScaler())
])


ordinal_features = ["education_level", "income_level", "smoking_status"]
education_order = ["No formal", "Highschool", "Graduate", "Postgraduate"]
income_order = ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
smoking_order = ["Never", "Former", "Current"]
ordinal_transformer = Pipeline(steps=[
    ("encoder", OrdinalEncoder(categories=[education_order, income_order, smoking_order], handle_unknown="use_encoded_value", unknown_value=-1)),
    ("scaler", StandardScaler())
])


nominal_features = ["gender", "ethnicity", "employment_status"]
categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])


numeric_features = [col for col in X_train.columns if col not in log_features + ordinal_features + nominal_features]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[
        ("log", log_transformer, log_features),
        ("ord", ordinal_transformer, ordinal_features),
        ("cat", categorical_transformer, nominal_features),
        ("num", numeric_transformer, numeric_features)],
    verbose_feature_names_out=False).set_output(transform="pandas")


def train_and_evaluate(
    classifier: BaseEstimator,
    preprocessor: ColumnTransformer,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_valid: Union[pd.DataFrame, np.ndarray],
    y_valid: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
    ax: Optional[Axes] = None
) -> tuple[float, Pipeline]:
    """
    Constructs a pipeline, trains it, evaluates ROC AUC, and plots the curve.
    
    Args:
        classifier: An initialized sklearn-compatible classifier (e.g., LogisticRegression()).
        preprocessor: The ColumnTransformer preprocessor instance.
        X_train, y_train: Training data.
        X_valid, y_valid: Validation data.
        model_name: Name of the model for the plot legend.
        ax: Optional matplotlib axes to plot on.
        
    Returns:
        float: The ROC AUC score.
        pipeline: The trained pipeline object.
    """
    # Build pipeline
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    # Train
    print(f"Training {model_name}...")
    model_pipeline.fit(X_train, y_train)
    
    # Predict probs
    y_pred_proba = model_pipeline.predict_proba(X_valid)[:, 1]
    
    # Eval and plot
    auc_score = plot_roc_curve(y_valid, y_pred_proba, label_name=model_name, ax=ax)
    
    return auc_score, model_pipeline



baseline_models = [
    ("Logistic Regression", LogisticRegression(solver='liblinear', random_state=CFG.SEED)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=10, random_state=CFG.SEED)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=CFG.SEED)),
    ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss', n_jobs=-1, random_state=CFG.SEED)),
    ("LightGBM", LGBMClassifier(n_estimators=100, max_depth=10, verbosity=-1, n_jobs=-1, random_state=CFG.SEED)),
    ("CatBoost", CatBoostClassifier(n_estimators=100, depth=6, verbose=0, random_state=CFG.SEED))
]


plt.figure(figsize=(12, 10))
ax = plt.gca()

results = {}

for name, model in baseline_models:
    score, trained_pipe = train_and_evaluate(
        classifier=model,
        preprocessor=preprocessor,
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        model_name=name,
        ax=ax
    )
    results[name] = score

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.title('Baseline Model Comparison (ROC AUC)')
plt.show()

print("\n----- Baseline Models -----")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_results:
    print(f"{name}: {score:.5f}")


def blend_preds(
    classifiers: list[tuple[str, BaseEstimator]],
    preprocessor: ColumnTransformer,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    submission_df: pd.DataFrame,
    weights: Optional[list[float]] = None,
    output_path: str = "submission.csv"
) -> pd.DataFrame:
    """
    Trains multiple models on the full dataset, blends their predictions, and saves to CSV.

    Args:
        classifiers: List of (name, model_instance) tuples.
        preprocessor: The ColumnTransformer instance.
        X_train, y_train: The FULL training features and target.
        X_test: The test dataframe (features only).
        submission_df: The sample_submission dataframe (to get IDs).
        weights: Optional list of weights for blending (e.g. [0.4, 0.3, 0.3]). 
                 Defaults to equal weighting.
        output_path: Filename for the output CSV.

    Returns:
        pd.DataFrame: The final submission dataframe.
    """
    blended_probs = np.zeros(len(X_test))
    
    if weights is None:
        weights = [1.0 / len(classifiers)] * len(classifiers)
    
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    for i, (name, clf) in enumerate(classifiers):
        
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        pipe.fit(X_train, y_train)
        
        probs = pipe.predict_proba(X_test)[:, 1]
        
        weight = weights[i]
        blended_probs += (probs * weight)

    final_sub = submission_df.copy()
    final_sub[CFG.TARGET] = blended_probs
    
    final_sub.to_csv(output_path, index=False)
    
    return final_sub


X_full = train.drop(columns=[CFG.TARGET])
y_full = train[CFG.TARGET]

# Using the same hyperparameters
top_models = [
    ("CatBoost", CatBoostClassifier(n_estimators=100, depth=6, verbose=0, random_state=42)),
    ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, eval_metric='logloss', n_jobs=-1, random_state=CFG.SEED)),
    ("LightGBM", LGBMClassifier(n_estimators=100, max_depth=10, verbosity=-1, n_jobs=-1, random_state=CFG.SEED))
]

blend_weights = [0.4, 0.3, 0.3] 


submission = blend_preds(
    classifiers=top_models,
    preprocessor=preprocessor,
    X_train=X_full,
    y_train=y_full,
    X_test=test,
    submission_df=sample,
    weights=blend_weights,
    output_path="submission.csv"
)

print(submission.head())

