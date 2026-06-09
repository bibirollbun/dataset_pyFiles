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
from pandas.api.types import CategoricalDtype
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import shap

train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

train.head(7)


train.info()


msno.dendrogram(train);

"""
The dendrogram uses a hierarchical clustering algorithm (courtesy of scipy) to bin variables against one another 
by their nullity correlation (measured in terms of binary distance). 
At each step of the tree the variables are split up based on which combination 
minimizes the distance of the remaining clusters. 
"""


def engineer_social_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create social interaction features with missing-value-safe transformations.

    Adds:
    - activity_ratio
    - drain_adjusted_activity
    - social_activity_index

    Returns
    -------
    df : pd.DataFrame
        Modified dataframe with new features.
    """
    df = df.copy()

    # Handle missing for numeric columns
    for col in ['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Encode 'Drained_after_socializing': Yes â†’ 1, No â†’ 0, NaN â†’ 0.5
    if 'Drained_after_socializing' in df.columns:
        df['Drained_after_socializing_encoded'] = (
            df['Drained_after_socializing']
            .map({'Yes': 1, 'No': 0})
            .fillna(0.5)
        )

    # Feature 1: Activity Ratio
    df['activity_ratio'] = df['Social_event_attendance'] / (df['Going_outside'] + 1)

    # Feature 2: Drain-adjusted Activity (lower if drained after socializing)
    df['drain_adjusted_activity'] = df['activity_ratio'] * (1 - df['Drained_after_socializing_encoded'])

    # Feature 3: Social Activity Index (posts + friends)
    df['social_activity_index'] = df['Friends_circle_size'] + df['Post_frequency']

    return df

train = engineer_social_features(train)


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
    Compute Theilâ€™s U (Uncertainty Coefficient) for two categorical variables.
    U(X|Y) â€” how much knowing Y reduces uncertainty in X.
    """
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = [n / total_occurrences for n in x_counter.values()]
    s_x = ss.entropy(p_x)
    return 1.0 if s_x == 0 else (s_x - s_xy) / s_x


def categorical_correlation(df: pd.DataFrame, method: str = 'theils_u') -> pd.DataFrame:
    """
    Compute the pairwise correlation matrix for categorical variables using Theilâ€™s U.

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

def plot_upper_triangle_heatmap(df: pd.DataFrame, method: str = 'theils_u', figsize=(10, 8)):
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
    plt.title(f"Theilâ€™s U Correlation (U(X|Y)) - Association Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_upper_triangle_heatmap_with_binned_numerics(df: pd.DataFrame, method: str = 'theils_u', quantiles: int = 20, figsize=(10, 8)):
    df_copy = df.copy()

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


plot_upper_triangle_heatmap_with_binned_numerics(train.drop('id', axis=1))


sns.countplot(x='Personality', data=train); # Imbalanced dataset


sns.histplot(data=train, x='Friends_circle_size', hue='Personality', kde=True);


sns.boxplot(x='Personality', y='Going_outside', data=train)


sns.violinplot(x='Personality', y='Time_spent_Alone', data=train);


sns.countplot(x='Drained_after_socializing', hue='Personality', data=train);


sns.pairplot(train, hue='Personality', vars=['Time_spent_Alone', 'Post_frequency', 'Friends_circle_size']);


def plot_personality_radar_chart(df, features, target_col='Personality'):
    """
    Plot a radar chart comparing mean feature values for each personality type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and personality labels.

    features : list of str
        Feature columns to plot on the radar chart.

    target_col : str
        Column representing the personality class (default = 'Personality').
    """
    mean_vals = df.groupby(target_col)[features].mean()
    normalized = (mean_vals - mean_vals.min()) / (mean_vals.max() - mean_vals.min())

    labels = features
    categories = list(normalized.index)
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    for cat in categories:
        values = normalized.loc[cat].tolist()
        values += values[:1]  # close the loop
        ax.plot(angles, values, label=cat)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)

    ax.set_title(f"Mean Behavioral Profile by {target_col}", size=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()

features_to_plot = [
    'Time_spent_Alone',
    'Going_outside',
    'Social_event_attendance',
    'Friends_circle_size',
    'Post_frequency',
    'activity_ratio',
    'drain_adjusted_activity',
    'social_activity_index'
]

plot_personality_radar_chart(train, features_to_plot, target_col='Personality');


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

def catboost_cv_predict(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    submission: pd.DataFrame,
    cat_features: list = None,
    n_splits: int = 5,
    seed: int = 42,
    threshold: float = 0.5,
    model_params: dict = None,
    output_file: str = "submission.csv",
    target_column: str = "Personality"
):
    if model_params is None:
        model_params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "eval_metric": "Logloss",
            "early_stopping_rounds": 50,
            "random_seed": seed,
            "verbose": 0
        }

    if cat_features is None:
        cat_features = X.select_dtypes(include="object").columns.tolist()

    for col in cat_features:
        X[col] = X[col].astype(str).fillna("Missing")
        X_test[col] = X_test[col].astype(str).fillna("Missing")

    # Ensure X_test has same columns as X
    assert list(X.columns) == list(X_test.columns), "Mismatch: X_test columns do not match X"

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = np.zeros((len(X_test), len(np.unique(y))))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # print(f"Training fold {fold+1}/{n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**model_params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features)

        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_test) / n_splits

    # Convert OOF predicted probabilities to class labels
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    class_names = model.classes_
    true_labels = pd.Categorical(y, categories=class_names).codes
    
    report = classification_report(true_labels, oof_pred_labels, target_names=class_names)
    print("\nClassification Report:\n", report)

    oof_logloss = log_loss(y, oof_preds)
    print(f"\nOut-of-Fold LogLoss: {oof_logloss:.5f}")

    final_preds = np.argmax(test_preds, axis=1)
    class_names = model.classes_
    predicted_labels = class_names[final_preds]

    submission[target_column] = predicted_labels
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to: {output_file}")

    final_model = CatBoostClassifier(**model_params)
    final_model.fit(X, y, cat_features=cat_features)
    return oof_logloss, submission, final_model

X, y = prepare_data(train, target_col="Personality")
test = engineer_social_features(test)
submission = prepare_submission(test, id_col="id", target_col="Personality")

logloss, submission, model = catboost_cv_predict(
    X=X,
    y=y,
    X_test=test.drop(columns=["id"]),
    submission=submission
)


def plot_shap_summary_catboost(model, X: pd.DataFrame, max_display: int = 10):
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

plot_shap_summary_catboost(model, X)

