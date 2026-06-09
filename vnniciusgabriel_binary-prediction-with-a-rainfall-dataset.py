import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

__import__('warnings').filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


df_train.info()


df_train.head()


df_train.isna().sum()


df_train.describe()


df_test.head()


df_test.isna().sum()


df_test.describe()


def continuous_univariate_analysis(
    df: pd.DataFrame,
    column: str,
    figsize = (12, 4),
    bins = 30,
    show_outliers = True,
    quantile_range = (0.25, 0.75)
):
    if column not in df.columns:
        raise ValueError(f'{column} not in the dataframe')

    stats_columns = [
        'mean', 'median', 'std', 'min', 'max', 'IQR', 'skewness', 'kurtosis',
        'missing_count', 'missing_percentage', 'num_outliers', 'outlier_percentage'
    ]
    stats_df = pd.DataFrame(index=[column], columns=stats_columns)
    
    data = df[column].dropna()

    stats_df.loc[column, 'mean'] = df[column].mean()
    stats_df.loc[column, 'median'] = df[column].median()
    stats_df.loc[column, 'std'] = df[column].std()
    stats_df.loc[column, 'min'] = df[column].min()
    stats_df.loc[column, 'max'] = df[column].max()
    
    q1, q3 = np.percentile(data, [quantile_range[0]*100, quantile_range[1]*100])
    iqr = q3 - q1
    stats_df.loc[column, 'IQR'] = iqr
    stats_df.loc[column, 'skewness'] = skew(data)
    stats_df.loc[column, 'kurtosis'] = kurtosis(data)
    
    missing_count = df[column].isna().sum()
    missing_percentage = (missing_count / len(df)) * 100
    stats_df.loc[column, 'missing_count'] = missing_count
    stats_df.loc[column, 'missing_percentage'] = missing_percentage

    lower_bound = q1 - 1.5 * iqr
    power_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > power_bound)]

    stats_df.loc[column, 'num_outliers'] = len(outliers)
    stats_df.loc[column, 'outlier_percentage'] = (len(outliers) / len(data)) * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    sns.histplot(data, bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f'Distribution of column: {column}')
    axes[0].axvline(data.mean(), color='red', linestyle='--', label='Mean')
    axes[0].axvline(data.median(), color='green', linestyle='--', label='Median')
    axes[0].legend()
    
    # Boxplot
    sns.boxplot(x=data, ax=axes[1], showfliers=show_outliers)
    axes[1].set_title(f'Boxplot of column: {column}')
    
    stats_text = (
        f"Mean: {data.mean():.2f}\n"
        f"Median: {data.median():.2f}\n"
        f"Std: {data.std():.2f}\n"
        f"IQR: {iqr:.2f}\n"
        f"Skew: {skew(data):.2f}\n"
        f"Outliers: {len(outliers)} ({(len(outliers)/len(data)*100):.1f}%)"
    )
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes, 
                fontsize=9, va='top', ha='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


continuous_univariate_analysis(df_train, 'day')


continuous_univariate_analysis(df_train, 'pressure')


continuous_univariate_analysis(df_train, 'maxtemp')


continuous_univariate_analysis(df_train, 'temparature')


continuous_univariate_analysis(df_train, 'mintemp')


continuous_univariate_analysis(df_train, 'dewpoint')


continuous_univariate_analysis(df_train, 'humidity')


continuous_univariate_analysis(df_train, 'cloud')


continuous_univariate_analysis(df_train, 'sunshine')


continuous_univariate_analysis(df_train, 'winddirection')


continuous_univariate_analysis(df_train, 'windspeed')


df_train['rainfall'].hist()


def impute_outliers_using_iqr(df: pd.DataFrame, columns: list[str],*, factor: float = 1.5, visualize: bool = False) -> None:

    df_original = df.copy()
    df_imputed = df.copy()
    
    imputation_counts = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
    
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers_lower = (df[col] < lower_bound).sum()
        outliers_upper = (df[col] > upper_bound).sum()
        imputation_counts[col] = {'lower': outliers_lower, 'upper': outliers_upper}
        
        df_imputed.loc[df_imputed[col] < lower_bound, col] = lower_bound
        df_imputed.loc[df_imputed[col] > upper_bound, col] = upper_bound
    
    if visualize and len(columns) > 0:

        n_cols, n_rows = 2, len(columns)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        
        for i, col in enumerate(columns):
        
            sns.boxplot(x=df_original[col], ax=axes[i, 0], orient='h')
            axes[i, 0].set_title(f'{col} - before imputation')
            axes[i, 0].set_xlabel('Value')
            
            sns.boxplot(x=df_imputed[col], ax=axes[i, 1], orient='h')
            axes[i, 1].set_title(f'{col} - after imputation')
            axes[i, 1].set_xlabel('Value')
            
            before_stats = (
                f"Mean: {df_original[col].mean():.2f}\n"
                f"Std: {df_original[col].std():.2f}\n"
                f"Min: {df_original[col].min():.2f}\n"
                f"Max: {df_original[col].max():.2f}"
            )
            
            after_stats = (
                f"Mean: {df_imputed[col].mean():.2f}\n"
                f"Std: {df_imputed[col].std():.2f}\n"
                f"Min: {df_imputed[col].min():.2f}\n"
                f"Max: {df_imputed[col].max():.2f}"
            )
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            imputation_info = (
                f"lower bound: {lower_bound:.2f}\n"
                f"upper bound: {upper_bound:.2f}\n"
                f"imputed: {imputation_counts[col]['lower']} lower, "
                f"{imputation_counts[col]['upper']} upper"
            )
            
            axes[i, 0].text(0.02, 0.95, before_stats, transform=axes[i, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            axes[i, 1].text(0.02, 0.95, after_stats, transform=axes[i, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            axes[i, 1].text(0.02, 0.05, imputation_info, transform=axes[i, 1].transAxes, 
                           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
        total_imputed = sum(d['lower'] + d['upper'] for d in imputation_counts.values())
        total_values = len(df) * len(columns)
        imputed_pct = 100 * total_imputed / total_values if total_values > 0 else 0
        
        plt.suptitle(
            f'outlier imputation (IQR method, factor={factor})\n'
            f'imputed {total_imputed} values ({imputed_pct:.1f}% of data points)', 
            fontsize=14
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95 if n_rows > 1 else 0.85)
        plt.show()
    
    return df_imputed


columns_with_outliers = ['pressure', 'dewpoint', 'mintemp', 'humidity', 'cloud', 'windspeed']
df_train_without_outliers = impute_outliers_using_iqr(df_train, columns_with_outliers, visualize=True)


def continuos_bivariate_analysis(df: pd.DataFrame, height: float = 2.5):
    sns.set_style('ticks')

    scatter_grid = sns.pairplot(
        df,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 'edgecolor': 'k'},
        height=height,
        corner=True
    )

    def annotate_corr(x, y, **kwargs):
        ax = kwargs.get('ax', plt.gca())
        r, _ = pearsonr(x, y)
        ax.annotate(f'r = {r:.2f}', xy=(0.1, 0.9), xycoords=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

    scatter_grid.map_lower(annotate_corr)

    plt.suptitle('Scatter Plot Matrix and Distributions', y=1.02, fontsize=24)

    plt.show()


continuos_bivariate_analysis(df_train_without_outliers.drop('id', axis=1))


df_train.isna().sum()


df_test.isna().sum()


df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].mean())
df_test.isna().sum()


df_train.duplicated(subset=['id']).sum()


df_test.duplicated(subset=['id']).sum()


corr_matrix = df_test.drop('id', axis=1).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1, vmax=1
)

plt.title("Correlation Heatmap", fontsize=16)
plt.show()


def perform_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """perform feature enginner to the df"""

    df = df.copy()

    # 1. encode day as cyclic to capture seasonality
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # 2. create interaction
    df['sunshine_minus_cloud'] = df['sunshine'] - df['cloud']
    df['sunshine_over_cloud'] = df['sunshine'] / (df['cloud'] + 1e-3)

    # 3. lag features (using 1-day lag)
    for col in ['cloud', 'sunshine', 'humidity']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(method='bfill')

    # 4. rolling statistics (3-day window)
    WINDOW = 3
    for col in ['cloud', 'sunshine', 'humidity']:
        df[f'{col}_roll_mean'] = df[col].rolling(window=WINDOW, min_periods=1).mean().fillna(method='bfill')

    # 5. feature combination
    df['cloud_humidity_interaction'] = df['cloud'] * df['humidity']
    df['sunshine_cloud_ratio'] = (df['sunshine'] / (df['cloud'] + 1e-5))

    # 6. meteorological feature
    df['temp_amplitude'] = df['maxtemp'] - df['mintemp']
    df['pressure_change'] = df['pressure'].diff().fillna(0)
                                                         
    # 7. additional time-based interactions
    df['temp_avg'] = (df['maxtemp'] + df['mintemp']) / 2
    df['cloud_seasonal'] = df['cloud'] * df['day_sin']
    df['sunshine_seasonal'] = df['sunshine'] * df['day_cos']
    df['humidity_seasonal'] = df['humidity'] * df['day_sin']
    
    COLUMNS_TO_REMOVE = ['day']
    df = df.drop(columns=COLUMNS_TO_REMOVE, axis=1)

    return df


df_train_processed = perform_feature_engineer(df_train.drop(['id'], axis=1))
df_train_processed.head()


corr_matriz = df_train_processed.corr()
corr_matriz['rainfall'].sort_values(ascending=False)


X, y = df_train_processed.drop(['rainfall'], axis=1), df_train_processed['rainfall']


min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)

print(f"shape: train: {len(X_train)}, test: {len(X_test)}")


from time import perf_counter
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {perf_counter() - start:.2f} seconds")
        return result
    return wrapper


@timeit
def train_model_and_evaluate(model):    
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    print(f"roc_auc_score: {roc_auc_score(y_test, y_pred):.4f}")


train_model_and_evaluate(XGBClassifier(random_state=42, scale_pos_weight=len(y[y==0]) / len(y[y==1])))


train_model_and_evaluate(LGBMClassifier(random_state=42, is_unbalance=True))


scale_pos_weight = [len(y_train[y_train==0]) / len(y_train[y_train==1])]

models = {
    'LGBMClassifier': LGBMClassifier(),
    'XGBClassifier': XGBClassifier(),
    'RandomForestClassifier': RandomForestClassifier()
}

params = {
    'LGBMClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 50],
        'max_depth': [-1, 5],
        'is_unbalance': [True],
        'boosting_type': ['gbdt']
    },
    'XGBClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'scale_pos_weight': scale_pos_weight,
        'subsample': [0.9],
        'colsample_bytree': [0.9],
        'objective': ['binary:logistic']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [None, 5],
        'class_weight': ['balanced']
    }
}

best_models = {}

@timeit
def run_grid_search():
    for model_name, model in models.items():
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best params for {model_name}: {grid_search.best_params_}")

        y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_proba)

        print(f"ROC AUC score for {model_name}: {score:.4f}")
        
run_grid_search()


voting_clf = VotingClassifier(
    estimators=[(model_name, model) for model_name, model in models.items()],
    voting='soft'
)

voting_clf.fit(X_train, y_train)

best_models['VotingClassifier'] = voting_clf


plt.figure(figsize=(10, 8))

for model_name, model in best_models.items():

    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f"{model_name} AUC: {auc_score:.2f}")

plt.plot([0, 1], [0, 1], '--k', lw=2)
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


model = best_models['XGBClassifier']

df_test_processed = perform_feature_engineer(df_test.drop('id', axis=1))
df_test_scaled = min_max_scaler.transform(df_test_processed)

y_pred = model.predict_proba(df_test_scaled)[:, 1]

submission = pd.DataFrame({
    'id': df_test['id'],
    'rainfall': y_pred
})

submission.to_csv('submission.csv', index=False)

