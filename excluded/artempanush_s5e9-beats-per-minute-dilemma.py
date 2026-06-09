import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import entropy
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "notebook_connected"

palette = "ch:s=.25,rot=-.25"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
train.head()


origin = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
origin.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
test.head()


train.info()


origin.info()


train = train.drop(columns=['id'])


train = pd.concat([train, origin], axis=0, ignore_index=True)


cols = train.columns
cols


train.isnull().sum()


# identify duplicates

duplicates = train.duplicated()
duplicates.sum()


#identify duplicates from columns with skewness

duplicates_subset = train.duplicated(subset = ['AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood'], keep=False)
duplicates_subset.sum()


train[duplicates_subset == True].head()


# train = train.drop_duplicates(subset=['AudioLoudness', 'VocalContent', 'AcousticQuality',
#        'InstrumentalScore', 'LivePerformanceLikelihood'], keep="first")


def histplots(df1, df2, features, labels, palette=("steelblue", "#b98c2a")):
    """
    Create histograms for two DataFrames in two columns.
    Each row contains histograms from df1 and df2 for the same feature.
    """

    n_features = len(features)
    n_rows = n_features  # one row per feature

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 3), constrained_layout=True)

    # Ensure axes iterable for single feature
    if n_features == 1:
        axes = [axes]

    for i, feat in enumerate(features):
        # Histogram for df1
        sns.histplot(ax=axes[i][0], data=df1, x=feat, color=palette[0], kde=True, bins=30)
        axes[i][0].set_title(f"{labels[0]} — {feat}", fontsize=11, fontweight="bold")
        axes[i][0].set_xlabel(feat, fontsize=9)
        axes[i][0].set_ylabel("Count", fontsize=9)
        axes[i][0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Histogram for df2
        sns.histplot(ax=axes[i][1], data=df2, x=feat, color=palette[1], kde=True, bins=30)
        axes[i][1].set_title(f"{labels[1]} — {feat}", fontsize=11, fontweight="bold")
        axes[i][1].set_xlabel(feat, fontsize=9)
        axes[i][1].set_ylabel("Count", fontsize=9)
        axes[i][1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        for ax in axes[i]:
            ax.tick_params(axis='both', which='major', labelsize=9)

    plt.suptitle("Comparative Histograms by Features", fontsize=14, fontweight="bold")
    plt.show()


feat = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']


histplots(train, test, feat, labels=("TRAIN", "TEST"), palette=("steelblue", "#b98c2a"))


def boxplots(df1, df2, features, labels, palette=("steelblue", "#b98c2a")):
    """
    Create compact boxplots for two DataFrames in two columns.
    Each row contains boxplots from df1 and df2 for the same feature.
    """

    n_features = len(features)
    n_rows = n_features  # one row per feature

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, n_rows * 3), constrained_layout=True)

    # Ensure axes iterable for single feature
    if n_features == 1:
        axes = [axes]

    for i, feat in enumerate(features):
        # Boxplot for df1
        sns.boxplot(ax=axes[i][0], y=df1[feat], color=palette[0])
        axes[i][0].set_title(f"{labels[0]} — {feat}", fontsize=11, fontweight="bold")
        axes[i][0].set_xlabel("")
        axes[i][0].set_ylabel(feat, fontsize=9)
        axes[i][0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Boxplot for df2
        sns.boxplot(ax=axes[i][1], y=df2[feat], color=palette[1])
        axes[i][1].set_title(f"{labels[1]} — {feat}", fontsize=11, fontweight="bold")
        axes[i][1].set_xlabel("")
        axes[i][1].set_ylabel(feat, fontsize=9)
        axes[i][1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        for ax in axes[i]:
            ax.tick_params(axis='both', which='major', labelsize=9)

    plt.suptitle("Compact Comparative Boxplots", fontsize=14, fontweight="bold")
    plt.show()


boxplots(train, test, feat, labels=("TRAIN", "TEST"), palette=("steelblue", "#b98c2a"))


def interactive_scatter_matrix(df, features=None, color=None):
    """
    Plot an interactive scatter matrix for numeric features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with numeric features.
    features : list of str, optional
        List of features to include (default: all numeric columns).
    color : str, optional
        Column name to use for coloring points.
    """
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()

    fig = px.scatter_matrix(
        df,
        dimensions=features,
        color=color,
        opacity=0.5,
        height=1100,
        width=1100
    )
    
    fig.update_layout(
        font=dict(size=6),   
        title=dict(font=dict(size=14))  
)
    fig.update_traces(diagonal_visible=False)  # remove diagonal histograms
    fig.update_layout(title="Interactive Scatter Matrix")
    fig.show()


interactive_scatter_matrix(train, features = cols, color = "BeatsPerMinute")


train['BeatsPerMinute'].describe()


plt.figure(figsize=(14, 4))
plt.subplots_adjust(wspace=0.6)

plt.subplot(1,2,1)
sns.boxplot(train, y = 'BeatsPerMinute',
            palette = palette)
plt.title('Target Value (BeatsPerMinute) Pie Plot')

plt.subplot(1,2,2)
sns.histplot(data = train, x='BeatsPerMinute', palette = palette, kde=True)
plt.title('Target Value (BeatsPerMinute) countplot', fontsize=12)
plt.xticks(rotation = 25)


def heatmap_plot(df, title_name):

    # corr = df.corr()
    # fig, axes = plt.subplots(figsize=(14, 8))
    # fig.text(0.12, 0.9, 'Strength of association between features and the target in training data', 
    #      fontsize=10, color='#666666', ha='left')
    # mask = np.zeros_like(corr)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(corr, mask=mask, linewidth = .3, cmap='Blues', annot=True, annot_kws={"fontsize":6})
    # plt.title(title_name)
    # plt.show()

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(12, 7))

    # Заголовок
    plt.title(title_name, fontsize=14, pad=40)  # pad увеличивает отступ от heatmap

    fig.set_facecolor('#f9fbfd')
    
    fig.text(
        0.123, 0.92, 
        'Strength of association between features and the target in training data', 
        fontsize=9, 
        color='#666666', 
        ha='left'
    )

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr, mask=mask, linewidth=.3, cmap='Blues', annot=True, annot_kws={"fontsize":6}
    )

plt.show()

heatmap_plot(train, 'Correlation of the Train Dataset')


correlations = train.corr(numeric_only=True)['BeatsPerMinute'].drop('BeatsPerMinute').sort_values(ascending=False)
correlations


def eda_summary(df: pd.DataFrame, target: str = None) -> pd.DataFrame:
    n_rows, n_cols = df.shape
    
    # Missing values (%)
    missing_pct = df.isna().sum().sum() / (n_rows * n_cols) * 100
    
    # Duplicate rows (%)
    duplicates_pct = df.duplicated().mean() * 100
    
    # Num:Cat ratio
    num_cols = df.select_dtypes(include=np.number).shape[1]
    cat_cols = df.select_dtypes(exclude=np.number).shape[1]
    ratio_num_cat = f"{num_cols}:{cat_cols}"
    
    # Mean correlation with target
    target_corr = None
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        corrs = df.corr(numeric_only=True)[target].drop(target).abs()
        target_corr = corrs.mean() if not corrs.empty else None
    
    # Noise level (proxy via 1-R2 from linear regression)
    noise_level = None
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        X = df.drop(columns=[target]).select_dtypes(include=np.number).dropna()
        y = df.loc[X.index, target]
        if not X.empty and y.nunique() > 1:
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            noise_level = 1 - r2
    
    # Outliers (%) using IQR
    outlier_frac = None
    numeric_df = df.select_dtypes(include=np.number).dropna()
    if not numeric_df.empty:
        outlier_counts = []
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_counts.append(((numeric_df[col] < lower) | (numeric_df[col] > upper)).mean())
        outlier_frac = np.mean(outlier_counts) * 100
    
    # Avg cardinality of categorical features
    avg_cardinality = None
    if cat_cols > 0:
        avg_cardinality = df.select_dtypes(exclude=np.number).nunique().mean()
    
    # Mean mutual information
    mi_score = None
    if target and target in df.columns:
        X = df.drop(columns=[target]).dropna()
        y = df.loc[X.index, target]
        if not X.empty and y.nunique() > 1:
            mi = mutual_info_regression(X.select_dtypes(include=np.number), y, discrete_features=False)
            if len(mi) > 0:
                mi_score = np.mean(mi)
    
    # Target balance (regression: coefficient of variation)
    target_balance = None
    if target and target in df.columns:
        y = df[target].dropna()
        if y.mean() != 0:
            target_balance = round(y.std() / abs(y.mean()), 3)
    
    # Final summary table
    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing values (%)",
            "Duplicate rows (%)",
            "Num:Cat ratio",
            "Mean correlation with target",
            "Noise level (1-R2)",
            "Outliers (%)",
            "Avg cardinality (categorical)",
            "Mean mutual information",
            "Target balance"
        ],
        "Value": [
            n_rows,
            n_cols,
            round(missing_pct, 2),
            round(duplicates_pct, 2),
            ratio_num_cat,
            round(target_corr, 3) if target_corr else None,
            round(noise_level, 3) if noise_level else None,
            round(outlier_frac, 2) if outlier_frac else None,
            round(avg_cardinality, 2) if avg_cardinality else None,
            round(mi_score, 3) if mi_score else None,
            target_balance
        ]
    })
    
    return summary


summary = eda_summary(train, target="BeatsPerMinute")
summary


def columns_complectation(df):

    """
    Create new features and other stuff
    """
    
    #new_features

    #times
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000

    #ratio
    df['ratio_acoustic_instrumental'] = df['AcousticQuality'] / (df['InstrumentalScore'] + 1e-8)
    df['ratio_rhythm_energy'] = df['RhythmScore'] / (df['Energy'] + 1e-8)
    df['ratio_vocal_instrumental'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-8)
    df['ratio_energy_audio'] = df['VocalContent'] / (df['AudioLoudness'] + 1e-8)

    #multiplication
    df['Live_with_Mood'] = df['LivePerformanceLikelihood'] * df['MoodScore']
    df['Rhythm_with_Duration'] = df['RhythmScore'] * df['TrackDurationMin']
    df['Rhythm_with_Audio'] = df['RhythmScore'] * df['AudioLoudness']
    df['Acoustic_with_Live'] = df['AcousticQuality'] * df['LivePerformanceLikelihood']
    df['Energy_with_Live'] = df['Energy'] * df['LivePerformanceLikelihood']
    df['Energy_with_Duration'] = df['Energy'] * df['TrackDurationMs']
    df['Energy_with_Vocal'] = df['Energy'] * df['VocalContent']
    df['Audio_with_Vocal'] = df['AudioLoudness'] * df['VocalContent']
    df['Audio_with_Acoustic'] = df['AudioLoudness'] * df['AcousticQuality']
    df['Audio_with_Instrumental'] = df['AudioLoudness'] * df['InstrumentalScore']
    df['Audio_with_Live'] = df['AudioLoudness'] * df['LivePerformanceLikelihood']

    #squares
    df["Energy_sq"] = df["Energy"] ** 2
    df['MoodScore_sq'] = df['MoodScore'] ** 2
    df['RhythmScore_sq'] = df["Energy"] ** 2
    df['VocalContent_sq'] = df['VocalContent'] ** 2
    df['AcousticQuality_sq'] = df['AcousticQuality'] ** 2

    #bins
    df['EnergyBin'] = pd.qcut(df['Energy'], 
                            q=[0, .2, .4, .6, .8, 1],
                            # q = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
                            labels = False)
    df['RhythmBin'] = pd.qcut(df['RhythmScore'], 
                            q=[0, .2, .4, .6, .8, 1],
                            # q = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
                            labels = False)

    return df


train = columns_complectation(train)
test = columns_complectation(test)


def heatmap_plot(df, title_name):

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(14, 8))

    # Заголовок
    plt.title(title_name, fontsize=14, pad=40)  # pad увеличивает отступ от heatmap

    fig.set_facecolor('#f9fbfd')
    
    fig.text(
        0.123, 0.92, 
        'Strength of association between features and the target in training data', 
        fontsize=9, 
        color='#666666', 
        ha='left'
    )

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr, mask=mask, linewidth=.3, cmap='Blues', annot=True, annot_kws={"fontsize":6}
    )

heatmap_plot(train, 'Correlation of the Train Dataset')


#choose columns for working without highly correlation

def drop_highly_correlated_features(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.9,
    started_features: list = None
) -> list:
    """
    Remove highly correlated features (> threshold), keeping specified started features.

    Logic:
    - If two features are strongly correlated:
        * remove the one that is NOT in started_features
        * if both or neither are in started_features → drop the one that has more strong correlations
        * if counts equal → drop the one with weaker correlation to target

    Parameters:
        df : DataFrame
        target : str — name of the target column
        threshold : float — correlation threshold (default = 0.9)
        started_features : list — features to always keep if possible
    
    Returns:
        List of remaining feature names (excluding target).
    """
    if started_features is None:
        started_features = []

    started_features = set(started_features)

    features = [col for col in df.columns if col != target]
    corr_matrix = df[features].corr().abs()
    target_corr = df[features].corrwith(df[target]).abs()

    to_drop = set()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            corr_val = corr_matrix.loc[f1, f2]

            if corr_val > threshold:
                # if one of them is started_feature → drop the other
                if f1 in started_features and f2 not in started_features:
                    to_drop.add(f2)
                    continue
                if f2 in started_features and f1 not in started_features:
                    to_drop.add(f1)
                    continue
                if f1 in started_features and f2 in started_features:
                    # both are started → skip deletion
                    continue

                # neither are started_features → use original logic
                f1_count = (corr_matrix.loc[f1] > threshold).sum()
                f2_count = (corr_matrix.loc[f2] > threshold).sum()

                if f1_count > f2_count:
                    to_drop.add(f1)
                elif f2_count > f1_count:
                    to_drop.add(f2)
                else:
                    # equal counts → drop one with weaker correlation to target
                    if target_corr[f1] >= target_corr[f2]:
                        to_drop.add(f2)
                    else:
                        to_drop.add(f1)

    remaining = [f for f in features if f not in to_drop]
    return remaining


X_col = drop_highly_correlated_features(train, target="BeatsPerMinute", threshold=0.9, started_features=cols)
X_col


y_col = 'BeatsPerMinute'


X_train, X_valid, y_train, y_valid = train_test_split(train[X_col], train[y_col], test_size=0.2, random_state=42)


# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_valid_scaled = scaler.transform(X_valid)


models_predictions = []
models_names = []
models_r2 = []


def modeling(model, model_name, X_train, y_train, X_valid, y_valid,
             models_names, models_predictions, models_r2):
    """
    Fit the model, make predictions, calculate RMSE and R2,
    and append results into the provided lists.
    """
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_valid)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    r2 = r2_score(y_valid, y_pred)

    # Append results into lists
    models_names.append(model_name)
    models_predictions.append(rmse)
    models_r2.append(r2)

    # Print results for quick inspection
    print(f"{model_name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return rmse, r2, y_pred


# def validation_modelling(model):

#     """
#     Performs K-fold cross-validation for a given model.
#     """
    
#     kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
#     scores = []

#     for train_idx, val_idx in kf.split(train[X_col], train[y_col]):

#         X_train_fold, X_valid_fold = train[X_col].iloc[train_idx], train[X_col].iloc[val_idx]
#         y_train_fold, y_valid_fold = train[y_col].iloc[train_idx], train[y_col].iloc[val_idx]
        
#         model.fit(X_train_fold, y_train_fold, eval_set = (X_valid_fold, y_valid_fold))

#         model_predict = model.predict(X_valid_fold)
#         rmse = np.sqrt(mean_squared_error(y_valid, model_predict))
#         scores.append(rmse)

#     return np.mean(scores), scores


cat_params = {
    'learning_rate': 0.007,          # Step size shrinkage for updating weights.
    'l2_leaf_reg': 34,               # L2 regularization coefficient for leaf values (helps avoid overfitting).
    'depth': 8,                      # Maximum depth of each tree.
    'iterations': 1700,              # Total number of boosting iterations (trees to build).
    'loss_function': 'RMSE',         # Objective function: Root Mean Squared Error for training.
    'eval_metric': 'RMSE',           # Evaluation metric used for validation and early stopping.
    'random_seed': 42,               # Random seed to ensure reproducibility.
    # "border_count": 410,           # Number of splits for numerical features (controls binarization granularity).
    'early_stopping_rounds': 200,    # Stop training if no improvement over these rounds.
    # 'od_type': "Iter",             # Overfitting detector type ("Iter" = stop after no progress in given rounds).
    # 'od_wait': 100                 # Number of rounds to wait before early stopping kicks in.
}


cat_model = CatBoostRegressor(**cat_params, verbose=False)


rmse_cat, r2_cat, pred_cat = modeling(cat_model, "CAT",
                                      X_train, y_train, X_valid, y_valid,
                                      models_names, models_predictions, models_r2)


cat_fi = cat_model.get_feature_importance(prettified=True)


plt.figure(figsize=(16, 5))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_cat, color='steelblue')
plt.xlabel('Y_valid', fontsize=12)
plt.ylabel('Predictions', fontsize=12)
plt.title('Pred vs Valid for Catboost')

plt.subplot(1,2,2)
sns.barplot(x='Importances', y='Feature Id', data=cat_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.yticks(fontsize=8)
plt.title('Feature Importance', fontsize=16)

plt.tight_layout()
plt.show()


lgbm_params = {
    'objective': "regression",        # The learning task, here set for regression problems.
    'metric': "rmse",                 # Evaluation metric: Root Mean Squared Error.
    'boosting_type': 'gbdt',          # Gradient Boosted Decision Trees (standard boosting method).
    'learning_rate': 0.001,           # Step size shrinkage to prevent overfitting.
    'max_depth': 6,                   # Maximum depth of a tree.
    'num_leaves': 64,                 # Maximum number of leaves per tree (controls model complexity).
    'lambda_l2': 7.0,                 # L2 regularization term on weights (alias for reg_lambda).
    'reg_lambda': 1,                  # Additional L2 regularization parameter.
    "n_estimators": 1000,             # Number of boosting iterations (trees to build).
    'feature_fraction': 0.85,         # Fraction of features used in each iteration (to prevent overfitting).
    'bagging_fraction': 0.85,         # Fraction of data samples used for training each iteration.
    # 'subsample': 0.66,              # Alias for bagging_fraction (randomly select part of data per iteration).
    # 'colsample_bytree': 0.63,       # Alias for feature_fraction (fraction of features for each tree).
    # 'reg_alpha': 9,                 # L1 regularization term on weights (sparsity).
    # 'reg_lambda': 4.5,              # L2 regularization term on weights (stability).
    'n_jobs': -1                      # Number of parallel threads (-1 means use all available cores).
}


lgbm_model = LGBMRegressor(**lgbm_params, verbose=-1)


rmse_lgbm, r2_lgbm, pred_lgbm = modeling(lgbm_model, "LGBM",
                                      X_train, y_train, X_valid, y_valid,
                                      models_names, models_predictions, models_r2)


lgbm_fi = pd.DataFrame({'Feature Id':X_col, 'LGBM_Importances':lgbm_model.feature_importances_})
lgbm_fi = lgbm_fi.sort_values(by='LGBM_Importances', ascending=False)


plt.figure(figsize=(15, 6))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_lgbm, color='steelblue')
plt.xlabel('Y_valid', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.title('Pred vs Valid for LightGBM')

plt.subplot(1,2,2)
sns.barplot(x='LGBM_Importances', y='Feature Id', data=lgbm_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance', fontsize=16)

plt.tight_layout()
plt.show()


xgb_params = {
    'objective': 'reg:squarederror',   # Regression objective: predicts continuous values using squared error.
    'eval_metric': 'rmse',             # Evaluation metric: Root Mean Squared Error.
    'grow_policy': 'lossguide',        # Tree growing strategy ("lossguide" grows leaf-wise, can improve accuracy).
    'min_child_weight': 2.5,           # Minimum sum of instance weights (hessian) needed in a child node.
    'max_leaves': 100,                 # Maximum number of leaves allowed in a tree.
    "n_estimators": 1100,              # Total number of boosting rounds (trees to build).
    "learning_rate": 0.009,            # Step size shrinkage to avoid overfitting.
    "max_depth": 7,                    # Maximum depth per tree (controls model complexity).
    "subsample": 0.8,                  # Fraction of training samples used per tree (prevents overfitting).
    "colsample_bytree": 0.8,           # Fraction of features used for building each tree.
    "random_state": 42,                # Random seed to ensure reproducibility.
    # "min_child_weight": 3,           # (alternative setting) Minimum sum of hessian in child nodes.
}

xgb_model = XGBRegressor(**xgb_params,verbose=False)


rmse_xgb, r2_xgb, pred_xgb = modeling(xgb_model, "XGB",
                                      X_train, y_train, X_valid, y_valid,
                                      models_names, models_predictions, models_r2)


xgb_fi = pd.DataFrame({'Feature Id':X_col, 'XGB_Importances':xgb_model.feature_importances_})
xgb_fi = xgb_fi.sort_values(by='XGB_Importances', ascending=False)


plt.figure(figsize=(15, 6))

plt.subplot(1,2,1)
plt.scatter(y_valid, pred_xgb, color='steelblue')
plt.xlabel('Y_valid', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.title('Pred vs Valid for XGBBoost')

plt.subplot(1,2,2)
sns.barplot(x='XGB_Importances', y='Feature Id', data=xgb_fi, color='#a7c2e1', orient='h')
plt.xlabel('Importances', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance', fontsize=16)

plt.tight_layout()
plt.show()


voting_model = VotingRegressor(estimators = [
    ('cat', cat_model),
    ('lgbm', lgbm_model),
    ('xgb', xgb_model)
], n_jobs=-1)
    


rmse_voting, r2_voting, pred_voting = modeling(voting_model, "VOT",
                                      X_train, y_train, X_valid, y_valid,
                                      models_names, models_predictions, models_r2)


y_preds = [pred_cat, pred_lgbm, pred_xgb]

def weight_search_coarse(y_valid, preds):
    """
    Perform a grid search for optimal weights (a, b, c)
    to minimize RMSE between true values and the ensemble prediction.
    """

    best_score = np.inf
    best_weights = (1.0, 0.0, 0.0)

    for a in np.linspace(0, 1, 21):  # step = 0.05
        for b in np.linspace(0, 1 - a, int((1 - a) / 0.05) + 1):
            c = 1.0 - a - b
            w = np.array([a, b, c])
            w_preds = preds[0]*w[0] + preds[1]*w[1] + preds[2]*w[2]

            score = np.sqrt(mean_squared_error(y_valid, w_preds))
            if score < best_score:
                best_score = score
                best_weights = (a, b, c)

    return best_score, best_weights


bs, bw = weight_search_coarse(y_valid, y_preds)
print(f'The best RMSE for fractional ensemble is {bs}')
print(f'The best weights for fractional ensemble is {bw}')


models_names.append('FRA')
pred_fra = pred_cat*bw[0] + pred_lgbm*bw[1] + pred_xgb*bw[2]
rmse_fra = np.sqrt(mean_squared_error(y_valid, pred_fra))
models_predictions.append(rmse_fra)


r2_fra = r2_score(y_valid, pred_fra)
models_r2.append(r2_fra)
print("R2 (fractional):", r2_fra)


models = pd.DataFrame({'model':models_names, 'prediction':models_predictions, 'r2':models_r2})
models


display(test)


idd = test['id']


scores = cat_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'BeatsPerMinute':scores})
submission.to_csv('sub_cat_27ver.csv', index=False)
submission


scores2 = lgbm_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'BeatsPerMinute':scores2})
submission.to_csv('sub_lgbm_27ver.csv', index=False)
submission


scores3 = xgb_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'BeatsPerMinute':scores3})
submission.to_csv('sub_xgb_27ver.csv', index=False)
submission


scores4 = scores*bw[0] + scores2*bw[1] + scores3*bw[2]
submission = pd.DataFrame({'id':idd, 'BeatsPerMinute':scores4})
submission.to_csv('sub_comb_23ver.csv', index=False)
submission

