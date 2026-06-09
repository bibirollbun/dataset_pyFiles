import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import entropy, skew, kurtosis
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from sklearn.ensemble import VotingRegressor

palette = "ch:s=.25,rot=-.25"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test.head()


train.info()


cols = train.columns
cols


#identify nulls

train.isnull().sum()


# identify duplicates

duplicates = train.duplicated()
duplicates.sum()


train.sort_values(by="accident_risk", ascending=False)


numeric_cols = train.select_dtypes(include=["number"]).columns
cat_cols = train.select_dtypes(include=["object", "boolean"]).columns



numeric_cols


cat_cols


def cat_plots(
    df1, df2,
    features,
    target,
    labels=("Train", "Test"),
    palette=("steelblue", "#b98c2a"),
    box_color="seagreen",
    figsize=(12, 4)
):
    """
    Comparative categorical plots for two datasets:
    - Left: normalized bar plots comparing feature distributions between datasets.
    - Right: boxplots showing target distribution for each feature in df1.
    
    """

    n_features = len(features)
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, axes = plt.subplots(
        n_features, 2,
        figsize=(figsize[0], figsize[1] * n_features),
        constrained_layout=True
    )

    # Handle single-feature case
    if n_features == 1:
        axes = [axes]

    for i, feat in enumerate(features):
        # --- Prepare normalized data for barplot ---
        df1_temp = df1[[feat]].copy().reset_index(drop=True).assign(source=labels[0])
        df2_temp = df2[[feat]].copy().reset_index(drop=True).assign(source=labels[1])
        combined = pd.concat([df1_temp, df2_temp], ignore_index=True)

        counts = (
            combined.groupby(["source", feat])
            .size()
            .reset_index(name="count")
        )
        counts["proportion"] = counts.groupby("source")["count"].transform(lambda x: x / x.sum())

        # --- Left: normalized barplot ---
        sns.barplot(
            ax=axes[i][0], data=counts, x=feat, y="proportion",
            hue="source", palette=palette
        )
        axes[i][0].legend_.remove()
        axes[i][0].set_title(f"{feat} — normalized distribution ({labels[0]} vs {labels[1]})",
                             fontsize=11, fontweight="bold")
        axes[i][0].set_xlabel(feat, fontsize=9)
        axes[i][0].set_ylabel("Proportion", fontsize=9)
        axes[i][0].tick_params(axis='x', rotation=45)
        axes[i][0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        axes[i][0].set_ylim(0, counts["proportion"].max() * 1.15)

        # Add percentage labels
        for container in axes[i][0].containers:
            axes[i][0].bar_label(
                container,
                labels=[f"{v.get_height() * 100:.1f}%" for v in container],
                fontsize=8, padding=2
            )

        # --- Right: boxplot for df1 ---
        sns.boxplot(ax=axes[i][1], data=df1, x=feat, y=target, color=box_color)
        axes[i][1].set_title(f"{feat} — {target} distribution ({labels[0]})",
                             fontsize=11, fontweight="bold")
        axes[i][1].set_xlabel(feat, fontsize=9)
        axes[i][1].set_ylabel(target, fontsize=9)
        axes[i][1].tick_params(axis='x', rotation=45)
        axes[i][1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # --- Global legend ---
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc='upper center',
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )

    # --- Main title ---
    plt.suptitle(
        "Comparative Feature Plots (Normalized Counts & Target Distributions)",
        fontsize=14, fontweight="bold", y=1.04
    )

    plt.show()


cat_plots(train, test, cat_cols, target='accident_risk')


train['accident_risk'].describe()


plt.figure(figsize=(14, 4))
plt.subplots_adjust(wspace=0.6)

plt.subplot(1,2,1)
sns.boxplot(train, y = 'accident_risk',
            palette = ["#2E8B57"])
plt.title('Target Value (accident) BoxPlot')

plt.subplot(1,2,2)
sns.histplot(data = train, x='accident_risk', palette = palette, kde=True)
plt.title('Target Value (accident_risk) countplot', fontsize=12)
plt.xticks(rotation = 25)


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

    # cv, entropy, skewness, kurtosis
    cv_list = []
    skew_list = []
    kurt_list = []
    entropy_list = []

    if not numeric_df.empty:
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if col_data.mean() != 0:
                cv_list.append(col_data.std() / col_data.mean())
            else:
                cv_list.append(np.nan)
            skew_list.append(skew(col_data))
            kurt_list.append(kurtosis(col_data))
        
            hist, bin_edges = np.histogram(col_data, bins=10, density=True)
            hist = hist[hist > 0]
            entropy_list.append(entropy(hist))

    cv_mean = np.nanmean(cv_list)
    skew_mean = np.nanmean(skew_list)
    kurt_mean = np.nanmean(kurt_list)
    entropy_mean = np.nanmean(entropy_list)

    
    # Final summary table
    summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing values (%)",
            "Duplicate rows (%)",
            "Num:Cat ratio",
            "Outliers (%)",
            "Avg cardinality (categorical)",
            "Mean CV", 
            "Mean skewness", 
            "Mean kurtosis", 
            "Mean entropy"
        ],
        "Value": [
            n_rows,
            n_cols,
            round(missing_pct, 2),
            round(duplicates_pct, 2),
            ratio_num_cat,
            round(outlier_frac, 2) if outlier_frac else None,
            round(avg_cardinality, 2) if avg_cardinality else None,
            round(cv_mean, 3) if not np.isnan(cv_mean) else None,
            round(skew_mean, 3) if not np.isnan(skew_mean) else None,
            round(kurt_mean, 3) if not np.isnan(kurt_mean) else None,
            round(entropy_mean, 3) if not np.isnan(entropy_mean) else None
        ]
    })
    
    return summary


summary = eda_summary(train, target="BeatsPerMinute")
summary


def columns_complectation(df):

    """
    Enhance the input DataFrame with additional engineered features derived from
    existing numerical and categorical variables.

    The function performs:
      1. Basic mathematical transformations.
      2. Feature interactions and combinations.
      3. Binary condition-based indicators.
    """

    ndf = df.copy()
    
    #NEW FEATURES

    #math transformations
    ndf['curvature_sq'] = ndf['curvature'] ** 2
    ndf['speed_limit_sq'] = ndf['speed_limit'] ** 2

    #combinations
    # ndf['speed_curvature'] = ndf['speed_limit'] * ndf['curvature']
    ndf['speed_curvature_sq'] = ndf['speed_limit_sq'] * ndf['curvature_sq']
    ndf['lightning_vs_weather'] = ndf['lighting'] + ndf['weather']
    ndf['lanes_vs_speed'] = ndf['num_lanes'] * ndf['speed_limit']

    #risk conditions
    ndf['curvature_speed_risk'] = ((ndf['curvature'] > 0.7) & (ndf['speed_limit'] >= 60)).astype(int)
    ndf['speed_weather_risk'] = ((ndf['weather'] != 'foggy') & (ndf['speed_limit'] >= 40)).astype(int)
    ndf['curvature_weather_risk'] = ((ndf['weather'] != 'foggy') & (ndf['curvature'] > 0.7)).astype(int)
    ndf['speed_lightning_risk'] = ((ndf['lighting'] == 'night') & (ndf['speed_limit'] >= 55)).astype(int)
    ndf['curvature_lightning_risk'] = ((ndf['lighting'] == 'night') & (ndf['curvature'] > 0.7)).astype(int)
    
    return ndf


train = columns_complectation(train)
test = columns_complectation(test)


cat_cols = train.select_dtypes(include=["object", "boolean"]).columns


for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


def heatmap_plot(df, title_name):
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


train.columns


X_col = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
       'weather', 'road_signs_present', 'public_road', 'time_of_day',
       'holiday', 
       'school_season', 
       'num_reported_accidents', 
       # 'speed_curvature',
       'speed_curvature_sq',
       'curvature_sq',
       'speed_limit_sq',
       'lightning_vs_weather',
       'lanes_vs_speed',
       'curvature_speed_risk',
       'speed_weather_risk',
       'curvature_weather_risk',
       'speed_lightning_risk',
       'curvature_lightning_risk'
        ]
y_col = 'accident_risk'


X_train, X_valid, y_train, y_valid = train_test_split(train[X_col], train[y_col], test_size=0.2, random_state=42)


X_train


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


cat_params = {
    'learning_rate': 0.05,          # Step size shrinkage for updating weights.
    'l2_leaf_reg': 34,               # L2 regularization coefficient for leaf values (helps avoid overfitting).
    'depth': 10,                      # Maximum depth of each tree.
    'iterations': 800,              # Total number of boosting iterations (trees to build).
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
    'learning_rate': 0.06,           # Step size shrinkage to prevent overfitting.
    'max_depth': 6,                   # Maximum depth of a tree.
    'num_leaves': 64,                 # Maximum number of leaves per tree (controls model complexity).
    # 'lambda_l2': 7.0,                 # L2 regularization term on weights (alias for reg_lambda).
    'reg_lambda': 1,                  # Additional L2 regularization parameter.
    "n_estimators": 500,             # Number of boosting iterations (trees to build).
    'feature_fraction': 0.8,         # Fraction of features used in each iteration (to prevent overfitting).
    'bagging_fraction': 0.8,         # Fraction of data samples used for training each iteration.
    # 'subsample': 0.75,              # Alias for bagging_fraction (randomly select part of data per iteration).
    # 'colsample_bytree': 0.75,       # Alias for feature_fraction (fraction of features for each tree).
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
    'max_leaves': 128,                 # Maximum number of leaves allowed in a tree.
    "n_estimators": 600,              # Total number of boosting rounds (trees to build).
    "learning_rate": 0.039,            # Step size shrinkage to avoid overfitting.
    "max_depth": 7,                    # Maximum depth per tree (controls model complexity).
    "subsample": 0.85,                  # Fraction of training samples used per tree (prevents overfitting).
    "colsample_bytree": 0.85,           # Fraction of features used for building each tree.
    # "reg_lambda": 1.0,                 # [NEW] L2 regularization term (same as lambda_l2 in LGBM).
    # "reg_alpha": 0.0,                  # [NEW] L1 regularization (not explicitly in your LGBM set, but common pair).
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
print(f'The best RMSE for fractional combinations is {bs}')
print(f'The best weights for fractional combinations is {bw}')


models_names.append('FRA')
pred_fra = pred_cat*bw[0] + pred_lgbm*bw[1] + pred_xgb*bw[2]
rmse_fra = np.sqrt(mean_squared_error(y_valid, pred_fra))
models_predictions.append(rmse_fra)


r2_fra = r2_score(y_valid, pred_fra)
models_r2.append(r2_fra)
print("R2 (fractional):", r2_fra)


models = pd.DataFrame({'model':models_names, 'prediction':models_predictions, 'r2':models_r2})
models


top_features = pd.DataFrame({
        'CAT': cat_fi['Feature Id'].head(5).reset_index(drop=True),
        'LGBM': lgbm_fi['Feature Id'].head(5).reset_index(drop=True),
        'XGB': xgb_fi['Feature Id'].head(5).reset_index(drop=True)
    })
top_features


display(test)


idd = test['id']


scores_cat = cat_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'accident_risk':scores_cat})
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv('sub_cat_6ver.csv', index=False)
submission


scores_lgbm = lgbm_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'accident_risk':scores_lgbm})
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv('sub_lgbm_6ver.csv', index=False)
submission


scores_xgb = xgb_model.predict(test[X_col])
submission = pd.DataFrame({'id':idd, 'accident_risk':scores_xgb})
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv('sub_xgb_6ver.csv', index=False)
submission


scores_fra = scores_cat*bw[0] + scores_lgbm*bw[1] + scores_xgb*bw[2]
submission = pd.DataFrame({'id':idd, 'accident_risk':scores_fra})
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv('sub_fra_6ver.csv', index=False)
submission

