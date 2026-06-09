import numpy as np 
import pandas as pd

import shap


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


import plotly.express as px 
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
xtest = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
ytest = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


print(train.shape)
print(xtest.shape)
print(ytest.shape)


test = pd.concat([xtest, ytest], axis=1)
test.drop(columns="id" , axis=1 , inplace=True)

df = pd.concat([train, test], axis=0, ignore_index=True)


df.drop(columns="id" , axis=1 , inplace=True)


df.sample(5)


print(df.shape)
df.dtypes


df.isna().sum().sum()


df.duplicated().sum()


df = df.drop_duplicates()


df.shape


### Split Categorical & Numerical  cols 
num_cols =  df.select_dtypes(include='number').columns.tolist()
cat_cols = df.select_dtypes(exclude='number').columns.tolist()
num_cols.remove("accident_risk")


num_cols


df[num_cols].describe()


# numerical features correlation
correlation_matrix = df[num_cols + ['accident_risk']].corr()
corr_df = correlation_matrix.reset_index().melt(id_vars='index')
fig = px.imshow(correlation_matrix,text_auto='.2f',title='Numerical_Featuer_corr + Target',)

fig.update_layout(width=800, height=600)
fig.show()


for col in cat_cols :
    print(f"type : {df[col].dtypes}")
    print(f"{col} nunique : {df[col].nunique()}")
    print(f"{col} unique : {df[col].unique()}\n")


fig , axes = plt.subplots(2,4, figsize=(16,8))
axes = axes.flatten()
cmap = plt.get_cmap("coolwarm")   
colors = cmap([1,0.9,0.29])
target = 'accident_risk'

for i, col in enumerate(cat_cols) :
    grouped = df.groupby(col)[target].mean()
    axes[i].bar(grouped.index.astype(str), grouped.values , color=colors) 
    
    axes[i].set_ylabel(f'Mean {target}')
    axes[i].set_title(f'{col} vs {target}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


def feature_engineering_pre_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['accidents_per_lane'] = df['num_reported_accidents'] / df['num_lanes'].replace(0, np.nan)
    df['accidents_per_lane'] = df['accidents_per_lane'].fillna(0)

    df['curvature_times_speed'] = df['curvature'] * df['speed_limit']
    df['speed_per_lane'] = df['speed_limit'] / df['num_lanes'].replace(0, np.nan)
    df['speed_per_lane'] = df['speed_per_lane'].fillna(df['speed_limit'])  # fallback

    df['curvature_bin'] = pd.cut(df['curvature'],
                                bins=[-1, 0.25, 0.5, 0.75, 1.0],
                                labels=['very_low','low','med','high']).astype(object)

    df['speed_bin'] = pd.cut(df['speed_limit'],
                            bins=[0, 35, 55, 70, 200],
                            labels=['low','medium','high','very_high']).astype(object)

    df['night_and_fog'] = ((df['lighting'] == 'night') & (df['weather'] == 'foggy')).astype(int)
    df['risky_weather'] = df['weather'].isin(['rainy','foggy']).astype(int)
    df['low_lanes'] = (df['num_lanes'] <= 2).astype(int)
    df['high_curvature_and_fast'] = ((df['curvature'] > 0.6) & (df['speed_limit'] >= 55)).astype(int)

    df['road_and_public'] = df['road_type'].astype(str) + '_' + df['public_road'].astype(str)
    df['time_school'] = df['time_of_day'].astype(str) + '_' + df['school_season'].astype(str)
    df['lighting_time'] = df['lighting'].astype(str) + '_' + df['time_of_day'].astype(str)

    lighting_map = {'daylight': 0, 'dim': 1, 'night': 2}
    df['lighting_ord'] = df['lighting'].map(lighting_map).fillna(0).astype(int)

    return df


def split_data(df: pd.DataFrame, target: str = 'accident_risk', test_size: float = 0.2, random_state: int = 42):
    df = df.copy()
    X = df.drop(columns=[target])
    y = df[target].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        shuffle=True)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train, y_test


def build_post_split_transformers(X_train: pd.DataFrame, mode: str = 'tree'):

    numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object','category','bool']).columns.tolist()

    if mode == 'tree':
        numeric_transformer = Pipeline(steps=[('passthrough', 'passthrough')])
        cat_transformer = Pipeline(steps=[('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]) # OrdinalEncoder Ø¨Ø±Ø§ÛŒ ØªØ¨Ø¯ÛŒÙ„ categorical Ø¨Ù‡ Ø§Ø¹Ø¯Ø§Ø¯ (fit Ø±ÙˆÛŒ train)
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols),('cat', cat_transformer, categorical_cols)], remainder='drop', sparse_threshold=0)

    elif mode == 'linear':# OneHot + StandardScaler
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        cat_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols),('cat', cat_transformer, categorical_cols)], remainder='drop', sparse_threshold=0)
    else:
        raise ValueError("mode must be 'tree' or 'linear'")

    return preprocessor, numeric_cols, categorical_cols


def prepare_data_pipeline(df: pd.DataFrame,
                          target: str = 'accident_risk',
                          test_size: float = 0.2,
                          random_state: int = 42,
                          mode: str = 'tree'):
    # 1) 
    df_fe = feature_engineering_pre_split(df)
    # 2) split
    X_train, X_test, y_train, y_test = split_data(df_fe, target=target,
                                                  test_size=test_size, random_state=random_state)

    # 3) 
    preprocessor, numeric_cols, categorical_cols = build_post_split_transformers(X_train, mode=mode)

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    feature_names = None
    try:
        if mode == 'linear':
            ohe = None
            for name, trans, cols in preprocessor.transformers_:
                if name == 'cat':
                    ohe = trans.named_steps['ohe']
            num_names = numeric_cols
            cat_names = []
            if ohe is not None:
                cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
            feature_names = num_names + cat_names
    except Exception:
        feature_names = None

    return {
        'X_train_prepared': X_train_prepared,
        'X_test_prepared': X_test_prepared,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'feature_names': feature_names,
        'X_train_raw': X_train,
        'X_test_raw': X_test
    }


def evaluate_models(result, show_importance=True):

    X_train_prepared = result['X_train_prepared']
    X_test_prepared = result['X_test_prepared']
    y_train = result['y_train']
    y_test = result['y_test']
    X_train_raw = result['X_train_raw']
    X_test_raw = result['X_test_raw']
    categorical_cols = result['categorical_cols']
    numeric_cols = result['numeric_cols']

    # 1) CatBoost
    print("ğŸš€ Training CatBoostRegressor ...")
    cat_features = [X_train_raw.columns.get_loc(c) for c in categorical_cols if c in X_train_raw.columns]
    cat_model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function='RMSE',
        verbose=False,
        random_seed=42)
    cat_model.fit(X_train_raw, y_train, cat_features=cat_features, eval_set=(X_test_raw, y_test), verbose=False)
    y_pred_cat = cat_model.predict(X_test_raw)

    # 2) LightGBM
    print("âš¡ Training LightGBMRegressor ...")
    lgb_model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=32,
        random_state=42
    )
    lgb_model.fit(X_train_prepared, y_train)
    y_pred_lgb = lgb_model.predict(X_test_prepared)

    # 3) XGBoost
    print("ğŸ”¥ Training XGBoostRegressor ...")
    xgb_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"  # Ø³Ø±ÛŒØ¹â€ŒØªØ±
    )
    xgb_model.fit(X_train_prepared, y_train)
    y_pred_xgb = xgb_model.predict(X_test_prepared)

    # Evaluate
    def evaluate(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\nğŸ“Š {name} Results:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  RÂ²:   {r2:.4f}")
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    results = {
        "CatBoost": evaluate(y_test, y_pred_cat, "CatBoost"),
        "LightGBM": evaluate(y_test, y_pred_lgb, "LightGBM"),
        "XGBoost": evaluate(y_test, y_pred_xgb, "XGBoost")
    }

  
    # Feature Importances (optional)
    if show_importance:
        import matplotlib.pyplot as plt

        print("\nğŸ“ˆ Top Feature Importances (CatBoost):")
        fi_cat = pd.DataFrame({
            "feature": cat_model.feature_names_,
            "importance": cat_model.feature_importances_
        }).sort_values(by="importance", ascending=False).head(15)

        plt.figure(figsize=(8, 5))
        plt.barh(fi_cat["feature"][::-1], fi_cat["importance"][::-1])
        plt.title("CatBoost Top Feature Importances")
        plt.tight_layout()
        plt.show()

    return results, {"cat": cat_model, "lgb": lgb_model, "xgb": xgb_model}



result = prepare_data_pipeline(df, target='accident_risk', test_size=0.2, random_state=42, mode='tree')



"""
Feature importance analysis:
 - Model internal importances (CatBoost.feature_importances_, LGB/XGB .feature_importances_)
 - Permutation importance (sklearn)
 - SHAP summary + dependence (for CatBoost and LightGBM/XGBoost)
 - Collinearity check and simple rules for dropping features
"""
TOP_K = 30   
RANDOM_STATE = 42

def summarize_importances(df_importances):
    df = df_importances.copy()
    df['importance_norm'] = df['importance'] / (df['importance'].sum() + 1e-12)
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    df['cum_importance'] = df['importance_norm'].cumsum()

    return df

# ---------- 1) 
def model_internal_importances(cat_model, lgb_model, xgb_model, result):
    X_train_raw = result['X_train_raw']  # dataframe (raw) used for CatBoost
    # For LGB/XGB we may rely on X_train_prepared and feature_names if available
    feature_names_raw = list(X_train_raw.columns)

    # CatBoost
    fi_cat = pd.DataFrame({
        'feature': feature_names_raw,
        'importance': cat_model.get_feature_importance()
    }).sort_values('importance', ascending=False)

    # LightGBM: if trained on prepared matrix, try to get feature names from result['feature_names'] else fallback
    try:
        lgb_feature_names = result.get('feature_names') or feature_names_raw
    except:
        lgb_feature_names = feature_names_raw

    fi_lgb = pd.DataFrame({
        'feature': lgb_feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # XGBoost
    try:
        xgb_feature_names = result.get('feature_names') or feature_names_raw
    except:
        xgb_feature_names = feature_names_raw

    fi_xgb = pd.DataFrame({
        'feature': xgb_feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return fi_cat, fi_lgb, fi_xgb

# ---------- 2) Permutation importance (robust) ----------
def compute_permutation_importance(model, X_test, y_test, mode='raw', n_repeats=10):
    """
    mode:
      - 'raw' -> X_test should be DataFrame (for CatBoost use raw df)
      - 'prepared' -> X_test is prepared matrix (for LGB/XGB)
    """
    r = permutation_importance(model, X_test, y_test,
                               n_repeats=n_repeats,
                               random_state=RANDOM_STATE,
                               scoring='neg_root_mean_squared_error')
    df = pd.DataFrame({
        'feature': X_test.columns if hasattr(X_test, 'columns') else [f'f{i}' for i in range(X_test.shape[1])],
        'importance': r.importances_mean,
        'std': r.importances_std
    }).sort_values('importance', ascending=False)
    return df

# ---------- 3) SHAP analysis ----------
def shap_analysis_cat(cat_model, X_sample, top_n=TOP_K):
    """
    X_sample: DataFrame used for CatBoost (raw features)
    """
    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(X_sample)
    # summary plot
    print("SHAP summary (CatBoost):")
    shap.summary_plot(shap_values, X_sample, max_display=top_n, show=True)

    return explainer, shap_values

def shap_analysis_tree(model, X_sample, top_n=TOP_K, model_name='LGB'):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    print(f"SHAP summary ({model_name}):")
    shap.summary_plot(shap_values, X_sample, max_display=top_n, show=True)

    return explainer, shap_values

# ---------- 4) Collinearity check 
def collinearity_check(X_df, threshold=0.95):
    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop, corr

# ---------- 5) Pipeline
def run_feature_analysis(result, cat_model, lgb_model, xgb_model, top_k=TOP_K):
    # raw dataframes
    X_train_raw = result['X_train_raw']
    X_test_raw = result['X_test_raw']
    X_train_prep = result['X_train_prepared']
    X_test_prep = result['X_test_prepared']
    y_train = result['y_train']
    y_test = result['y_test']

    # 1) internal importances
    fi_cat, fi_lgb, fi_xgb = model_internal_importances(cat_model, lgb_model, xgb_model, result)
    print("\nTop CatBoost internal features:")
    display(fi_cat.head(top_k))
    print("\nTop LightGBM internal features:")
    display(fi_lgb.head(top_k))
    print("\nTop XGBoost internal features:")
    display(fi_xgb.head(top_k))

    # 2) permutation importances (use raw X for CatBoost, prepared for others)
    print("\nComputing permutation importances (this may take time)...")
    perm_cat = compute_permutation_importance(cat_model, X_test_raw, y_test, mode='raw', n_repeats=10)
    # for LGB/XGB the X_test_prep might be numpy; convert to DataFrame if feature names exist
    if result.get('feature_names') is not None:
        X_test_prep_df = pd.DataFrame(X_test_prep, columns=result['feature_names'])
    else:
        # fallback: try to use X_test_raw columns (may be okay if transformer was passthrough)
        X_test_prep_df = pd.DataFrame(X_test_prep, columns=X_test_raw.columns[:X_test_prep.shape[1]])
    perm_lgb = compute_permutation_importance(lgb_model, X_test_prep_df, y_test, mode='prepared', n_repeats=10)
    perm_xgb = compute_permutation_importance(xgb_model, X_test_prep_df, y_test, mode='prepared', n_repeats=10)

    print("\nTop permutation importances (CatBoost):")
    display(perm_cat.head(top_k))
    print("\nTop permutation importances (LightGBM):")
    display(perm_lgb.head(top_k))
    print("\nTop permutation importances (XGBoost):")
    display(perm_xgb.head(top_k))

    # 3) SHAP (sample to save time)
    sample_n = min(2000, X_test_raw.shape[0])
    X_shap_sample = X_test_raw.sample(sample_n, random_state=RANDOM_STATE)
    print("\nRunning SHAP for CatBoost (this may be slow if sample large)...")
    shap_expl_cat, shap_vals_cat = shap_analysis_cat(cat_model, X_shap_sample, top_n=top_k)

    # for LGB/XGB shap: prefer prepared df with column names if available
    print("\nRunning SHAP for LightGBM ...")
    X_shap_prep = X_test_prep_df.sample(sample_n, random_state=RANDOM_STATE)
    shap_expl_lgb, shap_vals_lgb = shap_analysis_tree(lgb_model, X_shap_prep, top_n=top_k, model_name='LGB')

    # 4) collinearity on numeric features (raw)
    to_drop, corr = collinearity_check(X_train_raw.select_dtypes(include=[np.number]), threshold=0.95)
    print("\nHighly collinear features (threshold=0.95) suggested to drop or merge:")
    print(to_drop)

    # 5) Produce consolidated importance table (average ranks)
    # normalize and average ranks from internal and permutation for robustness
    def rank_df(df_imp, name_col='feature'):
        df = df_imp.copy()
        df = df[['feature','importance']].groupby('feature').sum().reset_index()
        df['rank'] = df['importance'].rank(ascending=False, method='dense')
        return df[['feature','rank']]

    r_cat_int = rank_df(fi_cat)
    r_cat_perm = rank_df(perm_cat)
    # merge ranks (left join on feature)
    merged = r_cat_int.merge(r_cat_perm, on='feature', how='outer', suffixes=('_int','_perm')).fillna(9999)
    merged['avg_rank'] = merged[['rank_int','rank_perm']].mean(axis=1)
    merged = merged.sort_values('avg_rank')
    print("\nConsolidated rank (CatBoost internal + permutation):")
    display(merged.head(30))

    return {
        'fi_cat': fi_cat, 'fi_lgb': fi_lgb, 'fi_xgb': fi_xgb,
        'perm_cat': perm_cat, 'perm_lgb': perm_lgb, 'perm_xgb': perm_xgb,
        'shap_cat_explainer': shap_expl_cat, 'shap_vals_cat': shap_vals_cat,
        'shap_lgb_explainer': shap_expl_lgb, 'shap_vals_lgb': shap_vals_lgb,
        'collinearity_to_drop': to_drop,
        'corr_matrix': corr,
        'consolidated_rank': merged
    }



results, models = run_feature_analysis(result, cat_model, lgb_model, xgb_model)

cat_model = models["cat"]
lgb_model = models["lgb"]
xgb_model = models["xgb"]


analysis_results = run_feature_analysis(result, cat_model, lgb_model, xgb_model)


keep_features = [
    'lighting_ord', 'speed_limit', 'curvature', 'risky_weather',
    'curvature_times_speed', 'num_reported_accidents',
    'curvature_bin', 'speed_bin', 'weather', 'lighting',
    'holiday', 'lighting_time', 'time_school', 'speed_per_lane', 'accidents_per_lane'
]
drop_features = [
    'time_of_day', 'night_and_fog', 'num_lanes', 'public_road',
    'road_and_public', 'road_type', 'road_signs_present',
    'high_curvature_and_fast', 'low_lanes', 'school_season'
]



data = prepare_data_pipeline(df)

# Ø¯Ø³ØªØ±Ø³ÛŒ Ø¨Ù‡ Ù‡Ø± Ø¨Ø®Ø´ Ø§Ø² Ø®Ø±ÙˆØ¬ÛŒ
X_train_prepared = data['X_train_prepared']
X_test_prepared = data['X_test_prepared']
y_train = data['y_train']
y_test = data['y_test']
preprocessor = data['preprocessor']
numeric_cols = data['numeric_cols']
categorical_cols = data['categorical_cols']
feature_names = data['feature_names']
X_train = data['X_train_raw']
X_test = data['X_test_raw']




# Ø­Ø°Ù� Ù�ÛŒÚ†Ø±Ù‡Ø§ÛŒ Ú©Ù…â€ŒØ§Ø«Ø±
X_train_reduced = X_train_prepared[keep_features].copy()
X_test_reduced = X_test_prepared[keep_features].copy()



from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_train, X_test, y_train, y_test, name):

        # Ø´Ù†Ø§Ø³Ø§ÛŒÛŒ Ø®ÙˆØ¯Ú©Ø§Ø± Ù�ÛŒÚ†Ø±Ù‡Ø§ÛŒ Ù…ØªÙ†ÛŒ Ø¨Ø±Ø§ÛŒ CatBoost
    cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

    if "CatBoost" in name:
        model.fit(X_train, y_train, cat_features=cat_features, verbose=0)
    else:
        model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"ğŸ“Š {name} Results (Reduced Features):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  RÂ²:   {r2:.4f}\n")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}

# Ù…Ø¯Ù„â€ŒÙ‡Ø§
cat_model_reduced = CatBoostRegressor(
    depth=6, learning_rate=0.05, iterations=500, verbose=0, random_state=42)
lgb_model_reduced = LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model_reduced = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)

# Ø§Ø±Ø²ÛŒØ§Ø¨ÛŒ
results_reduced = []
results_reduced.append(evaluate_model(cat_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, "CatBoost"))
results_reduced.append(evaluate_model(lgb_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, "LightGBM"))
results_reduced.append(evaluate_model(xgb_model_reduced, X_train_reduced, X_test_reduced, y_train, y_test, "XGBoost"))


