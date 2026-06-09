import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



                       
import sys                         
import gc                          
import random                      
import warnings                    
from datetime import datetime      

SEED = 42
random.seed(SEED)


warnings.filterwarnings("ignore")

import numpy as np                 
import pandas as pd                


import matplotlib.pyplot as plt    
import seaborn as sns              
import plotly.express as px
import plotly.graph_objects as go


from scipy import stats           
from collections import Counter    
from scipy.stats import fisher_exact
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import skew



from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib   


from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test.head()


original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')



original.head()


train = pd.concat([train, original], ignore_index=True)



train.shape


sns.set(style="whitegrid")

# ===  general overview ===
print("Shapes -> train:", train.shape, " test:", test.shape)
print("\nColumns:", train.columns.tolist())
print("\n-- head --")
print(train.head())
print("\n-- info --")
print(train.info())
print("\n-- describe (numeric) --")
print(train.describe().T)

# ===  missing values & duplicate ids ===
print("\nMissing values (train):")
print(train.isnull().sum().loc[lambda x: x>0])
print("\nMissing values (test):")
print(test.isnull().sum().loc[lambda x: x>0])

print("\nDuplicate ids (train):", train['id'].duplicated().sum())
print("Duplicate ids (test):", test['id'].duplicated().sum())

# ===  target: distribution  ===
print("\nBeatsPerMinute summary:")
print(train['BeatsPerMinute'].describe())
print("Skew:", train['BeatsPerMinute'].skew(), "Kurtosis:", train['BeatsPerMinute'].kurtosis())

plt.figure(figsize=(8,4))
sns.histplot(train['BeatsPerMinute'], kde=True, bins=60)
plt.title("Distribution of BeatsPerMinute")
plt.xlabel("BeatsPerMinute")
plt.show()

plt.figure(figsize=(6,2))
plt.boxplot(train['BeatsPerMinute'], vert=False)
plt.title("Boxplot BeatsPerMinute")
plt.show()



feature_cols = [col for col in train.columns if col not in ["id", "BeatsPerMinute"]]

print("\nPlotting distributions of explanatory variables...")
for f in feature_cols:
    plt.figure(figsize=(6,3))
    sns.histplot(train[f], kde=True, bins=50, color="steelblue")
    plt.title(f"Distribution of {f}")
    plt.xlabel(f)
    plt.ylabel("Count")
    plt.show()



feature_cols = [c for c in train.columns if c not in ['id', 'BeatsPerMinute']]
print("\nFeatures considered:", feature_cols)



corr = train[feature_cols + ['BeatsPerMinute']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation matrix (features + target)")
plt.show()

# top features correlated with target
print("\nAbsolute correlation with target (sorted):")
print(corr['BeatsPerMinute'].abs().sort_values(ascending=False))


# ===  boxplots of explanatory variables ===
print("\nPlotting boxplots of explanatory variables...")
for f in feature_cols:
    plt.figure(figsize=(6,2))
    sns.boxplot(x=train[f], color="lightcoral")
    plt.title(f"Boxplot of {f}")
    plt.xlabel(f)
    plt.show()


def generate_features(df_input):
    numeric_columns = df_input.select_dtypes(include=[np.number]).columns
    df_input[numeric_columns] = df_input[numeric_columns].fillna(df_input[numeric_columns].median())

    df_input['TrackDurationSec'] = df_input['TrackDurationMs'] / 1000.0
    df_input['Rhythm_x_Energy'] = df_input['RhythmScore'] * df_input['Energy']
    df_input['Energy_x_AudioLoudness'] = df_input['Energy'] * df_input['AudioLoudness']
    df_input['Rhythm_Audio_Mult'] = df_input['RhythmScore'] * df_input['AudioLoudness']
    df_input['Vocal_Acoustic_Ratio'] = df_input['VocalContent'] / (df_input['AcousticQuality'] + 1e-6)
    df_input['Energy_Mood_Mult'] = df_input['Energy'] * df_input['MoodScore']
    df_input['Instrumental_Live_Mult'] = df_input['InstrumentalScore'] * df_input['LivePerformanceLikelihood']

    poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_vals = poly_transformer.fit_transform(df_input[['RhythmScore', 'AudioLoudness', 'Energy']])
    poly_columns = [f'poly_feat_{i}' for i in range(poly_vals.shape[1])]
    df_input[poly_columns] = poly_vals

    for col in ['TrackDurationMs', 'AudioLoudness', 'VocalContent']:
        if col in df_input.columns and skew(df_input[col].dropna()) > 0.5:
            if df_input[col].min() < 0:
                shift_val = abs(df_input[col].min()) + 1
                df_input[f'log_{col}'] = np.log1p(df_input[col] + shift_val)
            else:
                df_input[f'log_{col}'] = np.log1p(df_input[col].clip(lower=0))

    df_input['Duration_Bin'] = pd.qcut(df_input['TrackDurationMs'], q=10, labels=False, duplicates='drop')
    df_input['Energy_Bin'] = pd.qcut(df_input['Energy'], q=5, labels=False, duplicates='drop')
    
    return df_input

train = generate_features(train)
test = generate_features(test)

feature_cols = [col for col in train.columns if col not in ['id', 'BeatsPerMinute'] and train[col].nunique() > 1]



y = train["BeatsPerMinute"]
X = train.drop(columns=["BeatsPerMinute", "id"])  


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)                     
X_test_scaled = scaler.transform(test.drop(columns=["id"])) 


X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=test.drop(columns=["id"]).columns, index=test.index)


print("X_train shape:", X_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y shape:", y.shape)


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize, differential_evolution
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

class AdvancedBlending:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.optimal_weights = None
        self.meta_model = None
        self.rank_weights = None
        
    def create_stratified_folds(self, y, n_bins=10):
       
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
    
    def ensemble_predictions(self, models, X_train, y_train, X_test, use_stratified=True):
        
        n_models = len(models)
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
      
        oof_preds = np.zeros((n_train, n_models))
        test_preds = np.zeros((n_test, n_models))
        feature_importance = {}
        
       
        if use_stratified:
            kf = self.create_stratified_folds(y_train)
            folds = list(kf.split(X_train, pd.qcut(y_train, q=10, labels=False, duplicates='drop')))
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            folds = list(kf.split(X_train, y_train))
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            fold_model_preds = []
            
            
            for model_idx, (model_name, model_func, params) in enumerate(models):
                try:
                    if model_name == 'lgb':
                        model = lgb.LGBMRegressor(**params)
                        model.fit(
                            X_fold_train, y_fold_train,
                            eval_set=[(X_fold_val, y_fold_val)],
                            callbacks=[lgb.early_stopping(150, verbose=False)]
                        )
                        if fold == 0:
                            feature_importance[model_name] = model.feature_importances_
                            
                    elif model_name == 'cat':
                        model = CatBoostRegressor(**params)
                        model.fit(
                            X_fold_train, y_fold_train,
                            eval_set=(X_fold_val, y_fold_val),
                            early_stopping_rounds=150,
                            verbose=0
                        )
                        if fold == 0:
                            feature_importance[model_name] = model.feature_importances_
                            
                    elif model_name == 'xgb':
                        model = xgb.XGBRegressor(**params)
                        model.fit(
                            X_fold_train, y_fold_train,
                            eval_set=[(X_fold_val, y_fold_val)],
                            early_stopping_rounds=150,
                            verbose=False
                        )
                        if fold == 0:
                            feature_importance[model_name] = model.feature_importances_
                    
                   
                    val_pred = model.predict(X_fold_val)
                    oof_preds[val_idx, model_idx] = val_pred
                    test_preds[:, model_idx] += model.predict(X_test) / self.n_splits
                    fold_model_preds.append(val_pred)
                    
                except Exception as e:
                    print(f"Erreur avec le modÃ¨le {model_name} au fold {fold}: {str(e)}")
                    oof_preds[val_idx, model_idx] = np.mean(y_fold_train)
                    test_preds[:, model_idx] += np.mean(y_fold_train) / self.n_splits
                    fold_model_preds.append(np.full(len(val_idx), np.mean(y_fold_train)))
            
           
            if len(fold_model_preds) > 0:
                fold_blend = np.mean(fold_model_preds, axis=0)
                fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_blend))
                fold_scores.append(fold_rmse)
                print(f"Fold {fold+1} RMSE (equal blend): {fold_rmse:.4f}")
        
        print(f"Moyenne CV RMSE (equal blend): {np.mean(fold_scores):.4f} Â± {np.std(fold_scores):.4f}")
        self.feature_importance = feature_importance
        return oof_preds, test_preds
    
    def optimize_weights_advanced(self, oof_preds, y_true):
       
        n_models = oof_preds.shape[1]
        
        
        def objective_genetic(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)  # Normalisation
            blended = np.dot(oof_preds, weights)
            return np.sqrt(mean_squared_error(y_true, blended))
        
        bounds_genetic = [(0.001, 0.998)] * n_models
        result_genetic = differential_evolution(
            objective_genetic, bounds_genetic, 
            seed=self.random_state, maxiter=300, popsize=20
        )
        weights_genetic = np.abs(result_genetic.x)
        weights_genetic = weights_genetic / np.sum(weights_genetic)
        
        
        def objective_nlp(weights):
            blended = np.dot(oof_preds, weights)
            return np.sqrt(mean_squared_error(y_true, blended))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}  # w >= 0
        ]
        bounds_nlp = [(0.001, 0.999)] * n_models
        init_guess = np.ones(n_models) / n_models
        
        result_nlp = minimize(
            objective_nlp, init_guess, 
            method='SLSQP', bounds=bounds_nlp, constraints=constraints
        )
        weights_nlp = result_nlp.x if result_nlp.success else weights_genetic
        
      
        best_rmse_grid = float('inf')
        best_weights_grid = None
        
       
        step = 0.05
        weight_ranges = []
        for i in range(n_models):
            weight_ranges.append(np.arange(0.0, 1.0 + step, step))
        
        
        np.random.seed(self.random_state)
        for _ in range(1000):  
            weights = np.random.dirichlet(np.ones(n_models), size=1)[0]  
            blended = np.dot(oof_preds, weights)
            rmse = np.sqrt(mean_squared_error(y_true, blended))
            if rmse < best_rmse_grid:
                best_rmse_grid = rmse
                best_weights_grid = weights.copy()
        
       
        rmse_genetic = objective_genetic(weights_genetic)
        rmse_nlp = objective_nlp(weights_nlp)
        
        results = [
            ('Genetic Algorithm', weights_genetic, rmse_genetic),
            ('NLP Optimization', weights_nlp, rmse_nlp),
            ('Grid Search', best_weights_grid, best_rmse_grid)
        ]
        
       
        best_method, best_weights, best_rmse = min(results, key=lambda x: x[2])
        
        print(f"\nRÃ©sultats d'optimisation des poids:")
        for method, weights, rmse in results:
            print(f"{method}: RMSE = {rmse:.6f}, Poids = {weights}")
        print(f"\nMeilleure mÃ©thode: {best_method} (RMSE: {best_rmse:.6f})")
        
        self.optimal_weights = best_weights
        return best_weights
    
    def train_meta_model(self, oof_preds, y_true):
       
        meta_models = {
            'ridge': Ridge(alpha=10.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=self.random_state),
            'mlp': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=self.random_state)
        }
        
        best_meta_model = None
        best_meta_rmse = float('inf')
        meta_results = {}
        
       
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for name, model in meta_models.items():
            cv_scores = []
            for train_idx, val_idx in kf.split(oof_preds):
                X_meta_train, X_meta_val = oof_preds[train_idx], oof_preds[val_idx]
                y_meta_train, y_meta_val = y_true[train_idx], y_true[val_idx]
                
                try:
                    model.fit(X_meta_train, y_meta_train)
                    pred_meta = model.predict(X_meta_val)
                    rmse = np.sqrt(mean_squared_error(y_meta_val, pred_meta))
                    cv_scores.append(rmse)
                except:
                    cv_scores.append(float('inf'))
            
            avg_rmse = np.mean(cv_scores)
            meta_results[name] = avg_rmse
            
            if avg_rmse < best_meta_rmse:
                best_meta_rmse = avg_rmse
                best_meta_model = model
        
       
        if best_meta_model is not None:
            best_meta_model.fit(oof_preds, y_true)
            self.meta_model = best_meta_model
        
        print(f"\nRÃ©sultats des meta-modÃ¨les:")
        for name, rmse in meta_results.items():
            print(f"{name}: {rmse:.6f}")
        print(f"Meilleur meta-modÃ¨le: RMSE = {best_meta_rmse:.6f}")
        
        return best_meta_model
    
    def rank_averaging(self, oof_preds, test_preds, y_true):
        
        n_models = oof_preds.shape[1]
        
        
        oof_ranks = np.zeros_like(oof_preds)
        test_ranks = np.zeros_like(test_preds)
        
        for i in range(n_models):
            oof_ranks[:, i] = rankdata(oof_preds[:, i])
            test_ranks[:, i] = rankdata(test_preds[:, i])
        
       
        def rank_objective(weights):
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            blended_ranks = np.dot(oof_ranks, weights)
            return np.sqrt(mean_squared_error(y_true, blended_ranks))
        
        bounds = [(0.001, 0.999)] * n_models
        result = differential_evolution(
            rank_objective, bounds, seed=self.random_state, maxiter=200
        )
        
        rank_weights = np.abs(result.x)
        rank_weights = rank_weights / np.sum(rank_weights)
        self.rank_weights = rank_weights
        
        return rank_weights
    
    def final_blend(self, oof_preds, test_preds, y_true):
        
        print("=== BLENDING AVANCÃ‰ ===")
        
       
        optimal_weights = self.optimize_weights_advanced(oof_preds, y_true)
        blend_optimal = np.dot(oof_preds, optimal_weights)
        rmse_optimal = np.sqrt(mean_squared_error(y_true, blend_optimal))
        
        
        meta_model = self.train_meta_model(oof_preds, y_true)
        blend_meta = meta_model.predict(oof_preds) if meta_model else blend_optimal
        rmse_meta = np.sqrt(mean_squared_error(y_true, blend_meta)) if meta_model else rmse_optimal
        
        
        rank_weights = self.rank_averaging(oof_preds, test_preds, y_true)
        oof_ranks = np.zeros_like(oof_preds)
        for i in range(oof_preds.shape[1]):
            oof_ranks[:, i] = rankdata(oof_preds[:, i])
        blend_rank = np.dot(oof_ranks, rank_weights)
        rmse_rank = np.sqrt(mean_squared_error(y_true, blend_rank))
        
       
        methods = [
            ('Optimal Weights', blend_optimal, rmse_optimal, 'weights'),
            ('Meta Model', blend_meta, rmse_meta, 'meta'),
            ('Rank Averaging', blend_rank, rmse_rank, 'rank')
        ]
        
        best_method, best_blend, best_rmse, best_type = min(methods, key=lambda x: x[2])
        
        print(f"\n=== RÃ‰SULTATS FINAUX ===")
        for method, _, rmse, _ in methods:
            print(f"{method}: RMSE = {rmse:.6f}")
        print(f"\nMeilleure mÃ©thode: {best_method} (RMSE: {best_rmse:.6f})")
        
       
        if best_type == 'weights':
            final_test_preds = np.dot(test_preds, optimal_weights)
        elif best_type == 'meta' and meta_model:
            final_test_preds = meta_model.predict(test_preds)
        else:  # rank
            test_ranks = np.zeros_like(test_preds)
            for i in range(test_preds.shape[1]):
                test_ranks[:, i] = rankdata(test_preds[:, i])
            final_test_preds = np.dot(test_ranks, rank_weights)
        
        return final_test_preds, best_rmse, best_method



lgb_params = {
    "colsample_bytree": 0.6977764240714932,
    "learning_rate": 0.010200660341891958,
    "max_depth": 3,
    "min_child_samples": 7,
    "n_estimators": 711,
    "num_leaves": 190,
    "reg_alpha": 1.1411660265926687e-05,
    "reg_lambda": 8.328266237197566,
    "subsample": 0.5254691782296235,
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "random_state": 42
}

cat_params = {
    "iterations": 1856,
    "learning_rate": 0.0034955405986767337,
    "depth": 6,
    "l2_leaf_reg": 0.013723978895316472,
    "bagging_temperature": 0.13216576435438537,
    "border_count": 99,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0
}

xgb_params = {
    'n_estimators': 264,
    'learning_rate': 0.016147510440075844,
    'max_depth': 3,
    'subsample': 0.6373874482675941,
    'colsample_bytree': 0.5937026385399601,
    'reg_alpha': 0.00034470953918926047,
    'reg_lambda': 1.6911581294744957e-07,
    'min_child_weight': 4,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0
}


X_scaled = X_scaled.values if hasattr(X_scaled, "values") else X_scaled
X_test_scaled = X_test_scaled.values if hasattr(X_test_scaled, "values") else X_test_scaled
y = y.values if hasattr(y, "values") else y


models = [
    ('lgb', 'lgb', lgb_params),
    ('cat', 'cat', cat_params),
    ('xgb', 'xgb', xgb_params)
]


advanced_blender = AdvancedBlending(n_splits=5, random_state=42)


print("GÃ©nÃ©ration des prÃ©dictions avec validation croisÃ©e stratifiÃ©e...")
oof_preds, test_preds = advanced_blender.ensemble_predictions(
    models, X_scaled, y, X_test_scaled, use_stratified=True
)


final_predictions, final_rmse, best_method = advanced_blender.final_blend(
    oof_preds, test_preds, y
)





def intelligent_clipping(predictions, train_target, percentile_range=(1, 99)):
    """Clipping intelligent basÃ© sur les percentiles des donnÃ©es d'entraÃ®nement"""
    lower_bound = np.percentile(train_target, percentile_range[0])
    upper_bound = np.percentile(train_target, percentile_range[1])
    
    
    range_extension = (upper_bound - lower_bound) * 0.05
    lower_bound = max(lower_bound - range_extension, 40)  
    upper_bound = min(upper_bound + range_extension, 200)  
    
    return np.clip(predictions, lower_bound, upper_bound)


final_predictions = intelligent_clipping(final_predictions, y)

print(f"\n=== RÃ‰SULTAT FINAL ===")
print(f"MÃ©thode choisie: {best_method}")
print(f"RMSE final: {final_rmse:.6f}")
print(f"Plage des prÃ©dictions finales: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")


if advanced_blender.optimal_weights is not None:
    print(f"\nPoids optimaux:")
    model_names = ['LightGBM', 'CatBoost', 'XGBoost']
    for i, (name, weight) in enumerate(zip(model_names, advanced_blender.optimal_weights)):
        print(f"  {name}: {weight:.4f}")


submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
submission['BeatsPerMinute'] = final_predictions



submission.to_csv('submission.csv', index=False)



print(submission.head())


