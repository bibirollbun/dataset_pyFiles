from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, log_loss
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import VotingClassifier

import copy
from sklearn.base import clone
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from sklearn.pipeline import Pipeline


!pip install fancyimpute  > /dev/null 2>&1


from fancyimpute import IterativeSVD, SoftImpute, NuclearNormMinimization


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
dataset1_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')


binary_map = {'Yes': 1, 'No': 0}
personality_map = {'Extrovert': 1, 'Introvert': 0}

for df in [train_df, dataset1_df, test_df]:
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if 'Personality' in df.columns:
        df['Personality'] = df['Personality'].map(personality_map).fillna(-1)

    for col in cat_cols:
        if set(df[col].dropna().unique()) <= {'Yes', 'No'}:
            df[col] = df[col].map(binary_map).fillna(-1)


class MergingFeaturesCreator(BaseEstimator, TransformerMixin):
    """
    Adds strong and soft match features by merging input DataFrame with a reference DataFrame using exact 
    and nearest-neighbor (soft) matching.
    """
    def __init__(self, merge_df, merge_cols, soft_match_distance=1):
        self.merge_df = merge_df.copy()
        self.merge_cols = merge_cols
        self.distance = soft_match_distance
        self.nn = NearestNeighbors(n_neighbors=1)
        self.columns = None
        self.merge_df_prepared = None


    def fit(self, X, y=None):
        if 'Personality' not in self.merge_df.columns:
            raise ValueError("'Personality' column is missing from merge_df")

        self.merge_df_prepared = (
            self.merge_df                    
            .rename(columns={'Personality': 'match_p'})
            .drop_duplicates(self.merge_cols)
            )       
        
        # Fit Nearest Neighbors
        self.nn.fit(self.merge_df_prepared[self.merge_cols].fillna(-1))
        return self

    def transform(self, X):
        X = X.copy()
        
        # Strong match     
        X = X.merge(self.merge_df_prepared, how='left', on=self.merge_cols)

        # Soft match
        # Soft match features
        distances, indices = self.nn.kneighbors(X[self.merge_cols].fillna(-1))
        X['nn_distance'] = distances.flatten()
        X['match_index'] = indices.flatten()
        match_p_array = self.merge_df_prepared['match_p'].values
        soft_match_p = np.full(len(X), -1)  
        
        # Apply soft match condition
        mask_close = X['nn_distance'] < self.distance
        valid_indices = X.loc[mask_close, 'match_index'].astype(int)
        
        # Only assign if index is within bounds (very safe)
        within_bounds = valid_indices < len(match_p_array)
        soft_match_values = match_p_array[valid_indices[within_bounds]]
        soft_match_p[np.where(mask_close)[0][within_bounds]] = soft_match_values        
        X['soft_match_p'] = soft_match_p

        # Clean up
        X.drop(columns=['match_index', 'nn_distance'], inplace=True, errors='ignore')
        # X.set_index('id', inplace=True)
        self.columns = X.columns.tolist()
        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


class CrossValidatedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Cross-validated target encoding for categorical features.

    This transformer applies target encoding to specified categorical columns using 
    out-of-fold (OOF) strategy during training to avoid data leakage. It also fits a 
    final encoder on the full data to enable encoding of unseen test data.
    """
    
    def __init__(self, encode_cols: list, n_splits: int = 5):
        self.n_splits = n_splits
        self.encode_cols = encode_cols
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.oof_encoded_ = None
        self.columns_ = None
        self.final_encoder_ = None  # Fitted on full data for test transform

    def fit(self, X, y):
        X = X.copy()
        self.oof_encoded_ = pd.DataFrame(index=X.index, columns=self.encode_cols)
        
        for train_idx, val_idx in self.kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y[train_idx]
            
            te = TargetEncoder(cols=self.encode_cols)
            te.fit(X_train[self.encode_cols], y_train)
            self.oof_encoded_.iloc[val_idx] = te.transform(X_val[self.encode_cols]).values

        # Fit final encoder on full data to use on future unseen data (test set)
        self.final_encoder_ = TargetEncoder(cols=self.encode_cols)
        self.final_encoder_.fit(X[self.encode_cols], y)
        return self

    def transform(self, X):
        X = X.copy()
        if X.shape[0] == self.oof_encoded_.shape[0] and (X.index == self.oof_encoded_.index).all():
            # transform fitted data
            encoded = self.oof_encoded_
        else:
            # transform new(test) data
            encoded = self.final_encoder_.transform(X[self.encode_cols])
        
        X = X.drop(columns=self.encode_cols)
        for col in self.encode_cols:
            X[col + "_te"] = encoded[col].astype(float).values
        self.columns_ = X.columns.tolist()
        return X

    def get_feature_names_out(self):
        return self.columns_



num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear','Drained_after_socializing', 'match_p', 'soft_match_p']
merge_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency', 'Stage_fear', 'Drained_after_socializing']

categorical_pipeline = Pipeline([
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')),
   ])

numeric_pipeline = Pipeline([
    ('scaler', MinMaxScaler().set_output(transform='pandas')),
   ])

num_cat_preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols),
], remainder='passthrough').set_output(transform='pandas')  


preprocessing_pipeline = Pipeline([
    ('merge_prep', MergingFeaturesCreator(merge_df=dataset1_df, merge_cols = merge_cols, soft_match_distance=1)),
    # ('target_encoder', CrossValidatedTargetEncoder(encode_cols=['Stage_fear', 'Drained_after_socializing'])),
    ('num_cat_preprocessor', num_cat_preprocessor),  
    ])


y = train_df['Personality']
X = preprocessing_pipeline.fit_transform(train_df.drop('Personality', axis=1), y)
X.info()


class ImputerWrapper:    
    """
     A wrapper class for imputation objects that supports both inductive (train-only)
     and transductive (train+val combined) imputation strategies.

     Attributes:
        name (str): Name identifier for the imputer.
        imputer (object): An imputer instance that implements `fit` and `transform`.
        is_inductive (bool): If True, uses inductive imputation (fit on train only).
                             If False, uses transductive imputation (fit on train + val).
        feature_names_ (list or np.ndarray): List of output feature names after transformation.
    """
    
    def __init__(self, name, imputer, is_inductive=True):
        self.name = name
        self.imputer = imputer
        self.is_inductive = is_inductive
        self.feature_names_ = None

    def fit_transform(self, X_train, X_val, y_train=None):
        input_features = X_train.columns

        if not self.is_inductive or not hasattr(self.imputer, "transform"):
            imputer = deepcopy(self.imputer)
            X_all = pd.concat([X_train, X_val])
            X_filled = imputer.fit_transform(X_all)
            columns = self._get_feature_names(imputer, input_features, X_all.shape[1])
            self.feature_names_ = columns
            return (
                pd.DataFrame(X_filled[:len(X_train)], index=X_train.index, columns=columns),
                pd.DataFrame(X_filled[len(X_train):], index=X_val.index, columns=columns)
            )
        else:
            imp = clone(self.imputer).fit(X_train, y_train)
            columns = self._get_feature_names(imp, input_features, X_train.shape[1])
            self.feature_names_ = columns

            X_train_imp_arr = imp.transform(X_train)
            X_val_imp_arr = imp.transform(X_val)

            assert X_train_imp_arr.shape[1] == len(columns), (
                f"Shape mismatch for {self.name} train: {X_train_imp_arr.shape[1]} vs columns {len(columns)}"
            )
            assert X_val_imp_arr.shape[1] == len(columns), (
                f"Shape mismatch for {self.name} val: {X_val_imp_arr.shape[1]} vs columns {len(columns)}"
            )

            return (
                pd.DataFrame(X_train_imp_arr, index=X_train.index, columns=columns),
                pd.DataFrame(X_val_imp_arr, index=X_val.index, columns=columns)
            )

    def _get_feature_names(self, imp, input_features, fallback_len):
        if hasattr(imp, 'get_feature_names_out'):
            return imp.get_feature_names_out(input_features=input_features)
        else:
            return list(input_features[:fallback_len])



class ClassifierEvaluator:
    """
    A wrapper for evaluating classification models, providing a unified interface
    for fitting, predicting, and scoring models on validation data.

    Attributes:
        name (str): Name identifier for the classifier.
        model (object): A cloned classifier instance that supports `fit` and `predict` or `predict_proba`.
    """
    def __init__(self, name, model):
        self.name = name
        self.model = clone(model)

    def fit_predict(self, X_train, y_train, X_val):
        self.model.fit(X_train, y_train)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_val)[:, 1]
        return self.model.predict(X_val)

    def score(self, y_true, y_pred):
        acc = np.mean((y_pred > 0.5) == y_true)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return {
            "accuracy": acc,
            "log_loss": log_loss(y_true, y_pred)
        }



class ImputerStrategyEvaluator:
    """
    Evaluates combinations of imputation strategies and classifiers using cross-validation.

    This class systematically evaluates the effect of different imputation strategies
    on classification performance by performing cross-validation and collecting
    out-of-fold (OOF) predictions and performance metrics.

    Attributes:
        imputers (dict): A dictionary of imputer name → imputer object.
        classifiers (dict): A dictionary of classifier name → classifier object.
        cv (StratifiedKFold): Stratified K-fold cross-validator.
        results (list): List of per-fold evaluation dictionaries.
        oof_preds (dict): Dictionary with keys '<imputer>__<classifier>' → OOF predictions array.
    """
    def __init__(self, imputers: dict, classifiers: dict, cv=5):
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        self.imputers = {
            name: ImputerWrapper(name, imp, is_inductive=name not in ['soft_impute', 'impute_svd'])
            for name, imp in imputers.items()
        }
        self.classifiers = {
            name: ClassifierEvaluator(name, clf) for name, clf in classifiers.items()
        }
        self.results = []
        self.oof_preds = {}


    def evaluate(self, X, y):
        n_samples = X.shape[0]
        for imp_name in self.imputers:
            for clf_name in self.classifiers:
                self.oof_preds[f"{imp_name}__{clf_name}"] = np.zeros(n_samples)

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for imp_name, imputer in self.imputers.items():
                X_train_imp, X_val_imp = imputer.fit_transform(X_train, X_val, y_train)

                for clf_name, clf_eval in self.classifiers.items():
                    preds = clf_eval.fit_predict(X_train_imp, y_train, X_val_imp)
                    scores = clf_eval.score(y_val, preds)
                    self.oof_preds[f"{imp_name}__{clf_name}"][val_idx] = preds

                    self.results.append({
                        'imputer': imp_name,
                        'classifier': clf_name,
                        **scores
                    })

    def get_oof_dataframe(self):
        return pd.DataFrame(self.oof_preds)

    def get_metric_summary(self):
        df = pd.DataFrame(self.results)
        return df.groupby(['imputer', 'classifier']).agg(['mean']).reset_index()



class SupervisedImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using supervised models trained per column.
    Optionally adds binary flags for missing entries.
    """
    
    def __init__(self, 
                 model=RandomForestClassifier(), 
                 columns=None,
                 random_state=42,
                 add_missing_flags=True):
        self.columns = columns
        self.model = model
        self.random_state = random_state
        self.models_ = {}
        self.add_missing_flags = add_missing_flags
        self.missing_flag_cols_ = []

    def fit(self, X, y=None):
        X = X.copy()
        self.columns_ = self.columns or X.columns[X.isnull().any()].tolist()
        self.missing_flag_cols_ = [f"{col}_missing_flag" for col in self.columns_]

        for col in self.columns_:
            model = copy.deepcopy(self.model)

            df_known = X[X[col].notnull()]
            if df_known.empty:
                continue

            features = df_known.drop(columns=[col])
            target = df_known[col]

            not_null_mask = features.notnull().all(axis=1)
            features = features[not_null_mask]
            target = target[not_null_mask]

            model.fit(features, target)
            self.models_[col] = model

        return self

    def transform(self, X):
        X = X.copy()

        # Store missing mask BEFORE imputation
        missing_masks = {}
        if self.add_missing_flags:
            for col in self.columns_:
                missing_masks[col] = X[col].isnull()

        # Perform supervised imputation without adding flags as features
        for col, model in self.models_.items():
            missing_mask = X[col].isnull()
            if not missing_mask.any():
                continue

            features = X.loc[missing_mask].drop(columns=[col])
            usable_mask = features.notnull().all(axis=1)
            features = features[usable_mask]
            if features.empty:
                continue

            preds = model.predict(features).flatten()
            indices_to_fill = X.loc[missing_mask].index[usable_mask]
            X.loc[indices_to_fill, col] = preds
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median()) 

        # Now add missing flags columns AFTER imputation (do not pass to models)
        if self.add_missing_flags:
            for col in self.columns_:
                flag_col = f"{col}_missing_flag"
                X[flag_col] = missing_masks[col].astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = list(self.columns_) if hasattr(self, 'columns_') else []
        return list(input_features) + self.missing_flag_cols_
                



imputer_dict = {
    'simple': SimpleImputer(strategy='median'),
    'knn': KNNImputer(n_neighbors=40, add_indicator=True, weights='distance'),
    'rf': SupervisedImputer(model=RandomForestRegressor()),
    'xgb': SupervisedImputer(model=XGBRegressor(use_label_encoder=False, eval_metric='logloss')),
    'catb': SupervisedImputer(model=CatBoostRegressor(random_seed=42,  verbose=False)),
    'mice_br': IterativeImputer(estimator=BayesianRidge(),  
                                max_iter=10,  
                                random_state=42,  
                                initial_strategy='median',  
                                imputation_order='ascending',
                                add_indicator=True,  
                                n_nearest_features=20),
    'mice_rf': IterativeImputer(estimator=RandomForestRegressor(), 
                                max_iter=10,  
                                random_state=42,  
                                initial_strategy='median',  
                                imputation_order='ascending',
                                add_indicator=True,  
                                n_nearest_features=20),
    'mice_cat': IterativeImputer(estimator=CatBoostRegressor(verbose=0), 
                                max_iter=10,  
                                random_state=42,  
                                initial_strategy='median',  
                                imputation_order='ascending',
                                add_indicator=True,  
                                n_nearest_features=20),
     'mice_lgbm': IterativeImputer(estimator=LGBMRegressor(verbose=-1), 
                                max_iter=20,  
                                random_state=42,  
                                initial_strategy='median',  
                                imputation_order='ascending',
                                add_indicator=True,  
                                n_nearest_features=20),
    'impute_svd': IterativeSVD(rank=10, verbose=0),               
    'soft_impute': SoftImpute(verbose=0),  
}


num_positive = (y == 1).sum()
num_negative = (y == 0).sum()
scale_pos_weight = num_negative / num_positive

classifier_dict = {
    'lgbm': LGBMClassifier(
        random_state=42,
        class_weight='balanced',
        verbose=-1,
    ),
    'xgb': XGBClassifier(
        random_state=42,
        verbosity=0,      
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
    ),
    'cat': CatBoostClassifier(
        random_state=42,
        verbose=0,       
        auto_class_weights='Balanced',
    ),
    'rf': RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        verbose=0
    )
}


test_imputers = ImputerStrategyEvaluator(imputers=imputer_dict, classifiers = classifier_dict, cv=7)
# test_imputers.evaluate(X, y)


# results = test_imputers.get_metric_summary()
# oof = test_imputers.get_oof_dataframe()
# oof.to_csv('oof.csv')


## upload results from previous steps (with merging but without target encoding)
results = pd.read_csv('/kaggle/input/imputer-strategies-assesmentoof/imputer_strategies_assesmentnew.csv', index_col = 0)
oof = pd.read_csv('/kaggle/input/imputer-strategies-assesmentoof/oof_new.csv', index_col = 0)


results = results.drop(results.index[0])
results['accuracy'] = pd.to_numeric(results['accuracy'], errors='coerce')
results['log_loss'] = pd.to_numeric(results['log_loss'], errors='coerce')
results.index = results.index.astype(int)

def style_results_table(df):
    styled = (
        df.style
        .set_sticky().set_table_attributes("style='width:100%'")
        .format({'accuracy': '{:.4f}', 'log_loss': '{:.4f}'})
        .bar(subset=['accuracy'], color='rgba(141, 93, 148, 0.6)', vmin=0.965, vmax=df['accuracy'].max())
        .bar(subset=['log_loss'], color='rgba(106, 95, 148, 0.6)', vmin=0, vmax=df['log_loss'].max())
        .set_caption("Model Performance Across Imputation Strategies")
        .set_properties(**{'text-align': 'center'})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "caption", "props": [("caption-side", "top"), ("font-weight", "bold")]}
        ])
    )
    return styled

style_results_table(results.sort_values(['log_loss', 'accuracy']))


def hill_climb_ensemble(oof_df, y_true, max_models=None):
    remaining = list(oof_df.columns)
    selected = []
    best_score = np.inf
    best_comb = None

    for _ in range(len(remaining) if max_models is None else max_models):
        improved = False
        for col in remaining:
            trial = selected + [col]
            avg_preds = oof_df[trial].mean(axis=1)
            score = log_loss(y_true, avg_preds)
            if score < best_score:
                best_score = score
                best_comb = trial
                improved = True
        if improved:
            selected = best_comb
            remaining = [c for c in remaining if c not in selected]
        else:
            break

    return selected, best_score


models_used, final_score = hill_climb_ensemble(oof, y)
print("Best model stack:", models_used)
print("Final log loss:", final_score)


X_full = preprocessing_pipeline.fit_transform(pd.concat([train_df.drop('Personality', axis=1), test_df]))
X_train = X_full[X_full.index.isin(train_df.index)]
X_test = X_full[X_full.index.isin(test_df.index)]


class SoftImputeWrapper(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible wrapper for the FancyImpute SoftImpute imputer.

    This wrapper fits SoftImpute on a predefined full dataset (hardcoded),
    performs matrix completion, and applies the learned imputation to
    subsets of data (e.g., train or test) by indexing into the full imputed matrix.

    Note:
    - The 'fit' method uses a hardcoded full dataset `X_full` that must be
      accessible in the enclosing scope.
    - SoftImpute requires fitting on the entire data to perform effective
      imputation, so this approach ensures consistency across train/test splits.
    - During transform, it returns imputed rows matching the indices of the input.
    """
    
    def __init__(self, **kwargs):
        self.model_params = kwargs
        self.model = SoftImpute(**kwargs)
        self._is_fitted = False

    def fit(self, X, y=None):
        X_to_fit = self._validate_array(X_full)       # Yes, this is hardcoded — no alternative way to integrate it into the pipeline has been found so far
        self.X_imputed = self.model.fit_transform(X_to_fit)
        self.X_imputed_df = pd.DataFrame(
            self.X_imputed,
            columns=X_full.columns,
            index=X_full.index
        )
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("Transformer is not fitted yet.")
        return np.asarray(self.X_imputed_df.loc[X.index].sort_index())

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def _validate_array(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values.astype(float)
        return np.asarray(X, dtype=float)


pipelines = {
    'mice_lgbm_cat': Pipeline([
        ('imputer', imputer_dict['mice_lgbm']),
        ('model',classifier_dict['cat'])
    ]),
    'soft_impute_rf':  Pipeline([
        ('imputer',  SoftImputeWrapper(verbose=0)),  
        ('model', classifier_dict['rf'])
    ]),
    'simple_cat': Pipeline([
        ('imputer', imputer_dict['simple']),
        ('model',classifier_dict['cat'])
    ]),
    'knn_rf': Pipeline([
        ('imputer',  imputer_dict['knn']),
        ('model', classifier_dict['rf'])
    ]),
}

# Create voting regressor
voting_clf = VotingClassifier(estimators=list(pipelines.items()), voting='soft')


voting_clf.fit(X_train, y)


preds = voting_clf.predict(X_test)
predictions = pd.DataFrame({'Personality': preds}, index=test_df.index)
predictions.index.name = 'id'
predictions['Personality'] = predictions['Personality'].map({1: 'Extrovert', 0: 'Introvert'})

# Save to CSV with index
predictions.to_csv('submission.csv')


predictions

