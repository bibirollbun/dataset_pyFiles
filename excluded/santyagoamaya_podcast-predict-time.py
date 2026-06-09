# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.base            import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model     import LinearRegression
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing    import StandardScaler, OrdinalEncoder          # <- label style
from sklearn.impute           import SimpleImputer
from sklearn.metrics          import mean_squared_error
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import PolynomialFeatures


train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv'), pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')



# Assuming train and test DataFrames are already defined
# Create DataFrames
pro_train = pd.DataFrame()
pro_test = pd.DataFrame()

# Fill DataFrames with data
pro_train['Genre'] = train['Genre']
pro_test['Genre'] = test['Genre']
pro_train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage']
pro_train['Host_Popularity_percentage'] = train['Host_Popularity_percentage']
y = train['Listening_Time_minutes']
pro_test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage']
pro_test['Host_Popularity_percentage'] = test['Host_Popularity_percentage']



# ------------ helpers -------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def num_preprocessor():
    """Imputer + scaler for numeric cols."""
    return Pipeline([("imp", SimpleImputer(strategy="mean")),
                     ("sc",  StandardScaler())])
def create_label_pipeline(cat_cols, num_cols):
    cat_transformer = Pipeline([("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                       unknown_value=-1))])
    
    preproc = ColumnTransformer(
                transformers=[
                    ("num", num_preprocessor(), num_cols),
                    ("cat", cat_transformer,     cat_cols)
                ])
    
    pipe = Pipeline([("preprocessor", preproc),
                     ("regressor",    LinearRegression())])
    return pipe
def train_per_category_models(X, y, cat_feature, num_features,
                              test_size=0.20, random_state=42):
    """
    1) global split
    2) inside the training fold: one LinearRegression for each category
    3) return dict with {category : (pipeline, rmse_val)} and a summary df
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size   = test_size,
                                                      random_state= random_state,
                                                      stratify    = X[cat_feature])

    models_dict = {}
    summary_rows = []

    for cat in sorted(X_train[cat_feature].unique()):
        # ­­­­­­­­­Select only this category
        idx_tr  = X_train[cat_feature] == cat
        idx_val = X_val[cat_feature]   == cat
        
        X_tr_cat,  y_tr_cat  = X_train.loc[idx_tr,  :], y_train.loc[idx_tr]
        X_val_cat, y_val_cat = X_val.loc[idx_val, :], y_val.loc[idx_val]
        
        # pipeline without the categorical column (it's constant now)
        preproc_cat = ColumnTransformer(
                        [("num", num_preprocessor(), num_features)],
                        remainder = "drop")

        pipe_cat = Pipeline([("preprocessor", preproc_cat),
                             ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                             ("regressor",    LinearRegression())])

        pipe_cat.fit(X_tr_cat, y_tr_cat)
        y_pred_val = pipe_cat.predict(X_val_cat)
        cat_rmse   = rmse(y_val_cat, y_pred_val)

        coef       = pipe_cat.named_steps["regressor"].coef_
        intercept  = pipe_cat.named_steps["regressor"].intercept_

        models_dict[cat] = {"model": pipe_cat,
                            "rmse" : cat_rmse,
                            "coef" : coef,
                            "intercept": intercept}

        summary_rows.append(dict(Genre=cat,
                                 RMSE = cat_rmse,
                                 Intercept = intercept,
                                 **{f"w_{f}":c for f,c in zip(num_features,coef)}))

    summary_df = pd.DataFrame(summary_rows)
    return models_dict, summary_df, (X_val, y_val)        # keep the global val-fold for later



class MultiGenreLinearModel(BaseEstimator, RegressorMixin):
    """
    One LinearRegression per category.
    The correct sub-model is picked automatically at predict time.
    """
    def __init__(self, categorical_feature="Genre", numerical_features=None,
                 test_size=0.20, random_state=42):
        self.categorical_feature = categorical_feature
        self.numerical_features  = numerical_features
        self.test_size           = test_size
        self.random_state        = random_state
        self.models_             = {}

    # ------------------------------------------------------------------ FIT
    def fit(self, X, y):
        self.numerical_features_ = (self.numerical_features
                                    if self.numerical_features is not None
                                    else [c for c in X.columns
                                          if c != self.categorical_feature])

        (self.models_,
         self.summary_,
         (self.X_val_, self.y_val_)) = train_per_category_models(
                                           X, y,
                                           cat_feature = self.categorical_feature,
                                           num_features= self.numerical_features_,
                                           test_size   = self.test_size,
                                           random_state= self.random_state)
        return self

    # ---------------------------------------------------------------- PREDICT
    def predict(self, X):
        # container that uses the SAME index labels as X
        preds = pd.Series(index=X.index, dtype=float)

        # iterate over groups of identical category
        for cat, idx in X.groupby(self.categorical_feature).groups.items():
            if cat not in self.models_:
                raise ValueError(f"Category '{cat}' not seen during training.")
            sub_X   = X.loc[idx, :]
            sub_pred = self.models_[cat]["model"].predict(sub_X)
            preds.loc[idx] = sub_pred    # safe because both are label-aligned

        return preds.values             # or simply `return preds`

    # ---------------------------------------------------------- convenience
    def per_category_rmse(self):
        return {k: v["rmse"] for k, v in self.models_.items()}

    def per_category_coefficients(self):
        return {k: dict(intercept = v["intercept"],
                        **{f: w for f, w in zip(self.numerical_features_, v["coef"])})
                for k, v in self.models_.items()}


cat_col  = "Genre"
num_cols = ["Guest_Popularity_percentage", "Host_Popularity_percentage"]

# Fit the global "label-encoder" pipeline (goal 1)
label_pipe = create_label_pipeline([cat_col], num_cols)
label_pipe.fit(pro_train, y)

# ---------- per-category models  (goals 2 & 3) -----------------------------
models_dict, summary_df, (X_val, y_val) = train_per_category_models(
                                            pro_train, y,
                                            cat_feature = cat_col,
                                            num_features= num_cols)

print("RMSE & coefficients per model")
display(summary_df)          # ← shows rmse, weights & intercepts

# ---------- unified multi-model predictor (goal 4) -------------------------
multi_model = MultiGenreLinearModel(categorical_feature = cat_col,
                                    numerical_features  = num_cols)

multi_model.fit(pro_train, y)

# overall validation performance (using the stored val split)
y_val_pred  = multi_model.predict(X_val)
overall_rmse= rmse(y_val, y_val_pred)

print(f"\nOVERALL validation RMSE (all categories together) : {overall_rmse:0.2f}")
print("\nPer-category RMSE :", multi_model.per_category_rmse())
print("\nPer-category coefficients :", multi_model.per_category_coefficients())


y_test_pred  = multi_model.predict(pro_test)


submission['Listening_Time_minutes'] = y_test_pred


import matplotlib.pyplot as plt
plt.boxplot(submission['Listening_Time_minutes'])


from sklearn.linear_model import BayesianRidge


def train_per_category_bayesian_models(X, y, cat_feature, num_features,
                              test_size=0.20, random_state=42):
    """
    1) global split
    2) inside the training fold: one BayesianRidge Regression for each category
    3) return dict with {category : (pipeline, rmse_val)} and a summary df
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size   = test_size,
                                                      random_state= random_state,
                                                      stratify    = X[cat_feature])

    models_dict = {}
    summary_rows = []

    for cat in sorted(X_train[cat_feature].unique()):
        # ­­­­­­­­­Select only this category
        idx_tr  = X_train[cat_feature] == cat
        idx_val = X_val[cat_feature]   == cat
        
        X_tr_cat,  y_tr_cat  = X_train.loc[idx_tr,  :], y_train.loc[idx_tr]
        X_val_cat, y_val_cat = X_val.loc[idx_val, :], y_val.loc[idx_val]
        
        # pipeline without the categorical column (it's constant now)
        preproc_cat = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )

        pipe_cat = Pipeline([("preprocessor", preproc_cat),
                             ("regressor",   BayesianRidge())])

        pipe_cat.fit(X_tr_cat, y_tr_cat)
        y_pred_val = pipe_cat.predict(X_val_cat)
        cat_rmse   = rmse(y_val_cat, y_pred_val)

        coef       = pipe_cat.named_steps["regressor"].coef_
        intercept  = pipe_cat.named_steps["regressor"].intercept_

        models_dict[cat] = {"model": pipe_cat,
                            "rmse" : cat_rmse,
                            "coef" : coef,
                            "intercept": intercept}

        summary_rows.append(dict(Genre=cat,
                                 RMSE = cat_rmse,
                                 Intercept = intercept,
                                 **{f"w_{f}":c for f,c in zip(num_features,coef)}))

    summary_df = pd.DataFrame(summary_rows)
    return models_dict, summary_df, (X_val, y_val)        # keep the global val-fold for later


class MultiGenreBayesianModel(MultiGenreLinearModel):
    def fit(self, X, y):
        self.numerical_features_ = (
            self.numerical_features
            if self.numerical_features is not None
            else [c for c in X.columns if c != self.categorical_feature]
        )

        (self.models_,
         self.summary_,
         (self.X_val_, self.y_val_)) = train_per_category_bayesian_models(
            X, y,
            cat_feature=self.categorical_feature,
            num_features=self.numerical_features_,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return self


cat_col  = "Genre"
num_cols = ["Guest_Popularity_percentage", "Host_Popularity_percentage"]

bayesian_models_dict, bayesian_summary_df, (X_val, y_val) = train_per_category_bayesian_models(
                                            pro_train, y,
                                            cat_feature = cat_col,
                                            num_features= num_cols)

print("RMSE & coefficients per model")
display(summary_df)          # ← shows rmse, weights & intercepts


multi_bayesian_model = MultiGenreBayesianModel(categorical_feature = cat_col,
                                    numerical_features  = num_cols)

multi_bayesian_model.fit(pro_train, y)

# overall validation performance (using the stored val split)
y_val_pred  = multi_bayesian_model.predict(X_val)
overall_rmse= rmse(y_val, y_val_pred)

print(f"\nOVERALL validation RMSE (all categories together) : {overall_rmse:0.2f}")
print("\nPer-category RMSE :", multi_bayesian_model.per_category_rmse())
print("\nPer-category coefficients :", multi_bayesian_model.per_category_coefficients())


y_test_pred_bayesian  = multi_bayesian_model.predict(pro_test)
submission['Listening_Time_minutes'] = y_test_pred_bayesian

plt.boxplot(submission['Listening_Time_minutes'])
submission.to_csv('multi_model_linear_model.csv', index=False)


from sklearn.ensemble import RandomForestRegressor


def num_preprocessor():
    return make_pipeline(
        SimpleImputer(strategy='mean'),  # Handles NaNs
        StandardScaler()                 # Optional: scales features
    )
def train_per_category_random_forest_models(X, y, cat_feature, num_features,
                                            test_size=0.20, random_state=42, 
                                            n_estimators=100, max_depth=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=test_size,
                                                      random_state=random_state,
                                                      stratify=X[cat_feature])

    models_dict = {}
    summary_rows = []

    for cat in sorted(X_train[cat_feature].unique()):
        idx_tr  = X_train[cat_feature] == cat
        idx_val = X_val[cat_feature]   == cat

        X_tr_cat,  y_tr_cat  = X_train.loc[idx_tr, :], y_train.loc[idx_tr]
        X_val_cat, y_val_cat = X_val.loc[idx_val, :], y_val.loc[idx_val]

        preproc_cat = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )

        pipe_cat = Pipeline([
            ("preprocessor", preproc_cat),
            ("regressor", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state))
        ])

        pipe_cat.fit(X_tr_cat, y_tr_cat)
        y_pred_val = pipe_cat.predict(X_val_cat)
        cat_rmse = rmse(y_val_cat, y_pred_val)

        models_dict[cat] = {"model": pipe_cat, "rmse": cat_rmse}
        summary_rows.append(dict(Genre=cat, RMSE=cat_rmse))

    summary_df = pd.DataFrame(summary_rows)
    return models_dict, summary_df, (X_val, y_val)


from sklearn.base import BaseEstimator, RegressorMixin

class MultiGenreRandomForestModel(BaseEstimator, RegressorMixin):
    """
    One RandomForestRegressor per category.
    The correct sub-model is picked automatically at predict time.
    """
    def __init__(self, categorical_feature="Genre", numerical_features=None,
                 test_size=0.20, random_state=42, n_estimators=100, max_depth=None):
        self.categorical_feature = categorical_feature
        self.numerical_features  = numerical_features
        self.test_size           = test_size
        self.random_state        = random_state
        self.n_estimators        = n_estimators
        self.max_depth           = max_depth
        self.models_             = {}

    # ------------------------------------------------------------------ FIT
    def fit(self, X, y):
        self.numerical_features_ = (self.numerical_features
                                    if self.numerical_features is not None
                                    else [c for c in X.columns
                                          if c != self.categorical_feature])

        (self.models_,
         self.summary_,
         (self.X_val_, self.y_val_)) = train_per_category_random_forest_models(
                                           X, y,
                                           cat_feature = self.categorical_feature,
                                           num_features= self.numerical_features_,
                                           test_size   = self.test_size,
                                           random_state= self.random_state,
                                           n_estimators= self.n_estimators,
                                           max_depth   = self.max_depth)
        return self

    # ---------------------------------------------------------------- PREDICT
    def predict(self, X):
        # container that uses the SAME index labels as X
        preds = pd.Series(index=X.index, dtype=float)

        # iterate over groups of identical category
        for cat, idx in X.groupby(self.categorical_feature).groups.items():
            if cat not in self.models_:
                raise ValueError(f"Category '{cat}' not seen during training.")
            sub_X   = X.loc[idx, :]
            sub_pred = self.models_[cat]["model"].predict(sub_X)
            preds.loc[idx] = sub_pred    # safe because both are label-aligned

        return preds.values             # or simply `return preds`

    # ---------------------------------------------------------- convenience
    def per_category_rmse(self):
        return {k: v["rmse"] for k, v in self.models_.items()}


# Create an instance of the model
model_random_forest = MultiGenreRandomForestModel(categorical_feature=cat_col, 
                                     numerical_features=num_cols, 
                                     test_size=0.2, 
                                     random_state=42, 
                                     n_estimators=100, 
                                     max_depth=5)

# Fit the model
model_random_forest.fit(pro_train, y)

# overall validation performance (using the stored val split)
y_val_pred_random  = model_random_forest.predict(X_val)
overall_rmse= rmse(y_val, y_val_pred_random)

print(f"\nOVERALL validation RMSE (all categories together) : {overall_rmse:0.2f}")
print("\nPer-category RMSE :", model_random_forest.per_category_rmse())


y_test_pred_forest  = model_random_forest.predict(pro_test)
submission['Listening_Time_minutes'] = y_test_pred_forest
plt.boxplot(submission['Listening_Time_minutes'])
submission.to_csv('multi_model_linear_model.csv', index=False)


import xgboost as xgb

def train_per_category_xgboost_models(X, y, cat_feature, num_features,
                                      test_size=0.20, random_state=42, 
                                      n_estimators=100, max_depth=6, 
                                      learning_rate=0.1, gamma=0):
    """
    1) global split
    2) inside the training fold: one XGBoost Regression for each category
    3) return dict with {category : (pipeline, rmse_val)} and a summary df
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size   = test_size,
                                                      random_state= random_state,
                                                      stratify    = X[cat_feature])

    models_dict = {}
    summary_rows = []

    for cat in sorted(X_train[cat_feature].unique()):
        # ­­­­­­­­­Select only this category
        idx_tr  = X_train[cat_feature] == cat
        idx_val = X_val[cat_feature]   == cat
        
        X_tr_cat,  y_tr_cat  = X_train.loc[idx_tr,  :], y_train.loc[idx_tr]
        X_val_cat, y_val_cat = X_val.loc[idx_val, :], y_val.loc[idx_val]
        
        # pipeline without the categorical column (it's constant now)
        preproc_cat = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )

        pipe_cat = Pipeline([("preprocessor", preproc_cat),
                             ("regressor",   xgb.XGBRegressor(n_estimators=n_estimators,
                                                               max_depth=max_depth,
                                                               learning_rate=learning_rate,
                                                               gamma=gamma,
                                                               n_jobs=-1,
                                                               random_state=random_state))])

        pipe_cat.fit(X_tr_cat, y_tr_cat)
        y_pred_val = pipe_cat.predict(X_val_cat)
        cat_rmse   = rmse(y_val_cat, y_pred_val)

        models_dict[cat] = {"model": pipe_cat,
                            "rmse" : cat_rmse}

        summary_rows.append(dict(Genre=cat,
                                 RMSE = cat_rmse))

    summary_df = pd.DataFrame(summary_rows)
    return models_dict, summary_df, (X_val, y_val)        # keep the global val-fold for later


class MultiGenreXGBoostModel(BaseEstimator, RegressorMixin):
    """
    One XGBoost Regressor per category.
    The correct sub-model is picked automatically at predict time.
    """
    def __init__(self, categorical_feature="Genre", numerical_features=None,
                 test_size=0.20, random_state=42, n_estimators=100, 
                 max_depth=6, learning_rate=0.1, gamma=0):
        self.categorical_feature = categorical_feature
        self.numerical_features  = numerical_features
        self.test_size           = test_size
        self.random_state        = random_state
        self.n_estimators        = n_estimators
        self.max_depth           = max_depth
        self.learning_rate       = learning_rate
        self.gamma               = gamma
        self.models_             = {}

    # ------------------------------------------------------------------ FIT
    def fit(self, X, y):
        self.numerical_features_ = (self.numerical_features
                                    if self.numerical_features is not None
                                    else [c for c in X.columns
                                          if c != self.categorical_feature])

        (self.models_,
         self.summary_,
         (self.X_val_, self.y_val_)) = train_per_category_xgboost_models(
                                           X, y,
                                           cat_feature = self.categorical_feature,
                                           num_features= self.numerical_features_,
                                           test_size   = self.test_size,
                                           random_state= self.random_state,
                                           n_estimators= self.n_estimators,
                                           max_depth   = self.max_depth,
                                           learning_rate= self.learning_rate,
                                           gamma       = self.gamma)
        return self

    # ---------------------------------------------------------------- PREDICT
    def predict(self, X):
        # container that uses the SAME index labels as X
        preds = pd.Series(index=X.index, dtype=float)

        # iterate over groups of identical category
        for cat, idx in X.groupby(self.categorical_feature).groups.items():
            if cat not in self.models_:
                raise ValueError(f"Category '{cat}' not seen during training.")
            sub_X   = X.loc[idx, :]
            sub_pred = self.models_[cat]["model"].predict(sub_X)
            preds.loc[idx] = sub_pred    # safe because both are label-aligned

        return preds.values             # or simply `return preds`

    # ---------------------------------------------------------- convenience
    def per_category_rmse(self):
        return {k: v["rmse"] for k, v in self.models_.items()}


model_xgboost = MultiGenreXGBoostModel(categorical_feature=cat_col, 
                                      numerical_features=num_cols, 
                                      test_size=0.2, 
                                      random_state=42, 
                                      n_estimators=100, 
                                      max_depth=6, 
                                      learning_rate=0.1, 
                                      gamma=0)

model_xgboost.fit(pro_train, y)
y_test_pred_xgboost = model_xgboost.predict(pro_test)
submission['Listening_Time_minutes'] = y_test_pred_xgboost
plt.boxplot(submission['Listening_Time_minutes'])



print("\nPer-category RMSE :", model_xgboost.per_category_rmse()) 


import lightgbm as lgb

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def num_preprocessor():
    # Define your numerical preprocessor here
    # For simplicity, I'm using a simple StandardScaler
    from sklearn.preprocessing import StandardScaler
    return StandardScaler()

def train_per_category_lightgbm_models(X, y, cat_feature, num_features,
                                       test_size=0.20, random_state=42, 
                                       n_estimators=100, max_depth=-1, 
                                       learning_rate=0.1, num_leaves=31):
    """
    1) global split
    2) inside the training fold: one LightGBM Regression for each category
    3) return dict with {category : (pipeline, rmse_val)} and a summary df
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size   = test_size,
                                                      random_state= random_state,
                                                      stratify    = X[cat_feature])

    models_dict = {}
    summary_rows = []

    for cat in sorted(X_train[cat_feature].unique()):
        # ­­­­­­­­­Select only this category
        idx_tr  = X_train[cat_feature] == cat
        idx_val = X_val[cat_feature]   == cat
        
        X_tr_cat,  y_tr_cat  = X_train.loc[idx_tr,  :], y_train.loc[idx_tr]
        X_val_cat, y_val_cat = X_val.loc[idx_val, :], y_val.loc[idx_val]
        
        # pipeline without the categorical column (it's constant now)
        preproc_cat = ColumnTransformer(
            [("num", num_preprocessor(), num_features)],
            remainder="drop"
        )

        pipe_cat = Pipeline([("preprocessor", preproc_cat),
                             ("regressor",   lgb.LGBMRegressor(n_estimators=n_estimators,
                                                               max_depth=max_depth,
                                                               learning_rate=learning_rate,
                                                               num_leaves=num_leaves,
                                                               random_state=random_state))])

        pipe_cat.fit(X_tr_cat, y_tr_cat)
        y_pred_val = pipe_cat.predict(X_val_cat)
        cat_rmse   = rmse(y_val_cat, y_pred_val)

        models_dict[cat] = {"model": pipe_cat,
                            "rmse" : cat_rmse}

        summary_rows.append(dict(Genre=cat,
                                 RMSE = cat_rmse))

    summary_df = pd.DataFrame(summary_rows)
    return models_dict, summary_df, (X_val, y_val)        # keep the global val-fold for later


class MultiGenreLightGBMModel(BaseEstimator, RegressorMixin):
    """
    One LightGBM Regressor per category.
    The correct sub-model is picked automatically at predict time.
    """
    def __init__(self, categorical_feature="Genre", numerical_features=None,
                 test_size=0.20, random_state=42, n_estimators=100, 
                 max_depth=-1, learning_rate=0.1, num_leaves=31):
        self.categorical_feature = categorical_feature
        self.numerical_features  = numerical_features
        self.test_size           = test_size
        self.random_state        = random_state
        self.n_estimators        = n_estimators
        self.max_depth           = max_depth
        self.learning_rate       = learning_rate
        self.num_leaves          = num_leaves
        self.models_             = {}

    # ------------------------------------------------------------------ FIT
    def fit(self, X, y):
        self.numerical_features_ = (self.numerical_features
                                    if self.numerical_features is not None
                                    else [c for c in X.columns
                                          if c != self.categorical_feature])

        (self.models_,
         self.summary_,
         (self.X_val_, self.y_val_)) = train_per_category_lightgbm_models(
                                           X, y,
                                           cat_feature = self.categorical_feature,
                                           num_features= self.numerical_features_,
                                           test_size   = self.test_size,
                                           random_state= self.random_state,
                                           n_estimators= self.n_estimators,
                                           max_depth   = self.max_depth,
                                           learning_rate= self.learning_rate,
                                           num_leaves  = self.num_leaves)
        return self

    # ---------------------------------------------------------------- PREDICT
    def predict(self, X):
        # container that uses the SAME index labels as X
        preds = pd.Series(index=X.index, dtype=float)

        # iterate over groups of identical category
        for cat, idx in X.groupby(self.categorical_feature).groups.items():
            if cat not in self.models_:
                raise ValueError(f"Category '{cat}' not seen during training.")
            sub_X   = X.loc[idx, :]
            sub_pred = self.models_[cat]["model"].predict(sub_X)
            preds.loc[idx] = sub_pred    # safe because both are label-aligned

        return preds.values             # or simply `return preds`

    # ---------------------------------------------------------- convenience
    def per_category_rmse(self):
        return {k: v["rmse"] for k, v in self.models_.items()}


model_lightgbm = MultiGenreLightGBMModel(categorical_feature=cat_col, 
                                         numerical_features=num_cols, 
                                         test_size=0.2, 
                                         random_state=42, 
                                         n_estimators=100, 
                                         max_depth=-1, 
                                         learning_rate=0.1, 
                                         num_leaves=31)

model_lightgbm.fit(pro_train, y)
y_test_pred_lightgbm = model_lightgbm.predict(pro_test)
submission['Listening_Time_minutes'] = y_test_pred_lightgbm
plt.boxplot(submission['Listening_Time_minutes'])


from sklearn.base import clone

class MultiGenrePickBestRMSEModel(BaseEstimator, RegressorMixin):
    """
    For each category, compares multiple base regressors and picks the one with lowest validation RMSE.
    At predict time, uses the per-category winner.
    """
    def __init__(self, categorical_feature='Genre', numerical_feature=None, base_models=None, test_size=0.2, random_state=42):
        self.categorical_feature = categorical_feature
        self.numerical_feature = numerical_feature  # <- Needed for DataFrame reconstruction
        self.base_models = base_models if base_models is not None else []
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        if len(self.base_models) == 0:
            raise ValueError("You must provide at least one base model.")

        self.models_ = {}      # {category: fitted estimator}
        self.rmse_ = {}        # {category: RMSE for chosen estimator}
        self.winners_ = {}     # {category: name/class of chosen estimator}

        # Get all categories
        categories = X[self.categorical_feature].unique()
        for cat in categories:
            mask = X[self.categorical_feature] == cat
            X_cat = X[mask]
            y_cat = y[mask] if hasattr(y, "loc") else np.asarray(y)[mask]

            # Split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_cat, y_cat, test_size=self.test_size, random_state=self.random_state
            )

            # Fit all models, pick best by RMSE on validation set
            best_rmse = np.inf
            best_model = None
            best_name = None
            for model in self.base_models:
                model_instance = clone(model)
                model_instance.fit(X_train, y_train)
                preds = model_instance.predict(X_val)
                rmse = np.sqrt(np.mean((preds - y_val) ** 2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_instance
                    best_name = type(model_instance).__name__

            self.models_[cat] = best_model
            self.rmse_[cat] = best_rmse
            self.winners_[cat] = best_name

        return self

    def predict(self, X):
        # Handle unseen categories with np.nan, can change to fallback if you wish
        preds = pd.Series(index=X.index, dtype=float)
        for cat, idx in X.groupby(self.categorical_feature).groups.items():
            if cat not in self.models_:
                preds.loc[idx] = np.nan # category not seen at train time
                continue
            sub_X = X.loc[idx]
            preds.loc[idx] = self.models_[cat].predict(sub_X)
        return preds.values

    def per_category_rmse(self):
        return dict(self.rmse_)
    
    def per_category_winner(self):
        return dict(self.winners_)


preprocessor = ColumnTransformer(
    [('num', SimpleImputer(strategy='mean'), num_cols)],
    remainder='passthrough'
)

combo_pipeline = Pipeline([
    ('imputer', preprocessor),
    ('combo', MultiGenrePickBestRMSEModel(
        categorical_feature='Genre', 
        numerical_feature=num_cols,
        base_models=[
            MultiGenreLightGBMModel(
                categorical_feature=cat_col,
                numerical_features=num_cols,
                test_size=0.2,
                random_state=42,
                n_estimators=100,
                max_depth=-1,
                learning_rate=0.1,
                num_leaves=31
            ),
            MultiGenreXGBoostModel(
                categorical_feature=cat_col,
                numerical_features=num_cols,
                test_size=0.2,
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                gamma=0
            ),
        ],
        test_size=0.2,
        random_state=42
    ))
])

combo_pipeline.fit(pro_train, y)
preds = combo_pipeline.predict(pro_test)

# Access per-category diagnostics from the fitted combo inside the pipeline
combo = combo_pipeline.named_steps['combo']
print(combo.per_category_rmse())
print(combo.per_category_winner())


submission['Listening_Time_minutes'] = preds
plt.boxplot(submission['Listening_Time_minutes'])
submission.to_csv('multi_model_xgcombine.csv', index=False)




