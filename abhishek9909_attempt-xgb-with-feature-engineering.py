from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


original_df = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


target_feature = list(train_df.columns)[-1]
numerical_features = [x for x in train_df.describe().columns if x != "id"]
categorical_features = [x for x in train_df.columns if x not in numerical_features and x != "id" and x != target_feature]
numerical_features, categorical_features


train_df = train_df.drop(columns = ["id"])
train_df = pd.concat([train_df, original_df], ignore_index = True)
train_df.tail(10)


## transformation of the categorical features.
oe = OrdinalEncoder()
train_df[categorical_features] = oe.fit_transform(train_df[categorical_features])
test_df[categorical_features] = oe.transform(test_df[categorical_features])
for col in categorical_features:
    train_df[col] = train_df[col].astype("category")
    test_df[col] = test_df[col].astype("category")

## transformation of the label.
le = LabelEncoder()
train_df[target_feature] = le.fit_transform(train_df[target_feature])
train_df.head()


full_y, full_x = train_df[target_feature], train_df.drop(columns = [target_feature])
train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size = 0.1, random_state = 42, stratify = full_y) ## use a lower test size fraction to get maximal results.


## settings from the designated notebook.
model_xg_multi = XGBClassifier(
    objective = "multi:softprob",
    num_class = len(np.unique(train_y)),
    n_estimators = 2000,
    learning_rate = 0.05,
    max_depth = 12,
    colsample_bytree = 0.467,
    early_stopping_rounds = 100,
    reg_alpha = 2.7,
    reg_lambda = 1.4,
    gamma = 0.26,
    enable_categorical = True,
    tree_method = 'hist',
    max_delta_step = 4,
    subsample = 0.86,
    random_state = 13,
    device = "cuda"
)
model_xg_multi.fit(train_x, train_y, eval_set = [(test_x, test_y)], verbose = 0)


def predict_and_score_xgboost_multi(model):
    y_pred_probs = model.predict_proba(test_x)
    top3_probs = np.argsort(y_pred_probs, axis = 1)[:, -3:][:, ::-1]
    get_best_and_full_accuracy_xgboost(test_y, top3_probs)

    return top3_probs

def _get_score(actual, predicted):
    score = 0.0
    hits = 0
    seen = set()
    for i, pred in enumerate(predicted):
        if pred == np.int64(actual) and pred not in seen:
            hits += 1
            score += hits / (i + 1.0)
            seen.add(pred)
    
    return score ## since actual is ONE entity.


def get_best_and_full_accuracy_xgboost(ptest_y, topk_probs):
    test_yl = ptest_y.tolist()
    first_acc_l = [x for idx, x in enumerate(test_yl) if np.int64(x) == topk_probs[idx][0]]
    score_accl_l = [_get_score(x ,topk_probs[idx]) for idx, x in enumerate(test_yl)]

    print(f"First accuracy: {(len(first_acc_l) / len(test_yl)):.2f}") ## only based on the highest class.
    print(f"Score accuracy: {(np.mean(score_accl_l)):.2f}") ## based on the function defined earlier.

top3_xg_multi = predict_and_score_xgboost_multi(model_xg_multi)


from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns

importances = model_xg_multi.feature_importances_
feature_names = model_xg_multi.get_booster().feature_names

df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def create_balanced_splits():
    train_splits = {}
    for t in range(len(le.classes_)):
        pos_idx = train_y[train_y == t].index
        pos_samples = train_x.loc[pos_idx]
        pos_labels = train_y.loc[pos_idx]

        n_pos = len(pos_idx)
        neg_idx = train_y[train_y != t].index
        neg_x = train_x.loc[neg_idx]
        neg_y = train_y.loc[neg_idx]
        
        neg_x_sampled, _, neg_y_sampled, _ = train_test_split(neg_x, neg_y, train_size = n_pos, stratify = neg_y, random_state = 42)
        binary_x = pd.concat([pos_samples, neg_x_sampled])
        binary_y = pd.Series([1] * n_pos + [0] * n_pos, index = binary_x.index)

        ## shuffle.
        binary_x = binary_x.sample(frac = 1, random_state = 42)
        binary_y = binary_y.loc[binary_x.index]

        train_splits[t] = (binary_x, binary_y)
    
    return train_splits

balanced_splits = create_balanced_splits()

def fit_logistic_regression_l1_reg(X_train, y_train):
    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)

    # Logistic Regression with L1 penalty (Lasso)
    logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)
    logreg_l1.fit(x_train_scaled, y_train)

    # Get coefficients
    coef = logreg_l1.coef_.ravel()

    # Select features with non-zero coefficients
    return coef


coeff_dict = {}
for target_name, (ptrain_x, ptrain_y) in balanced_splits.items():
    coeffs = fit_logistic_regression_l1_reg(ptrain_x, ptrain_y)
    cols = ptrain_x.columns
    zipped_cols = list(zip(coeffs, cols))
    zipped_cols = list(sorted(zipped_cols, key = lambda x : -abs(x[0])))

    ## print the top -3
    print(f"Top 3 most important features for {le.classes_[target_name]}: ")
    top3_cols = zipped_cols[:3]
    print(top3_cols)

    for coeff, col in zipped_cols:
        if col not in coeff_dict.keys():
            coeff_dict[col] = 0
        coeff_dict[col] += abs(coeff)

for feature, importance in coeff_dict.items():
    print("Feature: ", feature)
    print("Importance: ", importance)


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def univariate_feature_plots(feature_name):
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature_name], kde = True, bins = 30)
    plt.title(f"Histogram of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x = train_df[feature_name])
    plt.title(f"Box plot of {feature_name}")

    plt.tight_layout()
    plt.show()

    print(f"Statistics: for {feature_name}", end = '\n')
    print(f"Skewness: {train_df[feature_name].skew():.2f}")
    print(f"Number of missing values: {train_df[train_df[feature_name].isnull()].shape[0]}")


for feature in numerical_features:
    univariate_feature_plots(feature)


temp_feature = numerical_features[0]
humidity_feature = numerical_features[1]
train_df['Temperature Binned'] = pd.qcut(train_df[temp_feature], q = 4, labels = ['Low', 'Medium', 'High', 'Very High'])
train_df['Humidity Binned'] = pd.qcut(train_df[humidity_feature], q = 4, labels = ['Low', 'Medium', 'High', 'Very High'])


scaler = StandardScaler()
normalized = scaler.fit_transform(train_df[[temp_feature, humidity_feature]])
train_df['Temperature Humidity Index'] = normalized[:, 0] + normalized[:, 1]


univariate_feature_plots('Temperature Humidity Index')


moisture_feature = numerical_features[2]
normalized = scaler.fit_transform(train_df[[humidity_feature, moisture_feature]])
epsilon = 1e-6
train_df['Humidity Moisture Ratio'] = train_df[humidity_feature] / (train_df[moisture_feature] + epsilon)


univariate_feature_plots('Humidity Moisture Ratio')


import seaborn as sns
import matplotlib.pyplot as plt

for feature in numerical_features[3:]:
    plt.figure(figsize = (8, 4))  
    sns.boxplot(x = categorical_features[1], y = feature, data = train_df, )  
    plt.xticks(rotation = 90)
    plt.title(f"Boxplot of {feature} by Crop Type")
    plt.tight_layout()
    plt.show()


from sklearn.preprocessing import MinMaxScaler

m_scaler = MinMaxScaler()
normalized_mineral_vals = m_scaler.fit_transform(train_df[numerical_features[3:]])
nmv = normalized_mineral_vals ## simple name.
## adding the normalized columns into a new column.
train_df['Summarized Mineral Values'] = nmv[:, 0] + nmv[:, 1] + nmv[:, 2]
train_df.head()


univariate_feature_plots('Summarized Mineral Values')


smv = 'Summarized Mineral Values'
plt.figure(figsize = (8, 4))  
sns.boxplot(x = categorical_features[1], y = smv, data = train_df)  
plt.xticks(rotation = 90)
plt.title(f"Boxplot of {smv} by Crop Type")
plt.tight_layout()
plt.show()


def clip_numerical_values(numerical_feature):
    q1 = train_df[numerical_feature].quantile(0.25)
    q3 = train_df[numerical_feature].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    train_df[f"{numerical_feature}_clipped"] = train_df[numerical_feature].clip(lower = lower_bound, upper = upper_bound, )


train_df.columns


tbc_features = list(train_df.columns)[-2:]
tbc_features


for feature in tbc_features:
    clip_numerical_values(feature)


## drop certain columns.
nc_features = list(train_df.columns)[:2]
nc_features


trimmed_train_df = train_df.drop(columns = [*nc_features, *tbc_features])
trimmed_train_df.head()


numerical_features = [x for x in trimmed_train_df.describe().columns if x != "id"]
categorical_features = [x for x in trimmed_train_df.columns if x not in numerical_features and x != "id" and x != target_feature]
numerical_features, categorical_features


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

oe = OrdinalEncoder()
trimmed_train_df[categorical_features] = oe.fit_transform(trimmed_train_df[categorical_features])
for col in categorical_features:
    trimmed_train_df[col] = trimmed_train_df[col].astype("category")

## transformation of the label.
le = LabelEncoder()
trimmed_train_df[target_feature] = le.fit_transform(trimmed_train_df[target_feature])
trimmed_train_df.head()


### reduce each numerical feature value to 1 decimal point.
for feature in numerical_features:
    trimmed_train_df[feature] = trimmed_train_df[feature].round(1)


full_y, full_x = trimmed_train_df[target_feature], trimmed_train_df.drop(columns = [target_feature])
train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, test_size = 0.1, random_state = 42, stratify = full_y) ## use a lower test size fraction to get maximal results.


## settings from the designated notebook.
from xgboost import XGBClassifier
model_xg_multi_mf = XGBClassifier(
    objective = "multi:softprob",
    num_class = len(np.unique(train_y)),
    n_estimators = 2000,
    learning_rate = 0.05,
    max_depth = 12,
    colsmaple_bytree = 0.467,
    early_stopping_rounds = 100,
    reg_alpha = 2.7,
    reg_lambda = 1.4,
    gamma = 0.26,
    enable_categorical = True,
    tree_method = 'hist',
    max_delta_step = 4,
    subsample = 0.86,
    random_state = 13,
    device = "cuda"
)
model_xg_multi_mf.fit(train_x, train_y, eval_set = [(test_x, test_y)], verbose = 0)


top3_xg_multi_mf = predict_and_score_xgboost_multi(model_xg_multi_mf)


importances = model_xg_multi_mf.feature_importances_
feature_names = model_xg_multi_mf.get_booster().feature_names

df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df.head(20), x='Importance', y='Feature', palette='viridis')
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()


from scipy.stats import uniform, reciprocal, randint

def sample_param(distribution):
    low, high = distribution[:2]
    
    if len(distribution) == 3:
        kind = distribution[2]
        if kind == 'log-uniform':
            return reciprocal(low, high).rvs()
        elif kind == 'integer':
            return randint(low, high + 1).rvs()
    
    # Default: uniform
    return uniform(loc=low, scale=high - low).rvs()


original_df = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
target_feature = list(train_df.columns)[-1]
numerical_features = [x for x in train_df.describe().columns if x != "id"]
categorical_features = [x for x in train_df.columns if x not in numerical_features and x != "id" and x != target_feature]
numerical_features, categorical_features


train_df = train_df.drop(columns = ["id"])
train_df = pd.concat([train_df, original_df], ignore_index = True)


oe = OrdinalEncoder()
train_df[categorical_features] = oe.fit_transform(train_df[categorical_features])
for col in categorical_features:
    train_df[col] = train_df[col].astype("category")
le = LabelEncoder()
train_df[target_feature] = le.fit_transform(train_df[target_feature])
train_df.head()


full_y, full_x = train_df[target_feature], train_df.drop(columns = [target_feature])


full_x.columns


temp_feature, humid_feature, moist_feature = full_x.columns[:3]
mineral_features = full_x.columns[-3:]
mineral_features


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, MinMaxScaler
import xgboost as xgb

class Dropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.empty((X.shape[0], 0))

class TempHumidityIndexAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Assumes X has two columns: temperature and humidity
        return (X[:, 0] + X[:, 1]).reshape(-1, 1)

class HumidityMoistureRatioAdder(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X has shape (n_samples, 2): [humidity, moisture]
        ratio = X[:, 0] / (X[:, 1] + self.epsilon)
        return ratio.reshape(-1, 1)

class SummarizedMineralAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        return self
    def transform(self, X):
        nmv = self.scaler.transform(X)
        return np.sum(nmv, axis=1).reshape(-1, 1)

class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower = q1 - 1.5 * iqr
        self.upper = q3 + 1.5 * iqr
        return self
    def transform(self, X):
        return np.clip(X, self.lower, self.upper)

## binning temperature, humidity.
binning_pipeline = Pipeline(steps=[
    ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile'))
])

## adding temperature, humidity.
temp_hum_index_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('index_adder', TempHumidityIndexAdder())
])

# Humidity / Moisture Ratio
humidity_moisture_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ratio', HumidityMoistureRatioAdder()),
    ('clip', IQRClipper())
])

# Summarized Mineral Values
summarized_mineral_pipeline = Pipeline([
    ('adder', SummarizedMineralAdder()),
    ('clip', IQRClipper())
])

# Dropping not required features.
drop_binned_pipeline = Pipeline([
    ('drop', Dropper())
])

every_feature_preprocessor = ColumnTransformer([
    ('binning', binning_pipeline, [temp_feature, humid_feature]),
    ('temp_hum_index', temp_hum_index_pipeline, [temp_feature, humid_feature]),
    ('humidity_moisture_ratio', humidity_moisture_pipeline, [humid_feature, temp_feature]),
    ('summarized_minerals', summarized_mineral_pipeline, mineral_features),
    ('drop_binned', drop_binned_pipeline, [temp_feature, humid_feature])
])

only_binning_preprocessor = ColumnTransformer([
    ('binning', binning_pipeline, [temp_feature, humid_feature]),
    ('drop_binned', drop_binned_pipeline, [temp_feature, humid_feature])
])

def create_feature_engg_pipeline(base_model, include_all_features = True):
    preprocessor = every_feature_preprocessor if include_all_features else only_binning_preprocessor
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', base_model)
    ])


X, y = full_x, full_y


from scipy.special import expit

def smooth_map3_obj(preds, dtrain):
    """
    preds:  flat array of length N*K (raw scores)
    dtrain: DMatrix containing labels in [0..K-1]
    Returns (grad, hess) both flat arrays of same length
    """
    labels = dtrain.get_label().astype(int)
    N = labels.shape[0]
    K = int(preds.size / N)
    # reshape to (N, K)
    S = preds.reshape(N, K)
    
    grad = np.zeros_like(S)
    hess = np.zeros_like(S)
    
    # for each sample
    for i in range(N):
        y = labels[i]
        scores = S[i]
        # get the three largest *other* scores
        others = np.delete(scores, y)
        top3_idx = np.argsort(others)[-3:][::-1]  # indices in 'others'
        top3_scores = others[top3_idx]
        
        # build the surrogate:  L = -[ σ(s_y − t₁)
        #                         + (1/2) σ(s_y − t₂)
        #                         + (1/3) σ(s_y − t₃) ]
        diffs = scores[y] - top3_scores  # shape (≤3,)
        weights = np.array([1.0, 0.5, 1/3.0])
        σ = expit(diffs)
        
        # gradient w.r.t. s_y:
        #   ∂L/∂s_y = −Σ_k w_k σ'(diffs_k)
        # ∂L/∂t_k = +w_k σ'(diffs_k)
        sigp = σ * (1 - σ)
        # sum over however many we have (<3 if K<4)
        grad_y = -np.sum(weights[:len(sigp)] * sigp)
        hess_y =  np.sum(weights[:len(sigp)] * sigp * (1 - 2*σ))
        
        grad[i, y] = grad_y
        hess[i, y] = hess_y
        
        # distribute to the top-3 others
        # need to map back to original class indices
        idx_map = [j for j in range(K) if j != y]
        for rank, idx_o in enumerate(top3_idx):
            j = idx_map[idx_o]
            grad[i, j] =  weights[rank] * sigp[rank]
            hess[i, j] = weights[rank] * sigp[rank] * (1 - 2*σ[rank])
    
    # flatten back out
    return grad.ravel(), hess.ravel()


from sklearn.base import BaseEstimator, ClassifierMixin

class XGBTrainWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params, num_boost_round=2000, early_stopping_rounds=100, obj=None, eval_metric="mlogloss", verbose_eval=False):
        self.params = params.copy()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.obj = obj
        self.eval_metric = eval_metric
        self.verbose_eval = verbose_eval

    def fit(self, X, y, eval_set=None):
        # Build DMatrices
        # 1) Convert any pandas -> numpy
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_train = X.values
        else:
            X_train = X
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_train = y.values
        else:
            y_train = y

        # 2) Build DMatrix for train
        dtrain = xgb.DMatrix(X_train, label=y_train)

        # 3) Prepare eval list -- skip because too much code :(
        # evals = [(dtrain, "train")]
        # if eval_set is not None:
        #     X_val, y_val = eval_set
        #     # again, force numpy
        #     if isinstance(X_val, (pd.DataFrame, pd.Series)):
        #         X_val = X_val.values
        #     if isinstance(y_val, (pd.Series, pd.DataFrame)):
        #         y_val = y_val.values
        #     evals.append((xgb.DMatrix(X_val, label=y_val), "valid"))

        # train
        self.bst_ = xgb.train(
            self.params,
            dtrain,
            obj=self.obj,
#            evals=evals,
#            evals_result=self._evals_result if hasattr(self, "_evals_result") else {},
#            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )
        return self

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        probs = self.bst_.predict(dtest)
        # xgb.train with multi:softprob outputs flat or (n,K)?
        return probs.reshape(X.shape[0], -1)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# def create_custom_preprocessed_model_pipeline(custom_obj, all_params):
#     xgb_wrapper = XGBTrainWrapper(
#         params=all_params,
#         early_stopping_rounds=100,
#         obj=custom_obj,
#         eval_metric='mlogloss',
#         verbose_eval=50
#     )

#     return Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier',    xgb_wrapper)
#     ])


search_space = {
    'n_estimators':      (1000, 8000, 'integer'),
    'learning_rate':     (1e-2, 1e-1, 'log-uniform'),
    'max_depth':         (8, 16, 'integer'),
    'gamma':             (0, 0.5),
    'subsample':         (0.6, 1.0),
    'colsample_bytree':  (0.4, 0.5),
    'reg_alpha':         (1e-3, 5, 'log-uniform'), 
    'reg_lambda':        (0.5, 5, 'log-uniform'),
    'max_delta_step':    (3, 6, 'integer'),
}


def map3_score(model, x_test, y_test):
    y_pred_probs = model.predict_proba(x_test)
    top3_probs = np.argsort(y_pred_probs, axis = 1)[:, -3:][:, ::-1]
    return get_best_and_full_accuracy_xgboost(y_test, top3_probs)

def _get_score(actual, predicted):
    score = 0.0
    hits = 0
    seen = set()
    for i, pred in enumerate(predicted):
        if pred == np.int64(actual) and pred not in seen:
            hits += 1
            score += hits / (i + 1.0)
            seen.add(pred)
    
    return score ## since actual is ONE entity.


def get_best_and_full_accuracy_xgboost(ptest_y, topk_probs):
    test_yl = ptest_y.tolist()
    first_acc_l = [x for idx, x in enumerate(test_yl) if np.int64(x) == topk_probs[idx][0]]
    score_accl_l = [_get_score(x ,topk_probs[idx]) for idx, x in enumerate(test_yl)]

    return np.mean(score_accl_l)


from sklearn.model_selection import StratifiedKFold as skfold

def process_experiment(total_exps = 5, mode = "vanilla"):
    best_params = None
    best_score = 0
    param_score_dictionary = {}

    kf = skfold(n_splits = 5, shuffle = True, random_state = 42)

    for _ in range(total_exps):
        test_scores = []
        best_rounds = []
        params = {k: sample_param(v) for k, v in search_space.items()}

        for train_index, test_index in kf.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

            # Further split current train set into train and validation 
            X_train_fold, X_val, y_train_fold, y_val = train_test_split(X_train_fold, y_train_fold, test_size=0.2, random_state=42)

            all_params = {
                "objective": "multi:softprob",
                "num_class": len(np.unique(y)),
                "enable_categorical": True,
                "random_state": 13,
                "device": "cuda"
            }
            all_params.update(params)

            if mode != "custom_objective":
                base_model = xgb.XGBClassifier(**all_params)
                if mode == "all_features":
                    pipeline_model = create_feature_engg_pipeline(base_model)   
                    pipeline_model.fit(X_train_fold, y_train_fold)
                elif mode == "best_features":
                    pipeline_model = create_feature_engg_pipeline(base_model, include_all_features = False)
                    pipeline_model.fit(X_train_fold, y_train_fold)
                else:
                    pipeline_model = base_model
                    pipeline_model.fit(X_train_fold, y_train_fold, eval_set = [(X_val, y_val)])
            else:
                other_params = {
                    "early_stopping_rounds": 100,
                    "verbose_eval": 50
                }
                all_params.update(other_params)
                ## pipeline_model = create_custom_preprocessed_model_pipeline(smooth_map3_obj, all_params)
                pipeline_model = XGBTrainWrapper(
                     params=all_params,
                     early_stopping_rounds=100,
                     obj=smooth_map3_obj,
                     eval_metric='mlogloss',
                     verbose_eval=50
                )
                X_val_np, y_val_np = X_val.values, y_val.values
                pipeline_model.fit(X_train_fold, y_train_fold, eval_set = (X_val, y_val))


            test_score = map3_score(pipeline_model, X_test_fold, y_test_fold)
            test_scores.append(test_score)


        ## average score across all folds.
        average_score = np.mean(test_scores)
        if average_score > best_score:
            best_score = average_score
            best_params = params
        param_score_dictionary[str(params)] = average_score

    print(f"best parameters: {best_params}")
    print(f"best score: {best_score}")

    return param_score_dictionary


# vanilla_score_dict = process_experiment()


# best_feature_score_dict = process_experiment(mode = "best_features")


# all_feature_score_dict = process_experiment(mode = "all_features")


# custom_objective_score_dict = process_experiment(mode = "custom_objective")


params = {
    "n_estimators":      2000,
    "learning_rate":     0.05,
    "max_depth":         12,
    "gamma":             0.2,
    "subsample":         0.7,
    "colsample_bytree":  0.45,
    "reg_alpha":         1.7, 
    "reg_lambda":        2.6,
    "max_delta_step":    4,
    "objective": "multi:softprob",
    "num_class": len(np.unique(y)),
    "enable_categorical": True,
    "random_state": 13,
    "device": "cuda",
    
}
model_xg_multi_factor = xgb.XGBClassifier(**params)
model_xg_multi_best_pipeline = create_feature_engg_pipeline(model_xg_multi_factor, include_all_features = False)
model_xg_multi_best_pipeline.fit(X, y)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


y_pred_probs = model_xg_multi_best_pipeline.predict_proba(test_df)
top3_probs = np.argsort(y_pred_probs, axis = 1)[:, -3:][:, ::-1]
top3_probs.shape, top3_probs[0]


top_prob_labels = [[le.classes_[cl] for cl in prob_array] for prob_array in top3_probs]
top_prob_labels[0], len(top_prob_labels)


sample_submission_df[target_feature] = [" ".join(k) for k in top_prob_labels]
sample_submission_df.head()


sample_submission_df.to_csv("submission.csv", index = False)

