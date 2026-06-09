import torch

config = {
    "features": ['day', 
                 'pressure', 
                 'maxtemp', 
                 'temparature', 
                 'mintemp', 
                 'dewpoint', 
                 'humidity', 
                 'cloud', 
                 'sunshine', 
                 'winddirection', 
                 'windspeed'
                 ],
    # Note: in this notebook https://www.kaggle.com/code/hopesb/rain-fall-prediction/notebook they removed:
    # ["mintemp", "temparature", "maxtemp", "winddirection"
    "clustering_variables": ['day', 'temparature', 'sunshine', 'cloud', 'windspeed'],
    "n_clusters": 3,
    "n_lags": 5,
    "lag_columns": ['humidity', 'temparature', 'pressure', 'cloud', 'windspeed', 'dewpoint', 'sunshine'],
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "n_estimators": 100000 #100000

}


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
import seaborn as sns
warnings.filterwarnings("ignore", category=FutureWarning)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def plot_cv_scores(cv_scores, title="Cross-Validation Fold Scores"):
    """
    Plot sorted cross-validation scores from individual folds.
    """
    sorted_scores = sorted(enumerate(cv_scores, 1), key=lambda x: x[1])  # (fold_index, score)
    sorted_fold_indices, sorted_values = zip(*sorted_scores)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_values) + 1), sorted_values, marker='o', linestyle='--', color='b')
    plt.axhline(y=np.mean(cv_scores), color='r', linestyle='-', label=f'Mean: {np.mean(cv_scores):.4f}')
    plt.xlabel('Sorted Fold (by ROC AUC)', fontsize=12)
    plt.ylabel('ROC AUC', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(range(1, len(sorted_values) + 1, max(1, len(sorted_values) // 20)))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def evaluate_pipeline(pipeline, X, y, pipeline_name, n_splits=5, n_repeats=10, random_state=666):
    """
    Evaluate a pipeline using repeated stratified k-fold cross-validation and plot the results.
    
    Parameters:
        pipeline: The sklearn Pipeline to evaluate.
        X: Feature DataFrame.
        y: Target array or Series.
        pipeline_name: String, name of the pipeline (used for printing/plot titles).
        n_splits: Number of folds (default 5).
        n_repeats: Number of repeats (default 10).
        random_state: Random seed for reproducibility.
    
    Returns:
        cv_scores: Array of cross-validation scores.
    """
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=rskf, scoring=auc_scorer)
    mean_score = np.mean(cv_scores)
    print(f"{pipeline_name} Repeated CV AUC scores:", cv_scores)
    print(f"Mean AUC {pipeline_name}: {mean_score:.4f}")
    plot_cv_scores(cv_scores, title=f"{pipeline_name}: Cross-validation fold estimates")
    return cv_scores

def plot_feature_importance(pipeline, X, model_name=""):
    """
    Plot feature importance from a fitted sklearn pipeline.

    This function applies all transformation steps up to the classifier
    to obtain the final feature names, then extracts importance values
    from the classifier step.

    Parameters:
        pipeline: The sklearn Pipeline with steps including 'feature_eng',
                  'additional_fe', 'lag_features', 'imputer', 'scaler', and 'clf'.
        X: pandas DataFrame with the original input features.

    Raises:
        ValueError: If the classifier doesn't support feature importance.
    """
    # Check required steps
    required_steps = ['feature_eng', 'additional_fe', 'lag_features', 'imputer', 'scaler', 'clf']
    missing_steps = [step for step in required_steps if step not in pipeline.named_steps]
    if missing_steps:
        raise ValueError(f"Pipeline is missing required steps: {missing_steps}")

    # Apply full transformation pipeline (up to classifier)
    X_transformed = pipeline.named_steps['feature_eng'].transform(X)
    X_transformed = pipeline.named_steps['additional_fe'].transform(X_transformed)
    X_transformed = pipeline.named_steps['lag_features'].transform(X_transformed)

    # Convert to DataFrame if needed
    if isinstance(X_transformed, np.ndarray):
        raise ValueError("Transformed data is a NumPy array. Custom transformers must return DataFrames to preserve column names.")

    # Apply imputation and scaling
    X_transformed = pd.DataFrame(
        pipeline.named_steps['imputer'].transform(X_transformed),
        columns=X_transformed.columns,
        index=X_transformed.index
    )
    X_transformed = pd.DataFrame(
        pipeline.named_steps['scaler'].transform(X_transformed),
        columns=X_transformed.columns,
        index=X_transformed.index
    )

    feature_names = X_transformed.columns
    clf = pipeline.named_steps['clf']

    # Get importance values
    if hasattr(clf, "coef_"):
        # For linear models like LogisticRegression
        coefficients = clf.coef_[0]
        feat_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': coefficients
        })
        feat_importance['abs_importance'] = feat_importance['importance'].abs()
        feat_importance = feat_importance.sort_values('abs_importance', ascending=True)
        title = f"Feature Importance from Logistic Regression (Coefficients) {model_name}"

    elif hasattr(clf, "feature_importances_"):
        # For tree-based models like XGBoost, LightGBM
        importances = clf.feature_importances_
        feat_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        feat_importance = feat_importance.sort_values('importance', ascending=True)
        title = f"Feature Importance from Tree Model {model_name}"

    else:
        raise ValueError("Classifier does not have a known feature importance attribute.")

    # Plot
    plt.figure(figsize=(12, 12))
    plt.barh(feat_importance['feature'], feat_importance['importance'])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()




!head /kaggle/input/playground-series-s5e3/sample_submission.csv
!head /kaggle/input/playground-series-s5e3/train.csv
!head /kaggle/input/playground-series-s5e3/test.csv


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")#.set_index("id")
train.head()


train.describe().style.background_gradient(cmap='summer')


# Credits https://www.kaggle.com/code/dalloliogm/1-0-lb-rainfall-binary-prediction-nn/edit
def validate_day_alignment(train):
    # Verify that all days now have exactly 6 entries
    fixed_counts = train.groupby('day').size()
    print("Post-Fix Record Counts:", fixed_counts.value_counts())

    # Verify that all sequences are correct
    incorrect_sequences = []
    for day, group in train.groupby('day'):
        expected_ids = [(day - 1) + (365 * i) for i in range(6)]
        actual_ids = sorted(group['id'])
        if expected_ids != actual_ids:
            incorrect_sequences.append(day)

    if incorrect_sequences:
        print(f"ERROR: Some days still have incorrect ID sequences: {incorrect_sequences}")
    else:
        print("All day ID sequences are correctly aligned.")
        
def fix_day_misalignments(train):
    # Define the reassignment map
    reassignment_map = {
        1132: 38, 1251: 157, 1284: 190, 1290: 196, 1312: 218, 1318: 224, 
        1346: 252, 1352: 258, 1367: 273, 1373: 279, 1380: 286, 1382: 288, 
        1388: 294, 1395: 301, 1400: 306, 1037: 308, 1403: 309, 1404: 310, 
        1406: 312, 1407: 313, 1409: 315, 1414: 320, 1416: 322, 1420: 326, 
        1430: 336, 1438: 344, 1439: 345, 1445: 351, 1452: 358, 1453: 359, 
        1457: 363, 1458: 364, 1459: 365, 1210: 116, 1428: 334
    }

    # Apply the reassignments
    for misplaced_id, correct_day in reassignment_map.items():
        train.loc[train['id'] == misplaced_id, 'day'] = correct_day

    print(train.shape)
    # Verify that all days now have exactly 6 entries
    validate_day_alignment(train)

    return train
train = fix_day_misalignments(train)

print("Train shape", train.shape)
train = train.drop_duplicates()

train.head()


train.day.value_counts()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")#.set_index("id")
test.head()


train = train.drop(columns=["id"])
submission100 = pd.read_csv("/kaggle/input/cp-sat-ensemble-100/submission.csv")
train_extra=pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
train_extra.columns = train_extra.columns.str.replace(' ', '')
train_extra = train_extra[train_extra.columns].copy()
train_extra['rainfall'] = train_extra['rainfall'].map({'no': 0, 'yes': 1})
train_extra['humidity']=train_extra['humidity'].astype(float)
train_extra['cloud']=train_extra['cloud'].astype(float)
train_features=list(train)
train_extra=train_extra[train_features]


y_test = np.asarray(submission100['rainfall']).astype(int)
y_test

train_add = test.iloc[0:len(y_test),]
train_add["rainfall"] = y_test
train_add

train_add = train_add.drop(columns=["id"])

train = pd.concat([train, train_add, train_extra], axis=0, ignore_index=True)
train = train.drop_duplicates()
train


train.corr().style.background_gradient(cmap='winter')



!ls /kaggle/input/0-90-rainfall-top-lb-short-analysis
#sns.pairplot(train, kind="kde")


train.isnull().sum()



test.isnull().sum()



test[test.isnull().any(axis=1)]



from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class SeasonMonthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_encode_month=True, drop_original_day=False):
        self.one_hot_encode_month = one_hot_encode_month
        self.drop_original_day = drop_original_day

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_trans = X.copy()

        # Convert day -> month
        def day_to_month(day):
            if day % 365 <= 31: return 1
            elif day % 365 <= 59: return 2
            elif day % 365 <= 90: return 3
            elif day % 365 <= 120: return 4
            elif day % 365 <= 151: return 5
            elif day % 365 <= 181: return 6
            elif day % 365 <= 212: return 7
            elif day % 365 <= 243: return 8
            elif day % 365 <= 273: return 9
            elif day % 365 <= 304: return 10
            elif day % 365 <= 334: return 11
            else: return 12
        
        X_trans['month'] = X_trans['day'].apply(day_to_month)

        # Convert day -> season
        def day_to_season(day):
            if 80 <= day % 365 < 172:
                return 'spring'
            elif 172 <= day % 365 < 264:
                return 'summer'
            elif 264 <= day % 365 < 356:
                return 'autumn'
            else:
                return 'winter'

        X_trans['season'] = X_trans['day'].apply(day_to_season)

        # Add cyclical encoding of day
        X_trans['day_sin'] = np.sin(2 * np.pi * X_trans['day'] / 365)
        X_trans['day_cos'] = np.cos(2 * np.pi * X_trans['day'] / 365)

        # One-hot encode season
        X_trans = pd.get_dummies(X_trans, columns=['season'], drop_first=True)

        # Optional: one-hot encode month
        if self.one_hot_encode_month:
            X_trans = pd.get_dummies(X_trans, columns=['month'], prefix='month', drop_first=True)

        # Optional: drop original 'day' column
        if self.drop_original_day and 'day' in X_trans.columns:
            X_trans.drop(columns=['day'], inplace=True)

        return X_trans



from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

class AdditionalFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, encode_winddirection=True, drop_cols_for_logistic=True):
        self.encode_winddirection = encode_winddirection
        self.drop_cols_for_logistic = drop_cols_for_logistic
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        # Fit PowerTransformer on pressure if available
        if 'pressure' in X.columns:
            self.pt.fit(X[['pressure']])
        return self

    def transform(self, X):
        X_trans = X.copy()

        # ------------------- Sunshine Metrics -------------------
        if all(col in X.columns for col in ['sunshine', 'cloud', 'humidity']):
            X_trans['relative_sunshine'] = X_trans['sunshine'] / (100 - X_trans['cloud'] + 1)
            X_trans['sunshine_ratio'] = X_trans['sunshine'] / (X_trans['cloud'] + X_trans['humidity'] + 1e-5)
        if all(col in X.columns for col in ['sunshine', 'cloud']):
            X_trans['cloud_sun_ratio'] = X_trans['cloud'] / (X_trans['sunshine'] + 1)
        if 'sunshine' in X.columns:
            X_trans['sunshine_pct'] = X_trans['sunshine'] / 24.0

        # ------------------- Cloud Metrics -------------------
        if 'cloud' in X.columns:
            X_trans['cloud_gradient'] = X_trans['cloud'] - X_trans['cloud'].shift(1, fill_value=X_trans['cloud'].iloc[0])
            X_trans['cloud_category'] = pd.cut(X_trans['cloud'], bins=[0, 20, 50, 80, 100],
                                               labels=[0, 1, 2, 3], include_lowest=True).astype(float)
            X_trans['sky_opacity'] = X_trans['cloud'] / 100.0

        # ------------------- Temperature Metrics -------------------
        if all(col in X.columns for col in ['maxtemp', 'mintemp']):
            X_trans['temp_range'] = X_trans['maxtemp'] - X_trans['mintemp']
        if 'temparature' in X.columns:
            X_trans['temp_change'] = X_trans['temparature'] - X_trans['temparature'].shift(1, fill_value=X_trans['temparature'].iloc[0])
            X_trans['temp_ewm'] = X_trans['temparature'].ewm(span=10, adjust=False).mean()
            if 'humidity' in X.columns:
                X_trans['temp_humidity_interaction'] = X_trans['temparature'] + 0.2 * X_trans['humidity']

        # ------------------- Pressure Metrics -------------------
        if 'pressure' in X.columns:
            X_trans['pressure_rolling_mean'] = X_trans['pressure'].rolling(window=7, min_periods=1).mean()
            X_trans['pressure_rolling_std'] = X_trans['pressure'].rolling(window=7, min_periods=1).std()
            X_trans['pressure_diff'] = X_trans['pressure'] - X_trans['pressure'].shift(1, fill_value=X_trans['pressure'].iloc[0])
            X_trans['pressure'] = self.pt.transform(X_trans[['pressure']])

        # ------------------- Humidity Metrics -------------------
        if all(col in X.columns for col in ['temparature', 'dewpoint']):
            X_trans['dewpoint_depression'] = X_trans['temparature'] - X_trans['dewpoint']
            X_trans['rh_approx'] = 100 - (5 * X_trans['dewpoint_depression'])
        if all(col in X.columns for col in ['humidity', 'cloud']):
            X_trans['humidity_cloud_interaction'] = (X_trans['humidity'] * X_trans['cloud']) / 10000.0
            X_trans['inv_humidity_cloud'] = 100 - X_trans['humidity'] - X_trans['cloud']

        # ------------------- Dewpoint Metrics -------------------
        if 'temparature' in X.columns:
            X_trans['svp'] = 6.1078 * np.exp((17.27 * X_trans['temparature']) / (X_trans['temparature'] + 237.3))
        if all(col in X.columns for col in ['temparature', 'humidity']):
            X_trans['abs_humidity'] = (6.112 * np.exp((17.67 * X_trans['temparature']) / (X_trans['temparature'] + 243.5)) *
                                       X_trans['humidity'] * 2.1674) / (273.15 + X_trans['temparature'])

        # ------------------- Wind Direction -------------------
        if 'winddirection' in X.columns:
            X_trans['change_in_direction'] = abs(X_trans['winddirection'] - X_trans['winddirection'].shift(1, fill_value=X_trans['winddirection'].iloc[0]))
            
            if self.encode_winddirection:
                # Bin wind direction into 16 categories (22.5 degrees each)
                wind_bins = np.linspace(0, 360, 17)
                wind_labels = list(range(16))
                wind_cat = pd.cut(X_trans['winddirection'], bins=wind_bins, labels=wind_labels, include_lowest=True)
                wind_dummies = pd.get_dummies(wind_cat, prefix='winddir')
                X_trans = pd.concat([X_trans, wind_dummies], axis=1)

        # ------------------- Optional Feature Dropping -------------------
        if self.drop_cols_for_logistic:
            drop_cols = ['id', 'day', 'winddirection']
            X_trans.drop(columns=[col for col in drop_cols if col in X_trans.columns], inplace=True)

        return X_trans



from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, n_lags=5, include_target_lags=True):
        # Set default columns if not provided. Assume 'rainfall' might be in there.
        default_columns = ['humidity', 'temparature', 'pressure', 'sunshine', 'rainfall']
        self.columns = columns if columns is not None else default_columns
        # If not including target lags, remove 'rainfall' from the list.
        if not include_target_lags and 'rainfall' in self.columns:
            self.columns = [col for col in self.columns if col != 'rainfall']
        self.n_lags = n_lags
        self.include_target_lags = include_target_lags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_trans = X.copy()
        for col in self.columns:
            if col in X_trans.columns:
                for lag in range(1, self.n_lags + 1):
                    X_trans[f"{col}_lag_{lag}"] = X_trans[col].shift(lag)
            else:
                print(f"Warning: Column '{col}' not found in data. Skipping lag features for this column.")
        return X_trans



train.columns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

# Initialize the imputer to replace NaN with the mean value
imputer = SimpleImputer(strategy='mean')

# Impute missing values in the clustering variables
train_imputed = imputer.fit_transform(train[config["clustering_variables"]])
test_imputed = imputer.transform(test[config["clustering_variables"]])

# Then scale the imputed data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_imputed)
test_scaled = scaler.transform(test_imputed)

# Proceed with PCA and KMeans as before
pca = PCA(n_components=config.get("n_pca_components", 4), random_state=42)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled)

kmeans = KMeans(n_clusters=config["n_clusters"], random_state=42)
train['cluster'] = kmeans.fit_predict(train_pca)
test['cluster'] = kmeans.predict(test_pca)

train = pd.get_dummies(train, columns=['cluster'], prefix='cluster')
test = pd.get_dummies(test, columns=['cluster'], prefix='cluster')




import matplotlib.pyplot as plt

# Assume 'train_pca' is the 2D PCA-transformed data
# and 'cluster' column contains cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(train_pca[:, 0], train_pca[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Training Data with KMeans Clusters')
plt.colorbar(label='Cluster ID')
plt.grid(True)
plt.show()




# Refit PCA with all components
pca_full = PCA().fit(train_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.title('PCA Explained Variance')
plt.grid(True)
plt.show()



explained = pca.explained_variance_ratio_.sum()
print(f"PCA explains {explained:.2%} of the variance")


train.head()





from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def create_pipeline(lag_columns, n_lags, clf, include_target_lags=True):
    steps = [
        ('feature_eng', SeasonMonthTransformer()),
        ('additional_fe', AdditionalFeatureTransformer()),
        ('lag_features', LagFeatureTransformer(columns=lag_columns, n_lags=n_lags, include_target_lags=include_target_lags)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', clf)
    ]
    return Pipeline(steps)

# Define your feature columns and data
features = [*config["features"], *[c for c in train.columns if c.startswith('cluster')]]

X = train[features]
y = train['rainfall']

# Create the logistic regression pipeline
pipeline_lg = create_pipeline(
    n_lags = config["n_lags"],
    lag_columns = config["lag_columns"],
    clf=LogisticRegression(
        penalty='l1',        # L1 (Lasso) regularization
        C=1.0,               # adjust for stronger/weaker regularization
        max_iter=1000,
        random_state=42,
        solver='liblinear'   # supports L1 penalty
    ),
    include_target_lags = True
)

# Create the XGBoost pipeline 
pipeline_xgb = create_pipeline(
    n_lags = config["n_lags"],
    lag_columns = config["lag_columns"],
    clf=XGBClassifier(
        device=config["device"],
        n_estimators=config["n_estimators"],
        learning_rate=0.1,
        max_depth=6,
#        early_stopping_rounds=100,
        alpha=0.1,
        random_state=42,
        colsample_bytree=0.9, 
        subsample=0.9,
        use_label_encoder=False,    # Disable label encoder to avoid warnings
        eval_metric='auc'           # Set evaluation metric to AUC
    ),
    include_target_lags = True
)

# Create the XGBoost pipeline (with a different lag configuration)
pipeline_lgbm = create_pipeline(
    n_lags = config["n_lags"],
    lag_columns = config["lag_columns"],
    clf=LGBMClassifier(
        device="cpu", # config["device"], # lightGBM not compiled for CUDA in this environment
        n_estimators=config["n_estimators"],
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1  # This will suppress the warnings
    ),
    include_target_lags = True
)




cv_scores_lg   = evaluate_pipeline(pipeline_lg, X, y, "Logistic Regression")
mean_cv_scores_lg = np.mean(cv_scores_lg)
pipeline_lg.fit(X, y)
plot_feature_importance(pipeline_lg, X, "Feature Importance Logistic Regression")


cv_scores_lgbm = evaluate_pipeline(pipeline_lgbm, X, y, "LightGBM")
mean_cv_scores_lgbm = np.mean(cv_scores_lgbm)
pipeline_lgbm.fit(X, y)
plot_feature_importance(pipeline_lgbm, X, "Feature Importance LightGBM")



cv_scores_xgb  = evaluate_pipeline(pipeline_xgb, X, y, "XGBoost")
mean_cv_scores_xgb = np.mean(cv_scores_xgb)
pipeline_xgb.fit(X, y)
plot_feature_importance(pipeline_xgb, X, "Feature Importance XGBoost")






# Refit on the full training set to ensure coefficients are available
#pipeline.fit(X, y)



X_test = test[features]
X_test


# Build the ensemble using the prediction pipelines:
from sklearn.ensemble import VotingClassifier

results = {
    "LG": mean_cv_scores_lg,
    "XGB": mean_cv_scores_xgb,
    "LGBM": mean_cv_scores_lgbm
}
total_score = mean_cv_scores_lg + mean_cv_scores_xgb + mean_cv_scores_lgbm
weights = [mean_cv_scores_lg / total_score, mean_cv_scores_xgb / total_score, mean_cv_scores_lgbm / total_score]

ensemble = VotingClassifier(estimators=[
    ('LG', pipeline_lg),
    ('XGB', pipeline_xgb),
    ('LGBM', pipeline_lgbm)
], voting='soft', weights=weights)

# Fit the ensemble on the training set (using X that includes rainfall so that feature engineering runs as in training)
ensemble.fit(X, y)

## Predict on test data (which does NOT include rainfall)
#test_preds = ensemble.predict_proba(X_test)[:, 1]



ensemble.predict_proba(X_test)


def iterative_predict(pipeline, transformer, test_df, train_last_rainfall, train_feature_names, lag_feature_name="rainfall_lag_1"):
    """
    Iteratively predicts test data, updating the lag feature with each prediction, 
    while ensuring the transformed features match the training set.
    
    Parameters:
        pipeline: The fitted classifier (or ensemble) from your pipeline.
        transformer: A Pipeline containing the transformation steps (all steps except the classifier).
        test_df: DataFrame of raw test features (must include all columns present during training).
        train_last_rainfall: The last known rainfall value from the training set.
        train_feature_names: The list of feature names produced by transformer on training data.
        lag_feature_name: The column name for the lagged rainfall feature.
        
    Returns:gtr
        A list of predicted probabilities for each test row.
    """
    test = test_df.copy()
    predictions = []
    last_value = train_last_rainfall  # initialize with last known rainfall from training
    
    # Iterate over test rows in time order
    for idx in test.index:
        # Update the lag column with the latest known/predicted value
        test.loc[idx, lag_feature_name] = last_value
        
        # Transform the current row using the transformer
        X_current = test.loc[[idx]]
        X_current_trans = transformer.transform(X_current)
        # Reindex to ensure the columns match those from training
        X_current_trans = X_current_trans.reindex(columns=train_feature_names, fill_value=0)
        
        # Get the predicted probability
        pred_prob = pipeline.predict_proba(X_current_trans)[:, 1][0]
        predictions.append(pred_prob)
        
        # Update last_value for the next iteration
        last_value = pred_prob
    
    return predictions

# ------------------------------------------------------------------------------
# Setup: Extract transformation pipeline and training feature names

# Use one of your pipelines (here pipeline_lg) as representative
# Extract only the custom transformation steps that return a DataFrame
transformer = Pipeline(pipeline_lg.steps[:1])
train_trans = transformer.transform(X)
train_feature_names = train_trans.columns


# Get the last rainfall value from training (to initialize the lag)
train_last_rainfall = train['rainfall'].iloc[-1]

# Ensure that the test DataFrame has the 'rainfall_lag_1' column (even if as placeholder)
if "rainfall_lag_1" not in test.columns:
    test["rainfall_lag_1"] = np.nan

# Use the iterative_predict function with your ensemble (or any pipeline)
test_preds = iterative_predict(ensemble, transformer, test, train_last_rainfall, train_feature_names, lag_feature_name="rainfall_lag_1")



#pipeline.fit(X,y)
# Predict probabilities on the test set
#test_preds = pipeline.predict_proba(X_test)[:, 1]

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_preds
})

# Save the submission file
submission.to_csv('submission.csv', index=False)



test_preds[0:10]


submission.head()


sns.histplot(submission, x="rainfall")

