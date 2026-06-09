import os
import gc
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):    
    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        self.num_agg_funcs = ['mean', 'std', 'min', 'max', 'last']
        self.cat_agg_funcs = ['count', 'last', 'nunique']
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if 'customer_ID' not in X.columns:
            raise ValueError("Data must contain 'customer_ID' column")
        
        # Num feat engineer
        num_agg = X.groupby("customer_ID")[self.num_features].agg(self.num_agg_funcs)
        num_agg.columns = ['_'.join(x) for x in num_agg.columns]
        num_agg = num_agg.reset_index()
        
        # Cat feat engineer
        cat_agg = X.groupby("customer_ID")[self.cat_features].agg(self.cat_agg_funcs)
        cat_agg.columns = ['_'.join(x) for x in cat_agg.columns]
        cat_agg = cat_agg.reset_index()
        
        result = num_agg.merge(cat_agg, how='inner', on='customer_ID')
        return result

def create_feature_pipeline():
    train = pd.read_parquet('train_data.parquet')
    features = train.drop(['customer_ID', 'S_2', 'target'], axis=1).columns.tolist()
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", 
                    "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in features if col not in cat_features]
    
    feature_pipeline = Pipeline([
        ('feature_engineering', FeatureEngineeringTransformer(num_features, cat_features))
    ])
    return feature_pipeline, num_features, cat_features


def preprocess_data_with_pipeline():    
    feature_pipeline, num_features, cat_features = create_feature_pipeline()
    
    train_raw = pd.read_parquet('train_data.parquet')
    train_transformed = feature_pipeline.fit_transform(train_raw)
    
    train_labels = pd.read_csv('train_labels.csv')
    train_final = train_transformed.merge(train_labels, how='inner', on='customer_ID')
    
    test_raw = pd.read_parquet('test_data.parquet')
    test_final = feature_pipeline.transform(test_raw)
    
    train_final.to_parquet('train_fe.parquet', index=False)
    test_final.to_parquet('test_fe.parquet', index=False)
    return train_final, test_final, cat_features

train, test, cat_features_original = preprocess_data_with_pipeline()


cat_features_transformed = [f"{cf}_last" for cf in cat_features_original]
train[cat_features_transformed] = train[cat_features_transformed].fillna("missing").astype(str)
test[cat_features_transformed] = test[cat_features_transformed].apply(
    lambda x: x.cat.add_categories("missing").fillna("missing") 
    if x.dtype.name == 'category' 
    else x.fillna("missing")
)


for cat_col in cat_features_transformed:
    encoder = LabelEncoder()
    encoder.fit(train[cat_col].tolist() + test[cat_col].tolist())
    train[cat_col] = encoder.transform(train[cat_col])
    test[cat_col] = encoder.transform(test[cat_col])


# COMPETITION METRIC FROM Konstantin Yakovlev
# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

amex_scorer = make_scorer(amex_metric_mod, greater_is_better=True, needs_proba=True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)
n_folds = 5
TARGET = 'target'


features = [col for col in train.columns if col not in ['customer_ID', TARGET]]
X = train[features]
y = train[TARGET]

params = {
    'objective': 'binary',
    'metric': "binary_logloss",
    'boosting': 'dart',
    'seed': 42,
    'num_leaves': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.20,
    'bagging_freq': 10,
    'bagging_fraction': 0.50,
    'n_jobs': 32,
    'lambda_l2': 2,
    'min_data_in_leaf': 40
    }


lgb_model = lgb.LGBMClassifier(**params)
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print("="*50)
print("Running cross_validate with LightGBM")
print("="*50)

cv_results = cross_validate(
    estimator=lgb_model,
    X=X,
    y=y,
    cv=cv,
    scoring={'amex': amex_scorer},
    return_train_score=True,
    return_estimator=True,
    n_jobs=1,
)

print(f"Metrics for each folds: {cv_results['test_amex']}, OOF metric: {cv_results['test_amex'].mean():.6f}")


test_predictions_matrix = np.zeros((len(test), n_folds))

for fold_idx, model in enumerate(cv_results['estimator']):
    print(f"Предсказываем  для {fold_idx} фолдв")
    test_pred = model.predict_proba(test[features])[:, 1]
    test_predictions_matrix[:, fold_idx] = test_pred

test_predictions_avg = test_predictions_matrix.mean(axis=1)


test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions_avg})
test_df.to_csv('submission.csv', index = False)




