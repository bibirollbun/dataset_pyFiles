# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!python --version


# !pip install -r mreq.txt


import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import math


# !pip install -U scikit-learn


import sklearn
sklearn.__version__


import xgboost
xgboost.__version__


# from google.colab import drive
# drive.mount('/content/drive')

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
# test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# train = pd.read_csv('/content/drive/MyDrive/kaggle/data-fertilizer/train.csv')
# train = pd.read_csv('data/train.csv')


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer


def mapk(y_true, y_pred_proba, k=3, **kwargs):
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]

    ap_at_k_scores = []

    if y_pred_proba.ndim == 1:
        y_pred_proba = np.vstack([1 - y_pred_proba, y_pred_proba]).T

    for true_label, k_y_preds in zip(y_true, top_k_preds):
        score = 0.0
        # num_hits = 0.0

        for i, p in enumerate(k_y_preds):
            if p == true_label:
                # num_hits += 1.0
                score += 1.0 / (i + 1.0)
                break

        ap_at_k_scores.append(score)

    if not ap_at_k_scores:
        return 0.0

    return np.mean(ap_at_k_scores)


def create_features(df):
    df_copy = df.copy()
    epsilon = 1e-6

    df_copy['NbyP'] = df_copy['Nitrogen'] / (df_copy['Phosphorous'] + epsilon)
    df_copy['NbyK'] = df_copy['Nitrogen'] / (df_copy['Potassium'] + epsilon)
    df_copy['PbyK'] = df_copy['Phosphorous'] / (df_copy['Potassium'] + epsilon)
    df_copy['NPKsum'] = df_copy['Nitrogen'] + df_copy['Phosphorous'] + df_copy['Potassium']

    return df_copy


train_eng = create_features(train)
train_eng.head()


train_eng.info()


target='Fertilizer Name'
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'NbyP', 'NbyK', 'PbyK', 'NPKsum']
categorical_features = ['Soil Type', 'Crop Type']


X_train = train_eng.drop(columns=[target, 'id'])
y = train_eng[target]

X_train.head()


le = LabelEncoder()
y_train_encoded = le.fit_transform(y)
y_train_encoded


print(len(X_train))
print(len(y_train_encoded))


# whole pipeline for preprocessing

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
     ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])


from xgboost import XGBClassifier


import imblearn
imblearn.__version__


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


xgb_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('oversampler', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',
        # use_label_encoder=False,
        random_state=42
    ))
])


from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer


# param_dist = {
#     'classifier__n_estimators': randint(200, 1000),
#     'classifier__max_depth': randint(5, 15),
#     'classifier__learning_rate': uniform(loc=0.01, scale=0.29),
#     'classifier__subsample': uniform(loc=0.6, scale=0.4),
#     'classifier__colsample_bytree': uniform(loc=0.6, scale=0.4)
# }

# New parameter distribution as per previous results
param_dist = {
    'classifier__n_estimators': randint(400, 1500),
    'classifier__max_depth': randint(6, 16),
    'classifier__learning_rate': uniform(loc=0.01, scale=0.09),
    'classifier__subsample': uniform(loc=0.6, scale=0.4),
    'classifier__colsample_bytree': uniform(loc=0.5, scale=0.5),
    'classifier__lambda': uniform(loc=1, scale=4)
}


map3_scorer = make_scorer(mapk, response_method='predict_proba')


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# test for what is xgb pipeline returning
# just using for
# xgb_pipeline.fit(X_train, y_train_encoded)


# y_pred = xgb_pipeline.predict_proba(X_train)
# y_pred = xgb_pipeline.predict(X_train)


# y_pred[:5]


# just testing to know what exactly i am doing

# top_k_preds_ex_test = np.argsort(y_pred, axis=1)[:, ::-1][:, :3]
# top_k_preds_ex_test[:20]


# map3_score = mapk(y_train_encoded, y_pred, k=3)
# map3_score


random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,  # Use your full pipeline here
    param_distributions=param_dist,
    # n_iter=50,
    # n_iter=3,
    n_iter=3,
    cv=skf,
    scoring=map3_scorer,
    # n_jobs=-1,
    n_jobs = 1, # doing for debugging and gpu
    verbose=2,
    random_state=42
)


print("Starting Randomized Search for XGBoost...")
random_search.fit(X_train, y_train_encoded)

print("\nRandomized Search Complete.")
print(f"Best MAP@3 score found: {random_search.best_score_:.4f}")
print("Best parameters found:")
print(random_search.best_params_)

best_xgb_model = random_search.best_estimator_


# best_param_1 = {'colsample_bytree': np.float64(0.749816047538945), 'learning_rate': np.float64(0.28570714885887566), 'max_depth': 12, 'n_estimators': 900, 'subsample': np.float64(0.8387400631785948)}
# iter_1_xgb = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(**best_param_1, random_state=42))
# ])


# iter_1_xgb.fit(X_train, y_train_encoded)


# y_pred_1_iter = iter_1_xgb.predict_proba(X_train)


# y_pred_1_iter[:5]


# iter_1_score = mapk(y_train_encoded, y_pred_1_iter, k=3)
# iter_1_score




