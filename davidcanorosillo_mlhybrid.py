# install
!pip install openfe -U -q


import pandas as pd
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')


train_x = train.drop(columns=['Hardness'])
train_y = train['Hardness']


%%capture

from openfe import OpenFE, transform

# generate new features
ofe = OpenFE()
features = ofe.fit(data=train_x, label=train_y)


# transform the train and test data according to generated features.
train_f, test_f = transform(train_x, test, features, n_jobs=1) 


train_f['Hardness'] = train_y


train_f.describe().T


train_f.reset_index(drop=True).to_csv('train_OFE_features.csv', index='False')
test_f.to_csv('test_OFE_features.csv', index=False)


target = 'Hardness'
X, y = train_f.drop(target, axis=1), train_f[target]
X = X.drop('id', axis=1)
p_train = .3

X_train, y_train = X.iloc[:round(p_train*len(X))], y.iloc[:round(p_train*len(X))]
X_test, y_test = X.iloc[:round((1-p_train)*len(X))], y.iloc[:round((1-p_train)*len(X))]


pd.unique(X_train.dtypes)


import shap

def select_top_k_features(X, y, model, k=10):
    """
    Selects the top-k features using SHAP values.

    Parameters:
    - X: Feature matrix (DataFrame).
    - y: Target vector (Series or ndarray).
    - model: A pre-trained machine learning model compatible with SHAP.
    - k: Number of top features to select.

    Returns:
    - List of top-k feature names based on SHAP importance.
    """
    
    # Use SHAP to calculate feature importances
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Calculate mean absolute SHAP values for feature importance
    shap_importance = np.abs(shap_values[1]).mean(axis=0)  # For class 1 in binary classification
    feature_scores = pd.Series(shap_importance, index=X.columns)
    
    # Select top-k features
    selected_features = feature_scores.nlargest(k).index.tolist()
    print(f"Top {k} features based on SHAP importance:", selected_features)
    return selected_features

import lightgbm as lgb

model = lgb.LGBMRegressor(random_state=42)
model = model.fit(X_train,y_train)
k_feats = select_top_k_features(X_train, y_train, model, k=100)


from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

def test_feat_subset(X_train, y_train, X_test, y_test, feats):
    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model.fit(X_train[feats], y_train)
    preds = model.predict(X_test[feats])
    
    return preds

preds = test_feat_subset(X_train, y_train, X_test, y_test, k_feats)
mean_squared_error(preds, y_test)


k_feats_reduced = k_feats[:20]
preds = test_feat_subset(X_train, y_train, X_test, y_test, k_feats_reduced)
mean_squared_error(preds, y_test)


%timeit
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def recursive_feature_elimination(X, y, selected_features, n_features_to_select=5):
    model = lgb.LGBMRegressor(random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    X_selected = X[selected_features]
    rfe.fit(X_selected, y)
    final_features = X_selected.columns[rfe.support_].tolist()
    print("Final selected features after RFE:", final_features)
    return final_features

final_features = recursive_feature_elimination(
    X_train, y_train, k_feats, n_features_to_select=20
)


preds = test_feat_subset(X_train, y_train, X_test, y_test, final_features)
mean_squared_error(preds, y_test)




