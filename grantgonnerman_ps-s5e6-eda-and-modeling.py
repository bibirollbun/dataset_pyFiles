# Importing libraries 
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np
import optuna

from tqdm import tqdm
from IPython.display import clear_output
from ydata_profiling import ProfileReport

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, cohen_kappa_score, confusion_matrix, mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVC, SVR
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


#loading dataset
train = pd.read_csv('../input/playground-series-s5e6/train.csv', index_col = 'id')
test = pd.read_csv('../input/playground-series-s5e6/test.csv', index_col = 'id')
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv('../input/playground-series-s5e6/sample_submission.csv')

print('The dimession of the train dataset is:', train.shape)
print('The dimession of the original dataset is:', original.shape)
print('The dimession of the test dataset is:', test.shape)


train.info()


test.info()


train.describe()


test.describe()


print('Train duplicates: ', train.duplicated().sum())
print('original duplicates: ', original.duplicated().sum())
print('Test duplicates: ', test.duplicated().sum())


# Create the profile report
profile = ProfileReport(train, title="Pandas Profiling Report", explorative=True)
profile.to_notebook_iframe()


fig, ax = plt.subplots(2, 3, figsize=(18, 8))

sns.boxplot(data=train, x='Fertilizer Name', y='Temparature', ax=ax[0,0])
sns.boxplot(data=train, x='Fertilizer Name', y='Humidity', ax=ax[0,1])
sns.boxplot(data=train, x='Fertilizer Name', y='Moisture', ax=ax[0,2])
sns.boxplot(data=train, x='Fertilizer Name', y='Nitrogen', ax=ax[1,0])
sns.boxplot(data=train, x='Fertilizer Name', y='Potassium', ax=ax[1,1])
sns.boxplot(data=train, x='Fertilizer Name', y='Phosphorous', ax=ax[1,2])

plt.tight_layout()
plt.show();


fig, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.countplot(data=train, x='Soil Type', hue='Fertilizer Name', ax=ax[0])
sns.countplot(data=train, x='Crop Type', hue='Fertilizer Name', ax=ax[1])
ax[0].legend_.remove()

plt.tight_layout()
plt.show();


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = list(p[:k])  # Convert NumPy array to list
        if a in p:
            return 1.0 / (p.index(a) + 1)
        return 0.0

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


le = LabelEncoder()

# Baseline Modeling
X = train.drop(columns=["Fertilizer Name"], axis=1)
X = pd.get_dummies(train.drop(columns=["Fertilizer Name"]), columns=["Soil Type", "Crop Type"])
X['original'] = 0
y = train["Fertilizer Name"]
y_encoded = le.fit_transform(y)

# set up origninal data
X_original = original.drop(columns=["Fertilizer Name"], axis=1)
X_original = pd.get_dummies(original.drop(columns=["Fertilizer Name"]), columns=["Soil Type", "Crop Type"])
X_original['original'] = 1
y_original_encode = le.fit_transform(original["Fertilizer Name"])

# test data
test_base = test
test_base = pd.get_dummies(test, columns=["Soil Type", "Crop Type"])
test_base['original'] = 0

# Align all feature sets to ensure column consistency
X, X_original = X.align(X_original, join='outer', axis=1, fill_value=0)
X, test_base = X.align(test_base, join='outer', axis=1, fill_value=0)

num_classes = len(np.unique(y_encoded))
oof_meta_features = np.zeros((len(X), num_classes * 3))  # XGB, LGB, HGB
test_meta_features = []

hgb_map, xgb_map, lgb_map = list(), list(), list()
hgb_preds, xgb_preds, lgb_preds = list(), list(), list()
hgb_cv_score, xgb_cv_score, lgb_cv_score = list(), list(), list()
hgb_imp, xgb_imp, lgb_imp = list(), list(), list()

## Running 5 times CV
for i in range(5):
    print('loop: ', i+1)
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for fold, (train_ix, test_ix) in enumerate(skf.split(X, y_encoded)):
        
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y_encoded[train_ix].ravel(), y_encoded[test_ix].ravel()

        X_train = pd.concat([X_train, X_original, X_original, X_original, X_original, X_original], axis=0)
        y_train = np.concatenate([y_train, y_original_encode, y_original_encode, y_original_encode, y_original_encode, y_original_encode], axis=0)
    
        ## Building xgb model ##
        xgb_md = XGBClassifier(learning_rate = 0.1, 
                               n_estimators = 276, 
                               max_depth = 10, 
                               min_child_weight = 109, 
                               gamma = 0.016956098917901095, 
                               alpha = 0.0020043865461097815, 
                               colsample_bytree = 0.37021993853928137, 
                               enable_categorical=True,
                               tree_method='hist',
                               device='cuda',
                               subsample = 0.4018846579568579).fit(X_train, y_train)
        # extracting feature importances
        xgb_imp.append(xgb_md.feature_importances_)
        ## Predicting on X_test and test
        xgb_pred_1 = xgb_md.predict_proba(X_test)
        xgb_pred_2 = xgb_md.predict_proba(test_base)

        
        ## Building lgb model ##
        lgb_md = LGBMClassifier(n_estimators = 431, 
                                learning_rate = 0.1, 
                                num_leaves = 663, 
                                min_data_in_leaf = 51, 
                                min_child_weight = 0.09614241002958618, 
                                max_depth = 6, 
                                bagging_fraction = 0.8276967656687921, 
                                feature_fraction = 0.5187405202716673, 
                                lambda_l1 = 1.2448192075203763, 
                                lambda_l2 = 0.5986697619520711,
                                device='gpu',
                                verbose = -1).fit(X_train, y_train)
        # extracting feature importances
        lgb_imp.append(lgb_md.feature_importances_)
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict_proba(X_test)
        lgb_pred_2 = lgb_md.predict_proba(test_base)

        
        ## Building hgb model ##
        hgb_md = HistGradientBoostingClassifier(max_iter = 609, 
                                               learning_rate = 0.1, 
                                               l2_regularization = 0.1, 
                                               min_samples_leaf = 175, 
                                               max_leaf_nodes = 53, 
                                               max_depth = 5, 
                                               max_bins = 105).fit(X_train, y_train)
        ## Predicting on X_test and test
        hgb_pred_1 = hgb_md.predict_proba(X_test)
        hgb_pred_2 = hgb_md.predict_proba(test_base)
        
        # Computing metrics
        top3_xgb = np.argsort(xgb_pred_1, axis=1)[:, ::-1][:, :3]
        xgb_map.append(mapk(y_test, top3_xgb))
        top3_lgb = np.argsort(lgb_pred_1, axis=1)[:, ::-1][:, :3]
        lgb_map.append(mapk(y_test, top3_lgb))
        top3_hgb = np.argsort(hgb_pred_1, axis=1)[:, ::-1][:, :3]
        hgb_map.append(mapk(y_test, top3_hgb))
        
        # test preds
        xgb_preds.append(xgb_pred_2)
        lgb_preds.append(lgb_pred_2)
        hgb_preds.append(hgb_pred_2)

        # Fill OOF and test meta features
        oof_meta_features[test_ix] = np.hstack([xgb_pred_1, lgb_pred_1, hgb_pred_1])
        test_meta_features.append(np.hstack([xgb_pred_2, lgb_pred_2, hgb_pred_2]))

    clear_output()

xgb_cv_score = np.mean(xgb_map)
lgb_cv_score = np.mean(lgb_map)
hgb_cv_score = np.mean(hgb_map)

# Train Meta Model
meta_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
meta_model.fit(oof_meta_features, y_encoded)

# Predict on averaged test features
test_meta_avg = np.mean(test_meta_features, axis=0)
final_stack_preds = meta_model.predict_proba(test_meta_avg)
top3_stack = np.argsort(final_stack_preds, axis=1)[:, ::-1][:, :3]

print('The Mean Average Precision @ 3 of the xgb model over 5-folds (run 5 times) is:', xgb_cv_score)
print('The Mean Average Precision @ 3 of the lgb model over 5-folds (run 5 times) is:', lgb_cv_score)
print('The Mean Average Precision @ 3 of the hgb model over 5-folds (run 5 times) is:', hgb_cv_score)


# Plotting importances
data1 = pd.DataFrame(pd.DataFrame(xgb_imp, columns = X.columns).apply(np.mean, axis = 0))
data1['Feature'] = data1.index
data1.columns = ['XGB Score', 'Feature']
data1.reset_index(drop = True, inplace = True)
data1.sort_values(by = 'XGB Score', ascending = False, inplace = True)

data2 = pd.DataFrame(pd.DataFrame(lgb_imp, columns = X.columns).apply(np.mean, axis = 0))
data2['Feature'] = data2.index
data2.columns = ['LightGBM Score', 'Feature']
data2.reset_index(drop = True, inplace = True)
data2.sort_values(by = 'LightGBM Score', ascending = False, inplace = True)


fig, axes = plt.subplots(2, 1, figsize = (15, 25))
sns.barplot(ax = axes[0], data = data1, x = 'XGB Score', y = 'Feature', color = 'steelblue')
sns.barplot(ax = axes[1], data = data2, x = 'LightGBM Score', y = 'Feature', color = 'steelblue');


xgb_preds_mean = np.mean(xgb_preds, axis=0)
lgb_preds_mean = np.mean(lgb_preds, axis=0)
hgb_preds_mean = np.mean(hgb_preds, axis=0)

# straight average ensemble
avg_ensemble = (xgb_preds_mean + lgb_preds_mean + hgb_preds_mean) / 3
top3_avg_ensemble = np.argsort(avg_ensemble, axis=1)[:, ::-1][:, :3]

# weighted average ensemble
cv_scores = np.array([xgb_cv_score, lgb_cv_score, hgb_cv_score])
weights = cv_scores / cv_scores.sum()

weighted_ensemble = (weights[0] * xgb_preds_mean +
                     weights[1] * lgb_preds_mean +
                     weights[2] * hgb_preds_mean)

top3_weighted_ensemble = np.argsort(weighted_ensemble, axis=1)[:, ::-1][:, :3]

# rank averaging ensemble
from scipy.stats import rankdata

def rank_averaging(*probs):
    ranks = [np.apply_along_axis(rankdata, 1, -p) for p in probs]
    avg_rank = np.mean(ranks, axis=0)
    return avg_rank

rank_avg = rank_averaging(xgb_preds_mean, lgb_preds_mean, hgb_preds_mean)
top3_rank_ensemble = np.argsort(rank_avg, axis=1)[:, ::-1][:, :3]


# Creating submissions
xgb_test_preds = np.stack(xgb_preds, axis=0)
lgb_test_preds = np.stack(lgb_preds, axis=0)
hgb_test_preds = np.stack(hgb_preds, axis=0)

# Average predictions across folds/runs
xgb_mean_preds = np.mean(xgb_test_preds, axis=0)
lgb_mean_preds = np.mean(lgb_test_preds, axis=0)
hgb_mean_preds = np.mean(hgb_test_preds, axis=0)

# Get top-3 predicted class indices
xgb_top3_preds = np.argsort(xgb_mean_preds, axis=1)[:, ::-1][:, :3]
lgb_top3_preds = np.argsort(lgb_mean_preds, axis=1)[:, ::-1][:, :3]
hgb_top3_preds = np.argsort(hgb_mean_preds, axis=1)[:, ::-1][:, :3]

# Convert indices to original class names
xgb_test_shape = xgb_top3_preds.shape
xgb_top_3_predictions = le.inverse_transform(xgb_top3_preds.reshape(-1, 1)).reshape(xgb_test_shape)

lgb_test_shape = lgb_top3_preds.shape
lgb_top_3_predictions = le.inverse_transform(lgb_top3_preds.reshape(-1, 1)).reshape(lgb_test_shape)

hgb_test_shape = hgb_top3_preds.shape
hgb_top_3_predictions = le.inverse_transform(hgb_top3_preds.reshape(-1, 1)).reshape(hgb_test_shape)

avg_ensemble_shape = top3_avg_ensemble.shape
avg_ensemble_top_3_predictions = le.inverse_transform(top3_avg_ensemble.reshape(-1, 1)).reshape(avg_ensemble_shape)

weighted_ensemble_shape = top3_weighted_ensemble.shape
weighted_ensemble_top_3_predictions = le.inverse_transform(top3_weighted_ensemble.reshape(-1, 1)).reshape(weighted_ensemble_shape)

rank_avg_shape = top3_rank_ensemble.shape
rank_avg_top_3_predictions = le.inverse_transform(top3_rank_ensemble.reshape(-1, 1)).reshape(rank_avg_shape)

stacking_shape = top3_stack.shape
stacking_top_3_predictions = le.inverse_transform(top3_stack.reshape(-1, 1)).reshape(stacking_shape)

submission['Fertilizer Name'] = [' '.join(each) for each in xgb_top_3_predictions]
submission.to_csv('xgb_submission.csv', index = False)
submission['Fertilizer Name'] = [' '.join(each) for each in lgb_top_3_predictions]
submission.to_csv('lgb_submission.csv', index = False)
submission['Fertilizer Name'] = [' '.join(each) for each in hgb_top_3_predictions]
submission.to_csv('hgb_submission.csv', index = False)
submission['Fertilizer Name'] = [' '.join(each) for each in avg_ensemble_top_3_predictions]
submission.to_csv('avg_ensemble_submission.csv', index = False)
submission['Fertilizer Name'] = [' '.join(each) for each in weighted_ensemble_top_3_predictions]
submission.to_csv('weighted_ensemble_submission.csv', index = False)
submission['Fertilizer Name'] = [' '.join(each) for each in rank_avg_top_3_predictions]
submission.to_csv('rank_avg_ensemble_submission.csv', index = False)
submission['Fertilizer Name'] = [' '.join(each) for each in stacking_top_3_predictions]
submission.to_csv('stacking_ensemble_submission.csv', index = False)

