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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, cohen_kappa_score, confusion_matrix, mean_squared_error, mean_squared_log_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


#loading dataset
train = pd.read_csv('../input/playground-series-s5e7/train.csv', index_col = 'id')
test = pd.read_csv('../input/playground-series-s5e7/test.csv', index_col = 'id')
#original = pd.read_csv("")
submission = pd.read_csv('../input/playground-series-s5e7/sample_submission.csv')

print('The dimession of the train dataset is:', train.shape)
#print('The dimession of the original dataset is:', original.shape)
print('The dimession of the test dataset is:', test.shape)


train.info()


test.info()


train.describe()


test.describe()


print('Train duplicates: ', train.duplicated().sum())
#print('original duplicates: ', original.duplicated().sum())
print('Test duplicates: ', test.duplicated().sum())


train.isna().sum()


test.isna().sum()


# Create the profile report
profile = ProfileReport(train, title="Pandas Profiling Report", explorative=True)
profile.to_notebook_iframe()


# numeric columns
fig, ax = plt.subplots(1, 5, figsize=(18, 8))

sns.countplot(data=train, x='Time_spent_Alone', hue = 'Personality', ax=ax[0])
sns.countplot(data=train, x='Social_event_attendance', hue = 'Personality', ax=ax[1])
sns.countplot(data=train, x='Going_outside', hue = 'Personality', ax=ax[2])
sns.countplot(data=train, x='Friends_circle_size', hue = 'Personality', ax=ax[3])
sns.countplot(data=train, x='Post_frequency', hue = 'Personality', ax=ax[4])

plt.tight_layout()
plt.show();


fig, ax = plt.subplots(1, 3, figsize=(18, 8))

sns.countplot(data=train, x='Stage_fear', hue='Personality', ax=ax[0])
sns.countplot(data=train, x='Drained_after_socializing', hue='Personality', ax=ax[1])
sns.countplot(data=train, x='Personality', ax=ax[2])

plt.tight_layout()
plt.show();


fig, axes = plt.subplots(1, 2, figsize = (20, 8))

corr_matrix_train = train.drop(columns = 'Personality').corr(numeric_only=True)
corr_filtered_train = corr_matrix_train[np.abs(corr_matrix_train) > 0.2]
# Create a heatmap
sns.heatmap(corr_filtered_train, cmap='coolwarm', annot = True, ax = axes[0]);

corr_matrix_test = test.corr(numeric_only=True)
corr_filtered_test = corr_matrix_test[np.abs(corr_matrix_test) > 0.2]
# Create a heatmap
sns.heatmap(corr_filtered_test, cmap='coolwarm', annot = True, ax = axes[1]);


numeric_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

cluster_data = train.select_dtypes(include=np.number).copy()
cluster_data_test = test.select_dtypes(include=np.number).copy()

cluster_data[numeric_cols] = cluster_data[numeric_cols].fillna(cluster_data[numeric_cols].median())
cluster_data_test[numeric_cols] = cluster_data_test[numeric_cols].fillna(cluster_data_test[numeric_cols].median())

# Step 2: Scale the data (fit scaler on train, transform both train and test)
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)
cluster_scaled_test = scaler.transform(cluster_data_test)  # Use transform only on test!

# Step 3: Elbow Method to find optimal k
inertia = []
k_range = range(1, 20)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(cluster_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
train_labels = kmeans.fit_predict(cluster_scaled)
test_labels = kmeans.predict(cluster_scaled_test)  # predict on test data

# Add cluster labels to original DataFrames
train['cluster'] = train_labels
test['cluster'] = test_labels

# Step 5: PCA reduction for visualization on train data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(cluster_scaled)

plt.figure(figsize=(8, 6))
for i in range(optimal_k):
    plt.scatter(X_pca[train_labels == i, 0], X_pca[train_labels == i, 1], label=f'Cluster {i}')
    
# Project cluster centers to PCA space for plotting
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
            c='black', marker='X', s=200, label='Centroids')

plt.title('KMeans Clustering (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()


y = train["Personality"].values  # "Extrovert" or "Introvert"

# Unique values in y (ensure order is consistent)
class_names = ["Introvert", "Extrovert"]
class_markers = {'Introvert': 's', 'Extrovert': 'o'}
cluster_colors = ['red', 'blue', 'green', 'purple', 'orange']

plt.figure(figsize=(10, 6))

for cluster in range(optimal_k):
    for class_label in class_names:
        idx = np.where((train_labels == cluster) & (y == class_label))
        plt.scatter(
            X_pca[idx, 0], X_pca[idx, 1],
            c=cluster_colors[cluster % len(cluster_colors)],
            marker=class_markers[class_label],
            label=f'Cluster {cluster} - {class_label}',
            alpha=0.6,
            edgecolors='k'
        )

plt.title("KMeans Clusters with Introvert/Extrovert Overlay (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(loc='best', fontsize=8)
plt.grid(True)
plt.show()


train["cluster"] = train_labels

# Display cross-tabulation of clusters and class labels
cluster_class_counts = pd.crosstab(train["cluster"], train["Personality"])

print(cluster_class_counts)


numeric_cols = ['Time_spent_Alone','Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_cols = ["Stage_fear", "Drained_after_socializing"]


def fill_mode(col):
    mode = col.mode()
    if not mode.empty:
        return col.fillna(mode.iloc[0])
    else:
        return col.fillna("Unknown")


# Baseline Modeling
# defining x and y
X = train.drop(columns = 'Personality')
for col in cat_cols:
    X[col] = X[col].fillna("missing").astype('category')
X["Stage_fear"] = X["Stage_fear"].map({"No": 0, "Yes": 1, "missing": 2}).astype(int)
X["Drained_after_socializing"] = X["Drained_after_socializing"].map({"No": 0, "Yes": 1, "missing": 2}).astype(int)
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

le = LabelEncoder()
y = train['Personality']
y_encoded = le.fit_transform(y)

test_base = test
for col in cat_cols:
    test_base[col] = test_base[col].fillna("missing").astype('category')
test_base["Stage_fear"] = test_base["Stage_fear"].map({"No": 0, "Yes": 1, "missing": 2}).astype(int)
test_base["Drained_after_socializing"] = test_base["Drained_after_socializing"].map({"No": 0, "Yes": 1, "missing": 2}).astype(int)
test_base[numeric_cols] = test_base[numeric_cols].fillna(test_base[numeric_cols].median())


rf_acc, ada_acc, gb_acc, hgb_acc, xgb_acc, lgb_acc, cat_acc = list(), list(), list(), list(), list(), list(), list()
rf_preds, ada_preds, gb_preds, hgb_preds, xgb_preds, lgb_preds, cat_preds = list(), list(), list(), list(), list(), list(), list()
rf_cv_score, ada_cv_score, gb_cv_score, hgb_cv_score, xgb_cv_score, lgb_cv_score, cat_cv_score = list(), list(), list(), list(), list(), list(), list()
rf_imp, ada_imp, gb_imp, xgb_imp, lgb_imp, cat_imp = list(), list(), list(), list(), list(), list()

## Running 5 times CV
for i in range(5):
    print('loop: ', i+1)
    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for fold, (train_ix, test_ix) in enumerate(skf.split(X, y_encoded)):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y_encoded[train_ix], y_encoded[test_ix]
    
        ## Building RF model ##
        rf_md = RandomForestClassifier(n_estimators = 289, max_depth = 7, min_samples_split = 4, min_samples_leaf = 2).fit(X_train, y_train)
        # extracting feature importances
        rf_imp.append(rf_md.feature_importances_)
        ## Predicting on X_test and test
        rf_pred_1 = rf_md.predict(X_test)
        rf_pred_2 = rf_md.predict(test_base)

        
        ## Building RF model ##
        ada_md = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 4), learning_rate = 0.005, n_estimators = 758).fit(X_train, y_train)
        # extracting feature importances
        ada_imp.append(ada_md.feature_importances_)
        ## Predicting on X_test and test
        ada_pred_1 = ada_md.predict(X_test)
        ada_pred_2 = ada_md.predict(test_base)

        
        ## Building gb model ##
        gb_md = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 459, max_depth = 4, 
                                          min_samples_split = 12, min_samples_leaf = 9).fit(X_train, y_train)
        # extracting feature importances
        gb_imp.append(gb_md.feature_importances_)
        ## Predicting on X_test and test
        gb_pred_1 = gb_md.predict(X_test)
        gb_pred_2 = gb_md.predict(test_base)

        
        ## Building hgb model ##
        hgb_md = HistGradientBoostingClassifier(max_iter = 639, 
                                               learning_rate = 0.08, 
                                               l2_regularization = 0.05, 
                                               min_samples_leaf = 13, 
                                               max_leaf_nodes = 66, 
                                               max_depth = 7, 
                                               max_bins = 241).fit(X_train, y_train)
        ## Predicting on X_test and test
        hgb_pred_1 = hgb_md.predict(X_test)
        hgb_pred_2 = hgb_md.predict(test_base)

        
        ## Building xgb model ##
        xgb_md = XGBClassifier(learning_rate = 0.08, 
                              n_estimators = 976, 
                              max_depth = 7, 
                              min_child_weight = 40, 
                              gamma = 0.018854387471939887, 
                              alpha = 0.2227027545609019, 
                              colsample_bytree = 0.3391547521532657, 
                              subsample = 0.5342124813812622).fit(X_train, y_train)
        # extracting feature importances
        xgb_imp.append(xgb_md.feature_importances_)
        ## Predicting on X_test and test
        xgb_pred_1 = xgb_md.predict(X_test)
        xgb_pred_2 = xgb_md.predict(test_base)

        
        ## Building lgb model ##
        lgb_md = LGBMClassifier(n_estimators = 264, 
                               learning_rate = 0.1, 
                               num_leaves = 435, 
                               min_data_in_leaf = 145, 
                               min_child_weight = 0.026057381417051245, 
                               max_depth = 3, 
                               bagging_fraction = 0.5715184704479637, 
                               feature_fraction = 0.9519812966118504, 
                               lambda_l1 = 1.0112358932442056, 
                               lambda_l2 = 0.737167992862996,
                               verbose = -1).fit(X_train, y_train)
        # extracting feature importances
        lgb_imp.append(lgb_md.feature_importances_)
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict(X_test)
        lgb_pred_2 = lgb_md.predict(test_base)
        
        ## Building cat model ##
        cat_md = CatBoostClassifier(iterations = 639, 
                                   learning_rate = 0.08, 
                                   min_data_in_leaf = 82, 
                                   depth = 5, 
                                   random_strength = 0.07457693113220128, 
                                   bagging_temperature = 0.3082760691709291, 
                                   border_count = 44, 
                                   l2_leaf_reg = 100, 
                                   verbose = False).fit(X_train, y_train)
        # extracting feature importances
        cat_imp.append(cat_md.feature_importances_)
        ## Predicting on X_test and test
        cat_pred_1 = cat_md.predict(X_test)
        cat_pred_2 = cat_md.predict(test_base)
        
        ## Computing metrics
        rf_acc.append(accuracy_score(y_test, rf_pred_1))
        ada_acc.append(accuracy_score(y_test, ada_pred_1))
        gb_acc.append(accuracy_score(y_test, gb_pred_1))
        hgb_acc.append(accuracy_score(y_test, hgb_pred_1))
        xgb_acc.append(accuracy_score(y_test, xgb_pred_1))
        lgb_acc.append(accuracy_score(y_test, lgb_pred_1))
        cat_acc.append(accuracy_score(y_test, cat_pred_1))
        
        ## test preds
        rf_preds.append(rf_pred_2)
        ada_preds.append(ada_pred_2)
        gb_preds.append(gb_pred_2)
        hgb_preds.append(hgb_pred_2)
        xgb_preds.append(xgb_pred_2)
        lgb_preds.append(lgb_pred_2)
        cat_preds.append(cat_pred_2)
        
    clear_output()

rf_cv_score = np.mean(rf_acc)
ada_cv_score = np.mean(ada_acc)
gb_cv_score = np.mean(gb_acc)
hgb_cv_score = np.mean(hgb_acc)
xgb_cv_score = np.mean(xgb_acc)
lgb_cv_score = np.mean(lgb_acc)
cat_cv_score = np.mean(cat_acc)

print('The accuracy_score of the rf model over 5-folds (run 5 times) is:', rf_cv_score)
print('The accuracy_score of the ada model over 5-folds (run 5 times) is:', ada_cv_score)
print('The accuracy_score of the gb model over 5-folds (run 5 times) is:', gb_cv_score)
print('The accuracy_score of the hgb model over 5-folds (run 5 times) is:', hgb_cv_score)
print('The accuracy_score of the xgb model over 5-folds (run 5 times) is:', xgb_cv_score)
print('The accuracy_score of the lgb model over 5-folds (run 5 times) is:', lgb_cv_score)
print('The accuracy_score of the cat model over 5-folds (run 5 times) is:', cat_cv_score)


estimators = [
    ('rf', RandomForestClassifier(n_estimators=289, max_depth=7, min_samples_split=4, min_samples_leaf=2)),
    ('ada', AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4), learning_rate=0.005, n_estimators=758)),
    ('gb', GradientBoostingClassifier(learning_rate=0.02, n_estimators=459, max_depth=4, min_samples_split=12, min_samples_leaf=9)),
    ('hgb', HistGradientBoostingClassifier(max_iter=639, learning_rate=0.08, l2_regularization=0.05, min_samples_leaf=13, max_leaf_nodes=66, max_depth=7, max_bins=241)),
    ('xgb', XGBClassifier(learning_rate=0.08, n_estimators=976, max_depth=7, min_child_weight=40, gamma=0.01885,
                          alpha=0.2227, colsample_bytree=0.3391, subsample=0.5342,
                          use_label_encoder=False, eval_metric='logloss', verbosity=0)),
    ('lgb', LGBMClassifier(n_estimators=264, learning_rate=0.1, num_leaves=435, min_data_in_leaf=145,
                           min_child_weight=0.026, max_depth=3, bagging_fraction=0.5715,
                           feature_fraction=0.952, lambda_l1=1.01, lambda_l2=0.737,verbose = -1)),
    ('cat', CatBoostClassifier(iterations=639, learning_rate=0.08, depth=5, min_data_in_leaf=82,
                               random_strength=0.0745, bagging_temperature=0.308,
                               border_count=44, l2_leaf_reg=100, verbose=0))
]

voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

voting_scores = cross_val_score(voting_clf, X, y_encoded, cv=5, scoring='accuracy')
print("Voting Classifier CV Accuracy:", voting_scores.mean())

voting_clf.fit(X, y_encoded)
vote_preds = voting_clf.predict(test_base)


meta_model = LogisticRegression(max_iter=1000)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

stacking_scores = cross_val_score(stacking_clf, X, y_encoded, cv=5, scoring='accuracy')
print("Stacking Classifier CV Accuracy:", stacking_scores.mean())

stacking_clf.fit(X, y_encoded)
stack_preds = stacking_clf.predict(test_base)


stacking_ridge_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=RidgeClassifier(alpha=1.0),
    cv=5,
    n_jobs=-1,
)

stacking_scores_ridge = cross_val_score(stacking_ridge_clf, X, y_encoded, cv=5, scoring='accuracy')
print("Stacking Classifier (Ridge) CV Accuracy:", stacking_scores_ridge.mean())

stacking_ridge_clf.fit(X, y_encoded)
stack_preds_ridge = stacking_ridge_clf.predict(test_base)


# Plotting importances
data1 = pd.DataFrame(pd.DataFrame(rf_imp, columns = X.columns).apply(np.mean, axis = 0))
data1['Feature'] = data1.index
data1.columns = ['RF Score', 'Feature']
data1.reset_index(drop = True, inplace = True)
data1.sort_values(by = 'RF Score', ascending = False, inplace = True)

data2 = pd.DataFrame(pd.DataFrame(ada_imp, columns = X.columns).apply(np.mean, axis = 0))
data2['Feature'] = data2.index
data2.columns = ['Ada Score', 'Feature']
data2.reset_index(drop = True, inplace = True)
data2.sort_values(by = 'Ada Score', ascending = False, inplace = True)

data3 = pd.DataFrame(pd.DataFrame(gb_imp, columns = X.columns).apply(np.mean, axis = 0))
data3['Feature'] = data3.index
data3.columns = ['Gb Score', 'Feature']
data3.reset_index(drop = True, inplace = True)
data3.sort_values(by = 'Gb Score', ascending = False, inplace = True)

data4 = pd.DataFrame(pd.DataFrame(xgb_imp, columns = X.columns).apply(np.mean, axis = 0))
data4['Feature'] = data4.index
data4.columns = ['XGB Score', 'Feature']
data4.reset_index(drop = True, inplace = True)
data4.sort_values(by = 'XGB Score', ascending = False, inplace = True)

data5 = pd.DataFrame(pd.DataFrame(lgb_imp, columns = X.columns).apply(np.mean, axis = 0))
data5['Feature'] = data5.index
data5.columns = ['LightGBM Score', 'Feature']
data5.reset_index(drop = True, inplace = True)
data5.sort_values(by = 'LightGBM Score', ascending = False, inplace = True)

data6 = pd.DataFrame(pd.DataFrame(cat_imp, columns = X.columns).apply(np.mean, axis = 0))
data6['Feature'] = data6.index
data6.columns = ['CatBoost Score', 'Feature']
data6.reset_index(drop = True, inplace = True)
data6.sort_values(by = 'CatBoost Score', ascending = False, inplace = True)

fig, axes = plt.subplots(6, 1, figsize = (15, 55))
sns.barplot(ax = axes[0], data = data1, x = 'RF Score', y = 'Feature', color = 'steelblue')
sns.barplot(ax = axes[1], data = data2, x = 'Ada Score', y = 'Feature', color = 'steelblue')
sns.barplot(ax = axes[2], data = data3, x = 'Gb Score', y = 'Feature', color = 'steelblue')
sns.barplot(ax = axes[3], data = data4, x = 'XGB Score', y = 'Feature', color = 'steelblue')
sns.barplot(ax = axes[4], data = data5, x = 'LightGBM Score', y = 'Feature', color = 'steelblue')
sns.barplot(ax = axes[5], data = data6, x = 'CatBoost Score', y = 'Feature', color = 'steelblue');


# Creating submissions
rf_preds_test = pd.DataFrame(rf_preds).apply(np.mean, axis = 0)
ada_preds_test = pd.DataFrame(ada_preds).apply(np.mean, axis = 0)
gb_preds_test = pd.DataFrame(gb_preds).apply(np.mean, axis = 0)
hgb_preds_test = pd.DataFrame(hgb_preds).apply(np.mean, axis = 0)
lgb_preds_test = pd.DataFrame(lgb_preds).apply(np.mean, axis = 0)
xgb_preds_test = pd.DataFrame(xgb_preds).apply(np.mean, axis = 0)
cat_preds_test = pd.DataFrame(cat_preds).apply(np.mean, axis = 0)
vote_preds_test = pd.DataFrame(vote_preds)
stack_preds_test = pd.DataFrame(stack_preds)
stack_preds_ridge_test = pd.DataFrame(stack_preds_ridge)

# Convert indices to original class names
rf_preds_test = le.inverse_transform(rf_preds_test.astype(int))
ada_preds_test = le.inverse_transform(ada_preds_test.astype(int))
gb_preds_test = le.inverse_transform(gb_preds_test.astype(int))
hgb_preds_test = le.inverse_transform(hgb_preds_test.astype(int))
lgb_preds_test = le.inverse_transform(lgb_preds_test.astype(int))
xgb_preds_test = le.inverse_transform(xgb_preds_test.astype(int))
cat_preds_test = le.inverse_transform(cat_preds_test.astype(int))
vote_preds_test = le.inverse_transform(vote_preds_test.astype(int))
stack_preds_test = le.inverse_transform(stack_preds_test.astype(int))
stack_preds_ridge_test = le.inverse_transform(stack_preds_ridge_test.astype(int))

submission['Personality'] = rf_preds_test
submission.to_csv('rf_base_submission.csv', index = False)

submission['Personality'] = ada_preds_test
submission.to_csv('ada_base_submission.csv', index = False)

submission['Personality'] = gb_preds_test
submission.to_csv('gb_base_submission.csv', index = False)

submission['Personality'] = hgb_preds_test
submission.to_csv('hgb_base_submission.csv', index = False)

submission['Personality'] = xgb_preds_test
submission.to_csv('xgb_base_submission.csv', index = False)

submission['Personality'] = lgb_preds_test
submission.to_csv('lgb_base_submission.csv', index = False)

submission['Personality'] = cat_preds_test
submission.to_csv('cat_base_submission.csv', index = False)

submission['Personality'] = vote_preds_test
submission.to_csv('vote_base_submission.csv', index = False)

submission['Personality'] = stack_preds_test
submission.to_csv('stack_base_submission.csv', index = False)

submission['Personality'] = stack_preds_ridge_test
submission.to_csv('stack_ridge_base_submission.csv', index = False)

