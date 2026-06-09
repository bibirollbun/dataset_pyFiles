# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv'), pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv'), pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
train, test = train.drop(columns="id"), test.drop(columns="id")


train, test = pd.get_dummies(train, columns=['road_type', 'weather', 'time_of_day', 'lighting']), pd.get_dummies(test, columns=['road_type', 'weather', 'time_of_day', 'lighting'])


train_sample, test_sample = train.sample(5000), test.sample(5000)
y_sample = train_sample['accident_risk']


import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')
x_train_sample = train_sample['accident_risk']
fig, ax = plt.subplots()
ax.hist(x_train_sample, bins=8, linewidth=0.5, edgecolor="white")
plt.show()


scaler_features = train_sample.select_dtypes(include=["int64", "float64"])
scal_feats = [feat for feat in scaler_features.columns]


import seaborn as sns

train_melted = train_sample.melt(
    value_vars=scal_feats,   
    var_name="Variable",     # new column with the column names
    value_name="Value"       # new column with the values
)
print(train_melted)


# Horizontal boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_melted, x="Value", y="Variable", orient="h")
plt.show()


feats = train.select_dtypes(include=["int64", "float64"])
train_m = train.melt(
    value_vars=feats,   
    var_name="Variable",     
    value_name="Value"       
)
feats_t = test.select_dtypes(include=["int64","float64"])
test_m = test.melt(
    value_vars=feats_t,   
    var_name="Variable",     
    value_name="Value"      
)
plt.figure(figsize=(8, 5))
plt.title(label="Train")
sns.boxplot(data=train_m, x="Value", y="Variable", orient="h")
plt.show()


plt.figure(figsize=(8, 5))
plt.title(label="Test")
sns.boxplot(data=test_m, x="Value", y="Variable", orient="h")
plt.show()


print('len_train',len(train_m),'len_test', len(test_m))


Q3_train = train['num_reported_accidents'].quantile(0.75)
Q1_train =  train['num_reported_accidents'].quantile(0.25) 
print('Q3: ',Q3_train, 'Q1: ', Q1_train)
IQR_train = Q3_train - Q1_train
lower, upper = Q1_train -(1.5 * IQR_train), Q3_train + (1.5 * IQR_train) 
print('IQRtrain: ', IQR_train, 'lower: ', lower, 'upper: ',upper)


Q3_test = test['num_reported_accidents'].quantile(0.75)
Q1_test =  test['num_reported_accidents'].quantile(0.25) 
print('Q3: ',Q3_test, 'Q1: ', Q1_test)
IQR_test = Q3_test - Q1_test
lower_t, upper_t = Q1_test -(1.5 * IQR_test), Q3_test + (1.5 * IQR_test) 
print('IQR_test: ', IQR_test, 'lower: ', lower_t, 'upper: ',upper_t)


outlier_15_low_train = (train['num_reported_accidents'] < lower)
outliers_75_up_train = (train['num_reported_accidents'] > upper)
outlier_15_low_test = (test['num_reported_accidents'] < lower_t)
outliers_75_up_test = (test['num_reported_accidents'] > upper_t)


print('with outliers')
print('train: ', len(train))
print('test: ', len(test))
print('how many rows we will lose')
print('train \n',train['num_reported_accidents'][(outlier_15_low_train|outliers_75_up_train)])
print('test \n',test['num_reported_accidents'][(outlier_15_low_test|outliers_75_up_test)])



print('train_new \n',train['num_reported_accidents'][~(outlier_15_low_train|outliers_75_up_train)])
print('test_new \n',test['num_reported_accidents'][~(outlier_15_low_test|outliers_75_up_test)])
print(train.dtypes)


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import ipywidgets as widgets

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(train)


train.head()


corr_mat = train.corr()
plt.figure(figsize=(12, 10))  # Width=12, Height=10 inches
sns.heatmap(corr_mat, annot=True,  linewidth=0.3, fmt=".2f");
plt.imshow


y = train['accident_risk']
train = train.drop(columns="accident_risk")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
test_scaled = scaler.fit_transform(test)


def pca_analysis(n_components):
    pca = PCA(n_components=n_components)
    pca.fit_transform(x_scaled)
    varianza = pca.explained_variance_ratio_
    sing_valu = pca.singular_values_
    print(f"Varianza explicada por PC1: {varianza[0]:.2%}")
    print(f"Varianza explicada por PC2: {varianza[1]:.2%}")
    print(f"Varianza explicada por PC3: {varianza[2]:.2%}")
    print(f"Varianza explicada por PC4: {varianza[3]:.2%}")
    print(f"Varianza total explicada: {varianza.sum():.2%}")
    print(f"Singular values: {sing_valu}")
    return pca
slider = widgets.IntSlider(value=2, min=4, max=20, step=1, description="Components:")
interactive_plot = widgets.interactive(pca_analysis, n_components=slider)
display(interactive_plot)



import matplotlib.pyplot as plt
x_train_pca = pca_analysis(16).fit_transform(x_scaled)
x_test_pca = pca_analysis(16).fit_transform(x_test_scaled)
test_pca = pca_analysis(16).fit_transform(test_scaled)
pca_1_sample = x_train_pca[:500, 0]
pca_2_sample = x_train_pca[:500, 1]
y_train_pca_sample = y_train[:500]
plt.figure(figsize=(12, 10))
plt.scatter(pca_1_sample, pca_2_sample, c=y_train_pca_sample, cmap='viridis', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA projection (first 500 samples)')
plt.imshow


from sklearn import metrics
from scipy.spatial.distance import cdist

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 21)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(x_train_pca)
    
    distortions.append(sum(np.min(cdist(x_train_pca, kmeanModel.cluster_centers_, 'euclidean'), axis=1)**2) / x_train_pca.shape[0])
    
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = distortions[-1]
    mapping2[k] = inertias[-1]



print("Distortion values:")
for key, val in mapping1.items():
    print(f'{key} : {val}')
plt.figure(figsize=(10,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


# List of column names to create (pca_1 to pca_16)
pca_cols = [f'pca_{i}' for i in range(1, 17)]

# Add the PCA components to the x_train DataFrame
for i, col_name in enumerate(pca_cols):
    x_train[col_name] = x_train_pca[:, i]

# Add the PCA components to the x_test DataFrame
for i, col_name in enumerate(pca_cols):
    x_test[col_name] = x_test_pca[:, i]

# Add the PCA components to the test DataFrame
for i, col_name in enumerate(pca_cols):
    test[col_name] = test_pca[:, i]

X = x_train[pca_cols]
X_test = x_test[pca_cols]
Test = test[pca_cols]

print("X (Training Features) Head:")
print(X.head())
print("\nFeature columns in X:")
print(X.columns.tolist())


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=21, random_state=42)

x_train['cluster_pca'] = kmeans.fit_predict(X)

sns.scatterplot(x='pca_1', y='pca_2', hue='cluster_pca', palette='Set2', data=x_train)
plt.title("Clusters de pca_train")
plt.xlabel("pca_1")
plt.ylabel("pca_2")
plt.show()


x_test['cluster_pca'] = kmeans.predict(X_test)

sns.scatterplot(x='pca_1', y='pca_2', hue='cluster_pca', palette='Set2', data=x_test)
plt.title("Clusters de pca_test")
plt.xlabel("pca_1")
plt.ylabel("pca_2")
plt.show()

test['cluster_pca'] = kmeans.predict(Test)
sns.scatterplot(x='pca_1', y='pca_2', hue='cluster_pca', palette='Set2', data=test)
plt.title("Clusters de test")
plt.xlabel("pca_1")
plt.ylabel("pca_2")
plt.show()


x_train['accident_risk'] = y_train
corr_mat_x_train = x_train.corr()
plt.figure(figsize=(12, 10))  # Width=12, Height=10 inches
sns.heatmap(corr_mat_x_train, annot=True,  linewidth=0.3, fmt=".2f");
plt.imshow


x_train = x_train.drop(columns='accident_risk')
X = x_train
X_test = x_test
Test = test

kmeans = KMeans(n_clusters=21, random_state=42)

x_train['cluster_all'] = kmeans.fit_predict(X)

sns.scatterplot(x='speed_limit', y='curvature', hue='cluster_all', palette='Set2', data=x_train)
plt.title("Clusters de pca_train")
plt.xlabel("speed_limit")
plt.ylabel("curvature")
plt.show()


x_test['cluster_all'] = kmeans.predict(X_test)

sns.scatterplot(x='speed_limit', y='curvature', hue='cluster_all', palette='Set2', data=x_test)
plt.title("Clusters de pca_test")
plt.xlabel('speed_limit')
plt.ylabel('curvature')
plt.show()

test['cluster_all'] = kmeans.predict(Test)
sns.scatterplot(x='speed_limit', y='curvature', hue='cluster_all', palette='Set2', data=test)
plt.title("Clusters de test")
plt.xlabel('speed_limit')
plt.ylabel('curvature')
plt.show()


x_train['accident_risk'] = y_train
corr_mat_x_train = x_train.corr()
plt.figure(figsize=(12, 10))  # Width=12, Height=10 inches
sns.heatmap(corr_mat_x_train, annot=True,  linewidth=0.3, fmt=".2f");
plt.imshow


drop_columns = ['time_of_day_morning', 'time_of_day_evening', 'time_of_day_afternoon', 'road_type_urban', 'road_type_rural', 'road_type_highway', 'school_season', 'holiday', 'public_road', 'road_signs_present', 'num_lanes']
x_train = x_train.drop(columns='accident_risk')
x_train = x_train.drop(columns =drop_columns)
x_test = x_test.drop(columns=drop_columns)
test = test.drop(columns = drop_columns)
x_train['curvature_over_speed'] = x_train['curvature'] / (x_train['speed_limit'] + 1e-5)
x_test['curvature_over_speed'] = x_test['curvature'] / (x_test['speed_limit'] + 1e-5)
test['curvature_over_speed'] = test['curvature'] / (test['speed_limit'] + 1e-5)

# interaction between curvature and reported accidents
x_train['curv_accidents'] = x_train['curvature'] * x_train['num_reported_accidents']
x_test['curv_accidents'] = x_test['curvature'] * x_test['num_reported_accidents']
test['curv_accidents'] = test['curvature'] * test['num_reported_accidents']

# ratio of night lighting influence over total lighting
x_train['night_ratio'] = x_train['lighting_night'] / (x_train['lighting_daylight'] + x_train['lighting_dim'] + 1e-5)
x_test['night_ratio'] = x_test['lighting_night'] / (x_test['lighting_daylight'] + x_test['lighting_dim'] + 1e-5)
test['night_ratio'] = test['lighting_night'] / (test['lighting_daylight'] + test['lighting_dim'] + 1e-5)
x_train['log_curvature'] = np.log1p(x_train['curvature'])
x_test['log_curvature'] = np.log1p(x_test['curvature'])
test['log_curvature'] = np.log1p(test['curvature'])


x_train['curv_times_speed']= x_train['curvature'] * x_train['speed_limit']
x_test['curv_times_speed'] = x_test['curvature'] * x_test['speed_limit']
test['curv_times_speed']= test['curvature'] * test['speed_limit']
x_train['accident_powered'] = x_train['curv_times_speed']**x_train['num_reported_accidents']
x_test['accident_powered']=x_test['curv_times_speed']**x_test['num_reported_accidents']
test['accident_powered'] = test['curv_times_speed']**test['num_reported_accidents']


print(x_train.isna().sum())
print(x_test.isna().sum())
print(test.isna().sum())


scaler_X = StandardScaler()
x_train_scaled = scaler_X.fit_transform(x_train)
x_test_scaled = scaler_X.transform(x_test)
test_scaled = scaler_X.transform(test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

print(x_train_scaled, x_train_scaled.shape, x_train.shape)


# Check for any infinite values in the training data
x_np, x_test_np, test_np, y_np, y_test_np = (
    np.array(x_train_scaled), np.array(x_test_scaled),
    np.array(test_scaled), np.array(y_train_scaled),
    np.array(y_test_scaled)
)


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import gc

def k_fold_func(model, x_np, y_np):
    # Metrics lists
    rmse_list = []
    r2_list = []
    
    # Define KFold
    k_folds = KFold(n_splits=4, shuffle=True, random_state=42)
    models = {}
    # --- Cross-validation loop ---
    for fold, (train_idx, val_idx) in enumerate(k_folds.split(x_np)):
        X_train_fold, X_val_fold = x_np[train_idx], x_np[val_idx]
        y_train_fold, y_val_fold = y_np[train_idx], y_np[val_idx]
        
        # Fit Ridge model
        print(f'model fit started with fold {fold}')
        model_ = model.fit(X_train_fold, y_train_fold)
        
        # Predict
        print(f'predictions started in {fold}')
        y_pred = model_.predict(X_val_fold)
        # Metrics
        mse = mean_squared_error(y_val_fold, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_fold, y_pred)
        rmse_list.append(rmse)
        r2_list.append(r2)
        print(f'r2: {r2}, rmse: {rmse} ')
        models[f'{fold}'] = model_
        models[f'{fold}_y_pred'] = y_pred
        models[f'{fold}_rmse'] = rmse
        models[f'{fold}_r2'] = r2
        models[f'{fold}_idx'] = val_idx
        del model_, X_train_fold, X_val_fold, y_train_fold, y_val_fold, y_pred
        gc.collect()
        
    model_all = model.fit(x_np, y_np)
    # Free memory for the current model
        
    # --- Mean metrics across folds ---
    print("\nAverage results across folds:")
    print(f"Mean RMSE: {np.mean(rmse_list):.4f}")
    print(f"Mean R²:  {np.mean(r2_list):.4f}")
    return models, model_all


def differens(y_test_np, y_hat):
    diff = np.subtract(y_test_np, y_hat).flatten()
    plt.figure(figsize=(8,4))
    plt.plot(y_test_np[:500], label="True Values", marker='o', color='b')
    plt.plot(y_hat[:500], label="Predicted Values", marker='x', color='r')
    plt.bar(range(len(diff)), diff, alpha=0.4, label="Difference (True - Pred)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.title("Differences using np.subtract")
    plt.xlabel("Index")
    plt.ylabel("Value / Difference")
    plt.show()


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor, HuberRegressor
cv = 10
# LassoCV to pick features
lasso = LassoCV(cv=cv, random_state=42, max_iter=10000).fit(x_np, y_np.ravel())
selector = SelectFromModel(lasso, prefit=True, threshold='mean')  # or a chosen threshold

X_reduced = selector.transform(x_np)
X_test_reduced = selector.transform(x_test_np)
test_reduced = selector.transform(test_np)
print("Selected features:", X_reduced.shape)


num_features = X_reduced.shape[1]
pca_cols = [f'pca_{i}' for i in range(1, num_features + 1)]
X_df = pd.DataFrame(X_reduced, columns=pca_cols)
X_df['accident_risk'] = y_np.flatten()


plt.figure(figsize=(12, 10))  # Width=12, Height=10 inches
sns.heatmap(X_df.corr(), annot=True,  linewidth=0.3, fmt=".2f");
plt.imshow


huber_model = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)
estimator = RandomForestRegressor(
    n_estimators=120,
    max_depth=5,
    random_state=42
)

ransac_model = RANSACRegressor(
    estimator=estimator,
    min_samples = 50,
    max_trials=100,
    residual_threshold=None,   # auto-estimate
    loss='absolute_error',
    random_state=42
)
huber_models, huber_model_all= k_fold_func(huber_model, X_reduced, y_np)
ransac_models, ransac_model_all = k_fold_func(ransac_model, X_reduced.astype(np.float32), y_np.astype(np.float32).ravel())
y_test_val_huber = huber_model_all.predict(X_test_reduced)
y_test_val_ransac = ransac_model_all.predict(X_test_reduced.astype(np.float32))
mse_huber = mean_squared_error(y_test_val_huber, y_test_np)
rmse_huber = np.sqrt(mse_huber)
mse_ransac = mean_squared_error(y_test_val_ransac, y_test_np)
rmse_ransac = np.sqrt(mse_ransac)


print(f'rmse_huber: {rmse_huber} , rmse_ransac: {rmse_ransac}')


test_huber = huber_model_all.predict(test_reduced)
submission['accident_risk'] = scaler_y.inverse_transform(test_huber.reshape(-1,1))
submission.to_csv('huber_predictor.csv', index=False)
test_ransac = ransac_model_all.predict(test_reduced.astype(np.float32))
submission['accident_risk'] = scaler_y.inverse_transform(test_ransac.reshape(-1,1))
submission.to_csv('ransac_predictor.csv', index=False)


ridge_models, ridge_model_all = k_fold_func(Ridge(alpha=10), X_reduced, y_np)


random_forest_model = RandomForestRegressor(n_estimators=250, criterion='squared_error', max_depth=5, min_samples_split=5, min_samples_leaf=5)
random_models, random_model_all = k_fold_func(random_forest_model, X_reduced.astype(np.float32), y_np.astype(np.float32).ravel())


import xgboost 
params = {'objective': 'reg:squarederror','n_estimators':1000 , 'learning_rate': 0.01, 'max_depth': 10, 'random_state': 42}
xgb = xgboost.XGBRegressor(**params)
xgb_models, xgb_model_all = k_fold_func(xgb, X_reduced.astype(np.float32), y_np.astype(np.float32))


# unpack all models from hashtable models, random_models, xgb_models
# all prediction of this dictionary were made on X_test_reduced
ridge_model = ridge_models['0']
ridge_model_1 = ridge_models['1']
ridge_model_2 = ridge_models['2']
ridge_model_3 = ridge_models['3']
y_pred = ridge_models['0_y_pred']
y_pred_1 = ridge_models['1_y_pred']
y_pred_2 = ridge_models['2_y_pred']
y_pred_3 = ridge_models['3_y_pred']
y_pred_idx = ridge_models['0_idx']
y_pred_1_idx = ridge_models['1_idx']
y_pred_2_idx = ridge_models['2_idx']
y_pred_3_idx = ridge_models['3_idx']
random_model = random_models['0']
random_model_1 =random_models['1']
random_model_2 =random_models['2']
random_model_3 =random_models['3']
y_pred_random = random_models['0_y_pred']
y_pred_random_1 = random_models['1_y_pred']
y_pred_random_2 =random_models['2_y_pred']
y_pred_random_3 =random_models['3_y_pred']
y_pred_random_idx = random_models['0_idx']
y_pred_random_1_idx = random_models['1_idx']
y_pred_random_2_idx =random_models['2_idx']
y_pred_random_3_idx =random_models['3_idx']
xgb_model = xgb_models['0']
xgb_model_1 = xgb_models['1']
xgb_model_2 = xgb_models['2']
xgb_model_3 = xgb_models['3']
y_pred_xgb = xgb_models['0_y_pred']
y_pred_xgb_1 = xgb_models['1_y_pred']
y_pred_xgb_2 = xgb_models['2_y_pred']
y_pred_xgb_3 = xgb_models['3_y_pred']
y_pred_xgb_idx = xgb_models['0_idx']
y_pred_xgb_1_idx = xgb_models['1_idx']
y_pred_xgb_2_idx = xgb_models['2_idx']
y_pred_xgb_3_idx = xgb_models['3_idx']


idx_ridge = [y_pred_idx, y_pred_1_idx, y_pred_2_idx, y_pred_3_idx]
idx_random = [y_pred_random_idx, y_pred_random_1_idx, y_pred_random_2_idx, y_pred_random_3_idx]
idx_xgb = [y_pred_xgb_idx, y_pred_xgb_1_idx, y_pred_xgb_2_idx, y_pred_xgb_3_idx]
pred_ridge = [y_pred.reshape(-1), y_pred_1.reshape(-1), y_pred_2.reshape(-1), y_pred_3.reshape(-1)]
pred_random = [y_pred_random.reshape(-1), y_pred_random_1.reshape(-1), y_pred_random_2.reshape(-1), y_pred_random_3.reshape(-1)]
pred_xgb = [y_pred_xgb.reshape(-1), y_pred_xgb_1.reshape(-1), y_pred_xgb_2.reshape(-1), y_pred_xgb_3.reshape(-1)]


# Rigde OOF
ridge_idx = np.concatenate(idx_ridge)
ridge_pred = np.concatenate(pred_ridge)
ridge_order = np.argsort(ridge_idx)
ridge_df = pd.DataFrame({
    'index': ridge_idx[ridge_order],
    'true_y': y_np[ridge_idx[ridge_order]].reshape(-1),
    'ridge_pred': ridge_pred[ridge_order]
})

# RandomForest OOF
random_idx = np.concatenate(idx_random)
random_pred = np.concatenate(pred_random)
random_order = np.argsort(random_idx)
random_df = pd.DataFrame({
    'index': random_idx[random_order],
    'true_y': y_np[random_idx[random_order]].reshape(-1),
    'random_pred': random_pred[random_order]
})

# XGBoost OOF
xgb_idx = np.concatenate(idx_xgb)
xgb_pred = np.concatenate(pred_xgb)
xgb_order = np.argsort(xgb_idx)
xgb_df = pd.DataFrame({
    'index': xgb_idx[xgb_order],
    'true_y': y_np[xgb_idx[xgb_order]].reshape(-1),
    'xgb_pred': xgb_pred[xgb_order]
})

# --- Merge all models on the 'index' column ---
X_oof = ridge_df.merge(random_df[['index', 'random_pred']], on='index') \
                 .merge(xgb_df[['index', 'xgb_pred']], on='index')

X_oof = X_oof.reset_index(drop=True)
true_y_oof = X_oof['true_y']
drop_cols = ['true_y', 'index']
print(X_oof.head())
print(X_oof.describe())
print(X_oof.info())
X_oof = X_oof.drop(columns=drop_cols)


def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }
    
    # initializing the XGBoost model
    model = xgb      # If you have multiple GPUs, you can specify the GPU ID
                              
    # if you try in the computer without GPU, just remove the last 2 params (tree_method & device)
    
    score = cross_val_score(model, X_reduced.astype(np.float32), y_np.astype(np.float32), cv=4).mean()   # calculating score using cross-validation
    return score


import optuna

# Create and run the optimization process with 100 trials
study = optuna.create_study(study_name="example_xgboost_study", direction='maximize') 
study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)   

# Retrieve the best parameter values
best_params = study.best_params
print(f"\nBest parameters: {best_params}")




