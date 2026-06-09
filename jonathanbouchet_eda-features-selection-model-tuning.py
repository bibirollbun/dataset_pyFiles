import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 150)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import time
import re

import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# load data
df_train = pd.read_csv("/kaggle/input/molecular-machine-learning/train.csv", sep=",")
df_test = pd.read_csv("/kaggle/input/molecular-machine-learning/test.csv", sep=",")

# group some columns by type
features_others = ['Batch_ID', 'T80', 'Smiles']
features_base = ['Mass', 'HAcceptors', 'HDonors', 'LogP', 'Asphericity', 'Rg', 'TPSA', 'RingCount', 'NumRotatableBonds', 'NumHeteroatoms', 
                 'HOMOm1(eV)', 'HOMO(eV)', 'LUMO(eV)', 'LUMOp1(eV)', 'PrimeState', 'PrimeExcite(eV)', 'PrimeExcite(osc)', 'DipoleMoment(Debye)', 'SurfaceCharge', 'ChargeCorrection']
T_features= ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20']
S_features = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']
O_features = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15', 'O16', 'O17', 'O18', 'O19', 'O20']
TDOS_features = ['TDOS1.5', 'TDOS1.6', 'TDOS1.7', 'TDOS1.8', 'TDOS1.9', 'TDOS2.0', 'TDOS2.1', 'TDOS2.2', 'TDOS2.3', 'TDOS2.4', 'TDOS2.5', 'TDOS2.6', 'TDOS2.7', 'TDOS2.8', 'TDOS2.9', 'TDOS3.0', 'TDOS3.1', 
                 'TDOS3.2', 'TDOS3.3', 'TDOS3.4', 'TDOS3.5', 'TDOS3.6', 'TDOS3.7', 'TDOS3.8', 'TDOS3.9', 'TDOS4.0', 'TDOS4.1', 'TDOS4.2', 'TDOS4.3', 'TDOS4.4', 'TDOS4.5', 'TDOS4.6', 'TDOS4.7']
SDOS_features = ['SDOS2.5', 'SDOS2.6', 'SDOS2.7', 'SDOS2.8', 'SDOS2.9', 'SDOS3.0', 'SDOS3.1', 'SDOS3.2', 'SDOS3.3', 'SDOS3.4', 'SDOS3.5', 'SDOS3.6', 'SDOS3.7', 'SDOS3.8', 'SDOS3.9', 'SDOS4.0', 
                 'SDOS4.1', 'SDOS4.2', 'SDOS4.3', 'SDOS4.4', 'SDOS4.5', 'SDOS4.6', 'SDOS4.7', 'SDOS4.8', 'SDOS4.9', 'SDOS5.0', 'SDOS5.1', 'SDOS5.2', 'SDOS5.3', 'SDOS5.4']


df_train.head()


df_train.shape, df_test.shape


corr = df_train[T_features].corr()
corr.style.background_gradient(cmap='coolwarm')


corr = df_train[S_features].corr()
corr.style.background_gradient(cmap='coolwarm')


corr = df_train[O_features].corr()
corr.style.background_gradient(cmap='coolwarm')


corr = df_train[TDOS_features].corr()
corr.style.background_gradient(cmap='coolwarm')


corr = df_train[SDOS_features].corr()
corr.style.background_gradient(cmap='coolwarm')


df_pca = df_train[features_base + T_features + S_features + O_features + TDOS_features+ SDOS_features]
print(df_pca.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(df_pca) # Standardize the data
pca = PCA()
pca.fit(X_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)


plt.figure(figsize=(6,3))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Plot')
plt.grid(True)
n_components = np.argmax(cumulative_variance >= 0.95) + 1 # e.g., 95% variance
plt.axvline(x=n_components, color='r', linestyle='--', label=f'n_components = {n_components} (95% variance)')
plt.show()


from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, cross_validate, cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.feature_selection import SelectKBest, f_classif


num_cols: list[str] = features_base + T_features + S_features + O_features + TDOS_features+ SDOS_features
target: str = "T80"

X = df_train[num_cols]
y = df_train[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    shuffle=True,
    random_state=1234)

print(f"train: {X_train.shape}, test: {X_test.shape}")


numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), 
           ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
    ], 
    remainder="passthrough"
)

preprocessor


regressors = {'LinearRegression': LinearRegression(),
              'Ridge': Ridge(),
              'BayesianRidge': BayesianRidge(),
              'RandomForest': RandomForestRegressor(),
              'GradientBoostingRegressor': GradientBoostingRegressor(),
              'KnnRegressor': KNeighborsRegressor(),
              'BaggingRegressor': BaggingRegressor(),
              'SGDRegressor': SGDRegressor(),
              'XGBRegressor': xgb.XGBRegressor()
       }

pca = PCA(n_components=25, svd_solver='full') # ValueError: n_components=42 must be between 0 and min(n_samples, n_features)=26 with svd_solver='full'
random_state = 1234
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=1234)

metrics = ['neg_mean_squared_log_error', 'r2']
regressor_fitted = {}
cross_validate_res = [] # list to hold CV results

for cnt, (clf_name, clf) in enumerate(regressors.items()):
    
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            # ("pca", pca),
            (clf_name, clf)]
    )
    
    print(f"processing {clf_name}")
    res = cross_validate(pipe, X_train, y_train, cv=cv, return_train_score=True, scoring=metrics)
    pipe.fit(X_train, y_train)
    preds_train = pipe.predict(X_train)
    preds_test = pipe.predict(X_test)
    regressor_fitted[clf_name] = {"preds_train": preds_train, "preds_test": preds_test}
    res_df = pd.DataFrame(res).mean()
    res_df = pd.DataFrame(res_df).apply(pd.to_numeric).transpose()
    res_df['Regressor'] = clf_name
    cross_validate_res.append(res_df)


fig, ax = plt.subplots(2, len(regressors), figsize=(16,4))
cnt=0
for k,v in regressor_fitted.items():
    ax[0,cnt].scatter(y_train, v["preds_train"], s=2) 
    ax[0,cnt].axline([0,0],[100, 100], color='red', linewidth=1, linestyle="--", label="best line fit")

    ax[1,cnt].scatter(y_test, v["preds_test"], s=2)
    ax[1,cnt].axline([0,0],[100, 100], color='red', linewidth=1, linestyle="--", label="best line fit")
   
    ax[0,cnt].set_title(k, fontsize=12)
    ax[0,cnt].set_xlabel('ground truth (train)')
    ax[0,cnt].set_ylabel('prediction (train)')
    ax[1,cnt].set_xlabel('ground truth (test)')
    ax[1,cnt].set_ylabel('prediction (test)')

    ax[0,cnt].set_title(k, fontsize=12)

    for i in range(0,2):
        ax[i,cnt].xaxis.set_tick_params(labelsize=10)
        ax[i,cnt].yaxis.set_tick_params(labelsize=10)
        ax[i,cnt].xaxis.label.set_size(10)
        ax[i,cnt].yaxis.label.set_size(10)
    
    cnt += 1
plt.tight_layout()
plt.show()


pd.concat(cross_validate_res, ignore_index=True).style.background_gradient(cmap="viridis")


from sklearn.decomposition import PCA
pca = PCA(svd_solver='full')

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor), 
    ("pca", pca),
    ('classifier', RandomForestRegressor(random_state=1234))
 ])

param_grid =  {
    'pca__n_components': [5, 10, 15, 20, 25],
    'classifier__max_depth': [4, 6, 8, 10],
    'classifier__n_estimators':[50, 100, 200],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__criterion':['friedman_mse', 'absolute_error', 'squared_error', 'poisson']
}


grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_log_error', verbose=1)
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()


# Print the best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print(f"timing: {end-start}")

# Evaluate the best model on the train set
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
print("Train score:", train_score)

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)


train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

fig, ax = plt.subplots(2, 3, figsize=(8,4))
ax[0,0].scatter(y_train, train_preds, s=2) 
ax[0,0].set_xlabel('ground truth (train)')
ax[0,0].set_ylabel('prediction (train)')
ax[0,0].axline([0,0],[100,100], color='red', linewidth=1, linestyle="--", label="best line fit")
ax[0,1].hist(y_train - train_preds, bins='auto') 
ax[0,1].set_xlabel('residuals (train)')
ax[0,2].scatter(y_train, y_train - train_preds, s=1) 
ax[0,2].set_xlabel('ground truth (train)')
ax[0,2].set_ylabel('residuals (train)')

ax[1,0].scatter(y_test, test_preds, s=2) 
ax[1,0].set_xlabel('ground truth (test)')
ax[1,0].set_ylabel('prediction (test)')
ax[1,0].axline([0,0],[100,100], color='red', linewidth=1, linestyle="--", label="best line fit")
ax[1,1].hist(y_test - test_preds, bins='auto') 
ax[1,1].set_xlabel('residuals (test)')
ax[1,2].scatter(y_test, y_test - test_preds, s=2) 
ax[1,2].set_xlabel('ground truth (test)')
ax[1,2].set_ylabel('residuals (test)')

plt.tight_layout()
plt.show()


df_train2 = pd.concat([
    df_train[features_others + features_base], 
    pd.DataFrame(df_train[T_features].mean(axis=1), columns = ["average_T_features"]),
    pd.DataFrame(df_train[S_features].mean(axis=1), columns = ["average_S_features"]),
    pd.DataFrame(df_train[O_features].mean(axis=1), columns = ["average_O_features"]),
    pd.DataFrame(df_train[TDOS_features].mean(axis=1), columns = ["average_TDOS_features"]),
    pd.DataFrame(df_train[SDOS_features].mean(axis=1), columns = ["average_SDOS_features"]),
    ], axis=1)

df_train2.head()


df_train.iloc[0]['Smiles']


from itertools import chain
from collections import Counter

def extract_groups(smile: str):
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, smile)
    return matches

df_train2['Smiles_groups'] = df_train2['Smiles'].apply(lambda x: extract_groups(x))

smiles_groups = list(df_train2['Smiles_groups'])
smiles_list = list(chain(*smiles_groups))
smiles_groups_counter = Counter(smiles_list)

r = pd.DataFrame.from_dict(dict(smiles_groups_counter), orient="index").reset_index()
r.columns = ["sequence", "count"]
r.sort_values(by=["count"], inplace=True, ascending=False)


plt.figure(figsize=(12, 5))
sns.barplot(data=r, x='count', y='sequence', orient="h", color='darkblue')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()


# selecting the sequences for which the frequency is > 5
top_sequences = list(r[r["count"] >= 5]["sequence"])

# create a base dictionary
genre_dict = dict.fromkeys(top_sequences)
res = []
# loop over all sequences , loop over all keys and fill '1' if the current Smile has a sequence labelled, otherwise 0
for t in smiles_groups:
    tmp = dict.fromkeys(top_sequences)
    for k in tmp.keys():
        if k in t:
            tmp[k] = 1
        else:
            tmp[k] = 0
    res.append(tmp)
res_df = pd.DataFrame(res)
res_df.columns = [f"sequence_{x}" for x in res_df.columns]
# adding the total number of sequences(s) describing a Smile as another feature
res_df["all_top_sequences"] = res_df.sum(axis=1)


pd.DataFrame(pd.concat([df_train2[["Smiles"]], res_df], axis=1)).head(5).style.background_gradient(cmap="viridis")


df_train3 = pd.DataFrame(pd.concat([df_train2, res_df], axis=1))


num_cols: list[str] = ['Mass','HAcceptors','HDonors','LogP','Asphericity','Rg','TPSA','RingCount','NumRotatableBonds','NumHeteroatoms','HOMOm1(eV)','HOMO(eV)',
                       'LUMO(eV)','LUMOp1(eV)','PrimeState','PrimeExcite(eV)','PrimeExcite(osc)','DipoleMoment(Debye)','SurfaceCharge','ChargeCorrection',
                       'average_T_features','average_S_features','average_O_features','average_TDOS_features','average_SDOS_features','all_top_sequences']
bool_cols: list[str] = ['sequence_OCC(CC','sequence_CC','sequence_C','sequence_=O','sequence_-c3ccc(-c4cccs4','sequence_-c2ccc(-c3ccc(-c4cccs4']
target: str = "T80"

X = df_train3[num_cols + bool_cols]
y = df_train3[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    shuffle=True,
    random_state=1234)

print(f"train: {X_train.shape}, test: {X_test.shape}")


from sklearn.decomposition import PCA
pca = PCA(svd_solver='full')

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), 
           ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor), 
    ("pca", pca),
    ('classifier', RandomForestRegressor(random_state=1234))
 ])

param_grid =  {
    'pca__n_components': [5, 10, 15, 20, 25],
    'classifier__max_depth': [4, 6, 8, 10],
    'classifier__n_estimators':[50, 100, 200],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__criterion':['friedman_mse', 'absolute_error', 'squared_error', 'poisson']
}


grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_log_error', verbose=1)
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()


# Print the best hyperparameters and score
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
print(f"timing: {end-start}")

# Evaluate the best model on the train set
best_model = grid_search.best_estimator_
train_score = best_model.score(X_train, y_train)
print("Train score:", train_score)

# Evaluate the best model on the test set
test_score = best_model.score(X_test, y_test)
print("Test score:", test_score)


train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

fig, ax = plt.subplots(2, 3, figsize=(8,4))
ax[0,0].scatter(y_train, train_preds, s=2) 
ax[0,0].set_xlabel('ground truth (train)')
ax[0,0].set_ylabel('prediction (train)')
ax[0,0].axline([0,0],[100,100], color='red', linewidth=1, linestyle="--", label="best line fit")
ax[0,1].hist(y_train - train_preds, bins='auto') 
ax[0,1].set_xlabel('residuals (train)')
ax[0,2].scatter(y_train, y_train - train_preds, s=1) 
ax[0,2].set_xlabel('ground truth (train)')
ax[0,2].set_ylabel('residuals (train)')

ax[1,0].scatter(y_test, test_preds, s=2) 
ax[1,0].set_xlabel('ground truth (test)')
ax[1,0].set_ylabel('prediction (test)')
ax[1,0].axline([0,0],[100,100], color='red', linewidth=1, linestyle="--", label="best line fit")
ax[1,1].hist(y_test - test_preds, bins='auto') 
ax[1,1].set_xlabel('residuals (test)')
ax[1,2].scatter(y_test, y_test - test_preds, s=2) 
ax[1,2].set_xlabel('ground truth (test)')
ax[1,2].set_ylabel('residuals (test)')

plt.tight_layout()
plt.show()

