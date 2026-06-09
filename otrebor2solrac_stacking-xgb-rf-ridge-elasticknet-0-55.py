import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow.keras.backend as K


import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler,OrdinalEncoder,StandardScaler,FunctionTransformer,OneHotEncoder

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score,train_test_split,ShuffleSplit,KFold

import xgboost as xgb
from sklearn.svm import SVR
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,roc_auc_score,make_scorer


try:
    import umap
    import pingouin as pg
except:
    !pip install umap-learn
    !pip install pingouin
    import pingouin as pg
    import umap

sns.set_theme(style="darkgrid", palette="muted")


def r2_keras(y_true, y_pred):   
    y_true = K.cast(y_true, dtype="float32") 
    y_pred = K.cast(y_pred, dtype="float32") 
    
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(np.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon())).numpy()

def imbalace_features(data,cols,threshold=0.1):
    """
    Feature selection with low variance.
    """
    # Cols with one value
    cols_with_low_rate = []
    
    for col in cols:
        # Features with one value
        if data[col].nunique() == 1:
            cols_with_low_rate.append(col) 
            
        elif data[col].nunique() == 2:
            # Binary features with low rate.
            vals = data[col].value_counts().values
            max_val = max(vals)
            min_val = min(vals)
    
            if min_val/max_val < threshold:
                cols_with_low_rate.append(col)    
                
    return cols_with_low_rate

def roc_auc_corr(data,cols,target_col="y",num_cols=100):
    """
    Compute ROC-AUC as metric for correlation
    """
    corr_var = {}
    
    for col in cols:
        auc = roc_auc_score(data[col], data[target_col])
        corr_var[col] = auc
    
    ordered_cols = sorted(corr_var.items(), key=lambda item: item[1],reverse=True)
    
    return [var for var, val in ordered_cols[:num_cols]]


def plot_pca_explain(pca):
    """
    Plot explained variance by components
    """
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    variance_df = pd.DataFrame({
        'Principal Component': [f"PC{i+1}" for i in range(len(explained_variance_ratio))],
        'Explained Variance Ratio': explained_variance_ratio,
        'Cumulative Variance': cumulative_variance
    })
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, label='Individual Explained Variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance', color='red')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def phi_coefficient(contingency_table):
    """
    Calculate Phi coefficient for a 2x2 contingency table.
    """
    chi2, _, _, _ = chi2_contingency(contingency_table, correction=False)
    n = contingency_table.sum()
    return np.sqrt(chi2 / n)


def signed_phi_coefficient(contingency_table):
    """
    Compute Phi coefficient with sign.
    """
    a, b = contingency_table[0]
    c, d = contingency_table[1]
    
    numerator = (a * d) - (b * c)
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    
    return numerator / denominator if denominator != 0 else 0

def find_high_phi_pairs(df,threshold=0.5):
    """
    Find all pairs of binary columns with Phi coefficient > threshold.
    
    Parameters:
    - df: Pandas DataFrame with binary columns (0/1 values)
    - threshold: Minimum Phi coefficient to report (default: 0.5)
    
    Returns:
    - Dictionary
    """

    # Get only binary columns
    binary_cols = [col for col in df.columns 
                  if set(df[col].unique()).issubset({0, 1})]

    ocuppied_cols = []
    
    high_phi_pairs = []
    binary_data = df[binary_cols].values
    
    for i, j in combinations(range(len(binary_cols)), 2):      
        # Compute contingency table using NumPy
        contingency = np.histogram2d(
            binary_data[:, i], binary_data[:, j], bins=[2, 2]
        )[0]
        
        # Compute Phi coefficient
        # phi = phi_coefficient(contingency) # It does not have sign
        
        phi = signed_phi_coefficient(contingency)
        
        # Store pairs with high correlation
        if abs(phi) > threshold:
            high_phi_pairs.append((binary_cols[i], binary_cols[j], round(phi, 3)))

    result = {}
    used_cols = []

    # Columns and their correlated columns
    for key, sub_key, value in high_phi_pairs:
        if key in used_cols or sub_key in used_cols:
            continue
        
        if key not in result:
            result[key] = {}
            
        result[key][sub_key] = value
        used_cols.append(sub_key)

    return result


def merge_columns(df,operations={},remove_cols=False,): 
    """
        Create new columns based on their correalated ones.
    """
    cols_used = []
    for k, sub_dict in operations.items():
        
        if k in df.columns: 
            df["new_" + k] = df[k].copy()
            cols_used.append(k)
            
            for sub_k, value in sub_dict.items():
                if sub_k in cols_used:
                    continue
                
                if sub_k in df.columns:
                    if value > 0:
                        df["new_" + k] += df[sub_k]
                    else:
                        df["new_" + k] -= df[sub_k]
                    cols_used.append(sub_k)
                    
    if remove_cols:
        df.drop(columns=cols_used,inplace=True)
                    
    return df


def get_umap(data):
    reducer = umap.UMAP()
    umap_df = reducer.fit_transform(data)
    umap_df = pd.DataFrame(umap_df)
    umap_df.columns = ["UM_{}".format(i) for i in range(umap_df.shape[1])]

    return reducer, umap_df


def get_pca(data,n_components=10):
    pca = PCA(n_components=n_components) 
    pca_result = pca.fit_transform(data)
    pca_result = pd.DataFrame(pca_result)
    pca_result.columns = ["PC_{}".format(i) for i in range(pca_result.shape[1])]

    return pca,pca_result

def get_dbscan(data):
    reducer = DBSCAN()
    umap_df = reducer.fit_transform(data)
    umap_df = pd.DataFrame(umap_df)
    umap_df.columns = ["UM_{}".format(i) for i in range(umap_df.shape[1])]

    return reducer, umap_df    
    

def convert_to_int(X):
    return X.astype(int)

def filter_data(data,final_cols=[]):
    return data[final_cols]

def test_pipeline(data,pipeline):
    X_train, X_test, y_train, y_test = (
        train_test_split(
            data.iloc[:,1:], 
            data["y"], 
            test_size=0.2,
            random_state=42)
    )

    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    
    print(r2_keras(y_test,y_pred))


data = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip")
data.drop(columns=["ID"],inplace=True)


data.info()


data.sample(5)


data.y.plot.box()


data = data[data.y<=176].reset_index(drop=True)


plt.figure(figsize=(15,5))
sns.histplot(data["y"], bins=30, kde=True, stat="density", alpha=0.5)
plt.show()


data_bins = [0,80, 113, 300] 
data_labels = ['A', 'B','C']  # Define bin labels


# Binning y into categories
pd.cut(data['y'], bins=data_bins, labels=data_labels, right=True).value_counts()



cat_vars =  data.select_dtypes(include=['object']).columns

corr_cat = lambda x: pg.anova(data=data, dv="y", between=x)["np2"].values[0].round(3)

# Compute eta-squared for X0 (categorical) vs y (continuous)
anova_results = {col:corr_cat(col) for col in cat_vars}
sorted(anova_results.items(), key=lambda x :x[1],reverse=True)


cat_vars


pipeline = make_pipeline(
    ColumnTransformer(
        [
            ("cat", OrdinalEncoder(), cat_vars)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas") 
)

encoded_data = pipeline.fit_transform(data)
encoded_data[list(cat_vars) + ["y"]].corr()["y"].map(abs).sort_values(ascending=False)


# Plot variable X0 vs y
cat_df = data[list(cat_vars) + ["y"]]

plt.figure(figsize=(15,5))
sns.boxplot(data=cat_df.sort_values("X0"),x="X0",y="y")
plt.show()


# Plot variable X2 vs y
cat_df = data[list(cat_vars) + ["y"]]

plt.figure(figsize=(15,5))
sns.boxplot(data=cat_df.sort_values("X2"),x="X2",y="y")
plt.show()


# Plot variable X3 vs y
cat_df = data[list(cat_vars) + ["y"]]

plt.figure(figsize=(15,5))
sns.boxplot(data=cat_df.sort_values("X3"),x="X3",y="y")
plt.show()


int_columns = list(data.select_dtypes(include=['int']).columns)

print(f"Num int cols: {len(int_columns)}")


data[int_columns].mean().plot.hist(bins=20)


data[int_columns].mean().sort_values().where(lambda x: x>0.1).dropna()


# For instance:
data["X205"].value_counts()


new_cat_vars = ["X0","X3"]

# Select features
X = data[int_columns + new_cat_vars ]
y = data["y"]

# Fit OneHotEncoder to determine all categories
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Ensure it sees all possible categories
ohe.fit(X[new_cat_vars])  

one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")

pipeline_rf = make_pipeline(
    one_hot,
    RandomForestRegressor(n_estimators=150,max_depth=5,random_state=42)
)

custom_scorer = make_scorer(r2_keras)
cv_scores = cross_val_score(pipeline_rf, X, y, cv=5, scoring=custom_scorer,n_jobs=-1) 
print(cv_scores,np.mean(cv_scores))


test_pipeline(data[["y"] + int_columns + new_cat_vars ],pipeline_rf)


one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")

encoded_data = one_hot.fit_transform(data)

int_encoded_columns = list(encoded_data.select_dtypes(include=['int']).columns)
int_encoded_columns.sort()


high_phi_pairs = find_high_phi_pairs(encoded_data[int_encoded_columns],threshold=0.8)
high_phi_pairs["X0_a"]


# Create a contingency table
contingency_table = pd.crosstab(encoded_data["X0_a"], encoded_data["X172"])

# Plot heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Heatmap')
plt.xlabel('X0_a')
plt.ylabel('X172')
plt.show()


high_phi_pairs["X0_ap"]


# Create a contingency table
contingency_table = pd.crosstab(encoded_data["X0_ap"], encoded_data["X111"])

# Plot heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Heatmap')
plt.xlabel('X0_ap')
plt.ylabel('X111')
plt.show()


new_features_data = merge_columns(encoded_data.copy(),operations=high_phi_pairs,remove_cols=True)


new_features_data.info()


new_features_data[new_features_data.select_dtypes(include=['int']).columns].map(abs).mean().plot.hist(bins=30)


encoded_data.shape


new_cat_vars = ["X0","X3"]

X = data[int_columns + new_cat_vars ]
y = data["y"]

# Fit OneHotEncoder to determine all categories
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Ensure it sees all possible categories
ohe.fit(X[new_cat_vars])  

one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")

pipeline_rf = make_pipeline(
    one_hot,
    FunctionTransformer(merge_columns,kw_args={"operations": high_phi_pairs,"remove_cols":False}),
    RandomForestRegressor(n_estimators=150,max_depth=5,random_state=42)
)

custom_scorer = make_scorer(r2_keras)
cv_scores = cross_val_score(pipeline_rf, X, y, cv=5, scoring=custom_scorer,n_jobs=-1) 
print(cv_scores,np.mean(cv_scores))


test_pipeline(data[["y"] + int_columns + new_cat_vars ],pipeline_rf)


# Detect the features with low variance
new_int_cols = new_features_data.select_dtypes(include=['int']).columns
cols_without_var = imbalace_features(new_features_data,new_int_cols,threshold=0.025)


new_features_data[cols_without_var].map(abs).mean().div(0.01*data.shape[0]).sort_values(ascending=False)


final_cols = list(set(new_int_cols) - set(cols_without_var))
len(final_cols)


new_features_data[final_cols + ["y"]].corr()["y"].map(abs).sort_values(ascending=False).tail(20).plot.bar()


new_cat_vars = ["X0","X3"]

X = data[int_columns + new_cat_vars ]
y = data["y"]

# Fit OneHotEncoder to determine all categories
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Ensure it sees all possible categories
ohe.fit(X[new_cat_vars])  

one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")

pipeline_rf = make_pipeline(
    one_hot,
    FunctionTransformer(merge_columns,kw_args={"operations": high_phi_pairs,"remove_cols":True}),
    FunctionTransformer(filter_data,kw_args={"final_cols": final_cols}),
    RandomForestRegressor(n_estimators=150,max_depth=5,random_state=42)
)

custom_scorer = make_scorer(r2_keras)
cv_scores = cross_val_score(pipeline_rf, X, y, cv=5, scoring=custom_scorer,n_jobs=-1) 
print(cv_scores,np.mean(cv_scores))


test_pipeline(data,pipeline_rf)


pca_model,pca_df = get_pca(new_features_data[final_cols],n_components=10)


pca_df[['PC_0','PC_1']].plot.scatter(x="PC_0",y="PC_1")


plot_pca_explain(pca_model)


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(pca_df)

dbscan = DBSCAN(eps=0.3, min_samples=30)
labels = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap="viridis", edgecolors="k")
plt.title("Clustering con DBSCAN")
plt.show()


pd.Series(labels).value_counts()


aux = data[["y"]]
aux["labels"] = labels
aux.groupby("labels").y.describe().sort_values("min")


from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = (
    train_test_split(
        new_features_data[final_cols], 
        labels, 
        test_size=0.2,
        random_state=42)
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)


class PCADBSCANTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, eps=0.5, min_samples=5):
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.classifier = RandomForestClassifier(random_state=42)


    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.dbscan.fit(X_pca)
        self.cluster_labels_ = self.dbscan.labels_
        self.classifier.fit(X, self.cluster_labels_)
            
        return self

    def transform(self, X):
        clusters = self.classifier.predict(X)
        return np.c_[X, clusters] 


new_cat_vars = ["X0","X3"]

X = data[int_columns + new_cat_vars ]
y = data["y"]

# Fit OneHotEncoder to determine all categories
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Ensure it sees all possible categories
ohe.fit(X[new_cat_vars])  

one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")


pipeline_rf = make_pipeline(
    one_hot,
    FunctionTransformer(merge_columns,kw_args={"operations": high_phi_pairs,"remove_cols":True}),
    FunctionTransformer(filter_data,kw_args={"final_cols": final_cols}),
    PCADBSCANTransformer(n_components=10, eps=0.3, min_samples=30),
    RandomForestRegressor(n_estimators=150,max_depth=5,random_state=42)
)

custom_scorer = make_scorer(r2_keras)
cv_scores = cross_val_score(pipeline_rf, X, y, cv=5, scoring=custom_scorer,n_jobs=-1) 
print(cv_scores,np.mean(cv_scores))


test_pipeline(data,pipeline_rf)


pipeline_rf.fit(X, y)

# Access the fitted RandomForestRegressor from the pipeline
random_forest_model = pipeline_rf.named_steps['randomforestregressor']

# Get feature importances from the trained model
feature_importances = random_forest_model.feature_importances_

feature_importances = pd.DataFrame(zip(final_cols + ["clus"],feature_importances))
feature_importances.columns = ["Feature","Importance"]


feature_importances.sort_values("Importance",ascending=False).head(30)


final_cols2 = list(set(final_cols) - set(feature_importances[feature_importances.Importance<0.001].Feature.tolist()))


len(final_cols2)


class PCADBSCANTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, eps=0.5, min_samples=5):
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.classifier = RandomForestClassifier(random_state=42)


    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.dbscan.fit(X_pca)
        self.cluster_labels_ = self.dbscan.labels_
        self.classifier.fit(X, self.cluster_labels_)
            
        return self

    def transform(self, X):
        clusters = self.classifier.predict(X)
        return np.c_[X, clusters] 


new_cat_vars = ["X0","X3"]

X = data[int_columns + new_cat_vars ]
y = data["y"]

# Fit OneHotEncoder to determine all categories
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Ensure it sees all possible categories
ohe.fit(X[new_cat_vars])  

one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")


pipeline_rf = make_pipeline(
    one_hot,
    FunctionTransformer(merge_columns,kw_args={"operations": high_phi_pairs,"remove_cols":True}),
    FunctionTransformer(filter_data,kw_args={"final_cols": final_cols2}),
    PCADBSCANTransformer(n_components=10, eps=0.3, min_samples=30),
    RandomForestRegressor(n_estimators=150,max_depth=5,random_state=42)
)

custom_scorer = make_scorer(r2_keras)
cv_scores = cross_val_score(pipeline_rf, X, y, cv=5, scoring=custom_scorer,n_jobs=-1) 
print(cv_scores,np.mean(cv_scores))


test_pipeline(data,pipeline_rf)


class PCADBSCANTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, eps=0.5, min_samples=5):
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.classifier = RandomForestClassifier(random_state=42)


    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.dbscan.fit(X_pca)
        self.cluster_labels_ = self.dbscan.labels_
        self.classifier.fit(X, self.cluster_labels_)
            
        return self

    def transform(self, X):
        clusters = self.classifier.predict(X)
        return np.c_[X, clusters] 


new_cat_vars = ["X0","X3"]

X = data[int_columns + new_cat_vars ]
y = data["y"]

# Fit OneHotEncoder to determine all categories
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Ensure it sees all possible categories
ohe.fit(X[new_cat_vars])  

one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")


pipeline_xgb = make_pipeline(
    one_hot,
    FunctionTransformer(merge_columns,kw_args={"operations": high_phi_pairs,"remove_cols":True}),
    FunctionTransformer(filter_data,kw_args={"final_cols": final_cols2}),
    PCADBSCANTransformer(n_components=10, eps=0.3, min_samples=30),
    xgb.XGBRegressor(subsample=1.0,n_estimators=500,max_depth=3,learning_rate=0.01,colsample_bytree=1.0)
)

custom_scorer = make_scorer(r2_keras)
cv_scores = cross_val_score(pipeline_xgb, X, y, cv=5, scoring=custom_scorer,n_jobs=-1) 
print(cv_scores,np.mean(cv_scores))


test_pipeline(data,pipeline_xgb)


from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV


param_grid = {
    'randomforestregressor__n_estimators': [100,150, 200, 300],  # Number of trees
    'randomforestregressor__max_depth': [None, 5,10,50, 100, 150],  # Max depth of trees
    'randomforestregressor__min_samples_split': [2, 5, 10],  # Minimum samples to split a node
    'randomforestregressor__min_samples_leaf': [1, 2, 5],  # Minimum samples per leaf
    'randomforestregressor__max_features': [1.0,'sqrt', 'log2', None],  # Features considered per split
    'randomforestregressor__ccp_alpha': [0.0, 0.01, 0.1],  # Complexity pruning
    'randomforestregressor__max_samples': [None, 0.8, 0.9],  # Fraction of dataset used for training each tree
}

# grid_search = GridSearchCV(pipeline_rf, param_grid, cv=5, n_jobs=-1, verbose=2)
# random_search = RandomizedSearchCV(pipeline_rf, param_grid, n_iter=100, cv=5, n_jobs=-1, verbose=2,scoring=custom_scorer,random_state=42)
# halving_search = HalvingGridSearchCV(
#     pipeline_rf, param_grid, cv=5, factor=2, min_resources="exhaust", verbose=2, n_jobs=-1
# )
# random_search.fit(X, y)

# print("Best parameters:", random_search.best_params_)
# print("Best R² score:", random_search.best_score_)

# Fitting 5 folds for each of 100 candidates, totalling 500 fits
# Best parameters: {'randomforestregressor__n_estimators': 150, 'randomforestregressor__min_samples_split': 10, 'randomforestregressor__min_samples_leaf': 2, 'randomforestregressor__max_samples': 0.9, 'randomforestregressor__max_features': 1.0, 'randomforestregressor__max_depth': 5, 'randomforestregressor__ccp_alpha': 0.1}
# Best R² score: 0.5922175288200379



from sklearn.linear_model import ElasticNet

class PCADBSCANTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, eps=0.5, min_samples=5):
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.classifier = RandomForestClassifier(random_state=42)


    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        self.dbscan.fit(X_pca)
        self.cluster_labels_ = self.dbscan.labels_
        self.classifier.fit(X, self.cluster_labels_)
            
        return self

    def transform(self, X):
        clusters = self.classifier.predict(X)
        return np.c_[X, clusters] 


new_cat_vars = ["X0","X3"]

X = data[int_columns + new_cat_vars ]
y = data["y"]

# Fit OneHotEncoder to determine all categories
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Ensure it sees all possible categories
ohe.fit(X[new_cat_vars])  

one_hot = ColumnTransformer([
    ("onehot", make_pipeline(
        OneHotEncoder(categories=ohe.categories_,
                      sparse_output=False,
                      handle_unknown='ignore'),
        FunctionTransformer(convert_to_int)
    ), new_cat_vars)
], remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")



# Define the base models
xgb_model = xgb.XGBRegressor(subsample=1.0,n_estimators=500,max_depth=3,learning_rate=0.01,colsample_bytree=1.0)

rf_model =  RandomForestRegressor(n_estimators=150,
                          min_samples_split=10,
                          min_samples_leaf=2,
                          max_samples=0.9,
                          max_features=1.0,
                          max_depth=5,
                          ccp_alpha=0.1,
                          random_state=42)

# Define Stacking Regressor
stacked_model = StackingRegressor(
    estimators=[('rf', rf_model), ('gb', xgb_model),('ridge',Ridge())],
    final_estimator=ElasticNet(l1_ratio=0.1, alpha=1.4),
    passthrough=True
)


pipeline_stack = make_pipeline(
    one_hot,
    FunctionTransformer(merge_columns,kw_args={"operations": high_phi_pairs,"remove_cols":True}),
    FunctionTransformer(filter_data,kw_args={"final_cols": final_cols2}),
    PCADBSCANTransformer(n_components=10, eps=0.3, min_samples=30),
    stacked_model
)

custom_scorer = make_scorer(r2_keras)
cv_scores = cross_val_score(pipeline_stack, X, y, cv=5, scoring=custom_scorer,n_jobs=-1) 
print(cv_scores,np.mean(cv_scores))


test_pipeline(data,pipeline_stack)


data_test = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip")
data_test["y"] = pipeline_stack.predict(data_test)
data_test["y"].reset_index().rename(columns={"index":"ID"}).to_csv("result.csv",index=False)


from IPython.display import FileLink
FileLink('result.csv')  # Genera un enlace de descarga directa

