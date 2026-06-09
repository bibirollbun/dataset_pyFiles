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


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost.core')
warnings.filterwarnings("ignore", message="1 warning generated.")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train_data_set_path=r"/kaggle/input/playground-series-s5e8/train.csv"
df_train_data_set=pd.read_csv(train_data_set_path)
df_train_data_set.head()


df_train_data_set=df_train_data_set.sample(10000,random_state=42)


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Data Wrangling Libraries
import numpy as np
import pandas as pd

#Data Processing Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#Model Metricrs

from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score


# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier


# Model Search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import  RandomizedSearchCV


#Plotting Libraries

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import math


df_train_data_set.shape


df_train_data_set=df_train_data_set.set_index("id")


df_train_data_set.isnull().sum()


df_train_data_set.dtypes



df_train_data_set.describe(include=["number"])


df_train_data_set.describe(include=["object"])


numerical_variables=[index for index in df_train_data_set.dtypes.index if df_train_data_set.dtypes[index] =="int64"]
numerical_variables


categorical_variables=[index for index in df_train_data_set.dtypes.index if df_train_data_set.dtypes[index] =="object"]
categorical_variables



# Set a consistent theme
sns.set_theme(style="whitegrid")

# Replace with your actual DataFrame
df = df_train_data_set.copy()

# Identify variable types
numerical_variables = df.select_dtypes(include=['number']).columns.tolist()
categorical_variables = df.select_dtypes(include=['object', 'category']).columns.tolist()
hue_col = "y"  # Adjust if needed




# --- 1. Histograms with Percentiles ---
list_len = len(numerical_variables)
number_of_column = math.ceil(math.sqrt(list_len))
rows_plot = math.ceil(list_len / number_of_column)

fig, axes = plt.subplots(rows_plot, number_of_column, figsize=(5 * number_of_column, 5 * rows_plot))
axes = axes.reshape(rows_plot, number_of_column)

counter = 0
for num_column in numerical_variables:
    trace_x = counter // number_of_column
    trace_y = counter % number_of_column
    ax = axes[trace_x][trace_y]

    sns.histplot(ax=ax, data=df, x=num_column, kde=True,palette="YlGnBu")

    p25 = np.percentile(df[num_column].dropna(), 25)
    p50 = np.percentile(df[num_column].dropna(), 50)
    p75 = np.percentile(df[num_column].dropna(), 75)

    ax.axvline(p25, color='red', linestyle='--', label='25th percentile')
    ax.axvline(p50, color='green', linestyle='--', label='50th percentile')
    ax.axvline(p75, color='blue', linestyle='--', label='75th percentile')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'{num_column}')
    ax.legend()

    counter += 1

for i in range(counter, rows_plot * number_of_column):
    fig.delaxes(axes[i // number_of_column][i % number_of_column])

plt.tight_layout()
plt.show()



deciles_df=df_train_data_set.describe(include=["number"],percentiles=[.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
deciles_df


n5_decile=deciles_df.loc["95%"]
n5_decile


#Removing Features
"""
features = [col for col in numerical_variables if col != "y"]
#df_train_data_set_wo_outlaters=df_train_data_set

for feature in features:
    df_train_data_set=df_train_data_set.loc[df_train_data_set[feature]<=n5_decile[feature]]

    
print(df_train_data_set.shape)

"""


#Replacing Features
"""

features = [col for col in numerical_variables if col != "y"]
#df_train_data_set_wo_outlaters=df_train_data_set

for feature in features:
    df_train_data_set[feature] = df_train_data_set[feature].clip(upper=n5_decile[feature])
"""


# --- 2. Box Plots for Outlier Detection ---
fig, axes = plt.subplots(rows_plot, number_of_column, figsize=(5 * number_of_column, 5 * rows_plot))
axes = axes.reshape(rows_plot, number_of_column)

counter = 0
for num_column in numerical_variables:
    trace_x = counter // number_of_column
    trace_y = counter % number_of_column
    ax = axes[trace_x][trace_y]

    sns.boxplot(ax=ax,data=df, y=num_column, x=hue_col, palette="YlGnBu")
    ax.set_title(f'{num_column}')
    ax.set_ylabel('')
    ax.set_xlabel(hue_col)
    ax.tick_params(axis='x', rotation=90)

    counter += 1

for i in range(counter, rows_plot * number_of_column):
    fig.delaxes(axes[i // number_of_column][i % number_of_column])

plt.tight_layout()
plt.show()



# --- 3. Correlation Heatmap ---
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="YlGnBu", square=False, linewidths=.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.tight_layout()
plt.show()



# --- 4. Categorical Bar Plots with Percentages ---
list_len = len(categorical_variables)
number_of_column = math.ceil(math.sqrt(list_len))
rows_plot = math.ceil(list_len / number_of_column)

fig, axes = plt.subplots(rows_plot, number_of_column, figsize=(5 * number_of_column, 5 * rows_plot))
axes = axes.reshape(-1, number_of_column)

counter = 0
for cat_column in categorical_variables:
    trace_x = counter // number_of_column
    trace_y = counter % number_of_column
    ax = axes[trace_x][trace_y]

    count_data = df.groupby([cat_column, hue_col]).size().reset_index(name='count')
    total_per_category = count_data.groupby(cat_column)['count'].transform('sum')
    count_data['percent'] = count_data['count'] / total_per_category * 100

    sns.barplot(ax=ax, data=count_data, x=cat_column, y='percent',palette="YlGnBu", hue=hue_col)
    ax.set_ylabel('Percentage')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'{cat_column} vs {hue_col} (%)')

    counter += 1
for i in range(counter, rows_plot * number_of_column):
    fig.delaxes(axes[i // number_of_column][i % number_of_column])

plt.tight_layout()
plt.show()



df_sampled = df_train_data_set.sample(n=1000, random_state=42)
sns.pairplot(df_sampled, hue="y", kind="reg", corner=True,palette="YlGnBu")


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# Remove 'y' safely
features = [col for col in numerical_variables if col != "y"]

# Scale features
scaled = MinMaxScaler().fit_transform(df_train_data_set[features])
train_scaled = df_train_data_set.copy()
train_scaled[features] = scaled

# Define cluster counts per class
cluster_counts = {1: 8, 0: 4}

# Radar plot function
def make_radar(ax, values, labels, color='blue'):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax.plot(angles, values, color=color, linewidth=1.5)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='lightgray')
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

# Prepare subplots
fig, axes = plt.subplots(
    nrows=3,
    ncols=4,
    subplot_kw=dict(polar=True),
    figsize=(12, 8)
)
axes = axes.flatten()

idx = 0
for label, k in cluster_counts.items():
    class_data = train_scaled[train_scaled["y"] == label]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(class_data[features])

    for i, center in enumerate(kmeans.cluster_centers_):
        make_radar(
            axes[idx],
            center.tolist(),
            features,
            color='orange' if label == 1 else 'green'
        )
        axes[idx].set_title(f'{label} #{i+1}', fontsize=10)
        idx += 1

plt.tight_layout()
plt.show()



def one_hot_encoding(data_ohc,le,ohc,categorical_cols):

    """
    This function goes through each columns and
    
    1. creates the one hot encoded based on the values of the selected column
    2. removes the selected column from the data.
    3. Renames the new columns
    4. Creates a data frame with the new columns
    5. Concatenate the data frame (with the new columns) to the original data frame.
    
    """
    for col in categorical_cols:
        dat = le.fit_transform(data_ohc[col]).astype(int)
        data_ohc = data_ohc.drop(col, axis=1)
        new_dat = ohc.fit_transform(dat.reshape(-1, 1))
        col_names = ['_'.join([col, str(x)]) for x in le.classes_]
        new_df = pd.DataFrame(new_dat, index=data_ohc.index, columns=col_names).astype(int)
        data_ohc = pd.concat([data_ohc, new_df], axis=1)

    return data_ohc


def one_hot_encoding_ordinal_columns(data_ohc, ordinal_cols, le, ohc):
    mappings = {}
    for col in ordinal_cols:
        dat = le.fit_transform(data_ohc[col]).astype(int)
        data_ohc = data_ohc.drop(col, axis=1)
        new_df = pd.DataFrame(dat, index=data_ohc.index, columns=[col])
        new_df = new_df.astype(int)
        data_ohc = pd.concat([data_ohc, new_df], axis=1)
        mappings[col] = dict(zip(range(len(le.classes_)), le.classes_))
    
    return data_ohc, mappings


def log1p_regularization(data_ohc,col_to_scale):
    for column in col_to_scale:
        data_ohc[column] = np.log1p(data_ohc[column])
    return data_ohc

# using scalers
import joblib  # or pickle

mm = MinMaxScaler()
s=StandardScaler()


# Fit the scaler on training data
scaler = MinMaxScaler()
#scaler.fit(data_ohc[non_categorical_cols])

# Save the fitted scaler to a file
joblib.dump(scaler, 'minmax_scaler.pkl')


# Load the saved scaler
#scaler = joblib.load('minmax_scaler.pkl')
# Transform the prediction data
#data_ohc_pred[col_to_scale] = scaler.transform(data_ohc_pred[col_to_scale])



mask = df_train_data_set.dtypes == 'object'
categorical_cols = df_train_data_set.columns[mask]
categorical_cols


non_categorical_cols = df_train_data_set.select_dtypes(include=[np.number]).columns
non_categorical_cols


#From this list, it is possible to observe that we have an ordinal variable month
ordinal_cols=["month"]


categorical_cols=list(set(categorical_cols)-set(ordinal_cols))
non_categorical_cols=list(set(non_categorical_cols)-set(["y"]))


data_ohc = df_train_data_set.copy()#


# One Hot Encoding Categorical Columns

le = LabelEncoder()
ohc = OneHotEncoder(sparse=False)
data_ohc =one_hot_encoding(data_ohc,le,ohc,categorical_cols)

#na_columns=[column for column in data_ohc.columns if column.endswith("_nan")]
#na_columns

# One Hot Encoding Ordinal Columns
data_ohc, mappings = one_hot_encoding_ordinal_columns(data_ohc, ordinal_cols, le, ohc)

scaler = MinMaxScaler()
scaler.fit(data_ohc[non_categorical_cols])
data_ohc[non_categorical_cols] = scaler.transform(data_ohc[non_categorical_cols])


data_ohc


target_column='y'


def measure_error(y_true, y_pred, label,y_probs):
    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred),
                       'auc_score' : roc_auc_score(y_true, y_probs),
                     },
                      name=label)


def confusion_matrix_graph(y,y_pred):
    sns.set_context('talk')
    cm = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d')


# Set up X and y variables
y, X = data_ohc[target_column], data_ohc.drop(columns=target_column)
# Split the data into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


"""
from sklearn.model_selection import train_test_split

# suppose X, y are your full dataset features/labels
X_small, _, y_small, _ = train_test_split(
    X, y,
    train_size=10_000,            # or a fraction like 0.02
    stratify=y,                   # preserve class ratios
    random_state=42
)
"""


""""""
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}

param_grid_best = {
    'n_estimators': [500],
    'max_depth': [50],
    'min_samples_split': [2],
    'min_samples_leaf': [2],
    'max_features': ['sqrt'],
    'bootstrap': [False],
    'class_weight': [None]
}


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42
                           ,n_estimators= 500
                           ,min_samples_split= 2
                           ,min_samples_leaf=2
                           ,max_features= 'sqrt'
                           , max_depth= 50
                           ,class_weight= None
                           ,bootstrap=False)




search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=5,                # Number of combinations to try
    cv=5,                     # 5-fold cross-validation
    scoring='roc_auc',       # Or use 'f1', 'roc_auc' for classification
    verbose=1,
    n_jobs=-1,                # Use all processors
    random_state=42
)

search.fit(X_train, y_train)
best_rf = search.best_estimator_
print("Best Parameters:", search.best_params_)


#rf.fit(X_resampled, y_resampled)
#rf.fit(X, y)
#Fitting 5 folds for each of 30 candidates, totalling 150 fits
#Best Parameters: {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 50, 'class_weight': None, 'bootstrap': False}


y_probs_pred = best_rf.predict_proba(X_test)[:, 1]
y_pred=best_rf.predict(X_test)
random_forest_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
random_forest_metrics.name="RandomForest"
random_forest_metrics


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np

# Define the model
model = XGBClassifier(use_label_encoder=False, eval_metric='auc', verbosity=0)

# Define hyperparameter grid (tune these)
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Use ROC AUC as scoring metric
roc_auc = make_scorer(roc_auc_score, needs_proba=True)

# Setup randomized search
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50,            # number of random parameter combinations to try
    scoring=roc_auc,
    cv=3,                 # 3-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1             # use all CPU cores
)

# Run the search
random_search.fit(X_train, y_train)

# Best parameters and best score
print("Best params:", random_search.best_params_)
print("Best ROC AUC:", random_search.best_score_)

#Fitting 3 folds for each of 50 candidates, totalling 150 fits
#Best params: {'subsample': 0.8, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 0.6}
#Best ROC AUC: 0.966870142882675


y_probs_pred = random_search.predict_proba(X_test)[:, 1]
y_pred=random_search.predict(X_test)
random_search_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
random_search_metrics.name="XGBClassifier"
random_search_metrics


#CatBoost, XGBoost, LightGBM



# Fit Logistic Regression
from sklearn.linear_model import LogisticRegression
LR_L2 = LogisticRegression(penalty='l2', max_iter=500, solver='saga')
LR_L2.fit(X_train, y_train)




from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
# Combine models
estimators = [('LR_L2', LR_L2), ('RF', random_search)]
VC = VotingClassifier(estimators, voting='soft')
VC.fit(X_train, y_train)



y_probs_pred = LR_L2.predict_proba(X_test)[:, 1]
y_pred=LR_L2.predict(X_test)
logistic_regression_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
logistic_regression_metrics.name="LogisticRegression"
logistic_regression_metrics


y_probs_pred = VC.predict_proba(X_test)[:, 1]
y_pred=VC.predict(X_test)
voting_classiffier_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
voting_classiffier_metrics.name="Voting_Classiffier"
voting_classiffier_metrics



from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



tree_list = [15, 25, 50, 100, 200, 400]
# The parameters to be fit
param_grid = {'n_estimators': tree_list,
              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
              'subsample': [1.0, 0.5],
              'max_features': [1, 2, 3, 4]}

# Grid Search for Gradient Boosting
GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42),
                      param_grid=param_grid,
                      scoring='roc_auc',
                      n_jobs=-1)
GV_GBC.fit(X_train, y_train)
best_gbc = GV_GBC.best_estimator_

"""
# Combine models
estimators = [('LR_L2', LR_L2), ('GBC', best_gbc)]
VC = VotingClassifier(estimators, voting='soft')
VC.fit(X, y)
"""




y_probs_pred = best_gbc.predict_proba(X_test)[:, 1]
y_pred=best_gbc.predict(X_test)
best_gbc_classiffier_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
best_gbc_classiffier_metrics.name="GradientBoostingClassifier"
best_gbc_classiffier_metrics




from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))

param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.001]}

GV_ABC = GridSearchCV(ABC,
                      param_grid=param_grid,
                      scoring='accuracy',
                      n_jobs=-1)

GV_ABC = GV_ABC.fit(X_train, y_train)

# The best model
GV_ABC.best_estimator_




y_probs_pred = GV_ABC.predict_proba(X_test)[:, 1]
y_pred=GV_ABC.predict(X_test)
GV_ABC_classiffier_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
GV_ABC_classiffier_metrics.name="AdaBoostClassifier"
GV_ABC_classiffier_metrics


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    verbose=0,          # suppress training logs
    eval_metric='AUC',
    random_state=42
)

# --------------------
# Step 3: Hyperparameter search space
# --------------------
param_grid = {
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "iterations": [200, 500, 1000],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 64, 128]  # number of splits for numerical features
}

# --------------------
# Step 4: Randomized search
# --------------------
rs = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_grid,
    n_iter=20,              # number of random combinations to try
    scoring='accuracy',     # could also use f1, roc_auc
    cv=3,                   # 3-fold cross-validation
    verbose=1,
    n_jobs=-1,
    random_state=42
)

rs.fit(X_train, y_train)

# --------------------
# Step 5: Evaluate best model
# --------------------
print("Best params:", rs.best_params_)
print("Best CV score:", rs.best_score_)

best_cat = rs.best_estimator_


y_probs_pred = best_cat.predict_proba(X_test)[:, 1]
y_pred=best_cat.predict(X_test)
best_cat_classiffier_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
best_cat_classiffier_metrics.name="CatBoostClassifier"
best_cat_classiffier_metrics


# Combine models
estimators = [('best_cat', best_cat), ('random_search', random_search)]
VC_boost_xgb = VotingClassifier(estimators, voting='soft')
VC_boost_xgb.fit(X_train, y_train)



y_probs_pred = VC_boost_xgb.predict_proba(X_test)[:, 1]
y_pred=VC_boost_xgb.predict(X_test)
VCbest_cat_classiffier_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
VCbest_cat_classiffier_metrics.name="VotingCatBoostXCGBoost"
VCbest_cat_classiffier_metrics


from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

# Define base estimator (XGBoost)
xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=2
)

# Wrap it in BaggingClassifier
bag_xgb = BaggingClassifier(
    base_estimator=xgb_base,
    n_estimators=10,   # number of bagged models
    random_state=0,
    bootstrap=True     # sampling with replacement
)

# Fit on training data
bag_xgb.fit(X_train, y_train)




y_probs_pred = bag_xgb.predict_proba(X_test)[:, 1]
y_pred=bag_xgb.predict(X_test)
bag_xgb_classiffier_metrics=measure_error(y_test, y_pred, label,y_probs_pred)
bag_xgb_classiffier_metrics.name="BaggingXCGBoost"
bag_xgb_classiffier_metrics


result_compilation = [
    random_forest_metrics,
    random_search_metrics,
    logistic_regression_metrics,
    voting_classiffier_metrics,
    best_gbc_classiffier_metrics,
    GV_ABC_classiffier_metrics,
    best_cat_classiffier_metrics,
    VCbest_cat_classiffier_metrics,
    bag_xgb_classiffier_metrics
    
]

# Convert each Series to dict, then build DataFrame
results_df = pd.DataFrame([s.to_dict() for s in result_compilation])

results_df.insert(0, "Model", [
    "Random Forest",
    "Random Search",
    "Logistic Regression",
    "Voting Classifier",
    "Best GBC",
    "GridSearch ABC",
    "CatBoosClassifier",
    "Voting_CatBoost_XGB",
    "BaggingXCGBoost"
])

results_df


train_data_set_path=r"/kaggle/input/playground-series-s5e8/train.csv"
df_train_data_set=pd.read_csv(train_data_set_path)


data_ohc=df_train_data_set.sample(50000)


data_ohc=one_hot_encoding(data_ohc,le,ohc,categorical_cols)
data_ohc, mappings = one_hot_encoding_ordinal_columns(data_ohc, ordinal_cols, le, ohc)
data_ohc[non_categorical_cols] = scaler.transform(data_ohc[non_categorical_cols])


y, X = data_ohc[target_column], data_ohc.drop(columns=target_column)


# Define the model
model = XGBClassifier(use_label_encoder=False, eval_metric='auc', verbosity=0)

# Define hyperparameter grid (tune these)
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Use ROC AUC as scoring metric
roc_auc = make_scorer(roc_auc_score, needs_proba=True)

# Setup randomized search
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=50,            # number of random parameter combinations to try
    scoring=roc_auc,
    cv=3,                 # 3-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1             # use all CPU cores
)

# Run the search
random_search.fit(X, y)

# Best parameters and best score
print("Best params:", random_search.best_params_)
print("Best ROC AUC:", random_search.best_score_)

#Fitting 3 folds for each of 50 candidates, totalling 150 fits
#Best params: {'subsample': 0.8, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 0.6}
#Best ROC AUC: 0.966870142882675


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier

best_cat_params = {
    'learning_rate': 0.1,
    'l2_leaf_reg': 5,
    'iterations': 200,
    'depth': 10,
    'border_count': 64
}

cat_model = CatBoostClassifier(
    verbose=0,
    eval_metric='AUC',
    random_state=42,
    **best_cat_params   # unpack the dictionary here
)


cat_model.fit(X, y)



# Combine models
estimators = [('best_cat', cat_model), ('random_search', random_search)]
VC_boost_xgb = VotingClassifier(estimators, voting='soft')
VC_boost_xgb.fit(X_train, y_train)



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


y_probs = VC.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

auc_score = roc_auc_score(y_test, y_probs)

fpr, tpr, thresholds = roc_curve(y_test, y_probs)


import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='VotingClassifier')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.datasets import make_classification



# Use only 30% of the data to speed up training
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)

# Convert to NumPy arrays to avoid KeyError during indexing
X_subset = np.array(X_subset)
y_subset = np.array(y_subset)

# Initialize model and cross-validation
rf = RandomForestClassifier(random_state=42)


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics
accuracy_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []

# Perform cross-validation
for train_index, test_index in kf.split(X_subset, y_subset):
    X_train, X_test = X_subset[train_index], X_subset[test_index]
    y_train, y_test = y_subset[train_index], y_subset[test_index]
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    roc_auc_scores.append(roc_auc_score(y_test, y_proba))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))

# Plotting the metrics
folds = np.arange(1, 6)
plt.figure(figsize=(10, 6))
plt.plot(folds, accuracy_scores, marker='o', label='Accuracy')
plt.plot(folds, roc_auc_scores, marker='o', label='ROC AUC')
plt.plot(folds, precision_scores, marker='o', label='Precision')
plt.plot(folds, recall_scores, marker='o', label='Recall')
plt.title('Cross-Validation Metrics per Fold (5% Subset)')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(folds)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



"""

# Define a black-box random forest model
black_box_model = RandomForestClassifier(random_state = 123, max_depth=25,
                             max_features=10, n_estimators=100,
                             bootstrap=True)
# Train the model
black_box_model.fit(X_train, y_train.values.ravel())

#predicting test set
y_blackbox = black_box_model.predict(X_test)

# Use permutation_importance to calculate permutation feature importances
feature_importances = permutation_importance(estimator=black_box_model, X = X_train, y = y_train, n_repeats=5,
                                random_state=123, n_jobs=2)

feature_importances.importances.sh

def visualize_feature_importance(importance_array):
    # Sort the array based on mean value
    sorted_idx = importance_array.importances_mean.argsort()
    # Visualize the feature importances using boxplot
    fig, ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    fig.tight_layout()
    ax.boxplot(importance_array.importances[sorted_idx].T,
               vert=False, labels=X_train.columns[sorted_idx])
    ax.set_title("Permutation Importances (train set)")
    plt.savefig('feature_importance.png')
    plt.show()

"""



def visualize_feature_importance(importance_array, X_train):
    """
    Visualize permutation feature importance using seaborn boxplots.

    Parameters:
    - importance_array: sklearn.inspection._permutation_importance.PermutationImportance object
    - X_train: DataFrame, training features used to match column names
    """
    # Prepare DataFrame for seaborn
    importances = importance_array.importances
    feature_names = X_train.columns
    n_repeats = importances.shape[1]

    # Create a DataFrame where each row is one repeat of one feature's importance
    data = {
        "Feature": [],
        "Importance": []
    }
    
    for idx, feature_name in enumerate(feature_names):
        data["Feature"].extend([feature_name] * n_repeats)
        data["Importance"].extend(importances[idx])

    df = pd.DataFrame(data)

    # Sort features by mean importance
    order = df.groupby("Feature")["Importance"].mean().sort_values().index

    # Plot using seaborn
    plt.figure(figsize=(16, 10))
    sns.boxplot(x="Importance", y="Feature", data=df, order=order, orient='h')
    plt.title("Permutation Importances (train set)")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()



# Get feature importances
importances = rf.feature_importances_

feature_names = X.columns

# Sort and plot
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.title("Feature Importances from Random Forest")




#Fitting 3 folds for each of 50 candidates, totalling 150 fits
#Best params: {'subsample': 0.8, 'reg_lambda': 2, 'reg_alpha': 1, 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'gamma': 0.1, 'colsample_bytree': 0.6}
#Best ROC AUC: 0.966870142882675





df_test_data=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
df_test_data=df_test_data.set_index("id")
df_test_data


data_ohc_pred = df_test_data.copy()


#data_ohc_pred=imputing_data(data_ohc_pred,non_categorical_cols,imputer)
data_ohc_pred=one_hot_encoding(data_ohc_pred,le,ohc,categorical_cols)
data_ohc_pred, mappings = one_hot_encoding_ordinal_columns(data_ohc_pred, ordinal_cols, le, ohc)
data_ohc_pred[non_categorical_cols] = scaler.transform(data_ohc_pred[non_categorical_cols])
y_pred=VC_boost_xgb.predict(data_ohc_pred)



y_probs_pred =VC_boost_xgb.predict_proba(data_ohc_pred)[:, 1]  # Probabilities for the positive class
data_ohc_pred["y"]=y_probs_pred
results_probs_df=pd.DataFrame(data_ohc_pred["y"])
results_probs_df


results_probs_df.to_csv("results.csv")

