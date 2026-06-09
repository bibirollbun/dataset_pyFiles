# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='xgboost.core')
warnings.filterwarnings("ignore", message="1 warning generated.")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)




train_data_set_path=r"/kaggle/input/playground-series-s5e7/train.csv"
df_train_data_set=pd.read_csv(train_data_set_path)
df_train_data_set.head()


df_train_data_set=df_train_data_set.set_index("id")


df_train_data_set.isnull().sum()
#df_train_data_set.isna().sum()


df_train_data_set.dtypes


df_train_data_set.describe()


# The intention with this dataframe is to use it for imputing NaN values
mean_data_frame=df_train_data_set.describe().loc["mean"]
#median_data_frame=df_train_data_set.describe().loc["50%"]
mean_data_frame.loc["Time_spent_Alone"]


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df_train_data_set[["Time_spent_Alone","Social_event_attendance","Going_outside","Friends_circle_size","Post_frequency"]].items():
    sns.histplot(v, ax=axs[index], kde=True)
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


import seaborn as sns
sns.set_context('talk')
sns.pairplot(df_train_data_set,hue="Personality",corner=True)



import math
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Replace with your actual DataFrame and categorical column list
df = df_train_data_set
cat_columns = categorical_variables

# Automatically determine grid size
list_len = len(cat_columns)
number_of_column = math.ceil(math.sqrt(list_len))
rows_plot = math.ceil(list_len / number_of_column)

# Create subplots
fig, axes = plt.subplots(rows_plot, number_of_column, figsize=(5 * number_of_column, 5 * rows_plot))
axes = axes.reshape(-1, number_of_column)  # Ensure 2D shape

# Plot each categorical column
counter = 0
for cat_column in cat_columns:
    trace_x = counter // number_of_column
    trace_y = counter % number_of_column
    ax = axes[trace_x][trace_y]
    sns.countplot(ax=ax, x=cat_column, data=df, hue='Personality')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    counter += 1

# Hide any unused subplots
for i in range(counter, rows_plot * number_of_column):
    fig.delaxes(axes[i // number_of_column][i % number_of_column])

plt.tight_layout()
plt.show()




df_train_data_set.dtypes


df_train_data_set.dtypes.value_counts()


# Select the object (string) columns
mask = df_train_data_set.dtypes == 'object'
categorical_cols = df_train_data_set.columns[mask]
categorical_cols


df_unique = df_train_data_set[categorical_cols].nunique().to_frame().reset_index()
df_unique.columns = ['Variable','DistinctCount']
df_unique


# Select non categorical columns (string) columns
mask = df_train_data_set.dtypes == 'float64'
non_categorical_cols = df_train_data_set.columns[mask]
non_categorical_cols


#From this list, it is possible to observe that we have an ordinal variable month
ordinal_cols=["Personality"]


categorical_cols=set(categorical_cols)-set(ordinal_cols)



def imputing_data(data_ohc,non_categorical_cols,imputer):
# Fit on training data

    # Transform training data
    data_ohc[non_categorical_cols] = pd.DataFrame(
        imputer.transform(data_ohc[non_categorical_cols]),
        columns=data_ohc[non_categorical_cols].columns,
        index=data_ohc[non_categorical_cols].index
    )
    data_ohc[non_categorical_cols] = data_ohc[non_categorical_cols].round()

    return data_ohc




def data_imputation_using_mean_training_data(data_ohc,non_categorical_cols):
    for col in non_categorical_cols:
        data_ohc[col].fillna(data_ohc[col].mean(),inplace=True)
    return  data_ohc




#Reserved for Imputation using mean or median
def data_imputation_mean(col_to_escale,data_ohc):
    
    for column in col_to_scale:
        print(column)
        print(mean_data_frame.loc[column])
        data_ohc[column]=mean_data_frame.loc[column] #comes from test_data, describe
    
    return data_ohc

    


# Why using one hot encoding
#https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/

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

      



# Apply log1p transformation to selected columns
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

"""
# Load the saved scaler
scaler = joblib.load('minmax_scaler.pkl')
# Transform the prediction data
data_ohc_pred[col_to_scale] = scaler.transform(data_ohc_pred[col_to_scale])
"""


    



data_ohc = df_train_data_set.copy()#creating a copy of trianing data


 # Imputation
imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42, max_iter=20)
imputer.fit(data_ohc[non_categorical_cols])


data_ohc=imputing_data(data_ohc,non_categorical_cols,imputer) 

# One Hot Encoding Categorical Columns

le = LabelEncoder()
ohc = OneHotEncoder(sparse=False)
data_ohc =one_hot_encoding(data_ohc,le,ohc,categorical_cols)

#na_columns=[column for column in data_ohc.columns if column.endswith("_nan")]
#na_columns

# One Hot Encoding Ordinal Columns
data_ohc, mappings = one_hot_encoding_ordinal_columns(data_ohc, ordinal_cols, le, ohc)

# Data Regularization
data_ohc=log1p_regularization(data_ohc,non_categorical_cols)



data_ohc


target_column='Personality'


def measure_error(y_true, y_pred, label):
    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'f1': f1_score(y_true, y_pred)},
                      name=label)


def confusion_matrix_graph(y,y_pred):
    sns.set_context('talk')
    cm = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d')


# Set up X and y variables
y, X = data_ohc[target_column], data_ohc.drop(columns=target_column)
# Split the data into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
results_compilation_temp=pd.DataFrame(measure_error(y_test, y_pred, 'Random Forest'))
results_compilation_temp
confusion_matrix_graph(y_test,y_pred)



y_pred=model.predict(X)
model.fit(X,y)
y_pred=model.predict(X)
confusion_matrix_graph(y,y_pred)

results_compilation_RF=pd.DataFrame(measure_error(y, y_pred, 'Random Forest Full Data'))
results_compilation_RF



param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'class_weight': [None, 'balanced']
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=30,                # Number of combinations to try
    cv=5,                     # 5-fold cross-validation
    scoring='accuracy',       # Or use 'f1', 'roc_auc' for classification
    verbose=1,
    n_jobs=-1,                # Use all processors
    random_state=42
)

search.fit(X, y)
best_rf = search.best_estimator_
print("Best Parameters:", search.best_params_)


y_pred=best_rf.predict(X)
results_compilation_RF_hyperparameter=pd.DataFrame(measure_error(y, y_pred, 'Random Forest'))
results_compilation_RF_hyperparameter
confusion_matrix_graph(y,y_pred)




# Iterate through various possibilities for number of trees
tree_list = [15, 25, 50, 100, 200, 400]

# The parameters to be fit
param_grid = {'n_estimators': tree_list,
              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
              'subsample': [1.0, 0.5],
              'max_features': [1, 2, 3, 4]}

# The grid search object
GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42),
                      param_grid=param_grid,
                      scoring='accuracy',
                      n_jobs=-1)

# Do the grid search
GV_GBC = GV_GBC.fit(X_train, y_train)


# The best model
GV_GBC.best_estimator_
y_pred=GV_GBC.predict(X)
results_compilation_GV_GBC=pd.DataFrame(measure_error(y, y_pred, 'GradientBoostingClassifier'))
results_compilation_GV_GBC
 


result_compilation=pd.concat([results_compilation_RF,results_compilation_RF_hyperparameter,results_compilation_GV_GBC],axis=1)
result_compilation


df_test_data=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_test_data=df_test_data.set_index("id")
df_test_data


data_ohc_pred = df_test_data.copy()


data_ohc_pred=imputing_data(data_ohc_pred,non_categorical_cols,imputer)
data_ohc_pred=one_hot_encoding(data_ohc_pred,le,ohc,categorical_cols)
#data_ohc_pred, mappings = one_hot_encoding_ordinal_columns(data_ohc_pred, ordinal_cols, le, ohc)
data_ohc=log1p_regularization(data_ohc_pred,non_categorical_cols)

#y_pred=GV_GBC.predict(data_ohc_pred)
y_pred=model.predict(data_ohc_pred)
data_ohc_pred["Personality"]=y_pred
results_df=pd.DataFrame(data_ohc_pred["Personality"])
results_df["Personality"]=results_df["Personality"].map(mappings["Personality"])


results_df


results_df.to_csv("results.csv")

