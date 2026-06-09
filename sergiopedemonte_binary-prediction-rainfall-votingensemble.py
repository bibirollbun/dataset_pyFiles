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


import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier


# Load libraries
import numpy
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, LabelBinarizer, RobustScaler
from sklearn.feature_selection import SelectKBest, RFECV, f_classif
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier

import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA

plt.style.use('ggplot')
import seaborn as sns


# Import warnings, silent warns

import warnings
warnings.filterwarnings('ignore')


def calculate_missing_values(df):

    # Calculate missing values percentage by column, format as %

    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().sum() * 100 / len(df)

    # Format the percentage as a string with '%'
    missing_percentage = missing_percentage.apply(lambda x: f'{x:.2f}%')

    # Print the results
    return missing_percentage


# funtion "evaluate" to evaluate df with LogisticRegression and roc_auc score

def evaluate(X, y):
  """
  Evaluates a DataFrame using LogisticRegression and cross-validation.

  Args:
    df: The input DataFrame.

  Returns:
    The mean ROC AUC score across the cross-validation folds.
  """

  model = LogisticRegression()
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
  scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

  return scores.mean()



def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


def pca_inspired(df):
    X = pd.DataFrame()
    X["Feature1"] = df.GrLivArea + df.TotalBsmtSF
    X["Feature2"] = df.YearRemodAdd * df.TotalBsmtSF
    return X


def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca


pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]


def make_mi_scores(X, y, discrete_features):

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def load_df():
  path = "/kaggle/input/playground-series-s5e3/train.csv"
  df = pd.read_csv(path, index_col= "id")
  return df


path = "/kaggle/input/playground-series-s5e3/train.csv"
def load_data(path):
    df = pd.read_csv(path, index_col= "id")
    return df


df = load_data(path)
df.head()


path = "/kaggle/input/playground-series-s5e3/test.csv"
X_test = load_data(path)
X_test.head()


X_test.shape


path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"
submission = load_data(path)
submission.head()


calculate_missing_values(df)


calculate_missing_values(X_test)


# Shape of the dataset
print(df.shape)


# Data Types
print(df.dtypes)


# Descriptive Statistics
#set_option('precision', 1)
df.describe()


# Class Distribution
print(df.groupby('rainfall').size())


X = df.copy()
y = X.pop('rainfall')

evaluate(X,y).round(3)


# ANOVA correlation coefficient for feature selection

from sklearn.feature_selection import f_classif

# Assuming X is your feature matrix and y is your target variable
X = df.drop('rainfall', axis=1)
y = df['rainfall']

# Apply ANOVA F-value for feature selection
f_statistic, p_values = f_classif(X, y)

# Create a DataFrame to store the results
feature_scores = pd.DataFrame({'Feature': X.columns, 'F-Statistic': f_statistic, 'P-value': p_values})

# Sort the features by F-statistic in descending order
feature_scores = feature_scores.sort_values('F-Statistic', ascending=False)

# Select the top 4 features
selected_features = feature_scores['Feature'].head(7).tolist()

print("Top features based on ANOVA:")
selected_features


feature_scores


# create a new dataframe with selected_features

# Create a new DataFrame with only the selected features
df_selected = df[selected_features]

# Print the new DataFrame
print(df_selected.head())



X = df_selected.copy()
#y = X.pop('rainfall')

evaluate(X,y).round(3)


df = df_selected.copy()


df_selected.isnull().sum()


# # Skew for numerical features
print(df.skew())


# # Histograms
df.hist(figsize=(12, 8))
plt.show()


# # Density Plots
df.plot(kind='density', subplots=True, layout=(7, 3), figsize=(12, 8), sharex=False)
plt.show()


from scipy import stats
import pandas as pd
import numpy as np

# Reciprocal transformation – 1 / x
# Exponential transformation – exp(x)

def transform_and_compare(df):
    """
    Applies log, sqrt, and Box-Cox transformations to specified columns,
    compares their skewness, and returns the best transformation for each.
    """
    columns_to_transform = [col for col in df.columns if col not in ["day", "rainfall"]]
    results = {}

    for col in columns_to_transform:
      #Check for negative values
      if (df[col] < 0).any():
          print(f"Column '{col}' contains negative values, skipping Box-Cox and Log transforms")
      #    continue
      original_skew = df[col].skew()

      # Log Transform
      if (df[col] < 0).any():
        log_skew = np.nan  # Assign NaN to log_skew if there are negative values
      if (df[col] > 0).all():
        log_transformed = np.log1p(df[col]) #Adding 1 to avoid log(0)
        log_skew = log_transformed.skew()

      # Square Root Transform
      sqrt_transformed = np.sqrt(df[col])
      sqrt_skew = sqrt_transformed.skew()

      # PowerTransformer
      pt_transformed = PowerTransformer().fit_transform(df[col].values.reshape(-1, 1))
      pt_skew = pd.Series(pt_transformed.flatten()).skew()

      # Box-Cox Transform
      # if (df[col] < 0).any():
      #   boxcox_skew = np.nan
      if (df[col] > 0).all():
        # Check if the column contains constant values
        if np.all(df[col] == df[col].iloc[0]):
            print(f"Column '{col}' contains constant values, skipping Box-Cox transform")
            boxcox_skew = np.nan  # Assign NaN to boxcox_skew
            lmbda = np.nan  # Assign NaN to lmbda
        else:
            boxcox_transformed, lmbda = stats.boxcox(df[col] + 0.3) #Adding small value to avoid zero values
            boxcox_skew = pd.Series(boxcox_transformed).skew()

      # Compare skewness and choose best transform
      skewness_values = {
          "original": original_skew,
          "log": log_skew,
          "sqrt": sqrt_skew,
          "PowerTransform": pt_skew,
          "boxcox": boxcox_skew,
      }

      best_transform = min(skewness_values, key=lambda k: abs(skewness_values[k]))
      results[col] = {
          "original_skew": original_skew,
          "log_skew": log_skew,
          "sqrt_skew": sqrt_skew,
          "PowerTransform": pt_skew,
          "boxcox_skew": boxcox_skew,
          "boxcox_lambda": lmbda,
          "best_transform": best_transform
      }

    return pd.DataFrame(results)

# Assuming 'df' is your DataFrame
transform_df = transform_and_compare(df)
transform_df.T.head(10)


def transform(df):
  # Features Transforms
  pt = PowerTransformer()
  pt_features = ['cloud', 'humidity', "windspeed", "maxtemp", "dewpoint"]

  # Create a copy of the input DataFrame to avoid modifying the original DataFrame
  #df_transformed = df.copy()

  # Power Transformer
  pt = PowerTransformer()
  df[pt_features] = pt.fit_transform(df[pt_features])

  # sqrt transform
  sqrt_features = ['sunshine', "pressure"]
  df[sqrt_features] = np.sqrt(df[sqrt_features])

  return df # Return the transformed DataFrame


df_transformed = transform(df)


df_transformed.skew()


df_transformed.shape


df_transformed.isnull().sum()


df_transformed.plot(kind='density', subplots=True, layout=(7, 3), figsize=(12, 8), sharex=False)
plt.show()


X = df_transformed.copy()
#y = X.pop('rainfall')

evaluate(X,y).round(3)


df = df_transformed.copy()


df.isnull().sum()


# Box and Whisker Plots
df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(12, 8), sharex=False)
plt.show()


df_out_test = df.copy()


def scaling(df):
    features_to_scale = ['humidity', 'pressure']

    # Create a RobustScaler object
    robust_scaler = RobustScaler()

    # Apply RobustScaler to the specified features of your DataFrame
    df[features_to_scale] = robust_scaler.fit_transform(df[features_to_scale])

    return df


df_out_test = scaling(df)


# Box and Whisker Plots
df_out_test.plot(kind='box', subplots=True, layout=(4, 4), figsize=(12, 8), sharex=False)
plt.show()


X = df_out_test.copy()
#y = X.pop('rainfall')

evaluate(X,y).round(3)


df_out_test.isnull().sum()


df = df_out_test.copy()


# Scatter Plot Matrix
scatter_matrix(df, figsize=(12, 12))
plt.show()


# Correlation Matrix Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(numeric_only=True), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0, len(df.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
# Use df.columns to match the number of ticks for x-axis labels
ax.set_xticklabels(df.columns)
# Use df.columns to match the number of ticks for y-axis labels
ax.set_yticklabels(df.columns)
plt.show()


df_fe_test = df.copy()


df.columns


X = df.copy()


pca, X_pca, loadings = apply_pca(df)


mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
mi_scores


X_pca


loadings


plot_variance(pca)


# Restablecer los índices de ambos DataFrames
df = df.reset_index(drop=True)
X_pca = X_pca.reset_index(drop=True) # Now you can reset the index of the DataFrame
df_pca = pd.concat([df, X_pca[['PC2', 'PC1']]], axis=1)


df_pca


# Show dataframe sorted by PC1
idx = X_pca["PC1"].sort_values(ascending=False).index
cols = df.columns # remove the parenthesis
df.loc[idx, cols]


pca.explained_variance_ratio_


X = df_pca.copy()
#y = X.pop('rainfall')
evaluate(X,y).round(3)


df = df_pca.copy()


# Test lag features

def create_lag_features(df, cols, lags):
  """
  Creates lag features for given columns in a DataFrame.

  Args:
    df: The input DataFrame.
    cols: A list of column names to create lag features for.
    lags: A list of lag values.

  Returns:
    A DataFrame with the added lag features.
  """
  df_with_lags = df.copy()
  for col in cols:
    for lag in lags:
      df_with_lags[f'{col}_lag_{lag}'] = df_with_lags[col].shift(lag)
  return df_with_lags

# Example usage
lags = [1,2]  # Create lag features for 1 and 2 time steps
cols = ['cloud','sunshine', 'humidity', 'windspeed', 'dewpoint', 'maxtemp']


df = create_lag_features(df, cols, lags)
print(df.head())


df["rain"] = y


# apply simple imputer with strategy "mean"

# Assuming 'df' is your DataFrame and you want to impute missing values in all numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])



X = df.copy()
y = X.pop("rain")
evaluate(X,y).round(3)


# Apply ANOVA F-value for feature selection
f_statistic, p_values = f_classif(X, y)

# Create a DataFrame to store the results
feature_scores = pd.DataFrame({'Feature': X.columns, 'F-Statistic': f_statistic, 'P-value': p_values})

# Sort the features by F-statistic in descending order
feature_scores = feature_scores.sort_values('F-Statistic', ascending=False)

# Select the top 4 features
selected_features_with_lags = feature_scores['Feature'].head(6).tolist()

print("Top features based on ANOVA:")
selected_features_with_lags


feature_scores


X = df[selected_features_with_lags].copy()
#y = X.pop('rainfall')
evaluate(X,y).round(3)


df = df[selected_features_with_lags].copy()
df["rain"] = y


selected_features


def Preprocess(df):

  df = df[selected_features]
  df = transform(df)
  df = scaling(df)

  pca, X_pca, loadings = apply_pca(df)

  # Restablecer los índices de ambos DataFrames
  df = df.reset_index(drop=True)
  X_pca = X_pca.reset_index(drop=True) # Now you can reset the index of the DataFrame
  df = pd.concat([df, X_pca[['PC2', 'PC1']]], axis=1)

  df = create_lag_features(df, cols, lags)
  
  # Simpleimputer

  numerical_cols = df.select_dtypes(include=np.number).columns
  imputer = SimpleImputer(strategy='mean')
  df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

  df = df[selected_features_with_lags].copy()

#  df = Kmeans(df)
#  df = categorize_and_get_dummies(df)

  return df

X_test = Preprocess(X_test)

print(X_test.shape)


from sklearn.ensemble import VotingClassifier
# Define the base models
estimators = []
estimators.append(('logistic', LogisticRegression()))
estimators.append(('LinDiscr', LinearDiscriminantAnalysis()))
estimators.append(('ETrees', ExtraTreesClassifier(n_estimators=100)))

# Create the voting classifier
ensemble = VotingClassifier(estimators=estimators, voting='soft')


# predict X_test probability

# Assuming 'model' is your trained XGBoost model and 'X_test' is your test data
# Load the trained model (replace 'your_model.pkl' with the actual file path)
#model = XGBClassifier()
#model = CatBoostClassifier()
#model = LogisticRegression()
#model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
model = VotingClassifier(estimators=estimators, voting='soft')

#model.load_model('xgb_model.json') # Load the model

X = df.copy()

y = X.pop('rain')
model.fit(X,y)

# Make predictions on the test set
y_pred_prob = model.predict_proba(X_test)[:,1]

print(y_pred_prob[:10])


# Creating 'submission' dataframe to store predictions with ids
submission2 = pd.DataFrame({'id': submission.index, 'rainfall': y_pred_prob})

# Save submission2 (with predicted probabilities)
submission2.to_csv('submission_data.csv', index=False)

submission2

