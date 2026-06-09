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
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/microsoft-malware-prediction/train.csv',nrows=1000000, low_memory=False)
df.head()


df.info()


# Display all columns with their null value counts (not truncated)
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]  # Only columns with at least one null value
if not null_counts.empty:
    print(null_counts.to_string())
else:
    print('No columns with null values.')


# Drop columns with 10% or more missing data
threshold = 0.10
missing_fraction = df.isnull().mean()
cols_to_drop = missing_fraction[missing_fraction >= threshold].index.tolist()
df = df.drop(columns=cols_to_drop)
print(f"Dropped columns: {cols_to_drop}")


from sklearn.feature_selection import VarianceThreshold

# Select only numeric columns for variance threshold
numeric_df = df.select_dtypes(include=[np.number])

# Apply VarianceThreshold to drop features with low variance (default threshold=0)
selector = VarianceThreshold(threshold=0.01)
selector.fit(numeric_df)

# Get columns that are kept
features_kept = numeric_df.columns[selector.get_support(indices=True)].tolist()
features_dropped = [col for col in numeric_df.columns if col not in features_kept]

# Drop low variance features from the main dataframe
df = df.drop(columns=features_dropped)
print(f"Dropped low variance features: {features_dropped}")


df.drop(columns=["MachineIdentifier","ProductName" ])


df.duplicated().sum() # Check for duplicate rows in the dataframe


df.describe().T


df.describe(include='object').T


def identify_column_types(df, threshold_unique=10):
    continuous = []
    discrete = []
    categorical = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            if unique_count > threshold_unique:
                continuous.append(col)
            else:
                discrete.append(col)
        else:
            categorical.append(col)
    
    return continuous, discrete, categorical

continuous_cols, discrete_cols, categorical_cols = identify_column_types(df)
print(f"Continuous: {len(continuous_cols)}")
print(f"Discrete: {len(discrete_cols)}")
print(f"Categorical: {len(categorical_cols)}")


for col in continuous_cols:
    df[col].fillna(df[col].median(), inplace=True)


for col in discrete_cols + categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


missing_cols = df.columns[df.isnull().any()].tolist()
len(missing_cols) , missing_cols


for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound,
                       np.where(df[col] > upper_bound, upper_bound, df[col]))
print("Outliers in continuous columns have been capped using the IQR method.")


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
print("MinMax scaling applied to continuous columns.")


from sklearn.preprocessing import OrdinalEncoder

# Apply OrdinalEncoder to categorical columns
encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
print("Ordinal encoding applied to categorical columns.")


from sklearn.feature_selection import mutual_info_regression

# ----- INPUTS -----
# define categorical/discrete columns
target_col = 'HasDetections'
categorical_discrete = categorical_cols + discrete_cols

# ----- NUMERICAL FEATURES -----
corr_scores = df[continuous_cols].corrwith(df[target_col]).abs()

# ----- CATEGORICAL / DISCRETE FEATURES -----
mi_scores = pd.Series(
    mutual_info_regression(df[categorical_discrete], df[target_col], random_state=42),
    index=categorical_discrete
) if len(categorical_discrete) > 0 else pd.Series(dtype=float)

# ----- COMBINE AND SORT -----
combined_scores = pd.concat([corr_scores, mi_scores])
top_20_features = combined_scores.sort_values(ascending=False).head(21)

print("Top 20 correlated features with target:\n")
print(top_20_features)


top_20_features.sort_values().plot.barh(figsize=(8,6))
plt.title("Top 20 Features Most Correlated with Target")
plt.xlabel("Correlation / Mutual Information Score")
plt.show()


new_df = df[top_20_features.index.tolist()]
new_df.shape
print(new_df.info())


y = new_df['HasDetections']
X = new_df.drop(columns=['HasDetections'])
print (X.shape, y.shape)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



# ----- SPLIT DATA -----

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Optional: scale features (especially for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- DEFINE MODELS -----
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    #"SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# ----- TRAIN & EVALUATE -----
results = []
for name, model in models.items():
    # Choose scaled data for SVM, raw otherwise
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    results.append([name, acc, f1, roc])

# ----- RESULTS -----
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score', 'ROC-AUC'])
results_df = results_df.sort_values(by='ROC-AUC', ascending=False)

print("Model Performance Comparison:")
print(results_df)

