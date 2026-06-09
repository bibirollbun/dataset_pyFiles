# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train=pd.read_csv("../input/recruitment-task-for-gdsc-ml/MiNDAT.csv")
df_test=pd.read_csv("../input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv")


df_train.head()


df_train.describe()


df_train=df_train.drop('+U@',axis=1)
df_test=df_test.drop('+U@',axis=1)


df_train.isnull().sum()



columns_with_outliers = [
    '&%)LTaWRb', '.6AvGp', 'T\!', 'vzo."', '.o<m', '!;@Jw', 'ZVf', 'Jv[i', 
    'hp!', "0HU2N='U", '3I\y', '@V9', 'fPqsI', ']xq', 'ZrK', '9Z/5)2', 
    ';<"<i(T', '%IiL7w', '~7*', '^%a;', 'i]7V', '@wnsk>R'
]
for col in columns_with_outliers:
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_before = df_train[(df_train[col] < lower_bound) | (df_train[col] > upper_bound)].shape[0]
    
    df_train[col] = df_train[col].clip(lower=lower_bound, upper=upper_bound)
    
    outliers_after = df_train[(df_train[col] < lower_bound) | (df_train[col] > upper_bound)].shape[0]
    
    print(f"- Column '{col}': Capped {outliers_before} outliers. ({outliers_after} remain).")

print("\nOutlier capping process complete.")


if "LOCAL_IDENTIFIER" in df_train.columns:
    df_train = df_train.drop(columns=["LOCAL_IDENTIFIER"])

for i in df_train.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(6,4))
    
    sns.histplot(df_train[i].dropna(), bins=30, kde=True)
    plt.title(f"Distribution of {i}")
    plt.xlabel(i)
    plt.ylabel("Frequency")
    
    skewness = df_train[i].skew()
    print(f"Skewness of {i}: {skewness:.3f}")
    
    plt.show()


mean_cols = ['vzo."', 'hp!', '@wnsk>R', '&%)LTaWRb', '@V9', 'T\!', '.o<m', '~7*', '9Z/5)2', '%IiL7w', '!;@Jw', 'fPqsI', 'i]7V', ';<"<i(T', ']xq', '^%a;', "0HU2N='U", "ZrK", ".6AvGp", "3I\y"]
median_cols = ['ZVf', 'Jv[i']

for col in mean_cols:
    df_train[col].fillna(df_train[col].mean(), inplace=True)

for col in median_cols:
    df_train[col].fillna(df_train[col].median(), inplace=True)


    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # --- 1. Display Initial DataFrame State ---
    # Assuming 'df_train' is already loaded and contains missing values.
    print("--- df_train Before Imputation ---")
    print(df_train.head())
    print("\nMissing values in df_train before imputation:")
    print(df_train.isnull().sum())
    
    
    # --- 2. Separate Numerical and Categorical Columns ---
    # We'll impute only the numerical columns and then join them back.
    numerical_cols = df_train.select_dtypes(include=np.number).columns
    categorical_cols = df_train.select_dtypes(exclude=np.number).columns
    
    df_numerical = df_train[numerical_cols]
    df_categorical = df_train[categorical_cols]
    
    
    # --- 3. Set up the Imputation Pipeline for Numerical Data ---
    # BEST PRACTICE: KNN is distance-based, so it's crucial to scale your data first.
    # A Pipeline makes this easy and prevents data leakage.
    
    # Create a scaler object
    scaler = StandardScaler()
    
    # Create the KNN imputer object
    # n_neighbors (k) is the most important parameter to tune.
    # It's the number of neighbors that will be used to vote for the imputed value.
    knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
    
    # Create the pipeline that first scales the data, then imputes.
    pipeline = Pipeline([
        ('scaler', scaler),
        ('imputer', knn_imputer)
    ])
    
    
    # --- 4. Apply the Imputation to Numerical Columns ---
    # The pipeline will first scale the data, then impute the missing values.
    # The result is a NumPy array.
    imputed_numerical_data_scaled = pipeline.fit_transform(df_numerical)
    
    # We need to inverse_transform the scaling to get the values back in their original range.
    # We can access the scaler step from our pipeline to do this.
    imputed_numerical_data = pipeline.named_steps['scaler'].inverse_transform(imputed_numerical_data_scaled)
    
    # Convert the imputed NumPy array back into a pandas DataFrame.
    df_numerical_imputed = pd.DataFrame(imputed_numerical_data, columns=numerical_cols)
    
    
    # --- 5. Overwrite the Original DataFrame ---
    # Combine the imputed numerical columns with the original categorical columns.
    # We reset the index of the categorical dataframe to ensure a clean concatenation.
    df_train = pd.concat([df_numerical_imputed, df_categorical.reset_index(drop=True)], axis=1)
    
    # Ensure the column order is the same as the original DataFrame
    df_train = df_train[df_train.columns]
    
    
    print("\n\n--- df_train After Imputation ---")
    print(df_train.head())
    print("\nMissing values after imputation (checking the updated df_train):")
    print(df_train.isnull().sum())



from sklearn.preprocessing import OrdinalEncoder

# Step 1: Impute categorical columns with mode (from training set only)
cat_fill_values = {}
cat_cols = df_train.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:
    mode_val = df_train[col].mode()[0]    
    cat_fill_values[col] = mode_val
    df_train[col] = df_train[col].fillna(mode_val)

for col, value in cat_fill_values.items():
    if col in df_test.columns:  
        df_test[col] = df_test[col].fillna(value)

# Step 2: Ordinal Encoding (fit on train, transform both)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols])
df_test[cat_cols] = encoder.transform(df_test[cat_cols])



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --- 1. Setup and Data Loading ---

# Load the dataset
try:
    df = df_train
except FileNotFoundError:
    print("MiNDAT.csv not found. Please make sure the file is in the correct directory.")
    # Create a dummy dataframe for demonstration if the file is not found
    df = pd.DataFrame(np.random.randn(100, 30), columns=[f'col_{i}' for i in range(30)])

# The user-provided list of unimodal columns
unimodal_cols = [
    'vzo."', 'hp!', '@wnsk>R', '&%)LTaWRb', '@V9', 'T\!', '.o<m', '~7*',
    '9Z/5)2', '%IiL7w', '!;@Jw', 'fPqsI', 'i]7V', ';<"<i(T', ']xq',
    '^%a;', "0HU2N='U'", "ZrK", ".6AvGp", "3I\y", 'ZVf', 'Jv[i'
]

# Create a directory to save the PCA plots
output_dir = 'pca_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# --- 2. Identify Multimodal Columns and Binarize ---

# Identify multimodal columns by excluding unimodal ones
all_cols = df.columns.tolist()
multimodal_cols = [col for col in all_cols if col not in unimodal_cols]

# Filter for only the numeric multimodal columns, as PCA requires numeric input
numeric_multimodal_cols = df[multimodal_cols].select_dtypes(include=np.number).columns.tolist()

print(f"Found {len(numeric_multimodal_cols)} numeric multimodal columns to process.")

# Create a new dataframe to hold the original data and the new binary features
df_binarized = df.copy()


# --- 3. PCA Visualization and Binarization Loop ---

# Loop through each numeric multimodal column
for col in numeric_multimodal_cols:
    print(f"Processing column: {col}...")

    # --- Binarization Step ---
    # Calculate the mean of the original column data (ignoring NaNs)
    mean_val = df[col].mean()
    # Create the new binary feature name
    binary_col_name = f'{col}_binary'
    # Add the binary column to our new dataframe
    df_binarized[binary_col_name] = (df[col] > mean_val).astype(int)


    # --- PCA and Visualization Step ---
    # Drop missing values for PCA and visualization for this specific column
    # We create a temporary dataframe for this to avoid altering the main data
    temp_df = df[[col]].dropna()

    if temp_df.empty:
        print(f"  -> Skipping column '{col}' because it contains no valid data.")
        continue

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(temp_df[[col]])

    # Apply PCA. For a single column, this simplifies to centering the data.
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(scaled_data)

    # Create a dataframe for plotting
    plot_df = pd.DataFrame(data=principal_component, columns=['PC1'])
    # Add the binary feature to the plot dataframe (aligning by index)
    plot_df[binary_col_name] = df_binarized.loc[temp_df.index, binary_col_name]
    # We add a constant 'y' value to create a 1D scatter plot
    plot_df['y'] = 0

    # --- Plotting ---
    plt.figure(figsize=(12, 4))
    sns.scatterplot(
        x='PC1',
        y='y',
        hue=binary_col_name,
        data=plot_df,
        palette='viridis',
        s=100, # size of points
        alpha=0.7
    )

    plt.title(f'1D PCA of "{col}" (Colored by Binary Feature)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('')
    plt.yticks([]) # Hide the y-axis ticks
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.legend(title=f'Value > Mean({mean_val:.2f})')

    # Save the plot
    plot_filename = os.path.join(output_dir, f'pca_{col.replace("/", "_")}.png')
    plt.savefig(plot_filename)
    plt.close() # Close the plot to free up memory

print(f"\nAll PCA plots have been saved to the '{output_dir}' directory.")


# --- 4. Save the Final DataFrame ---

# Save the dataframe with all the new binary features to a new CSV file.
output_csv_path = 'MiNDAT_with_binary_features.csv'
df_binarized.to_csv(output_csv_path, index=False)

print(f"The final dataframe with binary features has been saved to '{output_csv_path}'")
print("\nFirst 5 rows of the new dataframe:")
print(df_binarized.head())



df_train.isnull().sum()



df_test.isnull().sum()


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- 1. Display Initial DataFrame State ---
# Assuming 'df_train' is already loaded and contains missing values.
print("--- df_test Before Imputation ---")
print(df_test.head())
print("\nMissing values in df_test before imputation:")
print(df_test.isnull().sum())


# --- 2. Separate Numerical and Categorical Columns ---
# We'll impute only the numerical columns and then join them back.
numerical_cols = df_test.select_dtypes(include=np.number).columns
categorical_cols = df_test.select_dtypes(exclude=np.number).columns

df_numerical = df_test[numerical_cols]
df_categorical = df_test[categorical_cols]


# --- 3. Set up the Imputation Pipeline for Numerical Data ---
# BEST PRACTICE: KNN is distance-based, so it's crucial to scale your data first.
# A Pipeline makes this easy and prevents data leakage.

# Create a scaler object
scaler = StandardScaler()

# Create the KNN imputer object
# n_neighbors (k) is the most important parameter to tune.
# It's the number of neighbors that will be used to vote for the imputed value.
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')

# Create the pipeline that first scales the data, then imputes.
pipeline = Pipeline([
    ('scaler', scaler),
    ('imputer', knn_imputer)
])


# --- 4. Apply the Imputation to Numerical Columns ---
# The pipeline will first scale the data, then impute the missing values.
# The result is a NumPy array.
imputed_numerical_data_scaled = pipeline.fit_transform(df_numerical)

# We need to inverse_transform the scaling to get the values back in their original range.
# We can access the scaler step from our pipeline to do this.
imputed_numerical_data = pipeline.named_steps['scaler'].inverse_transform(imputed_numerical_data_scaled)

# Convert the imputed NumPy array back into a pandas DataFrame.
df_numerical_imputed = pd.DataFrame(imputed_numerical_data, columns=numerical_cols)


# --- 5. Overwrite the Original DataFrame ---
# Combine the imputed numerical columns with the original categorical columns.
# We reset the index of the categorical dataframe to ensure a clean concatenation.
df_test = pd.concat([df_numerical_imputed, df_categorical.reset_index(drop=True)], axis=1)

# Ensure the column order is the same as the original DataFrame
df_test = df_test[df_test.columns]


print("\n\n--- df_train After Imputation ---")
print(df_test.head())
print("\nMissing values after imputation (checking the updated df_train):")
print(df_test.isnull().sum())



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# --- 1. Setup and Data Loading ---

# Load the dataset
try:
    df = df_test
except FileNotFoundError:
    print("MiNDAT.csv not found. Please make sure the file is in the correct directory.")
    # Create a dummy dataframe for demonstration if the file is not found
    df = pd.DataFrame(np.random.randn(100, 30), columns=[f'col_{i}' for i in range(30)])

# The user-provided list of unimodal columns
unimodal_cols = [
    'vzo."', 'hp!', '@wnsk>R', '&%)LTaWRb', '@V9', 'T\!', '.o<m', '~7*',
    '9Z/5)2', '%IiL7w', '!;@Jw', 'fPqsI', 'i]7V', ';<"<i(T', ']xq',
    '^%a;', "0HU2N='U'", "ZrK", ".6AvGp", "3I\y", 'ZVf', 'Jv[i'
]

# Create a directory to save the PCA plots
output_dir = 'pca_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# --- 2. Identify Multimodal Columns and Binarize ---

# Identify multimodal columns by excluding unimodal ones
all_cols = df.columns.tolist()
multimodal_cols = [col for col in all_cols if col not in unimodal_cols]

# Filter for only the numeric multimodal columns, as PCA requires numeric input
numeric_multimodal_cols = df[multimodal_cols].select_dtypes(include=np.number).columns.tolist()

print(f"Found {len(numeric_multimodal_cols)} numeric multimodal columns to process.")

# Create a new dataframe to hold the original data and the new binary features
df_binarized = df.copy()


# --- 3. PCA Visualization and Binarization Loop ---

# Loop through each numeric multimodal column
for col in numeric_multimodal_cols:
    print(f"Processing column: {col}...")

    # --- Binarization Step ---
    # Calculate the mean of the original column data (ignoring NaNs)
    mean_val = df[col].mean()
    # Create the new binary feature name
    binary_col_name = f'{col}_binary'
    # Add the binary column to our new dataframe
    df_binarized[binary_col_name] = (df[col] > mean_val).astype(int)


    # --- PCA and Visualization Step ---
    # Drop missing values for PCA and visualization for this specific column
    # We create a temporary dataframe for this to avoid altering the main data
    temp_df = df[[col]].dropna()

    if temp_df.empty:
        print(f"  -> Skipping column '{col}' because it contains no valid data.")
        continue

    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(temp_df[[col]])

    # Apply PCA. For a single column, this simplifies to centering the data.
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(scaled_data)

    # Create a dataframe for plotting
    plot_df = pd.DataFrame(data=principal_component, columns=['PC1'])
    # Add the binary feature to the plot dataframe (aligning by index)
    plot_df[binary_col_name] = df_binarized.loc[temp_df.index, binary_col_name]
    # We add a constant 'y' value to create a 1D scatter plot
    plot_df['y'] = 0

    # --- Plotting ---
    plt.figure(figsize=(12, 4))
    sns.scatterplot(
        x='PC1',
        y='y',
        hue=binary_col_name,
        data=plot_df,
        palette='viridis',
        s=100, # size of points
        alpha=0.7
    )

    plt.title(f'1D PCA of "{col}" (Colored by Binary Feature)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('')
    plt.yticks([]) # Hide the y-axis ticks
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.legend(title=f'Value > Mean({mean_val:.2f})')

    # Save the plot
    plot_filename = os.path.join(output_dir, f'pca_test_{col.replace("/", "_")}.png')
    plt.savefig(plot_filename)
    plt.close() # Close the plot to free up memory

print(f"\nAll PCA plots have been saved to the '{output_dir}' directory.")


# --- 4. Save the Final DataFrame ---

# Save the dataframe with all the new binary features to a new CSV file.
output_csv_path = 'MiNDAT_with_binary_test_features.csv'
df_binarized.to_csv(output_csv_path, index=False)

print(f"The final dataframe with binary features has been saved to '{output_csv_path}'")
print("\nFirst 5 rows of the new dataframe:")
print(df_binarized.head())



gg=pd.read_csv("../working/MiNDAT_with_binary_features.csv")
gf=pd.read_csv("../working/MiNDAT_with_binary_test_features.csv")


gg.isnull().sum()


gf.isnull().sum()


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "CORRUCYSTIC_DENSITY"
ID_COL = "LOCAL_IDENTIFIER"

def safe_binary_cols(df: pd.DataFrame):
    bad_substrings = {TARGET.lower(), ID_COL.lower(), "identifier", "id"}
    cols = []
    for c in df.columns:
        if c.endswith("_binary"):
            low = c.lower()
            if not any(b in low for b in bad_substrings):
                cols.append(c)
    return cols

UNIMODAL_COLS = [
    'vzo."','hp!','@wnsk>R','&%)LTaWRb','@V9','T\\!','.o<m','~7*',
    '9Z/5)2','%IiL7w','!;@Jw','fPqsI','i]7V',';<"<i(T',']xq',
    '^%a;',"0HU2N='U'","ZrK",".6AvGp","3I\\y",'ZVf','Jv[i'
]

class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.995):
        self.threshold = threshold
        self.keep_cols_ = None
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = pd.concat([X.reset_index(drop=True), pd.Series(y, name="__y__")], axis=1)
        corrs = df.corr(numeric_only=True)["__y__"].drop("__y__", errors="ignore").abs()
        suspicious = corrs[corrs > self.threshold]
        self.keep_cols_ = [c for c in X.columns if c not in suspicious.index]
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.keep_cols_]

class ColumnKeeper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: List[str]):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        return X[self.cols]

train = pd.read_csv("../working/MiNDAT_with_binary_features.csv")
test  = pd.read_csv("../working/MiNDAT_with_binary_test_features.csv")

drop_like_target = [c for c in train.columns if TARGET.lower() in c.lower() and c != TARGET]
train = train.drop(columns=drop_like_target, errors="ignore")
test  = test.drop(columns=drop_like_target, errors="ignore")

unimodal_present = [c for c in UNIMODAL_COLS if c in train.columns]
bin_cols = safe_binary_cols(train)
base_features = [c for c in (unimodal_present + bin_cols) if c in test.columns]
if len(base_features) == 0:
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    base_features = [c for c in num_cols if c not in [TARGET]]

X_full = train[base_features].copy()
y_full = train[TARGET].values
X_test_full = test[base_features].copy()

linear_pre = Pipeline([
    ("keep", ColumnKeeper(base_features)),
    ("impute", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(threshold=0.0)),
    ("corr", CorrelationDropper(threshold=0.995)),
    ("scale", StandardScaler(with_mean=True, with_std=True)),
    ("select", SelectKBest(score_func=f_regression, k="all"))
])

tree_pre = Pipeline([
    ("keep", ColumnKeeper(base_features)),
    ("impute", SimpleImputer(strategy="median")),
    ("var", VarianceThreshold(threshold=0.0)),
    ("corr", CorrelationDropper(threshold=0.995)),
    ("select", SelectKBest(score_func=f_regression, k="all"))
])

ridge = RidgeCV(alphas=np.logspace(-3, 3, 40), cv=5)
lasso = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=RANDOM_STATE, max_iter=20000)
enet  = ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,.99,1.0], alphas=np.logspace(-4, 1, 40),
                     cv=5, random_state=RANDOM_STATE, max_iter=20000)
hgb   = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE,
    learning_rate=0.03,
    max_depth=6,
    max_iter=1500,
    max_leaf_nodes=64,
    min_samples_leaf=20,
    l2_regularization=0.1
)

ridge_pipe = Pipeline([("pre", linear_pre), ("model", ridge)])
lasso_pipe = Pipeline([("pre", linear_pre), ("model", lasso)])
enet_pipe  = Pipeline([("pre", linear_pre), ("model", enet)])
hgb_pipe   = Pipeline([("pre", tree_pre),   ("model", hgb)])

def ttr(model, transformer):
    return TransformedTargetRegressor(regressor=model, transformer=transformer)

transformers = [
    QuantileTransformer(n_quantiles=min(1000, len(train)), output_distribution="normal", random_state=RANDOM_STATE),
    StandardScaler()
]

BASE_MODELS = []
for trf in transformers:
    BASE_MODELS.append(("ridge", ttr(ridge_pipe, trf)))
    BASE_MODELS.append(("lasso", ttr(lasso_pipe, trf)))
    BASE_MODELS.append(("enet",  ttr(enet_pipe,  trf)))
    BASE_MODELS.append(("hgb",   ttr(hgb_pipe,   trf)))

@dataclass
class OOFResult:
    oof: np.ndarray
    test_preds: np.ndarray
    models: List

def oof_predict(model, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame,
                n_splits: int = N_SPLITS, seed: int = RANDOM_STATE) -> OOFResult:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_fold_preds = []
    models = []
    for tr, va in kf.split(X, y):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]
        m = clone(model)
        m.fit(X_tr, y_tr)
        oof[va] = m.predict(X_va)
        test_fold_preds.append(m.predict(X_test))
        models.append(m)
    test_pred = np.mean(np.column_stack(test_fold_preds), axis=1)
    return OOFResult(oof=oof, test_preds=test_pred, models=models)

oof_dict = {}
test_pred_dict = {}
for name, mdl in BASE_MODELS:
    res = oof_predict(mdl, X_full, y_full, X_test_full, n_splits=N_SPLITS, seed=RANDOM_STATE)
    key = f"{name}_{mdl.transformer.__class__.__name__}"
    oof_dict[key] = res.oof
    test_pred_dict[key] = res.test_preds

for name in oof_dict:
    rmse = mean_squared_error(y_full, oof_dict[name], squared=False)
    print(f"[OOF] {name:>15s}: RMSE = {rmse:.6f}")

base_names = list(oof_dict.keys())
oof_stack_X = np.column_stack([oof_dict[n] for n in base_names])
test_stack_X = np.column_stack([test_pred_dict[n] for n in base_names])

meta = ElasticNetCV(l1_ratio=[.1,.5,.9], alphas=np.logspace(-5,2,50), cv=5, random_state=RANDOM_STATE, max_iter=20000)
meta.fit(oof_stack_X, y_full)
stack_preds = meta.predict(oof_stack_X)
stack_rmse = mean_squared_error(y_full, stack_preds, squared=False)
print(f"[Meta ElasticNet] OOF RMSE = {stack_rmse:.6f}")

stack_test_preds = meta.predict(test_stack_X)
blend_test_preds = np.median(test_stack_X, axis=1)

weights = np.linspace(0,1,11)
best_rmse = 1e9
best_w = 0.5
for w in weights:
    mix = w*stack_preds + (1-w)*np.median(oof_stack_X, axis=1)
    rmse = mean_squared_error(y_full, mix, squared=False)
    if rmse < best_rmse:
        best_rmse, best_w = rmse, w
print(f"Best blend weight = {best_w}, RMSE = {best_rmse:.6f}")

final_test_preds = best_w*stack_test_preds + (1-best_w)*blend_test_preds

submission = pd.read_csv("../input/recruitment-task-for-gdsc-ml/SPECIMEN.csv")
submission[TARGET] = final_test_preds
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


