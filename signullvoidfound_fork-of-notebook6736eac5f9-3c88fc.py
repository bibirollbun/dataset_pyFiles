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



cat_fill_values = {}

for j in df_train.select_dtypes(include=["object", "category"]).columns:
    mode_val = df_train[j].mode()[0]    
    cat_fill_values[j] = mode_val
    df_train[j] = df_train[j].fillna(mode_val)
    
for k, value in cat_fill_values.items():
    if k in df_test.columns:  
        df_test[k] = df_test[k].fillna(value)


from sklearn.preprocessing import OrdinalEncoder

# Step 1: Impute categorical columns with mode (you already did this)
cat_fill_values = {}

for j in df_train.select_dtypes(include=["object", "category"]).columns:
    mode_val = df_train[j].mode()[0]    
    cat_fill_values[j] = mode_val
    df_train[j] = df_train[j].fillna(mode_val)

for k, value in cat_fill_values.items():
    if k in df_test.columns:  
        df_test[k] = df_test[k].fillna(value)

# Step 2: Ordinal Encoding
cat_cols = df_train.select_dtypes(include=["object", "category"]).columns

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# Fit on train and transform both
df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols])
df_test[cat_cols]  = encoder.transform(df_test[cat_cols])



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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


# ----------------------
# 1. Load train data
# ----------------------
X = df_train.drop(columns=["CORRUCYSTIC_DENSITY"])
y = df_train["CORRUCYSTIC_DENSITY"]
feature_cols = X.columns.tolist()

# ----------------------
# 2. Split into train/validation (to calculate RMSE)
# ----------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------
#3. Build polynomial + lasso pipeline
# ----------------------
degree = 2
alpha = 0.001   # regularization strength, can be tuned

model = Pipeline([
    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=alpha, max_iter=10000, random_state=42))
])

model.fit(X_train, y_train)


# ----------------------
# 4. Validation RMSE
# ----------------------
y_val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print("Validation RMSE:", rmse)

# ----------------------
# 5. Predict on real test.csv
# ----------------------
X_real_test = df_test[feature_cols]   # use same features as training
y_pred = model.predict(X_real_test)

# ----------------------
# 6. Build submission
# ----------------------
submission = pd.read_csv("../input/recruitment-task-for-gdsc-ml/SPECIMEN.csv")
submission["CORRUCYSTIC_DENSITY"] = y_pred  # overwrite predictions

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created with same format as specimen.csv")

