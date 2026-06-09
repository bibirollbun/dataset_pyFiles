
# Cell 1 - Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

print("Libraries imported. Python version:", pd.__version__)



# Cell 2 - File path (you said the file is here)
csv_path = '/kaggle/input/boston-house-prices/housing.csv'
print("Looking for:", csv_path)
print("Files in /kaggle/input:")
print(os.listdir('/kaggle/input'))



# Cell 3 - Load data (robust: tries comma, then whitespace)
def load_boston_csv(path):
    # Try normal CSV read first
    try:
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            print("Read with pd.read_csv() : shape", df.shape)
            return df
    except Exception as e:
        print("pd.read_csv failed:", e)
    # Try delim_whitespace (some versions have space-separated values)
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        if df.shape[1] > 1:
            print("Read with delim_whitespace and header=None : shape", df.shape)
            return df
    except Exception as e:
        print("delim_whitespace read failed:", e)
    # Try engine='python' auto-detect
    try:
        df = pd.read_csv(path, sep=None, engine='python')
        print("Read with sep=None engine=python : shape", df.shape)
        return df
    except Exception as e:
        print("All read attempts failed:", e)
        raise FileNotFoundError(f"Couldn't read dataset at {path}. Make sure file exists and is CSV or whitespace-separated.")

df = load_boston_csv(csv_path)
print("Initial df shape:", df.shape)
display(df.head())
print("\nColumns in the dataset:")
print(df.columns)



# Cell 4 - If dataframe has headerless columns, fix for classic Boston dataset
boston_colnames = [
    "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX",
    "PTRATIO","B","LSTAT","MEDV"
]

# If df has one column but that column contains whitespace-separated data, try splitting:
if df.shape[1] == 1:
    # split strings into many columns
    df = df[ df.columns[0] ].str.split(expand=True)
    print("Split the single column into", df.shape[1], "columns.")

# If number of columns matches Boston features, set names:
if df.shape[1] == len(boston_colnames):
    df.columns = boston_colnames
    print("Assigned standard Boston column names.")
else:
    print("Number of columns:", df.shape[1])
    print("Column names (first 10):", list(df.columns[:10]))

# Convert possible object columns to numeric where possible
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nAfter conversion, dtypes:")
print(df.dtypes)

# Select target column safely:
if 'MEDV' in df.columns:
    target_col = 'MEDV'
else:
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found. Your CSV may not be parsed correctly. Check the file format.")
    target_col = numeric_cols[-1]   # last numeric column as fallback

print("Using target column:", target_col)
display(df.head())



# Cell 5 - Train a simple model
# Drop rows with missing values in target or features (simple handling)
df_clean = df.dropna(axis=0, subset=[target_col])
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

# If X has non-numeric or object columns, keep numeric only for now
X = X.select_dtypes(include=['int64','float64'])

print("Final training shapes: X:", X.shape, "y:", y.shape)
if X.shape[0] == 0 or X.shape[1] == 0:
    raise ValueError("No training data after cleaning. Check dataset content.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model trained. MSE: {mse:.4f}, R2: {r2:.4f}")



# Cell 6 - Save predictions file to notebook working directory
out = X_test.copy()
out['actual'] = y_test.values
out['predicted'] = y_pred
out.to_csv('predictions.csv', index=False)
print("Saved predictions.csv to notebook working directory.")


