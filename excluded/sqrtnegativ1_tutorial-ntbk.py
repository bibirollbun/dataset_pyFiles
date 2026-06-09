# There are plenty of better and faster libraries in the year 2086, but we're working in the past and have to deal with it.
import pandas as pd
import numpy as np

# The 'r' stands for "raw".
TRAIN_PATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"

# Read from CSV
df = pd.read_csv(TRAIN_PATH)


# Upgrade pip version
!pip install --upgrade scikit-learn


# Check if it worked.
# Default version of sklearn on Kaggle is 1.2.1.
# So we should see a version higher than that (around 1.7.1).
import sklearn
print(sklearn.__version__)


# Deal with NaNs as many models cannot handle them directly.
df.drop( # Drop all rows where...
    df[df['CORRUCYSTIC_DENSITY'].isna()].index, # the target ('CORRUCYSTIC_DENSITY') is NaN
    inplace=True # Do it "in-place".
)


cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist() # Columns with categorical data
num_cols = df.select_dtypes(include=[np.number]).columns.drop('CORRUCYSTIC_DENSITY').tolist() # Columns with numerical data, excluding the target


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = df[num_cols] # ignoring categorical columns in this ntbk because I am not paid enough to care.
y = df['CORRUCYSTIC_DENSITY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit the model on the training data
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# Test the model on the test data
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
y_pred = model.predict(X_test)
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")


# Train the model on the entire training data
model.fit(X, y)

# Make predictions on the UNK data.
df_unk = pd.read_csv(TEST_PATH)
X_unk = df_unk[num_cols]
y_unk_pred = model.predict(X_unk)

# Actually creating the submission CSV file
submission = pd.DataFrame({
    'LOCAL_IDENTIFIER': df_unk['LOCAL_IDENTIFIER'],
    'CORRUCYSTIC_DENSITY': y_unk_pred
})

submission['LOCAL_IDENTIFIER'] = submission['LOCAL_IDENTIFIER'].astype(int)
submission['CORRUCYSTIC_DENSITY'] = submission['CORRUCYSTIC_DENSITY'].astype(float)

submission.to_csv('submission.csv', index=False)
print(submission.head())

