# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()



# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

prediction_interval_competition_ii_house_price_path = kagglehub.competition_download('prediction-interval-competition-ii-house-price')

print('Data source import complete.')



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


sample = pd.read_csv(f"{prediction_interval_competition_ii_house_price_path}/sample_submission.csv", sep=",")


sample.head(2)



sample.shape


train=pd.read_csv(f"{prediction_interval_competition_ii_house_price_path}/dataset.csv",sep=",")
train.head(2)


test=pd.read_csv(f"{prediction_interval_competition_ii_house_price_path}/test.csv",sep=",")


test.head(2)


test.shape,train.shape


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (adjust the path as needed)
# df = pd.read_csv('/kaggle/input/your-dataset.csv')

# Select numerical columns only
num_cols = train.select_dtypes(include='number').columns

# Define number of columns and rows for subplot grid
cols = 3
rows = (len(num_cols) + cols - 1) // cols  # Ceiling division

plt.figure(figsize=(5 * cols, 4 * rows))  # Adjust size to fit all subplots nicely

for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(train[col].dropna(), kde=True, color='royalblue')
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()






import numpy as np
import pandas as pd

train = pd.DataFrame(train)
test = pd.DataFrame(test)

# Select numerical features
num_cols = train.select_dtypes(include='number').columns
num_t=test.select_dtypes(include='number').columns
# Calculate skewness
skewness = train[num_cols].skew().sort_values(ascending=False)
st=test[num_t].skew().sort_values(ascending=False)

print("Skewness of features:\n", skewness)

# Threshold for high skewness (you can tune this)
skew_threshold = 1

# List of highly skewed features
high_skew = skewness[skewness > skew_threshold].index
highst=st[st > skew_threshold].index
print("\nHighly skewed features:", list(high_skew))

# Apply log1p transform to reduce skewness on these features
for col in high_skew:
    train[col] = np.log1p(train[col])
for col in highst:
    test[col] = np.log1p(test[col])

# For moderately skewed (0.5 < skew <= 1), apply sqrt transform (optional)
mod_skew = skewness[(skewness > 0.5) & (skewness <= skew_threshold)].index
modst = st[(st > 0.5) & (st<= skew_threshold)].index

for col in mod_skew:
    train[col] = np.sqrt(train[col])
for col in modst:
    test[col]=np.sqrt(test[col])

print("\nTransformation applied! Check distributions again if needed.")



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (adjust the path as needed)
# df = pd.read_csv('/kaggle/input/your-dataset.csv')

# Select numerical columns only
num_cols = train.select_dtypes(include='number').columns

# Define number of columns and rows for subplot grid
cols = 3
rows = (len(num_cols) + cols - 1) // cols  # Ceiling division

plt.figure(figsize=(5 * cols, 4 * rows))  # Adjust size to fit all subplots nicely

for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(train[col].dropna(), kde=True, color='red')
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()



train.describe()


train.columns


test.columns


y=train["sale_price"]


y.shape


train=train.drop(columns=["id","sale_price","longitude","bath_half"],axis=1)


test=test.drop(columns=["id","longitude","bath_half"],axis=1)


train.isna().sum()


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="most_frequent")



df=imputer.fit_transform(train)
tf=imputer.transform(test)


from sklearn.preprocessing import RobustScaler,OrdinalEncoder
scaler=RobustScaler()
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)



dff=encoder.fit_transform(df)
tff=encoder.transform(tf)


dff=scaler.fit_transform(dff)
tff=scaler.transform(tff)


dff.shape,tff.shape


from sklearn.preprocessing import PolynomialFeatures


poly=PolynomialFeatures(degree=1,interaction_only=True)


dff=poly.fit_transform(dff)
tff=poly.transform(tff)


dff.shape,tff.shape


dff.shape,tff.shape



# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1)
# dff = kpca.fit_transform(dff)
# tff=kpca.transform(tff)





dff.shape,tff.shape


# from sklearn.decomposition import PCA
# pca=PCA(n_components=0.98)
# dff=pca.fit_transform(dff)
# tff=pca.transform(tff)


dff.shape,tff.shape


from xgboost import XGBRegressor
model=XGBRegressor(random_state=25)


model.fit(dff,y)


model.score(dff,y)


from xgboost import XGBRegressor
import numpy as np

# Winkler Score function
def winkler_score(y_true, y_lower, y_upper, alpha):
    y_true = np.array(y_true)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)

    score = np.where(
        (y_true >= y_lower) & (y_true <= y_upper),
        y_upper - y_lower,
        (y_upper - y_lower) + (2 / alpha) * np.where(
            y_true < y_lower, y_lower - y_true, y_true - y_upper
        )
    )

    return np.mean(score)

# Train Quantile Regressor
def train_quantile_model(X_train, y_train, quantile):
    model = XGBRegressor(random_state=25

    )
    model.set_params(objective='reg:quantileerror', quantile_alpha=quantile)
    model.fit(X_train, y_train)
    return model

# Predict intervals and calculate Winkler score (no CV)
def simple_predict_interval(dff, y, tff, alpha=0.3):
    # Train models for lower and upper quantiles
    lower_model = train_quantile_model(dff, y, quantile=alpha / 2)
    upper_model = train_quantile_model(dff, y, quantile=1 - alpha / 2)

    # Predict on training set to evaluate Winkler score
    y_lower = lower_model.predict(dff)
    y_upper = upper_model.predict(dff)
    winkler = winkler_score(y, y_lower, y_upper, alpha)

    # Predict on test set
    pi_lower = lower_model.predict(tff)
    pi_upper = upper_model.predict(tff)

    print(f"Winkler Score on full training set: {winkler:.4f}")
    return pi_lower, pi_upper



# dff: training features
# y: training target
# tff: test features (to predict PI on)

pi_lower, pi_upper = simple_predict_interval(dff, y, tff,0.5)



data = {
    "id": [i for i in range(200000,len(test)+200000)],  # Replace with actual IDs
    "pi_lower":pi_lower,
    "pi_upper":pi_upper,
}

# Create a DataFrame
tf = pd.DataFrame(data)

# Save as CSV
tf.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' created successfully!")


tf




