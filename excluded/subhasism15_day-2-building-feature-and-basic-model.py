import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Though we are using seaborn we still require it since plt ensure proper rendering for sns plots. Additionally it easier to build customizable subplots using matplotlib.
import seaborn as sns           # For its aesthetics and ease of creating statisctical plots especially with regards to grouping and aggregation.

sns.set(style='whitegrid')      # Horizontal gridlines on a white background

# Loading data from Kaggle input folders
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Data dimensions - For both Train and Test - Test does not have the target variable - Calories
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

print(f"Features in the Training data alongwith the Target variable - Calories: \n {train.columns}")


train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
test['BMI'] = test['Weight'] / ((test['Height'] / 100) ** 2)


plt.figure(figsize = (8,5))
sns.scatterplot(x = 'BMI', y = 'Calories', data = train, alpha = 0.3)
plt.title('Calories burned with BMI')
plt.show()


sns.lmplot(x = 'BMI', y = 'Calories', data = train, height = 5, aspect = 1.5)


sns.regplot(x = 'BMI', y = 'Calories', data = train, lowess = True, scatter_kws ={'alpha':0.3})


train['BMI_Bin'] = pd.cut(train['BMI'], bins = 50)

bin_means = train.groupby('BMI_Bin')['Calories'].mean().reset_index()

plt.figure(figsize=(10,5))
sns.barplot(x = 'BMI_Bin', y = 'Calories', data = bin_means)
plt.title("Average calories burnt by BMI group")
plt.xticks(rotation = 45)
plt.show()


corr = train['BMI'].corr(train['Calories'])
print(f"Correlation between BMI and Calories burnt: {corr:.4f}")


import statsmodels.api as sm
X = train['BMI']
X = sm.add_constant(X)
Y = train['Calories']
model = sm.OLS(Y,X).fit()
print(model.summary())


train['BMI2'] = train['BMI']**2
X_Poly = train[['BMI', 'BMI2']]
X_Poly=sm.add_constant(X_Poly)
Y=train['Calories']
poly_model = sm.OLS(Y,X_Poly).fit()
print(poly_model.summary())


#OLS Linear Regression Model

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

#Dummy Variables for Sex
train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)

features = ['Sex_male', 'Duration', 'Heart_Rate', 'Body_Temp']

# Ensure all features are float type
X = train[features].astype(float)
y = train['Calories'].astype(float)
X_test = test[features].astype(float)

# Add constant (intercept term)
X = sm.add_constant(X)
X_test = sm.add_constant(X_test)

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

train_preds = ols_model.predict(X)

#Owing to one of the variable - Body_Temp. Got some negative values hence making it 0. Need to figure out if it indeed is negative because in Day 1 notebook we could see a positive correlation. However, this could be because of the Sex too. Nevertheless need to look into it
train_preds = np.clip(train_preds, 0, None)
train_rmsle = np.sqrt(mean_squared_log_error(y, train_preds))
print(f"Training RMSLE (OLS, full data): {train_rmsle:.5f}")

test_preds = ols_model.predict(X_test)
test_preds = np.clip(test_preds, 0, None)

submission['Calories'] = test_preds
submission.to_csv('submission_ols_fulltrain.csv', index=False)
submission.head()


# Day 2: OLS Linear Regression Model (Full Train Data, No log1p)

# ============================
# STEP 1 â€“ Import Libraries
# ============================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train['Body_Temp2'] = train['Body_Temp']**2
test['Body_Temp2'] = test['Body_Temp']**2

train['Heart_Rate2'] = train['Heart_Rate']**2
test['Heart_Rate2'] = test['Heart_Rate']**2

train['Duration2'] = train['Duration']**2
test['Duration2'] = test['Duration']**2

train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)

features = ['Sex_male', 'Duration', 'Heart_Rate', 'Body_Temp', 'Body_Temp2', 'Heart_Rate2', 'Duration2']

# Ensure all features are float type
X = train[features].astype(float)
y = train['Calories'].astype(float)
X_test = test[features].astype(float)

# Add constant (intercept term)
X = sm.add_constant(X)
X_test = sm.add_constant(X_test)

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())


train_preds = ols_model.predict(X)
train_preds = np.clip(train_preds, 0, None)
train_rmsle = np.sqrt(mean_squared_log_error(y, train_preds))
print(f"Training RMSLE (OLS, full data): {train_rmsle:.5f}")


test_preds = ols_model.predict(X_test)
test_preds = np.clip(test_preds, 0, None)

submission['Calories'] = test_preds
submission.to_csv('submission_ols_fulltrain.csv', index=False)
submission.head()


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_log_error

# ============================
# STEP 2 â€“ Load Data
# ============================

train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train['Body_Temp2'] = train['Body_Temp'] ** 2
test['Body_Temp2'] = test['Body_Temp'] ** 2

train['Heart_Rate2'] = train['Heart_Rate'] ** 2
test['Heart_Rate2'] = test['Heart_Rate'] ** 2

train['Duration2'] = train['Duration'] ** 2
test['Duration2'] = test['Duration'] ** 2

train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)

features = ['Sex_male', 'Duration', 'Heart_Rate', 'Body_Temp', 'Body_Temp2', 'Heart_Rate2', 'Duration2']

# Ensure all features are float type
X = train[features].astype(float)
y = np.log1p(train['Calories']).astype(float)  # Apply log1p transform
X_test = test[features].astype(float)

# Add constant (intercept term)
X = sm.add_constant(X)
X_test = sm.add_constant(X_test)

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

train_preds_log = ols_model.predict(X)
train_preds = np.expm1(train_preds_log)  # Inverse transform
train_true = np.expm1(y)
train_preds = np.clip(train_preds, 0, None)

train_rmsle = np.sqrt(mean_squared_log_error(train_true, train_preds))
print(f"Training RMSLE (OLS with log1p): {train_rmsle:.5f}")

test_preds_log = ols_model.predict(X_test)
submission['Calories_log1p'] = test_preds_log  # Log-space prediction
submission['Calories'] = np.clip(np.expm1(test_preds_log), 0, None)  # Actual prediction
submission.drop('Calories_log1p', axis=1, inplace=True)
submission.to_csv('submission_ols_fulltrain_log1p.csv', index=False)
print(submission[['Calories']].head())


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_log_error

train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train['Body_Temp2'] = train['Body_Temp'] ** 2
test['Body_Temp2'] = test['Body_Temp'] ** 2

train['Heart_Rate2'] = train['Heart_Rate'] ** 2
test['Heart_Rate2'] = test['Heart_Rate'] ** 2

train['Duration2'] = train['Duration'] ** 2
test['Duration2'] = test['Duration'] ** 2

# One-hot encode 'Sex'
train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)

# Automatically bin 'Age' into 30 equal-width intervals using train bin edges
age_bins = pd.cut(train['Age'], bins=50).unique().categories
train['Age_Bin'] = pd.cut(train['Age'], bins=age_bins)
test['Age_Bin'] = pd.cut(test['Age'], bins=age_bins)

# One-hot encode Age bins
train = pd.get_dummies(train, columns=['Age_Bin'], drop_first=True)
test = pd.get_dummies(test, columns=['Age_Bin'], drop_first=True)

# Ensure test set has same dummy columns as train
missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    if 'Age_Bin_' in col:
        test[col] = 0

# Align test columns with train
test = test.reindex(columns=train.columns, fill_value=0)

# Dynamically extract age bin features
age_bin_features = [col for col in train.columns if col.startswith('Age_Bin_')]

features = ['Sex_male', 'Duration', 'Heart_Rate', 'Body_Temp',
            'Body_Temp2', 'Heart_Rate2', 'Duration2'] + age_bin_features

# Ensure all features are float type
X = train[features].astype(float)
y = np.log1p(train['Calories']).astype(float)
X_test = test[features].astype(float)

# Add constant (intercept term)
X = sm.add_constant(X)
X_test = sm.add_constant(X_test)

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

train_preds_log = ols_model.predict(X)
train_preds = np.expm1(train_preds_log)  # Inverse transform
train_true = np.expm1(y)
train_preds = np.clip(train_preds, 0, None)

train_rmsle = np.sqrt(mean_squared_log_error(train_true, train_preds))
print(f"Training RMSLE (OLS with log1p and 30 Equal-Width Age Bins): {train_rmsle:.5f}")

test_preds_log = ols_model.predict(X_test)
submission['Calories_log1p'] = test_preds_log  # Log-space prediction
submission['Calories'] = np.clip(np.expm1(test_preds_log), 0, None)  # Actual prediction
submission.drop('Calories_log1p', axis=1, inplace=True)
submission.to_csv('submission_ols_fulltrain_log1p_agebin.csv', index=False)
print(submission[['Calories']].head())


from IPython.display import HTML
import base64

def create_download_link(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return HTML(f'<a download="{filename}" href="data:text/csv;base64,{b64}" target="_blank">ðŸ“¥ Click here to download {filename}</a>')

# Show download link
create_download_link("submission_ols_fulltrain_log1p_agebin.csv")




