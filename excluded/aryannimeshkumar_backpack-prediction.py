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


import pandas as pd

test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv') 
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train.head()


train.info()


import pandas as pd

# Assuming train and train_extra are already loaded DataFrames
combined_df = pd.concat([train, train_extra], axis=0, ignore_index=True)

# Display the shape of the combined DataFrame
print(f"Combined dataset shape: {combined_df.shape}")



for col in combined_df.select_dtypes(include=['object']).columns:
    unique_values = combined_df[col].unique()
    print(f"Unique values in '{col}':\n", unique_values, "\n")


for col in combined_df.columns:
    nan_count = combined_df[col].isna().sum()
    print(f"Missing values in '{col}': {nan_count}")


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Function to check normality of the features
def check_normality(df, features):
    for feature in features:
        print(f"Checking normality for {feature}...")
        
        # Drop missing values
        data = df[feature].dropna()

        # 1. Visual Inspection
        # Histogram
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(data, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Histogram of {feature}')
        
        # QQ Plot
        plt.subplot(1, 2, 2)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'QQ Plot of {feature}')
        plt.show()

        # 2. Statistical Test: Shapiro-Wilk test
        stat, p_value = stats.shapiro(data)
        print(f"Shapiro-Wilk test for {feature}: Statistic={stat:.4f}, p-value={p_value:.4f}")
        if p_value > 0.05:
            print(f"The feature {feature} is likely normally distributed (Fail to Reject Null Hypothesis).")
        else:
            print(f"The feature {feature} is likely not normally distributed (Reject Null Hypothesis).")
        
        print("="*50)

# List of features to check for normality
features = ['Compartments', 'Weight Capacity (kg)']

# Check normality for the specified features in the combined_df DataFrame
check_normality(combined_df, features)



combined_df.fillna({
    'Brand': 'Unknown',
    'Material': 'Unknown',
    'Size': 'Unknown',
    'Laptop Compartment': 'Unknown',
    'Waterproof': 'Unknown',
    'Style': 'Unknown',
    'Color': 'Unknown'
}, inplace=True)

combined_df['Weight Capacity (kg)'] = combined_df['Weight Capacity (kg)'].fillna(combined_df['Weight Capacity (kg)'].mean())


import pandas as pd

# Sample data based on your unique values
data = {
    'Brand': ['Jansport', 'Under Armour', 'Nike', 'Adidas', 'Puma', 'Unknown'],
    'Material': ['Leather', 'Canvas', 'Nylon', 'Unknown', 'Polyester', 'Leather'],
    'Laptop Compartment': ['Yes', 'No', 'Unknown','Yes','No','Yes'],
    'Waterproof': ['No',  'Yes', 'Unknown','Yes','No','Yes'],
    'Style': ['Tote', 'Messenger', 'Unknown', 'Backpack', 'Tote', 'Messenger'],
    'Color': ['Black', 'Green', 'Red', 'Blue', 'Gray', 'Pink']
}

df = pd.DataFrame(data)

# List of categorical columns
categorical_columns = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Apply One-Hot Encoding using pandas get_dummies
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

# Display the encoded DataFrame
print(df_encoded)



combined_df.columns


from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

# Sample data
data = {
    'Size': ['Medium', 'Small', 'Large', 'Unknown', 'Medium', 'Small']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Define the order for the categories in 'Size'
size_order = ['Small', 'Medium', 'Large', 'Unknown']

# Instantiate the OrdinalEncoder with the specified order
encoder = OrdinalEncoder(categories=[size_order])

# Apply ordinal encoding to the 'Size' column
df['Size_Encoded'] = encoder.fit_transform(df[['Size']])

# Display the DataFrame with the encoded column
print(df)



nominal_features = ['Brand','Material','Laptop Compartment','Waterproof','Style','Color']

ordinal_features = ['Size']

numeric_features = ['Compartments','Weight Capacity (kg)']

all_features = nominal_features + ordinal_features + numeric_features


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# Define the transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
                               ])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[['Small', 'Medium', 'Large', 'Unknown']]))])

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric_transformer', numeric_transformer, numeric_features),
        ('nominal_transformer', nominal_transformer, nominal_features),
        ('ordinal_transformer', ordinal_transformer, ordinal_features)])


from sklearn.model_selection import train_test_split

X = combined_df[all_features]

y = combined_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"""size of training set: {len(X_train)}
size of testing set: {len(X_test)}""")


from xgboost import XGBRegressor
# Define the model pipeline with XGBoost
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(random_state=42))
])


# Fit the model
model_pipeline.fit(X_train, y_train)

# Predict on the test set
model_pipeline


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Fitting the model
model_pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred = model_pipeline.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")



y_test_pred = model_pipeline.predict(test)

# turn your predictions into a list
test_predictions = y_test_pred.tolist()

submission_df = pd.DataFrame({'id': test['id'], 'Price': test_predictions})
submission_df.to_csv('submission.csv', index=False)


submission_df.head()

