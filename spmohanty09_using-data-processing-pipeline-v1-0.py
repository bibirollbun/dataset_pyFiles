import pandas as pd
import numpy as np

df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df.head(5)


df.info()


df.describe()


# First, let's see what columns we actually have
print("Available columns:", df.columns.tolist())


# dropping the columns with numerical values
df_categorical = df.drop(['id', 'Weight Capacity (kg)', 'Price'], axis=1)

# Function to print value counts for each categorical column
def print_categorical_counts(df_categorical):
    print("Categorical Value Counts:\n")
    for column in df_categorical.columns:
        print(df_categorical[column].value_counts())
        print("=" * 30)

print_categorical_counts(df_categorical)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

# Creating subplots for each categorical column
n_cols = len(df_categorical.columns)
fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 4 * ((n_cols + 1) // 2)))
axes = axes.flatten()  # Flatten axes array for easier indexing

# Plot each categorical column
for idx, column in enumerate(df_categorical.columns):
    # Create value counts
    value_counts = df_categorical[column].value_counts()
    
    # Create bar plot
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[idx], palette='deep')
    
    # Customize the plot
    axes[idx].set_title(f'Distribution of {column}', pad=20)
    axes[idx].set_xlabel(column)
    axes[idx].set_ylabel('Count')
    
    # Rotate x-labels if they're too long
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Add value labels on top of each bar
    for i, v in enumerate(value_counts.values):
        axes[idx].text(i, v, str(v), ha='center', va='bottom')

# Remove empty subplots if any
for idx in range(len(df_categorical.columns), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


train_data=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_data.head(5)


train_data_features = train_data.drop(columns=['Price'])
train_data_target= train_data['Price']


from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class MyOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self

    def transform(self, X):
        print(f"Input shape: {X.shape}")  # Debugging statement
        transformed = self.encoder.transform(X)
        print(f"Output shape: {transformed.shape}")  # Debugging statement
        return transformed


# First, let's see what columns we actually have
print("Available columns:", train_data.columns.tolist())


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column, fill_value):
        self.column = column
        self.fill_value = fill_value
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X[self.column] = X[self.column].fillna(self.fill_value)
        return X


num_attribs = ['Weight Capacity (kg)']
cat_attribs = ['Brand', 
               'Material', 
               'Size', 
               'Compartments', 
               'Laptop Compartment', 
               'Waterproof', 
               'Style', 
               'Color']
# Numeric pipeline
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Categorical pipeline for other categorical attributes
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(sparse_output=False)),  # Use OneHotEncoder instead of LabelBinarizer
])

# Combine all pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_data_features, train_data_target, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)


X_train_prepared


import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

best_params = {
    'objective': 'mae',  
    'learning_rate': 0.05820157052133642,
    'n_estimators': 97,
    'max_depth': 10,
    'num_leaves': 22,
    'reg_alpha': 0.025041488939909966,  
    'reg_lambda': 0.04398938070361757,   
    'colsample_bytree': 0.513282987217108,  
    'verbose': -1,
    'n_jobs': -1,
    'device': 'gpu'
}

reg_model = lgb.LGBMRegressor(**best_params)
# reg_model = RandomForestRegressor()

reg_model.fit(X_train_prepared, y_train)


# Make predictions on the test set
y_pred = reg_model.predict(X_test_prepared)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE on the test set: {rmse:.5f}")


test_data=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


test_data.head(5)


test_data.info()


prepared_test_data=full_pipeline.fit_transform(test_data)
prepared_test_data


predictions = reg_model.predict(prepared_test_data)

# Creating a submission DataFrame
submission = pd.DataFrame({
    'id': test_data['id'],
    'Price': predictions
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

print(submission.head())




