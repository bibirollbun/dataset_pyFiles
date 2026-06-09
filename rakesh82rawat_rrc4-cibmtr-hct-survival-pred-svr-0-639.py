 # Installations-1
!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter


# Import dataset (df0)
df0=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')


df = df0.drop('ID', axis=1)


# Combining 2 target variables (efs and efs_time) into One Target- 

def kaplan(data=df, time_col = 'efs_time', event_col='efs'):
    
    kmf = KaplanMeierFitter()
    kmf.fit(data[time_col], event_observed=data[event_col])
    return kmf.survival_function_at_times(df[time_col]).values.flatten()


df['target'] = kaplan(data=df)


df.shape


df = df.drop(columns=['efs', 'efs_time'], errors='ignore')


df.duplicated()[df.duplicated()==True]
    #> No duplicates found


for col in df.select_dtypes(include='object').columns:
       df[col] = df[col].str.strip().str.lower().replace(
        {'n/a': None, 'na': None, 'nan': None, '-': None})


# cat_vars
cat_vars = [var for var in df.columns if df[var].dtype == "O"]

# num_vars
num_vars= [var for var in df.columns if df[var].dtype != "O" and var != 'target']


print('cat_vars: ',cat_vars,'\n')
print('num_vars: ',num_vars )  #num_vars is without target^

# Note that target is num_var


# Filling missing values for numerical columns with their median except target column
df[num_vars] = df[num_vars].fillna(df[num_vars].median())


# Filling missing values for categorical columns with their mode
df[cat_vars] = df[cat_vars].apply(
    lambda col: col.fillna(col.mode()[0] if not col.mode().empty else 'unknown'))


# Factorise Categorical vars (give unique number to unique cat_values) - 
for i in df[cat_vars]:
    df[i], _ = pd.factorize(df[i])


pd.set_option('display.max_columns', None)

df.head(2)


import matplotlib.pyplot as plt

# Get list of feature columns (excluding target)
X = [col for col in df.columns if col != 'target']

# Create a list to store feature names and their correlations
correlations = []

# Calculate correlation for each feature
for feature in X:
    correlation = df[feature].corr(df["target"])
    correlations.append((feature, correlation))

# Sort correlations in decreasing order of absolute value
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

# Print sorted correlations
for feature, correlation in correlations:
    print('Correlation for ', feature, ' = ', correlation)

# Extract features and correlation values for plotting
features = [item[0] for item in correlations]
correlation_values = [item[1] for item in correlations]

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(features, correlation_values, color='skyblue')
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Feature Correlations with Target')
plt.tight_layout()
plt.show()



df1=df0.copy()


df1['target']=df['target']


# Define features and target
X1 = df1.drop(columns=['efs', 'efs_time', 'ID', 'target', 'rituximab','hepatic_mild','renal_issue','hla_match_dqb1_low','tce_match'], errors='ignore')
target = df1['target']


# Re-Identify categorical and numerical columns
cat_vars1 = X1.select_dtypes(include=['object']).columns
num_vars1 = X1.select_dtypes(include=['int64', 'float64']).columns


# Preprocessing for numerical data: Imputation and Scaling
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


from sklearn.preprocessing import OneHotEncoder


# Preprocessing for categorical data: Imputation and One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


from sklearn.compose import ColumnTransformer


# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_vars1),
        ('cat', categorical_transformer, cat_vars1)
    ]
)


from sklearn.svm import SVR

# Support Vector Regressor setup
model6_SVR = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1
    ))
])


# Fit the model on the training data
model6_SVR.fit(X1, target)


train_pred6 = model6_SVR.predict(X1)
train_c_index6 = concordance_index(target, train_pred6)
print(f"Concordance Index on Training Data: {train_c_index6}")


# Make Predictions on TEST data

# Load the test data
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Predict survival outcomes on the test data 
prediction = model6_SVR.predict(test_data.drop(columns=['ID'], errors='ignore'))


# Add predictions to the test dataset
test_data['prediction'] = prediction

# Save predictions to a new CSV file
output_file_path = 'submission.csv'
test_data[['ID', 'prediction']].to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")

