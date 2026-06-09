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


import plotly.io as pio
pio.renderers.default = 'iframe'
pd.options.display.max_columns = None 


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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
print('num_vars: ',num_vars )

# Note that target is num_var


# Filling missing values for numerical columns with their median except target column
df[num_vars] = df[num_vars].fillna(df[num_vars].median())


# Filling missing values for categorical columns with their mode
df[cat_vars] = df[cat_vars].apply(
    lambda col: col.fillna(col.mode()[0] if not col.mode().empty else 'unknown'))


# Factorise Categorical vars - 
for i in df[cat_vars]:
    df[i], _ = pd.factorize(df[i])


pd.set_option('display.max_columns', None)

df.head()


# Get list of feature columns (excluding target)

X = [col for col in df.columns if col != 'target']

for feature in X:
    correlation = df[feature].corr(df["target"])
    print('Correlation for ', feature, ' = ',correlation)


df1=df0.copy()


df1['target']=df['target']


# Define features and target
X1 = df1.drop(columns=['efs', 'efs_time', 'ID', 'target', 'rituximab'], errors='ignore')
target = df1['target']


# Identify categorical and numerical columns
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


from sklearn.ensemble import GradientBoostingRegressor


# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingRegressor(random_state=42))
])


# Import necessary libraries for hyperparameter optimization
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from lifelines.utils import concordance_index

# Define parameter grid for hyperparameter optimization
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5, 7]
}

# Setup GridSearchCV with custom scoring function (concordance index)
grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5, 
    scoring=make_scorer(concordance_index), 
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X1, target)

# Retrieve best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Train the model on the full dataset using the best parameters
best_model.fit(X1, target)



# Evaluate the model on the full training data (using concordance index)
train_pred = best_model.predict(X1)
train_c_index = concordance_index(target, train_pred)
print(f"Concordance Index on Training Data: {train_c_index}")



# Load the test data
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Predict survival outcomes on the test data
prediction = best_model.predict(test_data.drop(columns=['ID'], errors='ignore'))


# Add predictions to the test dataset
test_data['prediction'] = prediction

# Save predictions to a new CSV file
output_file_path = 'submission.csv'
test_data[['ID', 'prediction']].to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")




