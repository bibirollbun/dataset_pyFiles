import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')

# Feature Engineering
def create_dataframe(df):
    df['Pass Rate'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['Pass Rate'] /= (df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)'] + 1e-5)  # Avoid division by zero
    
    df['Avg Grade per Unit'] = df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']
    df['Avg Grade per Unit'] /= (df['Curricular units 1st sem (evaluations)'] + df['Curricular units 2nd sem (evaluations)'] + 1e-5)
    return df

train = create_dataframe(train)
test = create_dataframe(test)

# Define features and target
X = train.drop(columns=['id', 'Target'])
y = train['Target']
X_test = test.drop(columns=['id'])

# Identify feature types
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Preprocessing pipeline
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)])

# Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X, y)

# Make predictions
y_pred = pipeline.predict(X_test)

# Prepare submission
submission = pd.DataFrame({'id': test['id'], 'Target': y_pred})
submission.to_csv('submission.csv', index=False)

print('cvs created')


