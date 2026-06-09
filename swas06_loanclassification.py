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


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC



df_train = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
df_train.shape,df_test.shape


df_train.head(3)


df_train.info()


df_train.isnull().sum()


numeric_features = ['id','person_age','person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income', 'cb_person_cred_hist_length']


categorical_features = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']


X = df_train.drop('loan_status',axis=1)
y=df_train['loan_status']


from sklearn.utils.class_weight import compute_class_weight
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessors
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
])
# Assuming all features are numeric in this case
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,numeric_features ),
    ('cat', categorical_transformer,categorical_features )
])

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced'),
    'XGBoost': XGBClassifier(class_weight='balanced')
}

plt.figure(figsize=(8, 6))

# Iterate over classifiers
for name, classifier in classifiers.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Customize plot
plt.title('ROC Curve for Different Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()



model =  XGBClassifier(class_weight='balanced')
pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

pipeline.fit(X_train, y_train)
# Test on the test set
best_preds = pipeline.predict(X_test)
final_accuracy = accuracy_score(y_test, best_preds)

print("Final Test Accuracy:", final_accuracy)


y_test_pred = pipeline.predict(df_test)


submission_df = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')

prediction_results = pd.DataFrame({
    'id':  submission_df.id ,
    'loan_paid_back': y_test_pred
})

submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())

