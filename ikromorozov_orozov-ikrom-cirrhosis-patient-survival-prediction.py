!pip install catboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df= pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
df


df.info()


df.isnull().sum()


df.drop(['id'], axis=1, inplace=True)


df['Status'].value_counts()


# Keep rows where 'Status' is not equal to 'Y'
df = df[df.Status != 'Y']


status_counts = df['Status'].value_counts()
print("Count of each Status:")
print(status_counts)
print()

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.pie(
    df['Status'].value_counts(),
    shadow=True,
    explode=[.1 for i in range(df['Status'].nunique())],  # Adjust the explode based on the number of unique values
    autopct='%1.f%%',
    textprops={'size': 14, 'color': 'white'}
)

ax.set_title('Status in Train Dataset', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.show()


selected_features = [

    'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
    'Tryglicerides', 'Platelets', 'Prothrombin'
]

# Create a subset correlation matrix for the selected features
subset_correlation_matrix = df[selected_features].corr()

# Plot the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    subset_correlation_matrix,
    annot=True,
    fmt=".2f",
    linewidths=.5
)

plt.show()


x = df.drop(['Status'], axis=1)
y = df['Status']


encoder = LabelEncoder()
y_train = encoder.fit_transform(y)


y_train_series = pd.Series(y_train, index=df.index)
df[selected_features].corrwith(y_train_series).sort_values(ascending=False)


# cat_attributes = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Drug']
cat_attributes = x.select_dtypes(include=['object']).columns.to_list()
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore')),
])

# numeric columns
num_cols = x.select_dtypes(include=['float64']).columns.to_list()
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# combine
preprocessor = ColumnTransformer([
    ('categorical', cat_pipeline, cat_attributes),
    ('numerical', num_pipeline, num_cols)
])


x_train = preprocessor.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


def estimate_model(y_test, y_pred, y_proba, model_name):
    print(f"Model: {model_name}")
    print(f"Accuracy Score: {metrics.accuracy_score(y_test, y_pred):.4f}")
    print("Log Loss:", metrics.log_loss(y_test, y_proba, labels=[0, 1, 2]))
    print(f"Classification Report:\n{metrics.classification_report(y_test, y_pred, zero_division=0)}")
    print('='*50)


models = {
    "SVM": SVC(kernel='linear', probability=True, decision_function_shape='ovo'),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(objective='multi:softmax', num_class=3),
    "LogisticRegression": LogisticRegression(multi_class='ovr', max_iter=500),
    "DecisionTree": DecisionTreeClassifier(),
}


for name, model in models.items():
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    estimate_model(y_test, y_pred, y_proba, name)


test_df = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
test_df


x_sub = preprocessor.transform(test_df)


model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y))
)
model.fit(x_train, y_train)
y_proba = model.predict_proba(x_sub)


submission = pd.DataFrame(y_proba, columns=['Status_C', 'Status_CL', 'Status_D'])
submission['id'] = test_df['id']
submission = submission[['id', 'Status_C', 'Status_CL', 'Status_D']]
submission.to_csv('submission.csv', index=False)

