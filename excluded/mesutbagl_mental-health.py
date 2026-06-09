import numpy as np 
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import joblib
# Suppress warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train_df=pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')


train_df.head(10)


train_df['Sleep Duration'].value_counts()


# a function merging categories
def consolidate_sleep_duration(duration):
    if duration in ['Less than 5 hours', '2-3 hours', '1-3 hours', 'No']:
        return 'Less than 5 hours'
    elif duration in ['3-4 hours', '4-5 hours']:
        return '3-4 hours'
    elif duration in ['5-6 hours', '4-6 hours', '6-7 hours', '1-6 hours']:
        return '5-6 hours'
    elif duration in ['7-8 hours', '6-8 hours', '8-9 hours']:
        return '7-8 hours'
    elif duration in ['More than 8 hours', '9-11 hours', '10-11 hours', '49 hours']:
        return 'More than 8 hours'
    else:
        return 'Other'

# apply
train_df['Sleep Duration'] = train_df['Sleep Duration'].apply(consolidate_sleep_duration)


test_df.head(10)


train_df.isnull().sum()


train_df['Working Professional or Student'].value_counts()


train_df=train_df.drop('id', axis=1)


test_id=test_df['id']
test_df=test_df.drop('id', axis=1)


train_df.describe()


train_df.info()


train_df.shape


test_df.shape


plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Participants', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Depression', y='CGPA', data=train_df, palette='Set2')
plt.title('CGPA vs Depression Status', fontsize=16)
plt.xlabel('Depression Status (0: No, 1: Yes)', fontsize=12)
plt.ylabel('CGPA', fontsize=12)
plt.grid()
plt.show()


plt.figure(figsize=(12, 8))
correlation_matrix = train_df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()


sns.pairplot(train_df, hue='Depression', vars=['Age', 'Academic Pressure', 'Work Pressure', 'CGPA'], palette='Set2')
plt.suptitle('Pair Plot of Key Features by Depression Status', y=1.02)
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(x='Depression', y='Study Satisfaction', data=train_df, palette='muted')
plt.title('Study Satisfaction by Depression Status', fontsize=16)
plt.xlabel('Depression Status (0: No, 1: Yes)', fontsize=12)
plt.ylabel('Study Satisfaction', fontsize=12)
plt.grid()
plt.show()


imputer = SimpleImputer(strategy='median')


impute_col = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Financial Stress']
train_df[impute_col] = imputer.fit_transform(train_df[impute_col])
test_df[impute_col] = imputer.fit_transform(test_df[impute_col])


train_df.isnull().sum()


train_df['Degree'].value_counts()


train_df['Profession'].fillna('other', inplace=True)
test_df['Profession'].fillna('other', inplace=True)


train_df['Dietary Habits'].fillna('Moderate', inplace=True)
train_df['Degree'].fillna(train_df['Degree'].mode()[0], inplace=True)
test_df['Dietary Habits'].fillna('Moderate', inplace=True)
test_df['Degree'].fillna(test_df['Degree'].mode()[0], inplace=True)


train_df.head()


train_df['Have you ever had suicidal thoughts ?'] = train_df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})
test_df['Have you ever had suicidal thoughts ?'] = test_df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})


train_df['Family History of Mental Illness'] = train_df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})
test_df['Family History of Mental Illness'] = test_df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})


categorical_col = train_df.select_dtypes(include=['object']).columns
numerical_col = train_df.select_dtypes(exclude=['object']).columns


for col in categorical_col:
    encoder = LabelEncoder()
    train_df[col] = encoder.fit_transform(train_df[col])  


X = train_df.drop(columns=['Name', 'Depression'])
y = train_df['Depression']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.4f}') 


categorical_col = test_df.select_dtypes(include=['object']).columns
numerical_col = test_df.select_dtypes(exclude=['object']).columns


for col in categorical_col:
    encoder = LabelEncoder()
    test_df[col] = encoder.fit_transform(test_df[col])  


test_df=test_df.drop('Name', axis=1)


final_pred = model.predict(test_df)


final_pred


submission = pd.DataFrame({
    'id': test_id,
    'Depression': final_pred 
})


submission.to_csv('submission.csv', index=False)


# Save the label encoder and model
joblib.dump(encoder, 'label_encoder.pkl')
joblib.dump(model, 'random_forest_model.pkl')




