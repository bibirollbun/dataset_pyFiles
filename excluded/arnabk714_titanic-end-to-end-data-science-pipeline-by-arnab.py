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


# ===============================
# Auto Data Scientist – Titanic (Google API version)
# ===============================

# 1. Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Google API imports
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import base64
from io import BytesIO

# 2. Google API Setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERVICE_ACCOUNT_FILE = "service_account.json"  # Upload this file in Kaggle

if GOOGLE_API_KEY:
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/documents'
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    docs_service = build('docs', 'v1', credentials=creds)
    print("✅ Google API mode active")
else:
    print("⚡ Running in offline mode")

# 3. Load Titanic Data
train = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv')
test = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/test.csv')

# 4. EDA + Cleaning + Feature Engineering as before
def eda_agent(df):
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Numeric Feature Correlation")
    plt.show()

# Call EDA function
eda_agent(train)

def clean_and_engineer(df):
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame directly
    
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Reassign the result
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Reassign the result
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Reassign the result
    
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Label encoding for categorical features
    for col in ['Sex', 'Embarked']:
        df[col] = LabelEncoder().fit_transform(df[col])
        
    # Drop unnecessary columns
    df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True, errors='ignore')
    
    return df

# Process the train and test data
train_proc = clean_and_engineer(train)
test_proc = clean_and_engineer(test)

# 5. Model Selection
X = train_proc.drop('Survived', axis=1)
y = train_proc['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Model training and selection based on F1 score
best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = f1_score(y_val, y_pred)
    print(name, "F1:", score)
    if score > best_f1:
        best_f1 = score
        best_model = model

print("Selected model:", type(best_model).__name__, "with F1:", best_f1)

# 6. Report Writer with Google Docs + Insights + Plots
def write_report(df, model):
    # Prepare insights
    report_text = f"Titanic Report\nModel: {type(model).__name__}\nF1 Score: {best_f1}\n\n"
    # Example insight
    pivot = df.groupby('Pclass')['Survived'].mean()
    report_text += f"Survival by Pclass: {pivot.to_dict()}\n"
    
    # Create Google Doc
    if GOOGLE_API_KEY:
        doc = docs_service.documents().create(body={'title': 'Titanic Agent Report'}).execute()
        doc_id = doc.get('documentId')
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': [
                {'insertText': {'location': {'index': 1}, 'text': report_text}}
            ]}
        ).execute()
        
        # Function to insert plot image
        def plot_to_doc(plot_func, desc=""):
            buf = BytesIO()
            plot_func()
            plt.savefig(buf, format="PNG", bbox_inches="tight")
            plt.close()
            buf.seek(0)
            img = buf.read()
            
            requests = []
            if desc:
                requests.append({'insertText': {'location': {'index': 1}, 'text': desc + "\n"}})
            requests.append({
                'insertInlineImage': {
                    'location': {'index': 1},
                    'uri': 'data:image/png;base64,' + base64.b64encode(img).decode('utf-8'),
                    'objectSize': {'height': {'magnitude': 300, 'unit': 'PT'},
                                   'width': {'magnitude': 500, 'unit': 'PT'}}
                }
            })
            docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': requests}).execute()
        
        # Example plots
        plot_to_doc(lambda: sns.histplot(data=df, x='Age', hue='Survived', bins=20), "Age vs Survival")
        plot_to_doc(lambda: sns.histplot(data=df, x='Fare', hue='Survived', bins=20), "Fare vs Survival")
        
        print("✅ Report created in Google Docs, Doc ID:", doc_id)
    else:
        print(report_text)

write_report(train, best_model)

# 7. Prediction & Submission
preds = best_model.predict(test_proc)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved!")


