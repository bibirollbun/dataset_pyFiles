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


import os  # Operating system interactions

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # High-level data visualization based on matplotlib
from scipy import stats

pd.set_option('display.max_rows', None)  # Display all rows in pandas DataFrame
pd.set_option('display.max_columns', None)  # Display all columns in the DataFrame

from sklearn.impute import SimpleImputer  # Handling missing values
from sklearn.preprocessing import OneHotEncoder  # Encoding categorical features
from sklearn.compose import ColumnTransformer  # Applying transformers to columns
from sklearn.pipeline import Pipeline  # Assembling steps for cross-validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Machine learning algorithm for classification
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score  # Cross-validation for evaluating scores


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')


test=pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')


train_df=pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')


print(train_df.shape)
print(test_df.shape)


train_df.head()


# Find all duplicates
all_duplicates = train_df[train_df.duplicated(keep=False)]
print(all_duplicates)


# Count duplicates
duplicate_count = train_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")


# Count duplicates
duplicate_count = test_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")


def nullpercent(df):
    value=(df.isnull().sum()/df.shape[0])*100
    print(value)


nullpercent(train_df)


# Include NaN values, zeros, and negatives in value counts
train_df['Academic Pressure'].value_counts(dropna=False)


train_df['CGPA'].value_counts(dropna=False)


train_df['Study Satisfaction'].value_counts(dropna=False)


train_df['Job Satisfaction'].value_counts(dropna=False)


train_df['Profession'].value_counts(dropna=False)


train_df['Working Professional or Student'].value_counts(dropna=False)


train_df['Sleep Duration'].value_counts()


import re

def extract_numeric_duration(value):
    if isinstance(value, str):
        # Extract numeric ranges or single values
        match = re.findall(r'\d+\.?\d*', value)  # Find all numbers
        if len(match) == 1:
            return float(match[0])  # Single numeric value
        elif len(match) == 2:
            # Average for a range (e.g., 7-8 hours → (7+8)/2)
            return (float(match[0]) + float(match[1])) / 2
    return np.nan  # Set non-numeric entries to NaN


# Apply function to column
train_df['Numeric Sleep Duration'] = train_df['Sleep Duration'].apply(extract_numeric_duration)


train_df.head(50)


train_df['Numeric Sleep Duration'].value_counts(dropna=False)


# Exclude values greater than 10 hours and less than 4.5 hours for mean calculation
valid_values = train_df[(train_df['Numeric Sleep Duration'] <= 10) & (train_df['Numeric Sleep Duration'] >= 4.5)]

# Calculate mean of valid values
mean_value = valid_values['Numeric Sleep Duration'].mean()

# Replace values where Numeric Sleep Duration > 10 or < 4.5 with the calculated mean
train_df['Numeric Sleep Duration'] = train_df['Numeric Sleep Duration'].apply(
    lambda x: mean_value if x > 10 or x < 4.5 else x
)


train_df['Numeric Sleep Duration'].value_counts(dropna=False)


# Calculate the mean of the column, ignoring NaN values
mean_value = train_df['Numeric Sleep Duration'].mean()

# Replace NaN values with the calculated mean
train_df['Numeric Sleep Duration'] = train_df['Numeric Sleep Duration'].fillna(mean_value)


train_df['Numeric Sleep Duration'].value_counts(dropna=False)


test_df['Numeric Sleep Duration'] = test_df['Sleep Duration'].apply(extract_numeric_duration)


test_df['Numeric Sleep Duration'].value_counts(dropna=False)


# Replace values where Numeric Sleep Duration > 10 or < 4.5 with the calculated mean
test_df['Numeric Sleep Duration'] = test_df['Numeric Sleep Duration'].apply(
    lambda x: mean_value if x > 10 or x < 4.5 else x
)


# Replace NaN values with the calculated mean
test_df['Numeric Sleep Duration'] = test_df['Numeric Sleep Duration'].fillna(mean_value)


test_df['Numeric Sleep Duration'].value_counts(dropna=False)


nullpercent(train_df)


# Drop the 'Sleep Duration' column from the DataFrame
train_df = train_df.drop(columns=['Sleep Duration'])
test_df = test_df.drop(columns=['Sleep Duration'])


train_df['Profession'].value_counts(dropna=False)


data = [
    "Student", "Academic", "Unemployed", "Profession", "Yogesh",
    "BCA", "MBA", "LLM", "PhD", np.nan, "Analyst", "Pranav",
    "Visakhapatnam", "M.Ed", "Moderate", "Nagpur", "B.Ed",
    "Unveil", "BBA", "MBBS", "Working Professional", "Medical Doctor",
    "City Manager", "FamilyVirar", "Dev", "BE", "B.Com",
    "Family Consultant", "Yuvraj", "Patna", "Unhealthy", "Surat",
    "MD", "City Consultant", "No", "MCA", "Surgeon", "M.Tech",
    "Simran", "B.Pharm", "Name", "Samar", "Manvi", "24th",
    "ME", "3M", "M.Pharm"
]


# Replace using apply with lambda
train_df["Profession"] = train_df["Profession"].apply(
    lambda x: 'other' if x in data or pd.isna(x) else x
)


train_df['Profession'].value_counts(dropna=False)


# Replace using apply with lambda
test_df["Profession"] = test_df["Profession"].apply(
    lambda x: 'other' if x in data or pd.isna(x) else x
)


test_df['Profession'].value_counts(dropna=False)


sns.displot(train_df['Profession'], kde=True)


sns.displot(test_df['Profession'], kde=True)


train_df['Depression'].value_counts()


print(nullpercent(train_df))
print('*************************')
print(nullpercent(test_df))


train_df['Academic Pressure'].value_counts(dropna=False)


train_df[train_df['Academic Pressure'].isna()].head()


train_df['Academic Pressure']=train_df['Academic Pressure'].fillna(0)
test_df['Academic Pressure']=test_df['Academic Pressure'].fillna(0)


print(nullpercent(train_df))
print('*************************')
print(nullpercent(test_df))


train_df['Work Pressure'].value_counts(dropna=False)


train_df[train_df['Work Pressure'].isna()].head()


train_df['Work Pressure']=train_df['Work Pressure'].fillna(0)
test_df['Work Pressure']=test_df['Work Pressure'].fillna(0)


train_df['Study Satisfaction'].value_counts(dropna=False)


train_df['Study Satisfaction']=train_df['Study Satisfaction'].fillna(0)
test_df['Study Satisfaction']=test_df['Study Satisfaction'].fillna(0)


train_df['Job Satisfaction']=train_df['Job Satisfaction'].fillna(0)
test_df['Job Satisfaction']=test_df['Job Satisfaction'].fillna(0)


train_df['Dietary Habits'].value_counts(dropna=False)


values_list = [
    'NaN', 'Yes', 'No', 'More Healthy', 'Class 12', 'Indoor', 'Male', 'Vegas', 'M.Tech',
    'Less Healthy', 'No Healthy', 'Hormonal', 'Electrician', '1.0', 'Mihir', 'Less than Healthy',
    '3', 'Gender', 'BSc', 'Pratham', '2', 'Educational', 'Naina', 'Raghav', 'Vivaan',
    '5 Unhealthy', 'Soham', '5 Healthy', 'Academic', 'MCA', 'Resistant', 'Mealy', 'Prachi',
    'Kolkata'
]


# Replace using apply with lambda
train_df["Dietary Habits"] = train_df["Dietary Habits"].apply(
    lambda x: 'other' if x in values_list or pd.isna(x) else x
)


# Replace using apply with lambda
test_df["Dietary Habits"] = test_df["Dietary Habits"].apply(
    lambda x: 'other' if x in values_list or pd.isna(x) else x
)


test_df['Dietary Habits'].value_counts(dropna=False)


train_df['Degree'].value_counts(dropna=False)


data_Degree = [
    np.nan, "M.Arch", "UX/UI Designer", "B.Sc", "Kalyan", "M", "LLBA", "NaN", "BArch", "L.Ed", "BPharm",
    "P.Com", "Nalini", "BEd", "B", "Degree", "Jhanvi", "Bhopal", "MEd", "LL B.Ed", "LLTech", "M_Tech",
    "5.88", "Pihu", "HCA", "Marsh", "Lata", "S.Arch", "BB", "LHM", "8.56", "Entrepreneur", "Aarav",
    "B.Student", "E.Tech", "M.S", "Navya", "Mihir", "RCA", "B B.Com", "LCA", "N.Pharm", "Doctor",
    "CGPA", "LLEd", "LLS", "Esha", "Working Professional", "Mthanya", "B.3.79", "K.Ed", "Mahika",
    "24", "M. Business Analyst", "Brithika", "ACA", "Badhya", "HR Manager", "Unite", "P.Pharm",
    "MPharm", "Data Scientist", "LL.Com", "Business Analyst", "H_Pharm", "Class 11", "20", "S.Tech",
    "Veda", "BH", "MPA", "S.Pharm", "Vrinda", "Bhavesh", "Brit", "B.B.Arch", "7.06", "B BA",
    "5.56", "Ritik", "B.03", "5.61", "0", "Plumber", "BPA", "Vivaan", "MTech", "29", "LLCom", "Advait",
    "BTech", "3.0", "B.M.Com", "Eshita", "M.UI", "B.H", "Mechanical Engineer", "I.Ed", "Magan", "B B.Tech",
    "M.B.Ed", "B Financial Analyst", "GCA", "G.Ed", "Rupak", "B.CA", "PCA", "J.Ed", "8.95", "Aadhya",
    "Banchal", "M.", "B.BA", "Moham", "B. Gender", "A.Ed", "Vibha", "B BCA", "B.Press", "Gagan",
    "Travel Consultant", "5.65", "B_Com", "E.Ed", "B._Pharm", "Pune", "Bian", "B.Study_Hours",
    "Kavya", "M.M.Ed", "BHCA"
]


# Replace using apply with lambda
train_df["Degree"] = train_df["Degree"].apply(
    lambda x: 'other' if x in data_Degree or pd.isna(x) else x
)


# Replace using apply with lambda
test_df["Degree"] = test_df["Degree"].apply(
    lambda x: 'other' if x in data_Degree or pd.isna(x) else x
)


test_df['Degree'].value_counts(dropna=False)


train_df['Financial Stress'].value_counts(dropna=False)


train_df = train_df.dropna(subset=['Financial Stress'])
test_df = test_df.dropna(subset=['Financial Stress'])


print(nullpercent(train_df))
print('*************************')
print(nullpercent(test_df))


train_df.drop(columns='CGPA', inplace=True)
test_df.drop(columns='CGPA', inplace=True)


train_df.head()


train_df['City'].value_counts(dropna=False)


test_df['City'].value_counts(dropna=False)


names = [
    "Mihir", "Nandini", "Mahi", "Vidya", "City", "Pratyush",
    "Harsha", "Saanvi", "Vidya", "Siddhesh", "Bhavna", "Vikram",
    "Keshav", "Nalini", "City", "Hrithik", "San Vasai-Virar",
    "Vaikot", "Leela", "Chemist", "Ghopal", "No", "More Delhi",
    "Saanvi", "Pratham", "Vidhi", "Abhinav", "Rolkata", "Parth",
    "Aditi", "Saurav", "Sara", "Less Delhi", "Golkata", "Is Kanpur",
    "Unaly", "Thani", "Lawyer", "Vaishnavi", "Ira", "Avni",
    "Mhopal", "Less than 5 hours", "Pratyush", "Malyan", "No.12",
    "Bhavna", "Molkata", "MCA", "M.Com", "Atharv", "Nalini",
    "Keshav", "Ayush", "M.Tech", "Researcher", "Vaishnavi",
    "Chhavi", "Parth", "Vidhi", "Tushar", "MSc", "No",
    "Rashi", "ME", "Ishanabad", "Armaan", "Kagan", "Kashish",
    "Ithal", "Nalyan", "Dhruv", "Galesabad", "Itheg", "Aaradhya",
    "Pooja", "Khushi", "Khaziabad", "Jhanvi", "Kibara", "Harsh",
    "Reyansh", "Morena", "Less Delhi", "Malyansh", "Aditya", "Plata",
    "Aishwarya", "3.0", "Less than 5 Kalyan", "Krishna", "Mira",
    "Moreadhyay", "Ishkarsh", "Raghavendra", "Kashk", "Gurgaon",
    "Tolkata", "Anvi", "Krinda", "Ayansh", "Shrey", "Ivaan",
    "Vaanya", "Gaurav", "Unirar"
]


# Replace using apply with lambda
train_df["City"] = train_df["City"].apply(
    lambda x: 'other' if x in names or pd.isna(x) else x
)


# Replace using apply with lambda
test_df["City"] = test_df["City"].apply(
    lambda x: 'other' if x in names or pd.isna(x) else x
)


train_df.drop(columns=['id', 'Name'], inplace=True)
test_df.drop(columns=['id', 'Name'], inplace=True)


train_df.head()


# Convert Gender column: Male -> 1, Female -> 0
train_df['Gender'] = train_df['Gender'].replace({'Male': 1, 'Female': 0})

# Convert 'Have you ever had suicidal thoughts?' column: Yes -> 1, No -> 0
train_df['Have you ever had suicidal thoughts ?'] = train_df['Have you ever had suicidal thoughts ?'].replace({'Yes': 1, 'No': 0})

# Convert 'Family History of Mental Illness' column: Yes -> 1, No -> 0
train_df['Family History of Mental Illness'] = train_df['Family History of Mental Illness'].replace({'Yes': 1, 'No': 0})

# Convert Gender column: Male -> 1, Female -> 0
test_df['Gender'] = test_df['Gender'].replace({'Male': 1, 'Female': 0})

# Convert 'Have you ever had suicidal thoughts?' column: Yes -> 1, No -> 0
test_df['Have you ever had suicidal thoughts ?'] = test_df['Have you ever had suicidal thoughts ?'].replace({'Yes': 1, 'No': 0})

# Convert 'Family History of Mental Illness' column: Yes -> 1, No -> 0
test_df['Family History of Mental Illness'] = test_df['Family History of Mental Illness'].replace({'Yes': 1, 'No': 0})


test_df.head()


train_df.columns.to_frame()



#Define the column names
numerical_columns = ['Age', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction',
                     'Work/Study Hours', 'Financial Stress', 'Numeric Sleep Duration']
categorical_columns = ['City', 'Working Professional or Student', 'Profession', 'Dietary Habits', 'Degree']


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Preprocessing pipelines
# For numerical columns, apply StandardScaler (and impute if necessary)
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values if any
    ('scaler', StandardScaler())
])

# For categorical columns, apply OneHotEncoding
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values if any
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])


# Combine the numerical and categorical pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])


# Apply the transformations
X = preprocessor.fit_transform(train_df.drop('Depression', axis=1))
y = train_df['Depression']
test_dataframe = preprocessor.transform(test_df)


# Define models and their parameters
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    #'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    #'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    #'SVC': SVC(),
    #'KNeighbors': KNeighborsClassifier()
}

params = {
    'LogisticRegression': {'C': [0.1, 1, 10, 100]},
    #'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    #'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    #'SVC': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
    #'KNeighbors': {'n_neighbors': [3, 5, 7, 9]}
}

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train models and find best parameters
best_models = {}
best_scores = {}
for name, model in models.items():
    print(f"Training {name}...")
    grid = GridSearchCV(model, params[name], cv=kf, scoring='accuracy')
    grid.fit(X, y)
    best_models[name] = grid.best_estimator_
    best_scores[name] = grid.best_score_
    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best score for {name}: {grid.best_score_}")


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Evaluate each model
validation_scores = {}
for name, model in best_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    validation_scores[name] = accuracy
    print(f"Evaluation of {name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_val, y_pred)}\n")


best_model_name = max(validation_scores, key=validation_scores.get)
best_model = best_models[best_model_name]
print(f"The best model is: {best_model_name} with accuracy: {validation_scores[best_model_name]}")


# Train the best model on the full training data
best_model.fit(X, y)


# Make predictions
predictions = best_model.predict(test_dataframe)


submission = pd.DataFrame({
        "id": test["id"],
        "Depression": predictions.astype('int')
    })

submission.to_csv('submission_.csv', index=False)

