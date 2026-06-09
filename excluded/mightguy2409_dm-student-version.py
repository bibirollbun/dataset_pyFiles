

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import math


data = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")


data.head()


print(data.isnull().values.any())


data.shape


print(data.columns.tolist())


data['Working Professional or Student'].value_counts()


ws_data = data.groupby('Working Professional or Student')
student_data = ws_data.get_group('Student').copy()
student_data


missing_counts = student_data.isna().sum()
missing_counts


total_rows = len(student_data)
missing_columns = missing_counts[missing_counts>0]
percent_missing = (missing_columns/total_rows)*100
percent_missing



miss_col = ['Profession','Work Pressure','Job Satisfaction']
student_data.drop(columns = miss_col,inplace=True)


student_data


student_data['Working Professional or Student'].value_counts()


student_data['id'].nunique()


student_data.drop(columns = ['id','Working Professional or Student'],inplace=True)
student_data


student_data.isna().sum()


numerical_cols = ['Academic Pressure', 'CGPA', 'Financial Stress']
categorical_cols = ['Study Satisfaction', 'Dietary Habits']


for col in numerical_cols:
    median_val = student_data[col].median()
    student_data[col] = student_data[col].fillna(median_val)
    print(f"Imputed missing values in '{col}' with median: {median_val}")


for col in categorical_cols:
    mode_val = student_data[col].mode()[0]
    student_data[col]=student_data[col].fillna(mode_val)
    print(f"Imputed missing values in '{col}' with mode: {mode_val}")


print("\nFinal missing value counts:")
print(student_data.isna().sum())


student_data


student_data.dtypes


student_data['Family History of Mental Illness'].value_counts()


student_data['City'].unique()


valid_cities = {
    'Visakhapatnam', 'Bangalore', 'Srinagar', 'Varanasi', 'Jaipur',
    'Pune', 'Thane', 'Chennai', 'Nagpur', 'Nashik', 'Vadodara',
    'Kalyan', 'Rajkot', 'Ahmedabad', 'Kolkata', 'Mumbai', 'Lucknow',
    'Indore', 'Surat', 'Ludhiana', 'Bhopal', 'Meerut', 'Agra',
    'Ghaziabad', 'Hyderabad', 'Vasai-Virar', 'Kanpur', 'Patna',
    'Faridabad', 'Delhi', 'Khaziabad' # Correct a typo to a standard name
}


def clean_city_column(city):
    """
    Cleans the city column by classifying entries as 'Other' if they
    are not in the set of valid city names.
    """
    if pd.isna(city):
        return np.nan # Keep NaNs for later imputation
    
    # Standardize the input for better matching (e.g., handle typos)
    city = str(city).strip()
    if city in valid_cities:
        return city
    
    # Catch a common typo and correct it
    if city == 'Khaziabad':
        return 'Ghaziabad'
        
    if city == 'Less Delhi':
        return 'Delhi'
    
    # If the entry is not a valid city, return 'Other'
    return 'Other'


student_data['City'] = student_data['City'].apply(clean_city_column)


mode_val = student_data['City'].mode()
student_data['City'] = student_data['City'].fillna(mode_val)
print(f"Imputed missing values in 'City' with mode: {mode_val}")


student_data['Have you ever had suicidal thoughts ?'].value_counts()


student_data['Dietary Habits'].value_counts()


mapping_dict = {
    # Map similar categories to the main ones
    'Less than Healthy': 'Unhealthy',
    'No Healthy': 'Unhealthy',
    'Less Healthy': 'Unhealthy',

    # Map incorrect/erroneous entries to 'Other'
    '3': 'Other',
    '1.0': 'Other',
    'Mihir': 'Other',
    'M.Tech': 'Other',
    'Male': 'Other',
    'Yes': 'Other',
    '2': 'Other'
}


mode_val = student_data['Dietary Habits'].mode()[0]
student_data['Dietary Habits'] =student_data['Dietary Habits'].fillna(mode_val)
print(f"Imputed missing values with mode: {mode_val}")


student_data['Dietary Habits'] = student_data['Dietary Habits'].replace(mapping_dict)


student_data['Degree'].unique()


valid_degrees = {
    'B.Pharm': 'B.Pharm', 'BPharm': 'B.Pharm',
    'BSC': 'B.Sc', 'B.Sc': 'B.Sc',
    'BA': 'B.A', 'B.A': 'B.A',
    'BCA': 'BCA',
    'M.Tech': 'M.Tech',
    'PhD': 'PhD',
    'Class 12': 'Class 12', 'Class 11': 'Class 11',
    'B.Ed': 'B.Ed', 'L.Ed': 'B.Ed', 'LL B.Ed': 'B.Ed', 'LLEd': 'B.Ed',
    'LLB': 'LLB',
    'BE': 'BE',
    'M.Ed': 'M.Ed',
    'MSc': 'M.Sc', 'M.Sc': 'M.Sc',
    'BHM': 'BHM', 'MHM': 'MHM',
    'M.Pharm': 'M.Pharm',
    'MCA': 'MCA',
    'MA': 'M.A', 'M.A': 'M.A',
    'B.Com': 'B.Com', 'M.Com': 'M.Com',
    'MD': 'MD', 'MBBS': 'MBBS',
    'MBA': 'MBA',
    'B.Arch': 'B.Arch', 'BArch': 'B.Arch',
    'LLM': 'LLM',
    'B.Tech': 'B.Tech', 'B.Tech': 'B.Tech',
    'BBA': 'BBA',
    'ΜΕ': 'ME', # Correct a potential encoding issue
    'B.Student': 'B.Student',
}


def clean_degree_column(degree):
    """
    Cleans and standardizes the degree column.
    If a degree is not in the valid_degrees dictionary, it returns 'Other'.
    """
    if pd.isna(degree):
        return np.nan
    
    # Check for exact matches first
    if degree in valid_degrees:
        return valid_degrees[degree]
    
    # A more robust check for variations (e.g., lowercase)
    for key, value in valid_degrees.items():
        if str(degree).strip().lower() == str(key).strip().lower():
            return value
    
    # If no match is found, classify as 'Other'
    return 'Other'


student_data['Degree'] = student_data['Degree'].apply(clean_degree_column)


print(student_data['Degree'].value_counts())


student_data['Sleep Duration'].nunique()


student_data['Sleep Duration'].value_counts()


def clean_sleep_duration(duration):
    if pd.isna(duration):
        return np.nan
    
    duration = str(duration).lower()
    
    if 'less than 5' in duration or 'than 5' in duration:
        return 'Less than 5 hours'
    elif '5-6' in duration:
        return '5-6 hours'
    elif '6-7' in duration:
        return '6-7 hours'
    elif '7-8' in duration or '8 hours' in duration:
        return '7-8 hours'
    elif 'more than 8' in duration or '10-11' in duration:
        return 'More than 8 hours'
    else:
        # For entries like '40-45 hours', 'Moderate', '45', and others
        return 'Other/Invalid'

# Apply the cleaning function to the 'Sleep Duration' column
student_data['Sleep Duration'] = student_data['Sleep Duration'].apply(clean_sleep_duration)


mode_sleep_duration = student_data['Sleep Duration'].mode()[0]
student_data['Sleep Duration'] = student_data['Sleep Duration'].fillna(mode_sleep_duration)
print(f"Imputed missing values in 'Sleep Duration' with mode: {mode_sleep_duration}")


student_data.columns


student_data = student_data.rename(columns = {'Have you ever had suicidal thoughts ?':'Suicidal Thoughts','Family History of Mental Illness':'Family History'})


student_data


student_data['Depression'].value_counts(normalize=True) * 100


student_depression_counts = student_data['Depression'].value_counts()
fig,ax = plt.subplots(figsize=(6,6))
ax.bar(['No Depression', 'Depression'], student_depression_counts.values,color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.7)
ax.set_title('Depression Distribution of Students (Target)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count')
for i, v in enumerate(student_depression_counts.values):
    ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
plt.show()



plt.figure(figsize=(8, 6))
plt.hist(student_data['Age'], bins=20, color='#FF6B6B', edgecolor='black', alpha=0.7)
plt.xlabel('Age (years)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Student Ages', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(student_data['CGPA'], fill=True, color='#95E1D3', linewidth=2)
plt.xlabel('CGPA', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.title('Density Plot of Student CGPA', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(student_data['Academic Pressure'], fill=True, color='#c994c7', linewidth=2)
plt.xlabel('Academic Pressure', fontsize=12, fontweight='bold')
plt.ylabel('Density', fontsize=12, fontweight='bold')
plt.title('Density Plot of Student Academic Pressure', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(y=student_data['Work/Study Hours'], color='#74B9FF')
plt.ylabel('Hours per Week', fontsize=12, fontweight='bold')
plt.title('Work/Study Hours Distribution - Box Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
gender_counts = student_data['Gender'].value_counts(dropna=False)
colors_gender = ['#6C5CE7', '#FD79A8']  # Male, Female
explode = [0.05]*len(gender_counts)

plt.pie(
    gender_counts.values,
    labels=gender_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors_gender[:len(gender_counts)],
    explode=explode,
    shadow=True,
    textprops={'fontsize':12, 'fontweight':'bold'}
)
plt.title('Gender Distribution', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()


diet_counts = student_data['Dietary Habits'].value_counts(dropna=False)
diet_counts


labels = diet_counts.index.tolist()
sizes = diet_counts.values
colors = ['#55EFC4', '#FFEAA7', '#FF7675', '#95A5A6']  # add gray for 'Other'
explode = [0.05]*len(sizes)

# Option 1: more precision
plt.figure(figsize=(7,7))
plt.pie(
    sizes, labels=labels, colors=colors[:len(labels)],
    autopct='%1.2f%%', startangle=90, explode=explode,
    pctdistance=0.85,
    textprops={'fontsize':12, 'fontweight':'bold'}
)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)
plt.title('Dietary Habits (Donut)', fontsize=14, fontweight='bold')
plt.axis('equal'); plt.tight_layout(); plt.show()


def lollipop(series, title, colors):
    counts = series.value_counts().reindex(['No','Yes'])
    x = np.arange(len(counts))
    plt.figure(figsize=(14,5))
    plt.hlines(y=x, xmin=0, xmax=counts.values, colors=colors, linewidth=3)
    plt.plot(counts.values, x, 'o', color='black')
    plt.yticks(x, counts.index); plt.xlabel('Count'); plt.title(title, fontweight='bold')
    for val, yi in zip(counts.values, x):
        plt.text(val, yi, f' {val}', va='center', fontsize=15, fontweight='bold')
    plt.tight_layout(); plt.show()




lollipop(student_data['Suicidal Thoughts'], 'Suicidal Thoughts (Lollipop)',['#00B894','#D63031'])
lollipop(student_data['Family History'], 'Family History (Lollipop)',['#74B9FF','#FD79A8'])


student_data


student_data['Dietary Habits'].unique()


student_data['Sleep Duration'].unique()


student_data['Suicidal Thoughts'] = student_data['Suicidal Thoughts'].map({'Yes' : 1, 'No':0})
student_data['Family History'] = student_data['Family History'].map({'Yes' : 1, 'No':0})
student_data['Dietary Habits'] = student_data['Dietary Habits'].map({'Healthy':2, 'Moderate':1, 'Unhealthy':-1, 'Other':0})
student_data['Gender'] = student_data['Gender'].map({'Male':0,'Female':1})


sleep_duration_map = {
    'Less than 5 hours': 0,      
    '5-6 hours': 1,
    '6-7 hours': 2,
    '7-8 hours': 3,
    'More than 8 hours': 4,      
    'Other/Invalid': -1          
}


student_data['Sleep Duration'] = student_data['Sleep Duration'].map(sleep_duration_map)


student_data['Name'].nunique()


student_data['City'].nunique()


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


student_data['Name']= label_encoder.fit_transform(student_data['Name'])
student_data['City']= label_encoder.fit_transform(student_data['City'])
student_data['Degree'] = label_encoder.fit_transform(student_data['Degree'])


student_data


X = student_data.drop(columns=['Depression'])
y = student_data['Depression']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify =y)


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dtree))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forrest Accuracy: ',accuracy_score(y_test,y_pred_rf))


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("AdaBoost accuracy:" ,accuracy_score(y_test,y_pred_ada))


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print('Neural Network accuracy:' ,accuracy_score(y_test,y_pred_mlp))


models = {
    'Logistic Regression': logreg,
    'Decision Tree': dtree,
    'Random Forest': rf,
    'MLP Neural Network': mlp,
    'AdaBoost': ada,
    
}

accuracies = {}

for name, model in models.items():
    # Some models (like CatBoost) may return predictions as float32; cast them to match y_test if needed
    preds = model.predict(X_test)
    if preds.dtype != y_test.dtype:
        preds = preds.astype(y_test.dtype)
    accuracies[name] = accuracy_score(y_test, preds)

# Create and display the accuracy table
accuracy_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
print(accuracy_df)

