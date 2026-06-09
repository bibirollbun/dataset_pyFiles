import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy import stats
from xgboost import XGBClassifier


test_df = pd.read_csv("/kaggle/input/train-test-and-sample-data/test.csv")
train_df = pd.read_csv("/kaggle/input/train-test-and-sample-data/train.csv")
sample_submission_df = pd.read_csv("/kaggle/input/train-test-and-sample-data/sample_submission.csv")


# Drop 'id' column (not useful for model training)
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


print ('Columns: ')
print (train_df.columns.values)
print ('-'*70)
print ('shape: ', train_df.shape)
print("____"*20)
print ('Columns: ')
print (test_df.columns.values)
print ('-'*70)
print ('shape: ', test_df.shape)


print (train_df.info())
print ('*'*20)
print (test_df.info())
print ('*'*20)
print(sample_submission_df.info())


#Categorical Features
# This will show the count, number of unique values, top values and most frquency of categorical columns.
train_df.describe(include=['O'])


# Check for missing values
print(train_df.isna().sum().sort_values())
print("___"*20)
print(test_df.isna().sum().sort_values())


# Print the sample submission data to know it.
sample_submission_df.head().style.set_properties(**{"background-color": "pink","color":"black","border": "1.5px solid Yellow"})


# This function is going to check the data types for columns, unique values and missing values.

def check(df):
    check_list=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        sum_null=df[col].isnull().sum()
        check_list.append([col,dtypes,nunique,sum_null])
    df_check=pd.DataFrame(check_list)
    df_check.columns=['column','dtypes','nunique','sum_null']
    return df_check 
print(check(train_df))
print("____"*20)
print(check(test_df))


# Drop duplicates
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()


# Check the shape of data.
print(f"The shape of train data: {train_df.shape}")
print("___"*20)
print(f"The shape of test data: {test_df.shape}")


print(train_df.isna().sum().sort_values())
print("___"*20)
print(test_df.isna().sum().sort_values())


# Split numeric and categorical columns
train_numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
train_categorical_columns = train_df.select_dtypes(include=['object']).columns

test_numeric_columns = test_df.select_dtypes(include=['float64', 'int64']).columns
test_categorical_columns = test_df.select_dtypes(include=['object']).columns


print(train_numeric_columns)
print("___"*20)
print(test_numeric_columns)


print(train_categorical_columns)
print("___"*20)
print(test_categorical_columns)


# Function to convert days to years
def convert_days_to_years(days):
    return round(days / 365.25, 0)  # Convert and round to 2 decimal places

# Apply the function to the Age column
train_df['Age'] = train_df['Age'].apply(convert_days_to_years)
test_df['Age'] = test_df['Age'].apply(convert_days_to_years)


# For categorical columns in train_df,  I create an imputer
imputer_train_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
train_df[train_categorical_columns] = imputer_train_cat.fit_transform(train_df[train_categorical_columns])

# For categorical columns in test_df,  I create another imputer
imputer_test_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
test_df[test_categorical_columns] = imputer_test_cat.fit_transform(test_df[test_categorical_columns])

# Create an imputer for numerical columns on train_df
imputer_train_num = SimpleImputer(strategy='median')
train_df[train_numeric_columns] = imputer_train_num.fit_transform(train_df[train_numeric_columns])

# Create an imputer for numerical columns on test_df
imputer_test_num = SimpleImputer(strategy='median')
test_df[test_numeric_columns] = imputer_test_num.fit_transform(test_df[test_numeric_columns])


print(train_df.tail())
print("____"*20)
print(test_df.tail())


print(train_df["Cholesterol"].describe())
print("____"*20)
print(test_df["Cholesterol"].describe())


sns.boxplot( x= train_df["Cholesterol"])
plt.title("This is the boxplot for checking the outliers of test data.")
plt.show()

sns.boxplot( x= test_df["Cholesterol"])
plt.title("This is the boxplot for checking the outliers of test data.")
plt.show()


train_df["Cholesterol"].hist(bins=30)
plt.show()

test_df["Cholesterol"].hist(bins=30)
plt.show()


sns.boxplot(train_df["Bilirubin"])
plt.show()

sns.boxplot(test_df["Bilirubin"])
plt.show()


train_df["Bilirubin"].hist(bins=30)
plt.show()

test_df["Bilirubin"].hist(bins=30)
plt.show()


sns.boxplot(train_df["Albumin"])
plt.show()

sns.boxplot(test_df["Albumin"])
plt.show()


train_df["Albumin"].hist(bins=30)
plt.show()

test_df["Albumin"].hist(bins=30)
plt.show()


sns.boxplot(train_df["Copper"])
plt.show()

sns.boxplot(test_df["Copper"])
plt.show()


train_df["Copper"].hist(bins=30)
plt.show()

test_df["Copper"].hist(bins=30)
plt.show()


sns.boxplot(train_df["Alk_Phos"])
plt.show()

sns.boxplot(test_df["Alk_Phos"])
plt.show()


train_df["Alk_Phos"].hist(bins=30)
plt.show()

test_df["Alk_Phos"].hist(bins=30)
plt.show()


sns.boxplot(train_df["SGOT"])
plt.show()

sns.boxplot(test_df["SGOT"])
plt.show()


train_df["SGOT"].hist(bins=30)
plt.show()

test_df["SGOT"].hist(bins=30)
plt.show()


sns.boxplot(train_df["Tryglicerides"])
plt.show()

sns.boxplot(test_df["Tryglicerides"])
plt.show()


train_df["Tryglicerides"].hist(bins=30)
plt.show()

test_df["Tryglicerides"].hist(bins=30)
plt.show()


sns.boxplot(train_df["Platelets"])
plt.show()

sns.boxplot(test_df["Platelets"])
plt.show()


train_df["Platelets"].hist(bins=30)
plt.show()

test_df["Platelets"].hist(bins=30)
plt.show()


sns.boxplot(train_df["Prothrombin"])
plt.show()

sns.boxplot(test_df["Prothrombin"])
plt.show()


train_df["Prothrombin"].hist(bins=30)
plt.show()

test_df["Prothrombin"].hist(bins=30)
plt.show()


# Create a function to apply log transformation
def log_transform(df, column):
    # Add a small constant to avoid log(0)
    df[column] = np.log1p(df[column])  # log1p is log(1 + x)
    return df

# Apply log transformation to the numeric columns in the training DataFrame
for column in train_numeric_columns:
    train_df = log_transform(train_df, column)
    

# Apply log transformation to the numeric columns in the testing DataFrame
for column in test_numeric_columns:
    test_df = log_transform(test_df, column)


print(train_df["Cholesterol"].describe())
print("___"*20)
print(test_df["Cholesterol"].describe())


sns.boxplot(x = train_df["Cholesterol"])
plt.show()

sns.boxplot(x = test_df["Cholesterol"])
plt.show()


sns.histplot(train_df["Cholesterol"], color="green", label="Cholesterol")
plt.title('Distribution of Cholesterol.')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Cholesterol"], color="green", label="Cholesterol")
plt.title('Distribution of Cholesterol.')
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["Bilirubin"].describe())
print("___"*20)
print(test_df["Bilirubin"].describe())


sns.boxplot(train_df["Bilirubin"])
plt.show()

sns.boxplot(test_df["Bilirubin"])
plt.show()


sns.histplot(train_df["Bilirubin"], color="green", label="Bilirubin")
plt.title('Distribution of Bilirubin.')
plt.xlabel('Bilirubin')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Bilirubin"], color="green", label="Bilirubin")
plt.title('Distribution of Bilirubin.')
plt.xlabel('Bilirubin')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["Albumin"].describe())
print("___"*20)
print(test_df["Albumin"].describe())


sns.boxplot(train_df["Albumin"])
plt.show()

sns.boxplot(test_df["Albumin"])
plt.show()


sns.histplot(train_df["Albumin"], color="green", label="Albumin")
plt.title('Distribution of Albumin.')
plt.xlabel('Albumin')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Albumin"], color="green", label="Albumin")
plt.title('Distribution of Albumin.')
plt.xlabel('Albumin')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["Copper"].describe())
print("___"*20)
print(test_df["Copper"].describe())


sns.boxplot(train_df["Copper"])
plt.show()

sns.boxplot(test_df["Copper"])
plt.show()


sns.histplot(train_df["Copper"], color="green", label="Copper")
plt.title('Distribution of Copper.')
plt.xlabel('Copper')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Copper"], color="green", label="Copper")
plt.title('Distribution of Copper.')
plt.xlabel('Copper')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["Alk_Phos"].describe())
print("___"*20)
print(test_df["Alk_Phos"].describe())


sns.boxplot(train_df["Alk_Phos"])
plt.show()

sns.boxplot(test_df["Alk_Phos"])
plt.show()


sns.histplot(train_df["Alk_Phos"], color="green", label="Alk_Phos")
plt.title('Distribution of Alk_Phos.')
plt.xlabel('Alk_Phos')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Alk_Phos"], color="green", label="Alk_Phos")
plt.title('Distribution of Alk_Phos.')
plt.xlabel('Alk_Phos')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["SGOT"].describe())
print("___"*20)
print(test_df["SGOT"].describe())


sns.boxplot(train_df["SGOT"])
plt.show()

sns.boxplot(test_df["SGOT"])
plt.show()


sns.histplot(train_df["SGOT"], color="green", label="SGOT")
plt.title('Distribution of SGOT.')
plt.xlabel('SGOT')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["SGOT"], color="green", label="SGOT")
plt.title('Distribution of SGOT.')
plt.xlabel('SGOT')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["Tryglicerides"].describe())
print("___"*20)
print(test_df["Tryglicerides"].describe())


sns.boxplot(train_df["Tryglicerides"])
plt.show()

sns.boxplot(test_df["Tryglicerides"])
plt.show()


sns.histplot(train_df["Tryglicerides"], color="green", label="Tryglicerides")
plt.title('Distribution of Tryglicerides.')
plt.xlabel('Tryglicerides')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Tryglicerides"], color="green", label="Tryglicerides")
plt.title('Distribution of Tryglicerides.')
plt.xlabel('Tryglicerides')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["Platelets"].describe())
print("___"*20)
print(test_df["Platelets"].describe())


sns.boxplot(train_df["Platelets"])
plt.show()

sns.boxplot(test_df["Platelets"])
plt.show()


sns.histplot(train_df["Platelets"], color="green", label="Platelets")
plt.title('Distribution of Platelets.')
plt.xlabel('Platelets')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Platelets"], color="green", label="Platelets")
plt.title('Distribution of Platelets.')
plt.xlabel('Platelets')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


print(train_df["Prothrombin"].describe())
print("___"*20)
print(test_df["Prothrombin"].describe())


sns.boxplot(train_df["Prothrombin"])
plt.show()

sns.boxplot(test_df["Prothrombin"])
plt.show()


sns.histplot(train_df["Prothrombin"], color="green", label="Prothrombin")
plt.title('Distribution of Prothrombin.')
plt.xlabel('Prothrombin')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

sns.histplot(test_df["Prothrombin"], color="green", label="Prothrombin")
plt.title('Distribution of Prothrombin.')
plt.xlabel('Prothrombin')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


# Check if there is any remaining missing values.
print(train_df.isnull().sum())
print("___"*20)
print(test_df.isnull().sum())


print(train_df.head(10))
print("____"*20)
print(test_df.head())


sns.histplot(train_df["Drug"], color="green", label="Drug")
plt.title('Distribution of Drug')
plt.xlabel('Drug')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)


sns.histplot(train_df["Age"], bins=30, kde=True, color="blue", label="Age")
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)


industry_counts = train_df["Bilirubin"].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(y=industry_counts.index, x=industry_counts.values, palette='husl')
plt.title('Distribution of Bilirubin Levels in Patients')
plt.xlabel('Number of Patients')
plt.ylabel('Bilirubin Level')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


sns.histplot(train_df['N_Days'], bins = 30, kde = True, color = 'blue')
plt.title('Distribution of N_Days')
plt.xlabel('N_Days')
plt.ylabel('Number of N_Days')
plt.show()


sns.histplot(train_df['Sex'], bins = 30, color = 'blue')
plt.title('Distribution of Sex')
plt.xlabel('Sex')
plt.ylabel('Number of Sex')
plt.show()



sns.histplot(train_df['Edema'], bins = 30, color = 'blue')
plt.title('Distribution of Edema Severity Levels')
plt.xlabel('Number of Patients')
plt.ylabel('Edema Severity Level')
plt.show()


# Encoding categorical variables
label_encoders = {}
for col in train_categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

for col in test_categorical_columns:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])
    label_encoders[col] = le
 


# Encode the target variable
target_encoder = LabelEncoder()
train_df['Status'] = target_encoder.fit_transform(train_df['Status'])


# Train-Test Split
X = train_df.drop(columns=['Status'])
y = train_df['Status']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize Models with Class Weights
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
}

# Define Hyperparameter Grids
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    },
}

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit and resample the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train & Evaluate Models with Resampled Data
model_results = {}
for name, model in models.items():
    print(f"Tuning {name} with SMOTE...")
    
    # Choose the appropriate parameter grid
    param_grid = param_grids[name]
    
    # Use GridSearchCV or RandomizedSearchCV based on your needs
    if name in ["Random Forest", "XGBoost"]:
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10,
                                     scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
    else:
        search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5,
                              verbose=2, n_jobs=-1)
    
    # Fit the model with hyperparameter tuning on resampled data
    search.fit(X_train_resampled, y_train_resampled)
    
    best_model = search.best_estimator_
    
    # Best model and evaluation
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    model_results[name] = {
        "Best Parameters": search.best_params_,
        "Accuracy": acc,
        "Classification Report": report,
        "Confusion Matrix": conf_matrix,
        "Best Model": best_model  # Save the best model here
    }


# Display Model Results
for name, results in model_results.items():
    print(f"\n{name} Results:")
    print("Best Parameters:", results["Best Parameters"])
    print("Accuracy:", results["Accuracy"])
    print("Classification Report:", results["Classification Report"])
    print("Confusion Matrix:\n", results["Confusion Matrix"])


# Choose Best Model (XGBoost was selected based on performance)
best_model = model_results["XGBoost"]["Best Model"]  # Ensure you're using the trained model

# Ensure that the target classes are in the right order
status_classes = target_encoder.classes_

# Get predicted probabilities for the test set using the best model
test_probs = best_model.predict_proba(test_df)

# Create a DataFrame to store the predictions
probabilities_df = pd.DataFrame(test_probs, columns=status_classes)

# Prepare the submission format
submission = pd.DataFrame({
    "id": range(15000, 15000 + len(test_probs)),  # Adjust starting ID as needed
    "Status_C": probabilities_df[status_classes[0]],
    "Status_CL": probabilities_df[status_classes[1]],
    "Status_D": probabilities_df[status_classes[2]]
})

# Save the submission to a CSV file
submission.to_csv("submission.csv", index=False)

# Output the first few rows to check the result
print(submission.head())

# Re-import the submission.csv to plot the results
submission_df = pd.read_csv("submission.csv")


# Create a bar plot to compare the probabilities for each ID
submission_df.set_index('id')[['Status_C', 'Status_CL', 'Status_D']].head(10).plot(kind='bar', figsize=(12, 6))
plt.title("Top 10 Predictions for Status Probabilities", fontsize=14)
plt.xlabel("ID", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.grid(True)
plt.show()





