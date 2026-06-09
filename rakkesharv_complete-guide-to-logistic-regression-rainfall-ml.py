























import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report,precision_score,recall_score

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")

# Assign column names
df.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

# Check dataset info
df.head()


# Check for missing values
print('Checking For Missing Values')
print('')
print(df.isnull().sum())
print('-'*88)


#Since there is no missing value, we can proceed to the next step:

# Split dataset
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# Model Fit
model = LogisticRegression()

model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print('-'*88)
print("F1 Score:",f1_score(y_test, y_pred))
print('-'*88)
print("Precision:",precision_score(y_test, y_pred))
print('-'*88)
print("Recall:",recall_score(y_test, y_pred))
print('-'*88)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print('-'*88)
print("Classification Report:\n", classification_report(y_test, y_pred))
print('-'*88)







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df.head()


#Checking Missing Values & Missing % !
missing_values = df.isnull().sum()  # Count missing values
total_values = len(df)  # Total number of rows
missing_percentage = (missing_values / total_values) * 100  # Calculate percentage

# Create a result DataFrame
missing_df = pd.DataFrame({"Missing Values": missing_values,
                           "Percentage (%)": missing_percentage})
missing_df


# Selecting numeric columns for scaling
numeric_columns = df.select_dtypes(include=['number']).columns
# drop('day')

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform numeric columns
train_scaled_df = df.copy()
train_scaled_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
train_scaled_df.head()

#Note: Since the Train and Test Data is separate so we do the Normalization
# to the whole DF, if not, only the TRAIN DATA should be Normalized first and then 
# TEST DATA should be Normalized during predict or testing, 
# this meaure is done to avoid - DATA LEAKAGE! 


# Feature and Target Variables
X= train_scaled_df.drop(['rainfall'], axis=1)
y=train_scaled_df['rainfall']


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)


model=LogisticRegression(C=0.08858667904100823, solver='lbfgs', penalty="l2",max_iter=1000)

# model = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=6, random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print('-'*88)
print("F1 Score:",f1_score(y_test, y_pred))
print('-'*88)
print("Precision:",precision_score(y_test, y_pred))
print('-'*88)
print("Recall:",recall_score(y_test, y_pred))
print('-'*88)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print('-'*88)
print("Classification Report:\n", classification_report(y_test, y_pred))
print('-'*88)



test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test.isna().sum()


# Since there is a single null value, we can impute this will median
test = test.fillna(test['winddirection'].median())


#Prediction using the Test dataset
test_df = pd.DataFrame(test)
test_scaled_df = test_df
scaler=MinMaxScaler()
test_scaled=scaler.fit_transform(test)

# Convert back to DataFrame
test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns)
test_scaled_df.head()


# Predict Model Values
pred = model.predict_proba(test_scaled_df)[:,1]


# add the values inside the sample_submission.csv
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
sub.head()


#submission
sub['rainfall'] = pred
sub.to_csv('submission_v4.csv', index=False)

