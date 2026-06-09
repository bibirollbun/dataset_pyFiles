import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

testDf = pd.read_csv('/kaggle/input/thisisit/test.csv')
trainDf = pd.read_csv('/kaggle/input/thisisit/train.csv')
sampleSumbission = pd.read_csv('/kaggle/input/thisisit/sample_submission.csv')


#trainDf.sample(5)
#trainDf.info()


numeric_features = trainDf.select_dtypes(include=['int64', 'float64']).columns.tolist()

# create correlation matrix
corr_matrix = trainDf[numeric_features].corr().abs()

# the upper triangle of correlation matrix
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# plot the heatmap of the upper triangle
plt.figure(figsize=(20, 16))
sns.heatmap(upper_triangle, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()





from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Ensure Target Encoding is Applied BEFORE Dropping 'Target'
target_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}

if 'Target' in trainDf.columns:
    trainDf['Target_Ordinal'] = trainDf['Target'].map(target_mapping)
    trainDf = trainDf.drop(columns=['Target'], errors='ignore')  # Drop original string Target column
else:
    raise KeyError("The 'Target' column is missing from trainDf. Ensure the dataset is loaded correctly.")

# Debug: Check if Target_Ordinal was created successfully
if 'Target_Ordinal' not in trainDf.columns:
    raise KeyError("Error: 'Target_Ordinal' was not created successfully. Check the mapping of 'Target'.")

# Debug: Print available columns
print("Columns after processing:\n", trainDf.columns)

#Feature Engineering Functions
def drop_unnecessary_columns(df):
    """Removes unnecessary columns inside the pipeline."""
    columns_to_drop = ['id', 'Application mode', 'Application order', 'International']
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

def bluecollarWhitecollar(df):
    """Categorizes occupations into Blue Collar (1) and White Collar (0)."""
    white_collar_jobs = {1, 2, 3, 4, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135}
    blue_collar_jobs = {5, 6, 7, 8, 9, 10, 151, 152, 153, 154, 161, 163, 171, 172, 173, 174, 175,
                        181, 182, 183, 191, 192, 193, 194, 195}

    if "Father's occupation" in df.columns and "Mother's occupation" in df.columns:
        df['Father_Blue_Collar'] = df["Father's occupation"].apply(lambda x: 1 if x in blue_collar_jobs else 0)
        df['Mother_Blue_Collar'] = df["Mother's occupation"].apply(lambda x: 1 if x in blue_collar_jobs else 0)

    return df.drop(columns=["Father's occupation", "Mother's occupation"], errors='ignore')

def categorize_education(df):
    """Converts educational qualifications into ordinal values."""
    education_mapping = {
        35: 0, 36: 0,  # No Formal Education
        37: 1, 38: 1, 26: 1, 11: 1,  # Primary Education
        30: 2, 29: 2, 19: 2, 18: 2, 1: 2, 22: 2, 39: 2, 9: 2, 10: 2, 12: 2, 14: 2, 27: 2, 20: 2, 25: 2, 13: 2, 15: 2, 42: 2,  # Secondary Education
        2: 3, 3: 3, 4: 3, 5: 3, 40: 3, 43: 3, 44: 3, 6: 3, 31: 3, 33: 3, 41: 3,  # Higher Education
        34: -1  # Unknown
    }

    if "Father's qualification" in df.columns and "Mother's qualification" in df.columns and "Previous qualification" in df.columns:
        df['Father_Education_Ordinal'] = df["Father's qualification"].map(education_mapping).fillna(-1)
        df['Mother_Education_Ordinal'] = df["Mother's qualification"].map(education_mapping).fillna(-1)
        df['Previous_Education_Ordinal'] = df["Previous qualification"].map(education_mapping).fillna(-1)

    return df.drop(columns=["Father's qualification", "Mother's qualification", "Previous qualification"], errors='ignore')

#Define Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('drop_cols', FunctionTransformer(drop_unnecessary_columns, validate=False), trainDf.columns.drop('Target_Ordinal', errors='ignore')),
    ('num', SimpleImputer(strategy='mean'), ['Admission grade', 'Age at enrollment', 'GDP', 'Inflation rate', 'Unemployment rate']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Marital status', 'Daytime/evening attendance', 'Scholarship holder']),
    ('blue_collar', FunctionTransformer(bluecollarWhitecollar, validate=False), ["Father's occupation", "Mother's occupation"]),
    ('edu', FunctionTransformer(categorize_education, validate=False), ["Father's qualification", "Mother's qualification", "Previous qualification"])
], remainder='passthrough')

#Create Decision Tree Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])

#Train/Test Split
X = trainDf.drop(columns=['Target_Ordinal'], errors='ignore')  # Features
y = trainDf['Target_Ordinal']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train the Model
pipeline.fit(X_train, y_train)

#Predictions & Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



testDf['Predicted_Status'] = pipeline.predict(testDf)
prediction_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
testDf['Predicted_Status'] = testDf['Predicted_Status'].map(prediction_mapping)

# Save Predictions as CSV
output_path = "/content/sample_submission.csv"
testDf.to_csv(output_path, index=False)

