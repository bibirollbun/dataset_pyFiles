import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

testDf = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')
trainDf = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
sampleSumbission = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')


trainDf.sample(5)
trainDf.info()


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

#Feature Engineering Functions
def bluecollarWhitecollar(df):
    """Categorizes occupations into Blue Collar (1) and White Collar (0)."""
    white_collar_jobs = {1, 2, 3, 4, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135}
    blue_collar_jobs = {5, 6, 7, 8, 9, 10, 151, 152, 153, 154, 161, 163, 171, 172, 173, 174, 175,
                        181, 182, 183, 191, 192, 193, 194, 195}

    if "Father's occupation" in df.columns and "Mother's occupation" in df.columns:
        df['Father_Blue_Collar'] = df["Father's occupation"].apply(lambda x: 1 if x in blue_collar_jobs else 0)
        df['Mother_Blue_Collar'] = df["Mother's occupation"].apply(lambda x: 1 if x in blue_collar_jobs else 0)

    return df

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

    return df
trainDf = categorize_education(trainDf)
trainDf = bluecollarWhitecollar(trainDf)


#Define Preprocessing Pipeline
preprocessor = ColumnTransformer([
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
X = trainDf.drop(columns=['Target'], errors='ignore')  # Features
y = trainDf['Target']  # Target

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



trainDf.sample(5)


testDf = bluecollarWhitecollar(testDf)
testDf = categorize_education(testDf)


# Predict using the trained model
y_pred_test = pipeline.predict(testDf)

testDf['Predicted_Target'] = y_pred_test

testDf_filtered = testDf.iloc[:, [0, -1]]

# Save to CSV
testDf_filtered.to_csv("submission.csv", index=False)




