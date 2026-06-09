import numpy as np
import pandas as pd
import os
import time
from sklearn.tree import DecisionTreeRegressor

inputFiles = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        inputFiles.append(os.path.join(dirname, filename))
        
        if filename == "train.csv":
            trainingData = pd.read_csv(os.path.join(dirname, filename))
            print("Training data found")
        elif filename == "x_test.csv":
            testData = pd.read_csv(os.path.join(dirname, filename))
            print("Testing data found")

print()
columns = trainingData.columns

insuranceDataFeatures = ['Years Lived','Yearly Earnings', 'Relationship Status', 'Dependent Count', 'Academic Standing', 'Job Title', 'Wellness Index', 'Coverage Class', 'Prior Claims', 'Automobile Age', 'Financial Rating', 'Coverage Period', 'Tobacco Use', 'Physical Activity', 'Asset Category'] # Full feature set of useful training data
# insuranceDataFeaturesTesting = ['Wellness Index', 'Financial Rating', 'Tobacco Use', 'Physical Activity'] # Temporary feature set for testing
X = trainingData[insuranceDataFeatures]
X_test = testData[insuranceDataFeatures]


def cleanTrainingAndTestingData():
    global trainingData, X, testData, X_test

    def formatSmokerData(x): # Formats smoker
        if x == 'Smoker':
            return 1
        elif x == 'Non-Smoker':
            return 0
        else:
            print("Missing or incorrect data")
            return np.nan

    def formatRelationshipData(x): # Formats relationship status
        if x == 'Spouse':
            return 1
        elif x == 'Unmarried':
            return 0
        elif x == 'Separated':
            return 2
        else:
            print("Missing or incorrect data")
            print(x)
            return np.nan
        
    def formatAcademicStandingData(x): # Formats academic standing
        if x == 'Secondary':
            return 0
        elif x == 'Undergraduate':
            return 1
        elif x == 'Graduate':
            return 2
        elif x == 'Doctorate':
            return 3
        else:
            print("Missing or incorrect data")
            return np.nan

    def formatJobTitleData(x): # Formats job title
        if x == 'Jobless':
            return 0
        elif x == 'Working':
            return 1
        elif x == 'Freelancer':
            return 2
        else:
            print("Missing or incorrect data")
            print(x)
            return np.nan

    def formatCoverageClassData(x): # Formats coverage class
        if x == 'Standard':
            return 0
        elif x == 'Full':
            return 1
        elif x == 'Superior':
            return 2
        else:
            print("Missing or incorrect data")
            return np.nan

    def formatPhysicalActivityData(x): # Formats physical activity
        if x == 'Seldom':
            return 0
        elif x == 'Once a Month':
            return 1
        elif x == 'Once a Week':
            return 2
        elif x == 'Everyday':
            return 3
        else:
            print("Missing or incorrect data")
            return np.nan

    def formatAssetCategoryData(x): # Formats asset category
        if x == 'Unit':
            return 0
        elif x == 'Flat':
            return 1
        elif x == 'Residence':
            return 2
        else:
            print("Missing or incorrect data")
            return np.nan

    def yearlyEarningsCorrection(row):
        if row.name >= 200: # Temporary as looping through the entire dataset takes too long; the higher the number, the more accurate the precitions will be
            return row['Yearly Earnings']
        if pd.isna(row['Yearly Earnings']):
            similarRows = trainingData[
                (trainingData['Years Lived'] == row['Years Lived']) &
                (trainingData['Academic Standing'] == row['Academic Standing']) &
                (trainingData['Job Title'] == row['Job Title']) &
                (~trainingData['Yearly Earnings'].isna())
            ]

            if not similarRows.empty:
                return similarRows['Yearly Earnings'].median()
            else:
                return np.nan  # Mark for dropping later
        return row['Yearly Earnings']

    def financialRatingCorrection(row):
        if row.name >= 200: # Temporary as looping through the entire dataset takes too long; the higher the number, the more accurate the precitions will be
            return row['Financial Rating']
        if pd.isna(row['Financial Rating']):
            similarRows = trainingData[
                (trainingData['Yearly Earnings'] == row['Yearly Earnings']) &
                (trainingData['Academic Standing'] == row['Academic Standing']) &
                (trainingData['Job Title'] == row['Job Title']) &
                (~trainingData['Financial Rating'].isna())
            ]

            if not similarRows.empty:
                return similarRows['Financial Rating'].median()
            else:
                return np.nan  # Mark for dropping later
        return row['Financial Rating']
    
    trainingData = trainingData.dropna(subset=['Relationship Status'])
    trainingData = trainingData.dropna(subset=['Client Review'])
    trainingData = trainingData.dropna(subset=['Dependent Count'])
    trainingData = trainingData.dropna(subset=['Prior Claims'])
    trainingData = trainingData.dropna(subset=['Job Title'])

    # Temporary fix to remove rows with NaN in testData

    testData = testData.dropna(subset=['Relationship Status'])
    testData = testData.dropna(subset=['Client Review'])
    testData = testData.dropna(subset=['Dependent Count'])
    testData = testData.dropna(subset=['Prior Claims'])
    testData = testData.dropna(subset=['Job Title'])

    X = X.loc[trainingData.index]
    X_test = X_test.loc[testData.index]

    X.loc[:, 'Tobacco Use'] = X['Tobacco Use'].apply(formatSmokerData) # type: ignore
    print("Formatted tobacco use training data")
    X.loc[:, 'Relationship Status'] = X['Relationship Status'].apply(formatRelationshipData) # type: ignore
    print("Formatted relationship status training data")
    X.loc[:, 'Academic Standing'] = X['Academic Standing'].apply(formatAcademicStandingData) # type: ignore
    print("Formatted academic standing training data")
    X.loc[:, 'Job Title'] = X['Job Title'].apply(formatJobTitleData) # type: ignore
    print("Formatted job title training data")
    X.loc[:, 'Coverage Class'] = X['Coverage Class'].apply(formatCoverageClassData) # type: ignore
    print("Formatted coverage class training data")
    X.loc[:, 'Physical Activity'] = X['Physical Activity'].apply(formatPhysicalActivityData) # type: ignore
    print("Formatted physical activity training data")
    X.loc[:, 'Asset Category'] = X['Asset Category'].apply(formatAssetCategoryData) # type: ignore
    print("Formatted asset category training data")

    X_test.loc[:, 'Tobacco Use'] = X_test['Tobacco Use'].apply(formatSmokerData) # type: ignore
    print("Formatted tobacco use data for test data")
    X_test.loc[:, 'Relationship Status'] = X_test['Relationship Status'].apply(formatRelationshipData) # type: ignore
    print("Formatted relationship status data for test data")
    time.sleep(5) # TESTING
    X_test.loc[:, 'Academic Standing'] = X_test['Academic Standing'].apply(formatAcademicStandingData) # type: ignore
    print("Formatted academic standing data for test data")
    X_test.loc[:, 'Job Title'] = X_test['Job Title'].apply(formatJobTitleData) # type: ignore
    print("Formatted job title data for test data")
    X_test.loc[:, 'Coverage Class'] = X_test['Coverage Class'].apply(formatCoverageClassData) # type: ignore
    print("Formatted coverage class data for test data")
    X_test.loc[:, 'Physical Activity'] = X_test['Physical Activity'].apply(formatPhysicalActivityData) # type: ignore
    print("Formatted physical activity data for test data")
    X_test.loc[:, 'Asset Category'] = X_test['Asset Category'].apply(formatAssetCategoryData) # type: ignore
    print("Formatted asset category data for test data")


    X.loc[:, 'Yearly Earnings'] = trainingData.apply(yearlyEarningsCorrection, axis=1)
    print("Estimated yearly earnings for missing values in training data")
    X.loc[:, 'Financial Rating'] = trainingData.apply(financialRatingCorrection, axis=1)
    print("Estimated financial ratings for missing values in training data")

    X_test.loc[:, 'Yearly Earnings'] = testData.apply(yearlyEarningsCorrection, axis=1)
    print("Estimated yearly earnings for missing values in test data")
    X_test.loc[:, 'Financial Rating'] = testData.apply(financialRatingCorrection, axis=1) 
    print("Estimated financial ratings for missing values in test data")

    # Drops rows where 'Yearly Earnings' or 'Financial Rating' is NaN
    nanIndices = X[X[['Yearly Earnings', 'Financial Rating']].isna().any(axis=1)].index
    X = X.drop(index=nanIndices)
    trainingData = trainingData.loc[X.index]

    X = X.reset_index(drop=True)
    trainingData = trainingData.reset_index(drop=True)


cleanTrainingAndTestingData()
y = trainingData["Policy Cost"]

insuranceModel = DecisionTreeRegressor(random_state=1)
insuranceModel.fit(X, y)

print("Training complete!")
print()
print()


# Testing the model with test data

predictions = insuranceModel.predict(X_test)
predictionNumber = 10

print("---PREDICTIONS---")
print()

print("Predictions for the first", str(predictionNumber), "valid entries:")
print(X_test.head(predictionNumber))
print()
print("Predictions:")
print(predictions[:predictionNumber])

print()
print("Predictions complete")

