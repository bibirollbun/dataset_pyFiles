# !pip install -q xgboost lightgbm catboost


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os, time

import seaborn as sns

import matplotlib.pyplot as plt


from sklearn import metrics

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report ,  ConfusionMatrixDisplay, 
                             roc_curve,  roc_auc_score)




# model
import xgboost as xgb
import lightgbm as lgb
import catboost as cb



class CFG:

    trainFile = "/kaggle/input/playground-series-s5e12/train.csv"
    testFile = "/kaggle/input/playground-series-s5e12/test.csv"
    sampleFile = "/kaggle/input/playground-series-s5e12/test.csv"
    targetCol = "diagnosed_diabetes"
    idCol = "id"
    


def printAllcolumnsValue(df, showAll=True):
    for col in df.columns:
        if showAll :
            print(f"{col} : {df[col].unique()}") # print unique value
        else: # only print catergory column
            if df[col].dtype == "object":
                print(f"{col} : {df[col].unique()}") # print unique value


trainDF = pd.read_csv(CFG.trainFile)
trainDF


trainDF.info()


trainDF.isnull().sum() # check Null 


testDF = pd.read_csv(CFG.testFile)
testDF


submit = pd.read_csv(CFG.sampleFile)
submit


testDF.isnull().sum() # check Null 


trainDF.columns


testDF.columns






printAllcolumnsValue(trainDF, showAll=False)


printAllcolumnsValue(testDF, showAll=False)


# Ordinal mappings each catergoy into numberic representation
genderMap = {"Female": 0, "Male": 1,  "Other": 2}
ethnicityMap = {"Hispanic": 0 , "White": 1, "Asian": 2, "Black": 3, "Other": 4}
eduLvlMap = {"Highschool": 0, "Graduate":1 , "Postgraduate": 2, "No formal": 3}
incomeLvlMap = {"Lower-Middle": 0 , "Upper-Middle":1 , "Low":2 , "Middle": 3, "High": 4}
smokingMap = {"Never": 0, "Current": 1,  "Former": 2}
employmentMap = {"Employed": 0,  "Retired": 1, "Student": 2,  "Unemployed": 3}





def cleanData(df):
    tempDF = df.copy()
    
    # convert catergoy into numberic column
    tempDF["gender"]= tempDF["gender"].map(genderMap)
    tempDF["ethnicity"]= tempDF["ethnicity"].map(ethnicityMap)
    tempDF["education_level"]= tempDF["education_level"].map(eduLvlMap)
    tempDF["income_level"] = tempDF["income_level"].map(incomeLvlMap)
    tempDF["employment_status"] = tempDF["employment_status"].map(employmentMap)
    tempDF["smoking_status"]= tempDF["smoking_status"].map(smokingMap)
    print(tempDF.head(5))
    print(tempDF.isnull().sum()) 
    return tempDF
    


def featureEngineering(df):
    tempDF = df.copy()
    
    tempDF['MAP'] = (tempDF['systolic_bp'] + 2 * tempDF['diastolic_bp']) / 3 # Mean Arterial Pressure
    tempDF['Pulse_Pressure'] = tempDF['systolic_bp'] - tempDF['diastolic_bp'] 
    tempDF["Non_HDL"] = tempDF["cholesterol_total"] - tempDF["hdl_cholesterol"]
    tempDF["Total_HDL_Ratio"] = tempDF["cholesterol_total"] / (tempDF["hdl_cholesterol"] + 1e-5)
    tempDF["TG_HDL_Ratio"]= tempDF["triglycerides"] / (tempDF["hdl_cholesterol"] + 1e-5)

    tempDF["Metabolic_Syndrome_Index"] = tempDF["bmi"] * tempDF["waist_to_hip_ratio"] # MSI index 
    daily_activity_hr = (tempDF["physical_activity_minutes_per_week"]/7)/60 # daily activity per hour
    tempDF["Active_Balance"]= daily_activity_hr - tempDF["screen_time_hours_per_day"]

    return tempDF
    


# clean train data
trainDF = cleanData(trainDF)


# Clean test data
testDF = cleanData(testDF)


trainDF = featureEngineering(trainDF)


trainDF


testDF = featureEngineering(testDF)


testDF


trainDF["family_history_diabetes"].value_counts()


trainDF["hypertension_history"].value_counts()


trainDF["cardiovascular_history"].value_counts()


trainDF.describe()


# trainDF[CFG.targetCol].astype(int).value_counts().plot(kind="bar", title="Diagnosed Diabetes Class");
plt.figure(figsize=(4,3))
sns.countplot(data=trainDF, x=CFG.targetCol)
plt.title("Diagnosed Diabetes Disbribution")
plt.show()



len(trainDF)


majorClassCount = trainDF[CFG.targetCol].value_counts().max()
majorClassCount


# oversample 
oversample_count = majorClassCount
def resample_group_oversample(group):
    '''
    Resample with condition Oversample , keep major
    '''
    n = len(group)
    if n < oversample_count:
        # Oversample with replacement if less than desired_count
        return group.sample(n=oversample_count, replace=True, random_state=42)
    else:
        # Keep as is if exactly desired_count
        return group




trainDF_balanced = trainDF.groupby(CFG.targetCol, group_keys=False).apply(resample_group_oversample)


trainDF_balanced


# trainDF[CFG.targetCol].astype(int).value_counts().plot(kind="bar", title="Diagnosed Diabetes Class");
plt.figure(figsize=(4,3))
sns.countplot(data=trainDF_balanced, x=CFG.targetCol)
plt.title("Balanced Diagnosed Diabetes Disbribution")
plt.show()



featureCols = [c for c in trainDF_balanced.columns if c not in [CFG.idCol , CFG.targetCol]]
print(featureCols)

X = trainDF_balanced[featureCols].copy()
y = trainDF_balanced[CFG.targetCol].astype(int) 


# Train/Test split 
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)


X_train.shape  , X_test.shape,  y_train.shape,  y_test.shape



model1 = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=3500,
    learning_rate = 8e-3,
    max_depth=5, 
    subsample=0.7,
    colsample_bytree=0.4,
    reg_lambda=2.0,
    random_state=42,
    treee_method='hist',
    early_stopping_rounds=200,
    n_jobs=4,
    verbose=1,
)
model1


model1.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)


predictProb = model1.predict_proba(X_test)[:, 1]
predict = model1.predict(X_test)
auc = roc_auc_score(y_test, predictProb)
print(f"Validation AUC: {auc:.5f}")


print("Accurary Score for XGBoost: ", accuracy_score(y_test, predict))
cm = confusion_matrix(y_test, predict)
report = classification_report(y_test, predict)
print('\n\rConfusion matrix for XGBoost:\n\r', cm)
print("\n\rClassification Report For XGBoost :\n\r", report)
xgbCMD=  ConfusionMatrixDisplay(cm)#.plot(ax=axs[0, 0]) 
xgbCMD.plot();
#     axs[0, 0].title.set_text("Confusion Matrix Random Forest")


finalPredictProb = model1.predict_proba(testDF[featureCols])[:, 1]


sub = pd.DataFrame()
sub["id"] = submit['id']
sub["diagnosed_diabetes"] = finalPredictProb


# sub = sub.loc["id", "diagnosed_diabetes"]
sub


sub.to_csv("submission.csv", index=False)


# final = pd.read_csv("subumission.csv")
# final




