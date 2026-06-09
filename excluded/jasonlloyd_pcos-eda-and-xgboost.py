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


trainFilepath = "/kaggle/input/exploring-predictive-health-factors/train.csv"
trainData = pd.read_csv(trainFilepath)

trainData.info()


trainData.Age.value_counts()


trainData["Age"] = trainData["Age"].replace({"20-25": "20-30",
                                             "15-20": "<20",
                                             "Less than 20": "<20",
                                             "35-44": "30-44", "25-30": "20-30",
                                             "45 and above": ">45",
                                             "30-35": "30-44",
                                             "30-25": "30-44",
                                             "30-40": "30-44",
                                             "Less than 20-25": "<20"})

trainData.Age.value_counts()


trainData.Weight_kg.value_counts()


trainData.PCOS.value_counts()


trainData.Hormonal_Imbalance.value_counts()


trainData["Hormonal_Imbalance"] = trainData["Hormonal_Imbalance"].replace({"Yes Significantly": "Yes"})

mask = trainData["Hormonal_Imbalance"] == "No, Yes, not diagnosed by a doctor"
trainData = trainData[~mask]

trainData.Hormonal_Imbalance.value_counts()


trainData.Hyperandrogenism.value_counts()


trainData.Hirsutism.value_counts()


mask = trainData["Hirsutism"] == "No, Yes, not diagnosed by a doctor"
trainData = trainData[~mask]

trainData.Hirsutism.value_counts()


trainData.Conception_Difficulty.value_counts()


trainData["Conception_Difficulty"] = trainData["Conception_Difficulty"].replace({"Yes, diagnosed by a doctor": "Yes"})

mask = trainData["Conception_Difficulty"] == "No, Yes, not diagnosed by a doctor"
trainData = trainData[~mask]

trainData.Conception_Difficulty.value_counts()


trainData.Insulin_Resistance.value_counts()


trainData.Exercise_Frequency.value_counts()


trainData["Exercise_Frequency"] = trainData["Exercise_Frequency"].replace({"6-8 hours": "6-8 Times a Week",
                                                                           "Less than 6 hours": "3-4 Times a Week"})

mask = trainData["Exercise_Frequency"] == "Less than usual"
trainData = trainData[~mask]

trainData.Exercise_Frequency.value_counts()


trainData.Exercise_Type.value_counts()


trainData["Exercise_Type"] = trainData["Exercise_Type"].replace({"Cardio (e.g., running, cycling, swimming)": "Cardio",
                                                                 "Cardio (e.g.": "Cardio",
                                                                 "Flexibility and balance (e.g., yoga, pilates)": "Flexibility and Balance",
                                                                 "Strength training (e.g., weightlifting, resistance exercises)": "Strength",
                                                                 "Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises)": "Cardio",
                                                                 "Cardio (e.g., running, cycling, swimming), Flexibility and balance (e.g., yoga, pilates)": "Cardio, Flexibility and Balance",
                                                                 "High-intensity interval training (HIIT)": "HIIT",
                                                                 "Cardio (e.g., running, cycling, swimming), Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)": "Cardio, Strength, Flexibility and Balance",
                                                                 "Strength training (e.g., weightlifting, resistance exercises), Flexibility and balance (e.g., yoga, pilates)": "Strength, Flexibility and Balance",
                                                                 "Flexibility and balance (e.g., yoga, pilates), None": "Flexibility and Balance",
                                                                 "Flexibility and balance (e.g., yoga, pilates), None": "Flexibility and Balance",
                                                                 "Cardio (e.g., running, cycling, swimming), None": "Cardio",
                                                                 "Strength training": "Strength",
                                                                 "Strength training (e.g.": "Strength", "Flexibility and balance (e.g.": "Flexibility and Balance"})

mask = trainData["Exercise_Type"] == "Somewhat"
trainData = trainData[~mask]

trainData.Exercise_Type.value_counts()


trainData.Exercise_Duration.value_counts()


trainData["Exercise_Duration"] = trainData["Exercise_Duration"].replace({"Less than 30 minutes": "<30 minutes",
                                                                         "More than 30 minutes": ">30 minutes",
                                                                         "20 minutes": "<30 minutes"})

mask = trainData["Exercise_Duration"] == "Less than 6 hours"
trainData = trainData[~mask]

trainData.Exercise_Duration.value_counts()


trainData.Sleep_Hours.value_counts()


trainData["Sleep_Hours"] = trainData["Sleep_Hours"].replace({"Less than 6 hours": "<6 hours",
                                                             "More than 12 hours": ">12 hours",})

trainData.Sleep_Hours.value_counts()


trainData.Exercise_Benefit.value_counts()


import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.filterwarnings("ignore", message=".*get_group.*", category=FutureWarning)

numericalFeatures = trainData.select_dtypes(exclude="object").copy()
categoricalFeatures = trainData.select_dtypes(include="object").drop("PCOS", axis=1).copy()


fig = plt.figure(figsize=(10,5))
plt.title("Distribution of Weight")
sns.histplot(numericalFeatures["Weight_kg"])
plt.show()

fig = plt.figure(figsize=(10,5))
plt.title("Swarmplot of Weight With PCOS Diagnosis")
sns.swarmplot(x=trainData["PCOS"], y=numericalFeatures["Weight_kg"])
plt.show()


fig = plt.figure(figsize=(10,5))
sns.histplot(x="PCOS", hue="PCOS", data=trainData, multiple="stack")
plt.xlabel("PCOS")
plt.show()

for i,col in enumerate(categoricalFeatures.columns):
    fig = plt.figure(figsize=(10,3))
    sns.histplot(x=col, hue="PCOS", data=trainData, multiple="stack")
    plt.xlabel(col)
    plt.gcf().autofmt_xdate()
    plt.show()


trainData[["Age", "PCOS"]].groupby(["Age", "PCOS"]).value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Age"], ascending=False).set_index("Age")


trainData[["Hormonal_Imbalance", "PCOS"]].groupby("Hormonal_Imbalance").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Hormonal_Imbalance", "PCOS"], ascending=False).set_index("Hormonal_Imbalance")


trainData[["Hyperandrogenism", "PCOS"]].groupby("Hyperandrogenism").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Hyperandrogenism", "PCOS"], ascending=False).set_index("Hyperandrogenism")


trainData[["Hirsutism", "PCOS"]].groupby("Hirsutism").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Hirsutism", "PCOS"], ascending=False).set_index("Hirsutism")


trainData[["Conception_Difficulty", "PCOS"]].groupby("Conception_Difficulty").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Conception_Difficulty", "PCOS"], ascending=False).set_index("Conception_Difficulty")


trainData[["Insulin_Resistance", "PCOS"]].groupby("Insulin_Resistance").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Insulin_Resistance", "PCOS"], ascending=False).set_index("Insulin_Resistance")


trainData[["Exercise_Frequency", "PCOS"]].groupby("Exercise_Frequency").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Exercise_Frequency", "PCOS"], ascending=False).set_index("Exercise_Frequency")


trainData[["Exercise_Type", "PCOS"]].groupby("Exercise_Type").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Exercise_Type", "PCOS"], ascending=False).set_index("Exercise_Type")


trainData[["Exercise_Duration", "PCOS"]].groupby("Exercise_Duration").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Exercise_Duration", "PCOS"], ascending=False).set_index("Exercise_Duration")


trainData[["Sleep_Hours", "PCOS"]].groupby("Sleep_Hours").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Sleep_Hours", "PCOS"], ascending=False).set_index("Sleep_Hours")


trainData[["Exercise_Benefit", "PCOS"]].groupby("Exercise_Benefit").value_counts(dropna=True, normalize=True).reset_index(name="Probability").sort_values(by=["Exercise_Benefit", "PCOS"], ascending=False).set_index("Exercise_Benefit")


from sklearn.model_selection import train_test_split

y = trainData["PCOS"]

featuresToDrop = ["Age", "Weight_kg", "ID", "PCOS", "Exercise_Frequency", "Exercise_Type", "Exercise_Duration", "Sleep_Hours", "Exercise_Benefit"]
X = trainData.drop(featuresToDrop, axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

X.head()


from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder

ordFeatures = ["Hormonal_Imbalance", "Hyperandrogenism", "Hirsutism", "Conception_Difficulty", "Insulin_Resistance"]

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, ordFeatures)
    ])

model = XGBRegressor(n_estimators=500, learning_rate=0.5, early_stopping_rounds=10, n_jobs=-1, random_state=1, eval_metric="auc")

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

le = LabelEncoder()
y_train_preproc = le.fit_transform(y_train)
y_valid_preproc = le.transform(y_valid)

X_train_preproc = preprocessor.fit_transform(X_train)
X_valid_preproc = preprocessor.transform(X_valid)

# Cross validation?
pipeline.fit(X_train, y_train_preproc, model__eval_set=[[X_train_preproc, y_train_preproc], [X_valid_preproc, y_valid_preproc]], model__verbose=False)

predictions = pipeline.predict(X_valid)

score = roc_auc_score(y_valid, predictions)
print("Area Under ROC Score:", score) # Best Area Under ROC Score: 0.8352272727272727


import numpy as np

feature_importance = pipeline.named_steps["model"].feature_importances_

feature_names = ordFeatures

sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 5))
plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance Analysis")
plt.show()


testFilepath = "/kaggle/input/exploring-predictive-health-factors/test.csv"
testData = pd.read_csv(testFilepath)


testData.Hormonal_Imbalance.value_counts()


testData.Hyperandrogenism.value_counts()


testData.Hirsutism.value_counts()


testData.Conception_Difficulty.value_counts()


testData["Conception_Difficulty"] = testData["Conception_Difficulty"].replace({"Somewhat": "Yes"})

testData.Conception_Difficulty.value_counts()


testData.Insulin_Resistance.value_counts()


testData["Insulin_Resistance"] = testData["Insulin_Resistance"].replace({"Yes Significantly": "Yes"})

testData.Insulin_Resistance.value_counts()


X_test = testData.drop(featuresToDrop, axis=1, errors="ignore")

testPredictions = pipeline.predict(X_test)

output = pd.DataFrame({"ID": testData.ID, "PCOS": testPredictions})
output.to_csv("submission.csv", index=False)

print("Output saved\n")

print(output.info(),"\n")
print(output.PCOS.describe())

plt.figure(figsize=(10, 3))
plt.title("Prediction Distribution")
sns.kdeplot(output["PCOS"], clip=(0, 1))
plt.show()

