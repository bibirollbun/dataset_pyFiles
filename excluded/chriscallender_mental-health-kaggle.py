import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")
test_ids = test["id"]

def clean(data):
    data = data.drop(["id", "Name", "City", "CGPA"], axis=1)

    data.rename(columns={"Working Professional or Student": "Work/Student",
                         "Sleep Duration": "Sleep",
                         "Dietary Habits": "Diet",
                         "Have you ever had suicidal thoughts ?": "Suicidal_Thoughts",
                         "Work/Study Hours": "Hours",
                         "Family History of Mental Illness": "Family_History",
                         "Financial Stress": "Financial_Stress",
                         "Academic Pressure": "Academic_Pressure",
                         "Work Pressure": "Work_Pressure",
                         "Study Satisfaction": "Study_Satisfaction",
                         "Job Satisfaction": "Job_Satisfaction"}, inplace=True)

    data.Diet.fillna("Unknown", inplace=True)
    data.Degree.fillna("Unknown", inplace=True)
    data.Profession.fillna("Student", inplace=True)
    data.Financial_Stress.fillna(data["Financial_Stress"].median(), inplace=True)



    data.Academic_Pressure.fillna(0, inplace=True)
    data.Work_Pressure.fillna(0, inplace=True)
    data["Pressure"] = data["Academic_Pressure"] + data["Work_Pressure"]

    data.Study_Satisfaction.fillna(0, inplace=True)
    data.Job_Satisfaction.fillna(0, inplace=True)
    data["Satisfaction"] = data["Study_Satisfaction"] + data["Job_Satisfaction"]

    data = data.drop(["Academic_Pressure", "Work_Pressure",
                      "Study_Satisfaction", "Job_Satisfaction"], axis=1)

    return data

train = clean(train)
test = clean(test)

le = preprocessing.LabelEncoder()

cols = ["Gender", "Work/Student", "Profession", "Sleep", "Diet",
        "Degree", "Suicidal_Thoughts", "Family_History"]

def label_data(data):
    for col in cols:
        data[col] = le.fit_transform(data[col])
        #print(le.classes_)

label_data(train)
label_data(test)

y = train["Depression"]
X = train.drop(["Depression"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=train["Depression"], random_state=42)

model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

predictions = model.predict(X_test)

print(accuracy_score(y_test, predictions))
print(recall_score(y_test, predictions))

submission_preds = model.predict(test)

df = pd.DataFrame({"id": test_ids.values,
                   "Depression": submission_preds})

df.to_csv("mental_health_kaggle_submission.csv", index=False)

