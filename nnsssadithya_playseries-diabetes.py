import pandas as pd


data = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv').drop('id',axis=1)
data.head(5)


data.info()


target_col = 'diagnosed_diabetes'


names = data.columns
encode = []
for i in range (len(names)):
    if data[names[i]].dtypes == object:
        encode.append(names[i])
print(encode)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for x in encode:
    data[x] = le.fit_transform(data[x])
    print(le.classes_)


data.head(5)


data.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt


data.hist(bins=50,figsize=(20,15))
plt.show()


data.describe()


correlation_figure, correlation_axis = plt.subplots(figsize = (30,25))
corr_mtrx = data.corr()
correlation_axis = sns.heatmap(corr_mtrx, annot= True)

plt.xticks(rotation = 30, horizontalalignment = 'right', fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()



corr_matrix = data.corr()
relates = corr_matrix[target_col]
relates


attributes = []
for i in range(len(relates)):
    if abs(relates.iloc[i]) > 0.01:
        attributes.append(data.columns[i])
print(attributes)


train_data = data[attributes]


train_data.info()


#Seperating outcomes from test data
label = train_data[target_col]
data = train_data.drop(target_col,axis=1)
data.head(5)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data,label,test_size=0.2, random_state=42)


print("Data Shape:",X_train.shape)
print("Label Shape",y_train.shape)


pd.Series(y_train).value_counts(normalize=True)


import warnings

from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
import time
warnings.filterwarnings("ignore")


SAFE_MODELS = {
    "LogisticRegression": {"max_iter": 1000, "n_jobs": -1},
    "RandomForestClassifier": {"n_estimators": 50, "n_jobs": -1},
    "ExtraTreesClassifier": {"n_estimators": 50, "n_jobs": -1},
    "GradientBoostingClassifier": {"n_estimators": 50},
    "AdaBoostClassifier": {"n_estimators": 50},
    "DecisionTreeClassifier": {},
    "GaussianNB": {},
    "KNeighborsClassifier": {"n_neighbors": 5, "n_jobs": -1},
    "LinearSVC": {"max_iter": 2000}
}



MAX_TIME_PER_MODEL = 180  # seconds

results = []

best_model = None
best_model_name = None
best_accuracy = -1
best_time = float("inf")

classifiers = [
    (name, cls)
    for name, cls in all_estimators(type_filter="classifier")
    if name in SAFE_MODELS
]




for name, Classifier in classifiers:
    try:
        start = time.time()

        model = Classifier(**SAFE_MODELS[name])
        model.fit(X_train, y_train)

        elapsed = time.time() - start

        # Skip if too slow
        if elapsed > MAX_TIME_PER_MODEL:
            print(f"⏭ Skipping {name} (too slow)")
            continue

        acc = model.score(X_val, y_val)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Training Time (s)": round(elapsed, 2)
        })

        # Track best model
        if acc > best_accuracy or (acc == best_accuracy and elapsed < best_time):
            best_accuracy = acc
            best_time = elapsed
            best_model = model
            best_model_name = name

        print(f"✅ {name} done in {elapsed:.2f}s")

    except Exception as e:
        print(f"❌ {name} failed: {e}")



results_df = pd.DataFrame(results).sort_values(
    by=["Accuracy", "Training Time (s)"],
    ascending=[False, True]
)

results_df


print(f"✅ Best Model Saved: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"Training Time: {best_time:.4f} seconds")


import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_model.score(X_val, y_val)


import lightgbm as lgb

lightgbm_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lightgbm_model.fit(X_train, y_train)
lightgbm_model.score(X_val, y_val)


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8
)

gb.fit(X_train, y_train)
gb.score(X_val,y_val)


test_data = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv').drop('id',axis=1)


names = test_data.columns
encode = []
for i in range (len(names)):
    if test_data[names[i]].dtypes == object:
        encode.append(names[i])
print(encode)


le = preprocessing.LabelEncoder()

for x in encode:
    test_data[x] = le.fit_transform(test_data[x])
    print(le.classes_)


test_data.head(5)


test_data.info()


attributes = ['age', 'physical_activity_minutes_per_week', 'diet_score', 'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
test = test_data[attributes]
test.head(5)


test_pred = xgb_model.predict_proba(test)


test_pred.shape


y_pred = (test_pred[:, 1] >= 0.5).astype(int)


col0 = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv").drop([target_col],axis=1)


col1 = pd.Series(y_pred, name=target_col)
submission = pd.concat([col0,col1],axis = 1)
submission.to_csv("submission.csv",index=False)

