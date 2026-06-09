import pandas as pd
import numpy as np
from scipy import stats
import random
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import squarify
%matplotlib inline

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")


df_train.head()


df_train.tail()


df_train.columns


df_train.describe()


df_train.info()


rename_dict = {
    "Working Professional or Student": "Employment",
    "Have you ever had suicidal thoughts ?": "Suicidal thoughts",
    "Work/Study Hours": "WS hours",
    "Family History of Mental Illness": "Family illness"
}

df_train.rename(columns=rename_dict, inplace=True)
df_test.rename(columns=rename_dict, inplace=True)


columns = ["Sleep Duration", "Dietary Habits", "Gender", "Employment", "Suicidal thoughts", "Depression", "Family illness"]

for df in [df_train, df_test]:
    if 'Name' in df.columns:
        df.drop(["Name"], axis=1, inplace=True)
    
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

columns_to_clean = ['City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree']

for col in columns_to_clean:
    value_counts = df_train[col].value_counts()
    rare_values = value_counts[value_counts < 13].index

    for df in [df_train, df_test]:
        df[col] = df[col].apply(lambda x: np.nan if x in rare_values else x)


for df in [df_train, df_test]:
    num_cols = df.select_dtypes(include=['int64', 'int32', 'float64']).columns

    for col in num_cols:
        if col not in ["CGPA", "id"]:
            col_min = df[col].min()
            col_max = df[col].max()

            if -32768 <= col_min and col_max <= 32767:
                if df[col].dtype == 'float64':
                    df[col] = df[col].round().astype("Int16")
                else:
                    df[col] = df[col].astype("Int16")


sleep_map = {
    "7-8 hours": 1,
    "more than 8 hours": 2,
    "5-6 hours": 3,
    "less than 5 hours": 4
}

diet_map = {
    "healthy": 1,
    "moderate": 2,
    "unhealthy": 3
}

gender_map = {"male": 0, "female": 1}
employment_map = {"working professional": 1, "student": 0}
yes_no_map = {"no": 0, "yes": 1}

WS_hours = {
    0: 1,
    1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 
    7: 4, 8: 4, 9: 4, 10: 5, 11: 5, 12: 5
}

education_levels = {
    "Class 12": 1,
    "BA": 2, "BSc": 2, "B.Com": 2, "B.Ed": 2, "BCA": 2, "B.Arch": 2, 
    "BBA": 2, "BHM": 2, "B.Pharm": 2, "BE": 2, "B.Tech": 2,
    "MA": 3, "MSc": 3, "M.Com": 3, "M.Ed": 3, "MCA": 3,
    "M.Tech": 3, "MBA": 3, "M.Pharm": 3, "ME": 3, "MHM": 3,
    "MD": 4, "LLB": 4, "LLM": 4, "MBBS": 4, "PhD": 4
}

for df in [df_train, df_test]:
    if "Sleep Duration" in df.columns:
        df["Sleep Duration"] = df["Sleep Duration"].map(sleep_map).astype("Int16")
    
    if "Dietary Habits" in df.columns:
        df["Dietary Habits"] = df["Dietary Habits"].map(diet_map).astype("Int16")
    
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(gender_map)
    
    if "Employment" in df.columns:
        df["Employment"] = df["Employment"].map(employment_map)
    
    if "Suicidal thoughts" in df.columns:
        df["Suicidal thoughts"] = df["Suicidal thoughts"].map(yes_no_map)
    
    if "Family illness" in df.columns:
        df["Family illness"] = df["Family illness"].map(yes_no_map)


for df in [df_train, df_test]:
    if "Degree" in df.columns:
        df["Degree"] = df["Degree"].map(education_levels).astype("Int16")

def assign_class(grade):
    if grade == 0:
        return 0
    elif 5.00 <= grade <= 5.99:
        return 1
    elif 6.00 <= grade <= 6.99:
        return 2
    elif 7.00 <= grade <= 7.99:
        return 3
    elif 8.00 <= grade <= 8.99:
        return 4
    elif 9.00 <= grade <= 10.00:
        return 5
    else:
        return None  


for df in [df_train, df_test]:
    
    if "WS hours" in df.columns:
        df["WS hours"] = df["WS hours"].map(WS_hours)
    
    if "CGPA" in df.columns:
        df["CGPA"] = df["CGPA"].fillna(0)
        df["CGPA"] = df["CGPA"].apply(assign_class)


df_train["Depression"].value_counts()


df_train["Depression"].value_counts(normalize = True)


print("There are {} duplicates in the dataset.".format(df_train.duplicated().sum()))


print("Checking for missing values in each column:")
print(df_train.isnull().sum())


test_ids = df_test['id']

df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)

target_column = 'Depression'

categorical_columns = df_train.select_dtypes(include=['object']).columns
numerical_columns = df_train.select_dtypes(exclude=['object']).columns

print("Target Column:", target_column)
print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


for col in df_train.columns:
    print(f"Columns: {col}")
    print(df_train[col].unique())
    print("-" * 33) 


for column in categorical_columns:
    print(f"\nValue counts in '{column}':\n{df_train[column].value_counts().head(50)}")


print("Checking for missing values in each column:")
print(df_train.isnull().sum())


print("Checking for missing values in each column:")
print(df_test.isnull().sum())


students_with_profession = df_train[(df_train["Employment"] == 0) & (df_train["Profession"].notna())]
print("Number of students who have a profession:", len(students_with_profession))


students_with_profession = df_test[(df_test["Employment"] == 0) & (df_test["Profession"].notna())]
print("Number of students who have a profession:", len(students_with_profession))



for df in [df_train, df_test]:
    mask = (df["Employment"] == 0) & (df["Profession"].notna())
    df.loc[mask, "Profession"] = np.nan


# i label encode this for 2 dataset for reason 
# in missing handling part you will see
professions = [
    "Teacher", "Content Writer", "Architect", "Consultant", "Pharmacist", "HR Manager",
    "Doctor", "Business Analyst", "Chemist", "Entrepreneur", "Chef", "Educational Consultant",
    "Data Scientist", "Lawyer", "Researcher", "Pilot", "Customer Support", "Marketing Manager",
    "Judge", "Travel Consultant", "Manager", "Sales Executive", "Plumber", "Electrician",
    "Financial Analyst", "Software Engineer", "Digital Marketer", "Civil Engineer",
    "UX/UI Designer", "Finanancial Analyst", "Accountant", "Mechanical Engineer",
    "Graphic Designer", "Research Analyst", "Investment Banker"
]

profession_mapping = {profession: idx for idx, profession in enumerate(professions)}

for df in [df_train, df_test]:
    df["Profession"] = df["Profession"].map(profession_mapping)
print("Jobs count")
for job, code in profession_mapping.items():
    print(f"{job}: {code}")


for df in [df_train, df_test]:
    df["Profession"] = df["Profession"].astype("object")

    df.loc[(df["Employment"] == 0) & (df["Profession"].isna()), "Profession"] = 35

    df.loc[df["Employment"] == 1, ["Academic Pressure", "Study Satisfaction"]] = 0 

    df.loc[df["Employment"] == 0, ["Work Pressure", "Job Satisfaction"]] = 0


print("Checking for missing values in each column:")
print(df_train.isnull().sum())


print("Checking for missing values in each column:")
print(df_test.isnull().sum())


df_train.loc[df_train["Profession"].isna(), ["Employment"]].value_counts()


df_test.loc[df_test["Profession"].isna(), ["Employment"]].value_counts()


np.random.seed(369)
random.seed(369)

for df in [df_train, df_test]:
    df["Profession"] = df["Profession"].astype("Int16")

    probabilities = df.loc[(df['Profession'].notna()) & (df['Profession'] != 35), 'Profession'].value_counts(normalize=True)

    missing_idx = df[df['Profession'].isna()].index

    df.loc[missing_idx, 'Profession'] = np.random.choice(
        probabilities.index, size=len(missing_idx), p=probabilities.values
    )

    df["Profession"] = df["Profession"].astype("Int16")


print("Checking for missing values in each column:")
print(df_train.isnull().sum())
print(df_test.isnull().sum())


columns_to_fill = [
    "City", 
    "Work Pressure", 
    "Study Satisfaction", 
    "Job Satisfaction", 
    "Sleep Duration", 
    "Dietary Habits", 
    "Degree",
    "Financial Stress",
    "Academic Pressure"
]

for df in [df_train, df_test]:
    for col in columns_to_fill:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)


print("Checking for missing values in each column:")
print(df_train.isnull().sum())
print(df_test.isnull().sum())


df_train.head()


# df_train.drop(["City"], axis = 1, inplace = True)


# plt.figure(figsize = (12,10), dpi = 80)

# corr = df_train.corr()

# sns.heatmap(corr, cmap = "coolwarm",
#            annot = True, fmt=".2f")

# plt.title("Correlogram of patient", fontsize = 22)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.show()


encoder = TargetEncoder(cols=['City', 'Profession'])

y_train = df_train['Depression'].astype(float)

df_train[['City_encoded', 'Profession_encoded']] = encoder.fit_transform(
    df_train[['City', 'Profession']], 
    y_train
)[['City', 'Profession']]

df_test[['City_encoded', 'Profession_encoded']] = encoder.transform(
    df_test[['City', 'Profession']]
)[['City', 'Profession']]

df_train = df_train.drop(['City', 'Profession'], axis=1)
df_test = df_test.drop(['City', 'Profession'], axis=1)


X_train = df_train.drop('Depression', axis=1)
y_train = df_train['Depression']
X_test = df_test.copy()


numerical_columns = X_train.select_dtypes(include=['float64', 'int64', 'int16']).columns.tolist()


numerical_columns


binary_cols = ['Gender', 'Employment', 'Suicidal thoughts', 'Family illness']

scale_cols = [c for c in X_train.columns if c not in binary_cols]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])


xgb_params = {
     'learning_rate': 0.298913248058474, 
     'max_depth': 9, 
     'min_child_weight': 3, 
     'n_estimators': 673, 
     'subsample': 0.5933970249700855, 
     'gamma': 2.597137534750985, 
     'reg_lambda': 0.11328048420927406, 
     'colsample_bytree': 0.1381203919800721
}

catboost_params = {
    'iterations': 145, 
    'depth': 7, 
    'learning_rate': 0.29930179265937246, 
    'l2_leaf_reg': 1.242352421942431, 
    'random_strength': 8.325681754379957, 
    'bagging_temperature': 0.7869848919618048, 
    'border_count': 139
}

hgb_params = {
    'learning_rate': 0.16299202834206894, 
    'max_iter': 250, 
    'max_depth': 4, 
    'l2_regularization': 7.1826466833939895,
    'early_stopping': True
}

xgb_model = XGBClassifier(**xgb_params, use_label_encoder=False, random_state=369)
catboost_model = CatBoostClassifier(**catboost_params, task_type="GPU", random_state=369, verbose=0)
hgb_model = HistGradientBoostingClassifier(**hgb_params, random_state=369)

stacking_ensemble = StackingClassifier(
    estimators=[
        ('catboost', catboost_model),
        ('xgb', xgb_model),
        ('hgb', hgb_model)
    ],
    final_estimator=LogisticRegression(),
    passthrough=False
)

scoring = make_scorer(accuracy_score)

cv_scores = cross_val_score(stacking_ensemble, X_train, y_train, cv=5, scoring=scoring)

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")

stacking_ensemble.fit(X_train, y_train)

y_hat_test = stacking_ensemble.predict(X_test)

submission = pd.DataFrame({'id': test_ids,
                       'class': y_hat_test})

submission.to_csv('/kaggle/working/submission.csv', index=False)

submission.head()


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# import pandas as pd

# RF_model = RandomForestClassifier(
#     random_state=369,
#     class_weight='balanced'
# )

# param_grid_rf = {
#     'n_estimators': [500],
#     'criterion': ['entropy'],
#     'max_depth': [9],
#     'min_samples_split': [6],
#     'min_samples_leaf': [5],
#     'max_features': [None],
#     'max_leaf_nodes': [80]
# }

# grid_search_rf = GridSearchCV(
#     estimator=RF_model,
#     param_grid=param_grid_rf,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1
# )

# grid_search_rf.fit(X_train, y_train)

# y_hat_test = grid_search_rf.predict(X_test)

# submission = pd.DataFrame({
#     'id': test_ids,
#     'class': y_hat_test
# })

# submission.to_csv('submission4.csv', index=False)
# print(submission.head())


y_train = y_train.astype(int)

xgb_model = XGBClassifier(
    n_jobs = 1,
    objective = 'binary:logistic',
    eval_metric = 'logloss',
    random_state = 369, 
    reg_lambda = 1.0,
    reg_alpha = 0.2,
    max_depth = 3,
    min_child_weight = 7,
    subsample = 0.8,
    colsample_bytree = 0.8,
    learning_rate = 0.05,
    n_estimators = 500
)

xgb_model.fit(X_train, y_train, verbose = False)

y_hat_test = xgb_model.predict(X_test)

submission = pd.DataFrame({
    'id': test_ids,
    'class': y_hat_test
})

submission.to_csv('submission2.csv', index=False)
print(submission.head())


# logreg = LogisticRegression(max_iter = 500, random_state = 369, class_weight = 'balanced')

# param_grid = {
#     'C': [500, 100],
#     'penalty': ['elasticnet'],
#     'solver': ['saga'],
#     'l1_ratio': [1.0]
# }

# grid_logreg = GridSearchCV(
#     estimator = logreg,
#     param_grid = param_grid,
#     cv = 5,
#     scoring = 'accuracy',
#     n_jobs = -1,
#     verbose = 1
# )

# grid_logreg.fit(X_train, y_train)

# print("Best Parameters:", grid_logreg.best_params_)
# print("Best CV Score:", grid_logreg.best_score_)

# y_hat_test = grid_logreg.predict(X_test)

# submission = pd.DataFrame({
#     'id': test_ids,
#     'class': y_hat_test
# })

# submission.to_csv('submission6.csv', index=False)
# print(submission.head())


# svm = SVC(probability = True, random_state = 369, class_weight = 'balanced')

# param_grid_svm = {
#     'kernel': ['linear'],
#     'C': [10],
#     'gamma': ['scale']                
# }

# grid_search_svm = GridSearchCV(
#     estimator = svm,
#     param_grid = param_grid_svm,
#     cv = 10,
#     scoring = 'accuracy',
#     n_jobs = -1,
#     verbose = 1
# )

# grid_search_svm.fit(X_train, y_train)

# print("Best Parameters:", grid_search_svm.best_params_)
# print("Best CV Score:", grid_search_svm.best_score_)

# y_hat_test = grid_search_svm.predict(X_test)

# submission = pd.DataFrame({
#     'id': test_ids,
#     'class': y_hat_test
# })

# submission.to_csv('submission7.csv', index=False)
# print(submission.head())


# mlp = MLPClassifier(
#     max_iter = 300, 
#     early_stopping = True, 
#     random_state = 369
#     )

# param_grid_mlp = {
#     'hidden_layer_sizes': [(32, 16)],
#     'activation': ['relu'],
#     'solver': ['adam'],
#     'alpha': [0.01],
#     'learning_rate': ['constant'],
#     'learning_rate_init': [0.001],
#     'tol': [1e-3],
#     'batch_size': [32]
# }

# grid_search_mlp = GridSearchCV(
#     estimator = mlp,
#     param_grid = param_grid_mlp,
#     cv = 10,
#     scoring = 'accuracy',
#     n_jobs = -1,
#     verbose = 1
# )

# grid_search_mlp.fit(X_train, y_train)

# print("Best Parameters:", grid_search_mlp.best_params_)
# print("Best CV Score:", grid_search_mlp.best_score_)

# y_hat_test = grid_search_mlp.predict(X_test)

# submission = pd.DataFrame({
#     'id': test_ids,
#     'class': y_hat_test
# })

# submission.to_csv('submission8.csv', index=False)
# print(submission.head())

