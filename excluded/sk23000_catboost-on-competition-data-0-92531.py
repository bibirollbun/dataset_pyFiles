# Loading train and test csv
import pandas as pd
import numpy as np 

df = pd.read_csv(r"/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e11/test.csv")
df.head()


df.shape, df_test.shape


df.columns


# Checking the percentage of other in features
other_cols = ['gender','education_level','loan_purpose']
for col in other_cols:
    print(f"{col} Other Percentage: {((df[col] == 'Other').sum()/df.shape[0]).round(2)*100}")



for cols in df.select_dtypes(include=['object']):
    print(f"{cols}: {df[cols].unique()}")



def apply_categorical_mappings(df):
    df = df.copy()

    # gender
    mapping = {
        "Other": "other",
        "Female": "female",
        "Male": "male",
    }
    df["gender"] = df["gender"].replace(mapping)

    # marital_status
    mapping = {
        "Single": "single",
        "Married": "married",
        "Divorced": "divorced",
        "Widowed": "widowed",
    }
    df["marital_status"] = df["marital_status"].replace(mapping)

    # education_level
    mapping = {
        "Other": "other",
        "High School": "high_school",
        "Master's": "masters",
        "Bachelor's": "bachelors",
        "PhD": "phd",
    }
    df["education_level"] = df["education_level"].replace(mapping)

    # loan_purpose
    mapping = {
        "Other": "other",
        "Debt consolidation": "debt_consolidation",
        "Education": "education",
        "Home": "home",
        "Vacation": "vacation",
        "Car": "car",
        "Medical": "medical",
        "Business": "business",
    }
    df["loan_purpose"] = df["loan_purpose"].replace(mapping)


    # fine grade_subgrade (a1..f5)
    mapping = {
        "A5": "a5", "A4": "a4", "A3": "a3", "A2": "a2", "A1": "a1",
        "B5": "b5", "B4": "b4", "B3": "b3", "B2": "b2", "B1": "b1",
        "C5": "c5", "C4": "c4", "C3": "c3", "C2": "c2", "C1": "c1",
        "D5": "d5", "D4": "d4", "D3": "d3", "D2": "d2", "D1": "d1",
        "E5": "e5", "E4": "e4", "E3": "e3", "E2": "e2", "E1": "e1",
        "F5": "f5", "F4": "f4", "F3": "f3", "F2": "f2", "F1": "f1",
    }
    df["grade_subgrade"] = df["grade_subgrade"].replace(mapping)

    # employment_status
    mapping = {
        "Self-employed": "self_employed",
        "Employed": "employed",
        "Unemployed": "unemployed",
        "Retired": "retired",
        "Student": "student",
    }
    df["employment_status"] = df["employment_status"].replace(mapping)

    return df


df = apply_categorical_mappings(df)
df_test  = apply_categorical_mappings(df_test)


# changing the target variable to binary form 
df['loan_paid_back'] = df['loan_paid_back'].astype('int64')


def apply_numeric_transforms(df):
    df = df.copy()
    
    # Create transformed versions (same as you did for df_logistic)
    df["annual_income_log"] = np.log1p(df["annual_income"])
    df["loan_amount_log"] = np.log1p(df["loan_amount"])
    df["debt_to_income_ratio_log"] = np.sqrt(df["debt_to_income_ratio"])
    
    return df



df = apply_numeric_transforms(df)
df_test = apply_numeric_transforms(df_test)


# DF for ensemble methods training
df_ensemble = df.drop(columns=['debt_to_income_ratio','loan_amount','annual_income'])

# DF for ensemble methods testing
df_test_ensemble = df_test.drop(columns=['debt_to_income_ratio','loan_amount','annual_income'])


df_test_ensemble


from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing
id_col = df_ensemble['id']
X = df_ensemble.drop(columns=['loan_paid_back','id'],axis = 1)
y = df_ensemble['loan_paid_back']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)


nominal_cols = ['gender','marital_status','education_level','employment_status','loan_purpose']
ordinal_cols = ["grade_subgrade"]

grade_order = ['a1', 'a2', 'a3', 'a4', 'a5', 'b1', 'b2', 'b3', 'b4', 'b5', 'c1',
       'c2', 'c3', 'c4', 'c5', 'd1', 'd2', 'd3', 'd4', 'd5', 'e1', 'e2',
       'e3', 'e4', 'e5', 'f1', 'f2', 'f3', 'f4', 'f5']

num_cols = X_train.drop(columns=nominal_cols + ordinal_cols).columns.tolist()



from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

preprocess_cat = ColumnTransformer(
    transformers=[
        ("nom", OneHotEncoder(drop=None, handle_unknown='ignore'), nominal_cols),
        ("ord", OrdinalEncoder(categories=[grade_order]), ordinal_cols),
        ("num", "passthrough", num_cols),
        ])

final_cat_clf = CatBoostClassifier(
    iterations=25000,
    learning_rate=0.02,
    depth=3,
    l2_leaf_reg=0.8,
    random_strength=0.5,
    bagging_temperature=0,
    border_count=3000,
    grow_policy='SymmetricTree',
    eval_metric='AUC',
    verbose=500,
    random_seed=42,
    use_best_model=False,   # set False since we're fitting on all data
    task_type='CPU',        # or 'CPU' if no GPU
    loss_function='Logloss',
)

final_pipe_cat = Pipeline(steps=[
    ("preprocess", preprocess_cat), 
    ("model", final_cat_clf),
])

final_pipe_cat.fit(X, y)


test_ids = df_test_ensemble["id"].copy()
X_test = df_test_ensemble.drop(columns=["id"])

test_proba_cat = final_pipe_cat.predict_proba(X_test)[:, 1]
test_proba_cat = np.clip(test_proba_cat, 0, 1)

submission_cat = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": test_proba_cat
})

submission_cat.to_csv("submission.csv", index=False)
print(submission_cat.head())


