import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")


X = train.drop(columns=["diagnosed_diabetes", "id"])
y = train["diagnosed_diabetes"]


X.columns


# Identify categorical columns dynamically
categorical_cols = X.select_dtypes(include="object").columns.tolist()

ordinal_features = [
    "education_level",
    "income_level",
    "smoking_status"
]

nominal_features = list(set(categorical_cols) - set(ordinal_features))
numeric_features = X.columns.difference(categorical_cols)

ordinal_categories = [
    ["No formal", "Highschool", "Graduate", "Postgraduate"],
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    ["Never", "Former", "Current"]
]


preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("ord", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=ordinal_categories))
        ]), ordinal_features),

        ("nom", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), nominal_features)
    ]
)



final_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ))
])


final_pipeline.fit(X, y)


# Save model
joblib.dump(final_pipeline, "diabetes_model.pkl")


print("âœ… Model saved successfully")


# fastapi
# uvicorn
# pandas
# numpy
# scikit-learn
# joblib


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes risk probability using ML",
    version="1.0"
)

# -----------------------------
# Load model & schema
# -----------------------------
model = joblib.load("/kaggle/working/diabetes_model.pkl")

FEATURE_COLUMNS = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history'
]

# -----------------------------
# Input Schema (ALL OPTIONAL)
# -----------------------------
class PatientData(BaseModel):
    age: Optional[int] = None
    alcohol_consumption_per_week: Optional[float] = None
    physical_activity_minutes_per_week: Optional[float] = None
    diet_score: Optional[float] = None
    sleep_hours_per_day: Optional[float] = None
    screen_time_hours_per_day: Optional[float] = None
    bmi: Optional[float] = None
    waist_to_hip_ratio: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None
    cholesterol_total: Optional[float] = None
    hdl_cholesterol: Optional[float] = None
    ldl_cholesterol: Optional[float] = None
    triglycerides: Optional[float] = None

    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    education_level: Optional[str] = None
    income_level: Optional[str] = None
    smoking_status: Optional[str] = None
    employment_status: Optional[str] = None

    family_history_diabetes: Optional[int] = None
    hypertension_history: Optional[int] = None
    cardiovascular_history: Optional[int] = None

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running ðŸš€"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_diabetes(data: PatientData):
    input_dict = data.dict()

    input_df = pd.DataFrame(columns=FEATURE_COLUMNS)

    for col in FEATURE_COLUMNS:
        input_df.loc[0, col] = input_dict.get(col, np.nan)

    input_df = input_df.apply(pd.to_numeric, errors="ignore")

    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.7:
        risk = "High"
    elif probability >= 0.4:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "diabetes_probability": round(float(probability), 4),
        "risk_level": risk
    }


# cd diabetes_fastapi
# uvicorn app:app --reload


# after above code runs server and fastapi will start running at below localhost and you can use swagger UI for testing fastapi endpoints
# http://127.0.0.1:8000/docs


# input :- 

#{
#  "Age": 45,
#  "BMI": 29.8,
#  "Gender": "Male",
#  "Ethnicity": "Asian",
#  "EducationLevel": "Graduate",
#  "IncomeLevel": "Middle",
#  "SmokingStatus": "Former",
#  "EmploymentStatus": "Employed"
#}


# output :- 

#{
#  "diabetes_probability": 0.7843,
#  "risk_level": "High"
#}

