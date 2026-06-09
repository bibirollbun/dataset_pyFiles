# ================================================================
#  FITNESS CALORIE BURN PREDICTOR
# ================================================================


# 1.IMPORTS

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# 2. DATA LOADING

def synthesize_dataset(n_samples=6000, seed=42):
    np.random.seed(seed)
    exercise_types = ['walking','running','cycling','rowing','elliptical','yoga','strength_training','swimming']
    mets = {'walking':3.5,'running':9,'cycling':7.5,'rowing':7,'elliptical':5,'yoga':2.5,'strength_training':6,'swimming':8}
    data=[]
    for _ in range(n_samples):
        et = np.random.choice(exercise_types, p=[0.25,0.15,0.15,0.05,0.10,0.10,0.10,0.10])
        duration = np.clip(np.random.normal(30,12), 5,180)
        weight = np.clip(np.random.normal(75,10), 40,140)
        hr = np.clip(np.random.normal(120 if et in ["running","cycling","rowing","swimming"] else 95,12),60,200)
        base = duration*(mets[et]*3.5*weight)/200
        calories = base*np.random.normal(1,0.08)
        data.append([et,duration,hr,weight,calories])
    return pd.DataFrame(data, columns=["exercise_type","duration_min","heart_rate","weight_kg","calories"])

DATA_PATHS = [
    "fitness_calories.csv",
    "/kaggle/input/fitness_calories/fitness_calories.csv",
    "data.csv"
]

df = None
for p in DATA_PATHS:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            print("Loaded dataset from",p)
            break
        except:
            pass

if df is None:
    print("⚠ No dataset found → Generating synthetic dataset...")
    df = synthesize_dataset()

print("\nDataset Shape:",df.shape)
df.head()


# 3. BASIC INFO

print("\nExercise type distribution:")
print(df["exercise_type"].value_counts(normalize=True).round(3))

print("\nStats:")
print(df.describe().round(2))

# Save a histogram
plt.hist(df["calories"], bins=40)
plt.title("Calorie Distribution")
plt.savefig("calories_hist.png")
plt.close()


# 4. MODEL PREP

X = df[["exercise_type","duration_min","heart_rate","weight_kg"]]
y = df["calories"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pre = ColumnTransformer([
    ("num", StandardScaler(), ["duration_min","heart_rate","weight_kg"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["exercise_type"])
])


# 5. RANDOM FOREST + TUNING

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

pipe = Pipeline([("pre",pre),("model",rf)])

params = {
    "model__n_estimators":[100,200,400],
    "model__max_depth":[None,12,18,24],
    "model__min_samples_leaf":[1,2,4]
}

search = RandomizedSearchCV(
    pipe, params, n_iter=10, cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42, verbose=1
)

search.fit(X_train,y_train)

best = search.best_estimator_
preds = best.predict(X_test)

print("\nBest Params:",search.best_params_)
print("MAE:", mean_absolute_error(y_test,preds))
print("RMSE:", mean_squared_error(y_test,preds,squared=False))
print("R2:", r2_score(y_test,preds))

joblib.dump(best,"calorie_predictor.joblib")
print("\nMODEL SAVED ✔ calorie_predictor.joblib")


# 6. CALORIE PREDICTOR FUNCTION

def predict_calories(exercise, duration, hr, weight):
    X_new = pd.DataFrame([{
        "exercise_type":exercise,
        "duration_min":duration,
        "heart_rate":hr,
        "weight_kg":weight
    }])
    return float(best.predict(X_new)[0])


# 7. WEEKLY WORKOUT CALORIE AGENT

WEIGHT = 75
HEIGHT = 170

WORKOUT = {
    "Day 1 - Leg Day":       (45,130),
    "Day 2 - HIIT":          (30,150),
    "Day 3 - Push":          (40,135),
    "Day 4 - Pull":          (40,128),
    "Day 5 - Core + Abs":    (35,140),
    "Day 6 - Full Body":     (38,145),
    "Day 7 - REST":          (0,80)
}

report = []

for day,(dur,hr) in WORKOUT.items():
    if dur==0:
        cal = 0
    else:
        cal = predict_calories("running", dur, hr, WEIGHT)
    report.append([day, round(cal,1)])

weekly = pd.DataFrame(report, columns=["Day","Predicted Calories Burned"])
print("\nWEEKLY FITNESS REPORT\n")
print(weekly)

print("\nTOTAL WEEKLY BURN:", weekly["Predicted Calories Burned"].sum(),"kcal")


# 8. BMI + INSIGHTS

BMI = WEIGHT / ((HEIGHT/100)**2)
print("\nYour BMI:",round(BMI,2))
print("BMI Status:", "Healthy ✔" if 18.5<=BMI<=24.9 else "Not Ideal ❌")

print("\n⚠ 500 kcal daily deficit → LOSE 0.45 KG weekly")


# 9. EXAMPLE CALL

print("\nExample → 30min running, HR150, 70kg =",
      predict_calories("running",30,150,70),"kcal")

print("\nOUTPUTS CREATED:")
print("✔ calorie_predictor.joblib")
print("✔ calories_hist.png")

workout_plan = {
    "Leg Day": {
        "estimated_calories": 520,
        "benefits": [
            "Builds lower body strength",
            "Boosts testosterone and growth hormones",
            "Strengthens quads, hamstrings, glutes",
            "Burns high calories due to big muscle groups",
            "Improves balance and athletic performance"
        ]
    },
    "HIIT Day": {
        "estimated_calories": 600,
        "benefits": [
            "Burns maximum fat in minimum time",
            "Improves cardiovascular fitness",
            "Boosts metabolism for 24–48 hours",
            "No equipment required",
            "Increases lung capacity (VO2 max)"
        ]
    },
    "Chest & Triceps Day": {
        "estimated_calories": 420,
        "benefits": [
            "Builds upper body pushing strength",
            "Enhances chest and arm muscle definition",
            "Reduces shoulder instability",
            "Improves posture",
            "Supports daily pushing activities"
        ]
    },
    "Back & Biceps Day": {
        "estimated_calories": 450,
        "benefits": [
            "Strengthens spine-supporting muscles",
            "Improves grip strength",
            "Fixes hunched posture",
            "Reduces lower back pain",
            "Creates V-shape aesthetic"
        ]
    },
    "Core & Abs Day": {
        "estimated_calories": 350,
        "benefits": [
            "Reduces belly fat",
            "Improves overall balance and stability",
            "Protects spine and improves posture",
            "Supports all major lifts",
            "Improves functional movement"
        ]
    },
    "Full Body Strength Day": {
        "estimated_calories": 580,
        "benefits": [
            "Works every muscle group together",
            "Best for fat loss + muscle gain",
            "Burns more calories than isolation workouts",
            "Improves functional strength",
            "Boosts hormone response (HGH, testosterone)"
        ]
    },
    "Rest / Active Recovery Day": {
        "estimated_calories": 120,
        "benefits": [
            "Repairs muscle tissues",
            "Prevents overtraining",
            "Reduces risk of injury",
            "Allows nervous system recovery",
            "Improves long-term progress"
        ]
    }
}

# DISPLAY WORKOUT PLAN CLEANLY
for day, details in workout_plan.items():
    print(f"\n==== {day.upper()} ====")
    print(f"Estimated Calories Burned: {details['estimated_calories']} kcal")
    print("Benefits:")
    for i, b in enumerate(details["benefits"], 1):
        print(f"{i}. {b}")



