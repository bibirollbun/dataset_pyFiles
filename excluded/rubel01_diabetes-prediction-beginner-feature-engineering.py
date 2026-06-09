import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


#eda


def feature_enginnering(df):
    df["Age_40+"] = (df["age"] >= 40).astype(int)
    df["Age_55+"] = (df["age"] >= 55).astype(int)
    df["Central_Obesity"] = (
    (df["gender"] == "Male") & (df["waist_to_hip_ratio"] > 0.9) |
    (df["gender"] == "Female") & (df["waist_to_hip_ratio"] > 0.85)
    ).astype(int)
    
    df["BMI_WHR"] = df["bmi"] * df["waist_to_hip_ratio"]

    df["Sleep_Screen_Stress"] = (
        (df["sleep_hours_per_day"] < 6) &
        (df["screen_time_hours_per_day"] > 6)
    ).astype(int)
    df["Alcohol_High"] = (df["alcohol_consumption_per_week"] > 14).astype(int)
    df["Alcohol_Zero"] = (df["alcohol_consumption_per_week"] == 0).astype(int)
    df["Low_Activity"] = (df["physical_activity_minutes_per_week"] < 150).astype(int)
    
    df["Hypertension"] = (
    (df["systolic_bp"] >= 130) |
    (df["diastolic_bp"] >= 80)
    ).astype(int)
    
    df["BP_Load"] = df["systolic_bp"] * df["diastolic_bp"]

    df["TG_HDL_Ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1)
    
    df["Lipid_Risk"] = (
        (df["triglycerides"] > 150) &
        (df["hdl_cholesterol"] < 40)
    ).astype(int)

    df["High_Heart_Rate"] = (df["heart_rate"] > 85).astype(int)
    df["Low_SES"] = (
    (df["education_level"].isin(["Low"])) |
    (df["income_level"].isin(["Low"]))
    ).astype(int)
    df["Genetic_Lifestyle_Risk"] = (
    df["family_history_diabetes"] *
    (df["bmi"] > 30).astype(int)
    )
    df["Comorbidity_Count"] = (
    df["hypertension_history"] +
    df["cardiovascular_history"] +
    df["family_history_diabetes"]
    )

    df["Silent_Diabetic"] = (
    (df["bmi"] < 25) &
    (df["physical_activity_minutes_per_week"] > 150) &
    (df["triglycerides"] > 150)
    ).astype(int)

    return df


train_featured = feature_enginnering(train)
test_featured = feature_enginnering(test)


# Identify target column (modify this!)
target_col = 'diagnosed_diabetes'  # CHANGE THIS to your target column name

# Separate features and target
X = train_featured.drop(columns=[target_col])
y = train_featured[target_col]
X_test = test_featured.copy()

# Find categorical columns automatically
categorical_cols = [col for col in X.columns 
                    if X[col].dtype == 'object' or X[col].nunique() < 20]

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")


for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


 
import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=1000,  # Large number for early stopping
    learning_rate=0.01,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


kf = KFold(n_splits = 2, shuffle = True, random_state = 42)

scores = cross_val_score(
    model, X_train, y_train,
    cv = kf,
    scoring = 'roc_auc',
    n_jobs = -1
)
print(scores.mean())


model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=100)],  
    
)


val_proba = model.predict_proba(X_val)[:, 1]  


auc_score = roc_auc_score(y_val, val_proba)
print(f"\n Model AUC Score: {auc_score:.6f}")

 
print("\n Making predictions on test set...")
test_proba = model.predict_proba(X_test)[:, 1] 


 
submission = pd.DataFrame({
    'id': test['id'],   
    'diagnosed_diabetes': test_proba  
})

submission.to_csv('submission.csv', index=False)

print(f"Submission saved: 'submission.csv'")
print(f"File shape: {submission.shape}")
print(f"Prediction range: {test_proba.min():.4f} to {test_proba.max():.4f}")
print("\n First 5 predictions:")
print(submission.head())




