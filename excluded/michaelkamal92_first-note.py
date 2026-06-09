# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import xgboost as xgb 
from sklearn.metrics import accuracy_score, classification_report
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


train.info()


train['diagnosed_diabetes'].value_counts()


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


#['gender','ethnicity','education_level','income_level','smoking_status','employment_status']

def extract(df):
    df['multi_all_obj1'] = df['gender'] * df['ethnicity'] * df['education_level'] * df['income_level'] * df['smoking_status'] * df['employment_status']
    df['multi_all_obj2'] = df['gender'] * df['ethnicity'] * df['education_level'] * df['income_level'] * df['smoking_status'] 
    df['multi_all_obj3'] = df['gender'] * df['ethnicity'] * df['education_level'] * df['income_level'] 
    df['multi_all_obj4'] = df['gender'] * df['ethnicity'] * df['education_level']
    df['multi_all_obj5'] = df['gender'] * df['ethnicity'] 

    
    return df


import pandas as pd
import numpy as np

def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ============================
    # 1) Clinical engineered features
    # ============================
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + df["pulse_pressure"] / 3

    df["cholesterol_hdl_ratio"] = df["cholesterol_total"] / df["hdl_cholesterol"]
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / df["hdl_cholesterol"]
    df["triglyceride_hdl_ratio"] = df["triglycerides"] / df["hdl_cholesterol"]
    df["non_hdl_cholesterol"] = df["cholesterol_total"] - df["hdl_cholesterol"]

    df["obesity_flag"] = (df["bmi"] >= 30).astype(int)
    df["bmi_age_ratio"] = df["bmi"] / df["age"]
    df["waist_bmi_product"] = df["waist_to_hip_ratio"] * df["bmi"]

    # ============================
    # 2) Lifestyle features
    # ============================
    df["alcohol_per_day"] = df["alcohol_consumption_per_week"] / 7
    df["physical_activity_hours"] = df["physical_activity_minutes_per_week"] / 60
    df["sedentary_ratio"] = df["screen_time_hours_per_day"] / 24
    df["sleep_deficit"] = (8 - df["sleep_hours_per_day"]).clip(lower=0)

    df["unhealthy_lifestyle_index"] = (
        (df["alcohol_consumption_per_week"] > 14).astype(int)
        + (df["screen_time_hours_per_day"] > 6).astype(int)
        + (df["sleep_hours_per_day"] < 6).astype(int)
        + (df["physical_activity_minutes_per_week"] < 60).astype(int)
    )

    # ============================
    # 3) Interaction features
    # ============================
    df["age_bmi_interaction"] = df["age"] * df["bmi"]
    df["bp_interaction"] = df["systolic_bp"] * df["diastolic_bp"]
    df["activity_diet_interaction"] = df["physical_activity_minutes_per_week"] * df["diet_score"]

    # ============================
    # 4) Medical history combined
    # ============================
    df["combined_history"] = (
        df["family_history_diabetes"]
        + df["hypertension_history"]
        + df["cardiovascular_history"]
    )
    df["chronic_risk_flag"] = (df["combined_history"] >= 2).astype(int)

    # ============================
    # 5) One-hot encoding
    # ============================
    categorical_cols = [
        "gender", "ethnicity", "education_level",
        "income_level", "smoking_status", "employment_status"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # ============================
    # 6) Drop ID if exists
    # ============================
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df



import pandas as pd
import numpy as np

def b(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # ----------------------------------
    # 1) Clinical Basic Derived Features
    # ----------------------------------
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + df["pulse_pressure"] / 3
    df["bp_ratio"] = df["systolic_bp"] / df["diastolic_bp"]
    
    df["cholesterol_hdl_ratio"] = df["cholesterol_total"] / df["hdl_cholesterol"]
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / df["hdl_cholesterol"]
    df["triglyceride_hdl_ratio"] = df["triglycerides"] / df["hdl_cholesterol"]
    
    df["non_hdl_cholesterol"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    df["lipid_index"] = df["cholesterol_total"] + df["triglycerides"] + df["ldl_cholesterol"]

    # ----------------------------------
    # 2) Obesity & Anthropometric
    # ----------------------------------
    df["obesity_flag"] = (df["bmi"] >= 30).astype(int)
    df["overweight_flag"] = (df["bmi"] >= 25).astype(int)
    df["bmi_age_ratio"] = df["bmi"] / df["age"]
    df["waist_bmi_product"] = df["waist_to_hip_ratio"] * df["bmi"]
    
    df["central_obesity_flag"] = (df["waist_to_hip_ratio"] > 0.9).astype(int)
    df["bmi_squared"] = df["bmi"] ** 2
    df["waist_hip_inverse"] = 1 / df["waist_to_hip_ratio"]

    # ----------------------------------
    # 3) Lifestyle features
    # ----------------------------------
    df["alcohol_per_day"] = df["alcohol_consumption_per_week"] / 7
    df["physical_activity_hours"] = df["physical_activity_minutes_per_week"] / 60
    df["sedentary_ratio"] = df["screen_time_hours_per_day"] / 24
    df["sleep_deficit"] = (8 - df["sleep_hours_per_day"]).clip(lower=0)
    
    df["activity_sleep_ratio"] = df["physical_activity_minutes_per_week"] / (df["sleep_hours_per_day"] * 60)
    df["activity_screen_ratio"] = df["physical_activity_minutes_per_week"] / (df["screen_time_hours_per_day"] + 0.1)

    df["unhealthy_lifestyle_index"] = (
        (df["alcohol_consumption_per_week"] > 14).astype(int)
        + (df["screen_time_hours_per_day"] > 6).astype(int)
        + (df["sleep_hours_per_day"] < 6).astype(int)
        + (df["physical_activity_minutes_per_week"] < 60).astype(int)
    )

    # ------------------------------
    # 4) Non-Linear Transformations
    # ------------------------------
    num_cols = [
        "age","bmi","cholesterol_total","triglycerides","ldl_cholesterol",
        "hdl_cholesterol","systolic_bp","diastolic_bp","heart_rate"
    ]
    
    for col in num_cols:
        df[f"log_{col}"] = np.log1p(df[col])
        df[f"sqrt_{col}"] = np.sqrt(df[col])
        df[f"{col}_squared"] = df[col] ** 2

    # ----------------------------------
    # 5) Interaction Features (Pairs)
    # ----------------------------------
    interactions = [
        ("age", "bmi"),
        ("age", "systolic_bp"),
        ("bmi", "cholesterol_total"),
        ("systolic_bp", "triglycerides"),
        ("bmi", "waist_to_hip_ratio"),
        ("physical_activity_minutes_per_week", "diet_score"),
        ("screen_time_hours_per_day", "bmi"),
        ("age", "cholesterol_total")
    ]
    
    for a, b in interactions:
        df[f"{a}_x_{b}"] = df[a] * df[b]

    # ----------------------------------
    # 6) Medical History Combined Features
    # ----------------------------------
    df["combined_history"] = (
        df["family_history_diabetes"]
        + df["hypertension_history"]
        + df["cardiovascular_history"]
    )
    df["chronic_risk_flag"] = (df["combined_history"] >= 2).astype(int)
    df["history_diabetes_hypertension"] = (
        df["family_history_diabetes"] & df["hypertension_history"]
    ).astype(int)

    # ----------------------------------
    # 7) Metabolic Syndrome Approx Score
    # ----------------------------------
    df["metabolic_syndrome_score"] = (
        (df["bmi"] >= 30).astype(int)
        + (df["systolic_bp"] >= 130).astype(int)
        + (df["diastolic_bp"] >= 85).astype(int)
        + (df["hdl_cholesterol"] < 40).astype(int)
        + (df["triglycerides"] >= 150).astype(int)
        + (df["waist_to_hip_ratio"] > 0.9).astype(int)
    )

    # ----------------------------------
    # 8) Quantile Features (Binning)
    # ----------------------------------
    df["age_quantile"] = pd.qcut(df["age"], q=5, labels=False)
    df["bmi_quantile"] = pd.qcut(df["bmi"], q=5, labels=False)
    df["cholesterol_quantile"] = pd.qcut(df["cholesterol_total"], q=5, labels=False)

    # ----------------------------------
    # 9) One-hot encoding for categorical
    # ----------------------------------
    categorical_cols = [
        "gender", "ethnicity", "education_level",
        "income_level", "smoking_status", "employment_status"
    ]
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # ----------------------------------
    # 10) Drop ID if present
    # ----------------------------------
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    return df



for col in train.select_dtypes(include=['object']).columns:
    train[col] = train[col].astype('category')
#    train[col] = train[col].astype('category').cat.codes + 1
for col in test.select_dtypes(include=['object']).columns:
    test[col] = test[col].astype('category')
#    test[col] = test[col].astype('category').cat.codes +1


#train = build(train)
#test = build(test)





train = b(train)
test = b(test)


#X = train.drop(['diagnosed_diabetes', 'id','gender','ethnicity','education_level','income_level','smoking_status','employment_status'],axis = 1)

X = train.drop(['diagnosed_diabetes'],axis = 1)
y = train['diagnosed_diabetes']


class_0 = X[y == 0]
class_1 = X[y == 1]

# حساب الفرق في الحجم بين الفئات
num_samples_class_0 = len(class_0)
num_samples_class_1 = len(class_1)

# إذا كانت الفئة 0 أصغر من الفئة 1، نقوم بتكرار عينات الفئة 0
if num_samples_class_0 < num_samples_class_1:
    class_0_resampled = class_0.sample(num_samples_class_1, replace=True, random_state=42)
    X_resampled = pd.concat([class_0_resampled, class_1])
    y_resampled = pd.concat([pd.Series([0]*len(class_0_resampled)), pd.Series([1]*len(class_1))])
else:
    class_1_resampled = class_1.sample(num_samples_class_0, replace=True, random_state=42)
    X_resampled = pd.concat([class_0, class_1_resampled])
    y_resampled = pd.concat([pd.Series([0]*len(class_0)), pd.Series([1]*len(class_1_resampled))])







model = xgb.XGBClassifier(
    n_estimators=120,
    objective= 'binary:logistic',  # الهدف: تصنيف ثنائي
    eval_metric='logloss',    
    use_label_encoder=False,   
    random_state=42,
    enable_categorical=True,
   # subsample=0.8,
   # colsample_bytree=0.8,
        #tree_method='gpu_hist',        
        #predictor='gpu_predictor', 
)
model.fit(X,y)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def train_keras_model(df: pd.DataFrame, target_col="diagnosed_diabetes"):
    # ------------------------------
    # 1) فصل الميزات عن الهدف
    # ------------------------------
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values

    # ------------------------------
    # 2) تقسيم البيانات
    # ------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------
    # 3) Scaling (مهم جدًا للشبكات العصبية)
    # ------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # ------------------------------
    # 4) بناء نموذج الشبكة العصبية
    # ------------------------------
    model = models.Sequential([
        
        layers.Input(shape=(X_train.shape[1],)),
        
        # Dense Block 1
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Dense Block 2
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Dense Block 3
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Output for binary classification
        layers.Dense(1, activation="sigmoid")
    ])

    # ------------------------------
    # 5) Compile
    # ------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # ------------------------------
    # 6) Callbacks
    # ------------------------------
    early_stop = callbacks.EarlyStopping(
        monitor="val_auc",
        patience=10,
        mode="max",
        restore_best_weights=True
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=5,
        mode="max"
    )

    # ------------------------------
    # 7) Train
    # ------------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=4096,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return model, scaler, history



# افترض أنك نفذت build_features_extended أولاً
#df_features = build_features_extended(df)

model, scaler, history = train_keras_model(train, target_col="diagnosed_diabetes")



#X_new = df_features.drop(columns=["diagnosed_diabetes"]).values
test = scaler.transform(test)
pred_probs = model.predict(test)



pred_probs


#test = test.drop(['id','gender','ethnicity','education_level','income_level','smoking_status','employment_status'],axis = 1)

#test = test.drop('id',axis=1)
#y_pred_proba = model.predict_proba(test)[:, 1]  


sub['diagnosed_diabetes'] = pred_probs
sub


sub.to_csv("submission.csv", index=False)

