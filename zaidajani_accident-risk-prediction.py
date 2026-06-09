import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestRegressor


from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_df.head()


train_df.groupby(['weather'], as_index=False)['accident_risk'].mean()


train_df.groupby(['speed_limit'], as_index=False)['num_reported_accidents'].mean()


# --- Risk Maps ---
road_risk = {
    "highway": 0.6,
    "rural": 0.4,
    "urban": 0.2
}

lighting_risk = {
    "daylight": 0.1,
    "dim": 0.4,
    "night": 0.8
}

weather_risk = {
    "clear": 0.1,
    "foggy": 0.5,
    "rainy": 0.8
}

# --- Risk Functions ---

def curvature_risk(curv):
    if curv < 5: return 0.1
    elif curv < 15: return 0.4
    elif curv < 30: return 0.7
    else: return 1.0

def speed_risk(speed):
    if speed < 40: return 0.2
    elif speed < 80: return 0.5
    else: return 0.9

# --- Slip Calculation Function ---

def calculate_slip(row):
    r_road = road_risk.get(row['road_type'], 0.5)
    r_curve = curvature_risk(row['curvature'])
    r_speed = speed_risk(row['speed_limit'])
    r_light = lighting_risk.get(row['lighting'], 0.4)
    r_weather = weather_risk.get(row['weather'], 0.4)

    score = (
        r_road * 0.25 +
        r_curve * 0.20 +
        r_speed * 0.25 +
        r_light * 0.10 +
        r_weather * 0.20
    )

    return round(score, 3)

# --- Apply to DataFrame ---
train_df['slipping_chances'] = train_df.apply(calculate_slip, axis=1)

train_df.head()


train_df.groupby(['slipping_chances'], as_index=False)['accident_risk'].mean().sort_values(by="accident_risk")


test_df['slipping_chances'] = test_df.apply(calculate_slip, axis=1)

test_df.head()


train_df.head(10)


# Weather, Road Sign, Public Road, Time_of_day, School Season, Num_reported_accidents

# Computing visibility factor

# Clear + True -> 0.31 + 0.2 = 0.55 (0.31 + 0.3524 = 0.6624)
# Clear + False -> 0.31 + 0.15 = 0.46 (0.31 + 0.3522 = 0.6622)

# Foggy + True -> 0.38 + 0.2 = 0.58 
# Foggy + False -> 0.38 + 0.15 = 0.43 

# Rainy + True -> 0.36 + 0.2 = 0.56 
# Rainy + False -> 0.36 + 0.15 = 0.44 

train_df.groupby(['weather'], as_index=False)['accident_risk'].mean()


train_df.groupby(['public_road'], as_index=False)['accident_risk'].mean()


def visibility_factor(weather, road_signs_present):
    """
    Acts as a multiplier: poor visibility amplifies existing risk.
    road_signs_present is assumed boolean: True = better navigation.
    """
    
    if weather == "clear":
        return 1.0 if road_signs_present else 1.1  # small effect

    elif weather == "foggy":
        return 1.2 if road_signs_present else 1.4  # big impact

    elif weather == "rainy":
        return 1.15 if road_signs_present else 1.3

    return 1.0  # fallback


train_df["visibility_factor"] = train_df.apply(
    lambda row: visibility_factor(row["weather"], row["road_signs_present"]),
    axis=1
)

test_df["visibility_factor"] = test_df.apply(
    lambda row: visibility_factor(row["weather"], row["road_signs_present"]),
    axis=1
)



# ------------------------------
# 3️⃣ Final Combined Accident Risk
# ------------------------------

# ------------------------------
# Final Combined Accident Risk for TRAIN
# ------------------------------
train_df["final_risk"] = (
    train_df["slipping_chances"] * train_df["visibility_factor"]
).round(3)

train_df["final_risk"] = train_df["final_risk"].clip(0, 1)


# ------------------------------
# Final Combined Accident Risk for TEST
# ------------------------------
test_df["final_risk"] = (
    test_df["slipping_chances"] * test_df["visibility_factor"]
).round(3)

test_df["final_risk"] = test_df["final_risk"].clip(0, 1)



train_df.head()


train_df.groupby(['visibility_factor'], as_index=False)['accident_risk'].mean().sort_values(by="accident_risk")


train_df.groupby(['visibility_factor'], as_index=False)['accident_risk'].mean().sort_values(by="accident_risk")


train_df.head(10)


train_df.groupby(['num_lanes'], as_index=False)['accident_risk'].mean().sort_values(by="accident_risk")


ohe = OneHotEncoder(sparse_output=False)
ode = OrdinalEncoder
SI = SimpleImputer(strategy='most_frequent')


ode_cols = ['road_type', 'weather', 'lighting']
ohe_cols = ['time_of_day']


correlation_matrix = train_df.corr(numeric_only=True)

# Create a heatmap using Seaborn
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")


X = train_df.drop([
    'accident_risk',   
    'final_risk',
    'id',           
    'num_lanes',
    'public_road',
    'holiday',
    'school_season',
    'num_reported_accidents'
], axis=1)

y = train_df['accident_risk']

X_test = test_df.drop([
    'final_risk',
    'id',
    'num_lanes',
    'public_road',
    'holiday',
    'school_season',
    'num_reported_accidents'
], axis=1)


numeric_cols = ['speed_limit','curvature','slipping_chances','visibility_factor']
categorical_cols = ['road_type','weather','lighting','time_of_day']

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

col_trans = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, categorical_cols)
], remainder='passthrough')

model = RandomForestRegressor(random_state=21)
pipefinalrfr = Pipeline([('prep', col_trans), ('model', model)])



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=21)


pipefinalrfr.fit(X_train, y_train)


def predict_one(sample_row):
    return round(pipefinalrfr.predict(sample_row)[0], 3)



sample = X_valid.sample(10, random_state=21)
preds = [predict_one(pd.DataFrame([row])) for idx, row in sample.iterrows()]
truth = y_valid.loc[sample.index].values

comparison = pd.DataFrame({
    "Predicted Risk": preds,
    "Actual Risk": truth
}).reset_index(drop=True)

comparison



from sklearn.metrics import mean_squared_log_error
y_pred = pipefinalrfr.predict(X_valid)

rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))

print(f"RMSLE: {rmsle:.5f}")


plt.figure(figsize=(7,7))
plt.scatter(y_valid, y_pred, alpha=0.3)
plt.xlabel("Actual accident_risk")
plt.ylabel("Predicted accident_risk")
plt.title("Predicted vs Actual Risk")
plt.grid(True)
plt.show()


test_predictions = pipefinalrfr.predict(X_test)

test_predictions = np.round(test_predictions, 4) 

submission = pd.DataFrame({
    "id": test_df["id"], 
    "accident_risk": test_predictions
})


submission.to_csv("submission.csv", index=False)
submission.head()




