import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_data.head() #First 5 data


train_data.tail() #Last 5 data


train_data.info() #Basic Info


train_data.describe()


train_data.drop_duplicates(inplace = True) #Drop and duplicates


sex_map = {
    "male" : 0,
    "female" : 1
}

train_data["Sex"] = train_data["Sex"].map(sex_map)
train_data.head()


sex_map = {
    "male" : 0,
    "female" : 1
}

test_data["Sex"] = test_data["Sex"].map(sex_map)
test_data.head()


def calculate_calories(age, weight, heart_rate, duration, sex):
    if sex == 0:
        calorie = ((age * 0.2017) - (weight * 0.09036) + (heart_rate * 0.6309) - 55.0969) * duration / 4.184

    else:
        calorie = ((age * 0.074) - (weight * 0.05741) + (heart_rate * 0.4472) - 20.4022) * duration / 4.18

    return calorie


calculate_calories(36, 82.0, 101.0, 26.0, 0)


#Average calories per gender

avg_cal_gen = train_data.groupby("Sex")["Calories"].mean()
avg_cal_gen.plot(kind = "bar", color = ["r", "g"], label = "Average Calorie Burnt Each Gender")
plt.title("Average Calories Burnt By Each Gender")
plt.xlabel("Gender")
plt.ylabel("Average")
plt.xticks(ticks=[0, 1], labels=["Female", "Male"], rotation=0)
plt.show()


#Number of Male and Female in the dataset
count_sex = train_data["Sex"].value_counts()
total = len(train_data["Sex"])
print(f"Total Number of Female in dataset: {count_sex[1]}, percentage: {count_sex[1] / total * 100:.2f}%")
print(f"Total Number of Male in dataset: {count_sex[0]}, percentage: {count_sex[0] / total * 100:.2f}%")


fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 columns

sns.boxplot(x=train_data["Duration"], ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title("Duration")

sns.boxplot(x=train_data["Heart_Rate"], ax=axes[0, 1], color="salmon")
axes[0, 1].set_title("Heart Rate")

sns.boxplot(x=train_data["Body_Temp"], ax=axes[1, 0], color="lightgreen")
axes[1, 0].set_title("Body Temperature")

sns.boxplot(x=train_data["Calories"], ax=axes[1, 1], color="plum")
axes[1, 1].set_title("Calories")

plt.tight_layout()
plt.show()


train_data = train_data.replace([float('inf'), float('-inf')], float('nan'))

fig, axs = plt.subplots(3, 3, figsize=(15, 15))

columns = [
    "Age","Height", 
    "Weight", "Duration", "Heart_Rate", 
    "Body_Temp", "Calories"
]

for i, column in enumerate(columns):
    ax = axs[i // 3, i % 3]
    sns.histplot(train_data[column], kde=True, color='skyblue', bins=50, ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

fig.delaxes(axs[2, 2])

plt.tight_layout()
plt.show()



#Scatter plot between Duration and Calorie

plt.scatter(train_data["Duration"] , train_data['Calories'], color = "b", label = "Duration / Calorie Scatter Plot")
plt.xlabel("Duration")
plt.ylabel("Calorie")
plt.title("Duration / Calorie Scatter Plot")
plt.show()


#Scatter plot between Heart_Rate and Calorie

plt.scatter(train_data["Heart_Rate"] , train_data['Calories'], color = "r", label = "Duration / Calorie Scatter Plot")
plt.xlabel("Heart_Rate")
plt.ylabel("Calorie")
plt.title("Heart_Rate / Calorie Scatter Plot")
plt.show()


#Scatter plot between Body_Temp and Calorie

plt.scatter(train_data["Body_Temp"] , train_data['Calories'], color = "g", label = "Body_Temp / Calorie Scatter Plot")
plt.xlabel("Body_Temp")
plt.ylabel("Calorie")
plt.title("Body_Temp / Calorie Scatter Plot")
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_data['Duration'], train_data['Heart_Rate'], train_data['Calories'], c='skyblue', marker='o')
ax.set_xlabel('Duration (min)')
ax.set_ylabel('Heart Rate (bpm)')
ax.set_zlabel('Calories')
ax.set_title('3D Scatter Plot: Duration, Heart Rate, and Calories')
plt.tight_layout()
plt.show()



spearman_corr = train_data.corr(method='spearman')

plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = add_constant(train_data[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']])

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Show the VIF
print(vif_data)


df1 = train_data.copy()
df1["Category"] = df1["Age"].map(lambda x : 0 if (x >= 20 and x<40) else ( 1 if (x >=40 and x < 60) else 2))
df1.head()


avg_cal_burned_age_cat = df1.groupby("Category")["Calories"].mean()
avg_cal_burned_age_cat.plot(kind = "bar", cmap = "coolwarm")
plt.xticks(
    ticks=range(len(avg_cal_burned_age_cat.index)), 
    labels=["Adult (20-40)", "Adult (40-60)", "Old (60+)"],
    rotation = 0
)
plt.xlabel("Age Category")
plt.ylabel("Average Calorie burnt")
plt.title("Average Calories Burnt by Different Age Category")
plt.show()


avg_cal_burned_age_cat = df1.groupby("Category")["Duration"].mean()
avg_cal_burned_age_cat.plot(kind = "bar", cmap = "coolwarm")
plt.xticks(
    ticks=range(len(avg_cal_burned_age_cat.index)), 
    labels=["Teenage (20-40)", "Adult (40-60)", "Old (60+)"],
    rotation = 0
)
plt.xlabel("Age Category")
plt.ylabel("Average Duration")
plt.title("Average Duration by Different Age Category")
plt.show()


def calculate_bmi(df):
    df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
    return df


df2 = train_data.copy()
df2 = calculate_bmi(df2)
df2.head()


df2.describe()["BMI"]


#Categorizing BMI into underweight, overweight , fit, obese1, obese2, obese3

bmi_bins = [0, 18.5, 25, 30, 35, 40, float("inf")]
labels = ["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Obese III"]

df2["BMI_Category"] = pd.cut(df2["BMI"], bins = bmi_bins , labels = labels, right = True)

df2.head()


import matplotlib.pyplot as plt

count_bmi_cat = df2["BMI_Category"].value_counts().sort_index()

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

count_bmi_cat.plot(kind="bar", color="lightgreen", edgecolor='black', ax=axs[0])
axs[0].set_title("BMI Category (Normal Scale)")
axs[0].set_xlabel("BMI Category")
axs[0].set_ylabel("Count")
axs[0].tick_params(axis='x', rotation=45)

count_bmi_cat.plot(kind="bar", color="lightgreen", edgecolor='black', ax=axs[1], log=True)
axs[1].set_title("BMI Category (Log Scale)")
axs[1].set_xlabel("BMI Category")
axs[1].set_ylabel("Log(Count)")
axs[1].tick_params(axis='x', rotation=45)

plt.suptitle("Total Number of People in Different BMI Categories")
plt.tight_layout()
plt.show()



#Average calorie burnt by different categoy

avg_cal_bur_each_bmi_cat = df2.groupby("BMI_Category", observed = False)["Calories"].mean().sort_values()
avg_cal_bur_each_bmi_cat.plot(kind = "bar", color = "purple")
plt.xlabel("BMI Categoty")
plt.ylabel("Average Calorie Burnt")
plt.xticks(rotation = 45)
plt.title("Average Calorie burnt by each category")
plt.show()


## Correlation in one - hot encode and then correlation heat map

df2 = pd.get_dummies(data = df2, columns = ["BMI_Category"], prefix = "OHE", prefix_sep = "_" , drop_first = False )

corr2 = df2[["OHE_Underweight", "OHE_Normal", "OHE_Overweight", "OHE_Obese I", "OHE_Obese II", "OHE_Obese III", "Calories"]].corr()
corr2


## Calulatin calorie burnt in duration 

def calorie_duration(data):
    calorie = data["Calories"]
    duration = data["Duration"]
    cal_dur = calorie / duration 
    data["cal/dur"] = cal_dur
    data["cal/dur"] = data["cal/dur"].replace(0, np.nan)

    return data


df3 = train_data.copy()

df3 = calorie_duration(df3)
df3.head()


plt.figure(figsize=(8, 5))
sns.histplot(df3['cal/dur'], kde=True, color='skyblue', bins=50)
plt.title("Distribution of Calories Burned per Minute")
plt.xlabel("Calories per Minute")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 5))
sns.boxplot(x='Sex', y='cal/dur', data=df3, palette='Set2')
plt.title("Calories per Minute by Sex")
plt.xlabel("Sex")
plt.ylabel("Calories per Minute")
plt.tight_layout()
plt.show()



#Categorizing BMI into underweight, overweight , fit, obese1, obese2, obese3

df3 = calculate_bmi(df3)
df3["BMI_Category"] = pd.cut(df3["BMI"], bins = bmi_bins , labels = labels, right = True)

plt.figure(figsize=(8, 5))
sns.boxplot(x = "BMI_Category", y = 'cal/dur', data = df3, palette='Set3')
plt.title("Calories per Minute by BMI Category")
plt.xlabel("BMI Category")
plt.ylabel("Calories per Minute")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df3[["Calories", "cal/dur"]].corr()


def calculate_rmsle(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    y_pred = np.maximum(y_pred, 0)
    
    log_y = np.log1p(y)
    log_y_pred = np.log1p(y_pred)
    
    squared_log_error = (log_y - log_y_pred) ** 2
    
    rmsle = np.sqrt(np.mean(squared_log_error))
    return rmsle


def kaggle_submission(Y):
    submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
    submission["Calories"] = Y
    submission.to_csv("submission.csv", index=False)
    print(f"✅ Submission file saved")
    return submission


# Load data
train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Process data
sex_map = {"male": 0, "female": 1}
train_data["Sex"] = train_data["Sex"].map(sex_map)
test_data["Sex"] = test_data["Sex"].map(sex_map)

# Calculate BMI
train_data = calculate_bmi(train_data)
test_data = calculate_bmi(test_data)


train_data.head()


test_data.head()


features = ['Age', 'Sex', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp']
X_train_full = train_data[features]
y_train_full = train_data['Calories']
X_test_submission = test_data[features]

print("Train data shape:", X_train_full.shape)
print("Test data shape:", X_test_submission.shape)
print("Any NaN in train:", X_train_full.isna().sum().sum())
print("Any NaN in test:", X_test_submission.isna().sum().sum())


from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_train_scaled = X_train_full.copy()
X_test_scaled = X_test_submission.copy()


features_to_scale = [col for col in features if col != 'Sex']
X_train_scaled[features_to_scale] = scaler.fit_transform(X_train_full[features_to_scale])
X_test_scaled[features_to_scale] = scaler.transform(X_test_submission[features_to_scale])


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train_full, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


y_val_pred = lr_model.predict(X_val)
y_val_pred = np.maximum(y_val_pred, 0)  # Enforce non-negative predictions
rmsle_val = calculate_rmsle(y_val, y_val_pred)
print(f"Validation RMSLE: {rmsle_val:.4f}")


lr_model.fit(X_train_scaled, y_train_full)


y_pred = lr_model.predict(X_test_scaled)


y_pred = np.maximum(y_pred, 0)

print("Prediction stats:")
print(f"Min: {y_pred.min():.2f}")
print(f"Max: {y_pred.max():.2f}")
print(f"Mean: {y_pred.mean():.2f}")
print(f"First few predictions: {y_pred[:5]}")

# Create submission
#submission = kaggle_submission(y_pred)


features = ['Age', 'Sex', 'BMI', 'Duration', 'Heart_Rate', 'Body_Temp']
X_train_full = train_data[features]
y_train_full = train_data['Calories']

print(X_train_full.shape)
print(y_train_full.shape)


X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full,
                                                   test_size = 0.2, random_state = 42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


from sklearn.ensemble import RandomForestRegressor

def rfr_model():
    rfr_model = RandomForestRegressor(max_depth = 20, min_samples_leaf = 2,
                                     min_samples_split=10, n_estimators=200)
    
    rfr_model.fit(X_train_full, y_train_full)

    return rfr_model


import lightgbm as lgb

def lgb_model():
    lgb_model = lgb.LGBMRegressor(boosting_type='dart', colsample_bytree=0.8, learning_rate=0.2,
                  max_depth=30, min_child_samples=100, n_estimators=500,
                  num_leaves=255, random_state=42, reg_alpha=1.0, reg_lambda=1.0,
                  subsample=0.8)

    lgb_model.fit(X_train_full, y_train_full)


import xgboost as xgb

def xgb_model():
    xgb_model = xgb.XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=1.0, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=0.5, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.07, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=10, max_leaves=None,
             min_child_weight=1, missing=np.nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=800, n_jobs=-1,
             num_parallel_tree=None, random_state=42)
    xgb_model.fit(X_train_full, y_train_full)

    return xgb_model


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


from sklearn.preprocessing import StandardScaler

X_train_sc = X_train.copy()
y_train_sc = y_train.copy()
X_test_sc = X_test.copy()
y_test_sc = y_test.copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sc)
X_test_scaled = scaler.transform(X_test_sc)

X_train_scaled.shape
X_test_scaled.shape


def rmsle_keras(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 0.0)
    y_true = tf.maximum(y_true, 0.0)

    log_true = tf.math.log1p(y_true)
    log_pred = tf.math.log1p(y_pred)

    squared_log_error = tf.square(log_true - log_pred)
    return tf.sqrt(tf.reduce_mean(squared_log_error))


from tensorflow.keras.optimizers import Adam

def neural_network():
    nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_dim=6),
    layers.BatchNormalization(),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(32, activation='relu'),
    layers.Dense(1)
    ])

    nn_model.compile(optimizer= Adam(learning_rate=0.001), loss='mse', metrics=['mae', rmsle_keras])
    nn_model.fit(X_train_scaled, y_train_sc, validation_split=0.2, epochs=30, batch_size=64)

    return nn_model


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def ada_model():
    base_estimator = DecisionTreeRegressor(max_depth=4)
    
    ada_reg = AdaBoostRegressor(estimator=base_estimator,
                                n_estimators=100,
                                learning_rate=0.8,
                                random_state=42)
    
    ada_reg.fit(X_train_full, y_train_full)

    return ada_reg


from sklearn.ensemble import StackingRegressor

def sts_model():
    model_lr = ("lr", LinearRegression())
    model_rfr = ("rfr", RandomForestRegressor(max_depth = 20, min_samples_leaf = 2,
                                     min_samples_split=10, n_estimators=200)) 
    model_lgb = ("lgb", lgb.LGBMRegressor(boosting_type='dart', colsample_bytree=0.8, learning_rate=0.2,
                  max_depth=30, min_child_samples=100, n_estimators=500,
                  num_leaves=255, random_state=42, reg_alpha=1.0, reg_lambda=1.0,
                  subsample=0.8))
    model_xgb = ("xgb", xgb.XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=1.0, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=0.5, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.07, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=10, max_leaves=None,
             min_child_weight=1, missing=np.nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=800, n_jobs=-1,
             num_parallel_tree=None, random_state=42))
    model_ada = ("ada", AdaBoostRegressor(
                                n_estimators=100,
                                learning_rate=0.8,
                                random_state=42))

    stacked_model = StackingRegressor(
        estimators=[model_lr, model_rfr, model_ada, model_xgb, model_lgb],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1,
        passthrough=True 
    )

    stacked_model.fit(X_train, y_train)

    return stacked_model


#model = sts_model()
#y_pred_train = model.predict(X_train_scaled)
#rmsle_train = calculate_rmsle(y_train, y_pred_train)
#print(f"Validation RMSLE: {rmsle_train:.4f}")


#y_pred_test = model.predict(X_test_scaled)
#rmsle_test = calculate_rmsle(y_test, y_pred_test)
#print(f"Validation RMSLE: {rmsle_test:.4f}")


#X_test_submission = test_data[features]
#y_pred = model.predict(X_test_submission)
#submission = kaggle_submission(y_pred)
#submission.head()


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error

from sklearn.model_selection import train_test_split

X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=75000, random_state=42)


#def rmsle(y_true, y_pred):
#    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(calculate_rmsle, greater_is_better=False)

param_distributions = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'base_estimator__max_depth': [1, 3, 5, 7],
    'base_estimator__min_samples_split': [2, 5, 10],
    'base_estimator__max_features': ['auto', 'sqrt', 'log2']
}


def hypertune():
    #Randomized Search
    rfr_hypermodel = RandomizedSearchCV(
        estimator= model,
        param_distributions=param_distributions,
        n_iter=50,  
        scoring=rmsle_scorer,
        n_jobs=-1,
        cv=3,
        random_state=42,
        verbose=1
    )
    rfr_hypermodel.fit(X_sample, y_sample)

    return rfr_hypermodel


#model = ada_model()
#rfr_hypermodel = hypertune()


#rfr_hypermodel.best_estimator_


#rfr_hypermodel.best_params_


from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error , mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train = train.drop_duplicates(subset = train.columns).reset_index(drop = True)
train["Sex"] = train["Sex"].map({"male" : 1, "female" : 0})
test["Sex"] = test["Sex"].map({"male" : 1, "female" : 0})


def add_cross_terms(df, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            df[f"{features[i]}_x_{features[j]}"] = df[features[i]] * df[features[j]]

    return df

numerical = ["Age", "Height" ,"Weight" ,"Duration" ,"Heart_Rate" ,"Body_Temp"]
train = add_cross_terms(train, numerical)
test = add_cross_terms(test, numerical)


train.head()


def predict_model_1(train):
    df = train.groupby(["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])["Calories"].min().reset_index()
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    test_copy = test.copy()
    test_copy["Sex"] = le.transform(test_copy["Sex"])

    df = add_cross_terms(df, numerical)
    test_copy = add_cross_terms(test_copy, numerical)

    df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
    df["Intensity"] = df["Heart_Rate"] / df["Duration"]
    test_copy["BMI"] = test_copy["Weight"] / (test_copy["Height"] / 100) ** 2
    test_copy["Intensity"] = test_copy["Heart_Rate"] / test_copy["Duration"]

    X = df.drop(columns = ["Calories"])
    y = np.log1p(df["Calories"])
    X_test = test_copy[X.columns]

    bins = KBinsDiscretizer(n_bins = 10, encode = "ordinal", strategy = 'quantile')
    duration_bins = bins.fit_transform(df[["Duration"]]).astype(int).flatten()
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    cat_preds = np.zeros(len(X_test))
    xgb_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_bins)):
        X_train, X_val = X.iloc[train_idx] , X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx] , y.iloc[val_idx]

        cat = CatBoostRegressor(verbose = 0, random_state = 42)
        cat.fit(X_train, y_train, eval_set = (X_val, y_val), early_stopping_rounds = 50)
        oof_preds[val_idx] += cat.predict(X_val) * 0.5
        cat_preds += cat.predict(X_test) / skf.n_splits

        xgb = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=10,
                           subsample=0.9, colsample_bytree=0.7, gamma=0.01,
                           max_delta_step=2, tree_method="hist", enable_categorical=True,
                           early_stopping_rounds=100, eval_metric="rmse", verbosity=0,
                           random_state=42)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        oof_preds[val_idx] += xgb.predict(X_val) * 0.5
        xgb_preds += xgb.predict(X_test) / skf.n_splits

    final_preds = 0.5 * cat_preds + 0.5 * xgb_preds
    return np.clip(np.expm1(final_preds), 1, 314)


pred1 = predict_model_1(train.copy())
submission["Calories"] = pred1
submission.to_csv("submission.csv", index = False)
print("submission.csv saved.")




