# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from lightgbm import LGBMClassifier
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import shap 


import warnings
warnings.filterwarnings('ignore')


path = '/kaggle/input/playground-series-s5e12/'
df = pd.read_csv(f"{path}train.csv")


df.head(5)


df.info()


categorical = [col for col in df.columns.tolist() if df[col].dtype == 'object' ]
numeric = [col for col in df.columns.tolist() if df[col].dtype != 'object']


numeric.remove('id')
numeric.remove('diagnosed_diabetes')


numeric_hist = ['age',  'alcohol_consumption_per_week',  'physical_activity_minutes_per_week',
                 'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi',
                 'waist_to_hip_ratio',  'systolic_bp', 'diastolic_bp', 'heart_rate', 
                'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']

historic_numeric_vars = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']


for col in numeric_hist:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col].dropna(), bins=50, color='steelblue')
    plt.title(f"Histogram of {col}")
    plt.show()



for col in categorical:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col].dropna(), bins=50, color='steelblue')
    plt.title(f"Histogram of {col}")
    plt.show()



df["bmi_waist"] = df["bmi"] * df["waist_to_hip_ratio"]
df["age_bmi"] = df["age"] * df["bmi"]
df["activity_bmi"] = df["physical_activity_minutes_per_week"] / (df["bmi"] + 1)

df["log_triglycerides"] = np.log1p(df["triglycerides"])
df["log_screen_time"] = np.log1p(df["screen_time_hours_per_day"])

df["obese"] = (df["bmi"] >= 30).astype(int)
df["overweight"] = ((df["bmi"] >= 25) & (df["bmi"] < 30)).astype(int)

df["hypertensive"] = ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(int)

df["low_hdl"] = (df["hdl_cholesterol"] < 40).astype(int)
df["high_triglycerides"] = (df["triglycerides"] > 150).astype(int)



df.columns


matrix = df[numeric].corr()

plt.figure(figsize=(14,12))
sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


df.info()


df_waist_to_hip_ratio_bmi = df[['waist_to_hip_ratio', 'bmi']]


X_scaled = StandardScaler().fit_transform(df_waist_to_hip_ratio_bmi)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

df['PCA_waist_to_hip_ratio_X_bmi'] = X_pca

#Explained variance
print(f"PCA explains {round(pca.explained_variance_ratio_[0], 4) * 100}% of the variance.") # 



# Principal axes (directions)
print(pca.components_)


df_ldl_cholesterol_to_cholesterol_total = df[['ldl_cholesterol', 'cholesterol_total']]


X_scaled_2 = StandardScaler().fit_transform(df_ldl_cholesterol_to_cholesterol_total)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

df['PCA_ldl_cholesterol_X_cholesterol_total'] = X_pca

#Explained variance
print(f"PCA explains {round(pca.explained_variance_ratio_[0], 4) * 100}% of the variance.") # 


df_pca = df.copy()
df_pca = df_pca.drop(columns=['ldl_cholesterol', 'cholesterol_total', 'waist_to_hip_ratio', 'bmi'])


numeric = ['age',
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
 'family_history_diabetes',
 'hypertension_history',
 'cardiovascular_history', 
# 'PCA_waist_to_hip_ratio_X_bmi',
#  'PCA_ldl_cholesterol_X_cholesterol_total'
               'bmi_waist', 'age_bmi',
       'activity_bmi', 'log_triglycerides', 'log_screen_time', 'obese',
       'overweight', 'hypertensive', 'low_hdl', 'high_triglycerides'
              ]


# selected_features = ['family_history_diabetes',
#                         'physical_activity_minutes_per_week',
#                         'age',
#                         'triglycerides',
#                         'bmi',
#                         'ldl_cholesterol',
#                         'diet_score',
#                         'hdl_cholesterol',
#                         'heart_rate',
#                         'screen_time_hours_per_day',
#                         'waist_to_hip_ratio',
#                         'systolic_bp',
#                         'cholesterol_total']

# df_selected_feat = df[selected_features + ['diagnosed_diabetes']]


categorical


X = df[numeric + categorical]
X = df.drop(columns = ['diagnosed_diabetes', 'id'])
y = df['diagnosed_diabetes']

X[categorical] = X[categorical].astype('category')




# model = LGBMClassifier(
#     objective='binary',
#     metric='auc',
#     boosting_type='gbdt',
#     num_leaves=31,
#     learning_rate=0.05,
#     feature_fraction=0.9, 
#     verbosity = -1
# )


# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# scores = cross_val_score(
#     model, 
#     X, y, 
#     cv=kf,
#     scoring="roc_auc",
#     n_jobs=-1
# )

# print("ROC-AUC scores:", scores)
# print("Mean ROC-AUC:", scores.mean())
# # print("Best iteration:", model.best_iteration_)



y.value_counts()/len(y)

pos = (y == 1).sum()
neg = (y == 0).sum()


# Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

params = {       'objective': 'binary',
                'metric':'auc',
                'boosting_type': 'gbdt',
                'num_leaves':31,
                'learning_rate':0.05,
                'feature_fraction':0.9,
                 'verbose': -1}

# Model
model = LGBMClassifier(
                        **params, 
                        is_unbalance = True
                        )

# Train with early stopping
model.fit(
    X_train, y_train,
    # eval_set=[(X_val, y_val)],
    # eval_metric="auc",
    #early_stopping_rounds=10,
   # 'verbose'= -1
)




y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)


roc_auc_train = roc_auc_score(y_train, y_train_pred)
roc_auc_val = roc_auc_score(y_test, y_pred)

print(f"Train AUC score {roc_auc_train}")
print(f"Validation AUC score {roc_auc_val}")


print(classification_report(y_test, y_pred))



cf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(cf_matrix, fmt=" ", cmap='Blues');



# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Labels for each cell
group_names = ["TN", "FP", "FN", "TP"]

# Create labels with counts
labels = [
    f"{group}\n{value}"
    for group, value in zip(group_names, cm.flatten())
]

labels = np.array(labels).reshape(2, 2)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=labels,
    fmt="",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    cbar=False
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()



selected_features = ['family_history_diabetes',
                        'physical_activity_minutes_per_week',
                        'age',
                        'triglycerides',
                        'bmi',
                        'ldl_cholesterol',
                        'diet_score',
                        'hdl_cholesterol',
                        'heart_rate',
                        'screen_time_hours_per_day',
                        'waist_to_hip_ratio',
                        'systolic_bp',
                        'cholesterol_total']




# import shap #shape values measures of contributions each predictor (feature) has in a machine learning model 
lista_feat = numeric + categorical

v = X_train.sample(2000)
clf = model
# Shap variables Train
explainer = shap.TreeExplainer(clf)
v = X_train.sample(2000)[lista_feat]
shap_values_train = explainer.shap_values(v)
shap.summary_plot(shap_values_train,v, plot_type='bar', max_display = 30)




# cose da fare: 
# - Classification Report
# - Shap Values
# - Feature engineering
# - PCA tra waist_to_hip_ratio e bmi
# - PCA tra ldl_cholesterol e cholesterol_total







