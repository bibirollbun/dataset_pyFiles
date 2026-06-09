import numpy as np 
import pandas as pd 
import math

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings("ignore")



import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


train_ds = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_ds = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


print("shape of train_ds : " , train_ds.shape)
print("shape of test_ds : " , test_ds.shape)


train_ds.dtypes


num_cols = ["age", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day","screen_time_hours_per_day", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol",  "ldl_cholesterol", "triglycerides" ]

cat_cols = ["alcohol_consumption_per_week","gender","ethnicity" ,"education_level","income_level","smoking_status","employment_status","family_history_diabetes","hypertension_history","cardiovascular_history" ]



train_ds.isnull().sum() 


y = train_ds["diagnosed_diabetes"]


fig = px.bar(
    y.value_counts(),
    title="Class Balance"
)
fig.show()


print("Percentage distribution:")
print(y.value_counts(normalize=True) * 100)


n = len(num_cols)
rows = math.ceil(n / 3)  
cols = 3

plt.figure(figsize=(18, 5 * rows))

for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(train_ds[col], kde=True)
    plt.title(f"Distribution: {col}")

plt.tight_layout()
plt.show()


n = len(num_cols)
rows = math.ceil(n / 3)    # 3 boxplots per row
cols = 3

plt.figure(figsize=(18, 4 * rows))

for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x=train_ds[col])
    plt.title(f"{col}")

plt.tight_layout()
plt.show()


skew_kurt = pd.DataFrame({
    'Skewness': train_ds[num_cols].skew(),
    'Kurtosis': train_ds[num_cols].kurt()
})

skew_kurt


for col in cat_cols:
    print(f"\nValue Counts for {col}:")
    print(train_ds[col].value_counts().head(10))


n = len(cat_cols)
rows = math.ceil(n / 3)  
cols = 3

plt.figure(figsize=(18, 5 * rows))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(rows, cols, i)
    train_ds[col].value_counts().head(10).plot(kind='bar')
    plt.title(f"Top 10 Categories: {col}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


rare_threshold = 0.01

for col in cat_cols:
    freq = train_ds[col].value_counts(normalize=True)
    rare_cats = freq[freq < rare_threshold].index.tolist()

    print(f"{col} — Rare Categories (<1%): {rare_cats}")


#Correlation matrix
plt.figure(figsize=(12,8))
corr = train_ds[num_cols].corr()

sns.heatmap(corr, annot=False, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()



##checking for duplicates
d = train_ds.duplicated().sum()
print("Duplicate Rows:", d)


#checking for const cols

const_cols = [col for col in train_ds.columns if train_ds[col].nunique() <= 1]
print("Const cols:", const_cols)


##checking for cols with high uniqeness
high_cardinality = [col for col in cat_cols if train_ds[col].nunique() > 50]
high_cardinality


n = len(num_cols)

r = math.ceil(n / 3)  
c = 3

plt.figure(figsize=(18, 5 * r))

for i, col in enumerate(num_cols, 1):
    plt.subplot(r, c, i)
    sns.kdeplot(train_ds[col], label='Train', shade=True)
    sns.kdeplot(test_ds[col], label='Test', shade=True)
    plt.title(f"{col}")
    plt.legend()

plt.tight_layout()
plt.show()



n = len(cat_cols)
r = math.ceil(n / 3)      
c = 3

plt.figure(figsize=(18, 5 * r))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(r, c, i)

    train_counts = train_ds[col].value_counts(normalize=True).head(10)

    test_counts = test_ds[col].value_counts(normalize=True).head(10)
    drift_df = pd.DataFrame({
        'Train': train_counts,
        'Test': test_counts
    }).fillna(0)

    drift_df.plot(kind='bar', ax=plt.gca())
    plt.title(col)
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()





