import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)



df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")



df.shape



df.head()



df.info()



df.describe().T




df.isnull().sum().sort_values(ascending=False)




num_cols = df.select_dtypes(include=np.number).columns

df[num_cols].hist(bins=30, figsize=(18, 14))
plt.suptitle("Distribution of Numerical Features")
plt.show()




cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure()
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
    plt.show()




sns.countplot(data=df, x="diagnosed_diabetes")
plt.title("Diabetes Distribution")
plt.show()




sns.boxplot(data=df, x="diagnosed_diabetes", y="age")
plt.title("Age vs Diabetes")
plt.show()




sns.boxplot(data=df, x="diagnosed_diabetes", y="bmi")
plt.title("BMI vs Diabetes")
plt.show()




features = [
    "physical_activity_minutes_per_week",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "diet_score"
]

for col in features:
    sns.boxplot(data=df, x="diagnosed_diabetes", y=col)
    plt.title(f"{col} vs Diabetes")
    plt.show()




corr = df[num_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()




df.groupby("diagnosed_diabetes")[num_cols].mean().T




pd.crosstab(df.gender, df.diagnosed_diabetes, normalize='index')




pd.crosstab(df.smoking_status, df.diagnosed_diabetes, normalize='index')



pd.crosstab(df.education_level, df.diagnosed_diabetes, normalize='index')


