import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pd.set_option('display.max_columns', 100)


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df.head()


df.info()


conteo = df["diagnosed_diabetes"].value_counts().sort_index()
total = conteo.sum()
porcetage = conteo / total

plt.bar(porcetage.index.astype(str), porcetage.values)

# Agregar porcentaje encima de cada barra
for i, v in enumerate(porcetage.values):
    plt.text(i, v + 0.01, f"{v:.2%}", 
             ha='center', va='bottom', fontsize=12)

plt.xlabel("Value")
plt.ylabel("Proportion")
plt.title("Diagnosed diabetes")
plt.ylim(0, 1.1)
plt.show()


import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if i != "diagnosed_diabetes":        
        sns.kdeplot(data=df, x=col, hue="diagnosed_diabetes", fill=True)
        plt.title(f"KDE de {col}")
        plt.show()


numeric_df = df.select_dtypes(include=["number"])
X_train, X_test, y_train, y_test = train_test_split(numeric_df.drop(columns=["diagnosed_diabetes"]),numeric_df["diagnosed_diabetes"],
    test_size=0.2, random_state=42, stratify=numeric_df["diagnosed_diabetes"])

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)


result = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42
)


importances = result["importances_mean"]
stds = result["importances_std"]

df_importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance_mean": importances,
    "importance_std": stds
})

# Ordenar de mayor a menor importancia
df_importances = df_importances.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

df_importances


plt.figure(figsize=(10,6))
plt.barh(df_importances["feature"], df_importances["importance_mean"])
plt.gca().invert_yaxis()
plt.title("Permutation Importance (RandomForest)")
plt.xlabel("Mean Importance")
plt.show()


numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)


X_train, X_test, y_train, y_test = train_test_split(numeric_df.drop(columns=["diagnosed_diabetes"]),numeric_df["diagnosed_diabetes"],
    test_size=0.2, random_state=42, stratify=numeric_df["diagnosed_diabetes"])

rf2 = RandomForestClassifier(random_state=42, n_jobs=-1)
rf2.fit(X_train, y_train)


result = permutation_importance(
    rf2, X_test, y_test, n_repeats=10, random_state=42
)


importances = result["importances_mean"]
stds = result["importances_std"]

df_importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance_mean": importances,
    "importance_std": stds
})

# Ordenar de mayor a menor importancia
df_importances = df_importances.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

df_importances




