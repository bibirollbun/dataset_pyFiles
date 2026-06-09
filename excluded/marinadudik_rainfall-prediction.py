import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


# 1ï¸�âƒ£ Ğ—Ğ°Ğ²Ğ°Ğ½Ñ‚Ğ°Ğ¶ĞµĞ½Ğ½Ñ� Ñ‚Ñ€ĞµĞ½ÑƒĞ²Ğ°Ğ»ÑŒĞ½Ğ¸Ñ… Ğ´Ğ°Ğ½Ğ¸Ñ…
df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")

# 2ï¸�âƒ£ ĞŸĞµÑ€ĞµĞ³Ğ»Ñ�Ğ´ Ğ¿ĞµÑ€ÑˆĞ¸Ñ… 5 Ñ€Ñ�Ğ´ĞºÑ–Ğ² (Ñ‰Ğ¾Ğ± Ğ¿Ğ¾Ğ±Ğ°Ñ‡Ğ¸Ñ‚Ğ¸ Ñ�Ñ‚Ñ€ÑƒĞºÑ‚ÑƒÑ€Ñƒ)
print("ğŸ”¹ ĞŸĞµÑ€ĞµĞ³Ğ»Ñ�Ğ´ Ğ¿ĞµÑ€ÑˆĞ¸Ñ… 5 Ñ€Ñ�Ğ´ĞºÑ–Ğ²:")
print(df.head())


# 3ï¸�âƒ£ ĞŸĞµÑ€ĞµĞ²Ñ–Ñ€ĞºĞ° Ğ·Ğ°Ğ³Ğ°Ğ»ÑŒĞ½Ğ¾Ñ— Ñ–Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ñ–Ñ— Ğ¿Ñ€Ğ¾ Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚
print("\nğŸ”¹ Ğ—Ğ°Ğ³Ğ°Ğ»ÑŒĞ½Ğ° Ñ–Ğ½Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ†Ñ–Ñ� Ğ¿Ñ€Ğ¾ Ğ´Ğ°Ğ½Ñ–:")
print(df.info())


# 4ï¸�âƒ£ ĞŸĞµÑ€ĞµĞ²Ñ–Ñ€ĞºĞ° Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ¸Ñ… Ğ·Ğ½Ğ°Ñ‡ĞµĞ½ÑŒ
print("\nğŸ”¹ ĞŸÑ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ñ– Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ½Ñ� Ğ² ĞºĞ¾Ğ¶Ğ½Ğ¾Ğ¼Ñƒ Ñ�Ñ‚Ğ¾Ğ²Ğ¿Ñ†Ñ–:")
print(df.isnull().sum())


# 5ï¸�âƒ£ Ğ�Ğ¿Ğ¸Ñ�Ğ¾Ğ²Ğ° Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ°
print("\nğŸ”¹ Ğ�Ğ¿Ğ¸Ñ�Ğ¾Ğ²Ğ° Ñ�Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ° Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ğ¸Ñ… Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ¸Ñ…:")
print(df.describe())


# 6ï¸�âƒ£ Ğ’Ñ–Ğ·ÑƒĞ°Ğ»Ñ–Ğ·Ğ°Ñ†Ñ–Ñ� Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�Ñƒ ĞºĞ»Ğ°Ñ�Ñ–Ğ² Ñƒ `rainfall`
plt.figure(figsize=(6, 4))
sns.countplot(x=df["rainfall"], palette="coolwarm")
plt.xlabel("Rainfall (0 - Ğ½ĞµĞ¼Ğ°Ñ” Ğ´Ğ¾Ñ‰Ñƒ, 1 - Ğ´Ğ¾Ñ‰)")
plt.ylabel("ĞšÑ–Ğ»ÑŒĞºÑ–Ñ�Ñ‚ÑŒ Ğ·Ñ€Ğ°Ğ·ĞºÑ–Ğ²")
plt.title("Ğ‘Ğ°Ğ»Ğ°Ğ½Ñ� ĞºĞ»Ğ°Ñ�Ñ–Ğ² Ñƒ Rainfall")
plt.show()


print("ğŸ”¹ Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº ÑƒÑ�Ñ–Ñ… Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ¸Ñ…:\n", df.columns)



df.drop(columns=["id"], inplace=True)



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("ĞšĞ¾Ñ€ĞµĞ»Ñ�Ñ†Ñ–Ğ¹Ğ½Ğ° Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñ�")
plt.show()



print(df[["dewpoint", "temparature", "mintemp", "maxtemp", "rainfall"]].corr()["rainfall"])



# Ğ¡ĞµÑ€ĞµĞ´Ğ½Ñ� Ñ‚ĞµĞ¼Ğ¿ĞµÑ€Ğ°Ñ‚ÑƒÑ€Ğ°
df["avg_temp"] = (df["maxtemp"] + df["mintemp"]) / 2


df.drop(columns=["maxtemp", "mintemp" ], inplace=True)


print(df[["dewpoint", "temparature", "avg_temp", "rainfall"]].corr()["rainfall"])


df[["dewpoint", "temparature", "avg_temp"]].hist(figsize=(12, 6), bins=30)
plt.show()



df.drop(columns=["temparature"], inplace=True)



df[["humidity", "cloud", "dewpoint"]].hist(figsize=(12, 6), bins=30)
plt.show()



df_grouped = df.groupby("day")["rainfall"].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=df_grouped.index, y=df_grouped.values, color="red")
plt.xlabel("Day")
plt.ylabel("Ğ™Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ñ–Ñ�Ñ‚ÑŒ Ğ´Ğ¾Ñ‰Ñƒ")
plt.title("Ğ™Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ñ–Ñ�Ñ‚ÑŒ Ğ´Ğ¾Ñ‰Ñƒ Ñƒ Ñ€Ñ–Ğ·Ğ½Ñ– Ğ´Ğ½Ñ– Ñ€Ğ¾ĞºÑƒ")
plt.show()



# Ğ¤ÑƒĞ½ĞºÑ†Ñ–Ñ� Ğ´Ğ»Ñ� Ñ�Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ½Ñ� Ñ�ĞµĞ·Ğ¾Ğ½Ñƒ
def assign_season(day):
    if day <= 90:
        return "winter"
    elif day <= 180:
        return "spring"
    elif day <= 270:
        return "summer"
    else:
        return "fall"

df["season"] = df["day"].apply(assign_season)



df_grouped = df.groupby("season")["rainfall"].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=df_grouped.index, y=df_grouped.values, color="red")
plt.xlabel("Season")
plt.ylabel("Ğ™Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ñ–Ñ�Ñ‚ÑŒ Ğ´Ğ¾Ñ‰Ñƒ")
plt.title("Ğ™Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ñ–Ñ�Ñ‚ÑŒ Ğ´Ğ¾Ñ‰Ñƒ Ñƒ Ñ€Ñ–Ğ·Ğ½Ñ– Ğ´Ğ½Ñ– Ñ€Ğ¾ĞºÑƒ")
plt.show()



df.drop(columns=["day"], inplace=True)
df = pd.get_dummies(df, columns=["season"], drop_first=True)



print(df.corr()["rainfall"].sort_values(ascending=False))



import matplotlib.pyplot as plt
import seaborn as sns

# Ğ“Ñ–Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¸
df.hist(figsize=(12, 10), bins=30)
plt.show()



import numpy as np

df["humidity_log"] = np.log1p(df["humidity"])
df["cloud_log"] = np.log1p(df["cloud"])
df["dewpoint_log"] = np.log1p(df["dewpoint"])

# Ğ’Ğ¸Ğ´Ğ°Ğ»Ñ�Ñ”Ğ¼Ğ¾ Ñ�Ñ‚Ğ°Ñ€Ñ– Ğ²ĞµÑ€Ñ�Ñ–Ñ— Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ¸Ñ…
df.drop(columns=["humidity", "cloud", "dewpoint"], inplace=True)



import matplotlib.pyplot as plt
import seaborn as sns

# Ğ“Ñ–Ñ�Ñ‚Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¸
df.hist(figsize=(12, 10), bins=30)
plt.show()



plt.figure(figsize=(10, 5))
sns.countplot(x="rainfall", data=df)
plt.title("Ğ Ğ¾Ğ·Ğ¿Ğ¾Ğ´Ñ–Ğ» rainfall")
plt.show()



# ĞŸÑ€Ğ¸Ğ¿ÑƒÑ�Ñ‚Ğ¸Ğ¼Ğ¾, Ñ‰Ğ¾ df - Ñ‚Ğ²Ñ–Ğ¹ Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼
X = df.drop(columns=["rainfall"])  # Ğ’Ñ�Ñ– Ñ„Ñ–Ñ‡Ñ–, Ğ¾ĞºÑ€Ñ–Ğ¼ Ñ†Ñ–Ğ»ÑŒĞ¾Ğ²Ğ¾Ñ— Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ¾Ñ—
y = df["rainfall"]  # Ğ¦Ñ–Ğ»ÑŒĞ¾Ğ²Ğ° Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ°



# Ğ†Ğ½Ñ–Ñ†Ñ–Ğ°Ğ»Ñ–Ğ·ÑƒÑ”Ğ¼Ğ¾ SMOTE
smote = SMOTE(random_state=42)

# Ğ’Ğ¸ĞºĞ¾Ğ½ÑƒÑ”Ğ¼Ğ¾ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ�ÑƒĞ²Ğ°Ğ½Ğ½Ñ�
X_resampled, y_resampled = smote.fit_resample(X, y)

# Ğ¡Ñ‚Ğ²Ğ¾Ñ€Ñ�Ñ”Ğ¼Ğ¾ Ğ½Ğ¾Ğ²Ğ¸Ğ¹ Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["rainfall"] = y_resampled



# ĞŸĞµÑ€ĞµĞ²Ñ–Ñ€Ñ�Ñ”Ğ¼Ğ¾ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ� Ğ¿Ñ–Ñ�Ğ»Ñ� SMOTE
rainfall_counts_resampled = df_resampled["rainfall"].value_counts()

# Ğ’Ñ–Ğ·ÑƒĞ°Ğ»Ñ–Ğ·ÑƒÑ”Ğ¼Ğ¾ Ğ±Ğ°Ğ»Ğ°Ğ½Ñ� ĞºĞ»Ğ°Ñ�Ñ–Ğ²
plt.figure(figsize=(6, 4))
sns.barplot(x=rainfall_counts_resampled.index, y=rainfall_counts_resampled.values, palette="coolwarm")
plt.xlabel("Rainfall (0 - Ğ�ĞµĞ¼Ğ°Ñ” Ğ´Ğ¾Ñ‰Ñƒ, 1 - Ğ”Ğ¾Ñ‰)")
plt.ylabel("ĞšÑ–Ğ»ÑŒĞºÑ–Ñ�Ñ‚ÑŒ Ğ·Ñ€Ğ°Ğ·ĞºÑ–Ğ²")
plt.title("Ğ Ğ¾Ğ·Ğ¿Ğ¾Ğ´Ñ–Ğ» ĞºĞ»Ğ°Ñ�Ñ–Ğ² Ğ¿Ñ–Ñ�Ğ»Ñ� SMOTE")
plt.show()

# Ğ’Ğ¸Ğ²Ğ¾Ğ´Ğ¸Ğ¼Ğ¾ ĞºÑ–Ğ»ÑŒĞºÑ–Ñ�Ñ‚ÑŒ Ğ·Ñ€Ğ°Ğ·ĞºÑ–Ğ² Ñƒ ĞºĞ¾Ğ¶Ğ½Ğ¾Ğ¼Ñƒ ĞºĞ»Ğ°Ñ�Ñ–
print(rainfall_counts_resampled)



test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
print(test_df.isnull().sum())



test_df.dropna(inplace=True)
print(test_df.isnull().sum())



test_df.drop(columns=["id"], inplace=True)



test_df["avg_temp"] = (test_df["maxtemp"] + test_df["mintemp"]) / 2
test_df.drop(columns=["maxtemp", "mintemp", "temparature"], inplace=True)




test_df["season"] = test_df["day"].apply(assign_season)
test_df.drop(columns=["day"], inplace=True)
test_df = pd.get_dummies(test_df, columns=["season"], drop_first=True)  # One-hot encoding



import numpy as np

test_df["humidity_log"] = np.log1p(test_df["humidity"])
test_df["cloud_log"] = np.log1p(test_df["cloud"])
test_df["dewpoint_log"] = np.log1p(test_df["dewpoint"])

test_df.drop(columns=["humidity", "cloud", "dewpoint"], inplace=True)



print("ğŸ”¹ Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº ÑƒÑ�Ñ–Ñ… Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ¸Ñ…:\n", df_resampled.columns)


print("ğŸ”¹ Ğ¡Ğ¿Ğ¸Ñ�Ğ¾Ğº ÑƒÑ�Ñ–Ñ… Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ¸Ñ…:\n", test_df.columns)


from sklearn.preprocessing import StandardScaler

# Ğ’Ğ¸Ğ·Ğ½Ğ°Ñ‡Ğ°Ñ”Ğ¼Ğ¾ X Ñ‚Ğ° y
X_train = df_resampled.drop(columns=["rainfall"])
y_train = df_resampled["rainfall"]

# ĞœĞ°Ñ�ÑˆÑ‚Ğ°Ğ±ÑƒÑ”Ğ¼Ğ¾ Ñ„Ñ–Ñ‡Ñ–
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(test_df)

# (Ğ¾Ğ¿Ñ†Ñ–Ğ¾Ğ½Ğ°Ğ»ÑŒĞ½Ğ¾) Ğ¿ĞµÑ€ĞµÑ‚Ğ²Ğ¾Ñ€Ğ¸Ñ‚Ğ¸ Ğ½Ğ°Ğ·Ğ°Ğ´ Ñƒ DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
test_scaled = pd.DataFrame(test_scaled, columns=X_train.columns)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
log_reg.fit(X_train_scaled, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)



# ĞŸĞµÑ€ĞµĞ´Ğ±Ğ°Ñ‡ĞµĞ½Ğ½Ñ� Ğ¹Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹
y_scores_log = log_reg.predict_proba(X_train_scaled)[:, 1]
y_scores_rf = rf_model.predict_proba(X_train_scaled)[:, 1]

# ĞŸĞ¾Ğ±ÑƒĞ´Ğ¾Ğ²Ğ° ĞºÑ€Ğ¸Ğ²Ğ¸Ñ…
fpr_log, tpr_log, _ = roc_curve(y_train, y_scores_log)
fpr_rf, tpr_rf, _ = roc_curve(y_train, y_scores_rf)

roc_auc_log = auc(fpr_log, tpr_log)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Ğ“Ñ€Ğ°Ñ„Ñ–Ğº
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})", linestyle="--")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-ĞºÑ€Ğ¸Ğ²Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»ĞµĞ¹")
plt.legend()
plt.grid(True)
plt.show()



# Ğ’Ğ¸ĞºĞ¾Ñ€Ğ¸Ñ�Ñ‚Ğ¾Ğ²ÑƒÑ”Ğ¼Ğ¾ ĞºÑ€Ğ°Ñ‰Ñƒ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ (Ğ¼Ğ¾Ğ¶ĞµÑˆ Ğ²Ğ¸Ğ±Ñ€Ğ°Ñ‚Ğ¸ Ğ·Ğ° AUC)
best_model = rf_model  # Ğ°Ğ±Ğ¾ log_reg

# ĞŸĞµÑ€ĞµĞ´Ğ±Ğ°Ñ‡Ğ°Ñ”Ğ¼Ğ¾ Ğ¹Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ñ–Ñ�Ñ‚ÑŒ Ğ´Ğ¾Ñ‰Ñƒ
rainfall_pred = best_model.predict_proba(test_scaled)[:, 1]



from sklearn.model_selection import train_test_split

# X_resampled_scaled â€” ÑƒĞ¶Ğµ Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğ¹
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)



# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ĞŸĞµÑ€ĞµĞ´Ğ±Ğ°Ñ‡ĞµĞ½Ğ½Ñ� Ğ¹Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹
y_val_pred_log = log_reg.predict_proba(X_val)[:, 1]
y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]

# ROC Ğ´Ğ»Ñ� Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_val, y_val_pred_log)
roc_auc_log = auc(fpr_log, tpr_log)

# ROC Ğ´Ğ»Ñ� Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_val, y_val_pred_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# ĞŸĞ¾Ğ±ÑƒĞ´Ğ¾Ğ²Ğ° Ğ³Ñ€Ğ°Ñ„Ñ–ĞºÑƒ
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})", linestyle="--")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-ĞºÑ€Ğ¸Ğ²Ğ° Ğ½Ğ° Ğ²Ğ°Ğ»Ñ–Ğ´Ğ°Ñ†Ñ–Ğ¹Ğ½Ğ¾Ğ¼Ñƒ Ğ½Ğ°Ğ±Ğ¾Ñ€Ñ–")
plt.legend()
plt.grid(True)
plt.show()



# Ğ’Ğ¸Ğ·Ğ½Ğ°Ñ‡Ğ°Ñ”Ğ¼Ğ¾ X Ñ‚Ğ° y
X_train = df_resampled.drop(columns=["rainfall"])
y_train = df_resampled["rainfall"]
# ĞŸĞµÑ€ĞµĞ½Ğ°Ğ²Ñ‡Ğ°Ñ”Ğ¼Ğ¾ Random Forest Ğ½Ğ° Ğ²Ñ�Ñ–Ñ… Ğ´Ğ°Ğ½Ğ¸Ñ…
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

# ĞŸĞµÑ€ĞµĞ´Ğ±Ğ°Ñ‡Ğ°Ñ”Ğ¼Ğ¾ Ğ¹Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ñ–Ñ�Ñ‚ÑŒ Ğ´Ğ¾Ñ‰Ñƒ
final_preds = final_model.predict_proba(test_scaled)[:, 1]

# Ğ¡Ñ‚Ğ²Ğ¾Ñ€Ñ�Ñ”Ğ¼Ğ¾ submission
submission = pd.DataFrame({
    "id": range(len(final_preds)),  # Ğ°Ğ±Ğ¾ test_original["id"] Ñ�ĞºÑ‰Ğ¾ Ñ‚Ğ¸ Ñ—Ñ— Ğ·Ğ±ĞµÑ€ĞµĞ³Ğ»Ğ°
    "rainfall": final_preds
})

submission.to_csv("submission.csv", index=False)
print("âœ… Ğ“Ğ¾Ñ‚Ğ¾Ğ²Ğ¾ Ğ´Ğ¾ Ğ·Ğ°Ğ²Ğ°Ğ½Ñ‚Ğ°Ğ¶ĞµĞ½Ğ½Ñ� Ğ½Ğ° Kaggle!")



submission.head()


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. Ğ¡Ñ‚Ğ²Ğ¾Ñ€Ñ�Ñ”Ğ¼Ğ¾ Ğ²Ğ»Ğ°Ñ�Ğ½Ğ¸Ğ¹ Ñ‚Ñ€Ğ°Ğ½Ñ�Ñ„Ğ¾Ñ€Ğ¼ĞµÑ€ Ğ´Ğ»Ñ� Ñ„Ñ–Ñ‡ĞµÑ–Ğ½Ğ¶ĞµĞ½ĞµÑ€Ñ–Ñ—
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ğ¡Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ½Ñ� avg_temp
        X["avg_temp"] = (X["maxtemp"] + X["mintemp"]) / 2

        # Ğ¡Ñ‚Ğ²Ğ¾Ñ€ĞµĞ½Ğ½Ñ� season Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ñ– day
        def assign_season(day):
            if day <= 90:
                return "winter"
            elif day <= 180:
                return "spring"
            elif day <= 270:
                return "summer"
            else:
                return "fall"

        X["season"] = X["day"].apply(assign_season)
        X = pd.get_dummies(X, columns=["season"], drop_first=True)

        # Ğ›Ğ¾Ğ³Ğ°Ñ€Ğ¸Ñ„Ğ¼ÑƒĞ²Ğ°Ğ½Ğ½Ñ�
        for col in ["humidity", "cloud", "dewpoint"]:
            X[col + "_log"] = np.log1p(X[col])

        # Ğ’Ğ¸Ğ´Ğ°Ğ»ĞµĞ½Ğ½Ñ� Ğ·Ğ°Ğ¹Ğ²Ğ¸Ñ… Ğ·Ğ¼Ñ–Ğ½Ğ½Ğ¸Ñ…
        X.drop(columns=["id", "day", "maxtemp", "mintemp", "temparature", 
                        "humidity", "cloud", "dewpoint"], inplace=True)

        return X

# 2. ĞŸĞ¾Ğ±ÑƒĞ´Ğ¾Ğ²Ğ° pipeline Ğ· Ñ–Ğ¼Ğ¿ÑƒÑ‚Ğ°Ñ†Ñ–Ñ”Ñ�
pipeline = ImbPipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("imputer", SimpleImputer(strategy="median")),
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])



# 3. ĞŸÑ€Ğ¸ĞºĞ»Ğ°Ğ´ Ğ²Ğ¸ĞºĞ¾Ñ€Ğ¸Ñ�Ñ‚Ğ°Ğ½Ğ½Ñ�
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
X = train_df.drop(columns=["rainfall"])
y = train_df["rainfall"]


from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np




models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'SVC': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}
for name, model in models.items():
    pipeline.set_params(model=model)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"{name}: Mean ROC AUC = {np.mean(scores):.4f}")


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Ğ�Ğ½Ğ¾Ğ²Ğ»Ñ�Ñ”Ğ¼Ğ¾ pipeline
pipeline.set_params(model=GradientBoostingClassifier())

# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¸ Ğ´Ğ»Ñ� Ğ¿Ğ¾ÑˆÑƒĞºÑƒ
param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7]
}

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¸
print("Ğ�Ğ°Ğ¹ĞºÑ€Ğ°Ñ‰Ñ– Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¸:", grid_search.best_params_)
print("ROC AUC:", grid_search.best_score_)



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Ğ�Ğ½Ğ¾Ğ²Ğ»Ñ�Ñ”Ğ¼Ğ¾ pipeline Ğ· LogisticRegression
pipeline.set_params(model=LogisticRegression(max_iter=1000, random_state=42))

# Ğ¡Ñ–Ñ‚ĞºĞ° Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ–Ğ² Ğ´Ğ»Ñ� Ğ¿Ğ¾ÑˆÑƒĞºÑƒ
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],          # Ñ�Ğ¸Ğ»Ğ° Ñ€ĞµĞ³ÑƒĞ»Ñ�Ñ€Ğ¸Ğ·Ğ°Ñ†Ñ–Ñ—
    'model__penalty': ['l2'],               # l1/l2 (Ğ´Ğ»Ñ� l1 Ñ‚Ñ€ĞµĞ±Ğ° solver='liblinear')
    'model__solver': ['lbfgs']              # Ñ�Ñ‚Ğ°Ğ±Ñ–Ğ»ÑŒĞ½Ğ¸Ğ¹ Ğ´Ğ»Ñ� l2
}

# Ğ—Ğ°Ğ¿ÑƒÑ�Ğº GridSearch
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X, y)

# ğŸ“Š Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¸
print("Ğ�Ğ°Ğ¹ĞºÑ€Ğ°Ñ‰Ñ– Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¸:", grid_search.best_params_)
print("ROC AUC:", grid_search.best_score_)



from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# Ğ’Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²Ğ»Ñ�Ñ”Ğ¼Ğ¾ Ğ±Ğ°Ğ·Ğ¾Ğ²Ñƒ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ
pipeline.set_params(model=CatBoostClassifier(verbose=0, random_state=42))

# ĞŸĞ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¸ Ğ´Ğ»Ñ� Ğ¿ĞµÑ€ĞµĞ±Ğ¾Ñ€Ñƒ
param_grid = {
    'model__depth': [4, 6, 8],
    'model__learning_rate': [0.03, 0.1],
    'model__iterations': [100, 200]
}

# GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

# Ğ�Ğ°Ğ²Ñ‡Ğ°Ğ½Ğ½Ñ�
grid_search.fit(X, y)

# Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¸
print("Ğ�Ğ°Ğ¹ĞºÑ€Ğ°Ñ‰Ñ– Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¸:", grid_search.best_params_)
print("ROC AUC:", grid_search.best_score_)



from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ğ¡Ñ‚Ğ²Ğ¾Ñ€Ñ�Ñ”Ğ¼Ğ¾ Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Ğ�Ğ±Ğ¸Ñ€Ğ°Ñ”Ğ¼Ğ¾ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ (Ğ½Ğ°Ğ¿Ñ€Ğ¸ĞºĞ»Ğ°Ğ´, Logistic Regression)
pipeline.set_params(model=LogisticRegression(C=0.01, solver='lbfgs', max_iter=1000, random_state=42))

# Ğ�Ñ†Ñ–Ğ½Ñ�Ñ”Ğ¼Ğ¾ ROC AUC Ñ‡ĞµÑ€ĞµĞ· ĞºÑ€Ğ¾Ñ�-Ğ²Ğ°Ğ»Ñ–Ğ´Ğ°Ñ†Ñ–Ñ�
scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')

# Ğ ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ğ¸
print("ROC AUC Ğ¿Ğ¾ Ñ„Ğ¾Ğ»Ğ´Ğ°Ñ…:", scores)
print("Ğ¡ĞµÑ€ĞµĞ´Ğ½Ñ–Ğ¹ ROC AUC:", scores.mean())



from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Train/test Ñ�Ğ¿Ğ»Ñ–Ñ‚
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. ĞœĞ¾Ğ´ĞµĞ»ÑŒ Ğ· Ğ½Ğ°Ğ¹ĞºÑ€Ğ°Ñ‰Ğ¸Ğ¼Ğ¸ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ°Ğ¼Ğ¸
best_logreg = LogisticRegression(C=0.01, solver='lbfgs', max_iter=1000, random_state=42)
pipeline.set_params(model=best_logreg)

# 3. Ğ�Ğ°Ğ²Ñ‡Ğ°Ğ½Ğ½Ñ�
pipeline.fit(X_train_lr, y_train_lr)

# 4. AUC Ğ½Ğ° Ñ‚Ñ€ĞµĞ½ÑƒĞ²Ğ°Ğ½Ğ½Ñ–
y_train_pred = pipeline.predict_proba(X_train_lr)[:, 1]
auc_train = roc_auc_score(y_train_lr, y_train_pred)

# 5. AUC Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ñ–
y_test_pred = pipeline.predict_proba(X_test_lr)[:, 1]
auc_test = roc_auc_score(y_test_lr, y_test_pred)

# 6. Ğ’Ğ¸Ğ²Ñ–Ğ´
print(f"ROC AUC Ğ½Ğ° Ñ‚Ñ€ĞµĞ½ÑƒĞ²Ğ°Ğ½Ğ½Ñ–: {auc_train:.4f}")
print(f"ROC AUC Ğ½Ğ° Ğ²Ğ°Ğ»Ñ–Ğ´Ğ°Ñ†Ñ–Ñ—: {auc_test:.4f}")



test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
# 1. ĞŸÑ–Ğ´Ğ³Ğ¾Ñ‚Ğ¾Ğ²ĞºĞ° test-Ğ´Ğ°Ğ½Ğ¸Ñ…
test_X = test_df.drop(columns=["id"])

# 2. ĞŸÑ€Ğ¾Ğ³Ğ½Ğ¾Ğ· Ñ–Ğ¼Ğ¾Ğ²Ñ–Ñ€Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹
submission_proba = pipeline.predict_proba(test_df)[:, 1]

# 3. Ğ¤Ğ¾Ñ€Ğ¼ÑƒÑ”Ğ¼Ğ¾ submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "rainfall": submission_proba
})

# 4. Ğ—Ğ±ĞµÑ€ĞµĞ¶ĞµĞ½Ğ½Ñ� Ñƒ CSV
submission.to_csv("logistic_submission.csv", index=False)

print("âœ… Submission Ñ„Ğ°Ğ¹Ğ» Ğ³Ğ¾Ñ‚Ğ¾Ğ²Ğ¸Ğ¹!")



# 7. ĞŸĞµÑ€ĞµĞ´Ğ±Ğ°Ñ‡ĞµĞ½Ğ½Ñ� Ğ´Ğ»Ñ� test.csv
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
predictions = grid_search.predict_proba(test_df)[:, 1]
submission = pd.DataFrame({"id": test_df["id"], "rainfall": predictions})
submission.to_csv("submission.csv", index=False)

