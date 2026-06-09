import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")

df.info()


df.head()


# check for missing value
print("Check for missing value:")
print(df.isna().sum())
print("-------------")
print("Check for duplicated value: ")
print(df.duplicated().sum())


category_cols = ["Stage_fear","Drained_after_socializing","Personality"]

for col in category_cols:
    df[col] = df[col].astype('category')

df.info()


import seaborn as sns
import matplotlib.pyplot as plt

cor = df.corr(numeric_only=True)

sns.heatmap(cor, annot=True, cmap='coolwarm')

plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# báº¯t Ä‘áº§u tá»« 1 vÃ¬ 0 lÃ  cá»™t id
col = numeric_cols[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cá»™t 1: Histogram
sns.histplot(data=df, x=col, kde=True, ax=axes[0])
axes[0].set_title(f"Histogram of {col}")
axes[0].set_xlabel(col)
axes[0].set_ylabel("Frequency")

# Cá»™t 2: Boxplot
sns.boxplot(data=df, x=col, ax=axes[1])
axes[1].set_title(f"Boxplot of {col}")
axes[1].set_xlabel(col)

plt.tight_layout()
plt.show()



col = numeric_cols[2]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cá»™t 1: Histogram
sns.histplot(data=df, x=col, kde=True, ax=axes[0])
axes[0].set_title(f"Histogram of {col}")
axes[0].set_xlabel(col)
axes[0].set_ylabel("Frequency")

# Cá»™t 2: Boxplot
sns.boxplot(data=df, x=col, ax=axes[1])
axes[1].set_title(f"Boxplot of {col}")
axes[1].set_xlabel(col)

plt.tight_layout()
plt.show()



col = numeric_cols[3]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cá»™t 1: Histogram
sns.histplot(data=df, x=col, kde=True, ax=axes[0])
axes[0].set_title(f"Histogram of {col}")
axes[0].set_xlabel(col)
axes[0].set_ylabel("Frequency")

# Cá»™t 2: Boxplot
sns.boxplot(data=df, x=col, ax=axes[1])
axes[1].set_title(f"Boxplot of {col}")
axes[1].set_xlabel(col)

plt.tight_layout()
plt.show()



col = numeric_cols[4]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cá»™t 1: Histogram
sns.histplot(data=df, x=col, kde=True, ax=axes[0])
axes[0].set_title(f"Histogram of {col}")
axes[0].set_xlabel(col)
axes[0].set_ylabel("Frequency")

# Cá»™t 2: Boxplot
sns.boxplot(data=df, x=col, ax=axes[1])
axes[1].set_title(f"Boxplot of {col}")
axes[1].set_xlabel(col)

plt.tight_layout()
plt.show()



col = numeric_cols[5]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Cá»™t 1: Histogram
sns.histplot(data=df, x=col, kde=True, ax=axes[0])
axes[0].set_title(f"Histogram of {col}")
axes[0].set_xlabel(col)
axes[0].set_ylabel("Frequency")

# Cá»™t 2: Boxplot
sns.boxplot(data=df, x=col, ax=axes[1])
axes[1].set_title(f"Boxplot of {col}")
axes[1].set_xlabel(col)

plt.tight_layout()
plt.show()



categorical_col = df.select_dtypes(include='category').columns.tolist()
col = categorical_col[0]

# Táº¡o figure vÃ  axes
fig, axes = plt.subplots(1, 1, figsize=(6, 4)) 

# Váº½ countplot
sns.countplot(data=df, x=col, ax=axes)
axes.set_title(f"Countplot of {col}")
axes.set_xlabel(col)
axes.set_ylabel("Count")

plt.tight_layout()
plt.show()



col = categorical_col[1]

# Táº¡o figure vÃ  axes
fig, axes = plt.subplots(1, 1, figsize=(6, 4)) 

# Váº½ countplot
sns.countplot(data=df, x=col, ax=axes)
axes.set_title(f"Countplot of {col}")
axes.set_xlabel(col)
axes.set_ylabel("Count")

plt.tight_layout()
plt.show()


col = categorical_col[2]

# Táº¡o figure vÃ  axes
fig, axes = plt.subplots(1, 1, figsize=(6, 4)) 

# Váº½ countplot
sns.countplot(data=df, x=col, ax=axes)
axes.set_title(f"Countplot of {col}")
axes.set_xlabel(col)
axes.set_ylabel("Count")

plt.tight_layout()
plt.show()


# 1. Fill missing values cho cá»™t sá»‘
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# 2. Fill missing values cho cá»™t phÃ¢n loáº¡i (object/category)
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val[0])

# Check ká»¹
print("Missing values sau khi fill:")
print(df.isna().sum().sum())  # pháº£i = 0



df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Náº¿u lÃºc train cÃ²n giá»¯ id â†’ cáº§n drop
X = df.drop(columns=["id", "Personality"], errors='ignore')  # loáº¡i bá»� cáº£ id láº«n label
y = df["Personality"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

# Kiá»ƒm tra láº§n cuá»‘i trÆ°á»›c khi train
assert not np.isnan(X_train.values).any(), "X_train cÃ²n NaN"
assert not np.isnan(X_test.values).any(), "X_test cÃ²n NaN"

scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        enable_categorical=True
    )
}

for mode in ["baseline"]:
    print(f"\n{'='*30} {mode.upper()} {'='*30}")

    for name, model in models.items():
        print(f"\n--- {name} ---")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap="Blues")
        plt.title(f"{name} - {mode.upper()}")
        plt.show()



from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Kiá»ƒm tra láº§n cuá»‘i trÆ°á»›c khi train
assert not np.isnan(X_train.values).any(), "X_train cÃ²n NaN"
assert not np.isnan(X_test.values).any(), "X_test cÃ²n NaN"

scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced',
        max_iter=1000
    ),
    "Random Forest": RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        enable_categorical=True
    )
}
         
for mode in ["smote"]:
    print(f"\n{'='*30} {mode.upper()} {'='*30}")

    for name, model in models.items():
        print(f"\n--- {name} ---")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap="Blues")
        plt.title(f"{name} - {mode.upper()}")
        plt.show()


submiss = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


category_cols = ["Stage_fear","Drained_after_socializing"]

for col in category_cols:
    submiss[col] = submiss[col].astype('category')


df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})


# TÃ¡ch ID Ä‘á»ƒ giá»¯ láº¡i (náº¿u cÃ³)
submission_ids = submiss["id"] if "id" in submiss.columns else np.arange(len(submiss))
X_submission = submiss.drop(columns=["id"], errors='ignore')


y_submission_pred = model.predict(X_submission)

mapped_pred = pd.Series(y_submission_pred).replace({0: "Extrovert", 1: "Introvert"})

submission_df = pd.DataFrame({
    "id": submission_ids,
    "Personality": mapped_pred
})

submission_df.to_csv("submission.csv", index=False)


