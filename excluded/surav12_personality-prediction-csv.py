import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
print(df_train)
print(df_test)
print(df_submission)


df_train.head()


df_test.head()


df_train.info()
df_test.info()


df_train.describe()
df_test.describe()


print(df_test.dtypes)


print(df_train.dtypes)


df_train.isnull().sum()


numerical_cols = [
    "Time_spent_Alone", "Social_event_attendance",
    "Going_outside", "Friends_circle_size", "Post_frequency"
]

for col in numerical_cols:
    df_train[col] = df_train[col].fillna(df_train[col].median())
    
for col in ['Stage_fear', 'Drained_after_socializing']:
    df_train[col] = df_train[col].fillna(df_train[col].mode()[0])



df_train.isnull().sum()


df_test.isnull().sum()


numerical_cols = [
    "Time_spent_Alone", "Social_event_attendance",
    "Going_outside", "Friends_circle_size", "Post_frequency"
]

for col in numerical_cols:
    df_test[col] = df_test[col].fillna(df_test[col].median())
    
for col in ['Stage_fear', 'Drained_after_socializing']:
    df_test[col] = df_test[col].fillna(df_test[col].mode()[0])


df_test.isnull().sum()


le_personality = LabelEncoder()
df_train["Personality"] = le_personality.fit_transform(df_train["Personality"])

encoders = {}
cols_to_encode = ["Stage_fear", "Drained_after_socializing"]

for col in cols_to_encode:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    encoders[col] = le

x = df_train[cols_to_encode]
y = df_train["Personality"]
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

df_test_encoded = df_test[cols_to_encode].copy()
for col in cols_to_encode:
    df_test_encoded[col] = encoders[col].transform(df_test[col])



print(df_train.head())    
print(df_test.head())     
print(df_submission.head())


models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(),
    "KNearestNeighbor": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    results[name] = acc
for name, acc in results.items():
    print(f"{name} Accuracy: {acc:.4f}")


models = {
    "CatBoostClassifier": CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
    "LightGBMClassifier": LGBMClassifier(n_estimators=100, random_state=42),
    "XGBClassifier": XGBClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    results[name] = acc
for name, acc in results.items():
    print(f"{name} Accuracy: {acc:.4f}")


sns.heatmap(x.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

low_var_cols = [col for col in x.columns if x[col].nunique() <= 1]
print("Low Variance or Constant Features:", low_var_cols)

print("Duplicate rows in train:", x.duplicated().sum())


pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x)
print(x_pca)


# Step 1: Define models
cat = CatBoostClassifier(verbose=0, random_state=42)
xgb = XGBClassifier(verbosity=0, random_state=42)
lgb = LGBMClassifier(verbose=0, random_state=42)

# Step 2: Create the Voting Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('cat', cat),
        ('xgb', xgb),
        ('lgb', lgb)
    ],
    voting='soft'
)

# Step 3: Fit the ensemble on full training data
ensemble.fit(x, y)

# Step 4: Predict using the trained model
x_test = df_test_encoded[["Stage_fear", "Drained_after_socializing"]]
predictions = ensemble.predict(x_test)

# Step 5: Decode predictions back to labels
predicted_labels = le_personality.inverse_transform(predictions)

# Step 6: Prepare submission
submission_df = pd.DataFrame({
    "id": df_test["id"],
    "Personality": predicted_labels
})

# Step 7: Save submission
submission_df.to_csv("submission.csv", index=False)
print("✅ Submission file saved correctly!")


