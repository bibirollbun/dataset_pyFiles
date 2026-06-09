import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv(r"/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv(r"/kaggle/input/playground-series-s5e7/test.csv")
submission=pd.read_csv(r"/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.info()


train.head()


test.info()


submission.info()


train.head()


test.head()


submission.head()


train.shape


test.shape


num_col = train.select_dtypes(include="number")

for col in num_col.columns:
    skew_val = train[col].skew()
    print(f"Skewness of {col}: {round(skew_val, 3)}")



skew_val = train["Time_spent_Alone"].skew()
print("Skewness:", skew_val)


num_col = train.select_dtypes(include="number")

for col in num_col.columns:
    train[col] = train[col].fillna(train[col].median())


cat_cols = train.select_dtypes(include="object")

for colx in cat_cols.columns:
    train[colx] = train[colx].fillna(train[colx].mode()[0])


# Handle numerical columns
num_col = test.select_dtypes(include="number")

for col in num_col.columns:
    test[col] = test[col].fillna(test[col].median())

# Handle categorical columns
cat_cols = test.select_dtypes(include="object")

for colx in cat_cols.columns:
    test[colx] = test[colx].fillna(test[colx].mode()[0])



train.isna().sum()
test.isna().sum()
train.duplicated().sum()
test.duplicated().sum()


train.head()


cat_cols


cat_cols = ["Stage_fear", "Drained_after_socializing", "Personality"]


from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
for colm in cat_cols:
    train[colm]=le.fit_transform(train[colm])


train.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Define Features and Target
X = train.drop(columns=["Personality"])
y = train["Personality"]

# Step 2: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Initialize Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Define Strong XGBoost Parameters (Tuned)
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

# Step 5: Evaluate with Cross-Validation
cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=skf, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", np.round(cv_scores, 4))
print("Mean Accuracy:", round(cv_scores.mean(), 4))

# Step 6: Train-Test Split for Final Model Training & Evaluation
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Step 7: Train on Full Training Set
xgb_model.fit(x_train, y_train)

# Step 8: Predict and Evaluate
y_pred = xgb_model.predict(x_test)
print("\nFinal Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Personality', palette='pastel')
plt.title('Target Distribution (Personality)')
plt.xlabel('Personality Type')
plt.ylabel('Count')
plt.show()



plt.figure(figsize=(12,8))
sns.heatmap(train.drop(columns=['Personality']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()



train.drop(columns=['Personality']).hist(figsize=(16, 12), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()



importances = xgb_model.feature_importances_
features = train.drop(columns=['Personality']).columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='Blues_r')
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.show()



test_cat_cols=test.select_dtypes(include="object")
for colss in test_cat_cols:
    test[colss]=le.fit_transform(test[colss])


# Step 1: Scale the test data
test_df_scaled = scaler.transform(test)

# Step 2: Predict personality labels
prediction = xgb_model.predict(test_df_scaled)

# Step 3: Create submission dataframe
submission_df = pd.DataFrame({
    'id': submission['id'], 
    'Personality': prediction
})

# Step 4: Map numeric predictions to labels
submission_df['Personality'] = submission_df['Personality'].map({0: 'Introvert', 1: 'Extrovert'})

# Step 5: Export to CSV
submission_df.to_csv("my_submission_xg.csv", index=False)
print('✅ Submission CSV saved successfully!')


