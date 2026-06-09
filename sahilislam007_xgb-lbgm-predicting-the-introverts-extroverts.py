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


for i in train.columns:
    sns.histplot(data=i,kde=True)
    plt.title(f"The Distribution Of {i}")
    plt.show()


train.head()


train.info()


cat_cols = ["Stage_fear", "Drained_after_socializing", "Personality"]

# Count plots for each categorical column
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x=col, palette="Set2")
    plt.title(f"Frequency of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



plt.figure(figsize=(8, 5))
sns.boxplot(data=train, x="Personality", y="Social_event_attendance", palette="coolwarm")
plt.title("Social Event Attendance by Personality Type")
plt.tight_layout()
plt.show()



train.head()


cat_cols


from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
for colm in cat_cols:
    train[colm]=le.fit_transform(train[colm])


# Step 1: Define features and target
X = train.drop(columns=["Personality"])  # Drop target and ID
y = train["Personality"]

# Step 2: Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Initialize the model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Step 2: Fit the model
xgb_model.fit(x_train_scaled, y_train)

# Step 3: Predict on test set
xgb_preds = xgb_model.predict(x_test_scaled)

# Step 4: Evaluate
print("Accuracy:", round(accuracy_score(y_test, xgb_preds), 4))
print("\nClassification Report:")
print(classification_report(y_test, xgb_preds))

# Step 5: Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, xgb_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Assume you have numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])



from xgboost import XGBClassifier

pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    ))
])



from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy Mean: {scores.mean():.4f}")



importances = pd.Series(xgb_model.feature_importances_, index=x_train.columns)
importances.sort_values().plot(kind='barh', color='slateblue')
plt.title("Feature Importance (XGBoost)")
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
    'id': submission['id'],  # Ensure 'id' exists in submission
    'Personality': prediction
})

# Step 4: Map numeric predictions to labels
submission_df['Personality'] = submission_df['Personality'].map({0: 'Introvert', 1: 'Extrovert'})

# Step 5: Export to CSV
submission_df.to_csv("my_submission_xx.csv", index=False)
print('✅ Submission CSV saved successfully!')


