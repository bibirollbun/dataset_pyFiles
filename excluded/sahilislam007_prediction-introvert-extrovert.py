import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import plotly.io as pio
pio.renderers.default = 'notebook'  # or 'iframe_connected'
pio.renderers.default = 'iframe_connected'


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


import seaborn as sns
import matplotlib.pyplot as plt

# List of numerical columns
num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside",
            "Friends_circle_size", "Post_frequency"]

# Plot histograms with KDE
for col in num_cols:
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.histplot(data=train, x=col, kde=True, color='mediumorchid',edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


cat_cols = ["Stage_fear", "Drained_after_socializing", "Personality"]

# Count plots for each categorical column
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=train, x=col, palette="Set2")
    plt.title(f"Frequency of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



import plotly.express as px

# Step 1: Count category frequencies
personality_counts = train["Personality"].value_counts().reset_index()
personality_counts.columns = ["Personality", "Count"]

# Step 2: Plot pie chart
fig = px.pie(personality_counts, names="Personality", values="Count",
             title="Personality Distribution",
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.show()



cat_cols = ["Stage_fear", "Drained_after_socializing", "Personality"]

for col in cat_cols:
    counts = train[col].value_counts().reset_index()
    counts.columns = [col, "Count"]

    fig = px.pie(counts, names=col, values="Count",
                 title=f"{col} Distribution",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.show()



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



from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_test, xgb_preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(8, 4))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("XGBoost Classification Metrics")
plt.tight_layout()
plt.show()



importances = pd.Series(xgb_model.feature_importances_, index=x_train.columns)
importances.sort_values().plot(kind='barh', color='slateblue')
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()



test_cat_cols=test.select_dtypes(include="object")
for colss in test_cat_cols:
    test[colss]=le.fit_transform(test[colss])


test_df=scaler.transform(test)
prediction=xgb_model.predict(test_df)


submission["Personality"]=prediction
submission.shape
submission_df = submission.reset_index(drop=True)
submission_df = pd.DataFrame({'id': submission_df["id"], 'Personality': prediction})
submission_df['Personality'] = submission_df['Personality'].map({0: 'Introvert', 1: 'Extrovert'})
submission_df
submission_df.to_csv(r"my_submission_x.csv", index=False)
print('save csv success')

