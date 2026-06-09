import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
%matplotlib inline


df=pd.read_excel("/kaggle/input/buy-or-not-purchase-intent-prediction-challenge/Training Dataset.xlsx")


df.head()


df.drop(columns=["ID","Customer ID","Manager","Techincal Manager"],axis=1,inplace=True)


df.shape


df.info()


mean=df["Estimated Win Rate"].mean()
df["Estimated Win Rate"]=df["Estimated Win Rate"].fillna(mean)

mode=df["City"].mode()[0]
df["City"]=df["City"].fillna(mode)


mode=df["Product"].mode()[0]
df["Product"]=df["Product"].fillna(mode)

mode=df["Competitor"].mode()[0]
df["Competitor"]=df["Competitor"].fillna(mode)

mode=df["Marketing Source"].mode()[0]
df["Marketing Source"]=df["Marketing Source"].fillna(mode)

mode=df["Division"].mode()[0]
df["Division"]=df["Division"].fillna(mode)


df.isnull().sum()


df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])


df['Start_Year'] = df['Start Date'].dt.year
df['Start_Month'] = df['Start Date'].dt.month
df['Start_Day'] = df['Start Date'].dt.day
df['Start_Weekday'] = df['Start Date'].dt.weekday  # Monday=0

# From End Date
df['End_Year'] = df['End Date'].dt.year
df['End_Month'] = df['End Date'].dt.month
df['End_Day'] = df['End Date'].dt.day
df['End_Weekday'] = df['End Date'].dt.weekday


df.drop(columns=["Start Date","End Date"],axis=1,inplace=True)


df.head()


cat_cols=df.select_dtypes(include=["object"]).columns

mappings = {}

for col in cat_cols:
    codes, uniques = pd.factorize(df[col])
    df[col] = codes
    mappings[col] = dict(enumerate(uniques))

# Show mappings for each column
for col, mapping in mappings.items():
    print(f"{col} mapping: {mapping}\n")


df["Result"].value_counts()


df.head()


X = df.drop(columns=['Result'])
y = df['Result']


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=1000,depth=6,learning_rate=0.1,loss_function='Logloss',random_seed=42,verbose=100)
model.fit(X_train,y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Predict labels
y_pred = model.predict(X_valid)

# Predict probabilities for AUC
y_pred_proba = model.predict_proba(X_valid)[:, 1]

# Metrics
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
auc = roc_auc_score(y_valid, y_pred_proba)
cm = confusion_matrix(y_valid, y_pred)
report = classification_report(y_valid, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", auc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)



test_df=pd.read_excel("/kaggle/input/buy-or-not-purchase-intent-prediction-challenge/Testing Dateset.xlsx")


test_df = test_df.reset_index(drop=True)


test_df.head()


Id = test_df.ID


test_df.drop(columns=["Manager","Techincal Manager","Customer ID","Result","ID"],axis=1,inplace=True)


test_df.isnull().sum()


mode=test_df["City"].mode()[0]
test_df["City"]=test_df["City"].fillna(mode)


mode=test_df["Product"].mode()[0]
test_df["Product"]=test_df["Product"].fillna(mode)

mode=test_df["Competitor"].mode()[0]
test_df["Competitor"]=test_df["Competitor"].fillna(mode)

mode=test_df["Marketing Source"].mode()[0]
test_df["Marketing Source"]=test_df["Marketing Source"].fillna(mode)

mode=test_df["Division"].mode()[0]
test_df["Division"]=test_df["Division"].fillna(mode)


mode=test_df["Unit Number"].mode()[0]
test_df["Unit Number"]=test_df["Unit Number"].fillna(mode)


test_df['Start Date'] = pd.to_datetime(test_df['Start Date'])
test_df['End Date'] = pd.to_datetime(test_df['End Date'])


# From Start Date
test_df['Start_Year'] = test_df['Start Date'].dt.year
test_df['Start_Month'] = test_df['Start Date'].dt.month
test_df['Start_Day'] = test_df['Start Date'].dt.day
test_df['Start_Weekday'] = test_df['Start Date'].dt.weekday  # Monday=0

# From End Date
test_df['End_Year'] = test_df['End Date'].dt.year
test_df['End_Month'] = test_df['End Date'].dt.month
test_df['End_Day'] = test_df['End Date'].dt.day
test_df['End_Weekday'] = test_df['End Date'].dt.weekday



test_df.drop(columns=["Start Date","End Date"],axis=1,inplace=True)


cat_cols=test_df.select_dtypes(include=["object"]).columns

mappings = {}

for col in cat_cols:
    codes, uniques = pd.factorize(test_df[col])
    test_df[col] = codes
    mappings[col] = dict(enumerate(uniques))

# Show mappings for each column
for col, mapping in mappings.items():
    print(f"{col} mapping: {mapping}\n")


test_df.head()


test_pred_proba = model.predict_proba(test_df)[:, 1]

test_pred_proba = (test_pred_proba >= 0.5).astype(int)
submission = pd.DataFrame({"ID": Id,"TARGET": test_pred_proba})

submission.to_csv("submission.csv", index=False)


df=pd.read_csv("/kaggle/working/submission.csv")

df.head()

