import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')



print(train_df.head())
print(test_df.head())


print(train_df.info())
print(train_df.describe())


print(train_df.isnull().sum())



plt.figure(figsize=(10,6))
sns.boxplot(data=train_df.drop(columns=["id", "rainfall"]))
plt.xticks(rotation=90)
plt.show()


print(train_df.columns)
print(test_df.columns)



test_ids = test_df["id"]
train_df = train_df.drop(columns=["id"], errors='ignore') 
test_df = test_df.drop(columns=["id"], errors='ignore')  
print(test_ids.head())



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X = train_df.drop(columns=["rainfall"])
y = train_df["rainfall"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred_prob = model.predict_proba(X_val_scaled)[:, 1]

roc_auc = roc_auc_score(y_val, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc}")



print(test_df.isnull().sum())

test_df.fillna(test_df.mean(), inplace=True)





X_test_scaled = scaler.transform(test_df)

rainfall_predictions = model.predict(X_test_scaled)

submission_df = pd.DataFrame({
    "id": test_ids,  
    "rainfall": rainfall_predictions  
})


submission_df.to_csv("submission.csv", index=False)
print("submission.csv created successfully!")

