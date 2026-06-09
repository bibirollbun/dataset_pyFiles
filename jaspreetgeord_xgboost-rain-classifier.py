import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

train_df.head()


train_df.describe()


plt.figure(figsize=(10, 8)) 
sns.heatmap(train_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.9)
plt.title("Correlation between Features")


plt.figure(figsize=(10, 8))
sns.histplot(train_df["rainfall"], bins=2)
plt.title("Rainfall Distribution", fontsize=14, fontweight="bold")


for col in train_df.columns:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=train_df, x="rainfall", y=col)
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x=col, hue="rainfall", kde=True)


train_data = train_df.iloc[:, :-1]  
label_data = train_df.iloc[:, -1]  


import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(train_data, label_data, test_size=0.2)

model = xgb.XGBClassifier(
    objective="binary:logistic",  
    n_estimators=10,
    learning_rate=0.1,
    max_depth=6,
)

# Train the model
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


Y_pred = model.predict(test_df)

submission = pd.DataFrame({
    "id": test_df["id"], 
    "rainfall": Y_pred  
})

submission.to_csv("submission.csv", index=False)

submission.head()

