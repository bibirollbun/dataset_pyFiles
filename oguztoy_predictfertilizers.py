import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df1 = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df1.head()


df2 = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
df2.head()


df1.shape, df2.shape


df = pd.concat([df1,df2])


df.sample(4)


df.info()


df.describe().T


df.isnull().sum()


categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(exclude=['object']).columns

print("\nCategorical Columns:", categorical_columns.tolist())
print("\nNumerical Columns:", numerical_columns.tolist())


for column in categorical_columns:
    num_unique = df[column].nunique()
    print(f"'{column}' has {num_unique} unique categories.")


for column in categorical_columns:
    values = df[column].value_counts()
    print(f"'{column}' has {values} values.")


categorical_columns = ['Soil Type', 'Crop Type', 'Fertilizer Name']

for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


numerical_columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=col, kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(10, 8))
corr = df[numerical_columns].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()



df['NPK_Total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
df['Humidity_Temp_Ratio'] = df['Humidity'] / df['Temparature']
df['Moisture_Deviation'] = df['Moisture'] - df['Moisture'].mean()
df["N_to_P"] = df["Nitrogen"] / (df["Phosphorous"] + 1e-5)
df["N_to_K"] = df["Nitrogen"] / (df["Potassium"] + 1e-5)
df["P_to_K"] = df["Phosphorous"] / (df["Potassium"] + 1e-5)
df["P_to_N"] = df["Phosphorous"] / (df["Nitrogen"] + 1e-5)
df["K_to_N"] = df["Potassium"] / (df["Nitrogen"] + 1e-5)
df["Env_Score"] = (df["Temparature"] * 0.4) + (df["Humidity"] * 0.3) + (df["Moisture"] * 0.3)


df = pd.get_dummies(df, columns=["Soil Type","Crop Type"], drop_first=True)


train=df[:750000]
test=df[750000:]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["Fertilizer Name"] = le.fit_transform(train["Fertilizer Name"])


train["Fertilizer Name"].value_counts()


x = train.drop(["id","Fertilizer Name"], axis=1)
y = train[["Fertilizer Name"]]
test = test.drop(["id","Fertilizer Name"], axis=1)


scaler = MinMaxScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8, stratify=y)


xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.25,
    reg_lambda=1.5,
    min_child_weight=3,
    tree_method="gpu_hist",
    predictor="gpu_predictor",
    objective="multi:softmax",
    eval_metric="mlogloss",
    use_label_encoder=False,
    verbosity=0
)

cat_model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    border_count=64,
    bagging_temperature=0.5,
    task_type="GPU",
    devices="0",
    loss_function="MultiClass",
    verbose=0
)

xgb_model.fit(x_train, y_train)
cat_model.fit(x_train, y_train)

xgb_preds = xgb_model.predict(x_test)
cat_preds = cat_model.predict(x_test)

xgb_acc = accuracy_score(y_test, xgb_preds)
cat_acc = accuracy_score(y_test, cat_preds)

print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print(f"CatBoost Accuracy: {cat_acc:.4f}")


model = xgb_model.fit(x, y)
proba = model.predict_proba(test)

top3_indices = proba.argsort(axis=1)[:, -3:][:, ::-1]  

top3_labels = le.inverse_transform(top3_indices.ravel())  
top3_labels = top3_labels.reshape(top3_indices.shape)     


joined_preds = [' '.join(row) for row in top3_labels]

submission = pd.DataFrame({
    "id": df2["id"],
    "Fertilizer Name": joined_preds
})

submission.to_csv("submission.csv", index=False)




