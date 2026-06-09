import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


def summarize_data(df, name="Dataset"):
    print(f"\n=== {name} Summary ===")
    print("Train Head", df.head())
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDescription:")
    print(df.describe().T)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nUnique values:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()}")


summarize_data(train, "Train")


def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


train = remove_outliers(train, ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'])


encoder = LabelEncoder()
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    train[col] = encoder.fit_transform(train[col])


fertilizer_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
reverse_map = {v: k for k, v in fertilizer_map.items()}


for col in ['Soil Type', 'Crop Type']:
    test[col] = encoder.fit_transform(test[col])


X = train.drop(['id', 'Fertilizer Name'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop('id', axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)


val_preds = lgbm.predict_proba(X_val)
val_score = top_k_accuracy_score(y_val, val_preds, k=3)
print(f"\nValidation MAP@3 Score: {val_score:.4f}")


test_preds = lgbm.predict_proba(X_test)
top3_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]


submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": [" ".join([reverse_map[p] for p in row]) for row in top3_preds]
})


submission.to_csv("submission.csv", index=False)
print("\nSubmission saved as submission.csv")




