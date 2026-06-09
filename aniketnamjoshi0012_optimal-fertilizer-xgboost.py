import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder



train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train


test


train.isnull().sum()


train.describe()


train.info()


train["Fertilizer Name"].value_counts()


train['Crop Type'].value_counts()


train['Soil Type'].value_counts()


from sklearn.preprocessing import LabelEncoder


fertilizer_encoder = LabelEncoder()
train['Fertilizer Name Enc'] = fertilizer_encoder.fit_transform(train['Fertilizer Name'])



train.head(5)


train.describe()


columns = [
    
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
    "Fertilizer Name Enc"
]

train_df = train[columns]
correlation = train_df.corr(method='pearson')


correlation


# Converting the categorical columns
categorical_cols = ["Soil Type", "Crop Type"]
for col in categorical_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


import seaborn as sns
import matplotlib.pyplot as plt

def plotshow(column):
    sns.histplot(data=train_df, x=column, bins=8)
    plt.show()

for col in columns:
    if col != 'id':
        plotshow(col)





# Preparing train data
X = train.drop(columns=["id", "Fertilizer Name", "Fertilizer Name Enc"])
y = train["Fertilizer Name Enc"]

X_test = test.drop(columns=["id"])


X


# XG Boost classifier parameters
model = XGBClassifier(
    enable_categorical=True,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss"
)

model.fit(X, y)

# Prediction
probs = model.predict_proba(X_test)


probs


top3_indices = np.argsort(probs, axis=1)[:, ::-1][:, :3]

top3_labels = [
    fertilizer_encoder.inverse_transform(row)
    for row in top3_indices
]

submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": [" ".join(row) for row in top3_labels]
})


submission.to_csv("submission.csv", index=False)

