import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


df_train = train.copy()
df_test = test.copy()
cat_features = ['Soil Type', 'Crop Type']
label_encoders = {}

for col in cat_features:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le

fertilizer_le = LabelEncoder()
df_train['Fertilizer Name'] = fertilizer_le.fit_transform(df_train['Fertilizer Name'])


features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous']
df_small = df_train.iloc[:750000]
X = df_small[features]
y = df_small['Fertilizer Name']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(fertilizer_le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    verbosity=0
)
model.fit(X_train, y_train)


def mapk(actual, predicted, k=5):
    score = 0.0
    for a, p in zip(actual, predicted):
        try:
            index = p.index(a)
        except ValueError:
            index = -1
        if index != -1 and index < k:
            score += 1.0 / (index + 1)
    return score / len(actual)

val_probs = model.predict_proba(X_val)
val_top5 = np.argsort(val_probs, axis=1)[:, -5:][:, ::-1]
val_top5_labels = [[fertilizer_le.classes_[i] for i in row] for row in val_top5]
val_true_labels = [fertilizer_le.classes_[i] for i in y_val]

map5_score = mapk(val_true_labels, val_top5_labels)
print("MAP@5 得分: ", map5_score)


X_test = df_test[features]
test_probs = model.predict_proba(X_test)
test_top5 = np.argsort(test_probs, axis=1)[:, -5:][:, ::-1]
test_top5_labels = [[fertilizer_le.classes_[i] for i in row] for row in test_top5]
submission = sample_submission.copy()
submission["Fertilizer Name"] = [" ".join(row) for row in test_top5_labels]
submission.to_csv("submission.csv", index=False)
print("✅ 提交文件保存为 submission.csv")

