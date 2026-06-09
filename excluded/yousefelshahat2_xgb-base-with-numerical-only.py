from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df.shape


df.head()


df.describe()


# null percentage
(df.isnull().sum()*100/df.shape[0])[lambda x: x>0]


target = "Personality"
cols = df.columns.tolist()
cat = [col for col in cols if df[col].dtype == "object" and col != target]
num = [col for col in cols if df[col].dtype != "object" and col != "id"]


if df[target].dtype == "object":
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
X_train, X_test, y_train, y_test = train_test_split(df[num], df[target], test_size = 0.1, stratify = df[target])
model = XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy_score(y_test, pred)


final = model.predict(df_test[num])
final_file = df_test["id"].copy()
final_file = pd.DataFrame(final_file)
final_file["Personality"] = le.inverse_transform(final)
final_file.to_csv("submission.csv", index = False)
final_file.head()




