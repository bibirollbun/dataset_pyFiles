# Load the Libraries
import pandas as pd
import numpy as np


# Reading the Data
df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')

print("The dimension of the train dataset is:", df_train.shape)
print("The dimension of the test dataset is:", df_test.shape)


df_train.head()


df_test.head()


df_train["Fertilizer Name"].value_counts(normalize=True)


train_dup = df_train.duplicated().sum()
test_dup = df_train.duplicated().sum()
print("Train Dataset Duplicates:", {train_dup})
print("Test Dataset Duplicates:", {test_dup})


# Checking for missing values
mv_train = df_train.isna().sum()
mv_test = df_test.isna().sum()

print("Missing Values in Train Dataset:")
print(f"{mv_train}\n")

print("Missing Values in Test Dataset:")
print(mv_test)


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10, 6))
sns.heatmap(df_train.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Features')
plt.show();


fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.boxplot(data=df_train, x="Fertilizer Name", y="Temparature", ax=ax[0], color="steelblue")
sns.boxplot(data=df_train, x="Fertilizer Name", y="Moisture", ax=ax[1], color="green");




fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.boxplot(data=df_train, x="Fertilizer Name", y="Potassium", ax=ax[0], color="steelblue")
sns.boxplot(data=df_train, x="Fertilizer Name", y="Nitrogen", ax=ax[1], color="green");


from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from lightgbm import LGBMClassifier


cat_columns = [i for i in df_train.columns if df_train[i].dtype == np.object_]
num_columns = [i for i in df_train.columns if i not in cat_columns]


label_enc = LabelEncoder()
for i in cat_columns[:-1]:
    df_train[i] = label_enc.fit_transform(df_train[i])
    df_test[i] = label_enc.transform(df_test[i])
df_train['Fertilizer Name'] = label_enc.fit_transform(df_train['Fertilizer Name'])


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


X = df_train.drop('Fertilizer Name', axis = 1)
y = df_train["Fertilizer Name"]


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(shape = (len(df_train), y.nunique()))
pred_prob = np.zeros(shape = (len(df_test), y.nunique()))

lgb_model = LGBMClassifier(
     n_estimators= 1100,
     learning_rate= 0.01,
     num_leaves= 169,
     max_depth= 10,
     min_child_samples= 19,
     subsample= 0.6420340301820501,
     colsample_bytree= 0.43403799235854973,
     reg_alpha= 6.294093849568123,
     reg_lambda= 5.5559072866866455,
     random_state=42,
     verbosity=-1
)

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print('#' * 15, i+1, '#' * 15)
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    lgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
    oof[valid_idx] = lgb_model.predict_proba(x_valid)
    pred_prob += lgb_model.predict_proba(df_test)

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]  
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"✅ FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")


top_3_preds = np.argsort(oof, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in y]
map3_score = mapk(actual, top_3_preds)
print(f'✅ Final MAP@3 Score: {map3_score:.5f} ')


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('lgm_sub_1', index=False)
print("✅ Submission file saved as 'submission.csv'")


submission.head()

