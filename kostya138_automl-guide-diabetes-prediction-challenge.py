import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
df_train = df_train.drop('id', axis = 1)
df_test = df_test.drop('id', axis = 1)
df_train.head(), df_test.head()



df_train.info()



df_train.isna().sum()



sns.countplot(data=df_train, x="diagnosed_diabetes")
plt.title("Target Distribution")
plt.show()



num_cols = df_train.select_dtypes(include=[np.number]).columns.drop("diagnosed_diabetes")

df_train[num_cols].hist(figsize=(16,12), bins=30)
plt.tight_layout()
plt.show()



plt.figure(figsize=(16,12))
for i, col in enumerate(num_cols[:9]):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=df_train[col])
    plt.title(col)
plt.tight_layout()
plt.show()



plt.figure(figsize=(14,10))
sns.heatmap(df_train[num_cols.union(['diagnosed_diabetes'])].corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()



from sklearn.preprocessing import LabelEncoder

cat_cols = df_train.select_dtypes(include=['object']).columns

df_train_le = df_train.copy()
df_test_le = df_test.copy()

for col in cat_cols:
    le = LabelEncoder()
    df_train_le[col] = le.fit_transform(df_train_le[col])
    df_test_le[col] = le.transform(df_test_le[col])



def remove_top50_outliers_quantile(df, low_q=0.01, high_q=0.99):
    df_clean = df.copy()
    cols = df_clean.select_dtypes(include=["int", "float"]).columns
    
    for col in cols:
        q_low = df_clean[col].quantile(low_q)
        q_high = df_clean[col].quantile(high_q)

        outliers = list(df_clean[df_clean[col] < q_low].index) + \
                   list(df_clean[df_clean[col] > q_high].index)

        df_clean = df_clean.drop(outliers[:50], errors="ignore")
    
    return df_clean

df_train_clean = remove_top50_outliers_quantile(df_train_le)



from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

automl = TabularAutoML(
    task=Task('binary'),
    timeout=1500,
    cpu_limit=4,
    general_params={'use_algos': ['cb']}
)

oof_pred = automl.fit_predict(
    df_train_clean,
    roles={'target': 'diagnosed_diabetes', 'category': cat_cols},
    verbose=0
)



from sklearn.metrics import roc_auc_score

score = roc_auc_score(df_train_clean["diagnosed_diabetes"], oof_pred.data[:,0])
score



test_pred = automl.predict(df_test_le).data[:, 0]
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

submission = pd.DataFrame({
    'id': sample['id'],
    'diagnosed_diabetes': test_pred
})

submission.to_csv("submission.csv", index=False)


