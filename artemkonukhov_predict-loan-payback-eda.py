import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


cat = train.select_dtypes("object").columns
num = list(set(train.columns) - set(cat) - set('id'))
target_name = "loan_paid_back"
drop_columns = ['id', "loan_paid_back"]


fig, axes = plt.subplots(len(cat) // 3, 3, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(cat):
    values = train[col].value_counts(normalize=True)
    axes[i].bar(values.index, values.values)
    axes[i].set_title(col + f" | {train[col].nunique()} unique")
    axes[i].tick_params(axis="x", rotation=90)
    
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, len(num) // 3 + int((len(num) % 3) > 0), figsize=(15, 15))
axes = axes.flatten()
for i, col in enumerate(num):
    sns.histplot(data=train, x=col, hue="loan_paid_back", ax=axes[i], kde=True)
    axes[i].set_title(col)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
    
plt.show()


corr = train.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(abs(corr[abs(corr) > .0]), annot=True, cmap='coolwarm', square=True)
plt.show()

