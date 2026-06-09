import pandas as pd

train_data = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv")

print(train_data.head())

print(train_data.info())

print("\nMissing Values:\n", train_data.isnull().sum())


print("\nSummary Statistics:\n", train_data.describe())
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    print(f"\n{col} unique values: {train_data[col].unique()}")


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=train_data, x='satisfaction')
plt.title('Satisfaction Distribution')
plt.show()



numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.drop(['id'])

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='satisfaction', y=col, data=train_data)
    plt.title(f'{col} vs Satisfaction')
    plt.tight_layout()
    plt.show()

