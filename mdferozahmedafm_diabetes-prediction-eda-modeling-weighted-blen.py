import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


df_train.head().style.background_gradient(cmap='gist_rainbow')


df_test.head().style.background_gradient(cmap='gist_rainbow')


df_train.describe().style.background_gradient(cmap='tab20c')



sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'


fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Diabetes Prediction: Train vs Test & Key Drivers', fontsize=24, weight='bold', y=0.96)


sns.countplot(data=df_train, x='diagnosed_diabetes', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Target Distribution (Train Set)', fontsize=16)
axes[0, 0].set_xlabel('Diagnosed Diabetes (0=No, 1=Yes)', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
for container in axes[0, 0].containers:
    axes[0, 0].bar_label(container, fmt='%.0f', fontsize=12)


sns.kdeplot(df_train['bmi'], fill=True, label='Train', color='#3498db', ax=axes[0, 1])
sns.kdeplot(df_test['bmi'], fill=True, label='Test', color='#e74c3c', ax=axes[0, 1])
axes[0, 1].set_title('Distribution Comparison: BMI', fontsize=16)
axes[0, 1].set_xlabel('BMI', fontsize=12)
axes[0, 1].legend()


train_gender = df_train['gender'].value_counts(normalize=True).reset_index()
train_gender['Set'] = 'Train'
test_gender = df_test['gender'].value_counts(normalize=True).reset_index()
test_gender['Set'] = 'Test'
gender_comp = pd.concat([train_gender, test_gender])

sns.barplot(data=gender_comp, x='gender', y='proportion', hue='Set', palette=['#3498db', '#e74c3c'], ax=axes[1, 0])
axes[1, 0].set_title('Gender Proportion: Train vs Test', fontsize=16)
axes[1, 0].set_ylabel('Proportion', fontsize=12)


numeric_df = df_train.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()[['diagnosed_diabetes']].sort_values(by='diagnosed_diabetes', ascending=False)

corr = corr.drop('diagnosed_diabetes')

sns.heatmap(corr.head(10), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 1], linewidths=1)
axes[1, 1].set_title('Top 10 Features Correlated with Diabetes', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


import os

base = "/kaggle/input/diabetes-prediction-eda-weighted-blend"

print("Files in folder:")
print(os.listdir(base))



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_single_model(path, output_path="submission.csv"):
    print("--- Loading Single Submission File ---")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path).set_index("id")

    print("Preview:")
    print(df.head())

    # Plot distribution
    plt.figure(figsize=(10, 4))
    sns.histplot(df["diagnosed_diabetes"], bins=100, kde=True, color='purple')
    plt.title("Prediction Probability Distribution")
    plt.show()

    df_out = df.reset_index()
    df_out.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df_out.head()


# Run it
single_path = "/kaggle/input/diabetes-prediction-eda-weighted-blend/submission.csv"
head = analyze_single_model(single_path)
print(head)


