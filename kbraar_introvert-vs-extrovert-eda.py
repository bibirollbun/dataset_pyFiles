import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train_df


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv') 
test_df


train_df.head()


train_df.describe()


train_df.info()


import matplotlib.pyplot as plt
import seaborn as sns

# Missing value'ları hesapla
missing_values = train_df.isnull().sum()
missing_values = missing_values[missing_values > 0]  

# Grafik çiz
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette="plasma")
plt.title("Missing Values by Feature")
plt.xlabel("Features")
plt.ylabel("Missing Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





personality_counts = train_df['Personality'].value_counts(dropna=False)

colors = plt.cm.tab20b.colors

fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(
    personality_counts.values,
    labels=personality_counts.index,
    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*personality_counts.sum())})',
    startangle=90,
    colors=colors
)
ax.set_title("Personality Distribution")
ax.axis('equal') 
plt.tight_layout()
plt.show()


categoric_cols = ['Stage_fear','Drained_after_socializing','Personality']
numeric_cols = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size', 'Post_frequency']


numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = train_df[(train_df[col] < Q1 - 1.5 * IQR) | (train_df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {outliers.shape[0]}")


from sklearn.preprocessing import LabelEncoder

df_corr = train_df.copy()
df_corr["Personality"] = LabelEncoder().fit_transform(df_corr["Personality"])
df_corr = df_corr.drop('id', axis=1)
corr = df_corr.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="cool", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


categoric_cols = [col for col in categoric_cols if col != "Personality"]
for col in categoric_cols:
    plt.figure(figsize=(8, 4))
    
    prop_df = (
        train_df.groupby(col)["Personality"]
        .value_counts(normalize=True)
        .rename("Rate")
        .reset_index()
    )
    # Barplot çiz
    sns.barplot(
        data=prop_df,
        x=col,
        y="Rate",
        hue="Personality",
        palette="Set1"
    )
    plt.title(f"{col} - Personality")
    plt.ylabel("Rate")
    plt.legend(title="Personality", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    

