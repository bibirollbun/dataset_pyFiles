import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')

df.drop(columns='id',inplace=True)


df.head()


print('shape : ',df.shape)
df.info()


(df.isna().sum() / df.shape[0]).sort_values(ascending=False)


df_num = df.select_dtypes(include='number')


df_num.describe()


# Replace infinite values with NaN to avoid warnings
df_num = df_num.replace([np.inf, -np.inf], np.nan)

# Set the style
sns.set_style("whitegrid")

# Plot histograms with KDE curve
num_features = df_num.columns
plt.figure(figsize=(15, 10))

for i, feature in enumerate(num_features):
    plt.subplot(2, 3, i + 1)  # Adjust layout based on the number of features
    sns.histplot(df_num[feature], bins=50, kde=True, color="royalblue")
    plt.axvline(df_num[feature].mean(), color='red', linestyle='dashed', linewidth=2, label="Mean")
    plt.axvline(df_num[feature].median(), color='green', linestyle='dashed', linewidth=2, label="Median")
    plt.legend()
    plt.title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()


df_num['Number_of_Ads'].value_counts()


target = 'Listening_Time_minutes'
# Scatter plots
fig, axes = plt.subplots(nrows=len(df_num.columns) - 1, ncols=1, figsize=(8, 20))

for i, col in enumerate(df_num.columns):
    if col != target:
        sns.scatterplot(x=df_num[col], y=df_num[target], alpha=0.5, ax=axes[i])
        axes[i].set_title(f"{col} vs. {target}")

plt.tight_layout()
plt.show()


# Compute correlation matrix
corr_matrix = df_num.corr()

# Display heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


from sklearn.feature_selection import mutual_info_regression


# Selecting numerical features (excluding the target)
X = df_num.drop(columns=["Listening_Time_minutes"]).fillna(df_num.median())
y = df_num["Listening_Time_minutes"]

# Compute mutual information
mi_scores = mutual_info_regression(X, y, random_state=42)

# Convert to DataFrame for better visualization
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='Mutual Information', ascending=False)

# Display results
print(mi_scores_df)


fig, axes = plt.subplots(nrows=len(df_num.columns), figsize=(10, 20))

for i, col in enumerate(df_num.columns):
    sns.boxplot(x=df_num[col], ax=axes[i], orient='h')
    axes[i].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()




'''df_num['Number_of_Ads'] = df_num['Number_of_Ads'].apply(lambda x: min(x, 3))


Q1,Q3 = df_num["Episode_Length_minutes"].quantile([0.25,0.75])
IQR = (Q3 - Q1) + 1.5* Q3
df_num.loc[df_num["Episode_Length_minutes"] > IQR, "Episode_Length_minutes"] = Q3'''


df_cat = df.select_dtypes(exclude='number')


df_cat.head()


df_cat.nunique()


# Features to plot
categorical_features = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

plt.figure(figsize=(15, 10))

for i, col in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=df[col], palette="viridis", order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# Countplot for combinations of categorical features
plt.figure(figsize=(16, 12))

# List of combinations of categorical features to plot
combinations = [
    ('Genre', 'Episode_Sentiment'),
    ('Publication_Day', 'Episode_Sentiment'),
    ('Publication_Time', 'Episode_Sentiment'),
    ('Genre', 'Publication_Day'),
    ('Genre', 'Publication_Time'),
    ('Publication_Day', 'Publication_Time')
]

for i, (x, hue) in enumerate(combinations, 1):
    plt.subplot(3, 2, i)
    sns.countplot(x=x, hue=hue, data=df_cat, palette="viridis")
    plt.title(f"Count of {hue} by {x}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# List of categorical features to loop through
categorical_features = ["Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

# Plotting
plt.figure(figsize=(12, 8))

for i, col in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=col, y='Listening_Time_minutes', data=df, palette="viridis")
    plt.title(f"Listening Time by {col}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


df.groupby("Genre")["Listening_Time_minutes"].describe()


df.groupby("Publication_Day")["Listening_Time_minutes"].describe()


df.groupby("Publication_Time")["Listening_Time_minutes"].describe()


df.groupby("Episode_Sentiment")["Listening_Time_minutes"].describe()




