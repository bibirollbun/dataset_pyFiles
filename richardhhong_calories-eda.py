import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


df_train.info()


df_train.head()


df_test.head()


# missing values
print("Train Set Missing Values")
print(df_train.isna().sum())
print( )
print("Test Set Missing Values")
print(df_test.isna().sum())


# summary statistics
print("Train Set Summary Statistics")
print(df_train.describe())
print( )
print("Test Set Summary Statistics")
print(df_test.describe())


len(df_train)


df_all = pd.concat([df_train, df_test]).reset_index(drop=True)

train_indices = len(df_train) - 1

df_all.describe()


df_train.drop(columns=['id'], inplace=True)
num_vars = df_train.select_dtypes(include=['int64','float64']).columns
cat_vars = ['Sex']


# distribution of Numeric Features
fig, axes = plt.subplots(4, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(num_vars):
    sns.histplot(df_train[feature], ax=axes[i], bins=min(30, df_train[feature].nunique()))
    axes[i].set_title(f"Distribution of {feature}")

axes[7].remove()
plt.tight_layout()
plt.show()


# Scatterplots between features
sns.pairplot(df_train[num_vars])
plt.show()


# correlation Matrix
corr_matrix = df_train[num_vars].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.xticks(rotation=45)
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# distribution of sex
fig, axes = plt.subplots(1, 2, figsize=(8, 8))
axes = axes.flatten()

sns.histplot(df_train['Sex'], ax=axes[0])
axes[0].set_title("Distribution of Sex for Training Set")
sns.histplot(df_test['Sex'], ax=axes[1])
axes[1].set_title("Distribution of Sex for Test Set")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(num_vars):
    sns.boxplot(x=df_train['Sex'], y=df_train[feature], ax=axes[i])
    plt.title(f"Relationship Between {feature} and Sex")

axes[7].remove()
plt.tight_layout()
plt.show()


# distribution of Numeric Features filtered by gender
fig, axes = plt.subplots(4, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(num_vars):
    sns.kdeplot(data=df_train, x=feature, ax=axes[i], hue='Sex', fill=True, alpha=0.3)
    axes[i].set_title(f"Distribution of {feature}")

axes[7].remove()
plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler

def make_features(df, test=False):
    df_temp = df.copy()

    # dummy encoding
    df_temp['Sex'] = df_temp['Sex'].astype('category')

    # log and exponential features
    df_temp['log_heart_rate'] = np.log(df_temp['Heart_Rate'])
    df_temp['log_duration'] = np.log(df_temp['Duration'])
    df_temp['exp_body_temp'] = np.exp(df_temp['Body_Temp']) # exponential relationship

    # new features, to be culled off in feature selection
    df_temp['BMI'] = df_temp['Weight'] / (df_temp['Height']/100)**2

    # transformations of outcome variable
    if test == False:
        df_temp['log_calories'] = np.log(df_temp['Calories'])
    
    return df_temp


df_train1 = make_features(df_train)


# domain_specific_features = ['BMI', 'Duration_Heart_Rate', ]
num_features = df_train1.select_dtypes(include=['int64','float64']).columns
feat_with_log = ['BMI', 'log_duration', 'log_heart_rate', 'exp_body_temp', 'Calories', 'log_calories']
feat_no_log = ['BMI', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories', 'log_calories']


corr_matrix_no_log = df_train1[feat_no_log].corr()
corr_matrix_log = df_train1[feat_with_log].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes = axes.flatten()

sns.heatmap(corr_matrix_no_log, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=1, ax=axes[0])
sns.heatmap(corr_matrix_log, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=1, ax=axes[1])
axes[0].set_title("Correlations without Log Features")
axes[1].set_title("Correlations with Log Features")

plt.show()


# scatterplots for relationship between log and exponential features and outcome
feat_with_log = ['BMI', 'log_duration', 'log_heart_rate', 'exp_body_temp']

fig, axes = plt.subplots(4, 2, figsize=(8, 8))

for i in range(len(feat_with_log)):
    # sns.scatterplot(data=df_train1, x=feat_with_log[i], y='Calories', ax=axes[i][0])
    sns.regplot(data=df_train1, x=feat_with_log[i], y='Calories', ax=axes[i][0], lowess=True, line_kws={"color":"red"})
    axes[i][0].set_title(f'Scatterplot of {feat_with_log[i]} and Calories')
    # sns.scatterplot(data=df_train1, x=feat_with_log[i], y='log_calories', ax=axes[i][1])
    sns.regplot(data=df_train1, x=feat_with_log[i], y='log_calories', ax=axes[i][1], lowess=True, line_kws={"color":"red"})
    axes[i][1].set_title(f'Scatterplot of {feat_with_log[i]} and log Calories')

plt.tight_layout()
plt.show()


df_train1.head()


def make_interactions(df, test = False, log = False):
    df_temp = df.copy()
    df_temp = pd.get_dummies(df_temp, columns=['Sex'])

    num_features = df_temp.drop(columns=['Calories']).select_dtypes(include=['int64', 'float64']).columns
    if log == True:
        num_features = df_temp.drop(columns=['Calories', 'log_calories']).select_dtypes(include=['int64', 'float64']).columns
    
    cat_features = ['Sex_female', 'Sex_male']

    num_only_int = []
    cat_num_int = []
    # numeric and numeric interactions
    for i in range(len(num_features)):
        for j in range(i+1, len(num_features)):
            df_temp[f"{num_features[i]}_{num_features[j]}"] = df_temp[num_features[i]] * df_temp[num_features[j]]
            num_only_int.append(f"{num_features[i]}_{num_features[j]}")

    # sex female and all numeric features
    for sex in cat_features:
        for i in range(len(num_features)):
            df_temp[f"{sex}_{num_features[i]}"] = df_temp[sex] * df_temp[num_features[i]]
            cat_num_int.append(f"{sex}_{num_features[i]}")

    return df_temp, num_only_int, cat_num_int


df_train2, num_only_int, cat_num_int = make_interactions(df_train2)
df_train2.head()


df_train2.info()


fig, axes = plt.subplots(8, 2, figsize=(8, 16))
axes = axes.flatten()

for i, feature in enumerate(num_only_int):
    # sns.scatterplot(data=df_train2, x=feature, y='Calories', ax=axes[i])
    sns.regplot(data=df_train2, x=feature, y='Calories', lowess=True, ax=axes[i], line_kws={"color": "red"})
    axes[i].set_title(f"{feature}")

axes[15].remove()
plt.tight_layout()
plt.show()


# correlation matrix for num only int
num_only_int2 = num_only_int + ['Calories']
corr_matrix = df_train2[num_only_int2].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=1)
plt.title("Correlations for Num Num Interactions")


from itertools import combinations

def bmi_to_weighttype(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi <= 24.9:
        return "NormalWeight"
    elif bmi <= 29.9:
        return "Overweight"
    else:
        return "Obesity"

def age_to_group(age):
    if age <= 18:
        return "Child"
    elif age <= 30:
        return "Young Adult"
    elif age <= 50:
        return "Adult"
    else:
        return "Senior"

def make_pairs(df):
    df_temp = df.copy()
    encode_columns = ['Sex', 'WeightType', 'AgeGroup']

    df_temp["BMI"] = df_temp['Weight'] / (df_temp['Height']/100)**2
    df_temp["WeightType"] = df_temp["BMI"].apply(bmi_to_weighttype).astype("category")
    df_temp["AgeGroup"] = df_temp["Age"].apply(age_to_group).astype("category")
    
    pair_size = [2,3]
    
    for r in pair_size:
        for cols in list(combinations(encode_columns, r)):
            new_col_name = '_'.join(cols)
            
            df_temp[new_col_name] = df_temp[list(cols)].astype(str).agg('_'.join, axis=1)
            df_temp[new_col_name] = df_temp[new_col_name].astype('category')

    return df_temp


df_train3 = make_pairs(df_train)
df_train3.head()


cat_features = df_train3.select_dtypes(include=['category']).columns

fig, axes = plt.subplots(3, 2, figsize=(12, 16))
axes = axes.flatten()

for i, feature in enumerate(cat_features):
    sns.boxplot(x=feature, y='Calories', data=df_train3, ax=axes[i])
    axes[i].set_title(f'Relationship Between {feature} and Calories')
    axes[i].tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()


# Calories minute
def make_cal_min(df):
    df_temp = df.copy()
    df_temp['Calories_Minute'] = df_temp['Calories'] / df_temp['Duration']
    return df_temp

def make_bmr(df):
    df_temp = df.copy()
    df_temp['BMR']=0
    df_temp.loc[df_temp.Sex=='male','BMR'] = df_temp['Weight'] * 9.65 + (df_temp['Height'] / 100) * 573 - df_temp['Age'] * 5.08 + 260
    df_temp.loc[df_temp.Sex=='female','BMR'] = df_temp['Weight'] * 7.38 + (df_temp['Height'] / 100) * 607 - df_temp['Age'] * 2.31 + 43
    return df_temp


df_train3['Sex'] = df_train3['Sex'].astype('category')
df_train4 = make_cal_min(df_train3)
df_train4 = make_bmr(df_train4)


# distribution of Calories_Minute and BMR
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(data=df_train4, x='Calories_Minute', ax=axes[0][0])
sns.kdeplot(data=df_train4, x='Calories_Minute', ax=axes[0][1], hue='Sex', fill=True, alpha=0.3)
sns.histplot(data=df_train4, x='BMR', ax=axes[1][0], bins=30)
sns.kdeplot(data=df_train4, x='BMR', ax=axes[1][1], hue='Sex', fill=True, alpha=0.3)

plt.tight_layout()
plt.show()


# Calories Minutes Heavily Skewed, remove outliers
df_train4 = df_train4.loc[df_train4['Calories_Minute'] < 11]


# relationship between features and Calories_Minute
num_features = ['Age', 'Height', "Weight", "Duration", "Heart_Rate", "Body_Temp", "BMI", "BMR"]

fig, axes = plt.subplots(4, 2, figsize=(12, 12))
axes = axes.flatten()

for i, feature in enumerate(num_features):
    # sns.scatterplot(data=df_train4, x=feature, y='Calories_Minute', ax=axes[i])
    sns.regplot(data=df_train4, x=feature, y='Calories_Minute', lowess=True, ax=axes[i], line_kws={"color": "red"})
    axes[i].set_title(f'{feature} and Calories_Minute')

plt.tight_layout()
plt.show()


# correlation matrix for stuff
num_features2 = num_features + ['Calories_Minute', 'Calories']
corr_matrix = df_train4[num_features2].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlations with Calories_Minute and BMR")
plt.show()

