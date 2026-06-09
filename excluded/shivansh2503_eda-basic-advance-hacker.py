# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df.info()


# Check for duplicates
# Remember to drop id column while checking duplicate as id is always unique which can create hinderance as for example you can see below

print("Duplicates before dropping id column: ",df.duplicated().sum())
print("Duplicates after dropping id column: ",df.drop('id', axis=1).duplicated().sum())


# Hence dropping id globally as it is just for numbering
ids = df['id']
df = df.drop('id', axis=1)


df.drop_duplicates(inplace=True)
df.info()


# Checking for nans
df.isna().sum()


# Seperating num and categorical columns
num_cols = [col for col in df.columns if df[col].dtype != 'object' and col != "Calories"]
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
target_col = df['Calories']


#------------------------------------------------------------------
# 1. Distribution: Is Calories normally distributed, skewed, or multimodal?
#------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Histogram + KDE plot
plt.subplot(1, 2, 1)
sns.histplot(target_col, kde=True, bins=30)
plt.title('Distribution of Calories')
plt.show()


# Q-Q plot (to check normality)
plt.subplot(1, 2, 2)
import scipy.stats as stats
stats.probplot(target_col, plot=plt)
plt.title('Q-Q Plot of Calories')

plt.tight_layout()
plt.show()


# Skewness and Kurtosis (quantitative measures)
print(f"Skewness: {df['Calories'].skew():.2f}")
print(f"Kurtosis: {df['Calories'].kurtosis():.2f}")


#------------------------------------------------------------------
# 2. Outliers: Are there extreme values needing treatment?
#------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x=target_col)
plt.title('Boxplot of Calories (Outlier Detection)')
plt.show()


# IQR Method to detect outliers
Q1 = df['Calories'].quantile(0.25)
Q3 = df['Calories'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Calories'] < lower_bound) | (df['Calories'] > upper_bound)]
print(f"Number of outliers (IQR method): {len(outliers)}")


outliers['Calories'].value_counts()


outliers['Sex'].value_counts()


# Plot outliers vs. non-outliers
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df.index, y='Calories', data=df, hue=df['Calories'].apply(
    lambda x: 'Outlier' if (x < lower_bound) | (x > upper_bound) else 'Normal'
), alpha=0.6)
plt.title('Outliers in Calories')
plt.show()


# Highlight outliers in the scatterplot with respect to Duration
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Duration', 
    y='Calories', 
    data=df,
    hue=df['Calories'].apply(
        lambda x: 'Outlier' if (x < lower_bound) | (x > upper_bound) else 'Normal'
    ),
    palette={'Normal': 'blue', 'Outlier': 'red'},
    alpha=0.6
)
plt.title('Duration vs. Calories (Outliers Highlighted)')
plt.legend(title='Data Type')
plt.show()


df


# Separate outliers and normal data
outlier = df[(df['Calories'] < lower_bound) | (df['Calories'] > upper_bound)]
normal_data = df[(df['Calories'] >= lower_bound) & (df['Calories'] <= upper_bound)]

# Summary stats for Duration in both groups
print("--- Outliers ---")
print(outliers['Duration'].describe())
print("\n--- Normal Data ---")
print(normal_data['Duration'].describe())

# Boxplot comparison
plt.figure(figsize=(8, 5))
sns.boxplot(data=df,x=df['Calories'].apply(
    lambda x: 'Outlier' if (x < lower_bound) | (x > upper_bound) else 'Normal'
), y='Duration')
plt.title('Duration Distribution: Outliers vs. Normal Data')
plt.show()


#Plot with regression lines for duration vs calories
plt.figure(figsize=(10, 6))
sns.regplot(x='Duration', y='Calories', data=normal_data, label='Normal Data', scatter_kws={'alpha':0.3})
sns.regplot(x='Duration', y='Calories', data=outliers, label='Outliers', scatter_kws={'alpha':0.3}, color='red')
plt.title('Regression: Duration vs. Calories (Outliers vs. Normal)')
plt.legend()
plt.show()


#------------------------------------------------------------------
# 3. Relationship with Features:
#    - Correlation with numerical features
#    - Relationship with categorical (Sex)
#------------------------------------------------------------------
# Correlation with numerical features
correlation_with_target = df[num_cols + ['Calories']].corr()['Calories'].sort_values(ascending=False)
print("\nCorrelation with Numerical Features:")
print(correlation_with_target.drop('Calories'))

# Visualize top correlations
top_correlated = correlation_with_target.drop('Calories').nlargest(3).index
for feature in top_correlated:
    sns.lmplot(x=feature, y='Calories', data=df, scatter_kws={'alpha':0.3})
    plt.title(f'Calories vs {feature}')
    plt.show()


# Relationship with categorical feature (Sex)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='Calories', data=df)
plt.title('Calories by Sex')
plt.show()


# Statistical test (t-test) to check if difference is significant

male_cals = df[df['Sex'] == 'male']['Calories']
female_cals = df[df['Sex'] == 'female']['Calories']
t_stat, p_val = ttest_ind(male_cals, female_cals)
print(f"\nT-test between Sex groups: p-value = {p_val:.4f}")
print("Tstat ", t_stat)


#Numerical distribution
# Set up subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(num_cols):
    # Histogram + KDE
    sns.histplot(df[feature], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f'{feature} Distribution')
    
    # Skewness & Kurtosis
    skew = df[feature].skew()
    kurt = df[feature].kurtosis()
    axes[i].annotate(f'Skew: {skew:.2f}\nKurt: {kurt:.2f}', 
                    xy=(0.7, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', alpha=0.8))

# Adjust layout
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
sns.boxplot(data=df[num_cols], orient='h')
plt.title('Outlier Detection (Numerical Features)')
plt.show()


# Class balance
sex_counts = df['Sex'].value_counts(normalize=True)
print("Sex Distribution (%):\n", sex_counts * 100)

# Impact on Calories
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='Calories', data=df)
plt.title('Calories by Sex')
plt.show()


# Generate BMI

#since height is in cms, therefore converting into meters for BMI
df['BMI'] = df['Weight'] / ((df['Height']/100)**2)

# Check distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['BMI'], kde=True, bins=30)
plt.title('BMI Distribution (After Creation)')
plt.show()

print(f"BMI Skewness: {df['BMI'].skew():.2f}, Kurtosis: {df['BMI'].kurtosis():.2f}")


plt.figure(figsize=(10, 4))
sns.histplot(df['BMI'], kde=True, bins=30)
plt.axvline(18.5, color='red', linestyle='--', label='Underweight')
plt.axvline(24.9, color='green', linestyle='--', label='Healthy')
plt.axvline(29.9, color='orange', linestyle='--', label='Overweight')
plt.legend()
plt.title('BMI Distribution (Corrected)')
plt.show()


print(df['BMI'].describe())


print(df[['BMI', 'Calories']].corr())


# Outlier handeling in BMI

# NOOB and Advance
# # We will keep the obese one but will clip the lower value BMI outlier as it is totally impractical

# df['BMI'] = df['BMI'].clip(lower=18.5)


## HACKER
df_1 = df.copy()
df_1['Sex'] = df_1['Sex'].map({"male":0, "female":1})
# Count extreme low BMI values
low_bmi = df_1[df_1['BMI'] < 18.5]  # 18.5 is the standard value by WHO. Whose BMI< 18.5 is considered as unhealthy in real world
print(f"Rows with BMI < 18.5: {len(low_bmi)} ({len(low_bmi)/len(df)*100:.2f}%)")

# Inspect their metadata (e.g., are they all from a specific age/sex group?)
print(low_bmi[['Age',"Sex", 'Weight', 'Height']].describe())


# Calculate minimum plausible weight for height to reach BMI 18.5
low_bmi['Min_Healthy_Weight'] = 18.5 * (low_bmi['Height']/100)**2
print(low_bmi[['Weight', 'Min_Healthy_Weight', "Height"]].head())


low_bmi["Sex"].value_counts()


df['BMI'] = df['BMI'].clip(lower=18.5)  # Cap underweight values
print(f"Post-clipping BMI range: {df['BMI'].min()}â€“{df['BMI'].max()}")


sns.histplot(df['BMI'], bins=30, kde=True)
plt.axvline(18.5, color='red', linestyle='--', label='Underweight Threshold')
plt.show()
print(f"BMI Skewness: {df['BMI'].skew():.2f}, Kurtosis: {df['BMI'].kurtosis():.2f}")



print(df[['BMI', 'Calories']].corr())


df['BMI_Duration'] = df['BMI'] * df['Duration']


# Check distribution
plt.figure(figsize=(10, 4))
sns.histplot(df['BMI_Duration'], kde=True, bins=30)
plt.title('BMI Distribution (After Creation)')
plt.show()

print(f"BMI_Duration Skewness: {df['BMI_Duration'].skew():.2f}, Kurtosis: {df['BMI_Duration'].kurtosis():.2f}")


print(df[['BMI_Duration', 'Calories']].corr())


df['BMI_Tertile'] = pd.qcut(df['BMI'], q=3, labels=['Low BMI', 'Medium BMI', 'High BMI'])

sns.lmplot(
    x='Duration', 
    y='Calories', 
    hue='BMI_Tertile',  # Use the pre-binned column
    data=df, 
    scatter_kws={'alpha': 0.1},
    line_kws={'lw': 1}
)
plt.title('Calories vs. Duration by BMI Group')
plt.show()


#------------------------> Checking non linear relationship before final verdict

df['BMI_Tertile'] = pd.qcut(df['BMI'], q=3, labels=['Low', 'Mid', 'High'])
sns.boxplot(x='BMI_Tertile', y='Calories', data=df)
plt.show()
df.drop("BMI_Tertile", axis=1, inplace = True)


df.drop(["BMI", "BMI_Duration"], axis=1, inplace=True)


# Bin Duration
df['Duration_Binned'] = pd.cut(
    df['Duration'],
    bins=[0, 15, 30],
    labels=['Short (1-15 mins)', 'Long (15-30 mins)']
)

# Check balance
print("\nDuration Bins Counts:")
print(df['Duration_Binned'].value_counts())

# Compare calorie burn across bins
plt.figure(figsize=(8, 5))
sns.boxplot(x='Duration_Binned', y='Calories', data=df)
plt.title('Calories by Duration Bins')
plt.show()


# Lets analyze this feature
calories_by_duration = df.groupby('Duration_Binned')['Calories'].describe()
print(calories_by_duration)


# Analyzing outliers

def outlier_summary(df, bin_col, target_col='Calories'):
    bins = df[bin_col].unique()
    for bin_name in bins:
        subset = df[df[bin_col] == bin_name][target_col]
        Q1 = subset.quantile(0.25)
        Q3 = subset.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
        print(f"\n{bin_name} Outliers:")
        print(f" - Count: {len(outliers)}")
        print(f" - % of Total: {len(outliers)/len(subset)*100:.2f}%")
        print(f" - Top 5 Extreme Values:\n{outliers.sort_values().tail(5)}")

outlier_summary(df, 'Duration_Binned')


# -------------------> Clipping short outliers as they are impractical

# Calculate IQR bounds for short duration
short_mask = df['Duration_Binned'] == 'Short (1-15 mins)'
Q1_short, Q3_short = df.loc[short_mask, 'Calories'].quantile([0.25, 0.75])
IQR_short = Q3_short - Q1_short
upper_bound_short = Q3_short + 3 * IQR_short  # Using 3xIQR for more conservative clipping

print(upper_bound_short)
# Clip values
df.loc[short_mask & (df['Calories'] > upper_bound_short), 'Calories'] = upper_bound_short
print(f"Clipped {sum(short_mask & (df['Calories'] > upper_bound_short))} short-duration outliers")


plt.figure(figsize=(10, 6))
sns.boxplot(x='Duration_Binned', y='Calories', data=df, showfliers=True)  # Set `showfliers=False` to hide outliers
plt.title('Calories by Duration Bin (With Outliers)')                                               
plt.show()


# Verify new distributions
print("\nAfter outlier treatment:")
print(df.groupby('Duration_Binned')['Calories'].describe()[['count', 'mean', 'std', 'min', 'max']])

# Visual confirmation
plt.figure(figsize=(10,5))
sns.boxplot(x='Duration_Binned', y='Calories', data=df, showfliers=False)
plt.title('Calories by Duration (Outliers Processed)')
plt.show()


# Verify final distribution
print(df[short_mask]['Calories'].describe())
sns.histplot(df[short_mask]['Calories'], bins=30)
plt.axvline(169, color='r', linestyle='--', label='Clipping Threshold')
plt.title('Short-Duration Calories After Capping')
plt.show()


# Clip HR to 40â€“200 bpm
df['Heart_Rate_Clipped'] = df['Heart_Rate'].clip(lower=40, upper=200)

# Compare before/after
print(f"Original HR Range: {df['Heart_Rate'].min()}â€“{df['Heart_Rate'].max()}")
print(f"Clipped HR Range: {df['Heart_Rate_Clipped'].min()}â€“{df['Heart_Rate_Clipped'].max()}")

# Plot impact
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['Heart_Rate'], bins=30)
plt.title('Original HR')
plt.subplot(1, 2, 2)
sns.histplot(df['Heart_Rate_Clipped'], bins=30)
plt.title('Clipped HR')
plt.tight_layout()
plt.show()


df.drop("Heart_Rate_Clipped", axis=1, inplace=True)


# Calculate % of max heart rate (assuming average age ~30)
df['HR_%Max'] = df['Heart_Rate'] / (190 - df['Age'])



print(df[['Heart_Rate', 'HR_%Max']].describe())


new_bins = [0, 0.55, 0.65, 0.75, 0.85, 0.95, np.inf]
new_labels = [
    'Resting (<55%)', 
    'Light (55-65%)', 
    'Moderate (65-75%)', 
    'Vigorous (75-85%)', 
    'High (85-95%)', 
    'Max (>95%)'
]


# Apply bins to HeartRate
df['HR_Zone'] = pd.cut(
    df['HR_%Max'],
    bins=new_bins,
    labels=new_labels
)

# Verify distribution
zone_distro = df['HR_Zone'].value_counts(normalize=True).sort_index()
print(zone_distro)


plt.figure(figsize=(12,6))
sns.boxplot(
    x='HR_Zone', 
    y='Calories', 
    data=df,
    order=new_labels,
    showfliers=False
)
plt.title('Calorie Distribution Across Optimized HR Zones')
plt.xticks(rotation=45)
plt.show()


# Mean calories per zone
print(df.groupby('HR_Zone')['Calories'].mean().sort_values(ascending=False))


import statsmodels.api as sm
model = sm.OLS(df['Calories'], pd.get_dummies(df['HR_Zone']))
print(model.fit().summary())


df['Is_Vigorous+'] = df['HR_Zone'].isin([
    'Vigorous (75-85%)', 
    'High (85-95%)', 
    'Max (>95%)'
]).astype(int)


df.info()


num_features = ['Duration', 'Heart_Rate', 'Body_Temp', 'HR_%Max']
for col in num_features:
    plt.figure(figsize=(10, 4))
    sns.regplot(x=df[col], y=df['Calories'], scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
    plt.title(f'Calories vs {col}')
    plt.show()

# What to Look For:

# Non-linearity: Curves in scatter plots â†’ Needs polynomial features.

# Heteroscedasticity: Fan-shaped residuals â†’ Consider log-transform of target.




# Add quadratic term to remove non linearity for linear models
df['Body_Temp_squared'] = df['Body_Temp']**2


# --------------------> Ordinal Encoding for HR_Zone
zone_order = ['Resting (<55%)', 'Light (55-65%)', 'Moderate (65-75%)', 'Vigorous (75-85%)', 'High (85-95%)', 'Max (>95%)']
df['HR_Zone_Ordinal'] = df['HR_Zone'].map({zone: i for i, zone in enumerate(zone_order)})


df["HR_Zone_Ordinal"] = df["HR_Zone_Ordinal"].astype(int)


# 1. Duration x HR_Zone (ordinal)
df['Duration_x_HR_Zone'] = df['Duration'] * df['HR_Zone_Ordinal']

# 2. Body_Temp x Is_Vigorous+
df['BodyTemp_x_Vigorous'] = df['Body_Temp'] * df['Is_Vigorous+']


# Check correlation with target
print(df[['Duration', 'Duration_x_HR_Zone', 'Calories']].corr()['Calories'])


sns.lmplot(x='Body_Temp', y='Calories', hue='Is_Vigorous+', data=df.sample(5000))
plt.title('Body Temp vs Calories by Vigorous+')
plt.show()


df.rename(columns={'Is_Vigorous+': 'Is_Vigorous_plus'}, inplace=True)


import statsmodels.formula.api as smf

model = smf.ols('Calories ~ Body_Temp + Is_Vigorous_plus + BodyTemp_x_Vigorous', data=df).fit()
print(model.summary())


# Duration x HR_Zone
plt.figure(figsize=(12, 6))
sns.lmplot(x='Duration', y='Calories', hue='HR_Zone', data=df.sample(5000), 
           scatter_kws={'alpha':0.2}, height=6, aspect=1.5)
plt.title('Calories vs Duration by HR Zone')
plt.show()

# What to Look For:

# Diverging Slopes: Different calorie rates per zone at same duration.


# Dropping 
df.drop(columns=['Duration_x_HR_Zone'], inplace=True)


cat_features = ['Sex', 'Duration_Binned', 'HR_Zone']
for col in cat_features:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col], y=df['Calories'])
    plt.title(f'Calories by {col}')
    plt.xticks(rotation=45)
    plt.show()

# What to Look For:

# Group Differences: Clear separation between categories (e.g., Male vs Female).

# Outliers: Extreme values within categories.


from scipy.stats import f_oneway
groups = df.groupby('HR_Zone')['Calories'].apply(list)
f_stat, p_val = f_oneway(*groups)
print(f"HR_Zone ANOVA: F={f_stat:.1f}, p={p_val:.3f}")


top_features = ["Age", "Height", 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'HR_%Max', 'Calories']
sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Top Features)')
plt.show()




