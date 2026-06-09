import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import zscore
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, levene, bartlett, shapiro, skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression


def feature_engineering(df):
    df = df.copy()
    df['Height_m'] = df['Height'] / 100
    df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
    df['Heart_Rate_ratio'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Cardio_Load'] = df['Duration'] * df['Heart_Rate']
    df['Temp_Heart_ratio'] = df['Body_Temp'] / df['Heart_Rate']
    df['Weight_per_cm'] = df['Weight'] / df['Height']
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=False)
    df['Heart_Rate_Duration_interaction'] = df['Heart_Rate'] * df['Duration']
    df['Body_Temp_Duration_interaction'] = df['Body_Temp'] * df['Duration']
    df['Log_Weight'] = np.log(df['Weight'])
    df['Log_Height'] = np.log(df['Height'])
    df['Log_Cardio_Load'] = np.log(df['Cardio_Load'])
    df['Log_BMI'] = np.log(df['BMI'])
    df['Heart_Rate_ratio_exp'] = np.exp(df['Heart_Rate_ratio'])
    df['Duration_sq'] = df['Duration'] ** 2
    df['Duration_cube'] = df['Duration'] ** 3
    df.drop(columns=['Height_m'], inplace=True)
    return df



test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')

sex_map = {'male': 1, 'female': 0}
test_df['Sex'] = test_df['Sex'].map(sex_map)

test_df = feature_engineering(test_df)


stats = test_df.describe().T
stats['mode'] = test_df.mode().iloc[0]
stats = stats[['mean', 'mode', 'std', 'min', 'max']]
print(stats)


num_cols = test_df.select_dtypes(include=np.number).columns.drop('id')

plt.figure(figsize=(16, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(6, 4, i)
    sns.histplot(test_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(6, 4, i)
    sns.boxplot(x=test_df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 12))
for i, col in enumerate(num_cols, 1):
    plt.subplot(6, 4, i)
    sns.boxplot(x='Sex', y=col, data=test_df)
    plt.title(f'Boxplot of {col} by Sex')
plt.tight_layout()
plt.show()


bins = [0, 18.5, 24.9, 29.9, 40]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
test_df['BMI_category'] = pd.cut(test_df['BMI'], bins=bins, labels=labels)

plt.figure(figsize=(8, 8))
test_df['BMI_category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
plt.title('BMI Distribution')
plt.ylabel('')
plt.show()



numeric_cols = test_df.select_dtypes(include=[np.number]).columns

corr_pearson = test_df[numeric_cols].corr(method='pearson')
corr_kendall = test_df[numeric_cols].corr(method='kendall')
corr_spearman = test_df[numeric_cols].corr(method='spearman')

top_5_pearson = corr_pearson['Calories'].sort_values(ascending=False).iloc[1:6]
top_5_kendall = corr_kendall['Calories'].sort_values(ascending=False).iloc[1:6]
top_5_spearman = corr_spearman['Calories'].sort_values(ascending=False).iloc[1:6]

print('Top 5 Pearson correlations with Calories:')
print(top_5_pearson)
print('\nTop 5 Kendall correlations with Calories:')
print(top_5_kendall)
print('\nTop 5 Spearman correlations with Calories:')
print(top_5_spearman)

# Pearson
plt.figure(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Pearson Correlation Heatmap')
plt.show()

# Kendall
plt.figure(figsize=(10, 8))
sns.heatmap(corr_kendall, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Kendall Correlation Heatmap')
plt.show()

# Spearman
plt.figure(figsize=(10, 8))
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Spearman Correlation Heatmap')
plt.show()


male_data = test_df[test_df['Sex'] == 1]['Calories']
female_data = test_df[test_df['Sex'] == 0]['Calories']

f_stat, p_val = f_oneway(male_data, female_data)
print(f'ANOVA test p-value: {p_val}')

t_stat, p_val = ttest_ind(male_data, female_data)
print(f'T-test p-value: {p_val}')

u_stat, p_val = mannwhitneyu(male_data, female_data)
print(f'Mann-Whitney U test p-value: {p_val}')

stat, p_val = levene(male_data, female_data)
print(f'Levene test p-value: {p_val}')

stat, p_val = bartlett(male_data, female_data)
print(f'Bartlett test p-value: {p_val}')


signal = np.mean(test_df['Calories'])
noise = np.std(test_df['Calories'])
snr = signal / noise
print(f'SNR: {snr}')


for col in numeric_cols:
    stat, p_value = shapiro(test_df[col].dropna())
    print(f"Column: {col}")
    print(f"  Shapiro-Wilk Test p-value: {p_value:.4f}")
    print(f"  {'Normal' if p_value > 0.05 else 'Not Normal'}")
    print('-' * 50)


for col in numeric_cols:
    skew_value = skew(test_df[col].dropna())
    kurt_value = kurtosis(test_df[col].dropna())
    
    print(f"Column: {col}")
    print(f"  Skewness: {skew_value:.4f} {'(Normal)' if abs(skew_value) < 1 else '(Non-Normal)'}")
    print(f"  Kurtosis: {kurt_value:.4f} {'(Normal)' if abs(kurt_value - 3) < 1 else '(Non-Normal)'}")
    print('-' * 50)



import scipy.stats as stats

numerical_df = test_df.select_dtypes(include=[np.number])

z_scores = zscore(numerical_df)

outliers_zscore = (abs(z_scores) > 3).sum(axis=0)

for i, outlier_count in enumerate(outliers_zscore):
    col = numerical_df.columns[i]
    if outlier_count > 0:
        print(f"Column '{col}' contains {outlier_count} outliers.")
    else:
        print(f"Column '{col}' does not contain outliers.")

# QQ plots
plt.figure(figsize=(16, 12))

for i, col in enumerate(numerical_df.columns, 1):
    plt.subplot(6, 4, i)
    stats.probplot(numerical_df[col].dropna().to_numpy(), dist="norm", plot=plt) 
    plt.title(f'QQ plot of {col}')

plt.tight_layout()
plt.show()


numerical_df_with_const = add_constant(numerical_df)

vif_data = pd.DataFrame()
vif_data["Variable"] = numerical_df_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(numerical_df_with_const.values, i) for i in range(len(numerical_df_with_const.columns))]

print(vif_data)


scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_df)

pca = PCA()
pca.fit(numerical_scaled)

plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.title('Explained Variance by Each Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.show()

print(f"Total explained variance: {sum(pca.explained_variance_ratio_)}")



target_variable = 'Calories'

model = LinearRegression()

rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(numerical_df, numerical_df[target_variable])

selected_features = numerical_df.columns[rfe.support_]
print(f"Selected features: {selected_features}")


X = numerical_df.drop(columns=[target_variable])  
y = numerical_df[target_variable]

mutual_info = mutual_info_regression(X, y)

mutual_info_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mutual_info})

mutual_info_df = mutual_info_df.sort_values(by='Mutual Information', ascending=False)

print(mutual_info_df.head(5))

plt.figure(figsize=(10, 6))
plt.barh(mutual_info_df['Feature'], mutual_info_df['Mutual Information'], color='skyblue')
plt.xlabel('Mutual Information')
plt.title('Top Features by Mutual Information with Target (Calories)')
plt.gca().invert_yaxis()  
plt.show()


