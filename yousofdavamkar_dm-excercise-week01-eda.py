import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


!pip install --upgrade numba pandas visions ydata_profiling ipywidgets


df.info()


from ydata_profiling import ProfileReport

type_schema = {
    "RefId": "Numeric",
    "IsBadBuy": "Boolean",
    "PurchDate": "DateTime",

    "Auction": "Categorical",

    "VehYear": "Numeric",
    "VehicleAge": "Numeric",
    "VehOdo": "Numeric",

    "Make": "Categorical",
    "Model": "Categorical",
    "Trim": "Categorical",
    "SubModel": "Categorical",
    "Color": "Categorical",

    "Transmission": "Categorical",
    "WheelTypeID": "Categorical",  
    "WheelType": "Categorical",

    "Nationality": "Categorical",
    "Size": "Categorical",
    "TopThreeAmericanName": "Categorical",

    "MMRAcquisitionAuctionAveragePrice": "Numeric",
    "MMRAcquisitionAuctionCleanPrice": "Numeric",
    "MMRAcquisitionRetailAveragePrice": "Numeric",
    "MMRAcquisitonRetailCleanPrice": "Numeric",
    "MMRCurrentAuctionAveragePrice": "Numeric",
    "MMRCurrentAuctionCleanPrice": "Numeric",
    "MMRCurrentRetailAveragePrice": "Numeric",
    "MMRCurrentRetailCleanPrice": "Numeric",

    "PRIMEUNIT": "Categorical",
    "AUCGUART": "Categorical",

    "BYRNO": "Categorical",
    "VNZIP1": "Categorical",
    "VNST": "Categorical",

    "VehBCost": "Numeric",
    "IsOnlineSale": "Boolean",
    "WarrantyCost": "Numeric",
}


profile = ProfileReport(
    df,
    title="Carvana Report with an Optimized Schema",
    type_schema=type_schema 
)

profile.to_file("ydata_profiling_report.html")


df_good_buy = df[df.IsBadBuy == 0]
df_bad_buy = df[df.IsBadBuy == 1]

profile_good = ProfileReport(
    df_good_buy,
    title="Good Buys (IsBadBuy = 0)",
    type_schema=type_schema 
)

profile_bad = ProfileReport(
    df_bad_buy,
    title="Bad Buys (IsBadBuy = 1)",
    type_schema=type_schema 
)

comparison_report = profile_good.compare(profile_bad)

comparison_report.to_file("carvana_comparison_report.html")


continuous_fields = [
    "VehYear",
    "VehicleAge",
    "VehOdo",
    "MMRAcquisitionAuctionAveragePrice",
    "MMRAcquisitionAuctionCleanPrice",
    "MMRAcquisitionRetailAveragePrice",
    "MMRAcquisitonRetailCleanPrice", 
    "MMRCurrentAuctionAveragePrice",
    "MMRCurrentAuctionCleanPrice",
    "MMRCurrentRetailAveragePrice",
    "MMRCurrentRetailCleanPrice",
    "VehBCost",
    "WarrantyCost"
]

summary_stats = df[continuous_fields].describe()
print("Summary statistics for continuous fields based on your schema:")
print(summary_stats)


import matplotlib.pyplot as plt
import seaborn as sns



df.dropna(subset=continuous_fields, inplace=True)
fig, axes = plt.subplots(nrows=len(continuous_fields), ncols=2, figsize=(12, 30))

fig.suptitle('Distribution and Outlier Analysis of Continuous Fields', fontsize=16, y=1.02)

for i, col in enumerate(continuous_fields):

    sns.histplot(df[col], kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {col}')
    axes[i, 0].set_xlabel('')  
    axes[i, 0].set_ylabel('') 

    sns.boxplot(x=df[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Box Plot of {col}')
    axes[i, 1].set_xlabel('')


plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

print("Distribution analysis plots have been saved to 'distribution_analysis.png'")

print("Outlier Analysis using the IQR Method:")
for col in continuous_fields:
    data = df[col].dropna()


    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(data)) * 100

    print(f"\n--- Field: {col} ---")
    print(f"Lower Bound for Outliers: {lower_bound:.2f}")
    print(f"Upper Bound for Outliers: {upper_bound:.2f}")
    print(f"Number of Outliers Found: {outlier_count}")
    print(f"Percentage of Data that are Outliers: {outlier_percentage:.2f}%")


import scipy.stats as stats

print("Normality Assessment using D'Agostino's K-squared test:")

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 25))
axes = axes.flatten() # Flatten the grid for easy looping
fig.suptitle('Q-Q Plots for Normality Assessment', fontsize=16, y=1.02)

for i, col in enumerate(continuous_fields):
    data_sample = df[col].sample(n=5000, random_state=1)
    k2, p_value = stats.normaltest(data_sample)
    print(f"\n--- Field: {col} ---")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: The data is likely NOT normally distributed.")
    else:
        print("Result: The data may be normally distributed.")

    stats.probplot(df[col], dist="norm", plot=axes[i])
    axes[i].set_title(f'Q-Q Plot of {col}')

plt.tight_layout(rect=[0, 0, 1, 1])

if len(continuous_fields) < len(axes):
    for j in range(len(continuous_fields), len(axes)):
        fig.delaxes(axes[j])
        
plt.show()

print("\nQ-Q plots have been saved to 'qq_plots_normality_corrected.png'")


categorical_fields = [key for key, value in type_schema.items() if value == 'Categorical']

for col in categorical_fields:
    distribution = df[col].value_counts(normalize=True) * 100
    rare_categories = distribution[distribution < 1]
    if not rare_categories.empty:
        print(f"\n--- Field: {col} ---")
        print("Rare Categories Found:")
        print(rare_categories)


from scipy.stats import mannwhitneyu, chi2_contingency

df.dropna(subset=continuous_fields + categorical_fields, inplace=True)

# --- 1. Analysis of Continuous Fields ---
print("--- Analysis of Continuous Fields vs. IsBadBuy ---")
fig_cont, axes_cont = plt.subplots(nrows=len(continuous_fields), ncols=2, figsize=(14, 40))
fig_cont.suptitle('Distribution of Continuous Fields by Target Class (IsBadBuy)', fontsize=16, y=1.01)

for i, col in enumerate(continuous_fields):
    sns.kdeplot(data=df, x=col, hue='IsBadBuy', fill=True, ax=axes_cont[i, 0], palette=['#3498db', '#e74c3c'])
    axes_cont[i, 0].set_title(f'{col} Distribution')
    sns.boxplot(data=df, x='IsBadBuy', y=col, ax=axes_cont[i, 1], palette=['#3498db', '#e74c3c'])
    axes_cont[i, 1].set_title(f'{col} by IsBadBuy')
    group_good_buy = df[df['IsBadBuy'] == 0][col]
    group_bad_buy = df[df['IsBadBuy'] == 1][col]
    
    stat, p_value = mannwhitneyu(group_good_buy, group_bad_buy)
    print(f"\nStatistical test for '{col}': p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("  -> The difference between Good Buys and Bad Buys is statistically significant.")
    else:
        print("  -> The difference is NOT statistically significant.")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('continuous_distributions_by_target.png')
print("\nPlots for continuous fields saved to 'continuous_distributions_by_target.png'")


# --- 2. Analysis of Categorical Fields ---
print("\n\n--- Analysis of Categorical Fields vs. IsBadBuy ---")

n_cats = len(categorical_fields)
n_cols_cat = 2
n_rows_cat = (n_cats + n_cols_cat - 1) // n_cols_cat

fig_cat, axes_cat = plt.subplots(nrows=n_rows_cat, ncols=n_cols_cat, figsize=(14, n_rows_cat * 5))
axes_cat = axes_cat.flatten()
fig_cat.suptitle('Distribution of Categorical Fields by Target Class (IsBadBuy)', fontsize=16, y=1.01)

for i, col in enumerate(categorical_fields):
    contingency_table = pd.crosstab(df[col], df['IsBadBuy'])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    print(f"\nStatistical test for '{col}': p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("  -> There is a significant association with IsBadBuy.")
    else:
        print("  -> There is NO significant association with IsBadBuy.")
        
    proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    proportions.plot(kind='bar', stacked=True, ax=axes_cat[i], color=['#3498db', '#e74c3c'])
    axes_cat[i].set_title(f'Proportion of IsBadBuy by {col}')
    axes_cat[i].set_ylabel('Proportion')
    axes_cat[i].legend(title='IsBadBuy', labels=['Good Buy (0)', 'Bad Buy (1)'])

if n_cats < len(axes_cat):
    for j in range(n_cats, len(axes_cat)):
        fig_cat.delaxes(axes_cat[j])

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()
print("\nPlots for categorical fields saved to 'categorical_distributions_by_target.png'")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


continuous_df = df[continuous_fields]
correlation_matrix = continuous_df.corr()


plt.figure(figsize=(12, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,         
    cmap='coolwarm',     
    fmt='.2f',           
    linewidths=.5,     
    vmin=-1, vmax=1      
)

plt.title('Correlation Matrix of Continuous Fields', fontsize=16)
plt.xticks(rotation=45, ha='right') 
plt.yticks(rotation=0)
plt.tight_layout() #

plt.show()

print("Correlation heatmap has been saved to 'correlation_heatmap.png'")

