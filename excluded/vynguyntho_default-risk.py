import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder


application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
bureau_credit = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# Missing values statistics
missing_values = missing_values_table(application_train)
missing_values = missing_values.reset_index()
missing_values.rename(columns={'index': 'Column Name'}, inplace=True)
print('The first 10 columns having the largest missing value ratio:')
missing_values.head(10)


missing_threshold = 70
column_to_drop = missing_values[missing_values["% of Total Values"] > missing_threshold]["Column Name"]

df_train = application_train.drop(columns=column_to_drop)  # Drop second column

# Missing values statistics
missing_values = missing_values_table(df_train)
missing_values = missing_values.reset_index()
missing_values.rename(columns={'index': 'Column Name'}, inplace=True)
missing_values.head(20)


missing_values_PD_rate =[]

for i in missing_values["Column Name"]:
    #print(i)
    df_missing_rows = df_train[df_train[['TARGET', i]].isnull().any(axis=1)][['TARGET', i]]
    df_missing_rows_PD = df_missing_rows[df_missing_rows['TARGET'] == 1 ]
    df_missing_rows_NonPD = df_missing_rows[df_missing_rows['TARGET'] == 0 ]

    x1 = df_missing_rows_PD.shape[0]
    x2 = df_missing_rows_NonPD.shape[0]
    PD_rate = x1/(x1+x2)
    missing_values_PD_rate.append(PD_rate)
missing_values['PD_rate'] = missing_values_PD_rate

#comparision with PD rate of full portfolio
df_train_PD = df_train[df_train['TARGET'] == 1 ]
df_train_NonPD = df_train[df_train['TARGET'] == 0 ]

x11 = df_train_PD.shape[0]
x22 = df_train_NonPD.shape[0]
missing_values['PD_rate_total'] = x11/(x11+x22)

missing_values


missing_values[(missing_values['PD_rate'] > missing_values['PD_rate_total']) & (missing_values['% of Total Values'] >= 50)].head(10)


numerical_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f'Numbers of columns are {df_train.shape[1]}')
print(f'umbers of numerical columns are  = {len(numerical_columns)}')
print(f'umbers of categorical columns are  = {len(categorical_columns)}')


def df_grouped(dataframe, selected_column):
    def format_as_percentage(s: float) -> str:
        return f"{round(s*100, 2)}%"
    df_grouped = dataframe.groupby(selected_column, as_index=False).agg(
    Customer_Count=('SK_ID_CURR', 'count'),  
    Total_Loan_Amount=('AMT_CREDIT', 'sum'),  
    Ticket_size=('AMT_CREDIT', 'mean'),
    Default_Count=('TARGET', lambda x: (x == 1).sum()),  # Số lượng khách hàng bị vỡ nợ (PD = 1)
    Non_Default_Count=('TARGET', lambda x: (x == 0).sum())  # Số KH không vỡ nợ
    )

    # PD rate
    df_grouped['PD_Rate'] = df_grouped['Default_Count'] / (df_grouped['Default_Count'] + df_grouped['Non_Default_Count'])
    df_grouped = df_grouped.sort_values(by='PD_Rate')
    
    # proportion
    df_grouped['Customer_Percentage'] = df_grouped['Customer_Count'] / df_grouped['Customer_Count'].sum()
    df_grouped['Loan_Percentage'] = df_grouped['Total_Loan_Amount'] / df_grouped['Total_Loan_Amount'].sum()
    
    # rounding
    df_grouped['Total_Loan_Amount'] = df_grouped['Total_Loan_Amount']/(10**6)
    df_grouped['Ticket_size'] = df_grouped['Ticket_size']/(10**6)
    df_grouped['PD_Rate'] = df_grouped['PD_Rate'].map(format_as_percentage)
    df_grouped['Customer_Percentage'] = df_grouped['Customer_Percentage'].map(format_as_percentage)
    df_grouped['Loan_Percentage'] = df_grouped['Loan_Percentage'].map(format_as_percentage)
    
    return df_grouped[[selected_column, 'Customer_Count', 'Customer_Percentage', 'Total_Loan_Amount', 'Loan_Percentage', 'Ticket_size', 'PD_Rate']]


def plot_numerical_features( dataframe ,selected_column, bins, xlabel_rotation: int = 0):
    plt.figure(figsize=(8, 5))
    sns.histplot(dataframe[dataframe['TARGET'] == 0]['selected_column'], bins=bins, color='blue', label='Target 0', kde=True, alpha=0.6)
    sns.histplot(dataframe[dataframe['TARGET'] == 1]['selected_column'], bins=bins, color='red', label='Target 1', kde=True, alpha=0.6)
    
    
    plt.title(f"Feature Distribution by {selected_column}")
    plt.xlabel(selected_column)
    plt.ylabel("Count contracts")
    plt.xticks(rotation=xlabel_rotation)
    plt.legend()
    plt.show()


def plot_categorical_features(dataframe, selected_column, sum_value, formula: str, xlabel_rotation: int = 0):
    
    if formula == 'count':
        agg_data = dataframe[selected_column].value_counts(normalize=True) 
    elif formula == 'sum':
        agg_data = dataframe.groupby(selected_column)[sum_value].sum()
        agg_data = sum_table / sum_table.sum()  # Normalize
    else:
        raise Exception("Invalid formula")
    agg_data = agg_data.sort_index(ascending=True)
    #PD rate
    pd_per_category = dataframe.groupby(selected_column)['TARGET'].mean()  # PD (Target=1 proportion per category)
    pd_per_category = pd_per_category.sort_index(ascending=True)
    #plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #Proportion
    #formula.plot(x=selected_column, kind='bar', stacked=True)
    sns.barplot(x=agg_data.index, y=agg_data.values*100, ax=axes[0], palette="coolwarm")
    axes[0].set_title(f"Proportion of {selected_column}")
    axes[0].set_xlabel(selected_column)
    axes[0].set_ylabel("Proportion")
    axes[0].tick_params(axis='x', labelrotation=xlabel_rotation)

    for i, v in enumerate(agg_data.values):
        axes[0].text(i, v + 0.01, f"{v:.2%}", ha='center', fontsize=10)

    
    #PD rate
    sns.barplot(x=pd_per_category.index, y=pd_per_category.values*100, ax=axes[1], palette="coolwarm")
    axes[1].set_title(f"PD rate by {selected_column}")
    axes[1].set_xlabel(selected_column)
    axes[1].set_ylabel("Proportion")
    axes[1].tick_params(axis='x', labelrotation=xlabel_rotation)

    for i, v in enumerate(pd_per_category.values):
        axes[1].text(i, v + (pd_per_category.max() * 0.01), f"{v:.2%}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()


selected_column = 'NAME_CONTRACT_TYPE'

print(f'Summary information of {selected_column}')

display(df_grouped(df_train, selected_column))

if selected_column in numerical_columns:
    plot_numerical_features(df_train,selected_column, 20)
else:
    formula = 'count'
    plot_categorical_features(df_train, selected_column, 'AMT_CREDIT', formula)


df_train['AGE_BIRTH'] = df_train['DAYS_BIRTH'] * (-1) / 365
df_train['AGE_BIRTH'] = df_train['AGE_BIRTH'].astype(int)
df_train['AGE_GROUP'] = np.where(df_train['AGE_BIRTH'] < 20, '<25',
                         np.where(df_train['AGE_BIRTH'] < 30, '25-30', 
                         np.where(df_train['AGE_BIRTH'] < 40, '30-40',
                         np.where(df_train['AGE_BIRTH'] < 50, '40-50', 
                         np.where(df_train['AGE_BIRTH'] < 60, '50-60', '60+')))))

categorical_columns.append('AGE_GROUP')

#display
selected_column = 'AGE_GROUP'
print(f'Summary information of {selected_column}')

display(df_grouped(df_train, selected_column))

if selected_column in numerical_columns:
    plot_numerical_features(df_train,selected_column, 20)
else:
    formula = 'count'
    plot_categorical_features(df_train, selected_column, 'AMT_CREDIT', formula)


selected_column = 'NAME_EDUCATION_TYPE'
sum_column = "AMT_CREDIT"

#display

print(f'Summary information of {selected_column}')

display(df_grouped(df_train, selected_column))

if selected_column in numerical_columns:
    plot_numerical_features(df_train,selected_column, 20, 45)
else:
    formula = 'count'
    plot_categorical_features(df_train, selected_column, sum_column, formula, 45)


df_train['OCCUPATION_TYPE'].unique()


def occupation_by_level(job: str):
    if job in ["Laborers", "Low-skill Laborers", "Drivers", "Cleaning staff", "Cooking staff", "Waiters/barmen staff", "Security staff"]:
        return 'Low-Skill & Manual Jobs'
    elif job in ["Core staff", "Sales staff", "Private service staff", "Secretaries", "HR staff", "Realty agents"]:
        return 'Mid-Skill & Office Support Jobs'
    elif job in ["Accountants", "Managers", "Medicine staff", "High skill tech staff", "IT staff"]: 
        return 'High-Skill & Professional Jobs'
    else:
        return 'N/A'

df_train['OCCUPATION_TYPE_GROUP'] = df_train['OCCUPATION_TYPE'].map(occupation_by_level)
selected_column = 'OCCUPATION_TYPE_GROUP'
sum_column = "AMT_CREDIT"

categorical_columns.append('OCCUPATION_TYPE_GROUP')

#display
print(f'Summary information of {selected_column}')

display(df_grouped(df_train, selected_column))

if selected_column in numerical_columns:
    plot_numerical_features(df_train,selected_column, 20, 45)
else:
    formula = 'count'
    plot_categorical_features(df_train, selected_column, sum_column, formula, 45)



subset = ['OCCUPATION_TYPE_GROUP', 'REGION_RATING_CLIENT']
formula = 'Count'
xlabel_rotation = 45

#proportion
df_grouped = df_train.groupby(subset).size().reset_index(name=formula)
    
total_count_per_A = df_grouped.groupby(subset[0])[formula].transform('sum')
df_grouped['Proportion'] = df_grouped[formula] / total_count_per_A

df_pivot_proportion = df_grouped.pivot_table(index=subset[0], columns=subset[1], values='Proportion', aggfunc='sum', fill_value=0)

#PD rate
pd_per_category = df_train.groupby(subset)['TARGET'].mean().reset_index()

#plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

#sns.barplot(x=subset[0], y='Proportion', hue=subset[1], data=df_grouped, palette='Set2', ax=axes[0])
ax0 = df_pivot_proportion.plot(kind='bar', stacked=True, ax=axes[0], cmap='Set2')

for p in ax0.patches:
    height = p.get_height()
    if height > 0: 
        ax0.annotate(f'{height:.2%}', 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.), 
                     xytext=(0, 5), 
                     textcoords='offset points', 
                     ha='center', va='center')
    
axes[0].set_title(f'Distribution of {subset[1]} by {subset[0]}')
axes[0].set_xlabel(subset[0])
axes[0].set_ylabel(f'{formula} of {subset[1]}')
axes[0].tick_params(axis='x', labelrotation=xlabel_rotation)

sns.barplot(x=subset[0], y='TARGET', hue=subset[1], data=pd_per_category, palette='Set3', ax=axes[1])

for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2%}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     xytext=(0, 5), 
                     textcoords='offset points', 
                     ha='center', va='bottom')
    
axes[1].set_title(f'PD rate of {subset[1]} by {subset[0]}')
axes[1].set_xlabel(subset[0])
axes[1].tick_params(axis='x', labelrotation=xlabel_rotation)

plt.tight_layout()
plt.show()



subset = ['OCCUPATION_TYPE_GROUP', 'NAME_EDUCATION_TYPE']
formula = 'Count'
xlabel_rotation = 45

#proportion
df_grouped = df_train.groupby(subset).size().reset_index(name=formula)
    
total_count_per_A = df_grouped.groupby(subset[0])[formula].transform('sum')
df_grouped['Proportion'] = 100*df_grouped[formula] / total_count_per_A

df_pivot_proportion = df_grouped.pivot_table(index=subset[0], columns=subset[1], values='Proportion', aggfunc='sum', fill_value=0)

#PD rate
pd_per_category = df_train.groupby(subset)['TARGET'].mean().reset_index()

#plot
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#sns.barplot(x=subset[0], y='Proportion', hue=subset[1], data=df_grouped, palette='Set2', ax=axes[0])
ax0 = df_pivot_proportion.plot(kind='bar', stacked=True, ax=axes[0], cmap='Set2')

    
axes[0].set_title(f'Distribution of {subset[1]} by {subset[0]}')
axes[0].set_xlabel(subset[0])
axes[0].set_ylabel(f'{formula} of {subset[1]}')
axes[0].tick_params(axis='x', labelrotation=xlabel_rotation)

sns.barplot(x=subset[0], y='TARGET', hue=subset[1], data=pd_per_category, palette='Set3', ax=axes[1])

for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2f}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     xytext=(0, 5), 
                     textcoords='offset points', 
                     ha='center', va='bottom')
    
axes[1].set_title(f'PD rate of {subset[1]} by {subset[0]}')
axes[1].set_xlabel(subset[0])
axes[1].tick_params(axis='x', labelrotation=xlabel_rotation)

plt.tight_layout()
plt.show()


def plot_subset_numerical(dataframe, subset, bins, xlabelrotation: int = 0, bw_adjust: float = 5.0):
   
    g = sns.displot(
        data=dataframe, 
        x=subset[1], 
        hue=subset[0], 
        kind='kde', 
        fill=False, 
        palette='Set2', 
        common_norm=False, 
        bw_adjust=bw_adjust,
        height=4, aspect=1.5  # Controls figure size
    )
    g.set_axis_labels(subset[1], "Density")
    g.fig.suptitle(f'{subset[1]} distribution by {subset[0]}', fontsize=12)
    plt.xticks(rotation=xlabelrotation)  # Rotate x-axis labels

    plt.show()


df_train['WORK_EXP_YEARS'] = df_train['DAYS_EMPLOYED'] * (-1) / 365
numerical_columns.append('WORK_EXP_YEARS')
subset = ['OCCUPATION_TYPE_GROUP', 'WORK_EXP_YEARS']
subset_2 = subset[1]
df_grouped = df_train[df_train[subset_2] > -1000][subset] #excluding outliers

plot_subset_numerical(df_grouped, subset, 50, 0, 2)


for i in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    subset = ['OCCUPATION_TYPE_GROUP', i]
    subset_2 = subset[1]
    df_grouped = df_train[subset]
    
    plot_subset_numerical(df_grouped, subset, 50, 0, 2)


subset = ['OCCUPATION_TYPE_GROUP', 'AMT_INCOME_TOTAL']
subset_2 = subset[1]

df_grouped = df_train[(df_train[subset_2] < 500000)][subset]
plot_subset_numerical(df_grouped, subset, 50, 0, 2.25)


bureau_credit['PAID_MONTHS'] = ((bureau_credit['DAYS_CREDIT_ENDDATE'] - bureau_credit['DAYS_CREDIT'] - bureau_credit['CREDIT_DAY_OVERDUE'])/30).round()
bureau_credit['TENOR'] = ((bureau_credit['DAYS_CREDIT_ENDDATE'] - bureau_credit['DAYS_CREDIT'])/30).round()
bureau_credit['BUCKET'] = np.where(bureau_credit['AMT_CREDIT_MAX_OVERDUE'] == 0 , 'B0',
                          np.where(bureau_credit['AMT_CREDIT_MAX_OVERDUE'] < 30, 'B1', 
                          np.where(bureau_credit['AMT_CREDIT_MAX_OVERDUE'] < 60, 'B2',
                          np.where(bureau_credit['AMT_CREDIT_MAX_OVERDUE'] < 90, 'B3', 
                          'B4+'))))

bureau_credit['AMT_ANNUITY_2'] = np.where(bureau_credit['CREDIT_ACTIVE'] == 'Closed', 0, bureau_credit['AMT_ANNUITY'])


merged_df_train = df_train.merge(
    bureau_credit[['SK_ID_CURR', 'CREDIT_ACTIVE', 'PAID_MONTHS', 'TENOR', 'AMT_ANNUITY_2', 'BUCKET']], 
    on='SK_ID_CURR', 
    how='left'
)

merged_df_train['TOTAL_AMT_ANNUITY'] = merged_df_train['AMT_ANNUITY'] + merged_df_train['AMT_ANNUITY_2'] 
merged_df_train['DTI'] = (merged_df_train['AMT_ANNUITY_2']/merged_df_train['AMT_INCOME_TOTAL'] ) * 100
numerical_columns.append('DTI')



subset = ['OCCUPATION_TYPE_GROUP', 'CREDIT_ACTIVE']
formula = 'Count'
xlabel_rotation = 45

#proportion
df_grouped = merged_df_train[merged_df_train['CREDIT_ACTIVE'].isin(['Closed', 'Active'])].groupby(subset).size().reset_index(name=formula)
    
total_count_per_A = df_grouped.groupby(subset[0])[formula].transform('sum')
df_grouped['Proportion'] = df_grouped[formula] / total_count_per_A

df_pivot_proportion = df_grouped.pivot_table(index=subset[0], columns=subset[1], values='Proportion', aggfunc='sum', fill_value=0)

#PD rate
pd_per_category = merged_df_train[merged_df_train['CREDIT_ACTIVE'].isin(['Closed', 'Active'])].groupby(subset)['TARGET'].mean().reset_index()

#plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

#sns.barplot(x=subset[0], y='Proportion', hue=subset[1], data=df_grouped, palette='Set2', ax=axes[0])
ax0 = df_pivot_proportion.plot(kind='bar', stacked=True, ax=axes[0], cmap='Set2')

for p in ax0.patches:
    height = p.get_height()
    if height > 0: 
        ax0.annotate(f'{height:.2%}', 
                     (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.), 
                     xytext=(0, 5), 
                     textcoords='offset points', 
                     ha='center', va='center')
    
axes[0].set_title(f'Distribution of {subset[1]} by {subset[0]}')
axes[0].set_xlabel(subset[0])
axes[0].set_ylabel(f'{formula} of {subset[1]}')
axes[0].tick_params(axis='x', labelrotation=xlabel_rotation)

sns.barplot(x=subset[0], y='TARGET', hue=subset[1], data=pd_per_category, palette='Set3', ax=axes[1])

for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2%}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     xytext=(0, 5), 
                     textcoords='offset points', 
                     ha='center', va='bottom')
    
axes[1].set_title(f'PD rate of {subset[1]} by {subset[0]}')
axes[1].set_xlabel(subset[0])
axes[1].tick_params(axis='x', labelrotation=xlabel_rotation)

plt.tight_layout()
plt.show()


merged_df_train['DTI'] = (merged_df_train['TOTAL_AMT_ANNUITY']/merged_df_train['AMT_INCOME_TOTAL'] ) * 100
numerical_columns.append('DTI')

merged_df_train['DTI_group'] = np.where(
    merged_df_train['DTI'] < 25, '<25',
    np.where(merged_df_train['DTI'] < 50, '25-50',
    np.where(merged_df_train['DTI'] < 100, '50-100',
    np.where(merged_df_train['DTI'] < 150, '100-150',
    np.where(merged_df_train['DTI'] < 200, '150-200', '200+')))))

categorical_columns.append('DTI_group')

subset = ['OCCUPATION_TYPE_GROUP', 'DTI_group']
formula = 'Count'
xlabel_rotation = 45

#proportion
df_grouped = merged_df_train[merged_df_train['CREDIT_ACTIVE'].isin(['Closed', 'Active'])].groupby(subset).size().reset_index(name=formula)
    
total_count_per_A = df_grouped.groupby(subset[0])[formula].transform('sum')
df_grouped['Proportion'] = 100*df_grouped[formula] / total_count_per_A

df_pivot_proportion = df_grouped.pivot_table(index=subset[0], columns=subset[1], values='Proportion', aggfunc='sum', fill_value=0)

#PD rate
pd_per_category = merged_df_train[(merged_df_train['DTI_group'].isin(['<25', '25-50', '200+'])) & (merged_df_train['CREDIT_ACTIVE'].isin(['Active', 'Closed']))].groupby(subset)['TARGET'].mean().reset_index()

#plot
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

#sns.barplot(x=subset[0], y='Proportion', hue=subset[1], data=df_grouped, palette='Set2', ax=axes[0])
ax0 = df_pivot_proportion.plot(kind='bar', stacked=True, ax=axes[0], cmap='Set2')
    
axes[0].set_title(f'Distribution of {subset[1]} by {subset[0]}')
axes[0].set_xlabel(subset[0])
axes[0].set_ylabel(f'{formula} of {subset[1]}')
axes[0].tick_params(axis='x', labelrotation=xlabel_rotation)

sns.barplot(x=subset[0], y='TARGET', hue=subset[1], data=pd_per_category, palette='Set3', ax=axes[1])

for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2f}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     xytext=(0, 5), 
                     textcoords='offset points', 
                     ha='center', va='bottom')
    
axes[1].set_title(f'PD rate of {subset[1]} by {subset[0]}')
axes[1].set_xlabel(subset[0])
axes[1].tick_params(axis='x', labelrotation=xlabel_rotation)

plt.tight_layout()
plt.show()




