import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print(f'PANDAS VERSION: {pd.__version__}')
print(f'NUMPY VERSION: {np.__version__}')
print(f'MATPLOTLIB VERSION: {matplotlib.__version__}')
print(f'SEABORN VERSION: {sns.__version__}')


df_train_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
df_test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
df_sample_submission_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')

print(f'''
Training Dataset:
    Shape: {df_train_data.shape} -> {df_train_data.shape[0]} rows and {df_train_data.shape[1]} columns
    Column Names: {", ".join(df_train_data.columns.to_list())}

{'='*100}

Test Dataset:
    Shape: {df_test_data.shape} -> {df_test_data.shape[0]} rows and {df_test_data.shape[1]} columns
    Column Names: {", ".join(df_test_data.columns.to_list())}

{'='*100}

Sample Submission Dataset:
    Shape: {df_sample_submission_data.shape} -> {df_sample_submission_data.shape[0]} rows and {df_sample_submission_data.shape[1]} columns
    Column Names: {", ".join(df_sample_submission_data.columns.to_list())}
''')


display(df_train_data.head())
display(df_test_data.head())


df_train_data.info()

print('+'*60)
print('+'*60)

df_test_data.info()


def categorize_missing_values(dataFrame):
    missing_percentage = dataFrame.isnull().mean() * 100
    
    more_than_50 = missing_percentage[missing_percentage > 50]
    between_30_and_50 = missing_percentage[(missing_percentage >= 30) & (missing_percentage <= 50)]
    less_than_30 = missing_percentage[missing_percentage < 30]
    
    more_than_50_percentage = more_than_50.apply(lambda x: f"{x:.2f}%")
    between_30_and_50_percentage = between_30_and_50.apply(lambda x: f"{x:.2f}%")
    less_than_30_percentage = less_than_30.apply(lambda x: f"{x:.2f}%")
    
    more_than_50_df = pd.DataFrame({
        'Category': ['More than 50% Missing'] * len(more_than_50),
        'Columns': more_than_50.index,
        'Missing Percentage': more_than_50_percentage.values
    })
    
    between_30_and_50_df = pd.DataFrame({
        'Category': ['30%-50% Missing'] * len(between_30_and_50),
        'Columns': between_30_and_50.index,
        'Missing Percentage': between_30_and_50_percentage.values
    })
    
    less_than_30_df = pd.DataFrame({
        'Category': ['Less than 30% Missing'] * len(less_than_30),
        'Columns': less_than_30.index,
        'Missing Percentage': less_than_30_percentage.values
    })
    
    categorized_df = pd.concat([more_than_50_df, between_30_and_50_df, less_than_30_df], axis=0, ignore_index=True)
    
    return categorized_df


display(categorize_missing_values(df_train_data))
display(categorize_missing_values(df_test_data))

fig = plt.figure(figsize=(36, 9))
ax1 = plt.subplot(1, 2, 1)
ax1 = sns.heatmap(df_train_data.isnull())
ax1.set_title('Missing Values Heatmap For The Training Set')
fig.add_subplot(ax1)
ax2 = plt.subplot(1, 2, 2)
ax2 = sns.heatmap(df_test_data.isnull())
ax2.set_title('Missing Values Heatmap For The Test Set')
fig.add_subplot(ax2)
fig.show()


display(df_train_data.describe())
display(df_test_data.describe())


df_train_data = df_train_data.drop(['mrd_hct', 'tce_match'], axis=1)
df_test_data = df_test_data.drop(['mrd_hct', 'tce_match'], axis=1)
print('The mrd_hct and tce_match columns dropped successfully')


fig = plt.figure()
ax = sns.countplot(data=df_train_data, x='tce_imm_match')
ax.set_title('tce_imm_match Distribution')
ax.set_xlabel('tce_imm_match')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
plt.show()


tce_imm_match_mode = df_train_data['tce_imm_match'].mode()[0]
df_train_data['tce_imm_match'] = df_train_data['tce_imm_match'].fillna(tce_imm_match_mode)
df_test_data['tce_imm_match'] = df_test_data['tce_imm_match'].fillna(tce_imm_match_mode)
print('Filling of the tce_imm_match column completed successfully')

