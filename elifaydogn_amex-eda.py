import pandas as pd
import numpy as np
import pickle, gc
from matplotlib import pyplot as plt


train_labels = pd.read_csv('../input/amex-default-prediction/train_labels.csv')
train_labels.head(2)


# Check for missing data and duplicated customer_IDs
train_labels.isna().any().any(), train_labels.customer_ID.duplicated().any()


label_stats = pd.DataFrame({'absolute': train_labels.target.value_counts(),
              'relative': train_labels.target.value_counts() / len(train_labels)})
label_stats['absolute upsampled'] =  label_stats.absolute * np.array([20, 1])
label_stats['relative upsampled'] = label_stats['absolute upsampled'] / label_stats['absolute upsampled'].sum()
label_stats


%%time
train = pd.read_feather('../input/amexfeather/train_data.ftr')
test = pd.read_feather('../input/amexfeather/test_data.ftr')
with pd.option_context("display.min_rows", 6):
    display(train)
    display(test)


print('Train statement dates: ', train.S_2.min(), train.S_2.max(), train.S_2.isna().any())
print('Test statement dates: ',  test.S_2.min(), test.S_2.max(), test.S_2.isna().any())



print(f'Train data memory usage: {train.memory_usage().sum() / 1e9} GBytes')
print(f'Test data memory usage:  {test.memory_usage().sum() / 1e9} GBytes')



train.info(max_cols=200, show_counts=True)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
train_sc = train.customer_ID.value_counts().value_counts().sort_index(ascending=False).rename('Train statements per customer')
ax1.pie(train_sc, labels=train_sc.index)
ax1.set_title(train_sc.name)
test_sc = test.customer_ID.value_counts().value_counts().sort_index(ascending=False).rename('Test statements per customer')
ax2.pie(test_sc, labels=test_sc.index)
ax2.set_title(test_sc.name)
plt.show()

# display(train.customer_ID.value_counts().value_counts().sort_index(ascending=False).rename('Train statements per customer'))
# display(train.customer_ID.value_counts().value_counts().sort_index(ascending=False).rename('Test statements per customer'))



temp = train.S_2.groupby(train.customer_ID).max()
plt.figure(figsize=(16, 4))
plt.hist(temp, bins=pd.date_range("2018-03-01", "2018-04-01", freq="d"),
         rwidth=0.8, color='#ffd700')
plt.title('When did the train customers get their last statements?', fontsize=20)
plt.xlabel('Last statement date per customer')
plt.ylabel('Count')
plt.gca().set_facecolor('#0057b8')
plt.show()
del temp

temp = test.S_2.groupby(test.customer_ID).max()
plt.figure(figsize=(16, 4))
plt.hist(temp, bins=pd.date_range("2019-04-01", "2019-11-01", freq="d"),
         rwidth=0.74, color='#ffd700')
plt.title('When did the test customers get their last statements?', fontsize=20)
plt.xlabel('Last statement date per customer')
plt.ylabel('Count')
plt.gca().set_facecolor('#0057b8')
plt.show()
del temp


temp = train.S_2.groupby(train.customer_ID).agg(['max', 'min'])
plt.figure(figsize=(16, 3))
plt.hist((temp['max'] - temp['min']).dt.days, bins=400, color='#ffd700')
plt.xlabel('days')
plt.ylabel('count')
plt.title('Number of days between first and last statement of customer (train)', fontsize=20)
plt.gca().set_facecolor('#0057b8')
plt.show()

temp = test.S_2.groupby(test.customer_ID).agg(['max', 'min'])
plt.figure(figsize=(16, 3))
plt.hist((temp['max'] - temp['min']).dt.days, bins=400, color='#ffd700')
plt.xlabel('days')
plt.ylabel('count')
plt.title('Number of days between first and last statement of customer (test)', fontsize=20)
plt.gca().set_facecolor('#0057b8')
plt.show()
del temp


temp = pd.concat([train[['customer_ID', 'S_2']], test[['customer_ID', 'S_2']]], axis=0)
temp.set_index('customer_ID', inplace=True)
temp['last_month'] = temp.groupby('customer_ID').S_2.max().dt.month
last_month = temp['last_month'].values

plt.figure(figsize=(16, 4))
plt.hist([temp.S_2[temp.last_month == 3],   # ending 03/18 -> training
          temp.S_2[temp.last_month == 4],   # ending 04/19 -> public lb
          temp.S_2[temp.last_month == 10]], # ending 10/19 -> private lb
         bins=pd.date_range("2017-03-01", "2019-11-01", freq="MS"),
         label=['Training', 'Public leaderboard', 'Private leaderboard'],
         stacked=True)
plt.xticks(pd.date_range("2017-03-01", "2019-11-01", freq="QS"))
plt.xlabel('Statement date')
plt.ylabel('Count')
plt.title('The three datasets over time', fontsize=20)
plt.legend()
plt.show()



for f in [ 'B_29', 'S_9','D_87']:#, 'D_88', 'R_26', 'R_27', 'D_108', 'D_110', 'D_111', 'B_39', 'B_42']:
    temp = pd.concat([train[[f, 'S_2']], test[[f, 'S_2']]], axis=0)
    temp['last_month'] = last_month
    temp['has_f'] = ~temp[f].isna() 

    plt.figure(figsize=(16, 4))
    plt.hist([temp.S_2[temp.has_f & (temp.last_month == 3)],   # ending 03/18 -> training
              temp.S_2[temp.has_f & (temp.last_month == 4)],   # ending 04/19 -> public lb
              temp.S_2[temp.has_f & (temp.last_month == 10)]], # ending 10/19 -> private lb
             bins=pd.date_range("2017-03-01", "2019-11-01", freq="MS"),
             label=['Training', 'Public leaderboard', 'Private leaderboard'],
             stacked=True)
    plt.xticks(pd.date_range("2017-03-01", "2019-11-01", freq="QS"))
    plt.xlabel('Statement date')
    plt.ylabel(f'Count of {f} non-null values')
    plt.title(f'{f} non-null values over time', fontsize=20)
    plt.legend()
    plt.show()



cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
plt.figure(figsize=(16, 16))
for i, f in enumerate(cat_features):
    plt.subplot(4, 3, i+1)
    temp = pd.DataFrame(train[f][train.target == 0].value_counts(dropna=False, normalize=True).sort_index().rename('count'))
    temp.index.name = 'value'
    temp.reset_index(inplace=True)
    plt.bar(temp.index, temp['count'], alpha=0.5, label='target=0')
    temp = pd.DataFrame(train[f][train.target == 1].value_counts(dropna=False, normalize=True).sort_index().rename('count'))
    temp.index.name = 'value'
    temp.reset_index(inplace=True)
    plt.bar(temp.index, temp['count'], alpha=0.5, label='target=1')
    plt.xlabel(f)
    plt.ylabel('frequency')
    plt.legend()
    plt.xticks(temp.index, temp.value)
plt.suptitle('Categorical features', fontsize=20, y=0.93)
plt.show()
del temp



bin_features = ['B_31', 'D_87']
plt.figure(figsize=(16, 4))
for i, f in enumerate(bin_features):
    plt.subplot(1, 2, i+1)
    temp = pd.DataFrame(train[f][train.target == 0].value_counts(dropna=False, normalize=True).sort_index().rename('count'))
    temp.index.name = 'value'
    temp.reset_index(inplace=True)
    plt.bar(temp.index, temp['count'], alpha=0.5, label='target=0')
    temp = pd.DataFrame(train[f][train.target == 1].value_counts(dropna=False, normalize=True).sort_index().rename('count'))
    temp.index.name = 'value'
    temp.reset_index(inplace=True)
    plt.bar(temp.index, temp['count'], alpha=0.5, label='target=1')
    plt.xlabel(f)
    plt.ylabel('frequency')
    plt.legend()
    plt.xticks(temp.index, temp.value)
plt.suptitle('Binary features', fontsize=20)
plt.show()
del temp


cont_features = sorted([f for f in train.columns if f not in cat_features + bin_features + ['customer_ID', 'target', 'S_2']])
print(len(cont_features))
# print(cont_features)
ncols = 4
for i, f in enumerate(cont_features):
    if i % ncols == 0: 
        if i > 0: plt.show()
        plt.figure(figsize=(16, 3))
        if i == 0: plt.suptitle('Continuous features', fontsize=20, y=1.02)
    plt.subplot(1, ncols, i % ncols + 1)
    plt.hist(train[f], bins=200)
    plt.xlabel(f)
plt.show()


def read_columns(name, features):
    """Read the specified columns of the train/test csv at full precision"""
    chunksize = 1000000
    chunklist = []
    with pd.read_csv(f"../input/amex-default-prediction/{name}_data.csv", chunksize=chunksize) as reader:
        for i, chunk in enumerate(reader):
            chunk = chunk[features] # keep only selected columns
            chunklist.append(chunk)
            print(i, end=' ')
            if i == 5: break
        print()
    df = pd.concat(chunklist, axis=0)
    return df

df = read_columns('train', ['B_19', 'S_13'])
df.info()


y = df.B_19
for i in np.linspace(0, 1, 11):
    plt.figure(figsize=(16, 3))
    plt.hist(y, bins=np.linspace(i, i+0.1, 101), rwidth=0.8, color='m')
    plt.xticks(np.linspace(i, i+0.1, 11))
    plt.title(f"B_19 histogram, part {int(i*10+1)}")
    plt.show()



y = df.S_13
for i in np.linspace(0, 1, 11):
    plt.figure(figsize=(16, 3))
    plt.hist(y, bins=np.linspace(i, i+0.1, 101), rwidth=0.8, color='c')
    plt.xticks(np.linspace(i, i+0.1, 11))
    plt.title(f"S_13 histogram, part {int(i*10+1)}")
    plt.show()


# Check missing values
missing_values = train.isnull().mean().sort_values(ascending=False)
missing_values[missing_values > 0].head(20)  # Show top 20 columns with missing values



# Group the features
missing = train.isnull().mean()
missing = missing[missing > 0]

# Separate categorical, binary, and continuous features
cat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
bin_features = ['B_31', 'D_87']
cont_features = [f for f in train.columns if f not in cat_features + bin_features + ['customer_ID', 'target', 'S_2']]

missing_cat = [col for col in cat_features if col in missing.index]
missing_bin = [col for col in bin_features if col in missing.index]
missing_cont = [col for col in cont_features if col in missing.index]

print("Missing categorical features:", missing_cat)
print("Missing binary features:", missing_bin)
print("Missing continuous features:", missing_cont[:10])  



# Categorical: Fill with the most frequent value (mode)
for col in missing_cat:
    train[col].fillna(train[col].mode()[0], inplace=True)

# Binary: Fill with the most frequent value (mode)
for col in missing_bin:
    train[col].fillna(train[col].mode()[0], inplace=True)

# Continuous: Fill with the median (mean is also possible, but median is more robust to outliers)
for col in missing_cont:
    train[col].fillna(train[col].median(), inplace=True)



# Final check to ensure all missing values have been filled
missing_after = train.isnull().mean()
missing_after[missing_after > 0] 

# If the output is empty, it means all missing values have been filled


outlier_summary = {}

for col in cont_features:
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = ((train[col] < lower) | (train[col] > upper)).sum()
    outlier_summary[col] = outlier_count

# Top 5 features with the most outliers
top_outlier_cols = pd.Series(outlier_summary).sort_values(ascending=False).head(5)
top_outlier_cols



import seaborn as sns
import matplotlib.pyplot as plt

for col in top_outlier_cols.index:
    plt.figure(figsize=(12, 4))
    sns.boxplot(data=train, x=train[col])
    plt.title(f'Boxplot of {col} (outliers: {top_outlier_cols[col]})')
    plt.show()



from sklearn.ensemble import IsolationForest

isof_results = {}
sample_train = train.sample(n=100_000, random_state=42)  # Alt kÃ¼me

for col in cont_features[:10]:
    clf = IsolationForest(n_estimators=50, contamination='auto', random_state=42, n_jobs=-1)
    preds = clf.fit_predict(sample_train[[col]])
    outlier_count = (preds == -1).sum()
    isof_results[col] = outlier_count





import seaborn as sns
import matplotlib.pyplot as plt

# Prepare data
heatmap_data = combined_outliers.copy().head(10).T  # Top 10 features, methods as rows

# Create a clustermap
sns.set(font_scale=1.0)
clustermap = sns.clustermap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    linewidths=0.5,
    figsize=(12, 6),
    col_cluster=False,  # Only cluster rows (methods)
    row_cluster=False   # Set True if you want method clustering
)

clustermap.ax_heatmap.set_title("Clustered Heatmap of Outlier Counts (Top 10 Features)", pad=20)
plt.show()






# IQR results (from previous step)
iqr_df = pd.Series(outlier_summary).sort_values(ascending=False)
iqr_df.name = "IQR Outliers"

# Isolation Forest results (from previous step)
isof_df = pd.Series(isof_results).sort_values(ascending=False)
isof_df.name = "Isolation Forest Outliers"

# Merge the two into a single DataFrame
combined_outliers = pd.concat([iqr_df, isof_df], axis=1)
combined_outliers = combined_outliers.dropna().astype(int)  # Drop any missing and convert to int
combined_outliers



# Calculate outlier percentages for both methods
iqr_percentage = (iqr_df / len(train)) * 100
isof_percentage = (isof_df / len(train)) * 100

# Combine them into a new DataFrame
outlier_percentages = pd.concat([iqr_percentage, isof_percentage], axis=1)
outlier_percentages.columns = ['IQR Outlier %', 'Isolation Forest Outlier %']
outlier_percentages = outlier_percentages.round(2)
outlier_percentages

# Align index to have consistent comparison
common_index = iqr_df.index.intersection(isof_df.index)
iqr_percentage = iqr_percentage.loc[common_index]
isof_percentage = isof_percentage.loc[common_index]

outlier_percentages = pd.concat([iqr_percentage, isof_percentage], axis=1)
outlier_percentages.columns = ['IQR Outlier %', 'Isolation Forest Outlier %']
outlier_percentages = outlier_percentages.round(2)
outlier_percentages.head(10)




import matplotlib.pyplot as plt
import seaborn as sns

# Stil ayarÄ±
sns.set_style("whitegrid")

# Grafik ayarlarÄ±
plt.figure(figsize=(14, 6))
ax = outlier_percentages.plot(
    kind='bar',
    width=0.75,
    color=['#4C72B0', '#DD8452'],
    edgecolor='black',
    figsize=(14, 6)
)

# BaÅŸlÄ±k ve etiketler
plt.title("Outlier Percentages by Feature (IQR vs Isolation Forest)", fontsize=16, weight='bold')
plt.ylabel("Outlier Percentage (%)", fontsize=12)
plt.xlabel("Feature", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Detection Method')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# DeÄŸer etiketleri (bar Ã¼stÃ¼ne yazdÄ±rma)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=9, padding=3)

plt.show()




from sklearn.ensemble import IsolationForest
import pandas as pd

# Use a random sample to improve performance
sample_train = train.sample(n=100_000, random_state=42)

overlap_ratio = {}

for col in combined_outliers.index:
    # IQR outlier detection on sampled data
    q1 = sample_train[col].quantile(0.25)
    q3 = sample_train[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers = ((sample_train[col] < lower) | (sample_train[col] > upper))

    # Isolation Forest on the same sample
    clf = IsolationForest(n_estimators=50, contamination='auto', random_state=42, n_jobs=-1)
    preds = clf.fit_predict(sample_train[[col]])
    isof_outliers = (preds == -1)

    # Overlap ratio calculation
    both_outliers = iqr_outliers & isof_outliers
    union_outliers = iqr_outliers | isof_outliers

    intersection_count = both_outliers.sum()
    union_count = union_outliers.sum()

    overlap_ratio[col] = intersection_count / union_count if union_count > 0 else 0

# Convert to DataFrame
overlap_df = pd.DataFrame.from_dict(overlap_ratio, orient='index', columns=['Overlap Ratio'])
overlap_df = overlap_df.sort_values(by='Overlap Ratio', ascending=False)
overlap_df.head(10)



import matplotlib.pyplot as plt
import seaborn as sns

# Use a colorful palette and add annotations
plt.figure(figsize=(12, 6))
barplot = sns.barplot(
    x=overlap_df.head(10).index,
    y=overlap_df['Overlap Ratio'].head(10),
    palette='cubehelix'
)

# Add value labels on top of bars
for i, val in enumerate(overlap_df['Overlap Ratio'].head(10)):
    barplot.text(i, val + 0.02, f"{val:.2f}", ha='center', fontsize=11, fontweight='bold')

plt.title('Top 10 Features by Outlier Overlap Ratio\n(IQR vs Isolation Forest)', fontsize=16, weight='bold')
plt.ylabel('Overlap Ratio')
plt.xlabel('Feature')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic style
sns.set_style('whitegrid')
plt.figure(figsize=(14, 7))

# Plot with clear colors, spacing and annotation
ax = combined_outliers.head(10).plot(
    kind='bar',
    color=sns.color_palette("Set2", 2),
    edgecolor='black',
    width=0.7
)

# Titles and labels
plt.title('Top 10 Features by Outlier Counts\n(IQR vs Isolation Forest)', fontsize=16, weight='bold', pad=20)
plt.ylabel('Number of Outliers', fontsize=12)
plt.xlabel('Feature', fontsize=12)

# Axis ticks
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

# Grid and legend
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(title='Method', fontsize=11, title_fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()




# All object-type columns excluding customer_ID
object_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'customer_ID']

for col in object_cols:
    print(f"\nColumn: {col}")
    print(train[col].unique()[:10])  # Print the first 10 unique values

# object tipinde customer_ID dÄ±ÅŸÄ±nda hiÃ§ sÃ¼tun yok


# Columns with the data type 'category'
cat_cols = [col for col in train.columns if train[col].dtype.name == 'category']
print("Categorical columns:", cat_cols)



# Convert categorical columns if not already 'category' type
for col in cat_cols:
    if train[col].dtype != 'category':
        train[col] = train[col].astype('category')

# Final check: display data types
print(train[cat_cols].dtypes)



