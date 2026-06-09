import pandas as pd
pd.set_option('display.max_rows', None)     
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 1200)
pd.set_option('display.max_colwidth', None)

kernels = pd.read_csv("/kaggle/input/kernelswithcalltoaction/KernelsWithCallToAction.csv")
kernels['CreationDate'] = pd.to_datetime(kernels['CreationDate'], errors='coerce')
print(kernels.head(10))


import matplotlib.pyplot as plt

kernels['Month'] = kernels['CreationDate'].dt.to_period('M')

monthlyKernelCounts = kernels.groupby('Month').size()
monthlyUserCounts = kernels.groupby('Month')['AuthorUserId'].nunique()

fig, ax = plt.subplots(figsize=(10, 5))
monthlyKernelCounts.sort_index().plot(kind='line', label='Kernels', ax=ax)
monthlyUserCounts.sort_index().plot(kind='line', label='Unique Users', ax=ax)

ax.set_title("Monthly Kernels Containing Upvote Requests")
ax.set_xlabel("Month")
ax.set_ylabel("Count")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import numpy as np

userMonthActivity = kernels.groupby(['AuthorUserId', 'Month']).size().unstack(fill_value=0)
userMonthActivity.columns = userMonthActivity.columns.to_timestamp()
total_kernels = userMonthActivity.sum(axis=1)

active_months = (userMonthActivity > 0).sum(axis=1)

first_month = userMonthActivity.apply(lambda row: row[row > 0].index.min(), axis=1)
last_month = userMonthActivity.apply(lambda row: row[row > 0].index.max(), axis=1)

kernels_per_month = total_kernels / active_months

user_summary = pd.DataFrame({
    'Total Kernels': total_kernels,
    'Active Months': active_months,
    'Avg Kernels/Month': kernels_per_month,
    'First Month': first_month,
    'Last Month': last_month
})

top_50_users = user_summary.sort_values(by='Total Kernels', ascending=False).head(50)
user_views = kernels.groupby('AuthorUserId')['TotalViews'].agg(['sum', 'mean'])
user_votes = kernels.groupby('AuthorUserId')['TotalVotes'].agg(['sum', 'mean'])
max_views = kernels.groupby('AuthorUserId')['TotalViews'].max()
max_votes = kernels.groupby('AuthorUserId')['TotalVotes'].max()

top_50_users['Total Views'] = user_views['sum']
top_50_users['Avg Views/Kernel'] = user_views['mean']
top_50_users['Total Votes'] = user_votes['sum']
top_50_users['Avg Votes/Kernel'] = user_votes['mean']
top_50_users['Max Views (Single Kernel)'] = max_views
top_50_users['Max Votes (Single Kernel)'] = max_votes

print(top_50_users.head(10))


top_10_avg_votes_per_kernel = top_50_users.sort_values(by='Avg Votes/Kernel', ascending=False).head(10)

cols_to_show = ['Total Kernels', 'Avg Kernels/Month', 'Avg Views/Kernel', 'Avg Votes/Kernel', 'Max Votes (Single Kernel)']
print(top_10_avg_votes_per_kernel[cols_to_show].head(10))

avg_kernels_per_month_top10 = top_10_avg_votes_per_kernel['Avg Kernels/Month'].mean()
print(f"Average 'Avg Kernels/Month' for top 10 authors: {avg_kernels_per_month_top10:.2f}")


top_10_most_upvoted_kernels = top_50_users.sort_values(by='Max Votes (Single Kernel)', ascending=False).head(10)
print(top_10_most_upvoted_kernels[cols_to_show])

avg_kernels_per_month_top10 = top_10_most_upvoted_kernels['Avg Kernels/Month'].mean()
print(f"Average 'Avg Kernels/Month' for top 10 authors: {avg_kernels_per_month_top10:.2f}")


parse_dates = ['CreationDate']

dtypes = {
    'Id': 'Int32',
    'AuthorUserId': 'Int32',
    'ScriptId': 'Int32'
}

kernelVersions = pd.read_csv(
    '/kaggle/input/meta-kaggle/KernelVersions.csv',
    dtype=dtypes,
    parse_dates=parse_dates,
    usecols=dtypes.keys() | set(parse_dates),
    low_memory=False
)

kernelVersions.head(10)


authors = top_10_most_upvoted_kernels.head(5).index.unique()
top_author_kernels = kernels[kernels['AuthorUserId'].isin(authors)]
top_script_ids = top_author_kernels['Id'].dropna().unique()

print(f"Found {len(top_script_ids)} kernels by top 5 most upvoted authors.")

filtered_versions = kernelVersions[kernelVersions['ScriptId'].isin(top_script_ids)]

versions_per_kernel = (
    filtered_versions
    .groupby(['AuthorUserId', 'ScriptId'])
    .size()
    .reset_index(name='VersionCount')
)

versions_per_author = (
    versions_per_kernel
    .groupby('AuthorUserId')['VersionCount']
    .sum()
    .reset_index(name='TotalKernelVersions')
    .sort_values(by='TotalKernelVersions', ascending=False)
)

print(versions_per_author)


import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(data=versions_per_kernel, x='AuthorUserId', y='VersionCount')
plt.title('Distribution of Kernel Versions per Value-Driven Author')
plt.xlabel("AuthorUserId")
plt.ylabel("Kernel Version Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


pivot = versions_per_kernel.pivot_table(
    index='AuthorUserId',
    columns='VersionCount',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(14, 6))
sns.heatmap(pivot, cmap='Blues', cbar_kws={'label': 'Number of Kernels'})
plt.title("Heatmap of Kernel Version Count Distribution per Value-Driven Author")
plt.xlabel("Version Count per Kernel")
plt.ylabel("AuthorUserId")
plt.tight_layout()
plt.show()


author_stats = versions_per_kernel.groupby('AuthorUserId').agg(
    TotalVersions=('VersionCount', 'sum'),
    KernelCount=('ScriptId', 'nunique')
)

author_stats['AvgUpdatesPerKernel'] = author_stats['TotalVersions'] / author_stats['KernelCount']

author_stats = author_stats.sort_values(by='AvgUpdatesPerKernel', ascending=False)

print(author_stats)


top_5_kernels_per_month = top_50_users.sort_values(by='Avg Kernels/Month', ascending=False).head(5)

authors = top_5_kernels_per_month.index.unique()
top_author_kernels = kernels[kernels['AuthorUserId'].isin(authors)]
top_script_ids = top_author_kernels['Id'].dropna().unique()

print(f"Found {len(top_script_ids)} kernels by top 5 high-volume authors.")

filtered_versions = kernelVersions[kernelVersions['ScriptId'].isin(top_script_ids)]

versions_per_kernel = (
    filtered_versions
    .groupby(['AuthorUserId', 'ScriptId'])
    .size()
    .reset_index(name='VersionCount')
)

versions_per_author = (
    versions_per_kernel
    .groupby('AuthorUserId')['VersionCount']
    .sum()
    .reset_index(name='TotalKernelVersions')
    .sort_values(by='TotalKernelVersions', ascending=False)
)

print(versions_per_author)

plt.figure(figsize=(12, 6))
sns.boxplot(data=versions_per_kernel, x='AuthorUserId', y='VersionCount')
plt.title('Distribution of Kernel Versions per High-Volume Author')
plt.xlabel("AuthorUserId")
plt.ylabel("Kernel Version Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


pivot = versions_per_kernel.pivot_table(
    index='AuthorUserId',
    columns='VersionCount',
    aggfunc='size',
    fill_value=0
)

plt.figure(figsize=(14, 6))
sns.heatmap(pivot, cmap='Blues', cbar_kws={'label': 'Number of Kernels'})
plt.title("Heatmap of Kernel Version Count Distribution per High-Volume Author")
plt.xlabel("Version Count per Kernel")
plt.ylabel("AuthorUserId")
plt.tight_layout()
plt.show()


heatmap_data = (
    versions_per_kernel
    .groupby(['AuthorUserId', 'VersionCount'])
    .size()
    .unstack(fill_value=0)
    .sort_index(axis=1)  
)

log_heatmap_data = np.log1p(heatmap_data)

plt.figure(figsize=(14, 6))
sns.heatmap(
    log_heatmap_data,
    cmap='Blues',
    annot=False,
    linewidths=0.5,
    cbar_kws={'label': 'Log(1 + Number of Kernels)'}
)
plt.title("Log-Scaled Heatmap of Kernel Version Count Distribution per High-Volume Author")
plt.xlabel("Version Count per Kernel")
plt.ylabel("AuthorUserId")
plt.tight_layout()
plt.show()


author_stats = versions_per_kernel.groupby('AuthorUserId').agg(
    TotalVersions=('VersionCount', 'sum'),
    KernelCount=('ScriptId', 'nunique')
)

author_stats['AvgUpdatesPerKernel'] = author_stats['TotalVersions'] / author_stats['KernelCount']

author_stats = author_stats.sort_values(by='AvgUpdatesPerKernel', ascending=False)

print(author_stats)


parse_dates = ['VoteDate']
dtypes = {
    'Id': 'int32',
    'UserId': 'Int32',
    'KernelVersionId': 'Int32'
}

kernelVotes = pd.read_csv(
    '/kaggle/input/meta-kaggle-kernel-votes/KernelVotes.csv',
    dtype=dtypes,
    parse_dates=parse_dates,
    usecols=dtypes.keys() | set(parse_dates),
    low_memory=False
)

kernelVotes.head(10)


top_5_kernels_per_month = top_50_users.sort_values(by='Avg Kernels/Month', ascending=False).head(5)
authors_from_max_kernels = top_5_kernels_per_month.index.unique()
authors_from_max_votes = top_10_most_upvoted_kernels.head(5).index.unique()

topAuthors = pd.Index(authors_from_max_kernels.union(authors_from_max_votes))

topKernels = kernels[kernels['AuthorUserId'].isin(topAuthors)]

votesWithAuthors = kernelVotes.merge(
    topKernels[['CurrentKernelVersionId', 'AuthorUserId']],
    left_on='KernelVersionId',
    right_on='CurrentKernelVersionId',
    how='inner'
)

votesWithAuthors['Month'] = votesWithAuthors['VoteDate'].dt.to_period('M')
monthlyVotes = (
    votesWithAuthors
    .groupby(['AuthorUserId', 'Month'])
    .size()
    .reset_index(name='MonthlyVotes')
)

monthlyVotes['Month'] = monthlyVotes['Month'].dt.to_timestamp()
monthlyVotes = monthlyVotes.sort_values(by=['AuthorUserId', 'Month'])
monthlyVotes['CumulativeVotes'] = (
    monthlyVotes
    .groupby('AuthorUserId')['MonthlyVotes']
    .cumsum()
)

monthlyVotes['RelativeMonth'] = (
    monthlyVotes
    .groupby('AuthorUserId')
    .cumcount()
)

pivot_relative = monthlyVotes.pivot(index='RelativeMonth', columns='AuthorUserId', values='CumulativeVotes')

fig, ax = plt.subplots(figsize=(12, 6))

for author_id in pivot_relative.columns:
    marker = '^' if author_id in authors_from_max_votes else 'o'
    ax.plot(
        pivot_relative.index,
        pivot_relative[author_id],
        marker=marker,
        label=str(author_id)
    )

ax.set_title("Cumulative upvotes Over Author Timeline")
ax.set_xlabel("Months")
ax.set_ylabel("Cumulative upvotes")
ax.legend(title="Author ID", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
plt.tight_layout()
plt.show()

