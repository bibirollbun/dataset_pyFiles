import kagglehub
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from IPython.display import display, Markdown, Latex
from kagglehub import KaggleDatasetAdapter, PolarsFrameType


path = kagglehub.dataset_download("kaggle/meta-kaggle")

# Languages lookup
kernel_languages_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "KernelLanguages.csv",
)

# The single Kernel Versions (a kernel has multiple versions)
kernel_versions_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "KernelVersions.csv",
)

# List of public kernels
kernels_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Kernels.csv"
)

print("Loaded and setup meta-kaggle dataset. Path to dataset files:", path)


# Get the number of unique notebooks (kernel Ids)
notebooks_count = kernels_df["Id"].nunique()
print('Unique notebooks: ', notebooks_count)


# Column ScriptLanguageId resolved into a static list for checking whether a kernel version is R or python
R_list      = [1,5,12,13,15,16]
python_list = [2,8,9,14]


# The following is courtesy of https://www.kaggle.com/code/carlmcbrideellis/kaggle-in-numbers
kernel_versions_df = kernel_versions_df.drop_duplicates(subset=['ScriptId'], keep='last')
R_number       = len(kernel_versions_df[kernel_versions_df['ScriptLanguageId'].isin(R_list)])
R_percent      = round(100/notebooks_count * R_number)
python_number  = len(kernel_versions_df[kernel_versions_df['ScriptLanguageId'].isin(python_list)])
python_percent = round(100/notebooks_count * python_number)

display(Markdown(f'**We have a total of {notebooks_count} kernels, of which {R_percent}% are written in R and {python_percent}% in Python.**'))


kernel_versions_df['CreationDate'] = pd.to_datetime(kernel_versions_df['CreationDate'])

# First and last kernel version
display(Markdown('#### Oldest Kernel: ' + str(kernel_versions_df['CreationDate'].min())))
display(Markdown('#### Newest Kernel: ' + str(kernel_versions_df['CreationDate'].max())))

# Filter the languages and assign more readable labels to them
filtered_df = kernel_versions_df[kernel_versions_df['ScriptLanguageId'].isin(R_list + python_list)].copy()
filtered_df['Language'] = filtered_df['ScriptLanguageId'].apply(
    lambda x: 'Python' if x in python_list else 'R'
)

# Group by date and language, count occurrences
daily_counts = (
    filtered_df
    .groupby([filtered_df['CreationDate'].dt.date, 'Language'])
    .size()
    .unstack(fill_value=0)
)

# Calculate total and percent usage
daily_percent = daily_counts.div(daily_counts.sum(axis=1), axis=0) * 100

plt.rcParams['axes.facecolor'] = 'whitesmoke'
plt.rcParams['figure.facecolor'] = 'white'
colors = {
    'Python': '#ffd343',
    'R': '#198CE7'      
}

# Smooth the daily percentage with a rolling average
daily_percent_smoothed = daily_percent.rolling(window=7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 7))
for lang in ['Python', 'R']:
    ax.plot(
        daily_percent_smoothed.index,
        daily_percent_smoothed[lang],
        label=lang,
        color=colors[lang],
        linewidth=2.5,
        alpha=0.85
    )

ax.set_title('% Usage of R and Python Kernels (specifically, Versions) on Kaggle Over Time', fontsize=14, pad=20)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Kernel Usage (%)', fontsize=14)
ax.legend(title='Language', loc='center right', fontsize=12, title_fontsize=13)
ax.grid(True, linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    r_kernels = kernel_versions_df[kernel_versions_df['ScriptLanguageId'].isin(R_list)]
    max_votes = r_kernels['TotalVotes'].max()
    max_vote_rows = r_kernels[r_kernels['TotalVotes'] == max_votes]

    display(max_vote_rows.head())


users_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "kaggle/meta-kaggle",
    "Users.csv",
)

r_kernel_counts = r_kernels['AuthorUserId'].value_counts()
# Merge with users_df to get usernames
r_kernel_counts_df = r_kernel_counts.rename('KernelCount').reset_index()
r_kernel_counts_df = r_kernel_counts_df.rename(columns={'index': 'AuthorUserId'})
r_kernel_counts_with_names = r_kernel_counts_df.merge(users_df[['Id', 'UserName']], left_on='AuthorUserId', right_on='Id', how='left')

# Get the top user
top_user_row = r_kernel_counts_with_names.iloc[0]
r_poweruser_username = top_user_row['UserName']
r_kernel_count = top_user_row['KernelCount']

display(Markdown(f"**Top R user (most R kernels) is @{r_poweruser_username} with {r_kernel_count} R kernels.**"))


display(r_kernel_counts_with_names[['UserName', 'KernelCount']].head(10))


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    python_kernels = kernel_versions_df[kernel_versions_df['ScriptLanguageId'].isin(python_list)]
    max_votes = python_kernels['TotalVotes'].max()
    max_vote_rows = python_kernels[python_kernels['TotalVotes'] == max_votes]
    
    display(max_vote_rows.head())

