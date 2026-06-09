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


from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from xgboost import XGBRegressor
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px
import warnings
# warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score


learn = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submit = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
org = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
print("learn shape: ", learn.shape)
print("test shape: ", test.shape)
print("submit shape: ", submit.shape)
print("org shape: ", org.shape)


learn


# Check Null values
def analyze_dataframe(df):
    # Create a DataFrame to store the results
    result_df = pd.DataFrame(index=df.columns, columns=['Unique Values', 'Missing Values'])

    # Populate the result DataFrame with the desired information
    for column in df.columns:
        unique_values = df[column].nunique()
        missing_values = df[column].isnull().sum()

        result_df.loc[column] = [unique_values, missing_values]
    return result_df


analyze_dataframe(learn)


analyze_dataframe(test)


analyze_dataframe(org)


# visualize target
TARGET = "Listening_Time_minutes"

fig, ax = plt.subplots(1,1,figsize=(8, 6))
palette = sns.color_palette('tab10', 3)
sns.kdeplot(data=learn, x=learn[TARGET], ax=ax, label='Train Target', color=palette[0], fill=True)
sns.kdeplot(data=org, x=org[TARGET], ax=ax, label='Original Target', color=palette[1], fill=True)
ax.set_title(f'Generated ground-truth vs Original ground-truth', fontsize=12)
ax.legend(title='Dataset', loc='upper right', labels=['Train Target', 'Original Target'])


# Distribution of numeric attributes
import matplotlib.pyplot as plt
import numpy as np

NUM_COLUMNS = [
    "Episode_Length_minutes",  # note: some outliers for test set
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage"
]

# Create a figure and a 5x4 grid of subplots
fig, axs = plt.subplots(3, 1, figsize=(20, 15))

# Flatten the 2D array of axes to make iteration easier
axs = axs.flatten()

# remove outliers from test
filtered_test = test[test.Episode_Length_minutes < org.Episode_Length_minutes.max()]

# Loop through each subplot and plot the data
for i in range(3):
    palette = sns.color_palette('tab10', 3)
    sns.kdeplot(data=learn, x=learn[NUM_COLUMNS[i]], ax=axs[i], label=f'Train', color=palette[0], fill=True)
    sns.kdeplot(data=org, x=org[NUM_COLUMNS[i]], ax=axs[i], label=f'Original', color=palette[1], fill=True)
    sns.kdeplot(data=filtered_test, x=filtered_test[NUM_COLUMNS[i]], ax=axs[i], label=f'Test', color=palette[2], fill=True)
    axs[i].set_title(f'Generated vs ground-truth', fontsize=12)
    axs[i].legend(title='Dataset', loc='upper right', labels=[f'Train {NUM_COLUMNS[i]}', f'Original {NUM_COLUMNS[i]}', f'Test {NUM_COLUMNS[i]}'])

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the figure
plt.show()


# correlation between numeric features and target
def plot_corr(df, method='pearson'):
    corr_mat = df.corr(method=method) # default pearson
    train_mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    cmap = sns.diverging_palette(100, 7, s = 75, l = 40, n = 5, center = 'light', as_cmap = True)

    plt.figure(figsize = (15, 10))
    sns.heatmap(corr_mat, annot = True, cmap = cmap, fmt = '.2f', center = 0,
                annot_kws = {'size': 12}, mask = train_mask).set_title('Correlations Among Numeric Features')
    plt.show()

plot_corr(learn[NUM_COLUMNS + [TARGET]], method='pearson')


learn.head()


# # Assuming 'df' is your dataframe
# sns.boxplot(x='Genre', y='Listening_Time_minutes', data=learn)

# plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
# plt.title('Distribution of Listening Time by Genre')  # Title of the chart
# plt.xlabel('Genre')  # Label for the x-axis
# plt.ylabel('Listening Time in Minutes')  # Label for the y-axis

# plt.show()  # Display the plot


# visualization of categorical columns vs target
CAT_COLUMNS = [
    "Podcast_Name",
    "Episode_Title",
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment",
    "Number_of_Ads"
]

# Assuming 'df' is your dataframe
for cat_col in CAT_COLUMNS:
    g = sns.catplot(x=cat_col, y='Listening_Time_minutes', kind='box', data=learn, height=5, aspect=2)
    g.set_xticklabels(rotation=45)
    g.fig.suptitle(f'Distribution of Listening Time by {cat_col}', y=1.02)  # Adjust title position
    g.set_axis_labels(cat_col, 'Listening Time in Minutes')
    plt.show()


