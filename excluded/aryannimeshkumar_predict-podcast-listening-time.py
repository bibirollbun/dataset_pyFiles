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


import pandas as pd

sample = pd.read_csv(r"/kaggle/input/playground-series-s5e4/sample_submission.csv")
test = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
train = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")


train.head()


train.shape


train.isna().sum()


for col in train.select_dtypes(include=['object', 'category']).columns:
    print(f"Column: {col}")
    print("Unique values:", train[col].unique())
    print("-" * 40)


def encode_podcast_data(df):
    # Episode_Num: extract number and convert to integer
    df['Episode_Num'] = df['Episode_Title'].str.extract(r'Episode (\d+)').astype(int)
    
    # Podcast_Name: label encoding
    df['Podcast_Name'] = df['Podcast_Name'].astype('category').cat.codes

    # Genre: label encoding
    df['Genre'] = df['Genre'].astype('category').cat.codes

    # Publication_Day: ordinal encoding
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Publication_Day'] = pd.Categorical(df['Publication_Day'], categories=day_order, ordered=True).codes

    # Publication_Time: ordinal mapping
    time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    df['Publication_Time'] = df['Publication_Time'].map(time_map)

    # Episode_Sentiment: ordinal mapping
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)

    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')

    return df



train = encode_podcast_data(train)
test = encode_podcast_data(test)


train.describe()


import pandas as pd
from scipy.stats import skew

episode_length_skew = skew(train['Episode_Length_minutes'].dropna())

guest_popularity_skew = skew(train['Guest_Popularity_percentage'].dropna())

print(f"Skewness of Episode Length (minutes): {episode_length_skew}")
print(f"Skewness of Guest Popularity (%): {guest_popularity_skew}")

if episode_length_skew > 0:
    print("Episode Length (minutes) is positively skewed (right-skewed).")
elif episode_length_skew < 0:
    print("Episode Length (minutes) is negatively skewed (left-skewed).")
else:
    print("Episode Length (minutes) is symmetric (no skewness).")

if guest_popularity_skew > 0:
    print("Guest Popularity (%) is positively skewed (right-skewed).")
elif guest_popularity_skew < 0:
    print("Guest Popularity (%) is negatively skewed (left-skewed).")
else:
    print("Guest Popularity (%) is symmetric (no skewness).")



def fill_missing_values(df):
    # Fill Guest_Popularity_percentage with median per Podcast_Name
    df["Guest_Popularity_percentage"] = df["Guest_Popularity_percentage"].fillna(
        df.groupby("Podcast_Name")["Guest_Popularity_percentage"].transform("median")
    )

    # Fill Episode_Length_minutes with median per Podcast_Name
    df["Episode_Length_minutes"] = df["Episode_Length_minutes"].fillna(
        df.groupby("Podcast_Name")["Episode_Length_minutes"].transform("median")
    )

    # Fill Number_of_Ads with 0
    df["Number_of_Ads"] = df["Number_of_Ads"].fillna(0)

    return df



train = fill_missing_values(train)
test = fill_missing_values(test)


train = train.drop_duplicates()


train.isna().sum()


train.shape


train.head()


def add_engineered_features(df):
    """
    Adds engineered features to the input DataFrame.
    Adds Engagement_Ratio only if Listening_Time_minutes is present.
    """
    df = df.copy()
    epsilon = 1e-6

    # Feature 1: Popularity Ratio
    df['Popularity_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + epsilon)

    # Feature 2: Combined Popularity Score (weighted average)
    df['Combined_Popularity'] = 0.6 * df['Host_Popularity_percentage'] + 0.4 * df['Guest_Popularity_percentage']

    # Feature 3: Harmonic Mean of Popularity
    df['Popularity_HarmonicMean'] = df.apply(
        lambda row: hmean([row['Host_Popularity_percentage'] + epsilon, row['Guest_Popularity_percentage'] + epsilon]),
        axis=1
    )

    # Feature 4: Engagement Ratio (only if target column exists)
    if 'Listening_Time_minutes' in df.columns:
        df['Engagement_Ratio'] = df['Listening_Time_minutes'] / (
            df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage'] + epsilon
        )

    return df



train = add_engineered_features(train)
test = add_engineered_features(test)


train = train.drop(columns=['Episode_Title'])
test = test.drop(columns=['Episode_Title'])


train.describe()


train.info()


import seaborn as sns
import matplotlib.pyplot as plt


corr_matrix = train.corr()

plt.figure(figsize=(15, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Select only numeric columns
numeric_columns = train.select_dtypes(include=['number']).columns

# Grid configuration: 4 rows, 2 columns
num_rows = 4
num_columns = 2
num_plots = len(numeric_columns)

# Create subplots
fig, axes = plt.subplots(num_rows, num_columns, figsize=(14, 16))
axes = axes.flatten()

# Create box plots with summary tables
for i, column in enumerate(numeric_columns):
    ax = axes[i]
    
    # Plot boxplot
    train.boxplot(column=column, ax=ax, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    ax.set_title(f'Box Plot of {column}', fontsize=10)
    ax.set_ylabel(column)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', labelsize=8)
    
    # Compute stats
    stats = train[column].describe()
    summary_values = {
        'Min': f"{stats['min']:.2f}",
        'Q1': f"{stats['25%']:.2f}",
        'Median': f"{stats['50%']:.2f}",
        'Q3': f"{stats['75%']:.2f}",
        'Max': f"{stats['max']:.2f}",
        'Mean': f"{stats['mean']:.2f}",
    }

    # Create a table with summary stats (upper right corner)
    table_data = [[key, val] for key, val in summary_values.items()]
    table = ax.table(cellText=table_data,
                     colLabels=['Stat', 'Value'],
                     cellLoc='left',
                     colLoc='left',
                     loc='upper right',
                     bbox=[0.6, 0.6, 0.35, 0.35])  # [x, y, width, height]
    
    table.set_fontsize(8)
    table.scale(1, 1.2)

# Remove extra subplots if needed
for j in range(num_plots, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()



# number_of_ads = train[train['Number_of_Ads'] > 4]
# number_of_ads

train = train[train['Number_of_Ads'] <= 4].reset_index(drop=True)
train = train[train['Episode_Length_minutes'] <= 150].reset_index(drop=True)


train.shape


 import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Select only numeric columns in test
numeric_columns = test.select_dtypes(include=['number']).columns

# Grid configuration: 4 rows, 2 columns
num_rows = 4
num_columns = 2
num_plots = len(numeric_columns)

# Create subplots
fig, axes = plt.subplots(num_rows, num_columns, figsize=(14, 16))
axes = axes.flatten()

# Create box plots with summary tables
for i, column in enumerate(numeric_columns):
    ax = axes[i]
    
    # Plot boxplot
    test.boxplot(column=column, ax=ax, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    ax.set_title(f'Box Plot of {column}', fontsize=10)
    ax.set_ylabel(column)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', labelsize=8)
    
    # Compute stats
    stats = test[column].describe()
    summary_values = {
        'Min': f"{stats['min']:.2f}",
        'Q1': f"{stats['25%']:.2f}",
        'Median': f"{stats['50%']:.2f}",
        'Q3': f"{stats['75%']:.2f}",
        'Max': f"{stats['max']:.2f}",
        'Mean': f"{stats['mean']:.2f}",
    }

    # Create summary table in top right corner
    table_data = [[key, val] for key, val in summary_values.items()]
    table = ax.table(cellText=table_data,
                     colLabels=['Stat', 'Value'],
                     cellLoc='left',
                     colLoc='left',
                     loc='upper right',
                     bbox=[0.6, 0.6, 0.35, 0.35])  # [x, y, width, height]
    
    table.set_fontsize(8)
    table.scale(1, 1.2)

# Remove extra subplots if any
for j in range(num_plots, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()



# number_of_ads = train[train['Number_of_Ads'] > 4]
# number_of_ads

test = test[test['Number_of_Ads'] <= 4].reset_index(drop=True)
test = test[test['Episode_Length_minutes'] <= 150].reset_index(drop=True)


def compare_features_to_listening_time(df, features, target='Listening_Time_minutes', bandwidth=1.0, bins=30):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity
    import numpy as np

    for feature in features:
        values = df[feature].dropna().values
        target_vals = df[target].dropna().values

        fig, ax1 = plt.subplots(figsize=(10, 6))

        sns.histplot(values, bins=bins, color='skyblue', alpha=0.5, stat='density', ax=ax1, label=feature)

        sns.kdeplot(values, color='darkorange', fill=True, linewidth=2, ax=ax1)
        # sns.kdeplot(target_vals, color='green', linestyle='--', linewidth=2, ax=ax1, label=target)

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(values[:, None])
        eval_point = np.median(values)
        prob_density = np.exp(kde.score_samples([[eval_point]]))[0]

        ax1.set_title(f'{feature} vs. {target}', fontsize=15, fontweight='bold')
        ax1.set_xlabel(feature, fontsize=13)
        ax1.set_ylabel('Density', fontsize=13)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        print(f"[{feature}] KDE prob density at median ({eval_point:.2f}): {prob_density:.6f}")
        plt.tight_layout()
        plt.show()



features_to_compare = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage'
]

compare_features_to_listening_time(train, features_to_compare)


plt.figure(figsize=(8, 6))
sns.boxplot(data=train, x='Episode_Sentiment', y='Listening_Time_minutes', palette='coolwarm')
plt.title('Listening Time by Episode Sentiment')
plt.xlabel('Episode Sentiment')
plt.ylabel('Listening Time (minutes)')
plt.tight_layout()
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

X = train.drop(columns=['Listening_Time_minutes']) 
y = train['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 
                  'Number_of_Ads'] 

categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']  # Categorical features

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  
    ('scaler', StandardScaler()) 
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('lgbm', LGBMRegressor(
        n_estimators=1000,  
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='regression',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
        random_state=42
        ))  
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print(f'RMSE: {rmse}')



y_test_pred = pipeline.predict(test)

# turn your predictions into a list
test_predictions = y_test_pred.tolist()

submission_df = pd.DataFrame({'id': test['id'], 'Listening_Time_minutes': test_predictions})
submission_df.to_csv('submission.csv', index=False)




