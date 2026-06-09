import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from category_encoders import MEstimateEncoder, TargetEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import gaussian_kde


import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=FutureWarning)


sns.set()
pd.set_option('display.float_format', '{:.2f}'.format)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_train.isna()


df_train.head()


df_check_values = pd.DataFrame({'NA_count': df_train.isna().sum(), 
                               'NA_%': round(df_train.isna().sum() / len(df_train) * 100, 2), 
                               'INF_count': df_train.isin([np.inf, -np.inf]).sum(),
                               'INF_%': df_train.isin([np.inf, -np.inf]).sum() / len(df_train) * 100,
                               'Zero_count': (df_train == 0).sum(),
                               'Zero_%': round((df_train == 0).sum() / len(df_train) * 100, 4)})


# df_check_values

def highlight_high(x):
    color = '#BE3D2A' if x > 10 else ''
    return f'background-color: {color}'

df_check_values.style.applymap(highlight_high)



df_train.duplicated().sum()


df_numeric = df_train[['Episode_Length_minutes', 'Host_Popularity_percentage', 'Listening_Time_minutes',
                       'Guest_Popularity_percentage', 'Number_of_Ads']].fillna(0)



fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
axes = axes.flatten()

for i, col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 
                       'Guest_Popularity_percentage', 'Number_of_Ads']):
    sns.scatterplot(data=df_numeric, x=col, y=df_numeric.Listening_Time_minutes, 
                    ax=axes[i])


# plt.tight_layout()
plt.show()


df_num_sample = df_numeric.sample(n=200, random_state=42)


fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
axes = axes.flatten()

for i, col in enumerate(['Episode_Length_minutes', 'Host_Popularity_percentage', 
                       'Guest_Popularity_percentage', 'Number_of_Ads']):
    sns.scatterplot(data=df_num_sample, x=col, y=df_num_sample.Listening_Time_minutes, 
                    ax=axes[i])


# plt.tight_layout()
fig.suptitle('Sample 200 Numeric Rows from the Dataset', fontsize=26, y=1.01)
plt.tight_layout()
plt.show()


# Let's check Number of Ads outliers

df_train[df_train.Number_of_Ads > 3]


ids_to_change = df_train[df_train.Number_of_Ads > 3]['id']
ids_to_change


df_train.loc[ids_to_change, 'Number_of_Ads'] = 0
df_train.loc[[211159, 495919], 'Number_of_Ads'] = 3


## Let's also explore Episode length outliers

df_train[df_train.Episode_Length_minutes >150]


df_train[df_train.Genre == 'Lifestyle'][['Episode_Length_minutes','Listening_Time_minutes']].mean()


df_train.loc[101637, 'Episode_Length_minutes'] = 64.8371


host_over100_percentage = df_train[df_train['Host_Popularity_percentage'] > 100]
guest_over100_percentage = df_train[df_train['Guest_Popularity_percentage'] > 100]
print(f'{len(host_over100_percentage)} records where the host_popularity_percentage exceeds 100%')
print(f'{len(guest_over100_percentage)} records where the guest_popularity_percentage exceeds 100%')


host_mean =  df_train[~(df_train['Host_Popularity_percentage'] > 100)]['Host_Popularity_percentage'].mean()
guest_mean = df_train[~(df_train['Guest_Popularity_percentage'] > 100)]['Guest_Popularity_percentage'].mean()


df_train.loc[host_over100_percentage['id'], 'Host_Popularity_percentage'] = host_mean
df_train.loc[guest_over100_percentage['id'], 'Guest_Popularity_percentage'] = guest_mean


highlisten_len_df = df_train[df_train['Episode_Length_minutes'] < df_train['Listening_Time_minutes']]
highlisten_len_df.agg({"Episode_Length_minutes" : ('count', 'mean', 'max'), "Listening_Time_minutes" : ('count','mean', 'max')})


df_train.loc[highlisten_len_df['id'], 'Listening_Time_minutes'] = df_train['Episode_Length_minutes']


df_train.describe()


df_categorical = df_train.select_dtypes('object')


df_categorical.describe(include='all')


fig, axes = plt.subplots(3, 2, figsize=(15, 20), gridspec_kw = {'height_ratios': [1.3, 0.3, 0.15]} )
axes = axes.flatten()

for i, col in enumerate(df_categorical.columns):
    sns.boxplot(data=df_train, y=col, x='Listening_Time_minutes', ax=axes[i], palette = 'Spectral')


# plt.tight_layout()
fig.suptitle('Distribution of Listening Time Across Categories', fontsize=26, y=1.01)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(15, 20), gridspec_kw = {'height_ratios': [1.3, 0.3, 0.15]} )
axes = axes.flatten()

for i, col in enumerate(df_categorical.columns):
    sns.boxplot(data=df_train, y=col, x=df_train['Listening_Time_minutes']/df_train['Episode_Length_minutes'].fillna(df_train['Listening_Time_minutes']+5), ax=axes[i], palette = 'Spectral')


# plt.tight_layout()
fig.suptitle('Listening/Episode Time Ratio Across Categories', fontsize=26, y=1.01)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.pointplot(df_train, y='Listening_Time_minutes', hue = 'Episode_Sentiment', x='Publication_Time')
plt.title('Mean Listening Time by Publication Time and Episode Sentiment')
plt.show()


weekday_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

df_train['Publication_Day'] = pd.Categorical(df_train['Publication_Day'], 
                                              categories=weekday_order, 
                                              ordered=True)

plt.figure(figsize=(9, 6))
sns.lineplot(df_train, y='Listening_Time_minutes', x='Publication_Day', markers='o', hue='Episode_Sentiment')
plt.ylim(41, 49)
plt.title('Mean Listening Time by Publication Day')
plt.show()


pareto_genre_plot = pd.DataFrame(df_train.Genre.value_counts())
pareto_genre_plot.sort_values('count', ascending=False)
pareto_genre_plot['cumulative_%'] = pareto_genre_plot['count'].cumsum() / pareto_genre_plot['count'].sum() * 100


plt.figure(figsize=(8, 5))
sns.barplot(data = pareto_genre_plot, x=pareto_genre_plot.index, y=pareto_genre_plot['count'], color='skyblue')
ax1 = plt.gca()
ax2 = ax1.twinx()
sns.lineplot(data=pareto_genre_plot, x=pareto_genre_plot.index, y='cumulative_%', marker='o', ax=ax2)
for i, row in pareto_genre_plot.iterrows():
    ax2.text(x=i, y=row['cumulative_%'] + 2,  # +2 to shift above the point
             s=f"{row['cumulative_%']:.1f}%", 
             ha='center', va='bottom', fontsize=10, color='black')

plt.title('Number of Podcasts in Different Genres')
ax1.set_ylabel("Count")
ax2.set_ylabel("Cumulative %")
ax2.set_ylim(0, 120)
ax1.grid(False) 
ax2.grid(False) 
ax1.tick_params(axis='x', labelrotation=45)

plt.title("Number of Podcasts in Different Genres")

plt.tight_layout()
plt.show()


def check_vif(df):
  df = df.select_dtypes(include=['number'])
  vif = pd.DataFrame()
  vif["feature"] = df.columns
  vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
  return vif


def transform_cat_df(df_with_cat: pd.DataFrame, target_name: str, smooth_factor: int) -> pd.DataFrame:
    """
    Transforms categorical columns using one-hot encoding for low-cardinality features, 
    frequency encoding for very high-cardinality features, and target encoding for others.
    Numeric columns are preserved and combined with encoded features.
    """
    
    cat_df = df_with_cat.select_dtypes(include=['object', 'category']).copy()
    final_df = pd.DataFrame(index=df_with_cat.index)  # Keep index aligned
    
    for col in cat_df.columns:
        nunique = cat_df[col].nunique()
        
        if nunique < 11:
            # One-Hot Encoding
            encoder = OneHotEncoder(drop='first', sparse=False)
            encoded_array = encoder.fit_transform(cat_df[[col]])
            encoded_cols = encoder.get_feature_names_out([col])
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=cat_df.index)
            final_df = final_df.join(encoded_df)
        
        elif nunique > 70:
            # Frequency Encoding
            freq_map = cat_df[col].value_counts(normalize=True).to_dict()
            freq_encoded = cat_df[col].map(freq_map)
            final_df[col + '_freq'] = freq_encoded
            
        else:
            # Target Encoding
            target_encoder = TargetEncoder(cols=[col], smoothing=smooth_factor)
            encoded_df = target_encoder.fit_transform(cat_df[[col]], df_with_cat[target_name])
            final_df = final_df.join(encoded_df)
    

    numeric_df = df_with_cat.select_dtypes(include=['int', 'float']).drop(columns=[target_name], errors='ignore')
    final_df = final_df.join(numeric_df)
    final_df[target_name] = df_with_cat[target_name]
    
    return final_df


# Handle NA values before proceeding with feature transformation and VIF/MI analysis

for col in df_train.columns:
    if df_train[col].isna().any():
        print(col)
        df_train[col].fillna(df_train[col].mean(), inplace=True)


## Let's convert categorical variables into numerical format.

new_df = transform_cat_df(df_train, target_name = 'Listening_Time_minutes', smooth_factor=4)
new_df


# COUNT VIF SCORES
vif_count = check_vif(new_df.drop(['Listening_Time_minutes', 'id'], axis=1))
vifs = vif_count.sort_values('VIF', ascending=False)
vifs.reset_index(drop=True, inplace=True)


plt.figure(figsize=(10, 8))
ax = plt.gca()
sns.barplot(data=vifs, y='feature', x='VIF', color='skyblue', ax=ax)

# Add VIF values as text labels on the bars
for i, row in vifs.iterrows():
    ax.text(x=row['VIF']+1, y=i, s=f"{row['VIF']:.1f}", 
            ha='left', va='center', fontsize=10, color='black')

plt.title('Variance Inflation Factor')
plt.tight_layout()
plt.show()


def count_mi_scores(X, y, entropy_estimate):
    """
    Calculates mutual information scores between features and a continuous target,
    and expresses how much information each feature provides about the target 
    as a percentage of the target's estimated entropy.
    """
    X = X.copy()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    
    # Estimate mutual information between each feature and the target
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    
    # Calculate the percentage of information each feature contributes
    info_percent = 100 * mi_scores / entropy_estimate
    
    mi_df = pd.DataFrame({
        'mi_scores': mi_scores,
        'info_percent': info_percent
    }, index=X.columns).sort_values(by='info_percent', ascending=True)
    
    return mi_df


def plot_mi_scores(df_scores):
    """
    Plots mutual information scores as a horizontal bar chart,
    annotated with the percentage of target information each feature explains.
    """
    df_sorted = df_scores.sort_values(by='mi_scores', ascending=True)
    scores = df_sorted['mi_scores']
    info_percent = df_sorted['info_percent']
    ticks = df_sorted.index
    width = np.arange(len(scores))

    # Plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.barh(width, scores, color='skyblue')
    ax.set_yticks(width)
    ax.set_yticklabels(ticks)
    ax.set_title("Proportion of Target Information Captured via Mutual Information", fontsize=14)
    ax.set_xlabel("MI Score")
    
    # Annotate with info_percent
    for i, (score, percent) in enumerate(zip(scores, info_percent)):
        ax.text(x=score + 0.01, y=i, s=f"{percent:.1f}%", 
                ha='left', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()


## KDE-Based Estimation of Differential Entropy from Sampled Continuous Data

# sample_for_entropy_count = new_df['Listening_Time_minutes'].sample(n=75000, random_state=42)
# kde = gaussian_kde(sample_for_entropy_count, bw_method='scott')  
# log_probs = np.log(kde.evaluate(sample_for_entropy_count))
# entropy = -np.mean(log_probs)


entropy = 4.649471119869842


mi_df = count_mi_scores(new_df.drop(['Listening_Time_minutes', 'id'], axis=1), new_df['Listening_Time_minutes'], entropy_estimate=entropy)


plot_mi_scores(mi_df)

