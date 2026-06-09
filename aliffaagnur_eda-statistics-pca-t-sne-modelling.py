%reload_ext cudf.pandas

import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import category_encoders as ce
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from cuml.decomposition import PCA
from cuml.manifold import TSNE

pd.set_option('display.float_format', '{:,.8f}'.format) 

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action = 'ignore', category = UserWarning)
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action='ignore', category = RuntimeWarning)
warnings.filterwarnings(action='ignore', category = SettingWithCopyWarning)


# LOAD DATASET

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv', sep=',')
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv', sep=',')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv', delimiter=',')

# CHECK SHAPE 
train_df.shape, test_df.shape, submission.shape



print('Train data : ')
display(train_df.head(5))

print('\nTest data')
display(test_df.head(5))

print('\nSubmission')
display(submission.head(5))


# PREPROCESSING BEFORE EDA

# MERGE THEM
combined_df = pd.concat((train_df, test_df), axis = 0)

# DROP DEPENDENT COLUMN
combined_df = combined_df.drop(labels = 'Listening_Time_minutes', axis = 1)

# DROP ALL MISSING VALUE
combined_df = combined_df.dropna()

combined_df


print('Train data : ')
display(train_df.info())

print('Test data : ')
display(test_df.info())


# NULL DATA COMPARISON 

# DISPLAY NULL VALUE\
print(f'Train data Null : ')
display(train_df.isna().sum())

print(f'\nTest data Null :')
display(test_df.isna().sum())

# ---------------------------------------------------

# CONVERT TO PERCENTAGE
train_null = train_df.isna().sum() / len(train_df)
test_null = test_df.isna().sum() / len(test_df)

# Mengubah ke bentuk data frame untuk perbandingan
null_comparison = pd.DataFrame({
    'train_null_proportion': train_null,
    'test_null_proportion': test_null
})

null_comparison = null_comparison.sort_values(by=['train_null_proportion', 'test_null_proportion'], ascending=False)

# Menampilkan hasil perbandingan
display(null_comparison)


# MAXIMUM VALUE FOR EACH NUMERIC COLUMNS

print(f'Maximum Episode Length : {train_df["Episode_Length_minutes"].max()}')
print(f'Maximum Number of Ads : {train_df["Number_of_Ads"].max()}')
print(f'Maximum Host Popularity Percentage : {train_df["Host_Popularity_percentage"].max()}')
print(f'Maximum Guest Popularity Percentage : {train_df["Guest_Popularity_percentage"].max()}')
print(f'Maximum Listening Time : {train_df["Listening_Time_minutes"].max()}')


# STATISTICS FOR LISTENING TIME

listening_time = train_df['Listening_Time_minutes'].describe()

listening_time_df = listening_time.reset_index()
listening_time_df.columns = ['Statistic', 'Value']

# PANDAS STYLER
styled_table = (listening_time_df.style
                .set_properties(**{'background-color': '#222', 
                                   'color': 'white', 
                                   'border-color': 'gray',
                                   'text-align': 'center'})
                .set_table_styles([{'selector': 'th', 
                                    'props': [('background-color', '#444'), 
                                              ('color', 'white'), 
                                              ('font-weight', 'bold'),
                                              ('text-align', 'center')]}])
                .set_caption("ğŸ“Š **Listening Time Statistics**"))

styled_table



# PODCAST NAME DISTRIBUTION COMPARISON

podcast_name = combined_df.groupby(by = 'Podcast_Name')['Genre'].count().sort_values(ascending = False)

dictionary = {'Podcast_Name' : podcast_name.index,
        'Count' : podcast_name.values }

df = pd.DataFrame(dictionary)


plt.figure(figsize=(12,15))

ax = sns.barplot(y = df['Podcast_Name'], x = df['Count'] ,palette= 'tab20', edgecolor = 'black', linewidth = 1.5, orient='h')

for bar in ax.containers:
    plt.bar_label(container= bar, fmt = '%d', label_type= 'edge', padding = 2, fontsize = 10, color = 'black')


plt.title('Top Podcast Name')
plt.show()


# WORDCLOUD

plt.figure(figsize = (12,6))

wordcloud = WordCloud(width = 400, height = 200, max_words = 50, random_state= 2025, colormap = 'rainbow')
wordcloud = wordcloud.generate(text = ' '.join(combined_df['Podcast_Name']))

plt.imshow(wordcloud)
plt.title('Podcast Name WordCloud')


# DISPLAY TOP 10 PODCAST WITH MOST DURATION EPISODE

duration_length = combined_df.groupby(by = 'Podcast_Name')['Episode_Length_minutes'].mean().sort_values(ascending = False).reset_index(drop=False).head(10)

plt.figure(figsize=(15,7))
ax = sns.barplot(data = duration_length, x = 'Episode_Length_minutes', y = 'Podcast_Name', palette='Set1', orient= 'h')

for bar in ax.containers:
    ax.bar_label(bar, fmt = '%.2f', label_type = 'edge', padding = 3, fontsize = 10, color = 'black')

plt.title('Top 10 Podcast with Most Duration Every Episode')
plt.show()


# PODCAST NAME BOX PLOT

plt.figure(figsize=(20,5))

sns.boxplot(data = combined_df, x = 'Podcast_Name', y = 'Episode_Length_minutes', palette='tab20')
plt.xticks(rotation = 40)
plt.ticklabel_format(axis = 'y', style = 'plain')

plt.title('Podcast Boxplot Based Duration each Episode')

plt.show()


# TOP 20 PODCAST WITH AVERAGE NUMBER OF ADS
podcast_ads = combined_df.groupby(by = 'Podcast_Name')['Number_of_Ads'].mean().sort_values(ascending= False).reset_index(drop= False).head(20)

custom_palette = [
    '#FF1493', '#FF4500', '#FFD700', '#32CD32', '#00BFFF',
    '#8A2BE2', '#FF69B4', '#FF6347', '#40E0D0', '#ADFF2F',
    '#DC143C', '#1E90FF', '#7B68EE', '#FF8C00', '#00FA9A',
    '#FF00FF', '#20B2AA', '#FFA07A', '#6A5ACD', '#EE82EE'
]


plt.figure(figsize=(12,7))

ax = sns.barplot(data = podcast_ads, x = 'Number_of_Ads', y = 'Podcast_Name', palette= custom_palette, edgecolor = 'black', linewidth = 1.5)

for bar in ax.containers:
    plt.bar_label(container= bar, fmt = '%.2f', label_type = 'edge', padding = 1.5, fontsize = 10, color = 'black' )

plt.title('Top 20 Podcast with Average Number of Ads')
plt.show()


# TOP 20 PODCAST WITH LONGEST AVERAGE LISTENERS

podcast_listeners = train_df.groupby(by = 'Podcast_Name')['Listening_Time_minutes'].mean().sort_values(ascending= False).reset_index(drop=False).head(20)

plt.figure(figsize=(13,8))

ax = sns.barplot(data = podcast_listeners, x = 'Listening_Time_minutes', y = 'Podcast_Name', palette= 'Set3', edgecolor='pink', linewidth = 1.5)
plt.title('Top 20 Podcast with the Longest Average Listeners Every Episode')

for bar in ax.containers:
    plt.bar_label(container = bar, fmt = '%.2f', label_type='center', padding = 3, fontsize = 11, color = 'black')

plt.show()


# GENRE DISTRIBUTION COMPARISON

train_genre = train_df['Genre'].sort_values(ascending = False)
test_genre  = test_df['Genre'].sort_values(ascending = False)

custom_palette_1 = ["#FF3B3F", "#FF9F1C", "#FFBF69", "#2EC4B6", "#E71D36"]
custom_palette_2 = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

plt.figure(figsize=(15,6))

# DISTRIBUTION FOR TRAIN DATA
plt.subplot(1,2,1)
ax1 = sns.countplot(x = train_genre, palette = custom_palette_1, edgecolor='black', linewidth = 1.2)
plt.title('Train data')
plt.xticks(rotation = 45)

for p in ax1.containers:
    ax1.bar_label(p, fmt='%d', label_type='edge', padding=3, fontsize=10, color='black')

# DISTRIBUTION FOR TEST DATA
plt.subplot(1,2,2)
ax2 = sns.countplot(x = test_genre, palette = custom_palette_2, edgecolor = 'black', linewidth = 1.5)
plt.title('Test data')
plt.xticks(rotation = 45)

for p in ax2.containers:
    ax2.bar_label(p, fmt='%d', label_type='edge', padding=3, fontsize=10, color='black')

plt.suptitle('Genre Distribution Comparison')

plt.show()


# PUBLICATION DATA DISTRIBUTION COMPARISON

publish_train = train_df['Publication_Day'].sort_values(ascending= False)
publish_test = test_df['Publication_Day'].sort_values(ascending= False)

custom_palette_3 = ["#6A0572", "#AB83A1", "#FAD02E", "#F28D35", "#BC4B51"]

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
ax1 = sns.countplot(x = publish_train, palette= custom_palette_1, edgecolor = 'black', linewidth = 1.5)
plt.xticks(rotation = 45)
plt.title('Train data')

for p in ax1.containers:
    ax1.bar_label(p, fmt='%d', label_type='edge', padding=3, fontsize=10, color='black')

plt.subplot(1,2,2)
ax2 = sns.countplot(x = publish_test, palette= custom_palette_2, edgecolor ='black', linewidth = 1.5)
plt.xticks(rotation = 45)
plt.title('Test data')

for p in ax2.containers:
    ax2.bar_label(p, fmt='%d', label_type='edge', padding=3, fontsize=10, color='black')

plt.suptitle('Publication Day Distribution Comparison')

plt.show()


# PUBLICATION TIME DISTRIBUTION COMPARISON

publish_time_train = train_df['Publication_Time'].sort_values(ascending = False)
publish_time_test  = test_df['Publication_Time'].sort_values(ascending = False)

custom_palette_3 = {
    'Morning': "#FFB74D",     
    'Afternoon': "#42A5F5",     
    'Evening': "#AB47BC",       
    'Night': "#1A237E",         
}

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
ax1 = sns.countplot(x = publish_time_train, palette= custom_palette_3, edgecolor = 'black', linewidth = 1.5)
plt.xticks(rotation = 45)
plt.title('Train data')

for p in ax1.containers:
    ax1.bar_label(p, fmt='%d', label_type='edge', padding=3, fontsize=10, color='black')

plt.subplot(1,2,2)
ax2 = sns.countplot(x = publish_time_test, palette= custom_palette_3, edgecolor ='black', linewidth = 1.5)
plt.xticks(rotation = 45)
plt.title('Test data')

for p in ax2.containers:
    ax2.bar_label(p, fmt='%d', label_type='edge', padding=3, fontsize=10, color='black')

plt.suptitle('Publication Time Distribution Comparison')

plt.show()


# SENTIMENT DISTRIBUTION COMPARISON

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
sentiment_train = train_df['Episode_Sentiment'].value_counts()
plt.pie(sentiment_train, labels = sentiment_train.index, autopct='%1.1f%%', startangle=90, shadow=True, colors=['#99ff99', '#ff9999', '#66b3ff'])
plt.title('Train data')


plt.subplot(1,2,2)
sentiment_test = test_df['Episode_Sentiment'].value_counts()
plt.pie(sentiment_train, labels = sentiment_train.index, autopct='%1.1f%%', startangle=90, shadow=True, colors=['#99ff99', '#ff9999', '#66b3ff'])
plt.title('Test data')

plt.suptitle('Average Sentiment')

plt.show()



# EPISODE LENGTH MINUTES DISTRIBUTION

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.histplot(x = train_df['Episode_Length_minutes'], color='green', bins = 20)
plt.title('Train data')

plt.subplot(1,2,2)
sns.histplot(x = test_df['Episode_Length_minutes'], color = 'green', bins = 20)
plt.title('Test data')

plt.suptitle('Episode Length Minutes Distribution')
plt.show()


print(f'Maximum Episode length in Train data : {train_df["Episode_Length_minutes"].max()}')
print(f'Maximum Episode length in Test data : {test_df["Episode_Length_minutes"].max()}')


# BOX PLOT FOR EPISODE LENGTH

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
sns.boxplot(y = train_df['Episode_Length_minutes'], color = 'green')
plt.title('Train data')

plt.subplot(1,2,2)
sns.boxplot(y = test_df['Episode_Length_minutes'], color = 'green')
plt.title('Test data')

plt.suptitle('Check Outlier for Episode Length Distribution')

plt.show()


# HOST POPULARITY PERCENTAGE DISTRIBUTION

host_popularity = train_df['Host_Popularity_percentage']

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,6))

sns.histplot(x = train_df['Host_Popularity_percentage'],color='skyblue', ax = axes[0])
axes[0].set_title('Train data')

sns.histplot(x = test_df['Host_Popularity_percentage'], color='skyblue', ax = axes[1])
axes[1].set_title('Test data')

plt.suptitle('Host Popularity Distribution Comparison')

plt.show()


# CHECK OUTLIER FROM POPULARITY DISTRIBUTION COMPARISON 

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6))

sns.boxplot(y = train_df['Host_Popularity_percentage'], color = 'lightblue', ax = axes[0])
axes[0].set_title('Train data')

sns.boxplot(y = test_df['Host_Popularity_percentage'], color = 'lightblue', ax = axes[1])
axes[1].set_title('Test data')

plt.suptitle('Check Outlier From Popularity Distribution')

plt.show()


# GUEST POPULARITY DISTRIBUTION 

fig, axes = plt.subplots(nrows= 1, ncols= 2, figsize=(12,6))

custom_palette_4 = [
    '#FF5733', '#FFC300', '#DAF7A6', '#581845', '#C70039',
    '#900C3F', '#FF33FF', '#33FF57', '#FFAA33', '#33A3FF',
    '#FF3366', '#66FF33', '#3366FF', '#33FFFF', '#FF33AA',
    '#AA33FF', '#33FFAA', '#AAFF33', '#FFA633', '#33FFA6'
]


sns.histplot(data = train_df, x = 'Guest_Popularity_percentage', bins = 20, color = 'green', ax = axes[0])
axes[0].set_title('Train data')

sns.histplot(data = test_df, x = 'Guest_Popularity_percentage', bins = 20, color = 'green', ax = axes[1])
axes[1].set_title('Test data')

plt.suptitle('Guest Popularity Distribution')


# LISTENING TIME DISTRIBUTION 

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)

sns.histplot(data = train_df, x = 'Listening_Time_minutes', color = 'gold', bins= 100)
plt.title('Listening Time Histogram')

plt.subplot(1,2,2)
sns.boxplot(data= train_df, y = 'Listening_Time_minutes', color = 'gold')
plt.title('Listening Time Boxplot')

plt.suptitle('Listening Time (Minutes) Distribution')

plt.show()


# IF MISSING VALUE FILLED WITH MEAN

concat_df = pd.concat((train_df, test_df), axis = 0)
missing_df = concat_df.copy()

extreme_missing_columns = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6))

for i, feature in enumerate(extreme_missing_columns):

    missing_df[feature] = missing_df[feature].fillna( missing_df[feature].mean())   # ---> FILL FEATURE WITH MISSING WITH VALUE OF THEIR MEAN

    sns.distplot(concat_df[feature], bins = 40, kde = True, hist= False, color='red', label = 'Original', ax= axes[i])
    sns.distplot(missing_df[feature], bins = 40, kde = True, hist= False, color='yellow', label = 'Filled with Mean', ax = axes[i])
    axes[i].set_title(feature)
    axes[i].legend()

plt.suptitle('What if a Missing Value filled with Mean')
plt.show()


# IF MISSING VALUE FILLED WITH MEDIAN

concat_df = pd.concat((train_df, test_df), axis = 0)
missing_df = concat_df.copy()

extreme_missing_columns = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6))

for i, feature in enumerate(extreme_missing_columns):

    missing_df[feature] = missing_df[feature].fillna( missing_df[feature].median())   # ---> FILL FEATURE WITH MISSING WITH VALUE OF THEIR MEAN

    sns.distplot(concat_df[feature], bins = 40, kde = True, hist= False, color='red', label = 'Original', ax= axes[i])
    sns.distplot(missing_df[feature], bins = 40, kde = True, hist= False, color='yellow', label = 'Filled with Median', ax = axes[i])
    axes[i].set_title(feature)
    axes[i].legend()

plt.suptitle('What if a Missing Value filled with Median')
plt.show()


# IF MISSING VALUE FILLED USING KNN-IMPUTER (YOU CAN UNCOMMENT THIS CELL)

#concat_df = pd.concat((train_df, test_df), axis = 0)
#missing_df = concat_df.copy()

#extreme_missing_columns = ['Episode_Length_minutes', 'Guest_Popularity_percentage']

#fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(12,6))

#for i, feature in enumerate(extreme_missing_columns):

#    # KNN IMPUTER
#    imputer = KNNImputer(n_neighbors= 3, weights= 'distance')
#    missing_df[feature] = imputer.fit_transform(missing_df[[feature]])

#    sns.distplot(concat_df[feature], bins = 40, kde = True, hist= False, color='red', label = 'Original', ax= axes[i])
#    sns.distplot(missing_df[feature], bins = 40, kde = True, hist= False, color='yellow', label = 'Filled with Median', ax = axes[i])
#    axes[i].set_title(feature)
#    axes[i].legend()

#plt.suptitle('What if a Missing Value filled Using KNN-Imputer')
#plt.show()



type(train_df)


# DISPLAY HEATMAP BETWEEN INDEPENDENT VARIABLE AND DEPENDENT VARIABLE 

# GET ONLY NUMERIC FEATURE
numerical_column = train_df.select_dtypes(include='number').drop(columns='id')


if hasattr(numerical_column, "to_pandas"):
    numerical_column = numerical_column.to_pandas()
    
# CREATE NON LINEAR CORRELATION
corr_matrix = numerical_column.corr(method = 'spearman')[['Listening_Time_minutes']]

# Create heatmap
sns.heatmap(
    corr_matrix.values,            # pass in plain numpy array
    vmin=-1, vmax=1,
    cmap="Spectral",
    annot=True, fmt=".2f",
    linewidths=0.5, linecolor="black",
    annot_kws={"size":12, "weight":"bold"},
    cbar_kws={"shrink":0.9, "aspect":40},
    yticklabels=corr_matrix.index,  # set the row labels yourself
    xticklabels=corr_matrix.columns
)

plt.title('Correlation with Listening Time', fontdict = {'size' : 12, 'weight' : 'bold'})
plt.show()


# VISUALIZE SCATTER PLOT 

# Drop the 'Listening_Time_minutes' column from numerical_column
numerical_columns = numerical_column.drop(columns='Listening_Time_minutes')

# CREATE CANVAS AND SUBPLOTS
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

# LOOP THROUGH NUMERICAL COLUMN
for i, feature in enumerate(numerical_columns):
    ax = axes[i // 2, i % 2]
    sns.scatterplot(
        x=train_df[feature], 
        y=train_df['Listening_Time_minutes'], 
        ax=ax,
        color='purple',  
        marker='o',  
        s=100,  
        edgecolor='black', 
        alpha=0.7  
    )
    
    ax.set_title(f'Correlation {feature} vs Listening Time', fontsize=10, fontweight='bold', color='darkblue')
    ax.set_xlabel(feature, fontsize=12, color='darkred', fontweight='light')
    ax.set_ylabel('Listening Time (minutes)', fontsize=10, color='darkred', fontweight='light')
    
    # ADD GRIDLINES
    ax.grid(True, linestyle='--', alpha=0.5)

    
    ax.tick_params(axis='x', rotation=45, labelsize=10, labelcolor='darkgreen')
    ax.tick_params(axis='y', rotation=0, labelsize=10, labelcolor='darkgreen')


plt.suptitle('Correlation of features that affect Listening Time')
plt.tight_layout()

plt.savefig('scatter_plot_listening_time.png', dpi=300, bbox_inches='tight') 

plt.show()



# CORRELATION INDEPENDEN AND CHECKING MULTICOLINEARITY

corr_matrix = numerical_column.corr(method = 'spearman')

plt.figure(figsize=(12,8))

sns.heatmap(data= corr_matrix.values, 
            vmin= -1, vmax= 1, 
            cmap= 'Spectral', 
            annot= True, fmt = '.2f', 
            linewidths= 0.5, 
            linecolor= 'gray', 
            annot_kws = {'size' : 12, 'weight' : 'bold'}, 
            cbar_kws= {'shrink' : 0.75, 'aspect' : 40})

plt.title('Checking Multicolinearity', fontdict = {'size' : 14, 'weight' : 'bold'}, pad= 10)
plt.show()


genre = train_df.groupby(by = 'Genre')['Listening_Time_minutes'].mean().sort_values(ascending= False).reset_index(drop=False).head(20)

plt.figure(figsize=(10,5))

ax = sns.barplot(data = genre, x = 'Listening_Time_minutes', y = 'Genre', palette= 'Set3', edgecolor='pink', linewidth = 1.5)
plt.title('Genre with Longest Listening Time')

for bar in ax.containers:
    plt.bar_label(container = bar, fmt = '%.2f', label_type='center', padding = 3, fontsize = 11, color = 'black')

plt.show()



train_df.groupby('Episode_Sentiment')['Listening_Time_minutes'].mean()



publication_day = train_df.groupby(by = 'Publication_Day')['Listening_Time_minutes'].mean().sort_values(ascending= False).reset_index(drop=False).head(20)

plt.figure(figsize=(10,5))

ax = sns.barplot(data = publication_day, x = 'Listening_Time_minutes', y = 'Publication_Day', palette= 'Set3', edgecolor='pink', linewidth = 1.5)
plt.title('Day with longest listening duration')

for bar in ax.containers:
    plt.bar_label(container = bar, fmt = '%.2f', label_type='center', padding = 3, fontsize = 11, color = 'black')

plt.show()


# PUBLICATION TIME WITH THE LONGEST AVERAGE LISTENERS

publication_time = train_df.groupby(by = 'Publication_Time')['Listening_Time_minutes'].mean().sort_values(ascending= False).reset_index(drop=False).head(20)

plt.figure(figsize=(10,5))

ax = sns.barplot(data = publication_time, x = 'Listening_Time_minutes', y = 'Publication_Time', palette= 'Set3', edgecolor='pink', linewidth = 1.5)
plt.title('Publication Time with longest listening duration')

for bar in ax.containers:
    plt.bar_label(container = bar, fmt = '%.2f', label_type='center', padding = 3, fontsize = 11, color = 'black')

plt.show()


# Groupby combined by day and time
combined = train_df.groupby(['Publication_Day', 'Publication_Time'])['Listening_Time_minutes'] \
                   .mean().reset_index()

# SORT TO DESCENDING
combined = combined.sort_values(by='Listening_Time_minutes', ascending=False)

# ------------------------------------------------------------------

plt.figure(figsize=(12,6))

ax = sns.barplot(data=combined, 
                 x='Listening_Time_minutes', 
                 y='Publication_Day', 
                 hue='Publication_Time', 
                 palette='Set3', 
                 edgecolor='pink', 
                 linewidth=1.5)

plt.title('Listening Duration by Day and Time of Publication')
plt.xlabel('Average Listening Time (minutes)')
plt.ylabel('Publication Day')

# LABEL FOR EACH BAR
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=9, color='black')

plt.legend(title='Publication Time', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# PREPROCESSING BEFORE DIMENSION REDUCTION


# ONE - HOT ENCODING
#podcast_name_encoded = pd.get_dummies(data = train_df['Podcast_Name'])
#genre_encoded = pd.get_dummies(data = train_df['Genre'])
#public_day_encoded = pd.get_dummies(data = train_df['Publication_Day'])
#public_time_encoded = pd.get_dummies(data = train_df['Publication_Time'])

# LABEL ENCODING
encoding = LabelEncoder()
train_df['Podcast_Name_encoded'] = encoding.fit_transform(train_df['Podcast_Name'])
train_df['Episode_Title_encoded'] = encoding.fit_transform(train_df['Episode_Title'])
train_df['Genre_encoded'] = encoding.fit_transform(train_df['Genre'])
train_df['Publication_Day_encoded'] = encoding.fit_transform(train_df['Publication_Day'])
train_df['Publication_Time_encoded'] = encoding.fit_transform(train_df['Publication_Time'])


# ----------------------------------------------------------------------------

# ORDINAL ENCODING
sentiment_label = {'Negative' : 0, 'Neutral' : 1, 'Positive' : 2}
train_df['Sentiment_encoded'] = train_df['Episode_Sentiment'].map(sentiment_label)

# ----------------------------------------------------------------------------

# IMPUTE MISSING VALUES
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].mean())
train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].mean())
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].fillna(train_df['Number_of_Ads'].mean())


# STANDARD SCALER FOR NUMERICAL FEATURE
scaler = StandardScaler()
train_df[['Episode_Length_minutes',	'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']] = scaler.fit_transform(train_df[['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']])

# STANDARD SCALER FOR CATEGORICAL FEATURE
scaler = StandardScaler()
train_df[['Podcast_Name_encoded','Episode_Title_encoded', 'Genre_encoded', 'Publication_Day_encoded', 'Publication_Time_encoded', 'Sentiment_encoded']] = scaler.fit_transform(train_df[['Podcast_Name_encoded','Episode_Title_encoded', 'Genre_encoded', 'Publication_Day_encoded', 'Publication_Time_encoded', 'Sentiment_encoded']])

# ----------------------------------------------------------------------------

# CHOOSE COLUMNS TO TRANSFORM
cols_to_transform = train_df[['Podcast_Name_encoded', 'Episode_Title_encoded','Genre_encoded','Publication_Day_encoded','Publication_Time_encoded',
                              'Episode_Length_minutes','Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Sentiment_encoded']]

# CONVERT TO NUMPY 
#data_array = cols_to_transform.to_numpy()

cols_to_transform


# PCA

import cudf

#cols_to_transform_gpu = cudf.from_pandas(cols_to_transform)  # ---> CONVERT PANDAS TO GPU BECAUSE WE USE CUML PCA 

# PCA
pca = PCA(n_components = 10)
data_pca = pca.fit_transform(cols_to_transform)

# DISPLAY PCA INFORMATION
print("Total explained variance by 10 components:")
for i, ratio in enumerate(pca.explained_variance_ratio_ ):
    print(f"Component {i+1}: {ratio*100:.2f}%")


# SHOW THE MOST INFLUENTIAL FEATURES IN EACH COMPONENT
components_df = pd.DataFrame(pca.components_.T.values,
                             index = cols_to_transform.columns , 
                             columns=[f'Component {i+1}' for i in range(pca.components_.shape[0])])
components_df= components_df.abs()  # ---> CONVERT TO ABSOLUTE VALUE

# SORT BY DESCENDING
components_df = components_df.sort_values(by = ['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7', 'Component 8', 'Component 9', 'Component 10'], ascending = [False, False, False, False, False, False, False, False, False, False])

components_df


explained_var = pca.explained_variance_ratio_  

plt.figure(figsize=(10,6))
plt.plot(range(1, len(explained_var)+1), explained_var * 100, marker='o', linestyle='--', color='b')
plt.title('Scree Plot - Explained Variance per Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.grid(True)
plt.xticks(np.arange(1, len(explained_var)+1))
plt.show()


components_df.iloc[:, :5]


# VISUALIZATION HEATMAP AND TAKE THE 5 FIRST COMPONENT
plt.figure(figsize=(12,6))
sns.heatmap(np.array(components_df.iloc[:, :]), annot=True, cmap='viridis', yticklabels=cols_to_transform.columns)
plt.title('Top Feature Contribution to First 5 Principal Components')
plt.ylabel('Features')
plt.xlabel('Principal Components')
plt.show()


%%time
# LETS VISUALIZE PCA ON SCATTER PLOT


# DISPLAY SCATTER PLOT
plt.figure(figsize=(15,25))

# 2D SCATTER PLOT
for i in range(9):
    plt.subplot(5,2,i+1)
    scatter = plt.scatter(x = data_pca.iloc[:, i], 
                          y = data_pca.iloc[:, i+1], 
                          c = train_df['Listening_Time_minutes'])
    
    plt.title(f'PC{i+1} And PC{i+2}')
    
    handles, labels = scatter.legend_elements()

    # ADJUST LEGEND
    plt.legend(handles, labels, title="Listening Time (minutes)", 
               loc='upper right',   
               fontsize=6,       
               title_fontsize=8,   # Ukuran font untuk judul
               markerscale=1,       # Ukuran marker di legenda
               borderpad=0.5,         # Jarak antara kotak legenda dan teks
               handlelength=1)      # Panjang garis legenda


plt.savefig('2d_pca.png', dpi = 300)
plt.show()


%%time
# FIT TSNE AND FINDING HIDDEN PATTERNS FOR 2D 

tsne = TSNE(n_components = 2,          
            perplexity = 50,         
            early_exaggeration = 12, 
            n_iter = 1000,           
            n_iter_without_progress = 300, 
            min_grad_norm = 1e-7,          
            metric = 'euclidean',               
            metric_params = None,          
            init = 'pca',                  
            verbose = 1,
            random_state = 2025,
            method = 'barnes_hut', 
            angle = 0.5)

# FIT TRANSFORM
data_reduced = tsne.fit_transform(cols_to_transform)


cols_to_transform.columns


# VISUALIZATION 2D TSNE

plt.figure(figsize=(20,20))
scatter = plt.scatter(x = data_reduced.loc[:, 0], y = data_reduced.loc[:, 1], c = train_df['Listening_Time_minutes'], cmap='viridis', alpha = 0.6)

handles, labels = scatter.legend_elements()
plt.legend(handles, labels, title="Listening Time (minutes)", 
               loc='upper right',   
               fontsize=10,       
               title_fontsize=10,   # Ukuran font untuk judul
               markerscale=1.5,       # Ukuran marker di legenda
               borderpad=0.7,         # Jarak antara kotak legenda dan teks
               handlelength=1.5)      # Panjang garis legenda

plt.title('t-SNE 2D Scatter Plot')

plt.savefig('2d_tsne_podcast.png', dpi = 300)


#from sklearn.utils import resample
#from sklearn.metrics import pairwise_distances

# TAKE ONLY 500 SUBSET
#X_tsne_subset = resample(data_reduced, n_samples=500, random_state=42)

# CALCULATE DISTANCE MATRIX
#distance_matrix = pairwise_distances(X_tsne_subset, metric='euclidean')

# PLOT HEATMAP
#plt.figure(figsize=(8, 6))
#sns.heatmap(distance_matrix, cmap="YlGnBu", cbar=True)

#plt.title('t-SNE Podcast Heatmap')
#plt.show()



# we only take 500 random subsets of the entire data, due to memory and computational limitations.
# on the heatmap, the distance between 2 data points is visible. The heatmap with a lighter color indicates that the distance between the 2 data points is close. 
# based on the conclusion, the distance of all data points in the heatmap is almost the same, which is close to each other.</strong>


%reset -f


import pandas as pd
import numpy as np
import copy
import seaborn as sns
from matplotlib import pyplot as plt
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
import catboost
import lightgbm



# LOAD DATASETS

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')

train_df.shape, test_df.shape, submission.shape


# ORIGINAL DATASET 

original_df = pd.read_csv(r'/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')

original_df = original_df.dropna(subset='Listening_Time_minutes')  # ---> DROP TARGET COLUMN WITH NULL VALUES
original_df = original_df.drop_duplicates()

original_df.shape


print('Train data : ')
display(train_df)

print('Test data : ')
display(test_df)

print('Original data : ')
display(original_df)


# COMBINE ORIGINAL DATA WITH TRAIN DATA
train_df = pd.concat([train_df, original_df], axis=0, ignore_index=True)

train_df


# CLIP FEATURE THAT HAS SIGNIFICANT OUTLIER

numeric_cols = train_df.select_dtypes(include = 'number').columns.drop(labels = 'Listening_Time_minutes')

display(train_df[numeric_cols].max())
display(test_df[numeric_cols].max())

# CLIPPING
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].clip(lower = None, upper = 120)
train_df['Number_of_Ads'] = train_df['Number_of_Ads'].clip(lower = None, upper = 3)

test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].clip(lower = None, upper = 120)
test_df['Number_of_Ads'] = test_df['Number_of_Ads'].clip(lower = None, upper= 3)



# LABEL ENCODING

categorical_columns = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])  



# SPLIT DATA

x = train_df.drop(labels = ['id', 'Listening_Time_minutes'], axis = 1)
y = train_df['Listening_Time_minutes']

test_df = test_df.drop(columns=['id'])

x.shape,  y.shape


%%time
# LGBM


# TRANSFORM INTO CATEGORICAL OBJECT/DATA TYPE
#for col in categorical_columns:        # ----> UNCOMMENT THIS IF U WANT USING TARGET ENCODER BELOW
#    x[col] = x[col].astype('category')
#    test_df[col] = test_df[col].astype('category')


# DEFINE KFOLD
kfold = KFold(n_splits=10, shuffle = False)
#rkfold = RepeatedKFold(n_splits = 10, n_repeats = 2, random_state= 2025)

# ---------------------------------------------------------------------

oof_train , oof_val , test_pred = [] , [] , []

repeat_num =  1
fold_num = 1
prev_repeat = 0
interaction_features = []

# DO KFOLD
for i, (train_index, val_index) in enumerate(kfold.split(x)):

    #current_repeat = i // 5 + 1            # ----> Applicable only when using rkfold to display information
    #if current_repeat != prev_repeat:
    #    print(f"\n>>> Repeat {current_repeat}")
    #    fold_num = 1
    #    prev_repeat = current_repeat

    # SPLIT
    x_train, x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    x_test = test_df.copy()

    # TARGET ENCODER
    #encoder = TargetEncoder(smoothing= 1)     # ---->  The RMSE value of using the target encoder is higher than using the label encoder.
    
    #x_train.loc[:, categorical_columns] = encoder.fit_transform(x_train[categorical_columns], y_train)
    #x_val.loc[:, categorical_columns]   = encoder.transform(x_val[categorical_columns])
    #x_test.loc[:, categorical_columns]  = encoder.transform(x_test[categorical_columns])

    
    #combined_features = list(x_train.columns)
    #
    # FEATURE INTERACTION ( FEATURE ENGINEERING)
    #for i_f, col1 in enumerate(combined_features):
    #    for col2 in combined_features[i_f + 1:]:
    #        new_col = f'{col1}_x_{col2}'
    #        x_train[new_col] = x_train[col1] * x_train[col2]
    #        x_val[new_col] = x_val[col1] * x_val[col2]
    #        x_test[new_col] = x_test[col1] * x_test[col2]
    #        # SAVE NAME OF FEATURE ONCE
    #        if i == 0:
    #            interaction_features.append(new_col)
    #
    #if i == 0:
    #    print(f'There are {len(interaction_features)} New Features')
    #    print(interaction_features)

    
    # LGBM 
    light = lightgbm.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=1024,
        max_depth=-1,
        colsample_bytree=0.7,
        max_bin=1024,
        objective='regression',
        random_state=42,
        force_col_wise= True,
        n_jobs=-1,
        device='cpu'
    )

    light.fit(x_train, y_train, 
              eval_set=(x_val, y_val), 
              eval_metric='RMSE' ,
              callbacks=[lightgbm.early_stopping(stopping_rounds = 100), 
                         lightgbm.log_evaluation(0), 
                         lightgbm.record_evaluation({})])
    
    # PREDICT
    y_pred_train = light.predict(x_train)
    y_pred_val   = light.predict(x_val)

    # CALCULATE RMSE
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_val   = mean_squared_error(y_val, y_pred_val, squared=False)


    # PRINT CURRENT SCORE
    print(f'Fold {fold_num} : Train data RMSE = {rmse_train}, Val data RMSE : {rmse_val}\n')
    fold_num += 1

    oof_train.append(rmse_train)
    oof_val.append(rmse_val)

    # PREDICT TEST DATA
    y_prediction = light.predict(x_test)
    test_pred.append(y_prediction)

print(f'\nOverall Train data RMSE : {np.mean(oof_train)}')
print(f'\nOverall Val data RMSE   : {np.mean(oof_val)}')


# FEATURE IMPORTANCES 

lightgbm.plot_importance(booster = light, importance_type = 'gain', figsize=(8,13))
plt.title("Feature Importance (Gain) LGBM")
plt.savefig('Feature Importance (Gain) LGBM', bbox_inches = 'tight')
plt.show()


lightgbm.plot_importance(booster = light, importance_type = 'split', figsize=(8,13))
plt.title('Split Importance LGBM')
plt.savefig('Split Importance .png', bbox_inches = 'tight')


# SAVE SUBMISSION

submission['Listening_Time_minutes'] = np.mean(test_pred, axis = 0)

submission.to_csv(r'submission.csv', index= False)


submission

