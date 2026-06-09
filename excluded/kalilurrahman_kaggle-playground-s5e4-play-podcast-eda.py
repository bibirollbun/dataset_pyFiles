import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from cycler import cycler


raw_light_palette = [
    (0, 122, 255), # Blue
    (255, 149, 0), # Orange
    (52, 199, 89), # Green
    (255, 59, 48), # Red
    (175, 82, 222),# Purple
    (255, 45, 85), # Pink
    (88, 86, 214), # Indigo
    (90, 200, 250),# Teal
    (255, 204, 0)  # Yellow
]

raw_dark_palette = [
    (10, 132, 255), # Blue
    (255, 159, 10), # Orange
    (48, 209, 88),  # Green
    (255, 69, 58),  # Red
    (191, 90, 242), # Purple
    (94, 92, 230),  # Indigo
    (255, 55, 95),  # Pink
    (100, 210, 255),# Teal
    (255, 214, 10)  # Yellow
]

raw_gray_light_palette = [
    (142, 142, 147),# Gray
    (174, 174, 178),# Gray (2)
    (199, 199, 204),# Gray (3)
    (209, 209, 214),# Gray (4)
    (229, 229, 234),# Gray (5)
    (242, 242, 247),# Gray (6)
]

raw_gray_dark_palette = [
    (142, 142, 147),# Gray
    (99, 99, 102),  # Gray (2)
    (72, 72, 74),   # Gray (3)
    (58, 58, 60),   # Gray (4)
    (44, 44, 46),   # Gray (5)
    (28, 28, 39),   # Gray (6)
]


light_palette = np.array(raw_light_palette)/255
dark_palette = np.array(raw_dark_palette)/255
gray_light_palette = np.array(raw_gray_light_palette)/255
gray_dark_palette = np.array(raw_gray_dark_palette)/255

mpl.rcParams['axes.prop_cycle'] = cycler('color',dark_palette)
mpl.rcParams['figure.facecolor']  = gray_dark_palette[-2]
mpl.rcParams['figure.edgecolor']  = gray_dark_palette[-2]
mpl.rcParams['axes.facecolor'] =  gray_dark_palette[-2]

white_color = gray_light_palette[-2]
mpl.rcParams['text.color'] = white_color
mpl.rcParams['axes.labelcolor'] = white_color
mpl.rcParams['axes.edgecolor'] = white_color
mpl.rcParams['xtick.color'] = white_color
mpl.rcParams['ytick.color'] = white_color

mpl.rcParams['figure.dpi'] = 200

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


sns.palplot(dark_palette)
sns.palplot(light_palette)
sns.palplot(gray_light_palette)
sns.palplot(gray_dark_palette)


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print(train.shape)
print(test.shape)
print(sample_submission.shape)


train.head(100)


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


train.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


test.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


zero_data = ((train.iloc[:,:50]==0).sum() / len(train) * 100)[::-1]
fig, ax = plt.subplots(1,1,figsize=(10, 19))

ax.barh(zero_data.index, 100, color='#dadada', height=0.6)
barh = ax.barh(zero_data.index, zero_data, color=light_palette[1], height=0.6)
ax.bar_label(barh, fmt='%.01f %%', color='black')
ax.spines[['left', 'bottom']].set_visible(False)

ax.set_xticks([])

ax.set_title('# of Zeros (by feature)', loc='center', fontweight='bold', fontsize=15)    
plt.show()


label_dict = {val:idx for idx, val in enumerate(sorted(train['Listening_Time_minutes'].unique()))}
train['Listening_Time_minutes'] = train['Listening_Time_minutes'].map(label_dict)


df=train
plt.figure(figsize=(10, 6))
sns.barplot(x='Genre', y='Host_Popularity_percentage', data=df)
plt.title('Host Popularity Percentage by Genre')
plt.xlabel('Genre')
plt.ylabel('Host Popularity Percentage')
plt.show()




## 2. Box Plot of Episode Length
plt.figure(figsize=(10, 6))
sns.boxplot(x='Genre', y='Episode_Length_minutes', data=df)
plt.title('Episode Length Distribution by Genre')
plt.xlabel('Genre')
plt.ylabel('Episode Length (minutes)')
plt.show()




## 3. Pie Chart for Episode Sentiment Distribution
sentiment_counts = df['Episode_Sentiment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Episode Sentiment')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()




## 4. Scatter Plot of Host Popularity vs Listening Time
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Host_Popularity_percentage', y='Listening_Time_minutes', hue='Episode_Sentiment', style='Publication_Day', data=df)
plt.title('Host Popularity vs Listening Time')
plt.xlabel('Host Popularity Percentage')
plt.ylabel('Listening Time (minutes)')
plt.legend()
plt.show()




## 5. Count Plot of Number of Ads by Genre
plt.figure(figsize=(10, 6))
sns.countplot(x='Number_of_Ads', hue='Genre', data=df)
plt.title('Count of Number of Ads by Genre')
plt.xlabel('Number of Ads')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(12,8))
corr_matrix = df[['Host_Popularity_percentage', 'Episode_Length_minutes', 
                 'Number_of_Ads', 'Listening_Time_minutes']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=np.triu(corr_matrix))
plt.title('Feature Correlation Matrix')
plt.show()



sns.pairplot(df, hue='Genre', 
            vars=['Host_Popularity_percentage', 'Listening_Time_minutes', 
                  'Number_of_Ads'],
            plot_kws={'alpha':0.8})
plt.suptitle('Multivariate Relationships by Genre', y=1.02)
plt.show()



from pandas.plotting import parallel_coordinates

plt.figure(figsize=(12,6))
parallel_coordinates(df[['Genre', 'Host_Popularity_percentage', 
                        'Listening_Time_minutes', 'Number_of_Ads']], 
                    'Genre', colormap='tab10')
plt.title('Multivariate Profile Analysis')
plt.ylabel('Normalized Values')
plt.xticks(rotation=45)
plt.show()



sns.pairplot(df, hue='Genre', 
            vars=['Host_Popularity_percentage', 'Listening_Time_minutes', 
                  'Number_of_Ads'],
            plot_kws={'alpha':0.8})
plt.suptitle('Multivariate Relationships by Genre', y=1.02)
plt.show()



plt.figure(figsize=(12,8))
sns.scatterplot(data=df, x='Host_Popularity_percentage', 
               y='Listening_Time_minutes', hue='Genre',
               size='Number_of_Ads', sizes=(50, 300), 
               alpha=0.8, palette='viridis')
plt.title('Host Popularity vs Listening Time with Ad Frequency')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()



g = sns.FacetGrid(df, col='Publication_Time', row='Genre', margin_titles=True)
g.map(sns.histplot, 'Listening_Time_minutes', kde=True)
g.set_axis_labels('Listening Time (minutes)', 'Frequency')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Listening Time Distribution by Genre & Publication Time')
plt.show()



g = sns.FacetGrid(df, col='Genre', height=5, aspect=1)
g.map(sns.regplot, 'Host_Popularity_percentage', 'Listening_Time_minutes', ci=None, scatter_kws={'alpha':0.5})
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Regression of Host Popularity on Listening Time by Genre')
plt.show()



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df['Host_Popularity_percentage'], 
           df['Listening_Time_minutes'], 
           df['Number_of_Ads'], 
           c=df['Genre'].astype('category').cat.codes, cmap='viridis', s=100)

ax.set_xlabel('Host Popularity Percentage')
ax.set_ylabel('Listening Time (minutes)')
ax.set_zlabel('Number of Ads')
plt.title('3D Scatter Plot of Podcast Metrics')
plt.show()



sentiment_heatmap = df.pivot_table(index='Publication_Day', 
                                    columns='Genre', 
                                    values='Episode_Sentiment', 
                                    aggfunc=lambda x: (x == 'Positive').sum())

plt.figure(figsize=(12,6))
sns.heatmap(sentiment_heatmap, annot=True, fmt='d', cmap='Blues')
plt.title('Heatmap of Positive Episode Sentiments by Genre and Publication Day')
plt.ylabel('Publication Day')
plt.xlabel('Genre')
plt.show()



plt.figure(figsize=(12,6))
sns.kdeplot(data=df, x='Listening_Time_minutes', hue='Genre', fill=True, common_norm=False)
plt.title('KDE Plot of Listening Time by Genre')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Density')
plt.show()



plt.figure(figsize=(12,6))
sns.boxenplot(x='Episode_Sentiment', y='Episode_Length_minutes', data=df)
plt.title('Boxen Plot of Episode Length by Sentiment')
plt.xlabel('Episode Sentiment')
plt.ylabel('Episode Length (minutes)')
plt.show()



from math import pi

# Prepare data for Radar Chart
categories = ['Host_Popularity_percentage', 'Listening_Time_minutes', 'Number_of_Ads']
values = df[categories].mean().values.flatten().tolist()

# Number of variables
num_vars = len(categories)

# Compute angle for each axis
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
values += values[:1]  # Repeat the first value to close the circle
angles += angles[:1]   # Repeat the first angle to close the circle

# Draw radar chart
ax = plt.subplot(111, polar=True)
ax.fill(angles, values, color='blue', alpha=0.25)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Average Podcast Metrics Radar Chart')
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a pivot table for heat map
pivot_table = df.pivot_table(index='Genre', columns='Episode_Sentiment', aggfunc='size', fill_value=0)

# Plot heat map
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Heat Map of Episode Themes by Genre and Sentiment')
plt.xlabel('Episode Sentiment')
plt.ylabel('Genre')
plt.show()


