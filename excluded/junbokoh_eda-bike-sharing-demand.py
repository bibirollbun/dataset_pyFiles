import numpy as np
import pandas as pd # íŒ�ë‹¤ìŠ¤ ì�„í�¬íŠ¸

# ë�°ì�´í„° ê²½ë¡œ
data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv') # í›ˆë ¨ ë�°ì�´í„°
test = pd.read_csv(data_path + 'test.csv')   # í…ŒìŠ¤íŠ¸ ë�°ì�´í„°
submission = pd.read_csv(data_path + 'sampleSubmission.csv') # ì œì¶œ ìƒ˜í”Œ ë�°ì�´í„°


train.shape, test.shape


train.head()


train.info()


import missingno as msno

msno.matrix(train, figsize=(12,5))


test.head()


test.info()


msno.matrix(test, figsize=(12,5))


submission.head()


print(type(train['datetime'][100])) # datetime ë�°ì�´í„° íƒ€ì�…
print(train['datetime'][100]) # datetime 100ë²ˆì§¸ ìš”ì†Œ
print(train['datetime'][100].split()) # ê³µë°± ê¸°ì¤€ìœ¼ë¡œ ë¬¸ì��ì—´ ë‚˜ëˆ„ê¸°
print(train['datetime'][100].split()[0]) # ë‚ ì§œ
print(train['datetime'][100].split()[1]) # ì‹œê°„


train = pd.read_csv(data_path + 'train.csv', parse_dates=["datetime"])

# train['datetime'] = pd.to_datetime(train['datetime']) 


type(train['datetime'][100])


print(train['datetime'][100].year) # ì—°ë�„
print(train['datetime'][100].month) # ì›”
print(train['datetime'][100].day) # ì�¼


train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
train.shape


train.head(1)


# ìš”ì�¼ ìˆ«ì��(0=ì›”ìš”ì�¼, 6=ì�¼ìš”ì�¼) ì¶”ê°€
train["dayofweek"] = train["datetime"].dt.dayofweek

# ìš”ì�¼ ë¬¸ì��ì—´('Monday', 'Tuesday' ...) ì¶”ê°€
train["weekday"] = train["datetime"].dt.strftime('%A')

# ê²°ê³¼ í™•ì�¸
train.head()


train.drop(columns='dayofweek', inplace=True)


# season ìˆ«ì��ë¥¼ ë¬¸ì��ì—´ë¡œ ë³€í™˜
season_mapping = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
train["season"] = train["season"].map(season_mapping)

# weather ìˆ«ì��ë¥¼ ìš”ì•½ë�œ ë¬¸ì��ì—´ë¡œ ë³€í™˜
weather_mapping = {
    1: "Clear/Partly Cloudy",
    2: "Mist/Cloudy",
    3: "Light Snow/Rain",
    4: "Heavy Rain/Snow"
}
train["weather"] = train["weather"].map(weather_mapping)

# ê²°ê³¼ í™•ì�¸
train.head()


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline


# ìŠ¤í…� 1 : mí–‰ nì—´ Figure ì¤€ë¹„
figure, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2)
mpl.rc('font', size=14)       # í�°íŠ¸ í�¬ê¸° ì„¤ì •
mpl.rc('axes', titlesize=15)  # ê°� ì¶•ì�˜ ì œëª© í�¬ê¸° ì„¤ì •
plt.tight_layout()            # ê·¸ë�˜í”„ ì‚¬ì�´ì—� ì—¬ë°± í™•ë³´ 
figure.set_size_inches(10, 9) # ì „ì²´ Figure í�¬ê¸°ë¥¼ 10x9ì�¸ì¹˜ë¡œ ì„¤ì • 

# ìŠ¤í…� 2 : ê°� ì¶•ì—� ì„œë¸Œí”Œë¡¯ í• ë‹¹
# ê°� ì¶•ì—� ì—°ë�„, ì›”, ì�¼, ì‹œê°„, ë¶„, ì´ˆë³„ í�‰ê·  ëŒ€ì—¬ ìˆ˜ëŸ‰ ë§‰ëŒ€ ê·¸ë�˜í”„ í• ë‹¹
sns.barplot(data=train, x="year", y="count", ax=ax1)
sns.barplot(data=train, x="month", y="count", ax=ax2)
sns.barplot(data=train, x="day", y="count", ax=ax3)
sns.barplot(data=train, x="hour", y="count", ax=ax4)
sns.barplot(data=train, x="minute", y="count", ax=ax5)
sns.barplot(data=train, x="second", y="count", ax=ax6)

# ìŠ¤í…� 3 : ì„¸ë¶€ ì„¤ì •
# 3-1 : ì„œë¸Œí”Œë¡¯ì—� ì œëª© ë‹¬ê¸°
ax1.set(ylabel='Count',title="Rental amounts by year")
ax2.set(xlabel='month',title="Rental amounts by month")
ax3.set(xlabel='day', title="Rental amounts by day")
ax4.set(xlabel='hour', title="Rental amounts by hour")
ax5.set(xlabel='minute', title="Rental amounts by minute")
ax6.set(xlabel='second', title="Rental amounts by second")

# 3-2 : 1í–‰ì—� ìœ„ì¹˜í•œ ì„œë¸Œí”Œë¡¯ë“¤ì�˜ xì¶• ë�¼ë²¨ 90ë�„ íšŒì „
ax3.tick_params(axis='x', labelrotation=90)
ax4.tick_params(axis='x', labelrotation=90)


# 'year-month' ì»¬ëŸ¼ ìƒ�ì„±
train["year_month"] = train["datetime"].dt.strftime('%Y-%m')

# ì—°ë�„-ì›”ë³„ ëŒ€ì—¬ëŸ‰ ì§‘ê³„
monthly_rentals = train.groupby("year_month")["count"].sum().reset_index()

# ì‹œê°�í™”
plt.figure(figsize=(12, 6))
sns.barplot(data=monthly_rentals, x="year_month", y="count")
plt.xticks(rotation=90)
plt.xlabel("Year-Month")
plt.ylabel("Total Rentals")
plt.title("Total Bike Rentals by Year-Month")
plt.show()


# ìŠ¤í…� 1 : mí–‰ nì—´ Figure ì¤€ë¹„
figure, axes = plt.subplots(nrows=2, ncols=2) # 2í–‰ 2ì—´
plt.tight_layout()
figure.set_size_inches(12, 10)

# ìŠ¤í…� 2 : ì„œë¸Œí”Œë¡¯ í• ë‹¹
# ê³„ì ˆ, ë‚ ì”¨, ê³µíœ´ì�¼, ê·¼ë¬´ì�¼ë³„ ëŒ€ì—¬ ìˆ˜ëŸ‰ ë°•ìŠ¤í”Œë¡¯
sns.boxplot(x='season', y='count', data=train, ax=axes[0, 0])
sns.boxplot(x='weather', y='count', data=train, ax=axes[0, 1])
sns.boxplot(x='holiday', y='count', data=train, ax=axes[1, 0])
sns.boxplot(x='workingday', y='count', data=train, ax=axes[1, 1])

# ìŠ¤í…� 3 : ì„¸ë¶€ ì„¤ì •
# 3-1 : ì„œë¸Œí”Œë¡¯ì—� ì œëª© ë‹¬ê¸°
axes[0, 0].set(title='Box Plot On Count Across Season')
axes[0, 1].set(title='Box Plot On Count Across Weather')
axes[1, 0].set(title='Box Plot On Count Across Holiday')
axes[1, 1].set(title='Box Plot On Count Across Working Day')

# 3-2 : xì¶• ë�¼ë²¨ ê²¹ì¹¨ í•´ê²°
axes[0, 1].tick_params('x', labelrotation=10) # 10ë�„ íšŒì „


# ìŠ¤í…� 1 : mí–‰ nì—´ Figure ì¤€ë¹„
mpl.rc('font', size=11)
figure, axes = plt.subplots(nrows=5) # 5í–‰ 1ì—´
figure.set_size_inches(12, 18)

# ìŠ¤í…� 2 : ì„œë¸Œí”Œë¡¯ í• ë‹¹
# ê·¼ë¬´ì�¼, ê³µíœ´ì�¼, ìš”ì�¼, ê³„ì ˆ, ë‚ ì”¨ì—� ë”°ë¥¸ ì‹œê°„ëŒ€ë³„ í�‰ê·  ëŒ€ì—¬ ìˆ˜ëŸ‰ í�¬ì�¸íŠ¸í”Œë¡¯
sns.pointplot(x='hour', y='count', data=train, hue='workingday', ax=axes[0])
sns.pointplot(x='hour', y='count', data=train, hue='holiday', ax=axes[1])
sns.pointplot(x='hour', y='count', data=train, hue='weekday', ax=axes[2])
sns.pointplot(x='hour', y='count', data=train, hue='season', ax=axes[3])
sns.pointplot(x='hour', y='count', data=train, hue='weather', ax=axes[4]);


correlation_matrix = train[["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation with Count")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.scatterplot(x="temp", y="count", data=train, ax=axes[0, 0])
axes[0, 0].set_title("Temperature vs. Count")

sns.scatterplot(x="humidity", y="count", data=train, ax=axes[0, 1])
axes[0, 1].set_title("Humidity vs. Count")

sns.scatterplot(x="windspeed", y="count", data=train, ax=axes[1, 0])
axes[1, 0].set_title("Windspeed vs. Count")

sns.boxplot(x="season", y="count", data=train, ax=axes[1, 1])
axes[1, 1].set_title("Season vs. Count")

plt.tight_layout()
plt.show()


corrMatt = train[["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"]]
corrMatt = corrMatt.corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)


# ìŠ¤í…� 1 : mí–‰ nì—´ Figure ì¤€ë¹„
mpl.rc('font', size=15)
figure, axes = plt.subplots(nrows=2, ncols=2) # 2í–‰ 2ì—´
plt.tight_layout()
figure.set_size_inches(7, 6)

# ìŠ¤í…� 2 : ì„œë¸Œí”Œë¡¯ í• ë‹¹
# ì˜¨ë�„, ì²´ê°� ì˜¨ë�„, í’�ì†�, ìŠµë�„ ë³„ ëŒ€ì—¬ ìˆ˜ëŸ‰ ì‚°ì �ë�„ ê·¸ë�˜í”„
sns.regplot(x='temp', y='count', data=train, ax=axes[0, 0], 
            scatter_kws={'color':'red', 'alpha': 0.2}, line_kws={'color': 'blue'})
sns.regplot(x='atemp', y='count', data=train, ax=axes[0, 1], 
            scatter_kws={'color':'red', 'alpha': 0.2}, line_kws={'color': 'blue'})
sns.regplot(x='windspeed', y='count', data=train, ax=axes[1, 0], 
            scatter_kws={'color':'skyblue', 'alpha': 0.2}, line_kws={'color': 'blue'})
sns.regplot(x='humidity', y='count', data=train, ax=axes[1, 1], 
            scatter_kws={'color':'purple', 'alpha': 0.2}, line_kws={'color': 'blue'});


# trainWithoutOutliers
trainWithoutOutliers = train[np.abs(train["count"] - train["count"].mean()) <= (3*train["count"].std())]

print(train.shape)
print(trainWithoutOutliers.shape)


from scipy import stats

# countê°’ì�˜ ë�°ì�´í„° ë¶„í�¬ë�„ë¥¼ íŒŒì•…
figure, axes = plt.subplots(ncols=2, nrows=2)
figure.set_size_inches(14, 12)

sns.distplot(train["count"], ax=axes[0][0])
stats.probplot(train["count"], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(trainWithoutOutliers["count"]), ax=axes[1][0])
stats.probplot(np.log1p(trainWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])




