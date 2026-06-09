import pandas 

pandas.__version__


!pip install -q hvplot


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas


df = pd.read_csv("../input/stanford-open-policing-project/police_project.csv")
df.head()


df.info()


df.describe()


df.shape


df.isnull().sum()


for column in df.columns:
    missing = df[column].isna().sum() / df.shape[0]
    print(f"{column:{20}}: ==============> {missing * 100:.2f}%")


df.dropna(axis=1, how='all').shape


df.drop('county_name', axis=1, inplace=True)


df.isnull().sum()


print(df.driver_gender.value_counts())

print(df.driver_gender.value_counts(normalize=True))


print(df['violation'].value_counts())

print(df['violation'].value_counts(normalize=True))


speed_violation = df[df.violation == 'Speeding']

fig = plt.figure(figsize=(12, 6))

plt.subplot(121)
sns.countplot(x='driver_gender', data=df)
plt.title('Men & Women Distribution')

plt.subplot(122)
sns.countplot(x='driver_gender', data=speed_violation)
plt.title('Men & Women Distribution (Violation = Speeding)')


print(df[df.violation == 'Speeding'].driver_gender.value_counts())

print(df[df.violation == 'Speeding'].driver_gender.value_counts(normalize=True))


df[df.violation == 'Speeding'].driver_gender.value_counts().hvplot.bar(height=350, width=350)


df['violation'].value_counts().hvplot.barh()


men = df.loc[df['driver_gender']=='F', 'violation'].value_counts().hvplot.barh(alpha=0.3) 
women = df.loc[df['driver_gender']=='M', 'violation'].value_counts().hvplot.barh(alpha=0.3)

women * men


plt.figure(figsize=(10, 12))

plt.subplot(2, 2, 1)
df[df.violation == 'Speeding'].driver_gender.value_counts().plot(kind="bar")
plt.title("Speeding violation for Men and Women", fontsize=15)

plt.subplot(2, 2, 2)
sns.countplot(x='violation', data=df, hue='driver_gender')
plt.title("Violation for Men and Women", fontsize=15)
plt.xticks(rotation=90);


df.loc[df.violation == "Speeding", "driver_gender"].value_counts(normalize=True)


print(df[df.driver_gender == "M"].violation.value_counts())

print(df[df.driver_gender == "M"].violation.value_counts(normalize=True))


print(df[df.driver_gender == "F"].violation.value_counts())

print(df[df.driver_gender == "F"].violation.value_counts(normalize=True))


sns.countplot(y='violation', data=df, hue='driver_gender', orient='h')
plt.title("Violation vs Driver Gender Distribution")
# plt.xticks(rotation=90)


plt.figure(figsize=(12, 18))

plt.subplot(4, 2, 1)
df[df.driver_gender == "F"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Women")

plt.subplot(4, 2, 2)
df[df.driver_gender == "M"].violation.value_counts(normalize=True).plot(kind="bar")
plt.title("Violation of Men")

plt.subplot(4, 2, 3)
sns.countplot(x='violation', data=df, hue='driver_gender')
plt.xticks(rotation=90)

plt.tight_layout();


print(df.search_conducted.value_counts())

print(df.search_conducted.value_counts(normalize=True))


print(df.loc[df.search_conducted, 'driver_gender'].value_counts())

print(df.loc[df.search_conducted, 'driver_gender'].value_counts(normalize=True))


searched = df.loc[df['search_conducted']==True, 'driver_gender'].value_counts().hvplot.barh(alpha=0.3) 
not_searched = df.loc[df['search_conducted']==False, 'driver_gender'].value_counts().hvplot.barh(alpha=0.3)

searched * not_searched


plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
sns.countplot(x='search_conducted', hue='driver_gender', data=df)
plt.title("Searched Conducted (3196-3%| True)/(88545-97%| False)")

plt.subplot(2, 2, 2)
searched = df[df['search_conducted']==True]
sns.countplot(x='driver_gender', data=searched)
plt.title("Search Conducted by Gender (2725-85%| Men)/(471-15%| Woman)")


df.groupby(['violation', 'driver_gender']).search_conducted.mean()


plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
df.search_conducted.value_counts().plot(kind="bar")
plt.title("Searched Cases")

plt.subplot(2, 2, 2)
df.loc[df.search_conducted, 'driver_gender'].value_counts().plot(kind="bar")
plt.title("Searched Men and Women")

plt.subplot(2, 2, 3)
df.groupby(['violation', 'driver_gender']).search_conducted.mean().plot(kind="bar")

plt.subplot(2, 2, 4)
sns.countplot(x='search_conducted', data=df, hue='driver_gender')


df.search_type.isnull().sum()


df.search_conducted.value_counts()


df[df.search_conducted == False].search_type.value_counts(dropna=False)


df.search_type.value_counts()


df.search_type.value_counts().hvplot.barh()


df.search_type.value_counts()


from collections import Counter

st = df.search_type.dropna()

search_count = Counter()
for search in st.str.split(','):
    search_count.update(search)


dict(search_count)


search_count_dict = dict(search_count)
pd.DataFrame(search_count_dict.items(), columns=['Search Type', 'Count']).hvplot.barh(x='Search Type', y='Count')


df.search_type.str.contains('Protective Frisk').sum()


df.search_type.str.contains('Protective Frisk').mean()


df.head()


print(df.stop_date.dtype)
print(df.stop_time.dtype)


df.stop_date


df['stop_date'] = pd.to_datetime(df.stop_date, format="%Y-%M-%d")
df["year"] = df.stop_date.dt.year


df.dtypes


df.year.value_counts()


df.year.value_counts().hvplot.barh()


df.columns


print(df.drugs_related_stop.value_counts())

print(df.drugs_related_stop.value_counts(normalize=True))


df["stop_time"] = pd.to_datetime(df.stop_time, format="%H:%M").dt.hour
df.head()


df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].value_counts()


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
(
    df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].
    value_counts().sort_index().plot(kind="bar")
)
plt.xlabel("Day Hour")
plt.ylabel("Count")

plt.subplot(2, 2, 2)
(
    df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].
    value_counts().sort_index().plot()
)
plt.xlabel("Day Hour")
plt.ylabel("Count")


(
    df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].
    value_counts().sort_index().plot(kind="bar")
)


(
    df.loc[df.sort_values(by="stop_time").drugs_related_stop, 'stop_time'].
    value_counts().sort_index().plot()
)


df.stop_time.sort_index().value_counts()


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
df.stop_time.sort_index().value_counts().sort_index().plot()
plt.xlabel("Day Hour")
plt.ylabel("Count")

plt.subplot(2, 2, 2)
df.stop_time.sort_index().value_counts().sort_index().plot(kind="bar")
plt.xlabel("Day Hour")
plt.ylabel("Count")


df.stop_time.sort_index().value_counts().sort_index().hvplot(height=300, width=450)


df.stop_time.sort_index().value_counts().sort_index().hvplot(kind="bar")


print(f"stop_duration Missing Values: {df.stop_duration.isnull().sum()}")
print(f"stop_duration Unique Values: {df.stop_duration.unique()}")


df.stop_duration.unique()


df.stop_duration.value_counts(dropna=False)


# ri.stop_duration.replace(['1', '2'], value=np.nan, inplace=True)
df.loc[(df.stop_duration == '1')| (df.stop_duration == '2'), 'stop_duration'] = np.nan


df.stop_duration.value_counts(dropna=False)


print(f"stop_duration Unique Values: {df.stop_duration.unique()}")

print(f"violation_raw Number of Unique Values: {df.violation_raw.nunique()}")
print(f"violation_raw Unique Values: {df.violation_raw.unique()}")


df.violation_raw.value_counts()


df.groupby('stop_duration').violation_raw.value_counts()


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
df['violation_raw'].value_counts().plot.barh()
plt.xlabel("Count")
plt.ylabel("Violation Raw")

plt.subplot(2, 2, 2)
df['stop_duration'].value_counts().plot.barh()
plt.xlabel("Count")
plt.ylabel("Stop Duration")

plt.tight_layout()


df['violation_raw'].value_counts().plot.barh()


df['stop_duration'].value_counts().plot.barh()


sns.catplot(x="stop_duration", data=df, hue="violation_raw", kind="count")


mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}
df['stop_minutes'] = df.stop_duration.map(mapping)


df.stop_minutes.value_counts()


df.groupby('violation_raw').stop_minutes.mean()


df.groupby('violation_raw').stop_minutes.agg(['mean', 'count'])


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
df.groupby('violation_raw').stop_minutes.mean().plot(rot=90)
plt.xlabel("Violation Raw")
plt.ylabel("Mean Stopping Time")

plt.subplot(2, 2, 2)
df.groupby('violation_raw').stop_minutes.mean().plot(kind="bar", rot=90)
plt.xlabel("Violation Raw")
plt.ylabel("Mean Stopping Time")

plt.tight_layout()


df.groupby('violation_raw').stop_minutes.mean().hvplot(rot=45, height=500)


df.groupby('violation_raw').stop_minutes.mean().hvplot(kind="bar", rot=45, height=500)


df.groupby("violation").driver_age.describe()


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.histplot(x='driver_age', data=df)

plt.subplot(2, 2, 2)
sns.kdeplot(x='driver_age', hue='violation', data=df)




sns.histplot(x='driver_age', data=df)


sns.kdeplot(x='driver_age', hue='violation', data=df)


df.hvplot.kde(y='driver_age', height=300, width=450)


df.hvplot.hist(y='driver_age', by='violation', height=300, width=500)


df.hvplot.hist(y='driver_age', by='violation', subplots=True, height=300, width=300).cols(3)

