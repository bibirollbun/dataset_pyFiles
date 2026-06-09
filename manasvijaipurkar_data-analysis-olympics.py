import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("/kaggle/input/winter-olympics-2022/train.csv")
data.head(10)


data.describe()


data.shape


data.columns


data.info


data.isnull()


data.isnull().count()


print("Number of rows: ", len(data))


print(f"Years:  {data['Year'].min()} to {data['Year'].max()}")


print(f"Country {data['Country'].nunique()}")


data.tail()


print(f"Total medals in dataset is: {data['Medal'].sum()}")


print(f"Average medals per country per games: {data['Medal'].mean()}")


print(f"Most medals by a country in one games: {data['Medal'].max()}")


# Simple group by country and sum medals
country_medals = data.groupby('Country')['Medal'].sum().sort_values(ascending = False)
print(country_medals)


plt.figure(figsize = (10, 5))
medal_per_year = data.groupby('Year')['Medal'].sum()
plt.plot(medal_per_year.index, medal_per_year.values, marker = 'o')
plt.title('Total Medals Awarded Each Olympic Games')
plt.xlabel("Year")
plt.ylabel("Medals")
plt.grid(True)
plt.show


plt.figure(figsize = (10, 5))
medal_per_year = data.groupby('Year')['Medal'].sum()
plt.bar(medal_per_year.index, medal_per_year.values)
plt.title('Total Medals Awarded Each Olympic Games')
plt.xlabel("Year")
plt.ylabel("Medals")
plt.grid(True)
plt.show


plt.figure(figsize = (10, 5))
medal_per_year = data.groupby('Year')['Medal'].sum()
plt.hist(medal_per_year.values, bins=10, edgecolor='black')
plt.title('Distribution of Total Medals per Olympic Games')
plt.xlabel("Medals")
plt.ylabel("Number of Games")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 10))
country_medals.head(3).plot(kind='bar')
plt.title('Top 10 Countries by Total Medals')
plt.xlabel('Country')
plt.ylabel('Total Medals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

