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


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df.sample(5)
df = df.sort_values(by='day')


df.info()


df.describe()


df.corr()


import seaborn as sns
import matplotlib.pyplot as plt


df['rainfall'].value_counts().plot(kind = 'pie', autopct = '%.2f')


sns.kdeplot(df['pressure'], fill=True)
plt.xlabel('Pressure Values')
plt.ylabel('Density')
plt.title('Pressure Distribution Curve')
plt.show()



sns.kdeplot(df['humidity'], fill=True)
plt.xlabel('Humidity Values')
plt.ylabel('Density')
plt.title('Humidity Distribution Curve')
plt.show()



sns.kdeplot(df['sunshine'], fill=True)
plt.xlabel('Sunshine Values')
plt.ylabel('Density')
plt.title('Sunshine Distribution Curve')
plt.show()



sns.kdeplot(df['pressure'], fill=True)
plt.xlabel('Pressure Values')
plt.ylabel('Density')
plt.title('Pressure Distribution Curve')
plt.show()



sns.kdeplot(df['cloud'], fill=True)
plt.xlabel('Cloud Values')
plt.ylabel('Density')
plt.title('Cloud Distribution Curve')
plt.show()



sns.kdeplot(df['temparature'], fill=True)
plt.xlabel('Temperature Values')
plt.ylabel('Density')
plt.title('Temperature Distribution Curve')
plt.show()



sns.kdeplot(df['windspeed'], fill=True)
plt.xlabel('Windspeed Values')
plt.ylabel('Density')
plt.title('Windspeed Distribution Curve')
plt.show()



sns.kdeplot(df['winddirection'], fill=True)
plt.xlabel('Wind direction Values')
plt.ylabel('Density')
plt.title('Wind direction Distribution Curve')
plt.show()



sns.kdeplot(df['dewpoint'], fill=True)
plt.xlabel('Dewpoint Values')
plt.ylabel('Density')
plt.title('Dewpoint Distribution Curve')
plt.show()



sns.kdeplot(df['rainfall'], fill=True)
plt.xlabel('Rainfall Values')
plt.ylabel('Density')
plt.title('Rainfall Distribution Curve')
plt.show()



plt.figure(figsize=(8,5))
sns.scatterplot(x=df['sunshine'], y=df['rainfall'])
plt.xlabel("Sunshine")
plt.ylabel("Rainfall")
plt.title("Sunshine vs Rainfall Scatter Plot")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['temparature'], y=df['rainfall'])
plt.xlabel("Temperature")
plt.ylabel("Rainfall")
plt.title("Temperature vs Rainfall Scatter Plot")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['windspeed'], y=df['rainfall'])
plt.xlabel("Windspeed")
plt.ylabel("Rainfall")
plt.title("Windspeed vs Rainfall Scatter Plot")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['cloud'], y=df['rainfall'])
plt.xlabel("Cloud")
plt.ylabel("Rainfall")
plt.title("Cloud vs Rainfall Scatter Plot")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['winddirection'], y=df['rainfall'])
plt.xlabel("Wind direction")
plt.ylabel("Rainfall")
plt.title("Wind direction vs Rainfall Scatter Plot")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['pressure'], y=df['rainfall'])
plt.xlabel("Pressure")
plt.ylabel("Rainfall")
plt.title("Pressure vs Rainfall Scatter Plot")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['humidity'], y=df['rainfall'])
plt.xlabel("Humidity")
plt.ylabel("Rainfall")
plt.title("Humidity vs Rainfall Scatter Plot")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x=df['dewpoint'], y=df['rainfall'])
plt.xlabel("Dewpoint")
plt.ylabel("Rainfall")
plt.title("Dewpoint vs Rainfall Scatter Plot")
plt.show()




