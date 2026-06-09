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


df = pd.read_csv("/kaggle/input/data-meta/Datasets.csv")
df


df.info()


df.describe()


df.describe().T


df.head(5)


df.tail(5)


df_new=df.drop("OwnerOrganizationId",axis=1)
df_new


df_new.head(17)


df_new["Medal"]


df_medalists = df_new[df_new["Medal"].isin([1, 2, 3])]
print(df_medalists)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_new = pd.DataFrame(df_new)


print(df_new.columns)



# Let's separate those who received medals from those who did not
df_new["Medal_Status"] = df_new["Medal"].apply(lambda x: "Won" if x in [1, 2, 3] else "No Winner")

# visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_new, x="Id", y="Medal", hue="Medal_Status", style="Medal_Status", s=100)

plt.title("Medal Winners and Non-Winners ")
plt.xlabel("Name")
plt.ylabel("Point")
plt.legend(title="Situation")
plt.grid(True)
plt.tight_layout()
plt.show()



pip install plotly



import pandas as pd
import plotly.express as px


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Medal status
df_new["Medal_Status"] = df_new["Medal"].apply(lambda x: "Won" if x in [1, 2, 3] else "No winner")

#Visual
plt.figure(figsize=(10, 6))

# SCATTERPLOT (dot chart)
sns.scatterplot(
    data=df_new,
    x="Id",
    y="Medal",
    hue="Medal_Status",
    alpha=0.6,
    s=100
)
plt.xticks(rotation=45)
plt.title("Scatterplot: Scores by Medal Standings")
plt.grid(True)
plt.tight_layout()
plt.show()

# STRIPPLOT (Distribution)
plt.figure(figsize=(8, 5))
sns.stripplot(data=df_new, x="Medal_Status", y="Medal", jitter=True)
plt.title("Stripplot: Point Distribution According to Medal Status")
plt.tight_layout()
plt.show()

# BARPLOT (Average comparison)
plt.figure(figsize=(8, 5))
sns.barplot(data=df_new, x="Medal_Status", y="Medal", estimator="mean")
plt.title("Barplot: Average Scores (Medal Standings)")
plt.tight_layout()
plt.show()



df_new


import pandas as pd
import numpy as np

# Example DataFrame with some NaN values

df_new = pd.DataFrame(df_new)

print("Original DataFrame:")
print(df_new)

# Replace all NaN values in the entire DataFrame with 0
df_filled = df.fillna(0)

print("\nDataFrame after filling NaN with 0:")
print(df_filled)


df_filled


print(df_filled.columns)


df_filled = pd.DataFrame(df_filled)

#Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df_filled['Medal'], kde=True, bins=10) # kde=True yoğunluk tahmini çizgisi ekler
plt.title('Distribution of Medal Values (NaN are set to 0)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # np.nan for example data

# --- Step 1: Creating a Sample DataFrame 

data = {'MedalAwardDate': ['08/05/2021', '08/06/2021', '08/07/2021', '08/08/2021', '08/09/2021',
                           '08/10/2021', '08/11/2021', '08/12/2021', '08/13/2021', '08/14/2021'],
        'Medal': [10, 12, 15, 11, 14, 0, 16, 18, 17, 0], # NaN'lar 0 yapılmış hali
        'TotalKernels': [100, 105, 110, 108, 112, 115, 0, 120, 125, 0]} # NaN'lar 0 yapılmış hali
df_filled = pd.DataFrame(data)

print("Original DataFrame (before date conversion):")
print(df_filled.info())
print("\n")

# --- Step 2: 'Convert column 'MedalAwardDate' to datetime format ---
# The 'format' parameter tells Pandas what date format to expect.
# For the format "08/05/2021" use '%m/%d/%Y' (Month/Day/Year)
df_filled['MedalAwardDate'] = pd.to_datetime(df_filled['MedalAwardDate'], format='%m/%d/%Y')

print("DataFrame after date conversion:")
print(df_filled.info())
print("\n")

# --- Step 3: Making the Visualization ---

plt.figure(figsize=(12, 6))
sns.lineplot(x='MedalAwardDate', y='Medal', data=df_filled, label='Medal', marker='o')
sns.lineplot(x='MedalAwardDate', y='TotalKernels', data=df_filled, label='TotalKernels', marker='x')
plt.title('Changes in Medal and Total Values Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45) # Improves readability of date labels
plt.tight_layout() # Improves the layout of the chart
plt.show()


import pandas as pd
import numpy as np

# Example DataFrame (mimicking your potential issue)
data = {'TotalVotes ': [100, 150, 200, 120, 180], # Notice the space after 'TotalVotes'
        'AnotherColumn': [1, 2, 3, 4, 5]}
df_filled = pd.DataFrame(data)


print("Columns in your DataFrame:")
print(df_filled.columns)



df_filled.columns = df_filled.columns.str.strip()


# 'TotalVotes'  work 
category_counts = df_filled['TotalVotes'].value_counts() # Notice no space after TotalVotes

print("\nColumn names after stripping whitespace:")
print(df_filled.columns)
print("\nValue counts for 'TotalVotes':")
print(category_counts)

# pie chart code:
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7, 7))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('TotalVotes Distribution')
plt.axis('equal') # Makes the cake round
plt.show()




