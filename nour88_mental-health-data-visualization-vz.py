# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


# Reading the data
train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv', index_col='id')


# Sneak peak
train.head()


missing_summary = train.isnull().sum().sort_values(ascending = False)
print(missing_summary)



# Count the occurrences of each value in the "Depression" column
depression_counts = train['Depression'].value_counts()

# Define labels and corresponding values
labels0 = ['No', 'Yes']
sizes0 = depression_counts.values




# Suicidal thoughts count
Suicidal_thoughts_counts = train['Have you ever had suicidal thoughts ?'].value_counts()

# Define labels and corresponding values
labels1 = ['No', 'Yes']
sizes1 = Suicidal_thoughts_counts.values





train['Family History of Mental Illness'].value_counts()


# mental illness family history count
family_history_counts = train['Family History of Mental Illness'].value_counts()

# Define labels and corresponding values
labels2 = ['No', 'Yes']
sizes2 = family_history_counts.values




fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(10, 10), facecolor = 'black')

# First pie chart
axs[0].pie(sizes0, labels= labels0, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'],textprops={'color': 'white'} )
axs[0].set_title('% of People with\n Depression ',color='white')

# Second pie chart
axs[1].pie(sizes1, labels= labels1, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'],textprops={'color': 'white'} )
axs[1].set_title('% of People with\n Suicidal Thoughts',color='white')

# Third pie chart
axs[2].pie(sizes2, labels= labels2, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'],textprops={'color': 'white'} )
axs[2].set_title('% of People with\n Family history of mental illness',color='white')

plt.show()




