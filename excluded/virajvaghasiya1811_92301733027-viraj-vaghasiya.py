# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/stack-overflow-data-analysis-ml-lhc-ict-2025'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install gdown



!gdown --id 1eHChVwaAwG66p9eDhISOROPIvKXnMg9W


# importing the zipfile module
from zipfile import ZipFile

# loading the temp.zip and creating a zip object
with ZipFile("/kaggle/working/Arjun_Assignment_data-20220427T165022Z-002.zip", 'r') as zObject:

    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path="/kaggle/working/DATA")


from zipfile import ZipFile

# loading the temp.zip and creating a zip object
with ZipFile("/kaggle/working/DATA/Arjun_Assignment_data/dataset-20210607T020316Z-001.zip", 'r') as zObject:
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path="/kaggle/working/DATA")


from zipfile import ZipFile

# loading the temp.zip and creating a zip object
with ZipFile("/kaggle/working/DATA/Arjun_Assignment_data/dataset-20210607T020316Z-002.zip", 'r') as zObject:
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path="/kaggle/working/DATA")


from zipfile import ZipFile

# loading the temp.zip and creating a zip object
with ZipFile("/kaggle/working/DATA/Arjun_Assignment_data/dataset-20210607T020316Z-003.zip", 'r') as zObject:
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path="/kaggle/working/DATA")


import pandas as pd
import numpy as np
import re
from collections import Counter

base_path = "/kaggle/working/DATA/dataset/"

posts = pd.read_csv(base_path + "posts_long.csv")
postlinks = pd.read_csv(base_path + "postLinks.csv")

posts.head(),postlinks.head()



posts['tag_count'] = posts['tags'].str.count(r'<')
posts[['id','tags','tag_count']].head()
posts['tag_count'].value_counts()


unique_tags = set()

for t in posts['tags'].dropna():
    tags = t.replace("<", " ").replace(">", " ").split()
    for tag in tags:
        unique_tags.add(tag)

len(unique_tags)



from collections import Counter

tag_counter = Counter()

for t in posts['tags'].dropna():
    # remove first and last <>
    clean = t.strip("<>")
    # split by >< to get individual tags
    tags = clean.split("><")     
    tag_counter.update(tags)

top_25 = tag_counter.most_common(25)
top_25



import matplotlib.pyplot as plt

top_500 = tag_counter.most_common(500)
freqs = [count for tag, count in top_500]

plt.figure(figsize=(10,5))
plt.hist(freqs, bins=30)
plt.title("Distribution of Top 500 Tags")
plt.xlabel("Frequency")
plt.ylabel("Count")
plt.show()



posts['creation_date'] = pd.to_datetime(posts['creation_date'])
posts['month'] = posts['creation_date'].dt.to_period("M")

dup = postlinks[postlinks['link_type_id'] == 3]
monthly_total = posts.groupby("month").size()
monthly_dup = posts[posts['id'].isin(dup['post_id'])].groupby("month").size()
ratio = (monthly_dup / monthly_total).fillna(0)

ratio["2008":"2018"].plot(figsize=(12,5))
plt.title("Duplicate Question Ratio per Month (2008–2018)")
plt.ylabel("Ratio")
plt.show()



dup = postlinks[postlinks['link_type_id'] == 3]
dup_ids = set(dup['post_id'])
tag_stats = {}
for _, row in posts.iterrows():
    if type(row['tags']) != str:
        continue
        
    clean = row['tags'].strip("<>")
    tags = clean.split("><")

    post_id = row['id']

    for tag in tags:
        if tag not in tag_stats:
            tag_stats[tag] = [0, 0]

        tag_stats[tag][1] += 1   

        if post_id in dup_ids:
            tag_stats[tag][0] += 1   

tag_percentage = {t: (d/tot)*100 for t, (d, tot) in tag_stats.items()}
top20 = dict(sorted(tag_percentage.items(), key=lambda x: x[1], reverse=True)[:20])
plt.figure(figsize=(14,5))
plt.bar(top20.keys(), top20.values())
plt.xticks(rotation=75)
plt.ylabel("Percentage")
plt.title("Top 20 Tags by Duplicate Question Rate")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

posts['creation_date'] = pd.to_datetime(posts['creation_date'])
postlinks['creation_date'] = pd.to_datetime(postlinks['creation_date'])

dup = postlinks[postlinks['link_type_id'] == 3]

merged = dup.merge(posts[['id', 'creation_date']], 
                   left_on='post_id', 
                   right_on='id', 
                   how='inner',
                   suffixes=('_link', '_post'))

merged['close_time_days'] = (merged['creation_date_link'] - merged['creation_date_post']).dt.days

merged['close_time_days'].dropna().plot(kind='hist', bins=30, figsize=(10,5))
plt.title("Estimated Time to Close Duplicate Questions (Using Link Creation Timestamp)")
plt.xlabel("Days")
plt.ylabel("Count")
plt.show()



import matplotlib.pyplot as plt

dup_ids = set(dup['post_id'])

dup_users = posts[posts['id'].isin(dup_ids)]['owner_user_id']

user_post_counts = posts[posts['owner_user_id'].isin(dup_users)]['owner_user_id'].value_counts()

plt.figure(figsize=(10,5))
plt.hist(user_post_counts, bins=20)
plt.title("User Activity Distribution for Duplicate-Question Authors")
plt.xlabel("Number of Posts by User")
plt.ylabel("User Count")
plt.show()



freqs_sorted = sorted(tag_counter.values(), reverse=True)
total_questions = len(posts)

coverage_500 = sum(freqs_sorted[:500]) / total_questions * 100
coverage_5000 = sum(freqs_sorted[:5000]) / total_questions * 100

coverage_500, coverage_5000



import pandas as pd

# Create a simple submission file
data = {
    'id': [1],
    'result': ['completed']
}

submission = pd.DataFrame(data)
submission.to_csv('/kaggle/working/submission.csv', index=False)

submission



import os
os.listdir('/kaggle/working/')





