# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Two lines Required to Plot Plotly
import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.offline as py

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from google.cloud import bigquery

client = bigquery.Client()


# List the tables in geo_openstreetmap dataset which resides in bigquery-public-data project:
dataset = client.get_dataset('bigquery-public-data.geo_openstreetmap')
tables = list(client.list_tables(dataset))
print([table.table_id for table in tables])


#By Anna Epishova https://www.kaggle.com/annaepishova/starter-geo-openstreetmap-bigquery-dataset

sql = '''
SELECT nodes.*
FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
JOIN UNNEST(all_tags) AS tags
WHERE tags.key = 'amenity'
  AND tags.value IN ('hospital',
    'clinic',
    'doctors')
LIMIT 10
'''
# Set up the query
query_job = client.query(sql)

# Make an API request  to run the query and return a pandas DataFrame
df = query_job.to_dataframe()
df.head(5)


sql = '''
SELECT nodes.*
FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
JOIN UNNEST(all_tags) AS tags
WHERE tags.key = 'amenity'
  AND tags.value IN ('stores',
    'clothes',
    'shoes')
LIMIT 10
'''
# Set up the query
query_job = client.query(sql)

# Make an API request  to run the query and return a pandas DataFrame
df = query_job.to_dataframe()
df.head()


#Fifth row, fourth column 

df.iloc[1,7]


df['all_tags'].value_counts()


df.iloc[5,7]


df.iloc[6,7]


df.head(7)


speeds_query = """
               WITH milan AS (
               SELECT ST_MAKEPOLYGON(ST_MAKELINE(
               [ST_GEOGPOINT(8.2768427 47.3489955),ST_GEOGPOINT(117.7535253 39.8865073),
               ST_GEOGPOINT(117.7536421 39.8863103)
               ]
               )) AS boundingbox
               )
               """


def run_query(shoes_query):
    return pd.read_sql_query(shoes_query, df) #Original was db


shoes_query = '''
SELECT hist.*
FROM `bigquery-public-data.geo_openstreetmap.history_nodes` AS hist
INNER JOIN UNNEST(all_tags) AS tags
INNER JOIN milan on ST_INTERSECTS(milan.boundingbox, hist.geometry)
WHERE tags.key = 'nice'
  AND tags.value IN ('stores',
    'clothes',
    'shoes')
  AND hist.id NOT IN (
    SELECT nodes.id
    FROM `bigquery-public-data.geo_openstreetmap.planet_nodes` AS nodes
    INNER JOIN UNNEST(all_tags) AS tags
    INNER JOIN milan on ST_INTERSECTS(milan.boundingbox, nodes.geometry)
    WHERE tags.key = 'nice'
      AND tags.value IN ('stores',
        'closes',
        'shoes')
)
'''
run_query(shoes_query)

