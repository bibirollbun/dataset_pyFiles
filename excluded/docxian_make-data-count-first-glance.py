# packages

# standard
import numpy as np
import pandas as pd
import time

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# XML
import xml.etree.ElementTree as ET


# configs
pd.set_option('display.max_rows', 100)


ls -l '../input/make-data-count-finding-data-references/'


ls -l '../input/make-data-count-finding-data-references/train/PDF'


ls -l '../input/make-data-count-finding-data-references/train/XML'


# load training labels file
df_train = pd.read_csv('../input/make-data-count-finding-data-references/train_labels.csv')


# preview
df_train.head()


# type distribution
df_train.type.value_counts().plot(kind='bar', color='darkblue')
plt.title('type')
plt.grid()
plt.show()


# look at rows with missing dataset_id
df_train_miss = df_train[df_train.dataset_id=='Missing']
df_train_miss.shape


# check missings
pd.crosstab(df_train_miss.dataset_id, df_train_miss.type)


# remove rows with missing data
df_train = df_train[df_train.dataset_id != 'Missing']
print(df_train.type.value_counts())
df_train.type.value_counts().plot(kind='bar', color='darkblue')
plt.title('type')
plt.grid()
plt.show()


# check article ids
article_freqs = df_train.article_id.value_counts()
article_freqs[0:100]


# check dataset ids
df_train.dataset_id.value_counts()


tree = ET.parse('../input/make-data-count-finding-data-references/train/XML/10.1002_2017jc013030.xml')


root = tree.getroot()
root.tag


for child in root[0]:
    print(child.tag, child.attrib)


for child in root[0][0]:
    print(child.tag, child.attrib)


for child in root[0][0][0]:
    print(child.tag, child.attrib)


for child in root[0][0][0][0]:
    print(child.tag, child.attrib)


for child in root[0][0][0][0][0]:
    print(child.tag, child.attrib, child.text)


# access to title
root[0][0][0][0][0][0].text

