# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train_df.isna().sum()


#testing using less data 
data={
    'id': [1,2,3,np.nan,5,np.nan,7],
        'Marks': [89,78,80,np.nan,92,98,np.nan]
         }
df=pd.DataFrame(data)



df.isna().sum()


df['id'].fillna(df['id'].mean(),inplace=True)


df['Marks'].fillna(method='ffill', inplace=True)

print(df['Marks'])

#df['Marks'].fillna(method='ffill', inplace=True)


df.isna().sum()
#testing completed with less data


 train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mean(),inplace=True)
# train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].median(),inplace=True)
#train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mode().iloc[0],inplace=True)
print(train_df['Weight Capacity (kg)'])



train_df.isnull().sum()


train_df["Size"].fillna(method='ffill',inplace=True)
print(train_df["Size"])


train_df.isnull().sum()


###Clean rest of the columns to remove NULL values.


#Label encoding involves converting each category into a unique integer
#initialize labelencoder()
label_encoder=LabelEncoder()
#fit and transform data
encoded_data=label_encoder.fit_transform(train_df['Style'])


#NOT REQUIRED. REMOVE THIS. Either use encoding or use dummies not both


# For Color and Brand columns, apply One-Hot Encoding

df_encoded = pd.get_dummies(train_df, columns=['Color','Brand','Style'], drop_first=False)
print(df_encoded.head())

#You can do this ONLY after cleaning the data
# Encoding is ALWAYS after cleaning. Please use get dummies for other columns too!!!



print(train_df['Color'].dtype)
print(train_df['Style'].dtype)
print(train_df['Brand'].dtype)





import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Load your dataset
#df =pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')

# -------------------------
# 1. Encoding 'Color' (Nominal data)
# Apply One-Hot Encoding for the 'Color' column (nominal)
#df_encoded = pd.get_dummies(train_df, columns=['Color'], drop_first=False)




# REMOVE THIS CELL

# -------------------------
# 2. Encoding 'Style' (Ordinal or Nominal)
# If 'Style' is ordinal, apply Ordinal Encoding:
style_order = ['Casual', 'Sporty', 'Formal']  # Modify the order as per your data
ordinal_encoder = OrdinalEncoder(categories=[style_order])

# Apply Ordinal Encoding to the 'Style' column
df_encoded['Style_encoded'] = ordinal_encoder.fit_transform(df_encoded[['Style']])

# If 'Style' is nominal (unordered), use One-Hot Encoding instead:
# df_encoded = pd.get_dummies(df_encoded, columns=['Style'], drop_first=False)

# Drop the original 'Style' column
df_encoded.drop('Style', axis=1, inplace=True)

# -------------------------
# 3. Encoding 'Brand' (Nominal data, use Label or One-Hot Encoding)
# If you prefer Label Encoding for 'Brand':
label_encoder = LabelEncoder()
df_encoded['Brand_encoded'] = label_encoder.fit_transform(df_encoded['Brand'])

# If you prefer One-Hot Encoding for 'Brand':
# df_encoded = pd.get_dummies(df_encoded, columns=['Brand'], drop_first=False)

# Drop the original 'Brand' column
df_encoded.drop('Brand', axis=1, inplace=True)

# -------------------------
# Check the transformed dataset
print(df_encoded.head())



#Once you complete get dummies. NOW your data is ready for train test and split.


