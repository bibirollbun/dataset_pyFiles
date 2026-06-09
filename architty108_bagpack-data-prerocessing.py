# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv")


df_extra = pd.read_csv(r"/kaggle/input/playground-series-s5e2/training_extra.csv")


df.info()


df.shape


df.isnull().sum()


df = df.dropna()


df.isnull().sum().sum()


df.sample(3)


df.nunique()


df.Style.unique()


df.shape


df.head(2)


brand_encoder = OneHotEncoder(sparse_output=False, drop=None)
new_brand = brand_encoder.fit_transform(df[['Brand']])
new_brand_feature_names = brand_encoder.get_feature_names_out()
temp_df = pd.DataFrame(new_brand, columns=new_brand_feature_names)
updated_df = pd.concat([temp_df, df], axis=1)


updated_df.dropna(inplace=True)


updated_df.shape


updated_df.drop(columns=['Brand','id'], inplace=True)


material_encoder = LabelEncoder()
updated_df.Material = Material = material_encoder.fit_transform(updated_df[['Material']])


updated_df.Size.unique()


size_encoder = OrdinalEncoder(categories=[['Small','Medium', 'Large']])
updated_df.Size = size_encoder.fit_transform(updated_df[['Size']])


laptopc_encoder = OrdinalEncoder(categories=[['No','Yes']])
updated_df['Laptop Compartment'] = laptopc_encoder.fit_transform(updated_df[['Laptop Compartment']])


waterproof_encoder = OrdinalEncoder(categories=[['No','Yes']])
updated_df['Waterproof'] = waterproof_encoder.fit_transform(updated_df[['Waterproof']])


style_encoder = OneHotEncoder(sparse_output=False, drop=None)
new_style = style_encoder.fit_transform(df[['Style']])
new_style_feature_names = style_encoder.get_feature_names_out()
temp_df = pd.DataFrame(new_style, columns=new_style_feature_names)
updated_df = pd.concat([temp_df, updated_df], axis=1)


updated_df.dropna(inplace=True)


updated_df


updated_df.drop(columns=['Color','Style'], inplace=True)


updated_df


import pickle


# exporting files

updated_df.to_csv("preprocessed_data.csv")

pickle.dump(brand_encoder, open("brand_encoder.pkl", 'wb'))
pickle.dump(material_encoder, open("material_encoder.pkl", 'wb'))
pickle.dump(size_encoder, open("size_encoder.pkl", 'wb'))
pickle.dump(laptopc_encoder, open("laptopc_encoder.pkl", 'wb'))
pickle.dump(waterproof_encoder, open("waterproof_encoder.pkl", 'wb'))
pickle.dump(style_encoder, open("style_encoder.pkl", 'wb'))




