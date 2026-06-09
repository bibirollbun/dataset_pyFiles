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


from typing import Literal


def load_data(data_split: Literal['train', "test"]) -> pd.DataFrame:
    train_name_file: str = "/kaggle/input/playground-series-s5e1/train.csv"
    test_name_file: str ="/kaggle/input/playground-series-s5e1/test.csv"
    if data_split == "train":
        return pd.read_csv(train_name_file)
    elif data_split == "test":
        return pd.read_csv(test_name_file)
    else:
        raise Exception(f"Incorrect data split: {data_split}")
    


df_train = load_data("train")
df_train.head()


df_train.info()


df_train.isnull().mean()


def optimize_types(my_df) -> pd.DataFrame:
    data = my_df.copy()

    data = data.drop("id", axis=1)   

    data['date'] = pd.to_datetime(data['date'])    

    categorical_columns = ['country', 'store', 'product']
    
    for col in categorical_columns:
        data[col] = data[col].str.lower().astype('category')

    data['num_sold'] = data['num_sold'].astype('float32')
    
    return data


def show_memory_optimization(original_data: pd.DataFrame, optimized_data: pd.DataFrame):
    original_memory = original_data.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / (1024 ** 2)  # MB
    memory_reduction = original_memory - optimized_memory
    reduction_percentage = (memory_reduction / original_memory) * 100
    
    print(f"Original Memory Used: {original_memory:.2f} MB")
    print(f"Optimized Memory Used: {optimized_memory:.2f} MB")
    print(f"Memory reduction: {memory_reduction:.2f} MB ({reduction_percentage:.2f}%)")
    
    return {
        "original_memory_mb": original_memory,
        "optimized_memory_mb": optimized_memory,
        "memory_reduction_mb": memory_reduction,
        "reduction_percentage": reduction_percentage,
    }


df_optimized = optimize_types(df_train)
show_memory_optimization(df_train, df_optimized)

