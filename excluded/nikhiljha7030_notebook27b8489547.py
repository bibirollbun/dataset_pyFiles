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


import numpy as np
import pandas as pd


num_entries = 50000


brands = ['Nike', 'Adidas', 'Jansport', 'Puma', 'Under Armour']
materials = ['Leather', 'Polyester', 'Nylon', 'Canvas']
sizes = ['Small', 'Medium', 'Large']
compartments = np.arange(1, 11)  # 1 to 10 compartments
laptop_compartment = ['Yes', 'No']
waterproof = ['Yes', 'No']
styles = ['Backpack', 'Messenger', 'Tote']
colors = ['Black', 'Blue', 'Red', 'Green', 'Gray', 'Pink']
weight_capacity = np.arange(5, 31)  # 5 kg to 30 kg


base_data = {
    'Brand': np.random.choice(brands, num_entries),
    'Material': np.random.choice(materials, num_entries),
    'Size': np.random.choice(sizes, num_entries),
    'Compartments': np.random.choice(compartments, num_entries),
    'Laptop Compartment': np.random.choice(laptop_compartment, num_entries),
    'Waterproof': np.random.choice(waterproof, num_entries),
    'Style': np.random.choice(styles, num_entries),
    'Color': np.random.choice(colors, num_entries),
    'Weight Capacity (kg)': np.random.choice(weight_capacity, num_entries),
    'Price': np.round(np.random.uniform(15, 150, num_entries), 2)
}

df_large = pd.DataFrame(base_data)


num_duplicates = int(0.05 * num_entries)
duplicates = df_large.sample(num_duplicates, replace=True)
df_large = pd.concat([df_large, duplicates], ignore_index=True)


missingness_fraction = 0.05
for col in df_large.columns:
    df_large.loc[df_large.sample(frac=missingness_fraction).index, col] = np.nan


noise_factor = 0.1  # 10% noise

df_large['Compartments'] = df_large['Compartments'] + np.random.randint(-1, 2, df_large.shape[0])
df_large['Weight Capacity (kg)'] = df_large['Weight Capacity (kg)'] + np.random.uniform(-2, 2, df_large.shape[0])
df_large['Price'] = df_large['Price'] * (1 + np.random.uniform(-noise_factor, noise_factor, df_large.shape[0]))



df_large['Compartments'] = df_large['Compartments'].clip(lower=1, upper=10)
df_large['Weight Capacity (kg)'] = df_large['Weight Capacity (kg)'].clip(lower=5, upper=30)
df_large['Price'] = df_large['Price'].clip(lower=15, upper=150)


# Displaying the modified dataset 
df_large.head() 


df_large.to_csv("noisy_student_bag_dataset.csv", index=False)




