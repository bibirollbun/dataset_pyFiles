import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pylab
import matplotlib.pyplot as plt
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Iterate through files in the specified directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import numpy as np
import pandas as pd
import os

input_dir = '/kaggle/input'

if not os.path.exists(input_dir):
    print(f"The directory {input_dir} does not exist.")
else:
    input_files = []
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            input_files.append(file_path)
            print(file_path)

    print(f"\nTotal number of files found: {len(input_files)}")



import pandas as pd
df=pd.read_csv('/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv')
df.head()


import pandas as pd
df=pd.read_csv('/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv')
df.isnull().sum()


import pandas as pd
df=pd.read_csv('/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv')
df.columns


import pandas as pd
df=pd.read_csv('/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv')
df.shape




