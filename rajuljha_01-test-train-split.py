# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GroupShuffleSplit


PROJECT_DIR = '/kaggle/input'
OUTPUT_DIR = '/kaggle/working'
METADATA_DIR = os.path.join('/kaggle/input/rsna-lumbar-metadata', 'data', 'processed_metadata')


# Ensure deterministic behavior
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)


df = pd.read_csv(os.path.join(METADATA_DIR, 'processed_metadata.csv'))
condition_types = df['condition'].unique().tolist()


def group_split(df, group_col, test_size=0.15, random_state=42):
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    split = splitter.split(df, groups=df[group_col])
    train_idx, test_idx = next(split)
    return df.iloc[train_idx], df.iloc[test_idx]


for condition in condition_types:
    print(condition, ":")
    # Remove spaces to write into file names
    condition = condition.replace(" ", "")
    
    # Read in the dataset exclusive to the condition
    df_condition = pd.read_csv(os.path.join(METADATA_DIR, 'processed_metadata_' + condition + '.csv'))
    
    # Split into train, validation and test
    train_val_df, test_df = group_split(df_condition, 'study_id')
    train_df, val_df = group_split(train_val_df, 'study_id')
    print(f"Number of samples in training set: {train_df.shape[0]}")
    print(f"Number of samples in validation set: {val_df.shape[0]}")
    print(f"Number of samples in test set: {test_df.shape[0]}")
    
    # Write out the splitted subsets
    WRITE_DIR = os.path.join(OUTPUT_DIR, condition)
    os.makedirs(WRITE_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(WRITE_DIR, 'train.csv'), index=False)
    val_df.to_csv  (os.path.join(WRITE_DIR, 'val.csv'),   index=False)
    test_df.to_csv (os.path.join(WRITE_DIR, 'test.csv'),  index=False)


!zip -r RSNA_Lumbar_01_test_train_split.zip /kaggle/working


!ls




