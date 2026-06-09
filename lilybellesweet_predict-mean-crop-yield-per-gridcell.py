# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def load_data(crop: str, mode: str="train"):
    # note that years represent an offset from model spinup;
    # soil co2 dataset has real year;
    # 0-30 are days before sowing, 31-238 are days after sowing
    #tasmax = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/tasmax_{crop}_{mode}.parquet")
    #tasmin = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/tasmin_{crop}_{mode}.parquet")
    #pr = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/pr_{crop}_{mode}.parquet")
    #rsds = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/rsds_{crop}_{mode}.parquet")
    soil_co2 = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/soil_co2_{crop}_{mode}.parquet")
    target = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/{mode}_solutions_{crop}.parquet")
    return {
        #'tasmax': tasmax,
        #'tasmin': tasmin,
        #'pr': pr,
        #'rsds': rsds,
        'soil_co2': soil_co2,
        'target': target,
    }


maize_train = load_data("maize", "train")
maize_train.keys()


wheat_train = load_data("wheat", "train")
wheat_train.keys()


# calculate mean maize yield at each grid cell
mean_maize_per_gridcell = maize_train['soil_co2'].join(maize_train['target']).groupby(
    ['lon','lat'])['yield'].mean()


# calculate mean wheat yield at each grid cell
mean_wheat_per_gridcell = wheat_train['soil_co2'].join(wheat_train['target']).groupby(
    ['lon','lat'])['yield'].mean()


soil_co2_maize_test = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/soil_co2_maize_test.parquet")


# join with IDs of datapoints to be predicted
maize_predictions = pd.merge(soil_co2_maize_test.reset_index(), 
                             mean_maize_per_gridcell.reset_index(), 
                             how='left', on=['lat','lon']).set_index('ID')[['yield']]


soil_co2_wheat_test = pd.read_parquet(f"/kaggle/input/ag-ml-leipzig-2025-future-crop/soil_co2_wheat_test.parquet")


# join with IDs of datapoints to be predicted
wheat_predictions = pd.merge(soil_co2_wheat_test.reset_index(), 
                             mean_wheat_per_gridcell.reset_index(), 
                             how='left', on=['lat','lon']).set_index('ID')[['yield']]


pd.concat([maize_predictions, wheat_predictions]).to_csv('submission.csv', index_label='ID')




