# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
import holidays

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Ensure data is loaded
df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df.head()


# Encode holidays, as these may effect sales
country_Holidays = {
    "Canada" : holidays.country_holidays("CA"),
    "Finland" : holidays.country_holidays("FI"),
    "Italy" : holidays.country_holidays("IT"),
    "Kenya" : holidays.country_holidays("KE"),
    "Norway" : holidays.country_holidays("NO"),
    "Singapore" : holidays.country_holidays("SG"),
}

# Predictions are ran on cube root of sum sold, this reverses that
undo = lambda x : x**3

# General data cleaning in one function
def dataCleaner(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    
    df2.dropna(inplace= True)

    # Encode dates cyclicly
    df2["year"] = df2["date"].map(lambda x : float(x[:4]))
    df2["month"] = df2["date"].map(lambda x : np.sin(2 * np.pi * float(x[5:7]) / 12))
    df2["day"] = df2["date"].map(lambda x : np.sin(2 * np.pi * float(x[8:]) / 31))
    
    df2["isHoliday"] = df2.apply(lambda x : x.date in country_Holidays[x.country], axis= 1)
    try:
        df2["num_sold"] = df2["num_sold"].map(lambda x : x**(1/3))
    except:
        pass

    df2 = df2.join(pd.get_dummies(df2["country"]))
    df2 = df2.join(pd.get_dummies(df2["store"]))
    df2 = df2.join(pd.get_dummies(df2["product"]))
    
    df2.drop(columns= ["id", "date", "country", "store", "product"], inplace= True)
    
    return df2


# Clean data
#data = dataCleaner(df)
#data.info()


# Make a copy, just in case, then split the data
#X = data.copy()
#y = X.pop("num_sold")

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state= 42)


# Test the Random Forest
#rf = RandomForestRegressor(random_state= 42)
#rf.fit(X_train, y_train)

#rf_pred = rf.predict(X_test)

#rf_pred_cubed = np.array([undo(x) for x in rf_pred])
#y_test_cubed = np.array([undo(x) for x in y_test])

#print("MAPE", MAPE(y_test_cubed, rf_pred_cubed))


#print("MAPE", MAPE(y_test, rf_pred))


#rf.score(X_test, y_test)


# Load train and test
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

# Clean data
train_cleaned = dataCleaner(train)
X_train = train_cleaned
y_train = X_train.pop("num_sold")
test_cleaned = dataCleaner(test.copy())

# Run predictions
rf = RandomForestRegressor(random_state= 42)
rf = rf.fit(X_train, y_train)

rf_pred = rf.predict(test_cleaned)
rf_pred_sub = np.array([undo(x) for x in rf_pred])

# Submit
output = pd.DataFrame({'id': test["id"], 
                       'num_sold': rf_pred_sub})
output.to_csv('submission.csv', index=False)

