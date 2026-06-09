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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.shape, test.shape, train_extra.shape


train = pd.concat([train, train_extra], ignore_index=True)


train.shape


train.sample(5)


train.info()


test.info()


# feature construction strategy adapted from vyacheslavbolotin notebook

dict_fen = {'Material':'NaN','Style':'NaN','Brand':'NaN','Size':'NaN','Waterproof':'NaN','Color':'NaN','Laptop Compartment':'NaN',}

def feh(df):
    
    df = df.fillna(dict_fen)

    map_size       = {'Small':    1.1,'Medium':  1.2,'Large':1.3,                                    'NaN':0}
    map_brand      = {'Jansport': 1.1,'Adidas':  1.2,'Nike': 1.3,'Puma':  1.4,'Under Armour':    1.5,'NaN':0}
    map_color      = {'Black':    1.1,'Green':   1.2,'Red':  1.3,'Blue':  1.4,'Gray':1.05,'Pink':1.5,'NaN':0}
    map_style      = {'Messenger':1.1,'Backpack':1.2,'Tote': 1.3,                                    'NaN':0}
    map_material   = {'Polyester':1.1,'Leather': 1.2,'Nylon':1.3,'Canvas':1.4,                       'NaN':0}
    map_waterproof = {'Yes':      1.1,'No':      1.0,                                                'NaN':0}
    map_laptop     = {'Yes':      1.1,'No':      1.0,                                                'NaN':0}
    
    df['Size_map']        = df['Size']              .map(map_size)
    df['Brand_map']       = df['Brand']             .map(map_brand)
    df['Color_map']       = df['Color']             .map(map_color)
    df['Style_map']       = df['Style']             .map(map_style)
    df['Material_map']    = df['Material']          .map(map_material)
    df['Waterproof_map']  = df['Waterproof']        .map(map_waterproof)
    df['Laptop_map']      = df['Laptop Compartment'].map(map_laptop)
    df['Compartments_map']= df['Compartments']      .apply(lambda x: x/1.1)
    
    df['_NaN_Material']   = df['Material']  .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Style']      = df['Style']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Brand']      = df['Brand']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Size']       = df['Size']      .apply(lambda x: 1 if x == 'NaN' else 0)                                    
    df['_NaN_Waterproof'] = df['Waterproof'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Color']      = df['Color']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Laptop']     = df['Laptop Compartment'].apply(lambda x: 1 if x == 'NaN' else 0)
    
    df['_7_NaNs'] = df['_NaN_Waterproof']+df['_NaN_Material']+df['_NaN_Laptop']+df['_NaN_Style']+df['_NaN_Brand']+df['_NaN_Size']+df['_NaN_Color']

    df = df.rename(columns={ 'Size_map':'x1', 'Brand_map':'x2', 'Color_map':'x3', 'Style_map':'x4', 'Material_map':'x5', 'Waterproof_map':'x6', 'Laptop_map':'x7', 'Compartments_map':'x8' } ) 

# feature construction inspired from @khsamaha notebook
    median_weight = df["Weight Capacity (kg)"].median()
    df["Weight Capacity (kg)"] = (
        df["Weight Capacity (kg)"].fillna(median_weight)
    )
    
    conditions = [
        (df["Weight Capacity (kg)"] <= 5),
        (df["Weight Capacity (kg)"]  > 5) & (df["Weight Capacity (kg)"] <= 15),
        (df["Weight Capacity (kg)"]  > 15) & (df["Weight Capacity (kg)"] <= 20),
        (df["Weight Capacity (kg)"]  > 20) & (df["Weight Capacity (kg)"] <= 25),
        (df["Weight Capacity (kg)"] > 25)
    ]
    choices = ['Light', 'Middle', 'Light_heavy', 'Middel_heavy','Heavy']
    df['Weight_Class'] = np.select(conditions, choices, default='')
    
    df["Weight Capacity (kg)"] = df["Weight Capacity (kg)"].astype("float64")
    df['Weight_Class'] = df['Weight_Class'].astype("category")

    return df

train=feh(train)
test=feh(test)


train.info()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(train.drop(columns='Price'),train['Price'],test_size=0.2)


from sklearn.base import TransformerMixin

# Custom transformer to convert object columns to category
class ObjectToCategory(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        # Convert all object columns to category
        self.categorical_columns_ = X.select_dtypes(include='object').columns.tolist()
        for col in self.categorical_columns_:
            X[col] = X[col].astype('category')
        return X



obj_to_cat = ObjectToCategory()
X_train_transformed = obj_to_cat.fit_transform(X_train)

# Extract the list of categorical columns
categorical_columns = obj_to_cat.categorical_columns_


from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from lightgbm import LGBMRegressor


# parameters for lgbm copied from @mirzamilanfarabi notebook
# now since we have converted all 'object' dtypes into 'category', we can leverage lightgbm's ability to encode categorical columns implicitly
# Create LGBM model with categorical columns passed as a parameter
model = LGBMRegressor(num_leaves=56,
                      learning_rate=0.022497950121387757,
                      n_estimators=1400,
                      min_child_samples=419,
                      subsample=0.9318779151947196,
                      colsample_bytree=0.5935633028324053,
                      reg_alpha=1.0644656664600252,
                      reg_lambda=0.3945627132333395,
                      min_split_gain=9.98148173286267e-07,
                      max_bin=1899,
                      min_data_in_leaf=403,
                      cat_features=categorical_columns)

# Perform cross-validation
rmse_scorer = make_scorer(mean_squared_error, squared=False)
scores = cross_val_score(model, X_train_transformed, y_train, cv=5, scoring=rmse_scorer)
print(f'Cross-Validation RMSE: {np.mean(scores):.4f}')

# Fit the model on the training set
model.fit(X_train_transformed, y_train)

# Test set predictions
X_test_transformed = obj_to_cat.transform(X_test)  # Apply the same transformation to test data
y_pred = model.predict(X_test_transformed)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test Set RMSE: {test_rmse:.4f}')


test_transformed = obj_to_cat.transform(test)  # Apply the same transformation to test data
# Make predictions on the test set
predictions = model.predict(test_transformed)

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],         # Ensure 'id' exists in the test set
    'Price': predictions      # Use predictions on the test set
})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)





