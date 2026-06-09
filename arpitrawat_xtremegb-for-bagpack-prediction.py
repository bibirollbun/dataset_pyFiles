%reload_ext cudf.pandas

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv',index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv',index_col='id')
train_ex = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",index_col='id')


train = pd.concat([train, train_ex], axis=0, ignore_index=True)
train.shape,test.shape


train.info()


test.info()


cat_cols=train.select_dtypes(include='object').columns.tolist()


from cuml.preprocessing import TargetEncoder
TE = TargetEncoder(n_folds=25, smooth=20, split_method='random', stat='mean')

features = test.columns.tolist()

for col in features:
    TE.fit(train[col], train['Price'])
    train[col] = TE.transform(train[col])
    test[col] = TE.transform(test[col])


# copied from @vyacheslavbolotin notebook
def fen(df):
    dict_fen = {
            'Material'          :'NaN',
            'Style'             :'NaN',
            'Brand'             :'NaN',
            'Size'              :'NaN',
            'Waterproof'        :'NaN',
            'Color'             :'NaN',
            'Laptop Compartment':'NaN',
        }
    
    df.fillna(dict_fen)
    
    df['_NaN_Material']   = df['Material']  .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Style']      = df['Style']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Brand']      = df['Brand']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Size']       = df['Size']      .apply(lambda x: 1 if x == 'NaN' else 0)                                    
    df['_NaN_Waterproof'] = df['Waterproof'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Color']      = df['Color']     .apply(lambda x: 1 if x == 'NaN' else 0)
    df['_NaN_Laptop']     = df['Laptop Compartment'].apply(lambda x: 1 if x == 'NaN' else 0)
    df['weight/compartment']=df['Weight Capacity (kg)']/df['Compartments']
    df['_7_NaNs'] = df['_NaN_Waterproof']+df['_NaN_Material']+df['_NaN_Laptop']+df['_NaN_Style']+df['_NaN_Brand']+df['_NaN_Size']+df['_NaN_Color']

    for cat in (cat_cols):
       df[f'{cat}_wc']=df[cat]/1000 + df['Weight Capacity (kg)']
       df[f'{cat}_cmp']=df[cat]/100 +df['Compartments']
        
       

    return df
train=fen(train)
test=fen(test)
 


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error


X=train.drop(columns=['Price'])
y=train['Price']



pipeline=Pipeline(steps=[('model',XGBRegressor(objective='reg:squarederror',
     device="cuda",
     tree_method='gpu_hist',                                                               
     max_depth=6,  
     colsample_bytree=0.7, 
     subsample=0.8,  
     n_estimators=500,  
     learning_rate=0.05,  
     min_child_weight=10,))])

# Custom RMSE scorer
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Perform cross-validation
scores = cross_val_score(pipeline, X, y, cv=3, scoring=rmse_scorer,error_score='raise')

# Print results
print(f"Cross-Validation RMSE Scores: {scores}")
print(f"Mean RMSE: {np.mean(scores):.4f}")


X.shape,test.shape


pipeline.fit(X,y)
# Make predictions on the test set
predictions = pipeline.predict(test)

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test.index,         # Ensure 'id' exists in the test set
    'Price': predictions      # Use predictions on the test set
})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)


submission.shape




