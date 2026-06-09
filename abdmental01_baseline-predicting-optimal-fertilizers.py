%%time

import pandas as pd 
import numpy as np


%%time

SEED = 42

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

train = pd.concat([train,original], axis=0, ignore_index=True)

%time

def n_fe(df):
    
    df['Total_Nutrients'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1)
    df['K_P_Ratio'] = df['Potassium'] / (df['Phosphorous'] + 1)
    df['Humidity_Moisture_Gap'] = df['Humidity'] - df['Moisture']
    
    def temp_zone(temp):
        if temp < 20:
            return 'Low'
        elif temp < 30:
            return 'Medium'
        else:
            return 'High'
    
    df['Temp_Zone'] = df['Temparature'].apply(temp_zone)    
    df['Soil_Crop'] = df['Soil Type'] + "_" + df['Crop Type']
    
    return df

train = n_fe(train)
test = n_fe(test)

cat_c = ["Soil Type","Crop Type",'Temp_Zone','Soil_Crop']

def update(df):

    for col in cat_c:
        df[col] = df[col].astype('category')
    return df

train = update(train)
test = update(test)

train.head()


%%time

print(f'Train Data Shape: {train.shape}')
print(f'Test Data Shape: {test.shape}')

print(f'\nTrain Data Duplicated Values: {train.duplicated().sum()}')
print(f'Test Data Duplicated Values: {test.duplicated().sum()}')


%%time

map_values = {'28-28': 0, '17-17-17': 1, '10-26-26': 2, 'DAP': 3, '20-20': 4, '14-35-14': 5, 'Urea': 6}
train['Fertilizer Name'] = train['Fertilizer Name'].map(map_values).values

train['Fertilizer Name'].value_counts()


%%time

def mapk(y_true, y_pred, k=3):
    import numpy as np

    if len(np.array(y_pred).shape) == 1:
        raise ValueError("y_pred must be a 2D probability array (n_samples, n_classes). Got 1D vector instead.")

    top_k_preds = np.argsort(y_pred, axis=1)[:, ::-1][:, :k]

    def apk(actual, predicted, k):
        if actual in predicted[:k]:
            return 1.0 / (predicted[:k].tolist().index(actual) + 1)
        return 0.0

    return np.mean([apk(a, p, k) for a, p in zip(y_true, top_k_preds)])


%%time

cat_c = ["Soil Type","Crop Type",'Temp_Zone','Soil_Crop']


encode_c = {'cat_c': cat_c}

base = AbdBase(train_data=train, test_data=test, target_column='Fertilizer Name',gpu=True,test_prob = True,
                 problem_type="classification", metric="custom", seed=SEED,ohe_fe=encode_c,ordinal_encoder=False,
                 n_splits=5,early_stop=True,num_classes=7,cat_features=False,custom_metric=mapk,
                 fold_type='SKF')


%%time

params = {
        'objective': 'multi:softprob',  'num_class': 7, 'max_depth': 7,'learning_rate': 0.03,
        'subsample': 0.8,'max_bin': 128,'colsample_bytree': 0.3, 'colsample_bylevel': 1,  'colsample_bynode': 1,  
        'eval_metric': 'mlogloss','n_estimators':10000}

results_CAT_1 = base.Train_ML(params,'XGB',e_stop=50)


%%time

pred = results_CAT_1[1]

I_Map = {0: '28-28', 1: '17-17-17', 2: '10-26-26', 3: 'DAP', 4: '20-20', 5: '14-35-14', 6: 'Urea'}
def inverse_map(x):
    return I_Map[x]
V_inverse_map = np.vectorize(inverse_map)
top_3_preds = V_inverse_map(np.argsort(pred, axis=1)[:, -3:][:, ::-1])

sample['Fertilizer Name'] = [' '.join(row) for row in top_3_preds]
sample.to_csv('submission.csv', index=False)

