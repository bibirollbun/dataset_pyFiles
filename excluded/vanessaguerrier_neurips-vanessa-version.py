!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df_train= pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df_train.describe()
df_test= pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
df_test






dataset1=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
dataset2=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv")
dataset3=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
dataset4=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")

dataset1.rename(columns={'TC_mean': 'Tc'}, inplace=True)


df_traindata= df_train.copy()
df_traindata = pd.concat([df_traindata, dataset1], ignore_index=True)
df_traindata =df_traindata = pd.concat([df_traindata, dataset2], ignore_index=True)
df_traindata = df_traindata = pd.concat([df_traindata, dataset3], ignore_index=True)
df_traindata = df_traindata = pd.concat([df_traindata, dataset4], ignore_index=True)


print(df_train.describe())
print(df_traindata.describe())

scaler = StandardScaler()
df_train[["Tg","FFV", "Tc","Density","Rg"]] = scaler.fit_transform(df_train[["Tg","FFV", "Tc","Density","Rg"]])
df_traindata[["Tg","FFV", "Tc","Density","Rg"]] = scaler.fit_transform(df_traindata[["Tg","FFV", "Tc","Density","Rg"]])


def calculate_error(df_test,df_testPred):
    weight=[]
    n5=[]
    error=0
    colonne= ["Tg","FFV", "Tc","Density","Rg"]
    for col in colonne:
        avalide= df_test.shape[0]
        n5.append((1/np.sqrt(avalide)))
    total =np.sum(np.array(n5))
    for i in range(5): 
        ri=df_test[colonne[i]].max()- df_test[colonne[i]].min()
        weight.append((5*n5[i])/(total* ri))
    weight=np.array(weight)
    for j in range(df_test.shape[0]):
        true_row = df_test.iloc[j].to_numpy()
        pred_row = df_testPred[j] if isinstance(df_testPred, np.ndarray) else df_testPred.iloc[j].to_numpy()
        difference = true_row - pred_row
        error += difference @ weight.T 
    return error/df_test.shape[0]
        
    


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def smiles_to_ecfp4(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
    

y= df_train[["Tg","FFV", "Tc","Density","Rg"]].copy()
fingerprints= df_train['SMILES'].apply(smiles_to_ecfp4)
X = np.stack(fingerprints.values)


def predict_missing_Value(y, df_train):
    for col in y.columns:
        df_known = df_train[["SMILES",col]].dropna()
        df_missing = df_train[["SMILES", col]].loc[df_train[["SMILES", col]].isnull().any(axis=1)]
        fingerp= df_known['SMILES'].apply(smiles_to_ecfp4)
    
        X_known = np.stack(fingerp.values)
        y_known = df_known[col]
        
        X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.33, random_state=42)
        
        fingerp= df_missing['SMILES'].apply(smiles_to_ecfp4)
        X_toPredict= np.stack(fingerp.values)
        
        model1=XGBRegressor(booster='gbtree',          
                             objective='reg:squarederror',
                             eval_metric='auc',         
                             n_estimators=500,          
                             learning_rate=0.05,        
                             max_depth=6,              
                             min_child_weight=1,       
                             gamma=0,                   
                             subsample=0.8,             
                             colsample_bytree=0.8,      
                             reg_alpha=0,               
                             reg_lambda=1   )
        model1.fit(X_train, y_train)
        # print('general Score for ',col," is : ", 'for reg',model.score(X_train,y_train),'for RandomForest',model1.score(X_train,y_train))
        # print('test Score for ',col," is : ", model.score(X_test,y_test),'for RandomForest',model1.score(X_test,y_test))
    
        y_predicted = model1.predict(X_toPredict)

        missing_index = df_missing.index
        if len(missing_index) != len(y_predicted):
            print(f"⚠️ Mismatch : {len(missing_index)} NaNs, {len(y_predicted)} preds")
        else: 
            df_train.loc[missing_index, col] = y_predicted
    return df_train

df_train= predict_missing_Value(y, df_train)


df_traindata= predict_missing_Value(y, df_traindata)


from sklearn.multioutput import MultiOutputRegressor
def print_error(df_train):
    y= df_train[["Tg","FFV", "Tc","Density","Rg"]].copy()
    print(y.shape)
    fingerprints= df_train['SMILES'].apply(smiles_to_ecfp4)
    X = np.stack(fingerprints.values)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
    base_model =XGBRegressor(booster='gbtree',          
                             objective='reg:squarederror',
                             eval_metric='auc',         
                             n_estimators=500,          
                             learning_rate=0.05,        
                             max_depth=6,              
                             min_child_weight=1,       
                             gamma=0,                   
                             subsample=0.8,             
                             colsample_bytree=0.8,      
                             reg_alpha=0,               
                             reg_lambda=1,              
                             scale_pos_weight=1,        
                             n_jobs=-1,                 
                             random_state=42,           
                             verbosity=1     )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    print(y_pred.shape)
    error= mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    print('error is : ',error.tolist() , 'and the mAEw is : ' , calculate_error(y_test,y_pred) )
    return model
model1= print_error(df_train)


model2=  print_error(df_traindata)


colonne= ["Tg","FFV", "Tc","Density","Rg"]
X_test1= np.stack(df_test['SMILES'].apply(smiles_to_ecfp4))
y_test1=np.array(model1.predict(X_test1))
df_testCopy= df_test.copy()
for i in range(len(colonne)): 
    df_testCopy[colonne[i]]= y_test1[:,i]

df_testCopy[["Tg","FFV", "Tc","Density","Rg"]]=scaler.inverse_transform(df_testCopy[["Tg","FFV", "Tc","Density","Rg"]])
df_testCopy


X_test1= np.stack(df_test['SMILES'].apply(smiles_to_ecfp4))
y_test1=np.array(model2.predict(X_test1))
df_testCopy= df_test.copy()
for i in range(len(colonne)): 
    df_testCopy[colonne[i]]= y_test1[:,i]
df_testCopy[["Tg","FFV", "Tc","Density","Rg"]]=scaler.inverse_transform(df_testCopy[["Tg","FFV", "Tc","Density","Rg"]])
df_testCopy.to_csv('/kaggle/working/submission.csv', index=False)
df_testCopy

