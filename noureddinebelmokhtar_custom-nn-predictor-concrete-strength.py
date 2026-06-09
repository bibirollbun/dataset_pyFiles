!pip install --upgrade scikit-learn==1.6.1


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


import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import joblib
import linecache
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


 train_data = pd.read_csv("/kaggle/input/train-val-cs/train_dataset.csv")
 test_data = pd.read_csv("/kaggle/input/train-val-cs/test_dataset.csv")


def normalization(param):
    df = pd.read_csv("/kaggle/input/concrete-strength-regression/test.csv")
    filename = f'/kaggle/input/scaler-cs/scaler_features_1.joblib'
    txt_fil =f"/kaggle/working/df_{param}_features.txt"
   
    X_test = df.drop(columns=['Row ID'])
    id_ = df[['Row ID']]

    # Load the scaler from the file
    loaded_scaler = joblib.load(filename)
    print(f"Scaler loaded from '{filename}'")

    X_test_scaled = loaded_scaler.transform(X_test)
    
    # Convert back to DataFrames for saving
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Step 4: Save to TXT (tab-separated or comma-separated)
    X_test_scaled.to_csv(txt_fil, header=None, index=False, sep='\t')

    print("Datasets saved as CSV and TXT.")
    print(txt_fil)
    return id_
param = 'csMPa'
id_ = normalization(param)


def neuron(type_fonc, n1):
    try:
        if type_fonc == 0:  #sigmoid
            a = 1/(1+math.exp(-n1))
        elif type_fonc == 1:   #identity
            a = n1
        elif type_fonc == 2:  #tanh
            a = math.tanh(n1) #2/(1+math.exp(-2 *n1))-1
        elif type_fonc == 3:  #relu
                if n1<=0:
                    a = 0
                if n1>0:
                    a = n1
        elif type_fonc == 4:  #leaky_relu
            if n1 > 0 :
                a = n1
            else:
                a = 0.01 * n1
    except OverflowError:
        a = float('inf')

    return (a)

def valid(test_fill, param, t_v_t):
    
    nn_fille_2=test_fill
    prediction = []
    origine_list = []
    final = 0
    r = 0
    for df in pd.read_csv(nn_fille_2, sep = "\t", header = None, iterator=True, chunksize =1 ):
 
        incrimente=0
        n_output_lear = 0
        k = 0
        j = 0
        i = 0
        
        var=0
        h_lears = None
        
        input_1 = 0
        outt = None
        n11=0
        type_f_a = []
        type_f_w_b = []
        n_neron = []
        
        #fichier1[0]=[]
        output_nerons_num=[]
        p=[]
       
        file_path = f"/kaggle/input/model-cs/seting_4.5135.txt"
        line_number = 1 
        line = linecache.getline(file_path, line_number)
        fichier1 = line.strip().split("\t")
        fichier1 = line.strip().split("\t")
        input_1 = int(fichier1[0])
        output = int(fichier1[1])
        h_lears = int(fichier1[2])
        n_output_lear=h_lears+1
        k = 3
        
        while k<3+h_lears+2:
            n_neron.append(int(fichier1[k]))
            k += 1
        data_kind = int(fichier1[k])
        k=k+1  #7
        output_nerons_num_size=int(fichier1[k])
        
        k=k+1
        
        for i in range(0,output_nerons_num_size):
            output_nerons_num.append(int(fichier1[k]))
           
            k += 1
        #n_neron.append(0)
        i=0
        p=[]
        #k=9
        n_type_f = int(fichier1[k])
        k += 1
        j = 0
        while j<data_kind:
            i = 0
            p=[]
            while i<n_type_f:
                p.append(int(fichier1[k]))
                i += 1
                k += 1
            p.append(p)
            p=[]
            j += 1
        j = 0
        while j<output_nerons_num_size:
            i = 0
            p=[]
            while i<n_type_f:
                p.append(int(fichier1[k]))
                i += 1
                k += 1
            p.append(p)
            p=[]
            j += 1
        pages = int(fichier1[k])
        rows = int(fichier1[k+1])
        colons = int(fichier1[k+2])
        x_factor = int(fichier1[k+3]) 
       
        k=k+3
        x_factor = int(fichier1[k+1])
        x_factor = int(fichier1[k+2])
        x_factor = int(fichier1[k+3])
        x_factor = int(fichier1[k+4])
        x_factor = (fichier1[k+5]) 
        x_factor = (fichier1[k+6]) 
        k=k+6
        x_factor = int(fichier1[k+1]) 
        x_factor = int(fichier1[k+2])
        x_factor = float(fichier1[k+3])
   
        x_factor = fichier1[k+4]
        
        x_factor = int(fichier1[k+5])
        x_factor = int(fichier1[k+6]) 
        x_factor = float(fichier1[k+7])
        x_factor = int(fichier1[k+8])
        k=k+9
        p=0
        pp=[]
        for p in range(0,h_lears):
            for i in range(0, data_kind):
                k+=1
                
        for p in range(0,output_nerons_num_size):
            k+=1
        
        for p in range(0,h_lears):
            for i in range(0, data_kind):
                pp.append(int(fichier1[k]))
                k+=1
            type_f_a.append(pp)
            pp=[]
            
        for p in range(0,output_nerons_num_size):
            pp.append(int(fichier1[k]))
            k+=1
        type_f_a.append(pp)
        p=[]
     
        pini2= [0 for _ in range(input_1 + output)]
        pp=[]
        weigth_01=[]
        p=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i]):
                for k in range(0,n_neron[i+1]):
                    p.append(258)
                pp.append(p)
                p=[]
            weigth_01.append(pp)
            pp=[]
        weigth=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i]):
                for k in range(0,n_neron[i+1]):
                    p.append(258)
                pp.append(p)
                p=[]
            weigth.append(pp)
            pp=[]
        weigthT= []
        for i in range(0,pages):
            for k in range(0,n_neron[i+1]):
                for j in range(0,n_neron[i]):
                    p.append(258)
                pp.append(p)
                p=[]
            weigthT.append(pp)
            pp=[]
        bais=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i+1]):
                p.append(0)
            bais.append(p)
            p=[]
        a=[]
        for i in range(0,pages+1):
            for j in range(0,n_neron[i]):
                p.append(0)
            a.append(p)
            p=[]
        n=[]
        for i in range(0,pages):
            for j in range(0,n_neron[i+1]):
                p.append(0)
            n.append(p)
            p=[]
        #initialisation de "out"
        out_file = open(nn_fille_2, 'r')
        in_out_validation=len(out_file.readlines())
        out=np.zeros((in_out_validation,output))
        out_file.close()
        
        #fin
        file_path = f"/kaggle/input/model-cs/w_4.5135.txt"
        line_number = 1
        line = linecache.getline(file_path, line_number)
        data = list(map(float, line.strip().split("\t")))
        incrimente=0
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigth[i][j][k] =float(data[incrimente])
                    incrimente+=1
        incrimente=0
     
        #fin
        
        file_path = f"/kaggle/input/model-cs/b_4.5135.txt"
        line_number = 1
        line = linecache.getline(file_path, line_number)
        data = list(map(float, line.strip().split("\t")))
        for i in range(0,pages):
            colon =n_neron[i+1]
            for k in range(0,colon):
                bais[i][k] =float(data[incrimente])
                incrimente+=1
        incrimente=0
        
        #fin
        
            
        file_path = f"/kaggle/input/model-cs/01table_4.5135.txt"
        line_number = 1
        line = linecache.getline(file_path, line_number)
        data = list(map(int, line.strip().split("\t")))  
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigth_01[i][j][k] =int(data[incrimente])
                    incrimente+=1
        incrimente=0
        
        data = []
    
        #fin
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigth[i][j][k] =weigth[i][j][k]*weigth_01[i][j][k]
    
   
        trans_a=df.values.tolist()
        #input append
        if t_v_t != "test":
            input_2 = input_1 + output
        else:
            input_2 = input_1
        for i in range(0, input_2):
            pini2[i]=trans_a[0][i]
            if i < input_1: 
                trans_2=pini2[i]
                a[0][i]=trans_2
            
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            for j in range(0,row):
                for k in range(0,colon):
                    weigthT[i][k][j] =weigth[i][j][k]
       
        for i in range(0,pages):
            colon =n_neron[i+1]
            row = n_neron[i]
            sweetch=0
            kind_neron=0
            for j in range(0,colon):
                for k in range(0,row):
                    n11 = a[i][k] * weigthT[i][j][k] + n11
                    
                n[i][j]=n11+bais[i][j]
                n11=0
                #calcul des a[i][j]
                
                a[i+1][j]=neuron(type_f_a[i][sweetch],n[i][j])
                kind_neron= kind_neron + 1
                if i<pages-1:
                    if kind_neron==n_neron[i+1]/data_kind:
                        sweetch= sweetch + 1
                        kind_neron=0
                if i==pages-1:
                    if kind_neron==output_nerons_num[sweetch]:
                        sweetch= sweetch + 1
                        kind_neron=0
                if i==h_lears and j<output:
                    out[r][j]=a[i+1][j]
                
                if i==0:
                    if j==input_1 :
                        j=input_1 
                    if j==input_1 :
                        j=input_1 
       
        prediction.append(a[n_output_lear][0])
        if t_v_t == 'train' or t_v_t == 'val':
            origine_list.append(pini2[input_1])
        r=r+1
  
    return prediction, origine_list
prediction = []
train_val_test =["train", "val", "test"]
for t_v_t in train_val_test:    
    param =  'csMPa'
    if t_v_t == "train":
        test_fill = "/kaggle/input/train-val-cs/train_dataset_1.txt"
    elif t_v_t == "val":
        test_fill = "/kaggle/input/train-val-cs/test_dataset_1.txt"
    elif t_v_t == "test":
        test_fill= f"/kaggle/working/df_{param}_features.txt"
    prediction, origine_list = valid(test_fill, param, t_v_t)
    
    if t_v_t != "test":
        print(f"{t_v_t} dataset :")
        mse = mean_squared_error(origine_list, prediction)       
        print(f"MSE error: {mse}")
        rmse = np.sqrt(mean_squared_error(origine_list, prediction))
        print(f"RMSE = {rmse:.4f}")
        mae = mean_absolute_error(origine_list, prediction)
        print(f"MAE = {mae:.4f}")
        r2 = r2_score(origine_list, prediction)
        print(f"R² = {r2:.4f}")
        plt.scatter(origine_list, prediction)
        plt.plot([min(origine_list), max(origine_list)], [min(origine_list), max(origine_list)], 'r--')  # diagonale idéale
        plt.xlabel("Vrai")
        plt.ylabel("Prédit")
        plt.title("Prédiction vs Réel")
        plt.grid()
        plt.show()



prediction = pd.DataFrame({param : prediction})
test_df = pd.concat([id_, prediction], axis=1)
test_df.to_csv("/kaggle/working/submission.csv", index=False)
print(test_df)

