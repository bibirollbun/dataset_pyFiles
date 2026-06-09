import pandas as pd

submissions = 'Added' # 'Added','dataset' -stored, 'public notebooks' -can be named!

if submissions == f'public notebooks':

    p  = '/kaggle/input/'
    s  = '/submission.csv'
    a1 = '/sub_logistic-regression_0.366356.csv'
    a2 = '/sub_logistic-regression_0.366237.csv'
    
    S  = pd.read_csv(p+ 'predicting-optimal-fertilizers'               +s)
    A1 = pd.read_csv(p+ 's05e06-fertilizer-optimization-ensemble'      +a1)
    A2 = pd.read_csv(p+ 's05e06-fertilizer-optimization-ensemble'      +a2)
    B  = pd.read_csv(p+ 'xgb-triple-ensemble-map-3-opt-no-fe-v1'       +s)
    C1 = pd.read_csv(p+ 'optimal-fertilizers-xgb'                      +s)
    C2 = pd.read_csv(p+ 'Optimal Fertilizers | NN+XGB+LGBM+CAT+HGB+YDF'+s)
    R  = pd.read_csv(p+ 'xgboost-predicting-optimal-fertilizers-r'     +s)
    V  = pd.read_csv(p+ 'Fertilizer Name predicting with Metamodels'   +s)
    L  = pd.read_csv(p+ 'My first blend'                               +s)
    
if submissions == f'dataset':

    p1 = '/kaggle/input/fertilizer-1-0'
    p2 = '/submission__LB_'
    s  = '.csv'
    
    S = pd.read_csv(p1+p2+ '0_35298__v09__' +  'Satya'              +s)
    A = pd.read_csv(p1+p2+ '0_36826__v07__' +  'Mahdi_Ravagi'       +s)
    B = pd.read_csv(p1+p2+ '0_36626__v05__' +  'Patryk'             +s)
    C = pd.read_csv(p1+p2+ '0_36657__v01__' +  'Mikhail_Naumov'     +s)
    R = pd.read_csv(p1+p2+ '0_29667__v20__' +  'Kheirallah Samaha'  +s)
    V = pd.read_csv(p1+p2+ '0_36863__v01__' +  'Vishnupriya'        +s)
    L = pd.read_csv(p1+p2+ '0_36863__v01__' +  'Lion-li-li'         +s)
    
if submissions == f'Added': # If two strings are equal to each other. Or a string of one

    p2 = '/submission__LB_'
    a = "___Added"
    s = '.csv'
    
    p1  = '/kaggle/input/fertilizer-1-2'
    # dfA = pd.read_csv(p1+p2+ '0_36826__v07__' +  'Mahdi_Ravagi'   +a+s)
    # dfB = pd.read_csv(p1+p2+ '0_36626__v05__' +  'Patryk'         +a+s)
    # dfC = pd.read_csv(p1+p2+ '0_36657__v01__' +  'Mikhail_Naumov' +a+s)
    
    # df  = dfA       #    LB = 0.36_827 
    # df  = dfB       #    LB = 0.36_704
    # df  = dfC       #    LB = 0.36_819

# FIRST REPLACE:

    p1  = '/kaggle/input/fertilizer-1-3'
    # dfA = pd.read_csv(p1+p2+ '0_36826__v07__' +  'Mahdi_Ravagi'   +a+s)
    # dfV = pd.read_csv(p1+p2+ '0_36863__v01__' +  'Vishnupriya'    +a+s)
    # dfC = pd.read_csv(p1+p2+ '0_36657__v01__' +  'Mikhail_Naumov' +a+s)
    
    # df  = dfA       #    LB = 0.36_849
    # df  = dfV       #    LB = 0.36_863
    # df  = dfC       #    LB = ?   

# SECOND REPLASE:
    p1  = '/kaggle/input/fertilizer-1-4'
    # dfA = pd.read_csv(p1+p2+ '0_36826__v07__' +  'Mahdi_Ravaghi'  +a+s)
    # dfV = pd.read_csv(p1+p2+ '0_36863__v01__' +  'Vishnupriya'    +a+s)
    # dfL = pd.read_csv(p1+p2+ '0_36855__v01__' +  'Lion-li-li'     +a+s)
    
    # df  = dfA       #    LB = ?
    # df  = dfV       #    LB = ?
    # df  = dfL       #    LB = ?

# LAST REPLASE:
    p1  = '/kaggle/input/fertilizer-1-5'
    dfA = pd.read_csv(p1+p2+ '0_36826__v07__' +  'Mahdi_Ravaghi'  +a+s)
    dfL = pd.read_csv(p1+p2+ '0_36855__v01__' +  'Lion-li-li'     +a+s)
    dfC = pd.read_csv(p1+p2+ '0_36826__v02__' +  'Mikhail_Naumov' +a+s)
    
    df  = dfA        #    LB = ?
    df  = dfL        #    LB = ?
    df  = dfC        #    LB = ?


df =dfL 


df.to_csv('submission.csv', index=False)

df

