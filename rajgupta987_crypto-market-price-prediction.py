!pip install koolbox scikit-learn==1.5.2


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import warnings 
import seaborn as sns 
import joblib 
import optuna 
from koolbox import Trainer   
from sklearn.model_selection import KFold 
from sklearn.linear_model import Ridge  
from lightgbm import LGBMRegressor 
from xgboost import XGBRegressor  
from scipy.stats import pearsonr as pr 
from sklearn.base import clone 
import gc 

warnings.filterwarnings('ignore')   


class CFG:
    train_path='/kaggle/input/drw-crypto-market-prediction/train.parquet'
    test_path='/kaggle/input/drw-crypto-market-prediction/test.parquet'
    submission_path='/kaggle/input/drw-crypto-market-prediction/sample_submission.csv' 
    target = "label"
    n_folds = 5
    seed = 42

    run_optuna = True
    n_optuna_trials = 500


train_data=pd.read_parquet(CFG.train_path)
test_data=pd.read_parquet(CFG.test_path)


def reduce_memory_usage(df): 
    initial_mem=df.memory_usage().sum()/1024**3
    print(f"starting memory:{initial_mem:.2f} GB")   
    for col in df.columns:  
        col_type=df[col].dtype
        
        col_min=df[col].min() 
        col_max=df[col].max()

        if str(col_type)[:3]=='int': 
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max: 
                df[col] = df[col].astype(np.int8) 
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max: 
                df[col] = df[col].astype(np.int16) 
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max: 
                df[col] = df[col].astype(np.int32) 
            elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max: 
                df[col] = df[col].astype(np.int64)   
        else :
            if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max: 
                df[col] = df[col].astype(np.float16)  
            elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max: 
                df[col] = df[col].astype(np.float32)   
            else : 
                df[col] = df[col].astype(np.float64)     
    final_mem=df.memory_usage().sum()/1024**3 
    reduced_by=100*(initial_mem - final_mem)/initial_mem
    print(f'initial memory : {initial_mem:.2f}GB') 
    print(f'final memory : {final_mem:.2f}GB') 
    print(f'reduced memory by : {reduced_by:.2f}%')
    return df


train_data = reduce_memory_usage(train_data)
test_data = reduce_memory_usage(test_data)


def add_features(df):   
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty'] 
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty']+1e-10)
    df['sell_pressure'] = df['sell_qty'] / (df['volume']+1e-10)
    df['log_volume'] = np.log1p(df['volume'])   

    df['effective_spread_proxy'] = np.abs(df['buy_qty']-df['sell_qty'])/(df['volume']+1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty']-df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] 
                                                              + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) /( df['buy_qty'] + df['sell_qty']
                                                                  + 1e-10)
    df['liquidity_ratio']=(df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    #Price Pressure indiactors
    
    df['net_order_flow'] = df['buy_qty']-df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] /( df['volume'] + 1e-10)
    df['volume_weighted_buy'] =df['volume'] * df['buy_qty']
    
    #Liquidity Depth Measures
    
    df['total_depth']=df['bid_qty']+ df['ask_qty']   
    df['depth_imbalance'] = (df['bid_qty']-df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 
                                                                    1e-10)  
    df['log_depth']=np.log1p(df['total_depth'])
    
    # Order flow Toxicity 
    
    df['kyle_lambda']=np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity']= np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth']
                                                                     + 1e-10)
    # Market ACTIVITY Indicators
    df['volume_depth_ratio'] = df['volume'] /(df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty']) 
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty']) 
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    df['realized_spread_proxy'] = 2*np.abs(df['net_order_flow'])/(df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])  
    
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])  
    
    df['trade_informativeness'] = df['net_order_flow']/(df['bid_qty'] + df['ask_qty'] +1e-10) 
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10) 
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] +1e-10) * df['volume'] 
    
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 
                                                                      1e-10) 
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 
                                                               1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10) 
    
    
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2  
   
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10) 


    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)


    df['net_buying_ratio'] = df['net_order_flow']/(df['volume']+1e-10) 
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume']) 
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume'] 
    
    return df.replace([np.inf,-np.inf],0).fillna(0)   

def create_aggregated_features(df,feature_list,prefix): 
    df[f'{prefix}_sum'] = df[feature_list].sum(axis=1)
    df[f'{prefix}_mean'] = df[feature_list].mean(axis=1)
    df[f'{prefix}_median'] = df[feature_list].median(axis=1)
    df[f'{prefix}_max'] = df[feature_list].max(axis=1)
    df[f'{prefix}_min'] = df[feature_list].min(axis=1)
    df[f'{prefix}_std'] = df[feature_list].std(axis=1)
    return df

def extract_time_features(df): 
    df['timestamp'] = pd.to_datetime(df['timestamp']) 
    df['hour'] = df['timestamp'].dt.hour 
    dt['minute'] = df['timestamp'].dt.minute
    dt['dayofweek'] = df['dayofweek'].dt.dayofweek
    dt['Y_M_D_H'] = df['timestamp'].dt.strftime('%Y-%m-%d-%H') 
    dt['Y_M_D_H_M'] = df['timestamp'].dt.strftime('%Y-%m-%d-%H-%M') 
    return df  


train_data = add_features(train_data)
test_data = add_features(test_data)


Negative_features_list = ['X563', 'X560', 'X486', 'X215', 'X7', 'X290', 'X581', 'X492',
       'X569', 'X590', 'X201', 'X625', 'X247', 'X283', 'X261', 'X626',
       'X193', 'X545', 'X195', 'X291', 'X703', 'X243', 'X297', 'X38',
       'X490', 'X252', 'X42', 'X41', 'X254', 'X706', 'X237', 'X40', 'X16',
       'X331', 'X47', 'X571', 'X302', 'X711', 'X280',
       'activity_intensity', 'X412', 'X580', 'X54', 'X679', 'X245',
       'X700', 'X622', 'X646', 'X223', 'X608', 'X695', 'X48', 'X319',
       'X621', 'X125', 'X539', 'X259', 'X651', 'X648', 'X278', 'X682',
       'X441', 'X5', 'X378', 'X565', 'X343', 'X654', 'X629', 'X14',
       'X448', 'ask_qty', 'bid_qty', 'X95', 'X127', 'X463', 'X698',
       'X482', 'X123', 'X46', 'X488', 'X572', 'X521', 'X30', 'X562',
       'X421', 'X578', 'X380', 'X303', 'X171', 'X731', 'X452', 'X690',
       'X221', 'X226', 'X456', 'X32', 'X643', 'X115', 'X676', 'X217',
       'X602', 'X329', 'X765', 'X507', 'X722', 'X708', 'X179', 'X636',
       'X210', 'X670', 'X767', 'X416', 'X218', 'X317', 'X204', 'X461',
       'X250', 'X208', 'X159', 'X385', 'X198', 'X450', 'X28', 'X457',
       'X400', 'X373', 'X272', 'X77', 'X154', 'X777', 'X256', 'X203',
       'X20', 'X326', 'X407', 'X476', 'X274', 'X424', 'X33', 'X267',
       'X443', 'X499', 'X113', 'X547', 'X382', 'X554', 'X583', 'X76',
       'X639', 'X81']  

Positive_features_list = ['X557', 'X566', 'X485', 'X194', 'X575', 'X584', 'X587', 'X493',
       'X216', 'X44', 'X253', 'X627', 'X50', 'X707', 'X36', 'X6', 'X287',
       'X262', 'X551', 'X244', 'X8', 'X286', 'X214', 'X236', 'X624',
       'X623', 'X483', 'X337', 'X15', 'X240', 'X200', 'X289', 'X39',
       'X699', 'X533', 'X222', 'X260', 'X325', 'X647', 'X299', 'X285',
       'X650', 'X577', 'X284', 'X559', 'X43', 'X202', 'X702', 'X251',
       'X704', 'X246', 'X418', 'X710', 'X683', 'X55', 'X620', 'X694',
       'X298', 'X31', 'X292', 'X574', 'X239', 'X605', 'X288', 'X494',
       'X121', 'X449', 'X462', 'X89', 'X696', 'X489', 'total_depth',
       'X678', 'X165', 'X628', 'X406', 'X675', 'X568', 'X294', 'X384',
       'X131', 'X481', 'X374', 'X119', 'X56', 'X133', 'X442', 'X277',
       'X422', 'X655', 'X9', 'X451', 'X455', 'X209', 'X440', 'X652',
       'X379', 'X323', 'X279', 'X611', 'X735', 'X766', 'X527', 'X205',
       'X129', 'X644', 'X21', 'X224', 'X196', 'X642', 'X313', 'X293',
       'X427', 'X258', 'X586', 'X225', 'X726', 'X691', 'X219', 'X335',
       'X4', 'X635', 'X506', 'X415', 'X173', 'X458', 'X117', 'X372',
       'X197', 'X27', 'X281', 'X672', 'X275', 'X500', 'X775', 'X469',
       'X680', 'X599', 'X35', 'X29', 'X477', 'X401', 'X330', 'X160',
       'X370', 'X266', 'X408', 'X640', 'X87', 'X70']         



for i in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]:
    train_data = create_aggregated_features(train_data, Negative_features_list[:i], f"Negative_features_{i}")
    test_data = create_aggregated_features(test_data, Negative_features_list[:i], f"Negative_features_{i}")
    train_data = create_aggregated_features(train_data, Positive_features_list[:i], f"Positive_features_{i}")
    test_data = create_aggregated_features(test_data, Positive_features_list[:i], f"Positive_features_{i}")


Features = ['X455', 'X557', 'X6', 'X259', 'X627', 'X450', 'X48', 'X44', 'X448', 'X291',
            'X563', 'X218', 'X574', 'X695', 'X41', 'X418', 'X644', 'X621', 'X648', 'X451',
            'X443', 'X643', 'Negative_features_10_sum', 'X722', 'X290', 'X200', 
            'Positive_features_10_sum', 'X458', 'X286', 'X571', 'X640', 'X559', 'X625',
            'X222', 'X201', 'X214', 'X193', 'X639', 'X197', 'X31', 'X323', 'X456', 
            'X319', 'X278', 'X260', 'X251', 'X651', 'X5', 'X652', 'X285', 'X584', 
            'X317', 'X412', 'X647', 'Positive_features_100_std', 'X42', 'X440', 'X225',
            'X575', 'X203', 'X683', 'X289', 'X204', 'X35', 'X562', 'X566', 'X605', 
            'X33', 'X293', 'Negative_features_5_sum', 'X219', 'X45', 'X587', 'X36',
            'X16', 'X449', 'X569', 'X287', 'X246', 'X56', 'X726', 'X8', 'X379', 'X292',
            'X679', 'X507', 'X406', 'Negative_features_150_std', 'X165', 'X329', 'X376', 
            'X15', 'X608', 'X626', 'X678', 'X628', 'X696', 'X457', 'X539', 'X469', 
            'X442', 'Positive_features_5_sum', 'Positive_features_60_std',
            'Positive_features_90_std', 'X580', 'X694', 'X491', 'X258', 'X454', 'X46', 
            'X196', 'X123', 'X765', 'X370', 'X215', 'Negative_features_100_std', 
            'X331', 'X125', 'X119', 'X254', 'X279', 'X4', 'X298', 'X551', 'X768', 'X675',
            'X226', 'X223', 'X43', 'X568', 'X444', 'X337', 'X320', 
            'Negative_features_60_std', 'X335', 'X581', 'X623', 'X294', 'X325', 'X572',
            'X554', 'X416', 'X486', 'X602', 'X484', 'X577', 'X735', 'X288', 'X250', 'X691',
            'X776', 'X766', 'X113', 'X611', 'X635','X666', 'X401', 'X489', 'X326',
            'X699', 'X718', 'X636', 'X767', 'X476', 'Negative_features_70_std', 'X154',
            'X221', 'X706', 'X506', 'X470', 'Positive_features_30_sum', 'X34', 'X21',
            'X284', 'X38', 'X245', 'X700', 'X620', 'X415', 'X407',
            'Positive_features_90_median', 'X32', 'X244', 'X670', 'X95', 'X343', 'X299',
            'X129', 'X421', 'X482', 'X14', 'X668', 'X704', 'X9', 'X216', 'X77', 'X247',
            'X155', 'Negative_features_90_std', 'Negative_features_50_sum', 'X477',
            'X642', 'X465', 'X160', 'X373', 'X629', 'X777', 'X609', 'X724', 'X422', 
            'X672', 'X441', 'X302', 'X646', 'X527', 'X707', 'X71', 'X39', 'X194', 
            'X728', 'X410', 'X682', 'Positive_features_70_std', 'X49', 'X127', 'X198', 
            'X229', 'X579', 'X599', 'X622', 'X601', 'X461', 'fill_probability', 'X121',
            'Positive_features_20_std', 'X676', 'Negative_features_30_sum', 'X573', 
            'X720', 'X50', 'X209', 'X690', 'X173', 'X89', 'X313', 'X439', 'X275', 
            'X171', 'X20', 'X30', 'X708', 'X237', 'X710', 'X217', 'X488', 
            'Negative_features_50_std', 'X133', 'X404', 'X281', 'X427', 'X28', 'X70', 
            'X650', 'X311', 'X731', 'X764', 'X385', 'X274', 'X51', 
            'Negative_features_150_median', 'X205', 'X283', 'X266', 'X408', 'X424', 
            'X610', 'X277', 'X261', 'X362', 'X230', 'X256', 'X280', 'X624', 'X447', 
            'X303', 'X175', 'X485', 'X462', 'X262', 'X115', 'Positive_features_50_sum', 
            'X117', 'Negative_features_30_std', 'X775', 'X565', 'X27',
            'Negative_features_80_std', 'X377', 'X400', 'X272', 'X698', 'X578', 
            'Negative_features_60_sum', 'X159', 'X664', 'X269', 'X87', 'X680', 'X179',
            'X508', 'X481', 'X499', 'X114', 'X545', 'X604', 'Positive_features_40_median',
            'Positive_features_50_std', 'X267', 'X253', 'X459', 'X118', 'X547', 'X76', 
            'X464', 'Positive_features_150_std', 'X479', 'X295', 'X161', 'X192', 'X7', 
            'X316', 'Positive_features_90_max', 'X23', 'X548', 'X494', 'X500', 'X174', 
            'X521', 'X533', 'X655', 'X236', 'X24', 'ask_ratio', 'X382', 'X492', 'X380',
            'X368', 'X29', 'X242', 'X148', 'X463', 'X238', 'X662', 'X598', 'X149', 'X124',
            'X81', 'Negative_features_70_sum', 'X606', 'Negative_features_40_sum', 
            'X243', 'X723', 'X169', 'X314', 'X157', 'X84', 'X692', 
            'Negative_features_5_mean', 'X402', 'X328', 'X612', 'X613', 'X252', 'X769', 
            'Positive_features_5_mean', 'X210', 'X322', 'X235', 'X334', 'X271', 'X656',
            'X409', 'Positive_features_5_std', 'X395', 'X255', 'X213', 'X107', 'X498', 
            'X446', 'X338', 'X530', 'X600', 'X112', 'X151', 'X583', 'X228', 'X78', 'X384',
            'X372', 'X139', 'Negative_features_80_sum', 'X371', 'sell_ratio', 'X211', 
            'X120', 'X341', 'net_order_flow', 'X550', 'X90', 'Positive_features_40_std', 
            'X300', 'X667', 'X344', 'X40', 'X65', 'X497', 'X413', 'X490', 'X483', 
            'Negative_features_40_median', 'X212', 'X398', 'X73', 'X231', 'buy_qty', 
            'X734', 'Positive_features_30_std', 'Positive_features_80_max', 'X475', 
            'X163', 'X386', 'X542', 'X472', 'Negative_features_90_sum', 'X619', 'X282', 
            'activity_intensity', 'X576', 'X336', 'X241', 'X560', 'X375', 'X417', 'X333',
            'X330', 'Positive_features_50_median', 'X590', 'Negative_features_10_mean',
            'X240', 'Negative_features_40_std']


X = train_data[Features] 
y = train_data[CFG.target]
X_test = test_data[Features]


def pearsonr(y_true, y_pred):
    return pr(y_true, y_pred)[0]


lgbm_params = {
    "boosting_type": "gbdt",
    "colsample_bytree": 0.5625888953382505,
    "learning_rate": 0.029312951475451557,
    "min_child_samples": 63,
    "min_child_weight": 0.11456572852335424,
    "n_estimators": 126,
    "n_jobs": -1,
    "num_leaves": 37,
    "random_state": 42,
    "reg_alpha": 85.2476527854083,
    "reg_lambda": 99.38305361388907, 
    "subsample" : 0.450669817684892,
    "verbose" : -1
} 

lgbm_goss_params ={
    "boosting_type":"goss",
    'colsample_bytree': 0.34695458228489784,
    'learning_rate':0.031023014900595287, 
    'min_child_samples':30, 
    'min_child_weight':0.4727729225033618,
    'n_estimators':220, 
    "n_jobs":-1, 
    "num_leaves":58, 
    "random_state":42, 
    "reg_alpha":38.665994901468224,
    "reg_lambda":92.76991677464294, 
    "subsample":0.4810891284493255,
    "verbose":-1
} 

xgb_params={
    'colsample_bylevel':0.4778015829774066,
    "colsample_bynode": 0.362764358742407,
    "colsample_bytree": 0.7107423488010493, 
    "gamma": 1.7094857725240398,
    "learning_rate": 0.02213323588455387,
    "max_depth":20, 
    "max_leaves":12, 
    "min_child_weight":16,
    "n_estimators":200, 
    "n_jobs":-1, 
    "random_state":42, 
    "reg_alpha":39.352415706891264,
    "reg_lambda":75.44843704068275,
    "subsample": 0.06566669853471274,
    "verbosity":0
}


fold_scores = {}
overall_scores = {}

oof_preds = {}
test_preds = {}


lgbm_trainer = Trainer(
    LGBMRegressor(**lgbm_params),
    cv=KFold(n_splits=5, shuffle=False),
    metric=pearsonr,
    task="regression",
    metric_precision=6
)

lgbm_trainer.fit(X, y)

fold_scores['LightGBM (gbdt)'] = lgbm_trainer.fold_scores  
overall_scores['LightGBM (gbdt)'] = [pearsonr(lgbm_trainer.oof_preds,y)] 
oof_preds['LightGBM (gbdt)'] = lgbm_trainer.oof_preds 
test_preds['LightGBM (gbdt)'] = lgbm_trainer.predict(X_test)


lgbm_goss_trainer=Trainer(
    LGBMRegressor(**lgbm_goss_params),
    cv=KFold(n_splits=5,shuffle=False), 
    metric=pearsonr,
    task='regression',
    metric_precision=6
) 
lgbm_goss_trainer.fit(X,y) 

fold_scores['LightGBM (goss)'] = lgbm_goss_trainer.fold_scores 
overall_scores['LightGBM (goss)'] =[pearsonr(lgbm_goss_trainer.oof_preds,y)] 
oof_preds['LightGBM (goss)'] = lgbm_goss_trainer.oof_preds 
test_preds['LightGBM (goss)'] = lgbm_goss_trainer.predict(X_test)


xgb_trainer=Trainer(
    XGBRegressor(**xgb_params), 
    cv=KFold(n_splits=5,shuffle=False),
    metric=pearsonr,
    task='regression',
    metric_precision=6
) 
xgb_trainer.fit(X,y) 

fold_scores['XGBoost'] = xgb_trainer.fold_scores 
overall_scores['XGBoost'] = [pearsonr(xgb_trainer.oof_preds,y)] 
oof_preds['XGBoost'] = xgb_trainer.oof_preds 
test_preds['XGBoost'] = xgb_trainer.predict(X_test)


X=pd.DataFrame(oof_preds)  
X_test=pd.DataFrame(test_preds)


def plot_weights(weights,title): 
    sorted_indices=np.argsort(weights[0])[::-1]
    sorted_coeffs=np.array(weights[0])[sorted_indices]
    sorted_model_names=np.array(list(oof_preds.keys()))[sorted_indices] 
    
    plt.figure(figsize=(10,weights.shape[1]*0.5))
    ax=sns.barplot(x=sorted_coeffs,y=sorted_model_names,palette='RdYlGn_r') 
    
    
    for i,(value,name) in enumerate(zip(sorted_coeffs,sorted_model_names)): 
        if value>=0: 
            ax.text(value,i,f'{value:.3f}',va='center',ha='left',color='black')
        else:
            ax.text(value,i,f'{value:.3f}',va='center',ha='right',color='black') 
        xlim=ax.get_xlim() 
        ax.set_xlim(xlim[0]-0.1*abs(xlim[0]),xlim[1]+0.1*abs(xlim[1]))
    
    plt.title(title)
    plt.xlabel("") 
    plt.ylabel("") 
    plt.tight_layout() 
    plt.show()


def objective(trial):
    params={
        'random_state':CFG.seed,
        'alpha': trial.suggest_float('alpha',0,1), 
        'tol': trial.suggest_float('tol',1e-6,1e-2), 
        'fit_intercept' : trial.suggest_categorical('fit_intercept',[True,False]), 
        'positive':trial.suggest_categorical('positive',[True,False])
    } 
    trainer=Trainer(
        Ridge(**params),
        cv=KFold(n_splits=5,shuffle=False), 
        metric=pearsonr,
        task='regression',
        verbose=False
    ) 
    trainer.fit(X,y)
    
    return pearsonr(trainer.oof_preds,y) 

if CFG.run_optuna: 
    sampler = optuna.samplers.TPESampler(seed=CFG.seed,multivariate = True,n_startup_trials=CFG.n_optuna_trials//10) 
    study = optuna.create_study(direction='maximize',sampler=sampler) 
    study.optimize(objective,n_trials=CFG.n_optuna_trials,n_jobs=-1) 
    best_params=study.best_params  
    ridge_params={
        'random_state':CFG.seed, 
        'alpha':best_params["alpha"],
        'tol':best_params['tol'], 
        'fit_intercept':best_params['fit_intercept'], 
        'positive':best_params['positive']
    } 
else:
    ridge_params={
        random_state:CFG.seed
    }


ridge_trainer=Trainer(
    Ridge(**ridge_params),
    cv=KFold(n_splits=5,shuffle=False),
    metric=pearsonr,
    task='regression', 
    metric_precision=6,
)   
ridge_trainer.fit(X,y) 

fold_scores['Ridge (ensemble)'] = ridge_trainer.fold_scores 
overall_scores['Ridge (ensemble)'] = [pearsonr(ridge_trainer.oof_preds,y)] 
ridge_test_preds = ridge_trainer.predict(X_test)


ridge_coeffs = np.zeros((1,X.shape[1])) 
for m in ridge_trainer.estimators:
    ridge_coeffs +=m.coef_ 
ridge_coeffs = ridge_coeffs / len(ridge_trainer.estimators)     

plot_weights(ridge_coeffs,"Ridge Coefficients")


sub = pd.read_csv(CFG.submission_path) 
sub['prediction']=ridge_test_preds  
sub.to_csv(f'submission.csv',index=False)
sub.head() 

