import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from category_encoders.cat_boost import CatBoostEncoder
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")


train = None
test = None
sub = None
orig = None
myenv = Path("/kaggle/input/playground-series-s5e10/train.csv")
if(myenv.is_file()):
    train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv',index_col='id')
    sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv',index_col='id')
    orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
    orig_2 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
    orig_3 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')
    orig = pd.concat([orig, orig_2, orig_3])

else:
    train = pd.read_csv('train.csv',index_col='id')
    test = pd.read_csv('test.csv',index_col='id')
    sub = pd.read_csv('sample_submission.csv',index_col='id')
    orig = pd.read_csv('synthetic_road_accidents_100k.csv')
    orig_2 = pd.read_csv('synthetic_road_accidents_10k.csv')
    orig_3 = pd.read_csv('synthetic_road_accidents_2k.csv')
    orig = pd.concat([orig, orig_2, orig_3])

print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Sub Shape:', sub.shape)
print('Orig Shape:', orig.shape)
train.head(3)


TARGET = 'accident_risk'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
print(f'{len(BASE)} Base Features:{BASE}')


ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # MIN
    min_map = orig.groupby(col)[TARGET].min()
    new_min_col_name = f"orig_min_{col}"
    min_map.name = new_min_col_name
    
    train = train.merge(min_map, on=col, how='left')
    test = test.merge(min_map, on=col, how='left')
    ORIG.append(new_min_col_name)

print(len(ORIG), 'Orig Features Created!!')
train['orig_mean_curvature'] = train['orig_mean_curvature'].fillna(orig[TARGET].mean())
test['orig_mean_curvature'] = test['orig_mean_curvature'].fillna(orig[TARGET].mean())

train['orig_min_curvature'] = train['orig_min_curvature'].fillna(orig[TARGET].mean())
test['orig_min_curvature'] = test['orig_min_curvature'].fillna(orig[TARGET].mean())


bools = test.select_dtypes(include='bool').columns.tolist()
train[bools] = train[bools].astype('int')
test[bools] = test[bools].astype('int')
cats = test.select_dtypes(exclude='number').columns.tolist()
numerics = test.select_dtypes(include='number').columns.tolist()
print(bools)
print(cats)
print(numerics)


cb = CatBoostEncoder()
cbtrain = cb.fit_transform(train.loc[:,cats],train.accident_risk)
cbtest = cb.transform(test.loc[:,cats])
cbtrain = pd.concat([cbtrain,train.loc[:,numerics]],axis=1)
cbtrain['accident_risk'] = train.accident_risk
cbtest = pd.concat([cbtest,test.loc[:,numerics]],axis=1)
train = pd.concat([cbtrain,train[numerics],train[['accident_risk']]],axis=1)
test = pd.concat([cbtest,test[numerics]],axis=1)


def GP(data):
    d = pd.DataFrame()
    d["gp_0"] = (((((data["speed_limit"] + data["num_reported_accidents"]) + data["num_reported_accidents"]) * data["orig_mean_lighting"]) / ((0.4149648547)) + ((52.5000000000) * ((data["curvature"] - ((0.4160095453) - data["orig_mean_weather"])) - ((0.4160095453) - data["orig_mean_lighting"])) * data["orig_mean_weather"]) / ((0.3094571829)))/2.0 ) 
    d["gp_1"] = (((np.sin((np.cos(data["curvature"])) / (np.sin(np.sin((np.cos(data["orig_mean_num_reported_accidents"])) / (data["orig_mean_weather"]))))) - np.sin((data["speed_limit"]) / (np.cos(-1.0000000000))) * 2.0) + np.cos(data["speed_limit"] / 2.0) * 2.0)/2.0  * 2.0 * 2.0) 
    d["gp_2"] = (((np.where(data["curvature"] > (0.4948371053), np.where(data["orig_mean_num_reported_accidents"] > (0.4025629163),  9.0 , 0 ), 0 ) - data["curvature"] * data["num_reported_accidents"]) + np.where((data["orig_mean_speed_limit"] + data["orig_mean_speed_limit"] / 2.0 / 2.0) <= data["orig_mean_lighting"], ((data["orig_mean_speed_limit"] + data["orig_mean_lighting"])) / ((0.3028397560)), 0 ))) 
    d["gp_3"] = (np.where(((0.0149999997) + data["orig_mean_lighting"])/2.0  <= (data["orig_mean_speed_limit"] + ((0.3026306927) + data["orig_mean_speed_limit"] / 2.0)/2.0 )/2.0 , np.sin((65.0000000000) * (np.where(data["orig_mean_speed_limit"] <= data["orig_mean_lighting"], (data["orig_mean_speed_limit"] + ((0.3026306927) + (0.0149999997))/2.0 )/2.0 , 0 ) + data["orig_mean_lighting"])), 0 )) 
    d["gp_4"] = ((np.where(data["orig_mean_num_reported_accidents"] > data["orig_mean_weather"], np.cos(np.where((0.5349999666) > data["curvature"], data["num_reported_accidents"], 0 )), 0 ) - np.cos(((0.5049999952) - np.where(data["orig_mean_weather"] > np.cos(data["curvature"]) * data["orig_mean_num_reported_accidents"], (data["curvature"] - np.cos(data["curvature"]) * 2.0), 0 ))))) 
    d["gp_5"] = ((np.where(np.cos(data["curvature"] * 2.0) <= (0.4149648547), np.where(data["orig_mean_weather"] <= (0.4149648547), data["curvature"], 0 ) * 2.0 * 2.0, 0 ) + (data["curvature"] * -1. * 2.0 * 2.0 * data["curvature"] + (data["curvature"] * 2.0 + np.cos(data["num_reported_accidents"]))/2.0 ))/2.0 ) 
    d["gp_6"] = (np.cos(((data["speed_limit"] - data["orig_min_lighting"]) + np.sin(((data["speed_limit"] - (0.0799999982)) + np.sin(((np.cos(data["speed_limit"]) / 2.0 + ((30.0000000000) - (0.3662542701)))/2.0  * (data["orig_min_lighting"] - (0.4025629163))) / ((0.4350000024))))) * 2.0))) 
    d["gp_7"] = (np.sin(np.sin((np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin((0.0149999997))))))))) / (data["orig_mean_num_reported_accidents"] * (np.where(data["curvature"] > (1.0000000000 - data["curvature"]), (data["orig_mean_num_reported_accidents"] - data["curvature"]), 0 ) + data["orig_mean_curvature"]) * -1.) * -1.))) 
    d["gp_8"] = (data["num_reported_accidents"] * data["num_reported_accidents"] * np.where((data["orig_mean_weather"] - np.cos((data["orig_mean_curvature"] + data["num_reported_accidents"])/2.0 )) > np.where((data["orig_mean_lighting"]) / (data["orig_mean_curvature"] * 2.0) <= data["orig_mean_weather"], data["num_reported_accidents"], 0 ), (data["orig_mean_num_reported_accidents"]) / (data["orig_mean_curvature"] * 2.0) * (0.2183499932) * -1., 0 )) 
    d["gp_9"] = ((np.where(data["orig_mean_lighting"] > (0.4149648547), np.where(data["orig_mean_lighting"] > data["orig_mean_num_reported_accidents"], data["orig_mean_num_reported_accidents"], 0 ), 0 ) + data["orig_mean_num_reported_accidents"]) * np.cos(((data["orig_mean_curvature"]) / ((0.4149648547)) + np.where(np.where(data["orig_mean_num_reported_accidents"] <= np.where(data["orig_mean_lighting"] <= (0.4149648547), (0.4149648547), 0 ), data["speed_limit"], 0 ) <= (65.0000000000), data["speed_limit"], 0 )))) 
    d["gp_10"] = (np.where(np.cos(data["weather"]) > data["num_reported_accidents"], np.where(data["orig_mean_curvature"] > ((0.1200000048) + np.where(data["orig_mean_speed_limit"] > data["weather"], data["orig_mean_lighting"], 0 ))/2.0 , np.where(np.sin(np.sin(np.where(data["orig_mean_lighting"] > np.where(data["weather"] <= data["orig_mean_lighting"], (0.4593729973), 0 ), data["orig_mean_lighting"], 0 ))) > data["orig_mean_curvature"], np.cos(data["orig_mean_weather"]), 0 ), 0 ), 0 )) 
    d["gp_11"] = ((data["orig_min_curvature"] * -1. + (np.where((0.3662542701) > data["orig_mean_speed_limit"], data["orig_min_curvature"], 0 ) * -1. + np.where(data["num_reported_accidents"] / 2.0 > np.cos(np.where((0.3662542701) > data["orig_mean_weather"], (2.5000000000), 0 )), data["orig_min_curvature"], 0 ) * np.where(data["orig_mean_speed_limit"] <= (0.3587038517), data["num_reported_accidents"], 0 ) * 2.0 * 2.0))) 
    d["gp_12"] = (np.where(data["curvature"] <= (0.5049999952), np.where(data["num_reported_accidents"] > (2.5000000000), data["num_reported_accidents"] * -1. * 2.0, 0 ), 0 )) 
    d["gp_13"] = (np.where(np.where((data["orig_mean_curvature"] + np.where((data["orig_mean_curvature"] + (0.0450000018)) > (0.4315392673), (0.0450000018), 0 )) > data["orig_mean_weather"], (0.4948371053), 0 ) <= data["orig_mean_curvature"], np.where((0.4315392673) > data["orig_mean_speed_limit"], (data["orig_mean_curvature"] - np.where((data["orig_mean_weather"] + (0.0450000018)) > (0.4315392673), data["curvature"], 0 ) * 2.0), 0 ), 0 )) 
    d["gp_14"] = (((0.0949999988) + np.where((data["curvature"]) / (np.cos((data["curvature"]) / (np.cos((data["curvature"] + ((0.3750000000) + (0.3993960619)))/2.0 )))) <= (0.3993960619), (((0.3750000000) - data["orig_mean_speed_limit"]) - (data["curvature"]) / (np.cos(data["curvature"]))), 0 ))) 
    d["gp_15"] = (np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(np.where(data["lighting"] <= np.where((data["orig_mean_weather"] - (0.3662542701)) <= np.where(data["lighting"] <= data["orig_mean_speed_limit"], data["orig_mean_weather"], 0 ), data["orig_mean_weather"], 0 ), (data["orig_mean_weather"] - (0.3662542701)), 0 ) * 2.0 * 2.0 * 2.0))))))))) 
    d["gp_16"] = ((np.where((data["orig_min_speed_limit"]) / ((data["orig_mean_lighting"] - data["orig_mean_curvature"] * 2.0)) > data["orig_mean_curvature"], data["orig_mean_lighting"] * -1., 0 )) / (((0.0799999982) * 2.0 + np.where(((data["orig_mean_lighting"] - data["orig_mean_curvature"] * 2.0)) / ((0.0799999982)) <= data["orig_min_speed_limit"], data["orig_mean_weather"], 0 )))) 
    d["gp_17"] = ((np.where((0.7450000048) > np.where((0.4160095453) > data["orig_mean_lighting"], data["curvature"], 0 ), data["orig_mean_weather"], 0 ) * -1. + (data["orig_mean_lighting"] - np.where(data["orig_mean_weather"] > np.where((0.4160095453) > data["orig_mean_weather"], data["curvature"], 0 ), np.where(np.where(data["orig_mean_lighting"] > data["orig_mean_speed_limit"], data["curvature"], 0 ) > data["orig_mean_speed_limit"], data["curvature"], 0 ), 0 )))/2.0 ) 
    d["gp_18"] = (((data["orig_min_lighting"] + np.where(data["orig_min_lighting"] > np.where(data["curvature"] > data["orig_mean_speed_limit"] * data["orig_mean_weather"] * data["orig_mean_speed_limit"] * 2.0, data["orig_mean_speed_limit"], 0 ), -1.0000000000, 0 )) + np.where(np.where((0.0449999981) > data["curvature"], data["curvature"] * 2.0 * 2.0, 0 ) > (0.0449999981), data["orig_mean_weather"], 0 ) * 2.0)) 
    d["gp_19"] = (np.where((np.sin(data["speed_limit"]) + np.sin(data["speed_limit"] * (0.3993960619)))/2.0  <= np.where(data["curvature"] <= np.sin(np.sin((0.3818522692))), (0.3993960619) * -1., 0 ), (((0.3993960619) * -1. + ((0.3993960619) * -1. + data["orig_mean_lighting"])) + data["orig_mean_lighting"]), 0 )) 
    d["gp_20"] = ((data["curvature"] - data["orig_mean_lighting"]) * (np.where((0.3609148860) > data["orig_mean_lighting"], np.where(np.where(data["curvature"] / 2.0 > (0.3609148860), np.cos((0.3609148860)), 0 ) > data["curvature"], np.cos(data["orig_mean_speed_limit"]), 0 ), 0 ) - (data["orig_mean_lighting"] - (data["weather"] - data["curvature"] / 2.0)))) 
    d["gp_21"] = (np.where(data["orig_mean_weather"] <= ((data["num_reported_accidents"] - ((data["orig_mean_weather"]) / (data["orig_mean_curvature"])) / (data["orig_mean_curvature"]) * (0.0149999997)) - (data["orig_mean_weather"]) / (data["orig_mean_curvature"])), np.where(data["num_reported_accidents"] * (0.0149999997) > data["orig_min_lighting"], (data["orig_mean_weather"]) / ( 4.0 ), 0 ), 0 ) * 2.0) 
    d["gp_22"] = (np.where(data["weather"] > (np.where(((0.4025629163) - data["orig_mean_lighting"] / 2.0 / 2.0 / 2.0) <= data["weather"], data["weather"], 0 ) + ((0.4025629163) - data["orig_mean_lighting"] / 2.0 / 2.0)), (data["orig_mean_lighting"] / 2.0 / 2.0 - data["orig_min_curvature"] * 2.0), 0 )) 
    d["gp_23"] = (data["orig_mean_lighting"] * (np.where(data["orig_mean_speed_limit"] <= np.where(data["curvature"] * data["orig_mean_speed_limit"] <= data["orig_mean_lighting"] / 2.0, np.where(data["orig_mean_weather"] / 2.0 <= np.where(data["orig_mean_num_reported_accidents"] <= (0.4025629163), data["orig_mean_num_reported_accidents"], 0 ) * data["orig_mean_lighting"], data["orig_mean_num_reported_accidents"], 0 ), 0 ), np.cos(data["orig_min_curvature"]), 0 ) - data["orig_mean_lighting"] / 2.0)) 
    d["gp_24"] = (np.where(np.where(data["orig_mean_weather"] <= np.where(data["orig_mean_curvature"] <= ((0.4315392673) + (0.3714956343)) / 2.0, (0.4160095453), 0 ), ((0.3662542701) + np.where((0.3662542701) <= data["orig_mean_weather"], (0.3662542701), 0 )) / 2.0, 0 ) <= np.where(data["curvature"] <= data["orig_mean_lighting"], data["curvature"], 0 ), (data["orig_mean_weather"] - (0.3662542701)), 0 )) 
    d["gp_25"] = (np.where(data["curvature"] <= np.where((0.4160095453) > data["orig_mean_weather"], np.where(data["orig_mean_lighting"] <= data["orig_mean_weather"], (0.5049999952), 0 ), 0 ), np.where(data["orig_mean_lighting"] <= (0.4160095453), np.where(data["weather"] <= (0.6399999857), (data["curvature"] + np.cos(data["orig_mean_lighting"])), 0 ) * 2.0 * 2.0, 0 ), 0 ) * data["orig_min_speed_limit"] * 2.0 * -1.) 
    d["gp_26"] = (((0.0799999982)) / (((0.3662542701) * -1. + data["curvature"])/2.0 ) * (data["orig_min_lighting"] + (((0.3662542701) * -1. + ((((0.3662542701) * -1. + data["orig_mean_weather"])/2.0  + data["orig_min_lighting"])/2.0  + (0.3662542701) * -1.)/2.0 )/2.0  + data["orig_mean_weather"])/2.0 )/2.0  * data["curvature"]) 
    d["gp_27"] = (data["num_reported_accidents"] * np.where(data["num_reported_accidents"] * np.where(np.where(np.sin((data["orig_mean_curvature"] - ((0.1200000048) - (0.2183499932)))) > (0.5799999833), data["orig_mean_weather"], 0 ) > data["orig_mean_lighting"], (0.1200000048), 0 ) > (0.2183499932), (((0.2183499932) - np.sin(data["num_reported_accidents"])) + data["num_reported_accidents"])/2.0 , 0 ) / 2.0) 
    d["gp_28"] = (np.cos(data["curvature"]) * (np.cos(np.cos((data["speed_limit"]) / (np.cos(np.cos(data["speed_limit"]))) * data["orig_min_curvature"]))) / (data["speed_limit"] * np.cos(((data["speed_limit"]) / (np.cos(data["orig_min_curvature"]))) / (((0.4025629163) + data["orig_min_curvature"])/2.0 ))) * -1.) 
    d["gp_29"] = (np.sin(np.sin(np.where(data["orig_mean_curvature"] > (data["orig_mean_num_reported_accidents"] - data["orig_min_lighting"]), data["orig_min_speed_limit"] * 2.0 * (data["orig_min_lighting"] * 2.0 * 2.0 * 2.0 * -1. + (data["orig_min_lighting"]) / ((data["orig_mean_curvature"] - np.sin((0.3714956343)))))/2.0 , 0 ))) * -1. * 2.0 * 2.0) 
    d["gp_30"] = (((0.0149999997) + np.where(np.sin(data["orig_mean_weather"] * data["speed_limit"]) <= data["curvature"], np.where(np.where(data["curvature"] <= np.sin(data["weather"] * -1. * data["speed_limit"]), (0.0149999997), 0 ) <= np.sin(data["orig_min_lighting"] * data["speed_limit"] * -1.), data["orig_min_curvature"] * -1., 0 ), 0 ))) 
    d["gp_31"] = (np.where((data["orig_mean_lighting"] - data["orig_mean_weather"]) <= data["curvature"], (data["orig_mean_lighting"] + (data["curvature"] - np.where((0.3750000000) > data["orig_mean_lighting"], np.where((0.3750000000) > data["weather"], data["num_reported_accidents"], 0 ), 0 )))/2.0  * np.where(data["curvature"] <= (0.5049999952), (data["orig_mean_lighting"]) / (-1.0000000000), 0 ), 0 )) 
    d["gp_32"] = (np.where((data["curvature"]) / ((0.4025629163)) > data["orig_mean_num_reported_accidents"], np.where(((0.0799999982) + ((0.0799999982) + np.where((0.4025629163) <= data["orig_mean_num_reported_accidents"], data["curvature"], 0 ))/2.0 )/2.0  <= (data["orig_mean_weather"] - data["curvature"]) / 2.0, (data["curvature"]) / (data["orig_mean_weather"]), 0 ), 0 ) * 2.0 * data["curvature"]) 
    d["gp_33"] = (np.where((0.3587038517) > (data["curvature"] + np.where(data["curvature"] <= data["orig_mean_curvature"], np.where(np.where(data["orig_mean_curvature"] <= (0.5049999952), data["curvature"], 0 ) <= data["curvature"], data["curvature"], 0 ), 0 ) * (0.1200000048)), (np.sin(np.sin((0.5049999952))) - data["orig_mean_curvature"] * 2.0), 0 )) 
    d["gp_34"] = (np.where(data["orig_mean_lighting"] > (0.4160095453), np.where(data["orig_mean_curvature"] > np.sin(data["num_reported_accidents"]) * data["num_reported_accidents"], np.where(data["orig_mean_curvature"] > (0.4160095453), (np.where((0.4160095453) > data["orig_mean_speed_limit"], np.sin(data["num_reported_accidents"]) * data["num_reported_accidents"], 0 ) + np.sin(data["num_reported_accidents"])), 0 ), 0 ), 0 ) * -1.) 
    d["gp_35"] = (np.where((np.where(np.cos(np.where(data["curvature"] <= np.cos(np.where(data["orig_mean_lighting"] <= (0.4160095453), (0.2183499932), 0 )), (0.4160095453), 0 )) > data["curvature"], data["orig_mean_lighting"] * data["curvature"], 0 )) / (np.cos((data["curvature"] / 2.0) / ((0.0149999997)))) > data["orig_mean_lighting"], data["orig_mean_lighting"], 0 )) 
    d["gp_36"] = (((0.4025629163) + (0.3198369741))/2.0  * (np.sin((((-1.0000000000) / (((0.3198369741) - (data["curvature"] + (0.3381867111))/2.0 ) / 2.0 * (0.4025629163)) - (data["curvature"] - (0.4025629163))) - data["curvature"])) + ((0.3198369741) - data["curvature"]))/2.0 ) 
    d["gp_37"] = (data["orig_min_lighting"] * np.cos(((data["speed_limit"] + data["orig_min_curvature"] * np.cos((data["speed_limit"]) / (np.cos(((data["speed_limit"] + data["orig_min_lighting"] * data["orig_mean_curvature"])/2.0 ) / ((0.3198369741))))) * 2.0 * 2.0)/2.0 ) / ((0.3198369741))) * 2.0 * 2.0 * 2.0) 
    d["gp_38"] = ((data["orig_min_curvature"] + (0.0249999985)) * (np.where(data["orig_min_curvature"] <= np.where(data["weather"] <= (0.3609148860), data["weather"], 0 ), data["num_reported_accidents"], 0 ) + ((0.3609148860)) / (data["weather"])) * np.cos((data["num_reported_accidents"]) / (data["weather"] * (data["weather"]) / (data["orig_mean_curvature"])))) 
    d["gp_39"] = (np.where(np.where(data["curvature"] > data["orig_mean_num_reported_accidents"], data["orig_mean_num_reported_accidents"], 0 ) > (0.4079468548), data["orig_mean_lighting"], 0 ) * (np.sin((((52.5000000000) + data["speed_limit"])/2.0  + np.where(data["speed_limit"] > (52.5000000000), np.where(np.sin((52.5000000000)) > data["curvature"], np.sin((52.5000000000)), 0 ), 0 ))) + (0.3920692801))) 
    d["gp_40"] = ((np.where(data["holiday"] * (0.4165354371) > data["orig_mean_weather"], data["orig_min_curvature"], 0 ) + np.where((0.3609148860) > data["orig_min_curvature"], np.where(data["orig_mean_weather"] > (0.3609148860), data["orig_min_curvature"] * (data["orig_min_curvature"]) / (np.sin((np.where(data["holiday"] > (0.4165354371), (0.3609148860), 0 ) + (0.0850000009) * -1.)/2.0 )), 0 ), 0 ))) 
    d["gp_41"] = (data["orig_mean_lighting"] * 2.0 * data["orig_mean_lighting"] * 2.0 * np.where(data["weather"] <= np.where(data["orig_mean_speed_limit"] <= data["orig_mean_lighting"] * 2.0 * np.where( 0.3772679269  <= data["weather"], np.where(data["orig_mean_speed_limit"] <=  0.3772679269 , data["orig_mean_curvature"], 0 ), 0 ), data["orig_mean_curvature"], 0 ), (0.4025629163), 0 )) 
    d["gp_42"] = (np.where(np.sin((data["orig_mean_lighting"] - (data["orig_mean_weather"] - (data["orig_mean_lighting"] - data["orig_mean_weather"] / 2.0)))) > (data["orig_mean_weather"] - (data["orig_mean_lighting"] - data["orig_mean_lighting"] / 2.0)), (data["orig_mean_weather"] - np.sin((data["orig_mean_lighting"] - data["orig_mean_weather"] / 2.0))) * -1., 0 )) 
    d["gp_43"] = (np.cos(np.where(data["orig_mean_speed_limit"] <= data["curvature"], np.where( 0.1228411496  <= data["weather"] * data["orig_mean_curvature"], data["orig_mean_lighting"], 0 ), 0 )) * np.where( 0.1228411496  <= data["curvature"] * np.where(data["orig_mean_curvature"] <= (0.3587038517), np.where( 0.1228411496  <= data["orig_mean_curvature"] * data["curvature"], data["orig_mean_lighting"], 0 ), 0 ), (0.3198369741), 0 )) 
    d["gp_44"] = (np.where(data["orig_mean_num_reported_accidents"] <= data["orig_mean_weather"] * np.where(data["orig_mean_num_reported_accidents"] > data["weather"], data["curvature"], 0 ), ((np.where(np.where(1.0000000000 > data["curvature"], data["orig_mean_num_reported_accidents"], 0 ) <= data["orig_mean_speed_limit"], 1.0000000000 * 2.0, 0 ) + np.where(data["orig_mean_num_reported_accidents"] > data["orig_mean_speed_limit"], data["curvature"], 0 ) * -1.) + data["curvature"] * -1.), 0 )) 
    d["gp_45"] = (np.where(np.sin( 8.0  * data["curvature"]) > np.where(np.cos((52.5000000000)) <= np.cos(data["num_reported_accidents"]) * 2.0,  8.0  * data["curvature"] * data["curvature"], 0 ), (np.sin(data["num_reported_accidents"]) *  8.0  * 2.0) / ((52.5000000000)) * data["curvature"], 0 )) 
    d["gp_46"] = (np.where(data["orig_mean_weather"] > np.where((data["orig_mean_weather"]) / (data["orig_mean_speed_limit"]) <= np.cos(data["weather"]), data["weather"], 0 ), (data["orig_mean_weather"]) / (np.cos(data["orig_mean_weather"] * 2.0)) * 2.0 * np.where((data["orig_mean_curvature"]) / (np.cos(data["weather"]) / 2.0) <= data["orig_mean_speed_limit"], data["orig_mean_weather"] * 2.0, 0 ), 0 )) 
    d["gp_47"] = (np.where((0.3587038517) > data["orig_mean_lighting"], np.where((0.3609148860) > data["orig_mean_curvature"], np.where(data["weather"] > data["orig_mean_lighting"], np.where(data["orig_mean_curvature"] > (0.3587038517), ((((data["weather"]) / (data["orig_mean_speed_limit"]) - data["orig_mean_speed_limit"]) + data["orig_mean_curvature"]) + (0.3587038517)) * -1., 0 ), 0 ), 0 ), 0 )) 
    d["gp_48"] = (data["orig_min_lighting"] * ((1.0000000000 + ((data["orig_min_curvature"]) / (((0.4025629163) - (data["orig_mean_curvature"] + data["orig_min_curvature"]))) + data["orig_mean_curvature"])) + (data["orig_min_lighting"]) / ((((0.4176416695) + 1.0000000000)/2.0  - (data["orig_mean_curvature"] + data["orig_min_curvature"])))) * -1.) 
    d["gp_49"] = ((np.where((0.5049999952) <= np.where((0.3381867111) / 2.0 <= data["curvature"], data["curvature"], 0 ), data["orig_mean_curvature"], 0 ) * (np.where(data["orig_min_curvature"] > (0.3381867111) / 2.0 / 2.0, data["lighting"], 0 ) * -1. + data["orig_mean_curvature"]) + np.where(data["orig_min_curvature"] > data["weather"] / 2.0, (0.5049999952), 0 ) * -1.)/2.0 ) 
    d["gp_50"] = (np.where(data["curvature"] <= data["orig_min_lighting"], np.where(np.where(data["curvature"] <= data["orig_min_curvature"], data["orig_min_lighting"], 0 ) <= np.where((0.3818522692) <= data["orig_mean_num_reported_accidents"], (0.3818522692), 0 ), np.where(np.where((0.3818522692) > np.where(data["weather"] > (0.3818522692), data["orig_mean_num_reported_accidents"], 0 ), data["weather"], 0 ) > data["orig_mean_speed_limit"], data["orig_mean_num_reported_accidents"], 0 ), 0 ), 0 )) 
    d["gp_51"] = (np.where((data["orig_mean_lighting"]) / ((0.3198369741)) <= data["curvature"], ((data["orig_mean_num_reported_accidents"]) / (data["orig_mean_speed_limit"])) / ((np.where(data["orig_mean_num_reported_accidents"] <= data["orig_mean_weather"], ((data["orig_mean_num_reported_accidents"]) / (data["orig_mean_speed_limit"])) / ((data["orig_mean_speed_limit"]) / (data["orig_mean_speed_limit"])), 0 ) + (0.3198369741))/2.0 ), 0 ) * -1.) 
    d["gp_52"] = (np.where(data["orig_mean_curvature"] > np.sin(data["curvature"]) * 2.0, (np.where(data["curvature"] > (0.1200000048), np.cos(data["orig_min_speed_limit"] * 2.0 * 2.0 * 2.0 * 2.0), 0 ) + np.where( 0.0777993351  > data["curvature"], data["orig_min_speed_limit"], 0 ) * 2.0 * 2.0) * -1., 0 )) 
    d["gp_53"] = ((np.where(data["curvature"] > (0.7450000048), np.where(data["curvature"] > (data["curvature"] - data["orig_mean_lighting"]), np.where(data["orig_mean_speed_limit"] <= data["orig_mean_lighting"], (np.sin(np.cos(data["orig_mean_lighting"])) - np.where(data["orig_mean_lighting"] <=  0.4791802168 , 1.0000000000, 0 ) * -1.), 0 ), 0 ), 0 ) - data["curvature"]) * data["orig_min_curvature"]) 
    d["gp_54"] = (np.where(data["orig_mean_curvature"] <= np.where(data["curvature"] <= data["orig_mean_num_reported_accidents"], (data["orig_mean_lighting"] + data["orig_min_speed_limit"] * 2.0 * 2.0)/2.0 , 0 ), np.where(data["orig_mean_curvature"] <= data["weather"], (np.where(data["orig_mean_lighting"] <= data["curvature"] * 2.0, data["curvature"] * 2.0 * 2.0 * 2.0, 0 ) + data["curvature"]), 0 ), 0 )) 
    d["gp_55"] = (data["orig_mean_curvature"] * data["orig_mean_curvature"] * np.sin((np.sin(np.sin((0.3381867111)))) / (np.sin((data["orig_mean_weather"] + (np.sin((0.3587038517))) / (np.sin(((0.3381867111) + ((0.5049999952)) / ((data["orig_mean_curvature"] + (0.4350000024) * -1.))))))))) * -1.) 
    d["gp_56"] = (np.where(np.where(data["orig_min_curvature"] > data["orig_mean_speed_limit"], data["orig_min_curvature"], 0 ) > data["orig_mean_weather"], ( 9.0  + (np.cos((0.3381867111)) - (data["orig_mean_speed_limit"] + np.cos((data["orig_min_curvature"]) / (np.sin((0.0799999982)))))/2.0 ))/2.0 , 0 )) 
    d["gp_57"] = ((np.where((0.4350000024) <= data["orig_mean_num_reported_accidents"], data["curvature"], 0 )) / (((0.4160095453) - np.where((((0.5049999952) - (0.4160095453)) - (data["curvature"] - (0.4350000024))) <= data["orig_mean_lighting"], (0.4350000024), 0 ))) * np.where(data["curvature"] <= (0.5049999952), data["curvature"], 0 )) 
    d["gp_58"] = ((data["orig_mean_num_reported_accidents"]) / ((data["orig_mean_curvature"]) / ((data["orig_mean_num_reported_accidents"]) / ((data["orig_mean_curvature"]) / ((data["orig_mean_num_reported_accidents"]) / ((data["orig_mean_curvature"]) / ((data["orig_mean_num_reported_accidents"]) / ((data["orig_mean_curvature"]) / ((data["orig_mean_num_reported_accidents"]) / ((data["orig_mean_curvature"]) / ((data["orig_mean_num_reported_accidents"] + (0.3818522692) * -1.)/2.0 ))))))))))) 
    d["gp_59"] = (np.where(data["curvature"] <= np.where(data["orig_mean_speed_limit"] <= data["curvature"] * 2.0,  0.2973037362 , 0 ), np.where(np.where(data["curvature"] <= (0.2850000262), data["orig_mean_speed_limit"], 0 ) <= np.where(data["curvature"] <= np.where(data["curvature"] <= (0.2850000262), data["orig_mean_speed_limit"], 0 ), np.where(data["orig_mean_curvature"] <= data["orig_mean_lighting"], data["orig_mean_lighting"], 0 ), 0 ), data["orig_mean_speed_limit"], 0 ), 0 ) / 2.0) 
    d["gp_60"] = (np.where(np.sin(data["orig_mean_weather"]) > (data["orig_mean_curvature"] + (np.sin((0.3381867111)) + np.where(data["orig_mean_speed_limit"] <= data["orig_mean_weather"], np.where(data["weather"] <= data["orig_mean_weather"], data["weather"], 0 ), 0 )))/2.0 , np.where(data["orig_mean_lighting"] > (0.3381867111), (np.where(data["curvature"] <= data["orig_mean_weather"], data["curvature"], 0 ) * 2.0 - (0.3609148860)), 0 ), 0 )) 
    d["gp_61"] = (data["orig_min_speed_limit"] * np.where(data["curvature"] / 2.0 <= np.sin(data["num_reported_accidents"] * np.where(np.sin(data["curvature"]) <= np.where(data["curvature"] / 2.0 <= np.where((0.4699999988) <= data["curvature"], (0.4699999988), 0 ), data["curvature"], 0 ), data["curvature"], 0 )), data["curvature"], 0 ) * 2.0 * 2.0) 
    d["gp_62"] = (np.where(data["orig_mean_speed_limit"] > np.where((0.3381867111) > data["weather"], data["weather"], 0 ), data["num_reported_accidents"] * data["num_reported_accidents"] * data["num_reported_accidents"] * (np.where(data["num_reported_accidents"] > 1.0000000000, np.where(data["orig_mean_curvature"] > (0.3381867111), data["orig_min_lighting"], 0 ), 0 ) - np.where(data["orig_mean_speed_limit"] <= data["num_reported_accidents"] / 2.0, data["orig_min_lighting"], 0 )), 0 )) 
    d["gp_63"] = (np.where((np.sin(((52.5000000000) + data["num_reported_accidents"])) * -1.) / (((0.4160095453) - data["orig_mean_lighting"])) > np.where(data["weather"] > data["orig_mean_lighting"], (((0.4160095453) +  8.0 ) + data["orig_mean_lighting"]), 0 ), ((0.4160095453)) / (( 4.0  - (data["num_reported_accidents"] +  8.0 ))), 0 )) 
    d["gp_64"] = ((np.where(data["orig_mean_lighting"] / 2.0 <= data["orig_min_curvature"], data["orig_mean_curvature"], 0 ) + (np.where((data["curvature"]) / ((data["orig_min_speed_limit"] + (data["orig_min_speed_limit"] + data["orig_mean_curvature"]))) <= data["orig_mean_lighting"], data["orig_mean_curvature"], 0 )) / ((data["orig_min_speed_limit"] + (data["orig_min_speed_limit"] + data["orig_mean_lighting"] / 2.0)/2.0 )/2.0 ))/2.0  *  0.1051528677 ) 
    d["gp_65"] = (np.where(np.where((np.sin(data["lighting"]) * -1. -  0.7123813033 ) > -1.0000000000, data["curvature"], 0 ) > np.where(data["curvature"] * -1. <= np.where(data["curvature"] > data["curvature"], data["curvature"], 0 ), np.sin( 0.7123813033 ), 0 ),  3.0 , 0 )) 
    d["gp_66"] = (np.where((data["curvature"]) / ((0.3818522692)) <= data["weather"], ((data["weather"] * -1. - data["curvature"]) - np.where(data["curvature"] > np.where(np.where((0.3587038517) <= data["weather"], (data["curvature"] - (0.3818522692)), 0 ) <= data["weather"], ((0.3587038517) - data["weather"]), 0 ), data["weather"] * -1., 0 )), 0 )) 
    d["gp_67"] = (np.where((0.4160095453) <= np.cos((np.cos(data["curvature"]) * data["speed_limit"] + np.cos(np.cos(data["curvature"])))/2.0 ), ((np.cos(np.cos(0.0000000000))) / ((0.4160095453))) / (((np.cos(data["speed_limit"]) * data["speed_limit"] + (0.3993960619))/2.0  + data["speed_limit"])/2.0 ), 0 )) 
    d["gp_68"] = (np.where((0.0000000000) > (np.sin(data["num_reported_accidents"]) + np.where(data["orig_mean_curvature"] > ((data["orig_mean_weather"]) / (np.sin(np.sin(data["orig_mean_num_reported_accidents"]))) - data["orig_mean_curvature"]), (data["orig_mean_speed_limit"] - ((data["orig_mean_weather"]) / (data["orig_mean_num_reported_accidents"]) - (0.4025629163))), 0 ))/2.0 , (0.3458362222), 0 )) 
    d["gp_69"] = (np.where((np.sin(data["orig_mean_curvature"]) - data["orig_min_curvature"]) <= data["orig_mean_num_reported_accidents"], np.where((0.3993960619) <= data["orig_mean_curvature"], (data["orig_mean_curvature"] + (-1.0000000000 + ((np.sin(data["orig_mean_curvature"]) - np.where(data["orig_mean_curvature"] <= (0.5027128458), data["orig_min_curvature"], 0 )) - data["orig_min_curvature"]) * data["orig_mean_num_reported_accidents"] * -1.)/2.0 ), 0 ), 0 )) 
    d["gp_70"] = (np.where((0.4699999988) <= data["curvature"], np.where(data["curvature"] <= np.sin(np.sin(((0.4699999988)) / (np.sin(data["speed_limit"])))), np.where((0.4699999988) <= data["speed_limit"], np.where(data["curvature"] <= np.sin(np.sin((data["speed_limit"]) / (np.sin(data["speed_limit"])))), (0.4699999988), 0 ), 0 ), 0 ), 0 )) 
    d["gp_71"] = (np.where((0.3745930791) > data["weather"], data["num_reported_accidents"] * data["num_reported_accidents"] * np.where(1.0000000000 > (data["orig_mean_lighting"]) / ((0.3169768453)), (np.where((0.3745930791) <= data["orig_mean_weather"], ((data["orig_mean_lighting"]) / ( 0.3826292455 ) + (0.3745930791))/2.0 , 0 ) + np.sin(data["num_reported_accidents"]))/2.0 , 0 ) * data["orig_min_curvature"], 0 )) 
    d["gp_72"] = ((0.3750000000) * (data["num_reported_accidents"] / 2.0 + (data["orig_min_curvature"] + np.where((0.0799999982) > data["orig_min_curvature"], data["num_reported_accidents"], 0 ))) * (np.where(np.where(data["orig_mean_lighting"] > ((0.3750000000) + data["orig_min_curvature"]), data["orig_min_curvature"], 0 ) > (0.0799999982) / 2.0, (0.3750000000), 0 ) - data["orig_min_curvature"])) 
    d["gp_73"] = (np.where(data["orig_mean_lighting"] > data["curvature"], ((np.where((0.4699999988) <= data["curvature"], np.where((0.3818522692) <= data["curvature"], np.where(data["orig_mean_curvature"] > (0.3818522692), (np.where(data["orig_mean_curvature"] <= data["orig_mean_weather"], data["curvature"], 0 )) / ((0.3818522692) * (0.4699999988)), 0 ), 0 ), 0 )) / (data["orig_mean_curvature"])) / ((0.3587038517)), 0 ) * -1.) 
    d["gp_74"] = (np.where(np.where(np.where(np.where(0.0000000000 > np.sin(data["num_reported_accidents"]), data["orig_mean_weather"], 0 ) >  0.3170784414 , np.cos(data["speed_limit"]), 0 ) > np.where(data["speed_limit"] > (52.5000000000), np.sin(data["num_reported_accidents"]), 0 ), np.cos(data["speed_limit"]), 0 ) > np.where(data["speed_limit"] > (52.5000000000), np.sin(data["num_reported_accidents"]), 0 ), np.cos(data["speed_limit"]), 0 )) 
    d["gp_75"] = (np.where(np.cos(((data["curvature"]) / (data["orig_mean_weather"]) + (data["curvature"]) / ((0.0149999997)))/2.0 ) <= np.where(data["curvature"] <= data["orig_mean_weather"], (np.where(data["curvature"] <= data["orig_mean_speed_limit"] / 2.0, data["orig_mean_weather"], 0 ) + data["curvature"] / 2.0)/2.0 , 0 ), (0.0149999997), 0 ) * -1.) 
    d["gp_76"] = ((np.sin(np.sin((data["orig_mean_lighting"] * -1.) / ((0.0149999997) * (data["orig_min_curvature"] - data["orig_mean_weather"]) / 2.0) * np.where(data["orig_mean_weather"] <=  0.3270739019 , data["orig_min_curvature"], 0 )) * 2.0 * 2.0 * 2.0) * 2.0 * 2.0 * 2.0) / ((52.5000000000))) 
    d["gp_77"] = (np.where(data["orig_mean_speed_limit"] > data["curvature"], ((data["orig_mean_lighting"] * -1. * np.where((0.4160095453) <= data["curvature"], data["orig_mean_lighting"], 0 ) - np.where(np.sin(data["orig_mean_speed_limit"]) <= data["curvature"], (0.5799999833), 0 )) - np.where(data["curvature"] <= np.where((0.4049999714) <= data["curvature"], data["orig_mean_lighting"], 0 ), data["orig_mean_lighting"], 0 )), 0 )) 
    d["gp_78"] = (np.where(data["orig_min_curvature"] <= data["orig_mean_weather"] * data["orig_mean_weather"], np.where(data["orig_mean_curvature"] <= (0.3920692801), np.where(data["orig_mean_weather"] <= data["num_reported_accidents"] * data["orig_mean_curvature"] / 2.0, np.where(data["num_reported_accidents"] * data["orig_mean_weather"] / 2.0 / 2.0 <= (0.3920692801),  0.9170994759  * -1., 0 ), 0 ), 0 ) / 2.0, 0 ) * -1.) 
    d["gp_79"] = (np.where((data["orig_mean_weather"] - data["orig_mean_speed_limit"]) <= (data["orig_mean_lighting"] - np.where(np.where(data["num_reported_accidents"] > data["orig_mean_weather"], data["orig_mean_weather"], 0 ) > np.where(data["orig_mean_weather"] > data["orig_mean_lighting"], (0.4160095453), 0 ), data["orig_mean_weather"], 0 )), ((0.3662542701) - data["orig_mean_num_reported_accidents"]), 0 )) 
    d["gp_80"] = (np.where(data["orig_mean_curvature"] > (data["orig_mean_weather"] + (data["orig_mean_speed_limit"] + (data["orig_mean_weather"] + (data["orig_mean_lighting"] + data["orig_min_curvature"]))) * np.where((data["orig_mean_lighting"] + data["orig_min_curvature"]) > data["orig_mean_weather"], data["orig_mean_lighting"], 0 ) / 2.0), (0.0149999997), 0 ) * 2.0 * 2.0 * 2.0 * 2.0) 
    d["gp_81"] = ((((0.0149999997)) / (((0.3662542701) - (1.0000000000 + data["curvature"] * (((0.3662542701) -  4.0 ) - np.where((0.3662542701) > data["lighting"], (0.0149999997), 0 )) * 2.0) * 2.0)) - (np.where(data["curvature"] > (0.3662542701), 1.0000000000, 0 )) / (data["speed_limit"]))) 
    d["gp_82"] = ((data["orig_mean_lighting"] * 2.0 + np.cos(data["curvature"] * 2.0 * 2.0 * 2.0 * 2.0)) * (data["curvature"] * 2.0 + data["curvature"] * 2.0) * 2.0 * np.where(np.cos(data["orig_min_curvature"] * 2.0 * 2.0) <= data["curvature"], data["orig_min_speed_limit"], 0 )) 
    d["gp_83"] = ((np.where( 0.3662985265  * data["curvature"] / 2.0 <= (data["orig_mean_lighting"] + np.where(data["orig_mean_speed_limit"] <= data["curvature"], data["curvature"] / 2.0, 0 ))/2.0  * np.sin(data["orig_mean_weather"]), (0.5027128458), 0 ) -  0.3662990034 ) / 2.0 / 2.0) 
    d["gp_84"] = (np.where((data["orig_min_curvature"]) / (data["orig_mean_speed_limit"]) * data["orig_min_curvature"] <= (0.0149999997), (((52.5000000000) * data["orig_mean_speed_limit"] * data["orig_min_curvature"] * data["curvature"] * -1. - data["orig_mean_speed_limit"])) / ((data["curvature"] - ((52.5000000000) * data["orig_mean_curvature"] - data["speed_limit"]))), 0 )) 
    d["gp_85"] = (np.where(data["orig_mean_lighting"] <= (0.4025629163), np.where(data["orig_min_curvature"] <= np.where(data["orig_mean_curvature"] * np.cos(data["orig_mean_lighting"] * 2.0) <= data["orig_mean_speed_limit"], np.where((0.4025629163) > data["orig_mean_curvature"], data["orig_mean_lighting"], 0 ), 0 ), data["orig_min_curvature"], 0 ) * 2.0 * 2.0 * 2.0, 0 )) 
    d["gp_86"] = (np.where((np.where((0.3818522692) > ((data["orig_mean_weather"] - (0.5049999952)) + (data["curvature"] - np.where(data["orig_min_speed_limit"] > data["orig_mean_curvature"], data["orig_min_speed_limit"], 0 ))), (0.5049999952), 0 ) + data["orig_mean_curvature"] * (data["orig_mean_weather"] + data["curvature"] * 2.0))/2.0  > (0.5049999952), data["orig_min_speed_limit"], 0 )) 
    d["gp_87"] = (data["orig_min_speed_limit"] * np.where(data["orig_mean_public_road"] <= data["orig_mean_lighting"], np.where(np.where((0.3829739690) <= data["orig_mean_public_road"], (0.3662542701), 0 ) <= data["orig_mean_curvature"], np.where(data["orig_mean_curvature"] <= (0.3750000000), ((0.5049999952) + data["orig_mean_lighting"]) * 2.0 * 2.0, 0 ) * 2.0 * 2.0, 0 ), 0 ) * -1.) 
    d["gp_88"] = (np.where(np.where((0.3198369741) > np.where((0.3198369741) > data["orig_mean_weather"], data["num_reported_accidents"], 0 ), (0.3198369741), 0 ) <= np.where(data["orig_mean_weather"] > data["num_reported_accidents"], data["orig_mean_weather"], 0 ), np.where(np.where(data["curvature"] > np.where(data["orig_mean_lighting"] > (0.3198369741), (0.1800000072), 0 ), (0.3609148860), 0 ) <= (0.3198369741), data["orig_mean_lighting"], 0 ), 0 ) / 2.0) 
    d["gp_89"] = (np.where((0.3609148860) <= data["orig_mean_speed_limit"], data["orig_mean_lighting"], 0 ) * (np.where((0.3614033461) <= data["orig_mean_curvature"], np.where(data["orig_mean_curvature"] <= data["orig_mean_weather"], data["orig_mean_lighting"], 0 ) * -1., 0 ) + np.where((0.3614033461) <= np.where(np.where((0.3614033461) <= data["orig_mean_weather"], data["orig_mean_curvature"], 0 ) <= (0.3614033461), data["orig_mean_lighting"], 0 ), data["orig_mean_speed_limit"], 0 ))) 
    d["gp_90"] = ((np.where(np.where(data["orig_min_speed_limit"] > data["orig_min_lighting"], np.where((0.3381867111) > data["orig_min_speed_limit"], data["orig_min_lighting"] / 2.0, 0 ) / 2.0, 0 ) > data["curvature"], (0.3662542701), 0 )) / ((np.where(data["curvature"] > data["curvature"] / 2.0, (data["orig_min_lighting"] + (0.3381867111))/2.0 , 0 ) + (0.3381867111))/2.0 )) 
    d["gp_91"] = (np.where((data["orig_min_curvature"]) / ((data["orig_mean_weather"] + data["orig_min_curvature"])/2.0 ) <= np.where(data["orig_mean_weather"] <= (0.4025629163), np.cos(data["num_reported_accidents"]), 0 ), (data["orig_min_curvature"]) / ((np.cos((np.cos(data["orig_min_curvature"]) + np.cos(data["speed_limit"]))/2.0 ) + data["orig_min_curvature"])/2.0 ), 0 ) * np.cos(data["speed_limit"])) 
    d["gp_92"] = (np.where((0.3381867111) <= data["num_reported_accidents"] * data["num_reported_accidents"] * np.where(data["curvature"] <= (0.3818522692), np.where(np.where(data["orig_mean_lighting"] <= np.where((0.3818522692) <= np.where(data["curvature"] <= (0.3381867111), (0.3818522692), 0 ), (0.3381867111), 0 ), (0.3750000000), 0 ) <= data["orig_mean_weather"], data["orig_min_speed_limit"], 0 ), 0 ), data["num_reported_accidents"], 0 ) * -1.) 
    d["gp_93"] = (data["orig_min_speed_limit"] * 2.0 * 2.0 * (data["orig_mean_lighting"] * 2.0 + data["orig_mean_lighting"] * 2.0 * np.cos(np.where(data["curvature"] > (0.3662542701), data["holiday"] * 2.0, 0 ))) * data["orig_mean_lighting"] * 2.0 * data["orig_mean_lighting"] * 2.0 * data["holiday"] * -1.) 
    d["gp_94"] = (np.where(((data["speed_limit"] - data["curvature"] * (52.5000000000)) - np.cos(np.where(((data["speed_limit"] - data["curvature"] * (52.5000000000)) - (0.0799999982)) <= (52.5000000000), (52.5000000000), 0 ))) > (52.5000000000), (0.0799999982), 0 )) 
    d["gp_95"] = (((np.cos((((data["curvature"] - np.where(np.where((0.3818522692) > (0.3458362222), data["curvature"], 0 ) > np.where(data["curvature"] > data["orig_mean_lighting"], ((0.3818522692)) / ((0.5799999833)), 0 ), (0.5799999833), 0 ))) / (((0.3818522692)) / ((52.5000000000)))) / ((0.3662542701)))) / ((0.3458362222))) / (data["speed_limit"])) 
    d["gp_96"] = (np.where(data["orig_mean_weather"] * -1. <= np.cos(data["num_reported_accidents"]), data["num_reported_accidents"], 0 ) * np.where(data["curvature"] <= data["orig_mean_weather"] * data["orig_mean_speed_limit"], np.cos(data["orig_mean_weather"]), 0 ) * data["orig_mean_weather"] * np.cos(data["num_reported_accidents"]) * data["num_reported_accidents"] * data["orig_mean_speed_limit"]) 
    d["gp_97"] = (np.where(data["curvature"] <= np.where((0.4025629163) <= data["orig_mean_weather"], np.sin(data["orig_min_lighting"] * data["num_reported_accidents"]), 0 ), data["curvature"] * 2.0, 0 ) * data["num_reported_accidents"] * data["num_reported_accidents"] * -1.) 
    d["gp_98"] = (np.where(data["orig_mean_curvature"] > (0.3818522692), (np.where(data["orig_mean_curvature"] > data["weather"], (0.3750000000), 0 ) - ( 0.7015008330  + np.where(data["orig_mean_speed_limit"] > data["curvature"], (data["curvature"]) / ((np.where((0.3587038517) > data["weather"], (0.3750000000), 0 ) + data["orig_mean_curvature"])/2.0 ), 0 ))/2.0 ), 0 )) 
    d["gp_99"] = (np.where(data["curvature"] <= np.cos(data["speed_limit"]), np.where(np.cos(np.where(np.cos(data["orig_mean_weather"]) <= data["curvature"], np.cos((np.cos((data["speed_limit"] + data["num_reported_accidents"])/2.0 ) + data["num_reported_accidents"])/2.0 ) * -1., 0 )) <= data["curvature"], np.cos(np.cos(data["curvature"])) * -1., 0 ), 0 )) 
    d["gp_100"] = (np.where((0.2533500195) > data["curvature"] * (0.5799999833), (data["orig_min_curvature"] * np.where(data["orig_min_curvature"] / 2.0 <= data["curvature"] * (0.5799999833), data["orig_mean_weather"], 0 )) / ((0.2556182742)) * -1. * 2.0, 0 ) * -1.) 
    d["gp_101"] = (np.where(data["orig_mean_num_lanes"] <= data["time_of_day"], np.where(data["time_of_day"] <= data["lighting"], np.where(data["orig_mean_num_lanes"] <= data["time_of_day"], np.where((data["time_of_day"] - np.sin(data["num_reported_accidents"])) <= data["time_of_day"], ((data["time_of_day"] - np.sin( 3.0 )) - np.sin(data["num_reported_accidents"])), 0 ), 0 ) * 2.0, 0 ), 0 ) * 2.0 * 2.0) 
    d["gp_102"] = ((np.where(data["speed_limit"] > data["curvature"], np.sin((data["speed_limit"] * np.sin((np.where(data["weather"] > data["orig_mean_num_reported_accidents"], np.where(data["weather"] > data["curvature"] / 2.0, data["curvature"] / 2.0, 0 ), 0 ) + np.sin(data["weather"])) * data["orig_mean_lighting"]) - data["speed_limit"])), 0 )) / (data["speed_limit"])) 
    d["gp_103"] = (np.sin(((0.3662542701)) / ((data["curvature"] * np.sin((data["orig_mean_lighting"]) / ((data["orig_mean_weather"] * data["orig_min_speed_limit"] * (0.3662542701) * data["curvature"] + (data["orig_mean_weather"] - data["curvature"]))/2.0 )) + (data["orig_mean_weather"] - data["curvature"]))/2.0 )) * data["orig_min_speed_limit"]) 
    d["gp_104"] = (np.where(data["orig_min_lighting"] <= ((data["curvature"] - 1.0000000000) - data["orig_min_speed_limit"]), (1.0000000000 - data["num_reported_accidents"]), 0 ) * 2.0 * 2.0) 
    d["gp_105"] = ((np.sin(((0.2556182742) + np.sin(((0.2556182742) / 2.0 + data["speed_limit"])))) * -1.) / ((data["speed_limit"] - (data["orig_mean_weather"]) / ((((0.2556182742) - data["orig_min_curvature"]) - data["orig_mean_weather"] / 2.0))))) 
    d["gp_106"] = (np.where(data["lighting"] <= np.where(np.where(np.where(data["lighting"] <= np.sin(np.sin(data["weather"])), data["orig_min_curvature"], 0 ) <= data["num_reported_accidents"], (0.0450000018), 0 ) <= np.sin(data["orig_min_curvature"]) / 2.0, data["weather"], 0 ), (0.0450000018), 0 )) 
    d["gp_107"] = (((np.where(data["curvature"] <= (data["orig_mean_speed_limit"] + (((0.2850000262) + (data["orig_mean_weather"] + data["curvature"])/2.0 )/2.0  + np.where(data["curvature"] > (np.cos((0.5049999952)) + data["orig_min_lighting"])/2.0 , data["orig_mean_weather"], 0 )))/2.0 , data["orig_min_lighting"], 0 ) * data["curvature"]) / ((0.2850000262))) / (data["orig_mean_speed_limit"])) 
    d["gp_108"] = (np.where(((0.4079468548) + data["orig_mean_weather"] * -1.)/2.0  > np.where(data["orig_min_lighting"] > data["orig_min_curvature"] / 2.0, np.where(data["orig_min_lighting"] > data["orig_min_curvature"] / 2.0, data["orig_min_curvature"] / 2.0 / 2.0 * -1., 0 ), 0 ), data["orig_min_lighting"] * -1., 0 )) 
    d["gp_109"] = (((np.cos((data["curvature"]) / (((data["orig_mean_lighting"]) / ((data["orig_mean_weather"] - data["curvature"])) - data["curvature"] * -1.))) - (data["orig_mean_weather"] - np.cos(data["orig_mean_weather"])))) / (((np.cos(data["orig_mean_weather"])) / ((data["orig_mean_lighting"] - data["curvature"])) - (65.0000000000)))) 
    d["gp_110"] = ((np.sin(data["speed_limit"] * (data["curvature"]) / ((0.0799999982))) + np.sin((data["curvature"] - data["speed_limit"] * ((data["curvature"] * 2.0 + np.sin(data["orig_min_curvature"] / 2.0 * -1. / 2.0))/2.0 ) / ((0.0799999982) / 2.0))))/2.0  * (0.0799999982)) 
    d["gp_111"] = ((np.sin(np.sin((np.sin((((data["orig_mean_curvature"]) / ((((0.4079468548) - data["orig_mean_lighting"] / 2.0)) / ((52.5000000000))) - data["orig_mean_lighting"]) - np.where(data["orig_mean_lighting"] <= data["orig_mean_curvature"], (0.4079468548), 0 ))) * 2.0 - (0.4079468548))) * 2.0) * 2.0) / (data["speed_limit"])) 
    d["gp_112"] = ((((np.sin((data["orig_min_curvature"] - (data["curvature"]) / ((data["orig_mean_lighting"]) / ((52.5000000000))))) + np.sin(((65.0000000000) - ((65.0000000000)) / (data["orig_mean_curvature"]))))) / (data["orig_mean_lighting"])) / ((65.0000000000))) 
    d["gp_113"] = ((np.cos(data["orig_mean_curvature"] * data["curvature"] * data["speed_limit"] * 2.0) - np.cos(data["speed_limit"] * data["curvature"] * data["speed_limit"] * data["curvature"] * 2.0)) * np.where(data["speed_limit"] > data["orig_mean_curvature"] * data["speed_limit"] * 2.0, data["orig_min_lighting"], 0 )) 
    d["gp_114"] = ((((data["orig_min_curvature"] - np.sin(((30.0000000000) * (30.0000000000) - (0.2850000262)) * data["curvature"] / 2.0 * (30.0000000000))) - np.sin((data["curvature"] / 2.0 * (30.0000000000) - data["orig_min_curvature"]) * (30.0000000000)))) / ((30.0000000000))) 
    d["gp_115"] = (np.where(data["orig_mean_speed_limit"] <= data["orig_min_curvature"] * 2.0, np.where((0.3662542701) <= data["orig_mean_lighting"], data["orig_mean_lighting"] * 2.0, 0 ) * np.where(data["orig_min_curvature"] <= (0.3458362222) * 2.0, ((0.3458362222) + np.where(data["orig_mean_weather"] <= np.where(data["orig_mean_weather"] <= (0.3662542701), data["orig_mean_lighting"], 0 ), data["orig_mean_lighting"], 0 ) * 2.0 * -1.), 0 ), 0 )) 
    d["gp_116"] = (np.where((0.3818522692) <= ((0.3198369741) + np.where(data["orig_mean_curvature"] <= np.sin(data["orig_mean_lighting"]), data["orig_mean_curvature"], 0 ))/2.0 , np.where(data["orig_mean_weather"] <= (0.3818522692), (np.sin(data["orig_mean_lighting"]) + np.where(data["orig_mean_lighting"] <= np.sin(data["orig_mean_curvature"]), np.sin(np.where((0.3818522692) <= (0.3818522692), (0.3818522692), 0 )), 0 ))/2.0 , 0 ), 0 )) 
    d["gp_117"] = (((np.sin(((52.5000000000)) / ((np.where((52.5000000000) > data["curvature"], (((data["orig_mean_weather"] + (0.3750000000))/2.0  + data["curvature"])/2.0  + data["curvature"])/2.0 , 0 ) - data["orig_min_lighting"])))) / ((((data["curvature"] + data["orig_min_lighting"])/2.0  + data["orig_min_lighting"])/2.0  + data["orig_mean_weather"])/2.0 )) / ((52.5000000000))) 
    d["gp_118"] = (((data["orig_min_curvature"] - ((np.where(((data["orig_mean_curvature"] - data["orig_min_curvature"]) - data["orig_mean_lighting"]) <= data["orig_min_curvature"], ((data["orig_mean_lighting"] - -1.0000000000) - -1.0000000000), 0 ) - np.where(-1.0000000000 <= -1.0000000000, data["orig_mean_lighting"], 0 )) - data["orig_min_curvature"])) - -1.0000000000) * (0.0249999985)) 
    d["gp_119"] = (np.where(np.where((0.1800000072) > (np.where(data["orig_mean_speed_limit"] > np.where(data["orig_min_curvature"] <= (0.1800000072), data["orig_mean_curvature"], 0 ), (0.3381867111), 0 ) - data["orig_min_curvature"]), np.where((0.1800000072) > data["orig_min_curvature"], data["orig_min_curvature"], 0 ), 0 ) > ((0.3198369741) - (0.1800000072)), (0.0450000018), 0 )) 
    d["gp_120"] = (np.where(np.where(data["curvature"] * 2.0 * 2.0 * 2.0 <=  0.3097389340 , data["num_reported_accidents"] / 2.0 / 2.0, 0 ) / 2.0 >  0.3097389340 , data["num_reported_accidents"], 0 )) 
    d["gp_121"] = (np.where(data["curvature"] > (data["curvature"]) / ((52.5000000000)), (np.cos(((((data["speed_limit"]) / (np.cos(((52.5000000000)) / (data["orig_mean_lighting"]))) - data["curvature"])) / (np.cos(((52.5000000000)) / (data["orig_mean_lighting"]))) - ((52.5000000000)) / (data["speed_limit"])))) / ((52.5000000000)), 0 )) 
    d["gp_122"] = (((data["orig_mean_lighting"] - (0.2850000262) * data["curvature"])) / ((np.cos((np.cos(data["orig_min_curvature"]) + data["orig_min_curvature"])/2.0 ) - (65.0000000000) * np.cos(np.sin(data["curvature"]) * np.cos(data["orig_mean_lighting"]) * 2.0 * 2.0)))) 
    d["gp_123"] = (np.sin((((52.5000000000)) / ((0.6649999619) * data["orig_mean_curvature"]) - np.cos(data["num_reported_accidents"] * ((52.5000000000)) / ((0.6649999619) * data["orig_mean_curvature"])))) * data["num_reported_accidents"] * np.sin(data["orig_mean_curvature"]) * -1. *  0.0945785269 ) 
    d["gp_124"] = (np.cos(((data["orig_mean_weather"] * data["orig_mean_weather"] - data["speed_limit"] * 2.0)) / ((data["orig_mean_lighting"] * 2.0 * data["orig_mean_lighting"] * data["orig_mean_weather"]) / ((0.3714956343)))) * (data["orig_mean_lighting"]) / ((0.3714956343)) * data["orig_min_curvature"]) 
    d["gp_125"] = (np.where(np.where(data["lighting"] > (data["orig_mean_weather"] - (0.0799999982)), data["num_reported_accidents"] * 2.0, 0 ) <= np.cos(data["num_reported_accidents"]), np.where(data["num_reported_accidents"] <= data["num_reported_accidents"], np.where((0.2850000262) <= data["num_reported_accidents"], np.where(np.sin(data["lighting"]) <= (0.2850000262), np.cos(data["num_reported_accidents"]), 0 ) * 2.0, 0 ) * 2.0, 0 ) * 2.0, 0 )) 
    d["gp_126"] = (((np.sin((data["speed_limit"] - ((52.5000000000) + ((1.0000000000 + (52.5000000000))) / ((np.where((0.3662542701) <= data["orig_mean_curvature"], 1.0000000000, 0 ) + data["orig_mean_weather"])))))) / ((np.where(data["speed_limit"] <= (52.5000000000), data["orig_mean_curvature"], 0 ) + data["orig_mean_curvature"]))) / (data["speed_limit"])) 
    d["gp_127"] = (np.where((np.cos((0.5049999952)) + np.where(data["orig_min_lighting"] <= np.where(data["curvature"] <= data["orig_min_lighting"], (0.5049999952), 0 ), data["orig_min_speed_limit"], 0 ))/2.0  > ((np.where(data["curvature"] <= (0.0799999982), data["orig_min_speed_limit"], 0 ) + (0.3458362222)) + data["curvature"]), data["curvature"], 0 ))
    return d


X = pd.concat([cbtrain[cbtrain.columns[:-1]],GP(cbtrain)],axis=1)
X_test = pd.concat([cbtest.copy(),GP(cbtest.copy())],axis=1)
y = cbtrain['accident_risk']


scores = []
test_preds = []
X_test = pd.concat([cbtest.copy(),GP(cbtest.copy())],axis=1)
kf = KFold(n_splits=30, shuffle=True, random_state=42)

model = HistGradientBoostingRegressor(
                        max_iter=500,
                        learning_rate=.1,
                        early_stopping = True,
                        validation_fraction = 0.1,
                        n_iter_no_change = 20,
                        max_depth=7,
                        random_state=42,verbose=0)

for train_index, test_index in kf.split(X):
    X_train = X.loc[train_index,:]
    X_val = X.loc[test_index,:]
    y_train = y[train_index]
    y_val = y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = mean_squared_error(y_val,y_pred)**.5
    print(score)
    scores.extend([score])
    y_pred = model.predict(X_test)
    test_preds.extend([y_pred])
print(np.mean(scores),np.std(scores))
    
test_preds = np.array(test_preds)
test_preds = np.mean(test_preds,axis=0)


sub = None
if(myenv.is_file()):
   sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv',index_col='id')
   sub['accident_risk'] = test_preds
   sub.to_csv("gpsubmission.csv")
else:
   sub = pd.read_csv('sample_submission.csv',index_col='id')
   sub['accident_risk'] = test_preds
   sub.to_csv("gpsubmission.csv")
sub.describe()

