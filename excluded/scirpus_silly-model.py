import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.linalg import lstsq
from sklearn.metrics import mean_squared_error
from category_encoders.cat_boost import CatBoostEncoder


train = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_train.csv')
train = train.set_index("id")
train["month"] = pd.to_datetime(train.harvest_date).dt.month
train["doy"] = pd.to_datetime(train.harvest_date).dt.day_of_year
train["dow"] = pd.to_datetime(train.harvest_date).dt.day_of_week
del train["harvest_date"]
del train["field_id"]
column_to_move = train.pop("yield_tpha")
train["yield_tpha"] = column_to_move
print(train.shape[0])
test = pd.read_csv('/kaggle/input/crop-yield-prediction-challenge/crop_yield_test.csv')
test = test.set_index("id")
test["month"] = pd.to_datetime(test.harvest_date).dt.month
test["doy"] = pd.to_datetime(test.harvest_date).dt.day_of_year
test["dow"] = pd.to_datetime(test.harvest_date).dt.day_of_week
del test["harvest_date"]
del test["field_id"]
print(test.shape[0])


cats = []
for c in train.columns:
    if(len(train[c].unique())<32):
        cats.append(c)
        train[cats] = train[cats].astype('category')
        test[cats] = test[cats].astype('category')
print(cats)    
numerics = list(set(train.columns[:-1]).difference(set(cats)))+list(['yield_tpha'])
print(numerics)    


cb = CatBoostEncoder()
cbtrain = cb.fit_transform(train.loc[:,cats],train.yield_tpha)
cbtest = cb.transform(test.loc[:,cats])
cbtrain = pd.concat([cbtrain,train.loc[:,numerics]],axis=1)
cbtest = pd.concat([cbtest,test.loc[:,numerics[:-1]]],axis=1)


#Evolved Code ...
def GP(data):
    return (6.266543865203857 +
0.099991522729397*((((((data["nitrogen_content"] / 2.0 / 2.0 + (data["nitrogen_content"] - 1.000000000000000))/2.0 ) / (data["nitrogen_content"]) + data["nitrogen_content"])/2.0  - 1.000000000000000)) / ( 7.0 )) +
0.100000001490116*(( 0.003272295696661  * data["fertilizer_amount"] - ( 7.0 ) / ((data["irrigation_frequency"] + (data["soil_moisture"] + data["fertilizer_amount"])/2.0  / 2.0)/2.0 ) * 2.0) * 2.0) +
0.100000001490116*(((((data["nitrogen_content"]) / (( 6.0 ) / ( 10.0 )) + data["fertilizer_amount"])/2.0  -  9.0  *  10.0 )) / ( 5.0  *  9.0 )) +
0.100000001490116*((((( 0.000313997355988 ) / ((data["potassium_content"] * data["pesticide_usage"] - data["pesticide_usage"])) - data["pesticide_usage"])) / ((data["month"] + data["potassium_content"])) +  1.0 )/2.0 ) +
0.100000001490116*( 10.0  * ((1.000000000000000) / ( 8.0 ) + ( 10.0 ) / (((data["fertilizer_amount"] + ( 10.0 ) / (-1.000000000000000))/2.0  - data["fertilizer_amount"])))) +
0.100000001490116*((((((data["total_rainfall"]) / (data["avg_temperature"]) + data["fertilizer_amount"])/2.0 ) / ( 8.0  * 2.0) - data["dow"])) / ( 8.0 ) * 2.0) +
0.100000001490116*((-1.000000000000000 + (( 10.0  * data["region"] * -1. + ( 8.0  + data["fertilizer_amount"]))/2.0 ) / ( 6.0  *  10.0 ))/2.0  * 2.0) +
0.099988698959351*(( 0.011381152085960  - ((( 9.0 ) / ( 10.0 ) / 2.0 +  10.0 ) * 2.0) / (data["sunlight_hours"])) * 2.0 * 2.0 * 2.0) +
0.100000001490116*((data["soil_moisture"] *  0.002678395016119  + ((( 8.0  - ( 10.0 ) / ((data["soil_moisture"] -  8.0 ))) - data["pesticide_usage"])) / ( 8.0 ))/2.0 ) +
0.099994353950024*(((((data["potassium_content"] + data["potassium_content"])/2.0  * ( 0.126162320375443  + data["potassium_content"])/2.0  +  0.980857849121094 )/2.0  + -1.000000000000000)) / (data["crop_type"])) +
0.100000001490116*(( 10.0 ) / ((data["total_rainfall"]) / ((( 10.0  * data["crop_type"] - data["total_rainfall"]) *  0.025377994403243  +  10.0 )/2.0  * 2.0) * -1.)) +
0.100000001490116*(((((((data["crop_type"] +  10.0 ) - data["crop_type"] *  10.0 ) + data["fertilizer_amount"])/2.0 ) / ( 8.0 )) / ( 8.0 ) + -1.000000000000000)) +
0.100000001490116*((((data["total_rainfall"] / 2.0) / ( 6.0 ) / 2.0 - data["avg_temperature"])) / (((data["avg_temperature"] + (data["avg_temperature"] + data["total_rainfall"])/2.0 )/2.0  + data["avg_temperature"])/2.0 )) +
0.100000001490116*(((((( 9.0  / 2.0 / 2.0 -  10.0 ) + data["fertilizer_amount"] / 2.0)/2.0 ) / ( 8.0  / 2.0) -  10.0 )) / ( 9.0 )) +
0.099997177720070*((((data["region"] - data["soil_ph"]) - -1.000000000000000)) / (data["soil_ph"] * (data["soil_ph"]) / ((data["region"] - data["soil_ph"]) * -1.)) / 2.0) +
0.100000001490116*(((data["phosphorus_content"] + -1.000000000000000)) / (((data["phosphorus_content"] + -1.000000000000000) * (data["phosphorus_content"] + -1.000000000000000) +  6.0 )) * data["phosphorus_content"]) +
0.100000001490116*( 0.052968036383390  * 2.0 * (1.000000000000000 - ( 0.052967797964811  * data["avg_temperature"] + ((data["pesticide_usage"] -  0.627578914165497 ) - data["region"]))/2.0 )) +
0.100000001490116*((((data["season"] - data["pesticide_usage"])) / ((data["irrigation_frequency"] - (( 9.0  + data["season"])/2.0  - data["irrigation_frequency"]))) +  0.256641924381256 )/2.0  / 2.0) +
0.100000001490116*((((data["potassium_content"] +  4.0 ) - ((( 6.0  * 2.0 +  6.0 )) / (data["soil_ph"]) + data["pesticide_usage"])/2.0 )) / ( 6.0  * 2.0)) +
0.100000001490116*(( 0.017025236040354  *  8.0  + ( 8.0  *  10.0  / 2.0) / (data["total_rainfall"] / 2.0) * -1.)) +
0.100000001490116*((((data["total_rainfall"]) / ((data["pesticide_usage"] +  9.0  / 2.0)/2.0 ) + ( 10.0  + data["pesticide_usage"])/2.0  *  10.0  * -1.)) / (data["total_rainfall"])) +
0.099988698959351*(( 0.062697187066078  -  0.062696948647499  * data["avg_temperature"] *  0.029440410435200  * data["avg_temperature"] *  0.062696948647499  * data["avg_temperature"] *  0.029440410435200 )) +
0.099997177720070*((( 10.0  * 2.0 - data["nitrogen_content"] * data["soil_moisture"] / 2.0)) / (data["dow"] * data["soil_moisture"] * 1.000000000000000 * -1.)) +
0.099994353950024*(((-1.000000000000000 + ( 0.876400947570801  + (-1.000000000000000 + data["potassium_content"])))) / (((-1.000000000000000) / ((data["potassium_content"] + data["avg_temperature"])) + data["avg_temperature"])/2.0 )) +
0.100000001490116*((((data["phosphorus_content"] * data["phosphorus_content"] + (data["potassium_content"] * data["potassium_content"] + data["potassium_content"] / 2.0)/2.0 )/2.0  + -1.000000000000000)) / ( 8.0 )) +
0.100000001490116*(((( 7.0  + (data["pesticide_usage"]) / (( 7.0  + data["month"]))) - data["pesticide_usage"])) / ((data["irrigation_frequency"] + data["irrigation_frequency"]) * 2.0)) +
0.100000001490116*(((((data["total_rainfall"] + data["total_rainfall"] / 2.0)/2.0  / 2.0 + data["total_rainfall"])/2.0  / 2.0 / 2.0 -  9.0  *  9.0 )) / (data["total_rainfall"])) +
0.100000001490116*(( 0.010025980882347  + ( 0.012616875581443  - (data["irrigation_frequency"]) / (((data["phosphorus_content"]) / ( 0.012616875581443 ) + data["total_rainfall"])/2.0 ))) * 2.0 * 2.0 * 2.0) +
0.100000001490116*(((data["phosphorus_content"] * data["nitrogen_content"] * (data["phosphorus_content"] -  0.852938354015350 ) + data["nitrogen_content"])/2.0  + -1.000000000000000)/2.0  / 2.0 / 2.0) +
0.100000001490116*(( 0.249044001102448  + data["irrigation_frequency"] * (data["month"]) / (( 0.214354559779167  + (data["total_rainfall"] * ( 7.0 ) / (data["soil_moisture"]) - data["total_rainfall"]))/2.0 ))/2.0 ) +
0.100000001490116*((( 9.0  *  8.0  - (data["fertilizer_amount"] / 2.0 - data["pesticide_usage"] * 2.0))) / (( 0.215911909937859  -  9.0  *  9.0 ))) +
0.099960438907146*(((((1.000000000000000 + data["phosphorus_content"])/2.0 ) / ( 10.0 ) + data["phosphorus_content"]) + data["phosphorus_content"]) * ((-1.000000000000000 + data["phosphorus_content"])/2.0 ) / ( 10.0 )) +
0.099991522729397*((( 2.0  - (data["potassium_content"] * data["potassium_content"] + data["nitrogen_content"])/2.0 )) / ((data["nitrogen_content"] * -1. - data["soil_moisture"] / 2.0))) +
0.099985875189304*(((-1.000000000000000 + ((((-1.000000000000000 + data["nitrogen_content"])) / (data["nitrogen_content"]) + data["nitrogen_content"])/2.0  + ( 0.095304034650326  + data["nitrogen_content"])/2.0 )/2.0 )) / ( 9.0 )) +
0.100000001490116*(( 0.077587857842445  - (( 10.0  +  10.0  * 2.0 * 2.0)/2.0 ) / ((data["soil_moisture"] * 2.0 * 2.0 + data["total_rainfall"])/2.0 )) * 2.0) +
0.100000001490116*(((data["pesticide_usage"] * data["crop_type"] + (data["pesticide_usage"] - (data["total_rainfall"]) / (( 8.0  + data["pesticide_usage"])/2.0 )))) / (data["total_rainfall"] * -1.)) +
0.099988698959351*(((((data["avg_temperature"]) / (( 5.0  +  6.0 )) - data["soil_moisture"]) + data["avg_temperature"])/2.0 ) / ( 5.0  * data["soil_moisture"] * -1.)) +
0.100000001490116*(( 0.073484674096107  - (data["pesticide_usage"]) / ((data["pesticide_usage"] * 2.0 + (data["pesticide_usage"] + ( 3.0  + (data["total_rainfall"]) / (data["pesticide_usage"])))))) * 2.0) +
0.100000001490116*((((data["fertilizer_amount"]) / (data["irrigation_frequency"] * 2.0) - data["irrigation_frequency"] * 2.0)) / (( 7.0  * 2.0 + (data["fertilizer_amount"]) / (data["irrigation_frequency"] * 2.0))/2.0 )) +
0.100000001490116*((((-1.000000000000000 + (-1.000000000000000 + data["total_rainfall"])/2.0 )/2.0  - data["avg_temperature"] * data["pesticide_usage"])) / ( 6.0  * data["avg_temperature"]) *  0.116187363862991 ) +
0.099997177720070*((( 10.0 ) / ((data["soil_ph"] * 2.0 * data["nitrogen_content"] - ((data["season"]) / (data["soil_ph"]) + data["total_rainfall"])/2.0 ))) / (data["nitrogen_content"])) +
0.099977396428585*((((data["phosphorus_content"]) / ((data["avg_temperature"] + data["phosphorus_content"])) -  0.021626954898238 ) -  0.014482263475657 )) +
0.099997177720070*((((data["pesticide_usage"] * -1. + ((data["nitrogen_content"] + (data["soil_ph"] + data["nitrogen_content"])/2.0 ) + data["nitrogen_content"]))/2.0 ) / ( 4.0 )) / ( 6.0 )) +
0.099997177720070*(( 0.081028006970882 ) / ((((data["avg_temperature"]) / ( 0.079909346997738  * data["avg_temperature"]) + (data["month"] - 1.000000000000000)) - data["doy"]))) +
0.099997177720070*((( 0.010360720567405  * data["total_rainfall"] -  7.0 )) / ((data["dow"] + (data["soil_moisture"] +  7.0  * data["region"])/2.0 ))) +
0.100000001490116*(((data["pesticide_usage"] - ((data["soil_moisture"]) / ( 7.0 ) +  7.0  / 2.0))) / ( 8.0  * data["season"]) * -1.) +
0.100000001490116*(( 0.124340325593948  + ( 0.124340325593948  * (data["fertilizer_amount"]) / (( 0.124340325593948  - (data["region"] + data["region"]))) * -1. + -1.000000000000000 * 2.0)/2.0 )) +
0.100000001490116*(((((( 6.0 ) / ( 6.0 ) + data["fertilizer_amount"])/2.0 ) / (( 9.0  +  8.0  * 2.0)/2.0 ) -  7.0 )) / ( 9.0 )) +
0.099997177720070*( 0.016404155641794  * (( 0.016404155641794  + (( 0.231781780719757 ) / (( 0.016404155641794  - ( 0.426158756017685  - data["pesticide_usage"]))) + data["soil_moisture"])/2.0 )/2.0  - data["pesticide_usage"])) +
0.100000001490116*((((data["total_rainfall"]) / (( 8.0  + data["pesticide_usage"])) - ( 9.0  + data["pesticide_usage"]) * 2.0)) / ((data["doy"] + data["total_rainfall"])/2.0 )) +
0.100000001490116*((( 8.0  - (((data["total_rainfall"]) / (data["month"]) - data["month"])) / (data["month"] * 2.0))) / ( 7.0  *  8.0  * -1.)) +
0.099946312606335*(((data["season"] - data["region"])) / ((((data["season"] - data["region"])) / (((data["season"] - data["avg_temperature"]) + data["season"])/2.0 ) + data["avg_temperature"])/2.0 )) +
0.100000001490116*((( 10.0  * -1. *  8.0  + (data["fertilizer_amount"] + (data["sunlight_hours"]) / (data["fertilizer_amount"]))/2.0 )) / (data["fertilizer_amount"])) +
0.100000001490116*(((( 0.236618578433990  * data["fertilizer_amount"]) / (data["month"]) - data["irrigation_frequency"])) / ((data["month"] / 2.0 + data["month"] * data["irrigation_frequency"])/2.0 ) * 2.0) +
0.099994353950024*(((data["nitrogen_content"] * data["dow"] + (data["nitrogen_content"] * 2.0 + (data["dow"] + data["nitrogen_content"])/2.0 )/2.0 )/2.0  - data["pesticide_usage"]) * 2.0 *  0.006791831459850 ) +
0.099997177720070*((((-1.000000000000000 + data["nitrogen_content"])) / ((data["nitrogen_content"] + data["avg_temperature"])) - ((data["avg_temperature"] + data["avg_temperature"] * 2.0)) / (data["sunlight_hours"]))) +
0.099997177720070*((((data["phosphorus_content"] * data["sunlight_hours"]) / ((data["avg_temperature"] + data["pesticide_usage"] * data["pesticide_usage"])/2.0 ) - data["pesticide_usage"] * data["pesticide_usage"])) / (data["sunlight_hours"])) +
0.100000001490116*((((((data["soil_moisture"]) / ( 6.0 ) -  10.0 ) + data["fertilizer_amount"] / 2.0)/2.0  *  0.169650599360466  -  7.0 )) / (data["month"] * 2.0)) +
0.100000001490116*(((data["soil_moisture"] * -1. / 2.0 / 2.0 + data["pesticide_usage"])/2.0 ) / ((data["soil_moisture"] - data["irrigation_frequency"] * 2.0 * data["irrigation_frequency"]))) +
0.099997177720070*( 0.020982271060348  * ((data["soil_ph"] - data["pesticide_usage"]) + (((data["soil_ph"]) / (data["avg_temperature"])) / ( 0.020982271060348 )) / (data["avg_temperature"]))/2.0 ) +
0.099997177720070*((((data["soil_ph"] - data["season"]) * ((data["season"]) / (data["soil_moisture"]) - (data["soil_ph"] - data["season"]))) / (data["soil_moisture"]) +  0.036718137562275 )) +
0.100000001490116*(( 0.009587766602635  + ( 0.009587766602635  + (data["irrigation_frequency"]) / (((-1.000000000000000) / ( 0.009587766602635 ) - data["total_rainfall"]))) * 2.0 * 2.0 * 2.0)) +
0.099983051419258*(((data["potassium_content"] * (-1.000000000000000 + (data["potassium_content"] + -1.000000000000000))/2.0  / 2.0 + (-1.000000000000000 + data["potassium_content"]))/2.0 ) / (data["region"])) +
0.099997177720070*(( 0.049181710928679 ) / ((( 5.0 ) / (( 6.0  - data["pesticide_usage"])) *  0.049181710928679  + ( 5.0 ) / (( 8.0  - data["pesticide_usage"]))))) +
0.100000001490116*(((data["fertilizer_amount"]) / ( 10.0 ) +  7.0 )/2.0  * ( 0.074082389473915  + ( 10.0 ) / (((data["fertilizer_amount"]) / (data["soil_ph"]) - data["fertilizer_amount"])))/2.0 ) +
0.100000001490116*((((data["pesticide_usage"] - (data["fertilizer_amount"]) / ( 5.0 )) +  10.0  * 2.0)) / (((data["fertilizer_amount"] / 2.0 - data["avg_temperature"]) - data["fertilizer_amount"]))) +
0.100000001490116*((( 9.0  * 2.0 * 2.0 - data["fertilizer_amount"] / 2.0 / 2.0)) / (((data["fertilizer_amount"] / 2.0 -  10.0  * 2.0) - data["fertilizer_amount"]))) +
0.099929355084896*(((data["region"] - data["dow"])) / ((((data["region"] - data["dow"]) - data["phosphorus_content"]) * data["region"] - data["dow"]))) +
0.099997177720070*(((data["potassium_content"] * data["potassium_content"] +  0.012816670350730  * data["soil_moisture"]) * data["potassium_content"] *  0.012816670350730  + (-1.000000000000000) / (data["soil_moisture"]))) +
0.099974565207958*((((-1.000000000000000 + (-1.000000000000000 + data["phosphorus_content"])) + (data["soil_moisture"] / 2.0) / (data["month"]) / 2.0)) / (data["soil_moisture"])) +
0.100000001490116*(( 0.069856181740761  + ( 9.0 ) / (((data["fertilizer_amount"] / 2.0 -  7.0  * 2.0 * 2.0) / 2.0 - data["fertilizer_amount"]))) * 2.0 * 2.0) +
0.095363102853298*(((( 0.006471873726696  -  0.009401561692357  * -1.) * -1.) / (data["region"]) +  0.004920960869640  * -1. * -1.000000000000000) * -1. * -1. * -1.) +
0.099988698959351*(((-1.000000000000000 + (data["potassium_content"] + (-1.000000000000000 + (data["potassium_content"] * 2.0 + data["potassium_content"])/2.0 )/2.0  * data["potassium_content"])/2.0 )/2.0 ) / ( 8.0 )) +
0.100000001490116*(((data["avg_temperature"] + (data["nitrogen_content"]) / (((data["pesticide_usage"] + data["avg_temperature"])) / ((data["avg_temperature"] - data["total_rainfall"]))))) / ((data["potassium_content"] - data["total_rainfall"]))) +
0.100000001490116*(((data["pesticide_usage"]) / ((data["pesticide_usage"] * -1. - (data["total_rainfall"] * data["potassium_content"] + data["pesticide_usage"] * data["pesticide_usage"])/2.0 )) * 2.0 +  0.029518373310566 )) +
0.100000001490116*(( 0.176760718226433  - (data["crop_type"]) / ((((data["crop_type"] + data["fertilizer_amount"])/2.0 ) / (( 0.176760718226433  - data["soil_ph"])) + data["fertilizer_amount"])/2.0 ) * 2.0)) +
0.100000001490116*(((( 7.0  + data["avg_temperature"])) / ((data["soil_ph"] - data["fertilizer_amount"])) + ((data["soil_ph"] + data["fertilizer_amount"])) / ((data["avg_temperature"] + data["sunlight_hours"])/2.0 ))/2.0 ) +
0.100000001490116*(((data["dow"] + ((data["dow"] - data["avg_temperature"] / 2.0) + data["pesticide_usage"])/2.0 ) - data["pesticide_usage"]) *  0.013648512773216 ) +
0.099997177720070*(((data["crop_type"]) / (((data["total_rainfall"]) / (data["soil_ph"]) - ((data["dow"]) / ( 0.146096259355545 ) + data["total_rainfall"]))) +  0.012218239717185 ) * 2.0 * 2.0) +
0.100000001490116*(( 0.010074141435325  * (( 6.0  - data["pesticide_usage"]) + ( 0.010074141435325 ) / (data["pesticide_usage"])) +  0.037123687565327 )/2.0 ) +
0.099971741437912*(( 0.034178502857685  - ( 0.237510502338409 ) / (((data["soil_ph"] + ( 0.065186277031898  -  0.034178502857685 ))/2.0 ) / (data["avg_temperature"])) *  0.021159892901778 )) +
0.099994353950024*((((data["soil_ph"] -  7.0 )) / ((data["month"]) / (((data["soil_ph"] +  0.511329770088196 ) - data["region"]) * -1.))) / (data["soil_ph"])) +
0.100000001490116*((( 8.0  * 2.0 + ( 10.0  * 2.0 - data["fertilizer_amount"] / 2.0 / 2.0))/2.0 ) / ((data["soil_moisture"] * -1. + data["fertilizer_amount"] * -1.)/2.0 )) +
0.099974565207958*(((data["potassium_content"] / 2.0 / 2.0 + data["potassium_content"] * data["potassium_content"])/2.0  + -1.000000000000000)/2.0  *  0.093218348920345 ) +
0.099988698959351*((data["avg_temperature"]) / ((data["crop_type"]) / (((data["phosphorus_content"] / 2.0) / (((data["avg_temperature"] + data["phosphorus_content"]) + data["avg_temperature"])/2.0 ) -  0.017299655824900 ))) / 2.0) +
0.099997177720070*(((( 0.250511944293976 ) / (data["doy"])) / (((data["doy"]) / (( 0.250511944293976 ) / (data["phosphorus_content"])) + data["phosphorus_content"])/2.0 )) / (data["phosphorus_content"] * -1.)) +
0.099977396428585*((( 0.008278133347631  * data["phosphorus_content"] * -1. * data["phosphorus_content"] * data["phosphorus_content"] * -1. -  0.001523971906863 ) -  0.001523971906863 ) * -1. * -1.) +
0.100000001490116*((( 0.131042987108231  - ( 7.0 ) / ((( 10.0  - (data["fertilizer_amount"]) / ( 10.0 )) + data["fertilizer_amount"])/2.0 ) * 2.0) +  0.070915475487709 )/2.0  * 2.0) +
0.100000001490116*((((data["pesticide_usage"] + ( 9.0  - data["fertilizer_amount"] / 2.0 / 2.0))/2.0  +  9.0 )) / ((data["fertilizer_amount"] / 2.0 / 2.0 - data["fertilizer_amount"]))) +
0.100000001490116*(( 0.160152718424797  + (data["region"]) / ((((data["crop_type"] * -1. - data["fertilizer_amount"]) + (1.000000000000000 -  10.0 ))/2.0  +  1.0 )/2.0 ))/2.0  * 2.0) +
0.100000001490116*(((((data["fertilizer_amount"] + data["month"] / 2.0)/2.0 ) / (data["month"] * 2.0) -  7.0 )) / ((data["month"] * 2.0 * 2.0 + data["irrigation_frequency"]))) +
0.099985875189304*((data["nitrogen_content"] - (((1.000000000000000) / (data["potassium_content"])) / (data["potassium_content"]) + ((data["pesticide_usage"]) / (data["nitrogen_content"])) / ( 8.0 ))) *  0.025761133059859 ) +
0.099994353950024*( 0.014228824526072  * (( 0.009688379243016 ) / (data["pesticide_usage"]) + (( 6.0  + ( 6.0  - data["pesticide_usage"]))/2.0  + ( 6.0  - data["pesticide_usage"]))/2.0 )/2.0 ) +
0.099997177720070*(((data["fertilizer_amount"] *  0.036003120243549  + ((( 0.036003120243549  - data["fertilizer_amount"])) / (data["sunlight_hours"]) - data["irrigation_frequency"]))/2.0 ) / (( 9.0  + data["irrigation_frequency"]))) +
0.099061883985996*(( 0.010522368364036  -  0.007831575348973 )) +
0.100000001490116*((( 0.060266986489296  - (data["crop_type"] * 2.0) / ((( 10.0  * 2.0 + data["fertilizer_amount"]) +  10.0 ))) +  0.009097578004003 ) * 2.0) +
0.099988698959351*((data["nitrogen_content"] - (data["avg_temperature"] - data["region"] * (data["nitrogen_content"] + data["nitrogen_content"]))) * 2.0 *  0.001497507444583 ) +
0.099997177720070*((((((((data["total_rainfall"]) / (data["region"]) + data["fertilizer_amount"])/2.0 ) / (data["region"]) - data["region"]) -  8.0 ) - data["irrigation_frequency"])) / (data["fertilizer_amount"])) +
0.099997177720070*(((data["pesticide_usage"]) / (((data["fertilizer_amount"] * data["nitrogen_content"]) / ((data["nitrogen_content"] + data["pesticide_usage"])) - data["fertilizer_amount"] * data["nitrogen_content"])) +  0.031228788197041 )) +
0.099983051419258*((((data["soil_moisture"]) / ((((data["soil_moisture"] -  0.282705366611481 )) / (data["doy"]) - data["avg_temperature"])) - data["avg_temperature"]) + data["soil_moisture"]) *  0.001931667793542 ) +
0.100000001490116*(( 0.010304930619895  * 2.0 * data["potassium_content"] * data["potassium_content"] + ( 10.0  * -1.) / (((data["soil_moisture"] -  0.036831624805927 ) + data["total_rainfall"])/2.0 ))) +
0.099997177720070*(( 0.087167046964169  - data["irrigation_frequency"] * (1.000000000000000) / ((((data["soil_moisture"]) / ((data["fertilizer_amount"] +  0.087167285382748 )/2.0 )) / ( 0.087167046964169 ) + data["fertilizer_amount"])/2.0 ))) +
0.100000001490116*(( 0.122159034013748  + (((data["avg_temperature"] + (( 9.0 ) / (data["potassium_content"])) / (data["phosphorus_content"]))/2.0  +  0.122159034013748 )) / (data["fertilizer_amount"] * -1.))/2.0 ) +
0.100000001490116*((data["region"] * (data["region"] - data["soil_ph"]) +  9.0 ) * ( 0.003932476975024  -  0.003932476975024  * (data["region"] - data["soil_ph"]))) +
0.099971741437912*(((1.000000000000000) / (((1.000000000000000 + ((1.000000000000000) / ((1.000000000000000 - data["phosphorus_content"])) - data["soil_moisture"]))/2.0  - data["soil_moisture"])) +  0.049729358404875 )/2.0 ) +
0.099997177720070*( 0.037479169666767  *  0.043842803686857  * data["soil_moisture"] * ((-1.000000000000000 +  0.017863277345896 ) +  0.017863277345896  * data["nitrogen_content"] * data["soil_moisture"])) +
0.099997177720070*((data["fertilizer_amount"] *  0.027825601398945  - ((data["region"]) / (data["potassium_content"]) + (data["pesticide_usage"] + data["region"])/2.0  / 2.0)/2.0 ) *  0.011813166551292  * 2.0) +
0.099988698959351*(((((data["pesticide_usage"]) / ( 9.0 )) / (((data["fertilizer_amount"] + -1.000000000000000)/2.0  - data["fertilizer_amount"])) -  0.069543376564980 ) +  0.000470638391562  * data["fertilizer_amount"])) +
0.099994353950024*(( 0.015366557985544  + ( 0.015366557985544  + (data["irrigation_frequency"]) / (((-1.000000000000000) / ( 0.015366557985544 ) - data["fertilizer_amount"])))) * 2.0 * 2.0) +
0.099985875189304*(( 9.0 ) / ((data["irrigation_frequency"] * (data["month"] *  9.0  -  9.0 ) * (data["irrigation_frequency"] - data["month"]) - data["total_rainfall"]))) +
0.099991522729397*((data["season"]) / ((( 5.0 ) / (( 0.025119787082076  + ( 4.0 ) / (( 4.0  * -1. - data["fertilizer_amount"])))) - data["fertilizer_amount"])) * 2.0) +
0.099997177720070*(((data["pesticide_usage"]) / (( 0.535527110099792  - data["pesticide_usage"]))) / ((( 6.0  + (data["pesticide_usage"] * data["sunlight_hours"] + data["pesticide_usage"]))/2.0  - data["sunlight_hours"]))) +
0.100000001490116*((data["potassium_content"] * (data["potassium_content"] + ((data["potassium_content"] - data["crop_type"])) / (((data["soil_ph"] -  8.0 ) + data["soil_ph"])))/2.0 ) / (data["avg_temperature"])) +
0.099963270127773*((-1.000000000000000) / ((((-1.000000000000000) / ((data["potassium_content"] + -1.000000000000000)/2.0 ) + data["nitrogen_content"])/2.0  + data["fertilizer_amount"] * data["potassium_content"] * data["nitrogen_content"])/2.0 )) +
0.099974565207958*((( 0.001484156004153  +  0.032476432621479 ) + ( 0.022511487826705 ) / ((( 0.001484156004153 ) / (data["pesticide_usage"] * data["nitrogen_content"]) - data["nitrogen_content"]) / 2.0))) +
0.099977396428585*(((data["soil_moisture"] -  0.028749234974384 ) *  0.001716852653772  -  0.046934854239225 )) +
0.099985875189304*((((data["season"] + ((data["dow"]) / (data["phosphorus_content"]) + 0.000000000000000)/2.0 )) / (( 0.019356016069651  + (1.000000000000000 - data["pesticide_usage"])))) / (data["sunlight_hours"])) +
0.099997177720070*(((-1.000000000000000) / (((data["fertilizer_amount"] + -1.000000000000000)/2.0  + (data["month"] + ( 2.0 ) / (data["fertilizer_amount"]))/2.0 )/2.0 ) + data["fertilizer_amount"] *  0.000198125882889 )) +
0.099983051419258*((data["phosphorus_content"]) / ( 8.0 ) * ((data["phosphorus_content"] + (data["nitrogen_content"] + ( 0.063726440072060  + (-1.000000000000000 + data["nitrogen_content"])/2.0 )/2.0 )/2.0 )/2.0  + -1.000000000000000)/2.0 ) +
0.099971741437912*(( 0.006598235573620  - (-1.000000000000000) / (((-1.000000000000000 + -1.000000000000000) * 2.0 * 2.0 * 2.0 - data["fertilizer_amount"]))) * 2.0 * 2.0 * 2.0) +
0.099858723580837*(((data["pesticide_usage"] * -1. + data["month"]) + ( 0.016599182039499 ) / (data["pesticide_usage"])) * 2.0 *  0.001375198713504 ) +
0.099985875189304*(( 0.190016791224480 ) / (((((( 0.074884906411171 ) / (data["pesticide_usage"])) / ( 0.098602794110775 ) + data["doy"])) / ((data["doy"] - data["pesticide_usage"])) - data["doy"]))) +
0.099983051419258*(((data["crop_type"] - data["total_rainfall"] *  0.009383918717504 )) / (((data["irrigation_frequency"] - data["pesticide_usage"]) -  10.0 ) *  9.0 )) +
0.099988698959351*(((data["potassium_content"] + -1.000000000000000)) / ((((-1.000000000000000 + -1.000000000000000)) / (data["pesticide_usage"]) + data["potassium_content"] * data["avg_temperature"]))) +
0.099997177720070*((((( 9.0  - data["pesticide_usage"]) - ( 10.0 ) / (( 4.0  - data["soil_ph"])))) / (( 4.0  - data["soil_ph"]))) / (data["total_rainfall"])) +
0.099977396428585*((data["potassium_content"] * ( 0.013742689043283  * data["potassium_content"] - data["pesticide_usage"] *  0.001586199156009 ) * data["potassium_content"] -  0.013742450624704 )) +
0.099355749785900*((( 0.007604362443089  * -1. + (( 0.007604362443089 ) / ( 0.821012914180756 ) * -1.) / ( 0.821012914180756 ) * data["phosphorus_content"] * -1.)) / ( 0.821012914180756 )) +
0.099997177720070*(((data["phosphorus_content"]) / (-1.000000000000000 * data["phosphorus_content"])) / (((data["phosphorus_content"]) / ((data["phosphorus_content"] + -1.000000000000000)) + data["fertilizer_amount"]))))


cbtest['yield_tpha'] = GP(cbtest)
cbtest[['yield_tpha']].to_csv('gpsubmission.csv')

