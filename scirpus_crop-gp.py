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


class MyGP:
    def __init__(self):
        pass

    def CalculateYield(self, data):
        return (self.GPI(data)+self.GPII(data)+self.GPIII(data))/3
    

    def GPI(self, data):
        return (6.266543865203857 +
                0.100000001490116*np.tanh(( 0.849214971065521 ) / (((( 0.849215209484100 ) / ((data["pesticide_usage"] - (6.300599098205566))) + (11.292642593383789))) / (((7.518319606781006) - data["pesticide_usage"])))) +
                0.100000001490116*np.tanh((((data["total_rainfall"]) / ((10.346044540405273)) - ((348.315795898437500) - ((data["fertilizer_amount"] - data["pesticide_usage"] * 2.0) - data["pesticide_usage"]) * 2.0))) / ((103.398391723632812))) +
                0.099971741437912*np.tanh((((6.211860656738281) - ((6.237246513366699) / 2.0 + data["avg_temperature"])/2.0  / 2.0)) / (((27.416938781738281) + (data["irrigation_frequency"] * data["crop_type"] - data["avg_temperature"])))) +
                0.099997177720070*np.tanh(((data["potassium_content"] - (1.311273574829102))) / ((((1.311273574829102) + ((1.311273574829102)) / (((1.311273574829102) - data["potassium_content"]) * 2.0))/2.0  + (6.313077926635742)))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - ((data["pesticide_usage"] - ((8.462858200073242) - data["pesticide_usage"]) * 2.0 * 2.0) + (165.387481689453125)))) / ((135.007110595703125) / 2.0)) +
                0.100000001490116*np.tanh((((data["total_rainfall"]) / (((135.007110595703125) + ((165.387481689453125) - data["fertilizer_amount"]))/2.0 ) - ((165.387481689453125) - data["fertilizer_amount"]))) / (data["fertilizer_amount"] / 2.0)) +
                0.100000001490116*np.tanh(( 0.751381814479828  + ((978.743408203125000)) / (((234.419799804687500) * -1. - (data["total_rainfall"] - (((978.743408203125000)) / (data["soil_moisture"]) - data["soil_moisture"])) * 2.0)))/2.0 ) +
                0.100000001490116*np.tanh(((0.388530671596527) - ((data["phosphorus_content"]) / ((((data["pesticide_usage"] - (2112.323730468750000))) / (data["soil_moisture"])) / (data["pesticide_usage"])) + (data["pesticide_usage"]) / ((14.573366165161133))))) +
                0.100000001490116*np.tanh(((((data["soil_moisture"] * data["potassium_content"] * data["potassium_content"]) / (data["avg_temperature"]) - data["potassium_content"]) - (data["avg_temperature"]) / (data["soil_moisture"]))) / (data["soil_moisture"])) +
                0.100000001490116*np.tanh(((((data["total_rainfall"]) / (((10.802579879760742) + data["pesticide_usage"])/2.0  / 2.0) - data["pesticide_usage"] * data["pesticide_usage"]) - data["avg_temperature"] * 2.0)) / (data["total_rainfall"])) +
                0.099988698959351*np.tanh((((22.050069808959961) - (data["soil_moisture"] -  7.0 ) * data["potassium_content"])) / ((data["soil_moisture"] - (data["soil_moisture"] * (22.050069808959961) + (6.234696388244629))/2.0 ))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] + ((165.387481689453125) + ((135.007110595703125) + data["fertilizer_amount"])/2.0 )/2.0 )/2.0 ) / (((85.944404602050781)) / (((data["fertilizer_amount"] - (165.387481689453125))) / (data["fertilizer_amount"])))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - (165.387481689453125))) / ((((197.597854614257812) + ((data["fertilizer_amount"] - (135.007110595703125))) / ((data["fertilizer_amount"]) / ((197.597854614257812))))/2.0  + (103.398391723632812))/2.0 )) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - (165.387481689453125))) / (((165.387481689453125) + (((data["fertilizer_amount"] - (165.387481689453125))) / ((data["fertilizer_amount"]) / ((165.387481689453125))) + (165.387481689453125))/2.0 )/2.0 )) +
                0.100000001490116*np.tanh(((((21.059318542480469) - data["avg_temperature"])) / (((14.471948623657227) - ((data["avg_temperature"]) / ((data["avg_temperature"] - (14.471948623657227)))) / ((26.583709716796875))))) / ((11.216367721557617))) +
                0.100000001490116*np.tanh(((( 0.034740693867207  * (data["sunlight_hours"] - (698.024658203125000) * 2.0) * 2.0 + data["total_rainfall"]) - (698.024658203125000))) / ((data["total_rainfall"] + (2833.532470703125000))/2.0 )) +
                0.099994353950024*np.tanh(((data["potassium_content"] - (1.360137820243835)) / 2.0) / ((((data["potassium_content"] * -1. - data["potassium_content"]) + (1.360137820243835)) + (6.088520050048828)))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - (((276.883239746093750)) / (((data["fertilizer_amount"]) / ((8.462858200073242))) / (data["pesticide_usage"])) + ((165.387481689453125) + (312.358825683593750))/2.0 )/2.0 )) / ((165.387481689453125))) +
                0.099997177720070*np.tanh(((data["nitrogen_content"] - ((45.155815124511719)) / (data["soil_moisture"]))) / (((-1.000000000000000) / ((data["soil_moisture"] + (data["nitrogen_content"] - (45.155815124511719)))) + (27.000251770019531))/2.0 )) +
                0.100000001490116*np.tanh(((((data["pesticide_usage"] - data["nitrogen_content"]) - data["nitrogen_content"]) - (8.017212867736816) / 2.0)) / ((data["nitrogen_content"] + (27.416938781738281) * -1.))) +
                0.100000001490116*np.tanh(((data["total_rainfall"] * -1. + ((6.244260787963867) * -1. * 2.0 + (978.743408203125000))/2.0 )) / ((data["total_rainfall"] + data["total_rainfall"] * data["region"] * -1.))) +
                0.099991522729397*np.tanh((((24.188686370849609) - data["avg_temperature"])) / (data["avg_temperature"] * (((21.059318542480469) - (((16.168031692504883)) / ((24.188686370849609)) + data["avg_temperature"])) + (16.168031692504883)))) +
                0.100000001490116*np.tanh(((( 6.0  + data["nitrogen_content"]) - data["pesticide_usage"])) / (((data["nitrogen_content"]) / (( 6.0  - data["pesticide_usage"])) + (24.979068756103516)))) +
                0.100000001490116*np.tanh(((data["nitrogen_content"] - (((((1.654641866683960)) / (data["nitrogen_content"]) + (data["avg_temperature"]) / (data["nitrogen_content"]))/2.0  + data["avg_temperature"])/2.0 ) / ( 8.0 ))) / (data["avg_temperature"])) +
                0.100000001490116*np.tanh((data["phosphorus_content"] * (data["phosphorus_content"] - 1.000000000000000)) / ((((1.000000000000000 - data["season"]) + (6.238955020904541)) + (6.238955020904541)))) +
                0.100000001490116*np.tanh((((data["soil_ph"] + (7.884884834289551))/2.0  - data["pesticide_usage"])) / ((((7.884884834289551) + (7.884884834289551))/2.0  + (((6.241507053375244) + (7.992333412170410)) + (10.802579879760742))))) +
                0.099991522729397*np.tanh(((data["nitrogen_content"] + ((1.736552000045776) * -1. - (1.736552000045776) * ( 0.008802654221654 ) / ((data["nitrogen_content"] - (1.736552000045776)))))/2.0 ) / ((8.017212867736816))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - (165.387481689453125))) / (((165.387481689453125) * 2.0 + (77.933166503906250) * ((data["fertilizer_amount"] - (94.881408691406250) * 2.0)) / (data["fertilizer_amount"]))/2.0 )) +
                0.100000001490116*np.tanh(((((773.473510742187500) + (773.473510742187500) / 2.0)/2.0  - data["total_rainfall"])) / ((((421.740692138671875) / 2.0 - data["total_rainfall"]) - (773.473510742187500)) * 2.0)) +
                0.100000001490116*np.tanh((((7.992333412170410) - data["pesticide_usage"])) / (((((7.992333412170410) - data["irrigation_frequency"]) * 2.0 + data["pesticide_usage"])/2.0  - data["irrigation_frequency"] * data["irrigation_frequency"])) * -1.) +
                0.099932186305523*np.tanh(( 8.0  * 2.0) / ((((6.617258071899414)) / ((data["region"] - (6.226549148559570))) - data["sunlight_hours"]))) +
                0.100000001490116*np.tanh((((165.387481689453125) - data["fertilizer_amount"])) / (((165.387481689453125) + (((data["fertilizer_amount"] + data["fertilizer_amount"] / 2.0)/2.0  + (165.387481689453125))/2.0  + data["fertilizer_amount"])/2.0 )/2.0  * -1.)) +
                0.100000001490116*np.tanh((((data["soil_moisture"] - (data["avg_temperature"]) / (data["nitrogen_content"] / 2.0)) - data["nitrogen_content"])) / (((data["soil_moisture"] + (1598.307128906250000))/2.0 ) / (data["nitrogen_content"]) / 2.0)) +
                0.100000001490116*np.tanh((((173.912139892578125) - data["fertilizer_amount"])) / ((((((173.912139892578125) - data["fertilizer_amount"]) - data["fertilizer_amount"])) / ((data["fertilizer_amount"]) / ((27.000000000000000))) - (173.912139892578125)))) +
                0.099994353950024*np.tanh(((data["potassium_content"] - ((8.017212867736816)) / (data["soil_ph"]))) / ((((data["soil_ph"]) / (((8.017212867736816)) / (data["soil_ph"])) + data["soil_ph"])) / (data["potassium_content"]))) +
                0.099983051419258*np.tanh(((data["potassium_content"] + ((1.593693971633911)) / (data["nitrogen_content"] * -1.))) / (((data["nitrogen_content"]) / ((data["potassium_content"] * -1.) / ((1.593693971633911))) + data["avg_temperature"]))) +
                0.100000001490116*np.tanh(((((165.387481689453125) - data["fertilizer_amount"]) - data["nitrogen_content"])) / (((((165.387481689453125) - data["fertilizer_amount"])) / ((data["total_rainfall"]) / ((218.777847290039062))) - (218.777847290039062)))) +
                0.100000001490116*np.tanh((data["phosphorus_content"] * data["phosphorus_content"] * data["nitrogen_content"] * (data["nitrogen_content"] * (6.241961479187012)) / ((2450.111816406250000)) * 2.0 -  0.006038905587047 ) * 2.0) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"]) / (((259.510589599609375) + ( 0.690656125545502  - data["nitrogen_content"]) * (6.316970825195312) * data["nitrogen_content"])/2.0 ) / 2.0 + (0.705683529376984) * -1.)) +
                0.099960438907146*np.tanh((( 0.003390074474737  * -1.) / ((data["irrigation_frequency"] - (6.252044677734375))) *  0.003390074474737 ) / ((data["irrigation_frequency"] - (6.247577667236328))) *  0.004539252258837 ) +
                0.100000001490116*np.tanh((((((data["total_rainfall"] - (data["sunlight_hours"] + (698.024658203125000) * -1.)/2.0 )) / ((6.235262870788574)) - data["total_rainfall"]) + (698.024658203125000))) / ((2172.331542968750000)) * -1.) +
                0.099997177720070*np.tanh(((data["nitrogen_content"] / 2.0 - (0.851893186569214))) / ((((0.929545640945435) * 2.0 + (data["nitrogen_content"] + (0.929545640945435))) + (data["nitrogen_content"] + (6.297054290771484))))) +
                0.099994353950024*np.tanh((((6.228028774261475) - data["region"])) / (((((6.228028774261475) - data["region"])) / ((data["region"] - (6.186131477355957))) + data["region"]))) +
                0.100000001490116*np.tanh((((85.944404602050781) * 2.0 - data["fertilizer_amount"])) / (((((135.007110595703125) - data["fertilizer_amount"])) / ((data["total_rainfall"]) / ((135.007110595703125) * 2.0)) - (226.795288085937500)))) +
                0.100000001490116*np.tanh((((72.000000000000000) - (data["fertilizer_amount"] - (6.213725090026855) / 2.0) / 2.0)) / ((((72.000000000000000) * -1. + data["fertilizer_amount"] / 2.0)/2.0  - data["fertilizer_amount"]))) +
                0.099997177720070*np.tanh(( 0.173723503947258  / 2.0 / 2.0 - ((((((60.500000000000000) - data["total_rainfall"]) - data["total_rainfall"])) / (data["avg_temperature"]) + (60.500000000000000))) / (data["total_rainfall"]))) +
                0.099997177720070*np.tanh(((data["soil_moisture"] - data["avg_temperature"])) / ((((49.057014465332031)) / (((data["soil_moisture"] - data["avg_temperature"]) - data["avg_temperature"])) + data["soil_moisture"] * (13.330560684204102)))) +
                0.099994353950024*np.tanh((6.263886451721191) *  9.0  * ((data["phosphorus_content"] * data["phosphorus_content"] - data["phosphorus_content"])) / ((( 9.0  + data["crop_type"]) + (1096.338378906250000))/2.0 )) +
                0.100000001490116*np.tanh((((85.944404602050781) * 2.0 + (((259.510589599609375)) / (((85.944404602050781) - ((135.007110595703125) - data["fertilizer_amount"]))) - data["fertilizer_amount"]))/2.0 ) / ((165.387481689453125) * -1.)) +
                0.100000001490116*np.tanh(((data["nitrogen_content"] - (data["pesticide_usage"] -  6.0 ))) / ((((data["pesticide_usage"] + data["nitrogen_content"])/2.0 ) / ((data["pesticide_usage"] - (6.235155582427979))) + (72.000000000000000))/2.0 )) +
                0.099997177720070*np.tanh(((data["nitrogen_content"] - (1.654641866683960))) / (((((1.654641866683960) + data["nitrogen_content"])) / (((data["nitrogen_content"] - (1.654641866683960)) + data["nitrogen_content"])) + (27.000000000000000)))) +
                0.100000001490116*np.tanh(((data["total_rainfall"] + ((data["total_rainfall"]) / ((data["total_rainfall"] + (data["total_rainfall"] - (1148.990112304687500)))/2.0 ) - (594.861328125000000)))/2.0 ) / ((data["total_rainfall"] + (1148.990112304687500)))) +
                0.099991522729397*np.tanh((((data["avg_temperature"] - (data["soil_moisture"] + ((45.155815124511719)) / ((data["avg_temperature"] - (38.487670898437500)))))) / (data["avg_temperature"])) / ((data["avg_temperature"] - (38.487670898437500)))) +
                0.099997177720070*np.tanh(((data["potassium_content"]) / ((((35.827083587646484) - ((6.096069335937500) + data["soil_moisture"])/2.0 ) + (6.096069335937500))/2.0 ) - ( 8.0 ) / ((data["total_rainfall"]) / ((6.096069335937500))))) +
                0.100000001490116*np.tanh(((((((data["pesticide_usage"] + data["pesticide_usage"])/2.0  - data["avg_temperature"]) + (32.845542907714844))/2.0 ) / ((6.277909278869629)) + (data["season"] - data["pesticide_usage"]))/2.0 ) / ((20.257869720458984))) +
                0.099994353950024*np.tanh(((data["potassium_content"] - ((1.685156345367432)) / (data["nitrogen_content"]))) / (((data["nitrogen_content"] + ((1354.437500000000000) + data["sunlight_hours"])/2.0 )/2.0 ) / ( 8.0 ) / 2.0 / 2.0)) +
                0.099994353950024*np.tanh(((data["soil_moisture"] - (data["avg_temperature"] + (6.293375015258789)))) / ((6.211860656738281) * (((12.003805160522461) + ((6.293375015258789) - data["avg_temperature"])) + (49.057014465332031)))) +
                0.100000001490116*np.tanh((((85.944404602050781) - (data["fertilizer_amount"] + ((268.091552734375000)) / ((((135.007110595703125) - data["fertilizer_amount"]) - (85.944404602050781))))/2.0 )) / ((85.944404602050781) * -1.) / 2.0) +
                0.099977396428585*np.tanh((( 0.052150975912809 ) / ((data["crop_type"] - (6.308335304260254))) / 2.0 - ( 0.057967200875282 ) / (((6.316774368286133) - data["crop_type"]))) *  0.000926256412640 ) +
                0.099980227649212*np.tanh( 0.232582867145538  * ((data["potassium_content"] + (data["nitrogen_content"] + (-1.000000000000000 + ( 0.053895007818937  + (data["potassium_content"] + (1.547548294067383))/2.0 )/2.0 )/2.0 )/2.0 )/2.0  + -1.000000000000000)/2.0 ) +
                0.100000001490116*np.tanh((((((226.795288085937500) + data["total_rainfall"] / 2.0)/2.0  + (data["fertilizer_amount"] - (268.091552734375000)))/2.0  + (data["fertilizer_amount"] - (268.091552734375000)))) / ((594.861328125000000))) +
                0.100000001490116*np.tanh((((data["fertilizer_amount"] + (((data["sunlight_hours"]) / ((85.944404602050781)) - (226.795288085937500))) / ((data["total_rainfall"]) / ((85.944404602050781))))/2.0  - (70.183563232421875))) / ((226.795288085937500))) +
                0.100000001490116*np.tanh(((((data["total_rainfall"] + data["soil_moisture"] * 2.0 * 2.0)/2.0 ) / ((24.188686370849609)) + data["pesticide_usage"] * 2.0 * -1.)) / ((0.000000000000000 + (196.336090087890625))/2.0 )) +
                0.100000001490116*np.tanh(((((data["phosphorus_content"]) / (data["pesticide_usage"]) + (data["fertilizer_amount"] - (103.398391723632812)))/2.0 ) / ((346.000000000000000)) - (data["pesticide_usage"] * 2.0) / (data["fertilizer_amount"]))) +
                0.100000001490116*np.tanh(((data["potassium_content"] + -1.000000000000000)/2.0 ) / ((((-1.000000000000000 + 0.000000000000000)/2.0  + (data["potassium_content"]) / ((-1.000000000000000 + data["potassium_content"])/2.0 ))/2.0  + data["avg_temperature"])/2.0 )) +
                0.100000001490116*np.tanh((((data["fertilizer_amount"] - data["avg_temperature"] * 2.0)) / ((268.091552734375000) * 2.0) + ((35.827083587646484)) / ((((268.091552734375000)) / (data["fertilizer_amount"]) - data["fertilizer_amount"])))/2.0 ) +
                0.100000001490116*np.tanh(( 0.084687732160091 ) / ((((559.807128906250000) + ((data["total_rainfall"] + (630.205627441406250))/2.0  + (data["total_rainfall"] - (559.807128906250000)))/2.0 )/2.0 ) / ((data["total_rainfall"] - (559.807128906250000))))) +
                0.100000001490116*np.tanh((((data["fertilizer_amount"]) / ( 8.0 ) - ((10.346044540405273) + data["pesticide_usage"]))) / (((10.346044540405273) + data["fertilizer_amount"])/2.0 )) +
                0.100000001490116*np.tanh(((((data["fertilizer_amount"]) / ((6.752473354339600)) + data["fertilizer_amount"])/2.0  - (85.944404602050781))) / (((data["fertilizer_amount"] + data["fertilizer_amount"]) + (85.944404602050781)))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - ((165.387481689453125) + ((1659.392822265625000)) / (data["fertilizer_amount"])))) / ((((1659.392822265625000)) / (((165.387481689453125) - data["fertilizer_amount"])) + (1659.392822265625000))/2.0 )) +
                0.099994353950024*np.tanh(((data["potassium_content"] + (((1.685156345367432) + (1.242614746093750))/2.0  + ((1.242614746093750)) / (data["nitrogen_content"]))/2.0  * -1.)/2.0 ) / (((1.242614746093750) + (6.117704868316650)))) +
                0.099997177720070*np.tanh((data["nitrogen_content"]) / ((((((1796.001342773437500) - data["nitrogen_content"]) - (data["sunlight_hours"]) / (data["nitrogen_content"]))) / (data["nitrogen_content"])) / (data["nitrogen_content"]))) +
                0.100000001490116*np.tanh((((27.000000000000000) - ((data["total_rainfall"] * data["soil_moisture"]) / ((22.050069808959961))) / ((22.050069808959961)))) / ((data["potassium_content"] - data["total_rainfall"]))) +
                0.100000001490116*np.tanh(( 0.034493692219257  + ((268.091552734375000)) / ((data["fertilizer_amount"] - data["fertilizer_amount"] * ((49.057014465332031) + (6.231836318969727))/2.0  * 2.0))) * 2.0 * 2.0) +
                0.099991522729397*np.tanh((( 7.0 ) / (data["nitrogen_content"] / 2.0) - data["potassium_content"] *  7.0 ) * ( 8.0 ) / (( 9.0  - data["sunlight_hours"]))) +
                0.099983051419258*np.tanh((-1.000000000000000 + (-1.000000000000000 + data["potassium_content"] * data["potassium_content"])/2.0  * (data["potassium_content"] + data["potassium_content"] * data["potassium_content"])/2.0 )/2.0  *  0.038939006626606 ) +
                0.099997177720070*np.tanh(((data["total_rainfall"] / 2.0 - (299.500000000000000))) / ((data["total_rainfall"] + ((data["total_rainfall"] + data["doy"] * 2.0) + (876.907836914062500) * 2.0)))) +
                0.099988698959351*np.tanh(((((data["total_rainfall"]) / (data["region"] * data["region"]) - data["season"]) - (6.266191005706787))) / (((234.419799804687500) + data["total_rainfall"])/2.0 ) * 2.0) +
                0.100000001490116*np.tanh(((((1.132035017013550) + data["phosphorus_content"])/2.0  + (data["pesticide_usage"]) / (((data["phosphorus_content"]) / ((-1.000000000000000 + data["pesticide_usage"])/2.0 ) - (7.992333412170410))))/2.0 ) / ((4.578602790832520))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - (276.883239746093750) / 2.0)) / ((data["fertilizer_amount"] - (data["fertilizer_amount"] *  7.0 ) / (data["doy"] * 2.0)) *  7.0 )) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"] - (312.358825683593750) / 2.0)) / (((data["fertilizer_amount"] - ((698.024658203125000) / 2.0) / (data["fertilizer_amount"]) * (142.500000000000000)) + (1926.235229492187500))/2.0 )) +
                0.100000001490116*np.tanh((( 0.354368537664413 ) / ( 3.0 ) - ((data["pesticide_usage"] * 2.0 - ((data["pesticide_usage"]) / ( 3.0  * 2.0) -  3.0 ))) / (data["fertilizer_amount"]))) +
                0.099997177720070*np.tanh((((70.183563232421875) + (((70.183563232421875)) / ((data["fertilizer_amount"] - (268.091552734375000))) - data["fertilizer_amount"]) / 2.0)) / (data["fertilizer_amount"] *  6.0  * -1.)) +
                0.099994353950024*np.tanh(((data["avg_temperature"] - ((20.257869720458984) - (data["avg_temperature"] - (6.316774368286133))))) / ((203.500000000000000) * ((20.257869720458984) - ((6.088520050048828) + data["doy"])/2.0 ))) +
                0.099980227649212*np.tanh(((((data["region"]) / ((167.500000000000000)) - ((data["crop_type"] - (6.332691192626953))) / ((data["dow"] - (6.297779083251953))))) / (data["region"])) / ((191.500000000000000))) +
                0.100000001490116*np.tanh((((data["soil_moisture"]) / ((6.636413574218750)) + ((data["total_rainfall"]) / (((454.571441650390625) + ( 6.0 ) / ( 5.0 ))/2.0 ) - data["pesticide_usage"]))/2.0 ) / ((37.216506958007812))) +
                0.100000001490116*np.tanh(((data["fertilizer_amount"]) / ((data["month"] * 2.0 - data["nitrogen_content"]) * (276.883239746093750)) + ((6.312264919281006)) / (((6.312264919281006) * 2.0 - data["fertilizer_amount"])))) +
                0.099997177720070*np.tanh(((((243.232345581054688) + (243.232345581054688) / 2.0)/2.0  - data["fertilizer_amount"])) / ((data["fertilizer_amount"] * (data["fertilizer_amount"]) / ((243.232345581054688)) - (2775.931152343750000) / 2.0))) +
                0.099997177720070*np.tanh((( 10.0  + data["nitrogen_content"])/2.0  + (data["nitrogen_content"] + data["pesticide_usage"] * -1.)) * ( 9.0 ) / ((data["pesticide_usage"] * -1. + (1659.392822265625000))/2.0 )) +
                0.099951967597008*np.tanh(((((23.343093872070312) + data["region"]) - data["soil_moisture"])) / ((data["soil_moisture"] - (23.343093872070312) * (25.849256515502930)))) +
                0.099997177720070*np.tanh(((data["soil_ph"] - ((data["pesticide_usage"] + ((8.288860321044922) + data["pesticide_usage"])/2.0 )/2.0  +  6.0 )/2.0 )) / ((data["pesticide_usage"] + data["soil_ph"])/2.0  * data["soil_ph"])) +
                0.100000001490116*np.tanh((((6.268942832946777) * 2.0) / (data["fertilizer_amount"]) * -1. + (data["fertilizer_amount"]) / (((2715.945312500000000) - (data["sunlight_hours"] + ((6.268942832946777)) / ((2112.323730468750000)))/2.0 )))/2.0 ) +
                0.099997177720070*np.tanh((((70.183563232421875) - (data["fertilizer_amount"] + (7.367926597595215))/2.0 )) / (((70.183563232421875) * 2.0 - (8.017212867736816) * data["fertilizer_amount"]))) +
                0.099991522729397*np.tanh(((data["potassium_content"] - 1.000000000000000)) / (((0.724937319755554) + ((1.000000000000000) / ((data["potassium_content"] - 1.000000000000000)) + data["avg_temperature"])))) +
                0.099988698959351*np.tanh(((((data["total_rainfall"]) / (data["irrigation_frequency"] / 2.0)) / (data["region"]) - data["irrigation_frequency"] * 2.0 * 2.0)) / (data["total_rainfall"])) +
                0.099997177720070*np.tanh(((data["nitrogen_content"]) / ((data["nitrogen_content"] * data["nitrogen_content"] -  5.0 )) - ((6.379856109619141)) / (data["nitrogen_content"])) *  0.002297401893884 ) +
                0.099997177720070*np.tanh(((data["fertilizer_amount"] - ((173.912139892578125) - data["soil_moisture"]))) / (((data["fertilizer_amount"]) / ((data["soil_moisture"] - (18.340356826782227))) + data["fertilizer_amount"] * (13.330560684204102)))) +
                0.099997177720070*np.tanh(((data["potassium_content"] - ((559.807128906250000)) / ((data["total_rainfall"] + (559.807128906250000) / 2.0)/2.0 ))) / ((273.306610107421875)) * 2.0 * 2.0 * 2.0 * 2.0) +
                0.100000001490116*np.tanh((((((285.628753662109375) - data["fertilizer_amount"]) - data["fertilizer_amount"])) / ((((268.091552734375000) - data["fertilizer_amount"]) + (49.057014465332031))/2.0 )) / (data["fertilizer_amount"] * -1.)) +
                0.099991522729397*np.tanh((((data["phosphorus_content"] - data["phosphorus_content"] * data["fertilizer_amount"]) + ((119.068183898925781) + (8.148440361022949)))/2.0 ) / ( 9.0  * data["fertilizer_amount"] * -1.)) +
                0.099991522729397*np.tanh(((((1598.307128906250000)) / (((2.950643062591553) + data["sunlight_hours"])/2.0 )) / (data["nitrogen_content"])) / ((data["sunlight_hours"] - data["nitrogen_content"] * (1598.307128906250000)))) +
                0.099983051419258*np.tanh(((data["soil_moisture"] - (29.369710922241211))) / ((((1160.449951171875000)) / (((29.369710922241211) - data["soil_moisture"]) * -1.) + ((29.369710922241211) + (1862.609497070312500))/2.0 )/2.0 )) +
                0.099994353950024*np.tanh(((data["soil_ph"] - (7.884884834289551))) / ((data["soil_ph"] - ((6.288680076599121) * data["soil_ph"]) / ((data["soil_ph"] - (5.574391365051270)))))) +
                0.099991522729397*np.tanh((((((data["season"] + (6.266191005706787)) - (33.685565948486328)) + data["avg_temperature"])) / ((data["avg_temperature"] - data["season"] * (6.275362491607666)))) / ((32.845542907714844))) +
                0.099985875189304*np.tanh(((((2580.422851562500000) - data["total_rainfall"]) - data["sunlight_hours"])) / ((data["sunlight_hours"] - data["total_rainfall"] * (454.571441650390625))) * 2.0 * 2.0 * 2.0) +
                0.099968917667866*np.tanh(((((data["soil_moisture"] - ((38.487670898437500)) / (data["potassium_content"])) + (data["potassium_content"]) / ((data["soil_moisture"] - (27.000251770019531))))) / ((27.000251770019531))) / ((38.487670898437500))) +
                0.099983051419258*np.tanh((data["potassium_content"] * (data["potassium_content"] * data["potassium_content"] - (1.640607118606567)) * data["potassium_content"] - ((1.640607118606567)) / (data["potassium_content"])) *  0.005845071282238 ) +
                0.099974565207958*np.tanh(((data["soil_moisture"] * -1.) / ((data["soil_moisture"] * -1. + (data["doy"] * -1. + data["sunlight_hours"])/2.0 )/2.0 )) / ((data["doy"] - (17.123147964477539)))) +
                0.100000001490116*np.tanh(((data["soil_ph"] - (5.445620536804199))) / ((((7.884884834289551) - data["soil_ph"]) + ((8.429489135742188)) / (((7.884884834289551) - data["soil_ph"]) / 2.0)))) +
                0.100000001490116*np.tanh(((data["total_rainfall"] - (698.024658203125000))) / ((155.000000000000000) * (((698.024658203125000)) / ((data["total_rainfall"] * -1. / 2.0 + (299.500000000000000))/2.0 ) + (155.000000000000000))/2.0 )) +
                0.099985875189304*np.tanh(( 0.422852367162704 ) / (((data["sunlight_hours"] - ((2112.323730468750000) - data["sunlight_hours"])) / 2.0 - ((data["sunlight_hours"] +  0.204073235392570 )/2.0  + (1482.576538085937500))/2.0 ))) +
                0.099997177720070*np.tanh(((data["pesticide_usage"] -  8.0 )) / (((data["pesticide_usage"] + (data["region"] - (data["pesticide_usage"]) / ((data["pesticide_usage"] - (6.235155582427979))))) - (119.500000000000000)))) +
                0.099957615137100*np.tanh(((data["potassium_content"] * data["potassium_content"] - data["potassium_content"]) * (data["potassium_content"] +  0.013331892900169 )/2.0  * data["potassium_content"] *  0.013331892900169  -  0.012619021348655 )) +
                0.100000001490116*np.tanh((((data["soil_moisture"] - ((data["soil_moisture"]) / (data["month"])) / (((27.000251770019531) - data["soil_moisture"]))) - (27.000251770019531))) / ((data["soil_moisture"] + (1290.424316406250000))/2.0 )) +
                0.099997177720070*np.tanh((data["season"]) / ((((1354.437500000000000) + ((6.283831119537354) * -1.) / (((6.253936767578125) - data["pesticide_usage"])))/2.0 ) / (( 8.0  - data["pesticide_usage"])))) +
                0.099974565207958*np.tanh(((data["nitrogen_content"] * data["phosphorus_content"] - ((1.593693971633911)) / (data["potassium_content"]))) / ((38.500000000000000))) +
                0.099985875189304*np.tanh((((data["total_rainfall"] - (698.024658203125000))) / ((((data["total_rainfall"] - (384.687500000000000))) / (((1014.747924804687500) - data["total_rainfall"])) + (161.094619750976562))/2.0 )) / ((161.094619750976562))) +
                0.099997177720070*np.tanh((((8.462858200073242) - data["pesticide_usage"])) / (((((6.692811965942383)) / (( 2.0  - data["pesticide_usage"])) + ((6.692811965942383) - data["pesticide_usage"])) + (167.500000000000000)))) +
                0.099977396428585*np.tanh(((((23.435443878173828) - data["avg_temperature"])) / ((23.435443878173828)) * data["avg_temperature"]) / ((((264.500000000000000)) / (((23.435443878173828) - data["avg_temperature"])) + (264.500000000000000)))) +
                0.099983051419258*np.tanh(((978.743408203125000)) / (((data["total_rainfall"] + data["total_rainfall"]) + (((38.487670898437500) - data["total_rainfall"])) / (data["nitrogen_content"])) * ((38.487670898437500) - data["fertilizer_amount"]))) +
                0.099889799952507*np.tanh(((data["phosphorus_content"]) / ((5.445620536804199)) * data["crop_type"]) / (((6.283831119537354)) / ((data["phosphorus_content"] - ((6.262550354003906) - (5.445620536804199)))) * (7.245791435241699))) +
                0.099991522729397*np.tanh((((data["total_rainfall"] + data["fertilizer_amount"]) + data["fertilizer_amount"]) - (942.533630371093750)) * ((data["fertilizer_amount"]) / ((942.533630371093750))) / ((data["fertilizer_amount"] + (1988.114746093750000)))) +
                0.099898278713226*np.tanh(((data["potassium_content"] *  0.002908468944952 ) / ((data["potassium_content"] * data["potassium_content"] - data["potassium_content"])) - (6.224250316619873) *  0.002908468944952 ) *  0.002908468944952 ) +
                0.099977396428585*np.tanh((((data["phosphorus_content"]) / ( 0.357182353734970 ) - (((127.037521362304688) + ((70.183563232421875) + (70.183563232421875)))) / (data["fertilizer_amount"])) * 2.0) / (data["fertilizer_amount"])) +
                0.099980227649212*np.tanh((((6.247834682464600)) / ((((454.571441650390625) * -1.) / (data["total_rainfall"]) - ((data["total_rainfall"] - (312.358825683593750))) / ((6.231836318969727))))) / ((312.358825683593750))) +
                0.100000001490116*np.tanh((((data["total_rainfall"] * 2.0 - (1160.449951171875000) * 2.0) - ((630.205627441406250) - data["sunlight_hours"]))) / ((data["total_rainfall"]) / ( 0.017316345125437 ))) +
                0.099997177720070*np.tanh((((data["potassium_content"] - ((7.367926597595215)) / (data["soil_ph"])) - ((data["potassium_content"]) / ((7.367926597595215))) / ((data["soil_ph"] - (7.367926597595215))))) / ((94.500000000000000))) +
                0.100000001490116*np.tanh((((6.296088218688965)) / ((((data["nitrogen_content"]) / (data["irrigation_frequency"]) + (215.500000000000000))/2.0  - ((215.500000000000000)) / (data["nitrogen_content"])))) / ((119.500000000000000))))


    def GPII(self, data):
        return (6.266543865203857 +
                0.099997177720070*np.tanh((((((data["potassium_content"] + data["potassium_content"])) / (data["avg_temperature"]) -  0.050175677984953 ) - ( 0.050175677984953 ) / ((data["potassium_content"] + data["potassium_content"]))) -  0.051641713827848 )) +
                0.100000001490116*np.tanh( 0.535162329673767  * ( 0.738755404949188  + ((data["potassium_content"] - data["pesticide_usage"]) / 2.0 + ((data["nitrogen_content"] + data["potassium_content"])/2.0  + data["nitrogen_content"])/2.0 )/2.0 )/2.0 ) +
                0.099994353950024*np.tanh((((data["nitrogen_content"] + (-1.000000000000000) / (data["nitrogen_content"]))/2.0  / 2.0 + data["nitrogen_content"])/2.0  + -1.000000000000000)/2.0  / 2.0 / 2.0) +
                0.099963270127773*np.tanh((data["potassium_content"]) / ((( 0.546187520027161  - data["potassium_content"]) + data["month"] * 2.0)/2.0 ) * ((data["potassium_content"] +  0.621677041053772 )/2.0  - 1.000000000000000)) +
                0.100000001490116*np.tanh(((((data["soil_moisture"] -  9.0 ) -  8.0 ) - data["dow"])) / ((data["soil_moisture"] - data["irrigation_frequency"]) *  9.0 )) +
                0.100000001490116*np.tanh(( 0.927028894424438  - (( 0.927028894424438  - (data["nitrogen_content"] + data["nitrogen_content"] / 2.0 / 2.0 / 2.0)/2.0 ) + data["pesticide_usage"] / 2.0)/2.0  / 2.0) / 2.0) +
                0.100000001490116*np.tanh( 0.037346847355366  * (((((data["fertilizer_amount"] - data["region"]) - data["region"]) - 1.000000000000000) - data["avg_temperature"]) + (-1.000000000000000) / ( 0.007531167939305 ))/2.0 ) +
                0.099988698959351*np.tanh(( 0.092675708234310  + (data["nitrogen_content"]) / (((data["nitrogen_content"]) / (( 0.682797133922577  - data["nitrogen_content"])) - data["avg_temperature"]))) * -1.) +
                0.099977396428585*np.tanh(( 0.059815898537636  + ((data["avg_temperature"] + data["avg_temperature"] * 2.0 * 2.0)) / (( 0.012376549653709  - data["sunlight_hours"]))) * 2.0) +
                0.100000001490116*np.tanh((((((( 0.023655897006392 ) / (data["pesticide_usage"]) - data["pesticide_usage"])) / ( 8.0 ) / 2.0 +  8.0 ) - data["pesticide_usage"]) / 2.0) / ( 9.0 )) +
                0.100000001490116*np.tanh(( 0.297099888324738  - ((data["crop_type"] +  0.295949757099152 )) / ((((data["region"] + data["month"]) + (data["total_rainfall"] + data["crop_type"])/2.0 )/2.0 ) / (data["crop_type"])))) +
                0.100000001490116*np.tanh(((((data["irrigation_frequency"] * -1. + data["total_rainfall"]) + data["total_rainfall"]) *  0.029971130192280  + data["fertilizer_amount"]) *  0.029971130192280  - data["month"]) / 2.0) +
                0.099971741437912*np.tanh((((((-1.000000000000000) / (data["potassium_content"]) + (data["potassium_content"] + (data["potassium_content"] + -1.000000000000000))/2.0 )) / (data["doy"])) / (data["doy"])) / (data["doy"])) +
                0.100000001490116*np.tanh((-1.000000000000000 + ((data["pesticide_usage"] / 2.0) / (( 0.626501739025116  -  8.0 )) +  0.029402978718281  * data["fertilizer_amount"] / 2.0)/2.0 ) * 2.0) +
                0.100000001490116*np.tanh((( 8.0  *  4.0  *  5.0  * -1. + data["fertilizer_amount"])) / (data["fertilizer_amount"]) * 2.0) +
                0.100000001490116*np.tanh((( 8.0  *  10.0  - ( 2.0  + data["fertilizer_amount"])/2.0 )) / ((( 10.0  + ( 10.0  + data["fertilizer_amount"])/2.0 )/2.0 ) / (-1.000000000000000))) +
                0.099985875189304*np.tanh((( 0.024171357974410  * 2.0 * -1. * -1.) / ((data["season"] -  5.0 )) -  0.027416713535786 ) * -1.) +
                0.100000001490116*np.tanh((( 6.0  - data["soil_ph"])) / (data["soil_ph"] * (((data["soil_ph"]) / (data["soil_ph"] * 2.0) + data["region"]) - data["soil_ph"] * 2.0))) +
                0.100000001490116*np.tanh(( 10.0 ) / (((data["fertilizer_amount"]) / ( 5.0 ) - data["fertilizer_amount"])) * ( 10.0  - ((data["fertilizer_amount"]) / ( 9.0 ) - data["pesticide_usage"]))) +
                0.100000001490116*np.tanh(((data["potassium_content"]) / ( 6.0 ) + ((data["avg_temperature"] +  7.0 )) / ((data["region"] - (data["avg_temperature"] * 2.0 + data["total_rainfall"])/2.0 ) / 2.0))) +
                0.099997177720070*np.tanh(((data["crop_type"] - data["potassium_content"]) * 2.0) / ((data["crop_type"] * data["crop_type"] - ((data["potassium_content"] * -1. - data["potassium_content"]) + data["total_rainfall"])/2.0 ))) +
                0.100000001490116*np.tanh(( 0.336128324270248  - (( 10.0 ) / ( 0.336128324270248 )) / (((data["total_rainfall"]) / (data["region"] * 2.0 * 2.0) + data["fertilizer_amount"])/2.0 )) * 2.0 * 2.0) +
                0.100000001490116*np.tanh((((data["phosphorus_content"] * data["phosphorus_content"] * (data["soil_moisture"] + data["soil_moisture"])/2.0  + data["soil_moisture"])/2.0  - data["avg_temperature"])) / ( 7.0  * data["soil_moisture"])) +
                0.100000001490116*np.tanh((data["fertilizer_amount"] * ( 0.013230803422630  * ( 10.0 ) / (( 7.0  *  7.0  - data["fertilizer_amount"])) +  0.013230088166893 )/2.0  + -1.000000000000000)) +
                0.099997177720070*np.tanh(( 0.023929124698043  * -1. + ((data["phosphorus_content"]) / (data["avg_temperature"]) * -1. + data["nitrogen_content"] * data["potassium_content"] * (data["phosphorus_content"]) / (data["avg_temperature"])))) +
                0.099997177720070*np.tanh(( 0.177692458033562  + ( 4.0 ) / (((( 0.171231076121330 ) / ((( 4.0 ) / ( 0.174747511744499 ) - data["soil_moisture"])) +  4.0 )/2.0  - data["soil_moisture"])))/2.0 ) +
                0.099988698959351*np.tanh((((data["phosphorus_content"] + data["phosphorus_content"] / 2.0)/2.0  + (-1.000000000000000) / (data["nitrogen_content"]))/2.0 ) / ((data["nitrogen_content"] - data["irrigation_frequency"]) * -1.)) +
                0.099988698959351*np.tanh(( 0.264484465122223  + (data["irrigation_frequency"]) / ((( 0.264484465122223 ) / (((data["soil_moisture"] +  0.264484465122223 )/2.0  - data["irrigation_frequency"])) - data["soil_moisture"])))/2.0 ) +
                0.100000001490116*np.tanh(((( 0.166462704539299  - data["fertilizer_amount"]) *  0.135871917009354  +  8.0  *  3.0 ) * -1.) / (data["irrigation_frequency"] * data["dow"]) * 2.0) +
                0.100000001490116*np.tanh(((((data["irrigation_frequency"] + ( 7.0  - data["avg_temperature"]))/2.0 ) / (data["irrigation_frequency"]) + ( 8.0  - data["pesticide_usage"]))/2.0 ) / ((data["irrigation_frequency"] + data["irrigation_frequency"]))) +
                0.100000001490116*np.tanh(( 0.052585136145353  + ( 7.0  * 2.0) / ((( 7.0  * 2.0 + data["region"]) * 2.0 + data["total_rainfall"])/2.0 ) * -1.) * 2.0 * 2.0) +
                0.100000001490116*np.tanh((( 4.0  - data["pesticide_usage"] / 2.0) / 2.0 + (-1.000000000000000 * 2.0) / ((data["soil_moisture"] -  8.0 )))/2.0  *  0.282940208911896 ) +
                0.100000001490116*np.tanh( 8.0  * (( 10.0 ) / (( 8.0  * 2.0 *  7.0  * -1. + ( 7.0  - data["total_rainfall"]))/2.0 ) +  0.033722169697285 )) +
                0.100000001490116*np.tanh((((data["nitrogen_content"]) / (data["region"] / 2.0) - (data["region"]) / ((data["soil_moisture"] + (data["nitrogen_content"]) / (data["dow"]))/2.0 ))) / (data["dow"])) +
                0.100000001490116*np.tanh((((data["soil_ph"] +  9.0 )/2.0  - data["pesticide_usage"])) / ((( 0.941228151321411  + (data["month"] +  10.0 )) + (data["month"] + data["avg_temperature"])/2.0 ))) +
                0.100000001490116*np.tanh((((data["potassium_content"] - data["pesticide_usage"]) + ( 0.027682550251484 ) / (( 0.035121209919453  - data["pesticide_usage"])) * data["dow"]) + data["season"]) *  0.027682550251484 ) +
                0.100000001490116*np.tanh((( 8.0  + data["pesticide_usage"] * -1.)/2.0 ) / ((( 8.0  + ((data["pesticide_usage"] * -1. +  7.0 )/2.0  + data["month"])/2.0 ) +  8.0 ))) +
                0.099985875189304*np.tanh(( 0.026250129565597  - ( 0.026250129565597  - ((1.000000000000000) / (data["soil_ph"])) / (data["soil_ph"])) * 2.0 * (data["month"] * -1. + data["soil_ph"]))) +
                0.100000001490116*np.tanh(( 0.049323569983244  *  0.049323569983244  * ( 0.049323569983244  * data["fertilizer_amount"] + data["fertilizer_amount"]) - ( 9.0 ) / (data["fertilizer_amount"]) *  7.0 )) +
                0.099963270127773*np.tanh(( 0.047955762594938  -  0.047955762594938  *  0.047955762594938  *  0.047955762594938  * data["avg_temperature"] * ( 0.661189019680023  * data["avg_temperature"] + data["avg_temperature"])/2.0 )) +
                0.100000001490116*np.tanh(((1.000000000000000) / (data["crop_type"]) + (data["pesticide_usage"] *  10.0 ) / (( 10.0  * (data["crop_type"] - data["pesticide_usage"]) - data["total_rainfall"])))) +
                0.100000001490116*np.tanh((-1.000000000000000 +  0.467525839805603  *  0.153349429368973  * ((data["fertilizer_amount"] - (data["fertilizer_amount"]) / (data["soil_moisture"])) *  0.153349429368973  + data["nitrogen_content"])/2.0 )) +
                0.099997177720070*np.tanh(((data["phosphorus_content"] + (data["phosphorus_content"] + ( 5.0  - data["pesticide_usage"]))/2.0 )/2.0 ) / (((data["region"] * -1.) / ( 5.0 ) + data["region"] * 2.0))) +
                0.100000001490116*np.tanh((((data["fertilizer_amount"] + (data["fertilizer_amount"]) / ( 5.0 ))/2.0  - (data["pesticide_usage"] + (data["pesticide_usage"] +  7.0  *  10.0 )))) / (data["fertilizer_amount"])) +
                0.100000001490116*np.tanh( 0.063034787774086  * ( 0.328045696020126  + (((1.000000000000000 - data["sunlight_hours"]) - data["sunlight_hours"] * 2.0) *  0.001162052387372  + data["pesticide_usage"])/2.0  * -1.)/2.0 ) +
                0.100000001490116*np.tanh(( 0.147818595170975  - (data["region"] * 2.0) / (( 7.0  * 2.0 * 2.0 + (data["pesticide_usage"] * -1. + data["fertilizer_amount"]))/2.0 )) * 2.0 * 2.0) +
                0.100000001490116*np.tanh(((data["pesticide_usage"] * -1. + ((data["pesticide_usage"] * -1. + data["soil_moisture"])/2.0  + data["nitrogen_content"] * data["nitrogen_content"])/2.0 )/2.0 ) / ( 3.0 ) *  0.076776280999184 ) +
                0.100000001490116*np.tanh(((( 10.0 ) / (( 10.0  * -1. - (data["total_rainfall"] +  9.0 )/2.0 )) +  0.036780364811420 ) * 2.0 +  0.009029390290380 ) * 2.0) +
                0.100000001490116*np.tanh((-1.000000000000000 / 2.0 +  7.0  * (data["crop_type"]) / (((data["total_rainfall"]) / (data["irrigation_frequency"] * data["irrigation_frequency"]) + data["fertilizer_amount"])/2.0 )) * -1.) +
                0.100000001490116*np.tanh(( 0.762894809246063  + (data["irrigation_frequency"]) / ((data["pesticide_usage"] + ((data["irrigation_frequency"] - data["soil_moisture"] / 2.0) + data["fertilizer_amount"] * -1.)/2.0 )/2.0  / 2.0 / 2.0))/2.0 ) +
                0.099988698959351*np.tanh((((data["potassium_content"] / 2.0 * data["potassium_content"] * data["nitrogen_content"] + -1.000000000000000)/2.0 ) / (data["nitrogen_content"])) / ((data["potassium_content"] / 2.0 + data["crop_type"]))) +
                0.099270984530449*np.tanh(( 0.014396909624338  * -1. + ( 0.105680726468563 ) / (data["region"]))) +
                0.099997177720070*np.tanh(( 0.187676474452019  - ((( 6.0  +  0.187676474452019 )) / (( 9.0  *  9.0  + ( 0.187676474452019  + data["total_rainfall"])/2.0 )/2.0 )) / ( 0.187676474452019 ))) +
                0.100000001490116*np.tanh((((((data["soil_moisture"]) / ( 7.0  / 2.0) + data["fertilizer_amount"])/2.0 ) / ( 7.0 ) / 2.0 - data["dow"])) / (( 6.0  + data["dow"]))) +
                0.100000001490116*np.tanh(((data["potassium_content"] / 2.0 * (data["potassium_content"] - data["nitrogen_content"] / 2.0) + data["nitrogen_content"])/2.0  + -1.000000000000000)/2.0  / 2.0 / 2.0) +
                0.100000001490116*np.tanh(((((data["potassium_content"] - data["pesticide_usage"]) + data["potassium_content"] * data["potassium_content"] * data["potassium_content"])/2.0  + data["phosphorus_content"] * 2.0)/2.0 ) / ( 10.0 ) / 2.0) +
                0.100000001490116*np.tanh( 0.013132336549461  * (data["season"] + (data["phosphorus_content"] * data["phosphorus_content"] - ((data["pesticide_usage"] - ( 0.011198761872947 ) / (data["pesticide_usage"])) - data["phosphorus_content"])))) +
                0.100000001490116*np.tanh(( 0.173702999949455  - (( 8.0  * 2.0 + (data["pesticide_usage"] + ( 10.0  + (data["pesticide_usage"] +  10.0 )/2.0 )/2.0 )/2.0 )) / (data["fertilizer_amount"])) * 2.0) +
                0.099985875189304*np.tanh((data["nitrogen_content"] *  0.050575267523527  + ((data["phosphorus_content"] + (-1.000000000000000 + -1.000000000000000))) / ((-1.000000000000000 + data["soil_moisture"])/2.0 ))/2.0 ) +
                0.100000001490116*np.tanh(((((data["total_rainfall"] + data["soil_moisture"] * data["season"])/2.0 ) / ( 6.0  *  7.0 ) -  10.0 )) / ( 6.0  * data["season"])) +
                0.099988698959351*np.tanh(( 8.0 ) / ((-1.000000000000000 + data["soil_moisture"])/2.0  * (-1.000000000000000 * 2.0 + data["soil_moisture"])/2.0  * (data["soil_moisture"] - data["doy"] / 2.0))) +
                0.099985875189304*np.tanh((((data["soil_moisture"] - data["avg_temperature"] * data["pesticide_usage"]) / 2.0 - (data["avg_temperature"] - data["soil_moisture"] * 2.0) * data["nitrogen_content"])) / (data["sunlight_hours"])) +
                0.099957615137100*np.tanh(data["region"] * ( 0.002546072704718  + data["region"] * ( 0.002546072704718  + (data["region"] * -1.) / (data["sunlight_hours"]))/2.0 )/2.0  * 2.0) +
                0.099991522729397*np.tanh(( 0.007317544892430  * data["nitrogen_content"] + ( 0.293173134326935 ) / (((data["nitrogen_content"] - 1.000000000000000) - data["soil_moisture"]))) * 2.0 * 2.0) +
                0.099971741437912*np.tanh((data["soil_moisture"] * (data["soil_moisture"] *  0.036581285297871  -  0.251496851444244 ) *  0.036581285297871  *  0.036581285297871  -  0.036581285297871 )) +
                0.100000001490116*np.tanh((( 10.0  + ( 9.0  - (data["fertilizer_amount"]) / ( 6.0 )) / 2.0)) / ((data["season"] -  10.0  * 2.0 * 2.0))) +
                0.099960438907146*np.tanh((((-1.000000000000000 - (data["month"]) / (data["soil_moisture"])) + data["potassium_content"])/2.0 ) / (((data["potassium_content"] + -1.000000000000000)/2.0  + data["region"]))) +
                0.099991522729397*np.tanh(((data["nitrogen_content"] - data["pesticide_usage"]) - ((data["pesticide_usage"]) / (( 9.0  - data["soil_ph"])) - data["soil_ph"] * data["potassium_content"])) *  0.005860806908458 ) +
                0.096304044127464*np.tanh(( 0.006865741685033  * -1. +  0.009230615571141  / 2.0) * -1.) +
                0.099983051419258*np.tanh(( 0.004337788559496  * data["potassium_content"] * data["phosphorus_content"] * data["potassium_content"] * data["phosphorus_content"] * data["nitrogen_content"] * 2.0 -  0.019510988146067 )) +
                0.099985875189304*np.tanh((data["season"] * (data["soil_ph"] - data["season"]) * (data["season"] - (data["soil_ph"] - data["nitrogen_content"])) * 2.0) / (data["sunlight_hours"]) * 2.0) +
                0.100000001490116*np.tanh(( 0.021366363391280  + ( 0.021366124972701  + (data["region"]) / (((data["region"]) / (( 5.0  - data["region"])) - data["fertilizer_amount"])))) * data["season"]) +
                0.100000001490116*np.tanh(( 0.087571166455746  - ( 8.0 ) / ((data["season"] + (data["season"] + (data["season"] + ( 0.087571166455746  + data["fertilizer_amount"])/2.0 ))))) * 2.0 * 2.0) +
                0.100000001490116*np.tanh((((data["fertilizer_amount"] - data["avg_temperature"] * 2.0) + (data["irrigation_frequency"] * -1.) / ( 0.049603234976530 ))/2.0 ) / ((data["month"] * 2.0) / ( 0.049603234976530 ))) +
                0.099985875189304*np.tanh((data["potassium_content"] * data["potassium_content"] * data["potassium_content"] * data["potassium_content"] * data["potassium_content"] - data["region"]) *  0.001154422992840  * 2.0) +
                0.100000001490116*np.tanh((( 0.221457064151764  + ((data["avg_temperature"] / 2.0 +  10.0 ) * -1.) / (data["fertilizer_amount"])) - ( 10.0  * 2.0 * 2.0) / (data["total_rainfall"]))) +
                0.100000001490116*np.tanh(((data["crop_type"] - ((data["soil_moisture"] + data["total_rainfall"])/2.0 ) / ( 8.0  *  7.0 ))) / ((data["dow"] - data["month"] * data["crop_type"]))) +
                0.099988698959351*np.tanh((data["potassium_content"] * (-1.000000000000000 +  0.214790150523186  * data["soil_ph"] * data["potassium_content"] * data["potassium_content"])/2.0  - data["potassium_content"]) *  0.024986034259200 ) +
                0.100000001490116*np.tanh((( 10.0  * ( 0.000411510554841  * data["fertilizer_amount"] *  0.009486438706517  * data["fertilizer_amount"] -  5.0 )) / (data["fertilizer_amount"]) +  0.362089246511459 )/2.0 ) +
                0.099991522729397*np.tanh((( 8.0  - (0.000000000000000 + ( 10.0 ) / ((data["soil_moisture"] - data["phosphorus_content"] *  9.0 ))) * 2.0)) / (data["sunlight_hours"])) +
                0.099985875189304*np.tanh(((((data["soil_ph"] + 1.000000000000000) - data["region"])) / ((data["soil_ph"] + (data["soil_ph"] - data["region"])))) / (data["soil_ph"]) * data["potassium_content"]) +
                0.100000001490116*np.tanh( 0.010043623857200  * (data["crop_type"] + ((((data["phosphorus_content"]) / (data["pesticide_usage"]) + data["pesticide_usage"])/2.0  + (data["crop_type"] - data["pesticide_usage"]))/2.0  - data["pesticide_usage"]))/2.0 ) +
                0.100000001490116*np.tanh((( 0.067230716347694  + (data["pesticide_usage"]) / (( 0.195726916193962  - data["fertilizer_amount"]))) + ( 0.067230716347694  + (data["region"] * 2.0) / (data["fertilizer_amount"] * -1.)))) +
                0.100000001490116*np.tanh((-1.000000000000000 * data["potassium_content"] * -1. * 2.0 * 2.0 - ((data["region"]) / (data["nitrogen_content"])) / (data["nitrogen_content"])) *  0.003299475414678 ) +
                0.100000001490116*np.tanh(( 0.081656239926815  - ( 6.0 ) / (((data["fertilizer_amount"] + data["dow"]) + data["nitrogen_content"])/2.0 )) * 2.0) +
                0.100000001490116*np.tanh(((data["soil_ph"] - (data["pesticide_usage"] + ((data["avg_temperature"] - data["soil_ph"])) / ((data["soil_ph"] + -1.000000000000000)/2.0 ))/2.0 )) / (data["soil_ph"] * data["avg_temperature"])) +
                0.099980227649212*np.tanh(((((-1.000000000000000 + -1.000000000000000)/2.0 ) / (data["nitrogen_content"]) / 2.0 + (-1.000000000000000 + data["nitrogen_content"])/2.0 )/2.0 ) / (data["month"]) / 2.0) +
                0.099997177720070*np.tanh(((((data["total_rainfall"]) / (( 7.0  + data["pesticide_usage"])) - ( 7.0  +  10.0 )) -  10.0  * 2.0)) / (data["total_rainfall"])) +
                0.099957615137100*np.tanh((( 0.158100649714470 ) / ((( 0.061151280999184  + data["doy"])/2.0  -  9.0 ))) / (data["doy"] * -1.)) +
                0.100000001490116*np.tanh(( 0.014222387224436  * data["potassium_content"] + ( 0.018388751894236  + ( 8.0  * 2.0) / ((( 8.0  -  9.0 ) - data["total_rainfall"])))) * 2.0) +
                0.099991522729397*np.tanh(((data["soil_moisture"] - ( 10.0 ) / ((data["soil_ph"]) / ( 10.0 )) * 2.0)) / (data["sunlight_hours"]) * 2.0) +
                0.100000001490116*np.tanh((data["fertilizer_amount"] *  0.000765562232118  + ( 9.0 ) / ((( 9.0 ) / ((data["potassium_content"] +  0.000765085394960 )/2.0  * -1.) + data["fertilizer_amount"])/2.0  * -1.))/2.0 ) +
                0.099994353950024*np.tanh((( 0.006203414406627  * data["phosphorus_content"]) / (data["pesticide_usage"])) / (((data["crop_type"] +  0.000091075919045  * data["pesticide_usage"])/2.0  - data["pesticide_usage"])) * data["phosphorus_content"]) +
                0.100000001490116*np.tanh(((((((data["fertilizer_amount"] - data["avg_temperature"]) - data["avg_temperature"])) / ((data["sunlight_hours"]) / (data["total_rainfall"])) - data["avg_temperature"]) -  10.0 )) / (data["total_rainfall"])) +
                0.099994353950024*np.tanh(( 0.020697837695479  - (data["dow"]) / (( 8.0  * ( 0.020697837695479  + (data["dow"] + data["dow"])) + data["total_rainfall"])/2.0 )) * 2.0 * 2.0) +
                0.099971741437912*np.tanh((((data["phosphorus_content"] + data["pesticide_usage"])/2.0 ) / (data["pesticide_usage"]) / 2.0 - (data["pesticide_usage"] -  7.0  * data["phosphorus_content"])) *  0.004620076157153 ) +
                0.100000001490116*np.tanh(( 0.025730138644576  + (data["dow"]) / ((((data["pesticide_usage"] * 2.0 - data["total_rainfall"])) / (data["season"]) - data["fertilizer_amount"]))) * 2.0 * 2.0 * 2.0) +
                0.099994353950024*np.tanh((data["nitrogen_content"] *  2.0  + (((data["potassium_content"] * data["potassium_content"] -  2.0 ) - 1.000000000000000) -  2.0 ))/2.0  *  0.026510482653975 ) +
                0.099957615137100*np.tanh(( 0.007928850129247  - (( 9.0  - ( 0.005457402672619 ) / (( 0.007928850129247  - ( 9.0 ) / (data["sunlight_hours"])) * 2.0))) / (data["sunlight_hours"]))) +
                0.099997177720070*np.tanh((( 9.0  - data["fertilizer_amount"] / 2.0 / 2.0 / 2.0 / 2.0)) / (((data["fertilizer_amount"] / 2.0 -  10.0 ) - data["fertilizer_amount"]))) +
                0.099994353950024*np.tanh(((((( 8.0 ) / ((data["pesticide_usage"] - data["soil_moisture"])) + data["soil_moisture"]) - data["pesticide_usage"]) - data["pesticide_usage"] * 2.0)) / (data["sunlight_hours"]) * 2.0) +
                0.099966093897820*np.tanh((((data["soil_ph"] -  7.0 )) / ((data["soil_ph"] + (( 6.0  - data["soil_ph"])) / ((data["soil_ph"] -  7.0 ))))) / (data["doy"])) +
                0.099991522729397*np.tanh((( 1.0  - (data["avg_temperature"]) / ( 9.0  * 2.0))) / (((data["potassium_content"]) / ((data["soil_moisture"] - data["avg_temperature"])) + data["soil_moisture"]))) +
                0.099994353950024*np.tanh(( 0.056044355034828  + (((data["avg_temperature"] + data["irrigation_frequency"]) + data["irrigation_frequency"])) / (((data["avg_temperature"] + data["avg_temperature"] * 2.0) * -1. - data["total_rainfall"])))) +
                0.099988698959351*np.tanh((((data["sunlight_hours"]) / ((data["pesticide_usage"] * 2.0 + data["avg_temperature"])) * data["potassium_content"] - (data["pesticide_usage"] + data["avg_temperature"]) * 2.0)) / (data["sunlight_hours"])) +
                0.099994353950024*np.tanh( 0.000110864668386  * (data["avg_temperature"] + data["soil_moisture"])/2.0  * (((( 3.0  + data["soil_moisture"])/2.0  - data["avg_temperature"]) - data["avg_temperature"]) + data["soil_moisture"])/2.0 ) +
                0.099983051419258*np.tanh((data["pesticide_usage"]) / ((data["potassium_content"] + (data["pesticide_usage"] + (data["sunlight_hours"] + (data["crop_type"] - data["pesticide_usage"]) * (data["potassium_content"] - data["sunlight_hours"])))/2.0 )/2.0 )) +
                0.099985875189304*np.tanh(((((data["avg_temperature"] - data["total_rainfall"])) / ( 10.0 ) / 2.0 + data["avg_temperature"])/2.0 ) / ((data["soil_moisture"] * -1. - (data["avg_temperature"] + data["total_rainfall"])/2.0 ))) +
                0.099994353950024*np.tanh(((data["season"] - data["region"])) / ((((data["season"] - data["irrigation_frequency"]) *  5.0  + (data["region"] - data["season"])) + data["season"]))) +
                0.099963270127773*np.tanh(( 0.019618039950728  * -1. * ( 0.019618039950728  +  0.019618039950728 )/2.0 ) / ((data["nitrogen_content"] + ((data["pesticide_usage"] + data["nitrogen_content"])/2.0 ) / (data["nitrogen_content"] * -1.))/2.0 )) +
                0.099997177720070*np.tanh(((((((data["month"] + data["total_rainfall"])) / (data["month"])) / ( 6.0 ) -  10.0 ) -  9.0 )) / ((data["sunlight_hours"]) / ( 10.0 ))) +
                0.100000001490116*np.tanh((((data["region"]) / (((data["region"]) / ((data["pesticide_usage"]) / (data["region"])) - data["total_rainfall"])) +  0.006028892006725 ) +  0.006028892006725 ) * 2.0) +
                0.100000001490116*np.tanh(( 0.003292561275885 ) / (((data["pesticide_usage"] * (( 0.004094601608813 ) / (data["pesticide_usage"]) - data["pesticide_usage"]) +  0.011050703935325 ) + data["pesticide_usage"])/2.0 )) +
                0.100000001490116*np.tanh((((-1.000000000000000 + data["potassium_content"]) - ( 0.007549287751317 ) / ((-1.000000000000000 + (data["potassium_content"] -  0.000385761348298 ))))) / (data["avg_temperature"])) +
                0.099997177720070*np.tanh(((((((data["avg_temperature"] * -1. + data["soil_moisture"]) + data["fertilizer_amount"])/2.0 ) / ( 6.0 ) -  6.0 ) -  6.0 )) / (data["fertilizer_amount"])) +
                0.099994353950024*np.tanh(((-1.000000000000000 + ((-1.000000000000000) / (data["nitrogen_content"])) / (data["nitrogen_content"]))/2.0  + data["nitrogen_content"] / 2.0)/2.0  *  0.038211591541767 ) +
                0.099991522729397*np.tanh((( 0.030535705387592  *  0.030535705387592  *  0.030535705387592  * data["total_rainfall"] * data["total_rainfall"] -  10.0 )) / (data["total_rainfall"])) +
                0.099850237369537*np.tanh(( 0.024476295337081 ) / (-1.000000000000000) * ((-1.000000000000000) / (data["irrigation_frequency"]) / 2.0 * data["avg_temperature"] + 1.000000000000000)/2.0  * -1.000000000000000) +
                0.100000001490116*np.tanh(((( 6.0 ) / ((( 9.0  * 2.0 +  0.421570867300034 ) * 2.0 - data["soil_moisture"])) -  9.0 )) / (data["sunlight_hours"])) +
                0.099997177720070*np.tanh(( 0.131655961275101  + ((data["avg_temperature"] + data["pesticide_usage"] * 2.0)/2.0 ) / (((data["avg_temperature"]) / ((data["pesticide_usage"] + data["pesticide_usage"])) - data["fertilizer_amount"])))/2.0 ) +
                0.099994353950024*np.tanh( 0.002874375088140  * (data["sunlight_hours"] *  0.004652977921069  - (( 9.0  - data["sunlight_hours"] *  0.004652977921069 ) + data["pesticide_usage"]))) +
                0.099997177720070*np.tanh(((1.000000000000000) / ((((data["total_rainfall"] +  0.716569125652313 )/2.0 ) / ( 6.0 ) + data["fertilizer_amount"])/2.0 ) -  0.009994748048484 ) * 2.0 * 2.0 * 2.0 * -1.) +
                0.099867194890976*np.tanh((((( 0.021912818774581  +  0.021912818774581 )/2.0 ) / ((-1.000000000000000 + data["potassium_content"] / 2.0)/2.0  * -1.)) / (data["avg_temperature"])) / (data["avg_temperature"]) / 2.0) +
                0.099994353950024*np.tanh(((-1.000000000000000) / ((data["region"] * data["potassium_content"] - data["soil_moisture"] / 2.0)) + data["potassium_content"]) *  0.007298232987523 ) +
                0.099988698959351*np.tanh((-1.000000000000000) / ((((( 9.0  + -1.000000000000000)/2.0 ) / (data["nitrogen_content"])) / ((data["phosphorus_content"] + -1.000000000000000)) + data["fertilizer_amount"])/2.0  * data["nitrogen_content"])) +
                0.099977396428585*np.tanh(( 0.009127857163548  + ( 0.009127857163548  + ( 0.009127857163548  + (((data["potassium_content"] -  7.0 ) -  7.0 )) / (data["total_rainfall"]))))) +
                0.099943488836288*np.tanh((( 0.018397573381662  * data["phosphorus_content"] * data["nitrogen_content"] -  0.012599947862327  * 2.0) * data["phosphorus_content"] * data["phosphorus_content"] -  0.008221628144383 )) +
                0.099977396428585*np.tanh( 0.006898405030370  * data["nitrogen_content"] * (data["nitrogen_content"] -  0.921803951263428  * 2.0) * data["doy"] *  0.004630328156054 ))


    def GPIII(self, data):
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


gp = MyGP()


cbtest['yield_tpha'] = gp.CalculateYield(cbtest)
cbtest[['yield_tpha']].to_csv('gpsubmission.csv')

