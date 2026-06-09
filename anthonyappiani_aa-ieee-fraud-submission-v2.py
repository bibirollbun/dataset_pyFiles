import pandas as pd
import joblib


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
submission.head()


test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test = test_id.merge(
    test_trans,on='TransactionID',how='right'
)
# test uses '-' to seperate id cols but train used '_'
test.columns = [col.replace('-', '_') for col in test.columns]
print(test.shape)


import gc
del test_id
del test_trans
gc.collect()


preprocessor = joblib.load('/kaggle/input/ieee-second-models/ieee_processor_v2.joblib')
model = joblib.load('/kaggle/input/ieee-second-models/ieee_final_model_dt_02.joblib')
print(type(model))


cat_features = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain',
               'R_emaildomain','M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo','id_12',
                'id_13','id_14','id_15', 'id_16','id_17','id_18','id_19','id_20','id_21','id_22','id_23','id_24',
               'id_25','id_26','id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36',
                'id_37', 'id_38']
print(len(cat_features))

under_100 = ['ProductCD', 'card4', 'card6', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 
             'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_18', 
             'id_22', 'id_23', 'id_24', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_34', 'id_35', 'id_36', 
             'id_37', 'id_38']
test[under_100] = test[under_100].astype(str)

num_features = [x for x in test.columns if x not in cat_features]

print(len(num_features))

features = num_features + under_100


x_test = preprocessor.transform(test[features])
print(x_test.shape)


test_pred = model.predict(x_test)
print(test_pred.shape)


submission.isFraud = test_pred
submission.head()


submission.to_csv('submission.csv', index = False, header=True)

