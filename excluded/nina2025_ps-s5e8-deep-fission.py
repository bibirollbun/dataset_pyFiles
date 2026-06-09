import pandas as pd

path = '/kaggle/input/31-august-2025-ps-s5e8/'


subm_1 = pd.read_csv(path + 'submission Top1.csv')
subm_2 = pd.read_csv(path + 'submission Top2.csv')
subm_3 = pd.read_csv(path + 'submission Top3.csv')
subm_4 = pd.read_csv(path + 'submission Top4.csv')


subm_11 = subm_1.iloc[      0 :  62_500]
subm_12 = subm_1.iloc[ 62_500 : 125_000]
subm_13 = subm_1.iloc[125_000 : 187_500]
subm_14 = subm_1.iloc[187_500 : 250_001]

subm_21 = subm_2.iloc[      0 :  62_500]
subm_22 = subm_2.iloc[ 62_500 : 125_000]
subm_23 = subm_2.iloc[125_000 : 187_500]
subm_24 = subm_2.iloc[187_500 : 250_001]

subm_31 = subm_3.iloc[      0 :  62_500]
subm_32 = subm_3.iloc[ 62_500 : 125_000]
subm_33 = subm_3.iloc[125_000 : 187_500]
subm_34 = subm_3.iloc[187_500 : 250_001]

subm_41 = subm_4.iloc[      0 :  62_500]
subm_42 = subm_4.iloc[ 62_500 : 125_000]
subm_43 = subm_4.iloc[125_000 : 187_500]
subm_44 = subm_4.iloc[187_500 : 250_001]


# subm_1234 = 
# subm_1243 = 
# subm_1324 = 
# subm_1342 =
# subm_1423 = 
# subm_1432 = .. and so on ..

subm_2143 = pd.concat([subm_21, subm_12, subm_43, subm_34], axis=0)
subm_3214 = pd.concat([subm_31, subm_22, subm_13, subm_14], axis=0)
subm_4321 = pd.concat([subm_41, subm_32, subm_23, subm_24], axis=0)
subm_1432 = pd.concat([subm_11, subm_42, subm_33, subm_34], axis=0)

subm_2233 = pd.concat([subm_21, subm_22, subm_33, subm_34], axis=0)

subm_3222 = pd.concat([subm_31, subm_22, subm_23, subm_24], axis=0)

subm_3322 = pd.concat([subm_31, subm_32, subm_23, subm_24], axis=0)


submission = subm_2143 # LB=0.96413
submission = subm_3214 # LB=0.96072

submission = subm_2233 # LB=0.97768

submission = subm_3222 # LB=0.97771

submission = subm_3322 # LB=

submission.to_csv('submission.csv', index=False)

submission

