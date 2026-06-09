import pandas as pd


path_to_ds = '/kaggle/input/31-juli-2025-dig4bio/'

subm_0 = pd.read_csv(path_to_ds + "submission_1_ridge_stacking.csv")
subm_1 = pd.read_csv(path_to_ds + "submission_2_linear_stacking.csv") # <- 0.60906
subm_2 = pd.read_csv(path_to_ds + "submission_3_rf_stacking.csv")
subm_3 = pd.read_csv(path_to_ds + "submission 0.60906.csv") # application does not match!??
subm_4 = pd.read_csv(path_to_ds + "submission 0.37957.csv")
subm_5 = pd.read_csv(path_to_ds + "submission 0.26361.csv")

subm = pd.read_csv("/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv")


# 0.58971

# subm['Glucose'] =  0.0100 * subm_0['Glucose'] +\
#                    0.0100 * subm_1['Glucose'] +\
#                    0.0100 * subm_2['Glucose'] +\
#                    0.9500 * subm_3['Glucose'] +\
#                    0.0100 * subm_4['Glucose'] +\
#                    0.0100 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.0100 * subm_0['Sodium Acetate'] +\
#                    0.0100 * subm_1['Sodium Acetate'] +\
#                    0.0100 * subm_2['Sodium Acetate'] +\
#                    0.9500 * subm_3['Sodium Acetate'] +\
#                    0.0100 * subm_4['Sodium Acetate'] +\
#                    0.0100 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    0.0100 * subm_0['Magnesium Sulfate'] +\
#                    0.0100 * subm_1['Magnesium Sulfate'] +\
#                    0.0100 * subm_2['Magnesium Sulfate'] +\
#                    0.9500 * subm_3['Magnesium Sulfate'] +\
#                    0.0100 * subm_4['Magnesium Sulfate'] +\
#                    0.0100 * subm_5['Magnesium Sulfate']


# 0.58892

subm['Glucose'] =  0.0100 * subm_0['Glucose'] +\
                   0.0100 * subm_1['Glucose'] +\
                   0.0100 * subm_2['Glucose'] +\
                   0.9500 * subm_3['Glucose'] +\
                   0.0100 * subm_4['Glucose'] +\
                   0.0100 * subm_5['Glucose']

subm['Sodium Acetate'] =\
                   0.0200 * subm_0['Sodium Acetate'] +\
                   0.0200 * subm_1['Sodium Acetate'] +\
                   0.0200 * subm_2['Sodium Acetate'] +\
                   0.9000 * subm_3['Sodium Acetate'] +\
                   0.0200 * subm_4['Sodium Acetate'] +\
                   0.0200 * subm_5['Sodium Acetate']

subm['Magnesium Sulfate'] =\
                   0.0100 * subm_0['Magnesium Sulfate'] +\
                   0.0100 * subm_1['Magnesium Sulfate'] +\
                   0.0100 * subm_2['Magnesium Sulfate'] +\
                   0.9500 * subm_3['Magnesium Sulfate'] +\
                   0.0100 * subm_4['Magnesium Sulfate'] +\
                   0.0100 * subm_5['Magnesium Sulfate']


# # 0.58876

# subm['Glucose'] =  0.0100 * subm_0['Glucose'] +\
#                    0.0100 * subm_1['Glucose'] +\
#                    0.0100 * subm_2['Glucose'] +\
#                    0.9500 * subm_3['Glucose'] +\
#                    0.0100 * subm_4['Glucose'] +\
#                    0.0100 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.0200 * subm_0['Sodium Acetate'] +\
#                    0.0200 * subm_1['Sodium Acetate'] +\
#                    0.0200 * subm_2['Sodium Acetate'] +\
#                    0.9000 * subm_3['Sodium Acetate'] +\
#                    0.0200 * subm_4['Sodium Acetate'] +\
#                    0.0200 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    0.0300 * subm_0['Magnesium Sulfate'] +\
#                    0.0300 * subm_1['Magnesium Sulfate'] +\
#                    0.0300 * subm_2['Magnesium Sulfate'] +\
#                    0.8500 * subm_3['Magnesium Sulfate'] +\
#                    0.0300 * subm_4['Magnesium Sulfate'] +\
#                    0.0300 * subm_5['Magnesium Sulfate']


# # 0.58939

# subm['Glucose'] =  0.0001 * subm_0['Glucose'] +\
#                    0.0001 * subm_1['Glucose'] +\
#                    0.0001 * subm_2['Glucose'] +\
#                    0.9850 * subm_3['Glucose'] +\
#                    0.0105 * subm_4['Glucose'] +\
#                    0.0042 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.0001 * subm_0['Sodium Acetate'] +\
#                    0.0001 * subm_1['Sodium Acetate'] +\
#                    0.0001 * subm_2['Sodium Acetate'] +\
#                    0.9850 * subm_3['Sodium Acetate'] +\
#                    0.0105 * subm_4['Sodium Acetate'] +\
#                    0.0042 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    0.0001 * subm_0['Magnesium Sulfate'] +\
#                    0.0001 * subm_1['Magnesium Sulfate'] +\
#                    0.0001 * subm_2['Magnesium Sulfate'] +\
#                    0.9850 * subm_3['Magnesium Sulfate'] +\
#                    0.0105 * subm_4['Magnesium Sulfate'] +\
#                    0.0042 * subm_5['Magnesium Sulfate']


# # 0.58978

# subm['Glucose'] =  0.0001 * subm_0['Glucose'] +\
#                    0.0001 * subm_1['Glucose'] +\
#                    0.0001 * subm_2['Glucose'] +\
#                    0.9995 * subm_3['Glucose'] +\
#                    0.0001 * subm_4['Glucose'] +\
#                    0.0001 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.0001 * subm_0['Sodium Acetate'] +\
#                    0.0001 * subm_1['Sodium Acetate'] +\
#                    0.0001 * subm_2['Sodium Acetate'] +\
#                    0.9995 * subm_3['Sodium Acetate'] +\
#                    0.0001 * subm_4['Sodium Acetate'] +\
#                    0.0001 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    0.0001 * subm_0['Magnesium Sulfate'] +\
#                    0.0001 * subm_1['Magnesium Sulfate'] +\
#                    0.0001 * subm_2['Magnesium Sulfate'] +\
#                    0.9995 * subm_3['Magnesium Sulfate'] +\
#                    0.0001 * subm_4['Magnesium Sulfate'] +\
#                    0.0001 * subm_5['Magnesium Sulfate']


# 0.58978

# subm['Glucose'] =  0.0000 * subm_0['Glucose'] +\
#                    0.0000 * subm_1['Glucose'] +\
#                    0.0000 * subm_2['Glucose'] +\
#                    1.0000 * subm_3['Glucose'] +\
#                    0.0000 * subm_4['Glucose'] +\
#                    0.0000 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.0000 * subm_0['Sodium Acetate'] +\
#                    0.0000 * subm_1['Sodium Acetate'] +\
#                    0.0000 * subm_2['Sodium Acetate'] +\
#                    1.0000 * subm_3['Sodium Acetate'] +\
#                    0.0000 * subm_4['Sodium Acetate'] +\
#                    0.0000 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    0.0000 * subm_0['Magnesium Sulfate'] +\
#                    0.0000 * subm_1['Magnesium Sulfate'] +\
#                    0.0000 * subm_2['Magnesium Sulfate'] +\
#                    1.0000 * subm_3['Magnesium Sulfate'] +\
#                    0.0000 * subm_4['Magnesium Sulfate'] +\
#                    0.0000 * subm_5['Magnesium Sulfate']


# 0.58897

# subm['Glucose'] =  -0.000 * subm_0['Glucose'] +\
#                    -0.000 * subm_1['Glucose'] +\
#                    -0.000 * subm_2['Glucose'] +\
#                    1.005  * subm_3['Glucose'] +\
#                    -0.000 * subm_4['Glucose'] +\
#                    -0.000 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    -0.001 * subm_0['Sodium Acetate'] +\
#                    -0.001 * subm_1['Sodium Acetate'] +\
#                    -0.001 * subm_2['Sodium Acetate'] +\
#                    1.005  * subm_3['Sodium Acetate'] +\
#                    -0.001 * subm_4['Sodium Acetate'] +\
#                    -0.001 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    -0.001 * subm_0['Magnesium Sulfate'] +\
#                    -0.001 * subm_1['Magnesium Sulfate'] +\
#                    -0.001 * subm_2['Magnesium Sulfate'] +\
#                    1.005  * subm_3['Magnesium Sulfate'] +\
#                    -0.001 * subm_4['Magnesium Sulfate'] +\
#                    -0.001 * subm_5['Magnesium Sulfate']


# 0.51135

subm['Glucose'] =  0.0000 * subm_0['Glucose'] +\
                   0.0000 * subm_1['Glucose'] +\
                   1.0000 * subm_2['Glucose'] +\
                   0.0000 * subm_3['Glucose'] +\
                   0.0000 * subm_4['Glucose'] +\
                   0.0000 * subm_5['Glucose']

subm['Sodium Acetate'] =\
                   0.0000 * subm_0['Sodium Acetate'] +\
                   0.0000 * subm_1['Sodium Acetate'] +\
                   1.0000 * subm_2['Sodium Acetate'] +\
                   0.0000 * subm_3['Sodium Acetate'] +\
                   0.0000 * subm_4['Sodium Acetate'] +\
                   0.0000 * subm_5['Sodium Acetate']

subm['Magnesium Sulfate'] =\
                   0.0000 * subm_0['Magnesium Sulfate'] +\
                   0.0000 * subm_1['Magnesium Sulfate'] +\
                   1.0000 * subm_2['Magnesium Sulfate'] +\
                   0.0000 * subm_3['Magnesium Sulfate'] +\
                   0.0000 * subm_4['Magnesium Sulfate'] +\
                   0.0000 * subm_5['Magnesium Sulfate']


# 0.60906

# subm['Glucose'] =  0.0000 * subm_0['Glucose'] +\
#                    1.0000 * subm_1['Glucose'] +\
#                    0.0000 * subm_2['Glucose'] +\
#                    0.0000 * subm_3['Glucose'] +\
#                    0.0000 * subm_4['Glucose'] +\
#                    0.0000 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.0000 * subm_0['Sodium Acetate'] +\
#                    1.0000 * subm_1['Sodium Acetate'] +\
#                    0.0000 * subm_2['Sodium Acetate'] +\
#                    0.0000 * subm_3['Sodium Acetate'] +\
#                    0.0000 * subm_4['Sodium Acetate'] +\
#                    0.0000 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    0.0000 * subm_0['Magnesium Sulfate'] +\
#                    1.0000 * subm_1['Magnesium Sulfate'] +\
#                    0.0000 * subm_2['Magnesium Sulfate'] +\
#                    0.0000 * subm_3['Magnesium Sulfate'] +\
#                    0.0000 * subm_4['Magnesium Sulfate'] +\
#                    0.0000 * subm_5['Magnesium Sulfate']


# # 0.60900

# subm['Glucose'] =  0.0001 * subm_0['Glucose'] +\
#                    0.9947 * subm_1['Glucose'] +\
#                    0.0001 * subm_2['Glucose'] +\
#                    0.0001 * subm_3['Glucose'] +\
#                    0.0030 * subm_4['Glucose'] +\
#                    0.0020 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.0001 * subm_0['Sodium Acetate'] +\
#                    0.9947 * subm_1['Sodium Acetate'] +\
#                    0.0001 * subm_2['Sodium Acetate'] +\
#                    0.0001 * subm_3['Sodium Acetate'] +\
#                    0.0030 * subm_4['Sodium Acetate'] +\
#                    0.0020 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate'] =\
#                    0.0001 * subm_0['Magnesium Sulfate'] +\
#                    0.9947 * subm_1['Magnesium Sulfate'] +\
#                    0.0001 * subm_2['Magnesium Sulfate'] +\
#                    0.0001 * subm_3['Magnesium Sulfate'] +\
#                    0.0030 * subm_4['Magnesium Sulfate'] +\
#                    0.0020 * subm_5['Magnesium Sulfate']


# # 60905

# subm['Glucose'] =\
#                    0.9995 * subm_1['Glucose'] +\
#                    0.0003 * subm_4['Glucose'] +\
#                    0.0002 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.9995 * subm_1['Sodium Acetate'] +\
#                    0.0003 * subm_4['Sodium Acetate'] +\
#                    0.0002 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    0.9995 * subm_1['Magnesium Sulfate'] +\
#                    0.0003 * subm_4['Magnesium Sulfate'] +\
#                    0.0002 * subm_5['Magnesium Sulfate']


# # 0.60905

# subm['Glucose'] =\
#                    0.9997 * subm_1['Glucose'] +\
#                    0.0002 * subm_4['Glucose'] +\
#                    0.0001 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.9997 * subm_1['Sodium Acetate'] +\
#                    0.0002 * subm_4['Sodium Acetate'] +\
#                    0.0001 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    0.9997 * subm_1['Magnesium Sulfate'] +\
#                    0.0002 * subm_4['Magnesium Sulfate'] +\
#                    0.0001 * subm_5['Magnesium Sulfate']


# # 0.60901

# subm['Glucose'] =\
#                    0.997 * subm_1['Glucose'] +\
#                    0.002 * subm_4['Glucose'] +\
#                    0.001 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    0.997 * subm_1['Sodium Acetate'] +\
#                    0.002 * subm_4['Sodium Acetate'] +\
#                    0.001 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    0.997 * subm_1['Magnesium Sulfate'] +\
#                    0.002 * subm_4['Magnesium Sulfate'] +\
#                    0.001 * subm_5['Magnesium Sulfate']


# 0.60916

# subm['Glucose'] =\
#                    1.000 * subm_1['Glucose'] +\
#                    -0.001 * subm_4['Glucose'] +\
#                    +0.001 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    1.000 * subm_1['Sodium Acetate'] +\
#                    -0.001 * subm_4['Sodium Acetate'] +\
#                    +0.001 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    1.000 * subm_1['Magnesium Sulfate'] +\
#                    -0.001 * subm_4['Magnesium Sulfate'] +\
#                    +0.001 * subm_5['Magnesium Sulfate']


# 0.60923

# subm['Glucose'] =\
#                    1.000 * subm_1['Glucose'] +\
#                    -0.0017 * subm_4['Glucose'] +\
#                    +0.0017 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    1.000 * subm_1['Sodium Acetate'] +\
#                    -0.0017 * subm_4['Sodium Acetate'] +\
#                    +0.0017 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    1.000 * subm_1['Magnesium Sulfate'] +\
#                    -0.0017 * subm_4['Magnesium Sulfate'] +\
#                    +0.0017 * subm_5['Magnesium Sulfate']


# 0.61067

# subm['Glucose'] =\
#                    1.000 * subm_1['Glucose'] +\
#                    -0.017 * subm_4['Glucose'] +\
#                    +0.017 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    1.000 * subm_1['Sodium Acetate'] +\
#                    -0.017 * subm_4['Sodium Acetate'] +\
#                    +0.017 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    1.000 * subm_1['Magnesium Sulfate'] +\
#                    -0.017 * subm_4['Magnesium Sulfate'] +\
#                    +0.017 * subm_5['Magnesium Sulfate']


# # 0.61226

# subm['Glucose'] =\
#                    1.000 * subm_1['Glucose'] +\
#                    -0.037 * subm_4['Glucose'] +\
#                    +0.037 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    1.000 * subm_1['Sodium Acetate'] +\
#                    -0.037 * subm_4['Sodium Acetate'] +\
#                    +0.037 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    1.000 * subm_1['Magnesium Sulfate'] +\
#                    -0.037 * subm_4['Magnesium Sulfate'] +\
#                    +0.037 * subm_5['Magnesium Sulfate']


# 0.61442

# subm['Glucose'] =\
#                    1.000 * subm_1['Glucose'] +\
#                    -0.077 * subm_4['Glucose'] +\
#                    +0.077 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    1.000 * subm_1['Sodium Acetate'] +\
#                    -0.077 * subm_4['Sodium Acetate'] +\
#                    +0.077 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    1.000 * subm_1['Magnesium Sulfate'] +\
#                    -0.077 * subm_4['Magnesium Sulfate'] +\
#                    +0.077 * subm_5['Magnesium Sulfate']


# 0.61391

# subm['Glucose'] =\
#                    1.000 * subm_1['Glucose'] +\
#                    -0.177 * subm_4['Glucose'] +\
#                    +0.177 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    1.000 * subm_1['Sodium Acetate'] +\
#                    -0.177 * subm_4['Sodium Acetate'] +\
#                    +0.177 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    1.000 * subm_1['Magnesium Sulfate'] +\
#                    -0.177 * subm_4['Magnesium Sulfate'] +\
#                    +0.177 * subm_5['Magnesium Sulfate']


# # 0.61523

# subm['Glucose'] =\
#                    1.000 * subm_1['Glucose'] +\
#                    -0.122 * subm_4['Glucose'] +\
#                    +0.122 * subm_5['Glucose']

# subm['Sodium Acetate'] =\
#                    1.000 * subm_1['Sodium Acetate'] +\
#                    -0.122 * subm_4['Sodium Acetate'] +\
#                    +0.122 * subm_5['Sodium Acetate']

# subm['Magnesium Sulfate']    =\
#                    1.000 * subm_1['Magnesium Sulfate'] +\
#                    -0.122 * subm_4['Magnesium Sulfate'] +\
#                    +0.122 * subm_5['Magnesium Sulfate']


# ?

subm['Glucose'] =\
                   1.000 * subm_1['Glucose'] +\
                   -0.154 * subm_4['Glucose'] +\
                   +0.121 * subm_5['Glucose']

subm['Sodium Acetate'] =\
                   1.000 * subm_1['Sodium Acetate'] +\
                   -0.154 * subm_4['Sodium Acetate'] +\
                   +0.121 * subm_5['Sodium Acetate']

subm['Magnesium Sulfate']    =\
                   1.000 * subm_1['Magnesium Sulfate'] +\
                   -0.154 * subm_4['Magnesium Sulfate'] +\
                   +0.121 * subm_5['Magnesium Sulfate']


subm.to_csv('submission.csv', index=False)
subm.head(8)

