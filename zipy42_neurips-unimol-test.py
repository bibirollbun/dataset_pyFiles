!pip install /kaggle/input/unimol2-dependencies-air/addict-2.4.0-py3-none-any.whl
!pip install /kaggle/input/unimol2-dependencies-air/rdkit-2025.3.5-cp311-cp311-manylinux_2_28_x86_64.whl


import sys
sys.path.append("/kaggle/input/unimol_tools-include-weights/transformers/default/1/unimol_tools-main")
from unimol_tools import MolTrain, MolPredict


import pandas as pd
import numpy as np
import os


train_df_original = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


# Create separate files for each property
properties = ['Tg', 'FFV', 'Tc']

for prop in properties:
    # Create a new DataFrame with SMILES and the current property
    prop_df = train_df_original[['SMILES', prop]].copy()
    
    # Remove rows where the property is null or empty
    prop_df.dropna(subset=[prop], inplace=True)
    
    # Define the output filename
    output_filename = f'train_{prop}.csv'
    
    # Save the new DataFrame to a csv file
    prop_df.to_csv(output_filename, index=False)
    
    print(f"Created {output_filename} with SMILES and {prop} data.")

import pandas as pd


#将Density与Rg放到一起
filtered_df = train_df_original.dropna(subset=['Density', 'Rg'])
result_df = filtered_df[['SMILES', 'Density', 'Rg']]
result_df.to_csv('train_Density_Rg.csv', index=False)
print("Created train_Density_Rg.csv with SMILES and Density_Rg data.")

print("数据与权重加载成功！")



import pandas as pd

# Load the datasets1
train_tc_df = pd.read_csv('/kaggle/working/train_Tc.csv')
dataset1_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')

# Standardize column names
dataset1_df.rename(columns={'TC_mean': 'Tc'}, inplace=True)

# Concatenate the dataframes
merged_tc_df = pd.concat([train_tc_df, dataset1_df], ignore_index=True)

# Save the merged dataframe back to 'train_Tc.csv', overwriting the original file
merged_tc_df.to_csv('train_Tc.csv', index=False)

print("old 'train_Tc.csv':", len(train_tc_df))
print("Total rows in the updated 'train_Tc.csv':", len(merged_tc_df))

# Load the datasets3
train_tg_df = pd.read_csv('/kaggle/working/train_Tg.csv')
dataset3_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')


# Concatenate the dataframes
merged_tg_df = pd.concat([train_tg_df, dataset3_df], ignore_index=True)

# Save the merged dataframe back to 'train_Tc.csv', overwriting the original file
merged_tg_df.to_csv('train_Tg.csv', index=False)

print("old 'train_Tg.csv':", len(train_tg_df))
print("Total rows in the updated 'train_Tg.csv':", len(merged_tg_df))


# Load the datasets4
train_ffv_df = pd.read_csv('/kaggle/working/train_FFV.csv')
dataset4_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')


# Concatenate the dataframes
merged_ffv_df = pd.concat([train_ffv_df, dataset4_df], ignore_index=True)

# Save the merged dataframe back to 'train_Tc.csv', overwriting the original file
merged_ffv_df.to_csv('train_FFV.csv', index=False)

print("old 'train_FFV.csv':", len(train_ffv_df))
print("Total rows in the updated 'train_FFV.csv':", len(merged_ffv_df))


target_col = 'Density_Rg'
train_df_filtered = pd.read_csv(f'/kaggle/working/train_{target_col}.csv')
print(f"用于训练 '{target_col}' 的样本数量: {len(train_df_filtered)}")

clf_train = MolTrain(
                task='multilabel_regression', 
                target_cols=['Density','Rg'],
                data_type='molecule',
                save_path=f'./unimol_model_{target_col}',
                split='scaffold',
                epochs=80, 
                batch_size=25,  
                kfold=8,
                metrics='r2',
                model_name='unimolv1', # avaliable: unimolv1, unimolv2
                early_stopping=10,
                # batch_size=15, 
                # metrics='r2',
                # model_name='unimolv1', # avaliable: unimolv1, unimolv2
                # model_size='84m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                )
# 使用筛选后的数据进行训练
print(f"开始为 '{target_col}' 训练模型...")
clf_train.fit(train_df_filtered)
print(f"模型 '{target_col}' 训练完成！")


target_col = 'Tc'
train_df_filtered = pd.read_csv(f'/kaggle/working/train_{target_col}.csv')
print(f"用于训练 '{target_col}' 的样本数量: {len(train_df_filtered)}")
clf_train = MolTrain(
                task='regression', 
                target_cols=[target_col],
                data_type='molecule',
                save_path=f'./unimol_model_{target_col}',
                split='scaffold',
                epochs=80, 
                batch_size=20,           
                kfold=8,
                metrics='r2',
                model_name='unimolv1', # avaliable: unimolv1, unimolv2
                early_stopping=10,
                )
# 使用筛选后的数据进行训练
print(f"开始为 '{target_col}' 训练模型...")
clf_train.fit(train_df_filtered)
print(f"模型 '{target_col}' 训练完成！")


target_col = 'Tg'
train_df_filtered = pd.read_csv(f'/kaggle/working/train_{target_col}.csv')
print(f"用于训练 '{target_col}' 的样本数量: {len(train_df_filtered)}")
clf_train = MolTrain(
                task='regression', 
                target_cols=[target_col],
                data_type='molecule',
                save_path=f'./unimol_model_{target_col}',
                split='scaffold',
                epochs=80, 
                batch_size=25,          
                kfold=8,
                metrics='r2',
                model_name='unimolv1', # avaliable: unimolv1, unimolv2
                early_stopping=10,
                )
# 使用筛选后的数据进行训练
print(f"开始为 '{target_col}' 训练模型...")
clf_train.fit(train_df_filtered)
print(f"模型 '{target_col}' 训练完成！")


target_col = 'FFV'
train_df_filtered = pd.read_csv(f'/kaggle/working/train_{target_col}.csv')
print(f"用于训练 '{target_col}' 的样本数量: {len(train_df_filtered)}")
clf_train = MolTrain(
                task='regression', 
                target_cols=[target_col],
                data_type='molecule',
                save_path=f'./unimol_model_{target_col}',
                epochs=30, 
                batch_size=10,
                metrics='r2',
                model_name='unimolv1', # avaliable: unimolv1, unimolv2

                )
# 使用筛选后的数据进行训练
print(f"开始为 '{target_col}' 训练模型...")
clf_train.fit(train_df_filtered)
print(f"模型 '{target_col}' 训练完成！")


import pandas as pd

# 读取测试数据以获取id列
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

submission_df = pd.DataFrame({'id': test_df['id']})

target_columns = ['Tg', 'FFV', 'Tc']
for pre in target_columns:
    print(f"加载模型并为 '{pre}' 进行预测...")

    clf_predict = MolPredict(load_model=f'./unimol_model_{pre}')
    submission_df[pre] = clf_predict.predict(test_df)  
    
    print(f"'{pre}' 预测完成！")
    print("-" * 30)

print("为 'Density' 和 'Rg' 进行预测...")
clf_predict_multi = MolPredict(load_model='./unimol_model_Density_Rg')
predictions = clf_predict_multi.predict(test_df)
submission_df['Density'] = predictions[:, 0]
submission_df['Rg'] = predictions[:, 1]
print("'Density' 和 'Rg' 预测完成！")
print("-" * 30)

submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("\n最终提交文件预览:")
print(submission_df.head())

