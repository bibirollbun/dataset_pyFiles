!pip install pycaret


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import *
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
extra_train=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
sub=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


df


extra_train


sub


df.info()


df.describe().T


df.nunique()


def viz1():
    sns.set_style("whitegrid")
    
    # 1. Brand Distribution (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Brand')
    plt.title('Distribution of Brands')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


viz1()


def viz2():
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Material', y='Price')
    plt.title('Price Distribution by Material')
    plt.xlabel('Material')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


viz2()


def viz3():
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Price', bins=20, kde=True)
    plt.title('Distribution of Prices')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


viz3()


def viz4():
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()


viz4()


def viz5():
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Size', hue='Waterproof')
    plt.title('Size Distribution by Waterproof Feature')
    plt.xlabel('Size')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


viz5()


train = pd.concat([df, extra_train], ignore_index=True)


train


import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")



data=setup(
    data=train,
    target='Price',
    categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'], 
    ignore_features = ['id'],
    use_gpu=True
)


xgb = create_model('xgboost')   
lgbm = create_model('lightgbm') 
cb = create_model('catboost')  


predictions_xgb = predict_model(xgb, data=test)


predictions_lgbm= predict_model(lgbm, data=test)


predictions_cat=predict_model(cb, data=test)


metrics_xgb = pull()

# Get evaluation metrics for LightGBM
metrics_lgbm = pull()

# Get evaluation metrics for CatBoost
metrics_cb = pull()


metrics_xgb


metrics_lgbm


metrics_cb


predictions_xgb


test


def create_submission_file(df, output_path='submission.csv'):
    
    # Select only required columns
    submission_df = df[['id', 'prediction_label']]
    
    # Rename the prediction column to Price
    submission_df = submission_df.rename(columns={'prediction_label': 'Price'})
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission file created successfully at: {output_path}")
    print(f"Shape of submission file: {submission_df.shape}")
    
    # Display first few rows to verify
    print("\nFirst few rows of the submission file:")
    print(submission_df.head())


create_submission_file(predictions_xgb, 'submission.csv')


submission = pd.read_csv('submission.csv')
submission




