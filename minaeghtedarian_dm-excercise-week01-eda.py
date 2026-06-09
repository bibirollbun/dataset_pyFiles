!pip install --upgrade numba pandas visions ydata_profiling


import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.info()
df.head()


!pip install ydata_profiling


pip list | grep widgets


pip install --upgrade ipywidgets


from ydata_profiling import ProfileReport

profile = ProfileReport(
    df,
    title="Carvana_Data_Dictionary EDA",
    type_schema={
        "IsBadBuy": "categorical",
        "Auction": "categorical",
        "VehYear": "categorical",
        "Make": "categorical",
        "Model": "categorical",
        "Trim": "categorical",
        "SubModel": "categorical",
        "Color": "categorical",
        "Transmission": "categorical",
        "WheelType": "categorical",
        "Nationality": "categorical",
        "Size": "categorical",
        "TopThreeAmericanName": "categorical",
        "PRIMEUNIT": "categorical",
        "AUCGUART": "categorical",
        "VNST": "categorical",
        "IsOnlineSale": "categorical",
    })
profile.to_file("your_dataset_profile_report.html")



df_IsBadBuy_0=df[df["IsBadBuy"]==0]
df_IsBadBuy_1=df[df["IsBadBuy"]==1]

profile_0=ProfileReport(df_IsBadBuy_0,title="Carvana_Data_Dictionary EDA 0",
    type_schema={
        "IsBadBuy": "categorical",
        "Auction": "categorical",
        "VehYear": "categorical",
        "Make": "categorical",
        "Model": "categorical",
        "Trim": "categorical",
        "SubModel": "categorical",
        "Color": "categorical",
        "Transmission": "categorical",
        "WheelType": "categorical",
        "Nationality": "categorical",
        "Size": "categorical",
        "TopThreeAmericanName": "categorical",
        "PRIMEUNIT": "categorical",
        "AUCGUART": "categorical",
        "VNST": "categorical",
        "IsOnlineSale": "categorical",
    })
profile_1=ProfileReport(df_IsBadBuy_1,title="Carvana_Data_Dictionary EDA 1",
    type_schema={
        "IsBadBuy": "categorical",
        "Auction": "categorical",
        "VehYear": "categorical",
        "Make": "categorical",
        "Model": "categorical",
        "Trim": "categorical",
        "SubModel": "categorical",
        "Color": "categorical",
        "Transmission": "categorical",
        "WheelType": "categorical",
        "Nationality": "categorical",
        "Size": "categorical",
        "TopThreeAmericanName": "categorical",
        "PRIMEUNIT": "categorical",
        "AUCGUART": "categorical",
        "VNST": "categorical",
        "IsOnlineSale": "categorical",
    })
comparison_report=profile_0.compare(profile_1)
comparison_report.to_file("comparison.html")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

continuous_columns = [
    "VehYear",
    "VehicleAge",
    "VehOdo",
    "MMRAcquisitionAuctionAveragePrice",
    "MMRAcquisitionAuctionCleanPrice",
    "MMRAcquisitionRetailAveragePrice",
    "MMRAcquisitonRetailCleanPrice",
    "MMRCurrentAuctionAveragePrice",
    "MMRCurrentAuctionCleanPrice",
    "MMRCurrentRetailAveragePrice",
    "MMRCurrentRetailCleanPrice",
    "BYRNO",
    "VNZIP1",
    "VehBCost",
    "WarrantyCost"
]
continuous_data = df[continuous_columns]

correlation_matrix = continuous_data.corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            annot_kws={"size": 8},
            mask=np.triu(np.ones_like(correlation_matrix),k=1),
            cbar_kws={'label': 'Spearman Correlation'})
plt.title('Correlation Matrix Heatmap')
plt.show()


