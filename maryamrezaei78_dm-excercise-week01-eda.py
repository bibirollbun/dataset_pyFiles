import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


#!pip install ydata_profiling
#!pip install --upgrade numba pandas visions ydata_profiling
#!pip install ipywidgets==7.6.5


df.info()


from ydata_profiling import ProfileReport

# Generate a profile report
profile = ProfileReport(df, 
                        title="Don't Get Kicked! data EDA",
                        type_schema={
                            "IsBadBuy": "categorical",
                            "Auction": "categorical", 
                            "Make": "categorical", 
                            "Model": "categorical",
                            "Trim": "categorical",
                            "SubModel": "categorical",
                            "Color": "categorical",
                            "Transmission": "categorical",
                            "WheelTypeID": "categorical",
                            "WheelType": "categorical",
                            "Nationality": "categorical",
                            "Size": "categorical",
                            "TopThreeAmericanName": "categorical",
                            "PRIMEUNIT": "categorical",
                            "AcquisitionType": "categorical",
                            "AUCGUART": "categorical",
                            "VNST": "categorical",
                            "IsOnlineSale": "categorical"
                        },
                       explorative = True)

# Save the report to an HTML file
#profile.to_file("your_dataset_profile_report.html")


profile.to_notebook_iframe()


from ydata_profiling import ProfileReport

df_IsBadBuy_0 = df[df['IsBadBuy'] == 0]
df_IsBadBuy_1 = df[df['IsBadBuy'] == 1]


# Generate a profile report
profile_0 = ProfileReport(df_IsBadBuy_0, title="Don't Get Kicked! EDA 0", minimal=True,type_schema={
                            "IsBadBuy": "categorical",
                            "Auction": "categorical", 
                            "Make": "categorical", 
                            "Model": "categorical",
                            "Trim": "categorical",
                            "SubModel": "categorical",
                            "Color": "categorical",
                            "Transmission": "categorical",
                            "WheelTypeID": "categorical",
                            "WheelType": "categorical",
                            "Nationality": "categorical",
                            "Size": "categorical",
                            "TopThreeAmericanName": "categorical",
                            "PRIMEUNIT": "categorical",
                            "AcquisitionType": "categorical",
                            "AUCGUART": "categorical",
                            "VNST": "categorical",
                            "IsOnlineSale": "categorical"
                        })
profile_1 = ProfileReport(df_IsBadBuy_1, title="Don't Get Kicked! EDA 1", minimal=True, type_schema={
                            "IsBadBuy": "categorical",
                            "Auction": "categorical", 
                            "Make": "categorical", 
                            "Model": "categorical",
                            "Trim": "categorical",
                            "SubModel": "categorical",
                            "Color": "categorical",
                            "Transmission": "categorical",
                            "WheelTypeID": "categorical",
                            "WheelType": "categorical",
                            "Nationality": "categorical",
                            "Size": "categorical",
                            "TopThreeAmericanName": "categorical",
                            "PRIMEUNIT": "categorical",
                            "AcquisitionType": "categorical",
                            "AUCGUART": "categorical",
                            "VNST": "categorical",
                            "IsOnlineSale": "categorical"
                        })

comparison_report = profile_0.compare(profile_1)
comparison_report.to_file("comparison.html")


comparison_report.to_notebook_iframe()


numerical_columns = [
    'VehYear', 'VehicleAge', 'VehOdo', 
    'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',
    'VehBCost', 'WarrantyCost'
]


import seaborn as sns
import matplotlib.pyplot as plt

# Subset the dataframe to only include your numerical columns
numerical_df = df[numerical_columns]

# Compute correlation matrix
corr_matrix = numerical_df.corr()

# Plot the heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

