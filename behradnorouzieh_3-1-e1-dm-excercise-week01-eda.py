import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


#!pip install --upgrade numba pandas visions ydata_profiling


!cat "/kaggle/input/DontGetKicked/Carvana_Data_Dictionary.txt"


df.info()


!pip install ydata_profiling


from ydata_profiling import ProfileReport

categorical_fields= ("IsBadBuy","Auction","VehYear","VehicleAge","Make","Model","Trim","SubModel","Color","Transmission",
                     "WheelTypeID","WheelType","Nationality","Size","TopThreeAmericanName","PRIMEUNIT","AUCGUART","BYRNO",
                     "VNZIP1","VNST","IsOnlineSale")

cat_dict = {}
for field in categorical_fields:
    cat_dict[field] = 'categorical'

profile = ProfileReport(df, title="\"Don't Get Kicked!\" data EDA", type_schema = cat_dict)


import numpy as np

def frequency_table(variable):
    
    # Get unique elements and their counts
    unique_elements, counts = np.unique(variable.dropna(), return_counts=True)

    # Calculate percentages
    percentages = (counts / len(variable)) * 100

    # Create a dictionary to store the value counts and percentages
    value_counts_and_percentages = zip(unique_elements, counts, percentages)

    # Print the value counts and percentages
    for i, j, k in value_counts_and_percentages:
        print(f"{i}: Count: {j}, Percentage: {k:.2f}%")
    return


for field in categorical_fields:
    print(f"{field} frequency table:\n")
    frequency_table(df[field])
    print("-"* 30)

