import pandas as pd
from ydata_profiling import ProfileReport
!pip install --upgrade numba pandas visions ydata_profiling


df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.info()


df.columns


profile = ProfileReport(df, title="Carvana", 
type_schema = {"IsBadBuy" : "categorical",
"Auction" : "categorical",
"VehYear" : "categorical",
"VehicleAge" : "categorical",
"Make" : "categorical",
"Model" : "categorical",
"Trim" : "categorical",
"SubModel" : "categorical",
"Color" : "categorical",
"Transmission" : "categorical",
"WheelTypeID" : "categorical",
"WheelType" : "categorical",
"Nationality" : "categorical",
"Size" : "categorical",
"TopThreeAmericanName" : "categorical",
"PRIMEUNIT" : "categorical",
"AUCGUART" : "categorical",
"VNST" : "categorical",
"IsOnlineSale": "categorical"})


profile.to_file("your_dataset_profile_report.html")

