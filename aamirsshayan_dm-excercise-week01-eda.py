import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')



df.info()


!pip install ydata_profiling



from ydata_profiling import ProfileReport

# Generate a profile report
profile = ProfileReport(df, title="Carvana data EDA", type_schema = {
    "RefID": "categorical",
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
    "AUCGUART": "categorical",
    "BYRNO": "categorical",
    "VNZIP1": "categorical",
    "VNST": "categorical",
    "IsOnlineSale": "categorical"
})

# Save the report to an HTML file
profile.to_file("Carvana_profile_report.html")



