import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.head()



df.info()


!pip install --upgrade ydata-profiling


!pip install ipywidgets --upgrade



rom ydata_profiling import ProfileReport

profile = ProfileReport(
    df,
    title="car buying EDA",
    correlations={"auto": {"calculate": False}},  # جلوگیری از ارور مربوط به autocorrelation
    type_schema={
        "Auction": "categorical", "IsBadBuy": "categorical",
        "Make": "categorical", "Model": "categorical",
        "Trim": "categorical", "SubModel": "categorical",
        "Color": "categorical", "Transmission": "categorical",
        "WheelType": "categorical", "Nationality": "categorical",
        "Size": "categorical", "TopThreeAmericanName": "categorical",
        "PRIMEUNIT": "categorical", "AUCGUART": "categorical", "VNST": "categorical"
    }
)

profile.to_file("your_dataset_profile_report.html")


profile.to_notebook_iframe()


from ydata_profiling import ProfileReport

df_default_0 = df[df.IsBadBuy == 0].copy()
df_default_1 = df[df.IsBadBuy == 1].copy()

# Generate a profile report
profile_0 = ProfileReport(df_default_0, title="Carvana Data EDA 0",minimal=True,
                                                      type_schema = {"IsBadBuy": "categorical","PurchDate": "categorical", "Auction": "categorical",
                                                                     "Make": "categorical","Model": "categorical", "Trim": "categorical",
                                                                     "SubModel": "categorical","Color": "categorical", "Transmission": "categorical",
                                                                     "WheelTypeID": "categorical", "WheelType": "categorical","Nationality": "categorical", 
                                                                     "Size": "categorical", "TopThreeAmericanName": "categorical","PRIMEUNIT": "categorical", 
                                                                     "AUCGUART": "categorical", "BYRNO": "categorical", "VNZIP1": "categorical",
                                                                     "VNST": "categorical", "IsOnlineSale": "categorical"})

profile_1 = ProfileReport(df_default_1, title="Carvana Data EDA 1",minimal=True, 
                                                      type_schema = {"IsBadBuy": "categorical","PurchDate": "categorical", "Auction": "categorical",
                                                                     "Make": "categorical","Model": "categorical", "Trim": "categorical",
                                                                     "SubModel": "categorical","Color": "categorical", "Transmission": "categorical",
                                                                     "WheelTypeID": "categorical", "WheelType": "categorical","Nationality": "categorical", 
                                                                     "Size": "categorical", "TopThreeAmericanName": "categorical","PRIMEUNIT": "categorical", 
                                                                     "AUCGUART": "categorical", "BYRNO": "categorical", "VNZIP1": "categorical",
                                                                     "VNST": "categorical", "IsOnlineSale": "categorical"})

comparison_report = profile_0.compare(profile_1)
comparison_report.to_file("Carvana_dataset_comparison.html")

