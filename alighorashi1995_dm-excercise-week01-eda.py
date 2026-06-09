import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df
df.info()


!pip install ydata_profiling


!pip install --upgrade numba pandas visions ydata_profiling


from ydata_profiling import ProfileReport

# Select categorical fields
categorical_fields = ["RefId", "IsBadBuy", "Auction", "VehYear", "Make", "Model", "Trim", "SubModel", "Color", "Transmission", "WheelTypeID", "WheelType", "Nationality", "Size", "TopThreeAmericanName", "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1", "VNST", "IsOnlineSale"]
# Convert to dictionary with values as "categorical"WheelTypeID
type_schema = {field: "categorical" for field in categorical_fields}
# Generate a profile report
profile = ProfileReport(df, title="Carvana data EDA", type_schema = type_schema)

# Save the report to an HTML file
profile.to_file("your_dataset_profile_report.html")


from ydata_profiling import ProfileReport

df_IsBadBuy_0 = df[df["IsBadBuy"] == 0]
df_IsBadBuy_1 = df[df["IsBadBuy"] == 1]

# Generate a profile report
profile_0 = ProfileReport(df_IsBadBuy_0, title="Carvana EDA 0",minimal=True, type_schema = type_schema)
profile_1 = ProfileReport(df_IsBadBuy_1, title="Carvana EDA 1",minimal=True, type_schema = type_schema)

comparison_report = profile_0.compare(profile_1)
comparison_report.to_file("comparison.html")

