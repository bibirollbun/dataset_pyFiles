import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
print("Shape:", df.shape)


df.info()


with open('/kaggle/input/carvana-data-dictionary/Carvana_Data_Dictionary.txt', 'r') as f:
    dictionary_text = f.read()

print(dictionary_text[:1000])  # Ù�Ù‚Ø· 1000 Ú©Ø§Ø±Ø§Ú©ØªØ± Ø§ÙˆÙ„Ø´ Ø¨Ø±Ø§ÛŒ Ù†Ù…ÙˆÙ†Ù‡



!pip install ydata_profiling


!pip install ipywidgets==7.7.1
!jupyter nbextension enable --py widgetsnbextension



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

