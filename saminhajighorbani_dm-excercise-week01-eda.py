import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.info


#!pip install --upgrade numba pandas visions ydata_profiling
!pip install ydata_profiling


!pip install ipywidgets==7.6.5


!pip install ipywidgets==8.0.6 --quiet
!jupyter nbextension enable --py widgetsnbextension --sys-prefix


from ydata_profiling import ProfileReport
numeric_df = df.select_dtypes(include=['int64', 'float64'])
categorical_df = df.select_dtypes(include=['object', 'category'])
profile_numeric = ProfileReport(numeric_df, title="Numeric Features EDA", explorative=True)
profile_numeric.to_file("Numeric_EDA_Report.html")
profile_categorical = ProfileReport(categorical_df, title="Categorical Features EDA", explorative=True)
profile_categorical.to_file("Categorical_EDA_Report.html")
#profile = ProfileReport(df, title="car", type_schema = {"IsBadBuy": "categorical", "IsOnlineSale": "categorical"})
# Save the report to an HTML file
#profile.to_file("carvana_profile_report.html")


from IPython.display import IFrame
IFrame(src='Numeric_EDA_Report.html', width=1000, height=600)
from IPython.display import IFrame
IFrame(src='Categorical_EDA_Report.html', width=1000, height=600)


from ydata_profiling import ProfileReport
#Ø¯ÛŒØªØ§Ù�Ø±ÛŒÙ… Ø±Ùˆ Ø¯Ùˆ Ù‚Ø³Ù…Øª Ú©Ø±Ø¯ÛŒÙ… Ø¯Ø± ØªØ³Øª Ùˆ ØªØ±ÛŒÙ† Ù…ÛŒØªÙˆØ§Ù† Ø§Ø² Ø§ÛŒÙ† Ø§Ø³ØªÙ�Ø§Ø¯Ù‡ Ú©Ø±Ø¯ ØªØ§ ØªÙˆØ²ÛŒØ¹ Ø¯Ø§Ø¯Ù‡ Ø¯Ø± Ù‡Ø± Ú©Ø¯Ø§Ù… Ø±Ø§ Ø¯ÛŒØ¯ Ú©Ù‡ Ù…Ø´Ø§Ø¨Ù‡ Ø¨Ø§Ø´Ø¯

df_IsBadBuy_0 = df[df.IsBadBuy == 0]
df_IsBadBuy_1 = df[df.IsBadBuy == 1]

# Generate a profile reportÙ…ÛŒØ®ÙˆØ§Ù‡ÛŒÙ… Ø§Ø² ÙˆÛŒÚ˜Ú¯ÛŒ Ø±ÛŒÙ¾ÙˆØ±Øª Ù¾Ø±ÙˆÙ�Ø§ÛŒÙ„ Ø¨Ø±Ø§ÛŒ Ù…Ù‚Ø§ÛŒØ³Ù‡ Ø§ÛŒÙ† Ø¯Ùˆ Ú¯Ø±ÙˆÙ‡ Ø§Ø³ØªÙ�Ø§Ø¯Ù‡ Ú©Ù†ÛŒÙ…
#lÙˆÙ‚ØªÛŒ Ù…ÛŒÙ†Ø§Ù…Ù„ Ø±Ùˆ ØªØ±Ùˆ Ú¯Ø°Ø§Ø´ØªÛŒÙ… Ø®Ø±ÙˆØ¬ÛŒ Ø§ÛŒÙ†ØªØ±Ø§Ú©ØªÛŒÙˆ Ùˆ Ú©Ø±ÙˆÙ„ÛŒØ´Ù† Ø­Ø°Ù� Ù…ÛŒØ´ÙˆØ¯
profile_0 = ProfileReport(df_IsBadBuy_0, title="EDA 0",minimal=True)
profile_1 = ProfileReport(df_IsBadBuy_1, title="EDA 1",minimal=True)

comparison_report = profile_0.compare(profile_1)
comparison_report.to_file("comparison.html")
from IPython.display import IFrame
IFrame(src="comparison.html", width=1000, height=600)

