import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


!pip install --upgrade numba pandas visions ydata_profiling


pip install --upgrade jupyterlab ipywidgets


from ydata_profiling import ProfileReport  

profile = ProfileReport(df, title="EDA Report", explorative=True)  
profile.to_file("eda_report.html")


 !pip install sweetviz

    # Load Packages
import sweetviz as sv

    # Analyse Dataset
report = sv.analyze(df)

    # View and Save
report.show_html()


pip install --upgrade ipywidgets


from ydata_profiling import ProfileReport

df_0 = df[df.IsBadBuy  == 0]
df_1 = df[df.IsBadBuy  == 1]

# # Generate a profile report
profile_0 = ProfileReport(df_0, title="DGK 0",minimal=True)
profile_1 = ProfileReport(df_1, title="DGK 1",minimal=True)

comparison_report = profile_0.compare(profile_1)
comparison_report.to_file("comparison.html")


import pandas as pd  
from ydata_profiling import ProfileReport  

# Sample dataframe df (assuming df is already created)  
# df = pd.read_csv('your_data.csv')  # load your dataframe  

df_0 = df[df.IsBadBuy == 0]  
df_1 = df[df.IsBadBuy == 1]  

# Generate a profile report  
profile_0 = ProfileReport(df_0, title="DGK 0", minimal=True)  
profile_1 = ProfileReport(df_1, title="DGK 1", minimal=True)  

# Comparison report  
comparison_report = profile_0.compare(profile_1)  

# Extract profile data to DataFrame  
summary_0 = profile_0.get_description()['variables']  
summary_1 = profile_1.get_description()['variables']  

# Combine summaries to compare  
combined_summary = pd.concat([summary_0, summary_1], axis=1, keys=['DGK 0', 'DGK 1'])  

# Save to CSV  
combined_summary.to_csv("comparison_summary.csv")  

# Optional: Save HTML report as well  
comparison_report.to_file("comparison.html")

