import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


##1.
df.info()



!pip install ydata_profiling


!pip install --upgrade numba pandas visions ydata_profiling


from ydata_profiling import ProfileReport

#Genrate a profile reporr

profile=ProfileReport(df,title="Carnavas data EDA")

#savw as HTML File

profile.to_file("your_dataswt_profile_report_html")

