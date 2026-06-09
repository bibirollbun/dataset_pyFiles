import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


!pip install --upgrade packaging shapely wurlitzer jupyterlab



!pip install --upgrade numba pandas visions ydata_profiling


df.info()


from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="DontGetKicked",minimal=True, type_schema = {"Auction": "categorical", "Make": "categorical", "Model": "categorical", "Trim": "categorical", "SubModel": "categorical", "VNST": "categorical"})

profile.to_file("your_dataset_profile_report.html")

