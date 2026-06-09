pip install ydata-profiling


import warnings
warnings.filterwarnings("ignore")
import pandas as pd


file_path = "/kaggle/input/playground-series-s5e11/train.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)


from ydata_profiling import ProfileReport

# Create and save the profiling report
profile = ProfileReport(df, explorative=True)
profile.to_file("ydata_profiling_report.html")



profile.to_notebook_iframe()




