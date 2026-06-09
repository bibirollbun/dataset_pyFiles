!pip install sweetviz

import pandas as pd
import numpy as np

import sweetviz as sv
from IPython.display import IFrame


# i/o setting
folder = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
fp = f"{folder}/train_labels.csv"

# read data
df = pd.read_csv(fp)
df


def make_report(df):
    # setting
    sv.config.category_max_cardinality_for_summary_report = 1000
    
    # make report and output
    report = sv.analyze(df)
    report.show_html('report.html')
    
    # show in notebook
    return IFrame('report.html', width=1000, height=600)


# read data
df = pd.read_csv(fp)

# make report
make_report(df)


# read data
df = pd.read_csv(fp)

# filter negative data
df = df[df["Number of motors"]==0]

# make report
make_report(df)


# read data
df = pd.read_csv(fp)

# filter positive data
df = df[df["Number of motors"]!=0]

# make report
make_report(df)


# read data
df = pd.read_csv(fp)

# filter positive data
df = df[df["Number of motors"]==1]

# make report
make_report(df)


# read data
df = pd.read_csv(fp)

# filter positive data
df = df[~df["Number of motors"].isin([0, 1])]

# make report
make_report(df)




