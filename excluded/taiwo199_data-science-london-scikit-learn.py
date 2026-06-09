import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")


path = "/kaggle/input/data-science-london-scikit-learn/train.csv"


df = pd.read_csv(path)
df.shape


df.head()

