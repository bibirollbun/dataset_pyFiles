import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
print("DataFrame Info:")
print(df.info())


from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Carvana Training Data EDA", minimal=False)
profile.to_file("Carvana_training_EDA.html")



from IPython.display import IFrame

IFrame("Carvana_training_EDA.html", width=1000, height=600)



from ydata_profiling import ProfileReport

comparison_report = ProfileReport(df[df.IsBadBuy == 0], 
                                  title="Carvana EDA - Good Buy",
                                  minimal=True,
                                  type_schema={"IsBadBuy": "categorical"}).compare(
                    ProfileReport(df[df.IsBadBuy == 1],
                                  title="Carvana EDA - Bad Buy",
                                  minimal=True,
                                  type_schema={"IsBadBuy": "categorical"}))

comparison_report.to_file("Carvana_Good_vs_Bad_Buy.html")



from IPython.display import IFrame

IFrame("Carvana_Good_vs_Bad_Buy.html", width=1000, height=600)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Continuous Features")
plt.show()


