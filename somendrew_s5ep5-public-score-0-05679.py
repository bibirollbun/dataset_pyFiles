import numpy as np
import pandas as pd


ag = pd.read_csv("/kaggle/input/autoglucon-calories/submission.csv")
avg = pd.read_csv("/kaggle/input/avg-calories/submission.csv")
st = pd.read_csv("/kaggle/input/stacking-calories/submission.csv")
cat = pd.read_csv("/kaggle/input/cat-boostikaran/submission.csv")
one = pd.read_csv("/kaggle/input/tukregaladkibaazi/tukregaladkibaazi.csv")


ag.rename(columns = {'Calories': 'ag'},inplace = True)
avg.rename(columns = {'Calories': 'avg'},inplace = True)
st.rename(columns = {'Calories': 'st'},inplace = True)
cat.rename(columns = {'Calories': 'cat'},inplace = True)
one.rename(columns = {'Calories': 'one'},inplace = True)


df = pd.concat([ag,avg,st,cat,one], axis = 1)
df.head()


df.drop(["id"],axis = 1 , inplace = True)
df.head()


df['avg_st'] = (df['avg']+df['st'])/2
df['all'] = (df['avg']+df['st']+df['ag']+df['cat'])/4
df['st_ag'] = (df['st']+df['ag'])/2
df['st_ag_cat'] = (df['st']+df['ag']+df["cat"])/3



df['one_st_ag_cat'] = (df['st']+df['ag']+df["cat"]+df["one"])/4
df['one_ag'] = (df['one']+df['ag'])/2
df['one_cat'] = (df['one']+df['cat'])/2

df["final"] = (df['one_cat']+df["one_ag"]+df["one_st_ag_cat"])


df.head()


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
print("Submission file saved as .csv!")
submission['Calories'] = df['avg_st']
submission.to_csv('aavg_st.csv', index=False)
print("Submission file saved as .csv!")
submission['Calories'] = df['all']
submission.to_csv('all.csv', index=False)
print("Submission file saved as .csv!")
submission['Calories'] = df['st_ag']
submission.to_csv('st_ag.csv', index=False)
print("Submission file saved as .csv!")

submission['Calories'] = df['st_ag_cat']
submission.to_csv('st_ag_cat.csv', index=False)
print("Submission file saved as .csv!")


submission['Calories'] = df['one_st_ag_cat']
submission.to_csv('one_st_ag_cat.csv', index=False)
print("Submission file saved as .csv!")

submission['Calories'] = df['one_ag']
submission.to_csv('one_ag.csv', index=False)
print("Submission file saved as .csv!")

submission['Calories'] = df['one_cat']
submission.to_csv('one_cat.csv', index=False)
print("Submission file saved as .csv!")

submission['Calories'] = df['final']
submission.to_csv('final.csv', index=False)
print("Submission file saved as .csv!")




