import pandas as pd
import plotly.express as px

metadata_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'  
df = pd.read_csv(metadata_path,  low_memory=False)




fig = px.pie(df, names='target', title='Distribution of Benign vs Malignant Cases', hole=0.3)
fig.show()




fig = px.bar(df['target'].value_counts(), x=df['target'].unique(), y=df['target'].value_counts(),
             title="Target Class Count (Benign vs Malignant)",
             labels={'x': 'Target (0=Benign, 1=Malignant)', 'y': 'Count'})
fig.show()



fig = px.bar(df['target'].value_counts(), 
             x=df['target'].unique(), 
             y=df['target'].value_counts(), 
             title="Target Class Count (Benign vs Malignant) - Log Scale",
             labels={'x': 'Target (0=Benign, 1=Malignant)', 'y': 'Count'},
             log_y=True)
fig.show()



fig = px.bar(df, x=df['iddx_1'].value_counts().index, y=df['iddx_1'].value_counts().values,
             title="First Level Lesion Diagnosis Count", labels={'x': 'iddx_1', 'y': 'Count'})
fig.show()



fig = px.bar(df, x=df['iddx_1'].value_counts().index, y=df['iddx_1'].value_counts().values,
             title="First Level Lesion Diagnosis Count", labels={'x': 'iddx_1', 'y': 'Count'},log_y = True)
fig.show()



counts = df['iddx_1'].value_counts()
print(counts)
# Create bar plot
fig = px.bar(x=counts.index,y= counts.values,
             title="1st Level Lesion Diagnosis Count",
             labels={'x': 'iddx_1', 'y': 'Count'},log_y=True)

fig.show()


counts = df['iddx_2'].value_counts()
print(counts)
# Create bar plot
fig = px.bar(x=counts.index,y= counts.values,
             title="2nd Level Lesion Diagnosis Count",
             labels={'x': 'iddx_2', 'y': 'Count'})

fig.show()


counts = df['iddx_3'].value_counts()
print(counts)
# Create bar plot
fig = px.bar(x=counts.index,y= counts.values,
             title="3rd Level Lesion Diagnosis Count",
             labels={'x': 'iddx_3', 'y': 'Count'})

fig.show()


counts = df['iddx_4'].value_counts()
print(counts)
# Create bar plot
fig = px.bar(x=counts.index,y= counts.values,
             title="4th Level Lesion Diagnosis Count",
             labels={'x': 'iddx_4', 'y': 'Count'})

fig.show()


counts = df['iddx_5'].value_counts()
print(counts)
# Create bar plot
fig = px.bar(x=counts.index,y= counts.values,
             title="5th Level Lesion Diagnosis Count",
             labels={'x': 'iddx_5', 'y': 'Count'})

fig.show()


fig = px.histogram(df, x='age_approx', color='target', barmode='group',
                   title='Distribution of Target Class Across Age Groups')
fig.show()



fig = px.imshow(df.groupby(['iddx_1', 'iddx_2']).size().unstack(),
                labels=dict(x="Second Level", y="First Level", color="Count"),
                title="Lesion Diagnosis Heatmap ")
fig.show()



import plotly.express as px

diagnosis_by_site = df.groupby(['iddx_1', 'anatom_site_general']).size().reset_index(name='count')
print(diagnosis_by_site)
fig = px.bar(diagnosis_by_site, x='anatom_site_general', y='count', color='iddx_1',
             title='Diagnosis by Body Site', labels={'x': 'Anatomical Site', 'y': 'Count'})
fig.show()



import plotly.express as px

diagnosis_by_site = df.groupby(['iddx_2', 'anatom_site_general']).size().reset_index(name='count')
print(diagnosis_by_site)
fig = px.bar(diagnosis_by_site, x='anatom_site_general', y='count', color='iddx_2',
             title='Diagnosis by Body Site', labels={'x': 'Anatomical Site', 'y': 'Count'})
fig.show()



import plotly.express as px

diagnosis_by_site = df.groupby(['iddx_3', 'anatom_site_general']).size().reset_index(name='count')
print(diagnosis_by_site)
fig = px.bar(diagnosis_by_site, x='anatom_site_general', y='count', color='iddx_3',
             title='Diagnosis by Body Site', labels={'x': 'Anatomical Site', 'y': 'Count'})
fig.show()



import plotly.express as px

diagnosis_by_site = df.groupby(['iddx_4', 'anatom_site_general']).size().reset_index(name='count')
print(diagnosis_by_site)
fig = px.bar(diagnosis_by_site, x='anatom_site_general', y='count', color='iddx_4',
             title='Diagnosis by Body Site', labels={'x': 'Anatomical Site', 'y': 'Count'})
fig.show()



import plotly.express as px

diagnosis_by_site = df.groupby(['iddx_5', 'anatom_site_general']).size().reset_index(name='count')
print(diagnosis_by_site)



