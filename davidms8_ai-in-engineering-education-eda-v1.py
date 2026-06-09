import numpy as np 
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud


df = pd.read_csv('/kaggle/input/eda-competition-by-ashok/AI in Engineering Education.csv')
df.head()


df.info()


df.columns.unique()


df = df.rename(columns = {'Timestamp' : 'Time', 'Username' : 'User',
       'Which department of engineering are you affiliated with? ' : 'Department affiliated',
       '  How would you rate your current knowledge of AI?   ' : 'Knowledge AI',
       'Which AI tools do you use?(dalle, midjourney, chapgpt, claude,gemini, grammarly...)' : 'IA Tools Used',
       'How often do you use AI tools?' : 'IA Tools frequency',
       'For what purposes do you use AI tools?' : 'Purposes use IA',
       'On a scale of 0 to 10, how much have AI tools increased your productivity? (0 = not at all, 10 = significantly)' : 'Productivity Scale - IA Tools',
       'On a scale of 0 to 10, how much have AI tools increased your laziness? (0 = not at all, 10 = significantly)' : 'Laziness Scale - IA Tools',
       'Which tasks do you think are not well-solved by AI tools in your experience?(EG. writing original stories)' : 'unsolved tasks AI Tools',
       'Do you think AI will take your job in the future?' : 'Replacing work future - AI',
       'In your opinion, will AI ever take over humans?' : 'IA VS Humans',
       'Do you want to work on developing the best AI? ': 'Developing the best AI',
       'What do you wish AI could do better to make your life easier?' : 'IA - Life easier',
       'Anything you want to share about AI.' : 'Anything you want to share about AI.'})
df


df_da = df['Department affiliated'].value_counts().reset_index()
df_da.columns = ['Department affiliated', 'Cantidad']

fig = px.pie(df_da, values = 'Cantidad', names = 'Department affiliated',
             opacity = 0.85)

fig.update_layout(title_text='Which department of engineering are you affiliated with?', title_x=0.5)
fig.update_traces(marker = dict(line = dict(color = 'black', width = 2.5)))

fig.show()


df_k = df['Knowledge AI'].value_counts().reset_index()
df_k.columns = ['Knowledge AI', 'Cantidad']

fig = px.pie(df_k, values = 'Cantidad', names = 'Knowledge AI',
             opacity = 0.85)

fig.update_layout(title_text='How would you rate your current knowledge of AI? ', title_x=0.5)
fig.update_traces(marker = dict(line = dict(color = 'black', width = 2.5)))

fig.show()



relation_da_k = df.value_counts(subset=['Department affiliated', 'Knowledge AI']).reset_index()
relation_da_k.columns = ['Department affiliated', 'Knowledge AI', 'Cantidad']


fig = px.line_polar(relation_da_k, r = 'Cantidad', theta = 'Department affiliated', line_close = True,
                    color = 'Knowledge AI')
fig.update_traces(fill = 'toself')
fig.update_layout(title_text='Relationship between engineering departments and AI knowledge', title_x=0.5)
fig.show() 


df['IA Tools Used'] = df['IA Tools Used'].str.split(';')

df_exploded = df.explode('IA Tools Used').reset_index(drop=True)

df_u = df_exploded['IA Tools Used'].value_counts().reset_index()
df_u.columns = ['IA Tools Used', 'Cantidad']
df_u


fig = px.bar(df_u, x = 'Cantidad', y = 'IA Tools Used', color = 'IA Tools Used',
             text_auto = True)
fig.update_layout(title_text='Which AI tools do you use?', title_x=0.5)
fig.show()


df_f = df['IA Tools frequency'].value_counts().reset_index()
df_f.columns = ['IA Tools Used', 'Cantidad']

fig = px.pie(df_f, values = 'Cantidad', names = 'IA Tools Used',
             opacity = 0.85)

fig.update_layout(title_text='How often do you use AI tools?', title_x=0.5)
fig.update_traces(marker = dict(line = dict(color = 'black', width = 2.5)))

fig.show()


df['Purposes use IA'] = df['Purposes use IA'].str.split(';')

df_exploded = df.explode('Purposes use IA').reset_index(drop=True)

df_p = df_exploded['Purposes use IA'].value_counts().reset_index()
df_p.columns = ['Purposes use IA', 'Cantidad']
df_p


fig = px.bar(df_p, x = 'Cantidad', y = 'Purposes use IA', color = 'Purposes use IA',
             text_auto = True)
fig.update_layout(title_text='For what purposes do you use AI tools?', title_x=0.5)
fig.show() 



df_s = df['Productivity Scale - IA Tools'].value_counts().reset_index()
df_s.columns = ['Productivity Scale - IA Tools', 'Cantidad']

fig = px.bar(df_s, x='Productivity Scale - IA Tools', y='Cantidad', color='Cantidad', color_continuous_scale='Viridis', title="On a scale of 0 to 10, how much have AI tools increased your productivity? (0 = not at all, 10 = significantly)")

fig.show()


df_l = df['Laziness Scale - IA Tools'].value_counts().reset_index()
df_l.columns = ['Laziness Scale - IA Tools', 'Cantidad']

fig = px.bar(df_l, x='Laziness Scale - IA Tools', y='Cantidad', color='Cantidad', color_continuous_scale='Viridis', title="On a scale of 0 to 10, how much have AI tools increased your laziness? (0 = not at all, 10 = significantly)")

fig.show()


df['unsolved tasks AI Tools'] = df['unsolved tasks AI Tools'].str.split(';')

df_exploded = df.explode('unsolved tasks AI Tools').reset_index(drop=True)

df_ut = df_exploded['unsolved tasks AI Tools'].value_counts().reset_index()
df_ut.columns = ['unsolved tasks AI Tools', 'Cantidad']
df_ut


df_top5 = df_ut.head(6)

fig = px.bar(df_top5, y='unsolved tasks AI Tools', x='Cantidad', title='Which tasks do you think are not well-solved by AI tools in your experience?')

fig.show()


df_r = df['Replacing work future - AI'].value_counts().reset_index()
df_r.columns = ['Replacing work future - AI', 'Cantidad']

fig = px.pie(df_r, values = 'Cantidad', names = 'Replacing work future - AI',
             opacity = 0.85)

fig.update_layout(title_text='Do you think AI will take your job in the future?', title_x=0.5)
fig.update_traces(marker = dict(line = dict(color = 'black', width = 2.5)))

fig.show()


df_d = df['Developing the best AI'].value_counts().reset_index()
df_d.columns = ['Developing the best AI', 'Cantidad']

fig = px.pie(df_d, values = 'Cantidad', names = 'Developing the best AI',
             opacity = 0.85)

fig.update_layout(title_text='Do you want to work on developing the best AI?', title_x=0.5)
fig.update_traces(marker = dict(line = dict(color = 'black', width = 2.5)))

fig.show()


df_e = df['IA - Life easier'].value_counts().reset_index()
df_e.columns = ['IA - Life easier', 'Cantidad']

fig = px.bar(df_e, y = 'Cantidad', x = 'IA - Life easier',
             opacity = 0.85)

fig.update_layout(title_text='What do you wish AI could do better to make your life easier?', title_x=0.5)
fig.update_traces(marker = dict(line = dict(color = 'black', width = 2.5)))

fig.show()


texto = texto_unido = df['Anything you want to share about AI.'].str.cat(sep=', ')

wc = WordCloud(width = 400, height = 300, background_color = "white",
               colormap = "magma")
wc.generate(texto)

plt.axis("off")
plt.imshow(wc, interpolation = "bilinear")

