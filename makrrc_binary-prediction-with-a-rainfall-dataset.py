#pip install pandas-profiling


import numpy as np
import pandas as pd


#from ydata_profiling import ProfileReport


import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train


test.shape


train[train['rainfall'] == 0]


train.describe().T


train.info()


#profile = ProfileReport(df, title="Profiling Report")


# Compute the correlation matrix
corr = train.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


features = train.columns


from sklearn.preprocessing import StandardScaler

x = train.loc[:, features].values
x = StandardScaler().fit_transform(x)


x


np.mean(x),np.std(x)


from sklearn.decomposition import PCA
n_components=12
pca_rain = PCA(n_components=n_components)
principalComponents_rain = pca_rain.fit_transform(x)


explained_variance = pca_rain.explained_variance_ratio_
explained_variance


# Simulando um dataset de exemplo
columns = columns = train.columns[1:]

np.random.seed(42)
df = pd.DataFrame(np.random.rand(100, len(columns)), columns=columns)

# Simular "rainfall" com muitos zeros (variável desbalanceada)
df['rainfall'] = np.random.choice([0, 0, 0, 0.5, 1, 2, 5, 10], size=100, p=[0.7, 0.1, 0.1, 0.03, 0.03, 0.02, 0.01, 0.01])

# Normalizar os dados (exceto "rainfall", que é nossa variável-alvo)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(columns=['rainfall']))

# Aplicar PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

# Criar DataFrame com componentes principais
pca_df = pd.DataFrame(principal_components, columns=['PCA1', 'PCA2'])
pca_df['rainfall'] = df['rainfall']

# Criar o biplot destacando "rainfall"
fig, ax = plt.subplots(figsize=(10, 7))

# Definir um esquema de cores baseado na quantidade de chuva
scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['rainfall'], cmap='coolwarm', alpha=0.7)

# Adicionar barra de cores
cbar = plt.colorbar(scatter)
cbar.set_label("Rainfall (mm)")

# Adicionar vetores das variáveis no gráfico
componentes = pca.components_.T
scaling_factor = 5.5  # Ajuste para melhorar visibilidade das setas

for i, col in enumerate(df.columns[:-1]):  # Exclui "rainfall" das setas
    x_vector = componentes[i, 0] * scaling_factor
    y_vector = componentes[i, 1] * scaling_factor
    ax.arrow(0, 0, x_vector, y_vector, color='red', alpha=0.7,
             head_width=0.05, head_length=0.07)
    ax.text(x_vector * 1.2, y_vector * 1.2, col, color='black', fontsize=10, ha='center')

# Configurações do gráfico
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Biplot do PCA - Destacando Rainfall")
ax.grid(True, linestyle='--', alpha=0.6)

plt.show()


train['rainfall'].hist()



df = pd.DataFrame({
    'winddirection': train['winddirection'],  # Direções do vento
    'rainfall': train['rainfall']  # 30% de chance de chover
})

# Criando a variável de ocorrência de chuva (1 se choveu, 0 se não choveu)
df['rain_occurred'] = (df['rainfall'] > 0).astype(int)

# Contar quantos dias chuvosos ocorreram para cada direção do vento
rain_counts = df.groupby('winddirection')['rain_occurred'].sum().sort_values(ascending=False)

# Ordenar direções do vento corretamente
directions_order = train['winddirection']
rain_counts = rain_counts.reindex(directions_order, fill_value=0).sort_values(ascending=False)

# Criar gráfico de barras
plt.figure(figsize=(8, 5))
plt.bar(rain_counts.index.astype(str), rain_counts.values, color='royalblue', alpha=0.7, edgecolor='black')

# Configurações do gráfico
plt.xlabel("Direção do Vento (graus)")
plt.ylabel("Ocorrências de Chuva")
plt.title("Frequência de Ocorrência de Chuvas por Direção do Vento")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Exibir o gráfico
plt.show()


df = pd.DataFrame({
    'humidity': train['humidity'],  # Direções do vento
    'rainfall': train['rainfall']  # 30% de chance de chover
})

# Criando a variável de ocorrência de chuva (1 se choveu, 0 se não choveu)
df['rain_occurred'] = (df['rainfall'] > 0).astype(int)

# Contar quantos dias chuvosos ocorreram para cada direção do vento
rain_counts = df.groupby('humidity')['rain_occurred'].sum().sort_values(ascending=False)

# Ordenar valores de humidade corretamente
directions_order = train['humidity']
rain_counts = rain_counts.reindex(directions_order, fill_value=0).sort_values(ascending=False)

# Criar gráfico de barras
plt.figure(figsize=(8, 5))
plt.bar(rain_counts.index.astype(str), rain_counts.values, color='royalblue', alpha=0.7, edgecolor='black')

# Configurações do gráfico
plt.xlabel("")
plt.ylabel("Ocorrências de Chuva")
plt.title("Frequência de Ocorrência de Chuvas por Humidade")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Exibir o gráfico
plt.show()





df = pd.DataFrame({
    'cloud': train['cloud'],  # Direções do vento
    'rainfall': train['rainfall']  # 30% de chance de chover
})

# Criando a variável de ocorrência de chuva (1 se choveu, 0 se não choveu)
df['rain_occurred'] = (df['rainfall'] > 0).astype(int)

# Contar quantos dias chuvosos ocorreram para cada direção do vento
rain_counts = df.groupby('cloud')['rain_occurred'].sum().sort_values(ascending=False)

# Ordenar direções do vento corretamente
directions_order = train['cloud']
rain_counts = rain_counts.reindex(directions_order, fill_value=0).sort_values(ascending=False)

# Criar gráfico de barras
plt.figure(figsize=(8, 5))
plt.bar(rain_counts.index.astype(str), rain_counts.values, color='royalblue', alpha=0.7, edgecolor='black')

# Configurações do gráfico
plt.xlabel("Direção do Vento (graus)")
plt.ylabel("Ocorrências de Chuva")
plt.title("Frequência de Ocorrência de Chuvas por Direção do Vento")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Exibir o gráfico
plt.show()


train['winddirection'].unique()


train.columns


train['rainfall'].unique()


#14
df = pd.DataFrame({
    'day': train['day'],
    'pressure': train['pressure'],
    'maxtemp': train['maxtemp'],
    'temparature': train['temparature'],
    'mintemp': train['mintemp'],
    'dewpoint': train['dewpoint'],
    'humidity': train['humidity'],
    'cloud': train['cloud'],
    'sunshine': train['sunshine'],
    'winddirection': train['winddirection'],
    'windspeed': train['windspeed'],
    'rainfall': train['rainfall']
    # 'pressure_maxtemp': train['pressure']+train['maxtemp'],
    # 'pressure_winddirection': train['pressure']+train['winddirection'],
    # 'pressure_windspeed': train['pressure']+train['windspeed'],
    # 'pressure_dewpoint': train['pressure']+train['dewpoint']
})



#13
# Lista de colunas para visualizar (exceto 'rainfall')
columns_name = [
    'day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
           'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'
          # 'pressure_maxtemp', 'pressure_winddirection', 'pressure_windspeed','pressure_dewpoint'
]


df1 = pd.DataFrame({})


columns = list(columns)  # transforme em lista


for i, coli in enumerate(columns_name):
    for j, colj in enumerate(columns_name):
        duplo = coli+'_'+colj
        if(
           coli != colj 
           and coli not in colj
           and duplo.count('_') < 2
          ):
            print(coli+'_'+colj)
            df1[coli+'_'+colj] = train[coli]+train[colj]
            columns.append(coli+'_'+colj)

# columns = list(columns)  # transforme em lista

# for coli in columns:
#     for colj in columns:
#         if coli != colj:
#             print(coli+'_'+colj)
#             df1[coli+'_'+colj] = train[coli] + train[colj]
#             columns.append(coli+'_'+colj)
            


columns


columns_test = []
df_test = pd.DataFrame({})


columns_name


for i, coli in enumerate(columns_name):
    for j, colj in enumerate(columns_name):
        duplo = coli+'_'+colj
        print('---'+coli+'_'+colj+'---')
        if(
           coli != colj 
           and coli not in colj
           and duplo.count('_') < 2
          ):
            print(coli+'_'+colj)
            df_test[coli+'_'+colj] = test[coli]+test[colj]
            columns_test.append(coli+'_'+colj)
            


columns_test


df_test


# df1['pressure'+'_'+'maxtemp'] = train['pressure']+train['maxtemp']


df1


columns_test





# df[['pressure_maxtemp','pressure_winddirection']]


df1.dtypes [ df1.dtypes != 'float64' ]


#13
# Lista de colunas para visualizar (exceto 'rainfall')
# columns = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 
#            'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed',
#           'pressure_maxtemp', 'pressure_winddirection']

# print(enumerate(columns))
# exit(0)


columns[1:]


len(columns[1:])



df1 = pd.concat([df1, train['rainfall']], axis=1)
print(df1)


# Definir as colunas a serem plotadas
columns = df1.columns.tolist()
# columns.remove('rainfall')  # Removemos 'rainfall' pois ele será o alvo

# Calcular a grade de subplots dinamicamente
num_vars = len(columns)
ncols = 3  # Número de colunas fixo
nrows = int(np.ceil(num_vars / ncols))  # Número de linhas necessário

# Criar a grade de subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, nrows * 4))
fig.suptitle("Relação das Variáveis com a Ocorrência de Chuva (rainfall)", fontsize=16)

# Garantir que axes seja um array 2D mesmo se houver uma única linha
axes = np.array(axes).reshape(nrows, ncols)

# Criar gráficos
for i, col in enumerate(columns):
    row, col_idx = divmod(i, ncols)  # Definir posição no grid de subplots
    ax = axes[row, col_idx]

    # Criar boxplot para variáveis numéricas
    if df1[col].dtype in ['float64', 'int64']:
        sns.boxplot(x=df1['rainfall'], y=df1[col], ax=ax, palette='coolwarm')
        ax.set_xlabel("Choveu? (0=Não, 1=Sim)")
        ax.set_ylabel(col)
    else:
        # Criar gráfico de barras para variáveis categóricas
        sns.countplot(x=df1[col], hue=df1['rainfall'], ax=ax, palette='coolwarm')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.set_title(f"{col} vs Rainfall")

# Remover subplots vazios (se houver)
for i in range(num_vars, nrows * ncols):
    fig.delaxes(axes.flatten()[i])

# Ajustar espaçamentos
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Exibir o gráfico
plt.show()


train[train['day'] == 365]


test[test['day'] == 365]


qtde_day_year = 365
qtde_year = ( train['id'].count() / qtde_day_year )
qtde_year


qtde_day_year = 365
qtde_year = ( test['id'].count() / qtde_day_year )
qtde_year


train[ (train['id'] >= 1450) & (train['id'] <= 1460) ]


##asdasdad


train[ (train['id'] == 1452) ]


train.loc[ (train['id'] == 1452), 'day' ] = train[ (train['id'] == 1451) ]['day'].values[0] + 1
train.loc[ (train['id'] == 1453), 'day' ] = train[ (train['id'] == 1452) ]['day'].values[0] + 1
train.loc[ (train['id'] == 1457), 'day' ] = train[ (train['id'] == 1456) ]['day'].values[0] + 1
train.loc[ (train['id'] == 1458), 'day' ] = train[ (train['id'] == 1457) ]['day'].values[0] + 1
train.loc[ (train['id'] == 1459), 'day' ] = train[ (train['id'] == 1458) ]['day'].values[0] + 1


train[ (train['id'] >= 1450) & (train['id'] <= 1460) ]


def generateYearColumn(df, day):
    year = 1
    ponteiro = 0
    for value in df[day]:
        df.loc[ ponteiro, 'year' ] = year
        if value == 365:
            year+=1
        ponteiro+=1
        
    return df
        


def generateMonthColumn(df, day):
    # Criando a coluna month
    df['month'] = pd.to_datetime(df[day], format='%j').dt.month
    return df


train = generateYearColumn(train, 'day')


train = generateMonthColumn(train, 'day')


train


test


test = generateYearColumn(test, 'day')


test = generateMonthColumn(test, 'day')


test


train[train['day']==32]


train.groupby('year').count()


train[ (train['year'] == 4) & (train['day'] == 4) ]


# Group by year and month and calculate the mean of rainfall.
df_grouped = train.groupby(['year', 'month'])['rainfall'].mean().reset_index()

# Create the line chart
plt.figure(figsize=(10, 6))
sns.lineplot(x='month', y='rainfall', hue='year', marker='o', data=df_grouped, palette='tab10')

# Configure the chart
plt.xlabel('Mês')
plt.ylabel('Precipitação Média (Rainfall)')
plt.title('Comparação da Precipitação Média Mensal em Diferentes Anos')
plt.xticks(range(1, 13))  # Mostrar os meses de 1 a 12
plt.legend(title='Ano')
plt.grid()

# Display the graph
plt.show()


# Group by year and month and calculate the sum of rainfall ( cumulated).
df_grouped = train.groupby(['year', 'month'])['rainfall'].sum().reset_index()

#Create the graphic of line
plt.figure(figsize=(10, 6))
sns.lineplot(x='month', y='rainfall', hue='year', marker='o', data=df_grouped, palette='tab10')

# Configurate the graphic
plt.xlabel('Mês')
plt.ylabel('Precipitação Média (Rainfall)')
plt.title('Comparação da Precipitação Média Mensal em Diferentes Anos')
plt.xticks(range(1, 13))  # Mostrar os meses de 1 a 12
plt.legend(title='Ano')
plt.grid()

#Show the graphic
plt.show()


train


train['rainfall'].hist()


colunm_sub = ['pressure_cloud', 'maxtemp_cloud', 'temparature_cloud', 'mintemp_cloud', 'dewpoint_cloud', 
'humidity_cloud', 'cloud_pressure', 'cloud_mintemp', 'cloud_temparature', 'cloud_mintemp',
'sunshine_cloud', 'windspeed_cloud','id','rainfall']


df_test = pd.concat([df_test,test['id']], axis=1)
df_test


df1 = pd.concat([df1, train[['id','rainfall']]], axis=1)
df1


# submission = train[['id','pressure','temparature','cloud','sunshine', 'windspeed','month','year', 'rainfall']]
submission = df1[colunm_sub]
submission


colunm_sub[:-1]


df_test[colunm_sub[:-1]]


submission_test = df_test[colunm_sub[:-1]]
submission_test


test


test.isnull().sum()


test[test['winddirection'].isnull()]


test[test['pressure'] == 1007.8]


test.loc[ ( test['id'] == 2707 ), 'winddirection'] = 230.0


test[test['id'] == 2707]


submission.to_csv('eda_variation.csv')
print('eda binary_prediction_variation_rainfall v1')


# test.to_csv('result/eda_test.csv')
# print('eda test binary_prediction_rainfall v1')


submission_test.to_csv('cross_variation_eda_test.csv')
print('eda test cross_variation_binary_prediction_rainfall v1')

