#--------------------------------------------
#---- Импортируем библиотеки и загружаем данные
#--------------------------------------------

import numpy as np 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


penguin_df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')

penguin_df


#--------------------------------------------
#---- Смотрим на описательные статистики
#--------------------------------------------

penguin_df.describe()


sns.scatterplot(data=penguin_df, x="culmen_length_mm", y="culmen_depth_mm", hue="sex")



sns.scatterplot(data=penguin_df, x="culmen_length_mm", y="flipper_length_mm", hue="sex")



sns.scatterplot(data=penguin_df, x="culmen_length_mm", y="body_mass_g", hue="sex")



# Уберем лишние значения
penguin_df = penguin_df[penguin_df['sex']!='.'] 
penguin_df = penguin_df[penguin_df['flipper_length_mm'] > 0]

# Гугл сказал, что средняя длина ласт у пингвинов около 110 мм, у нас есть значение 
# 5000 мм. Уберем их за счет фильтра <1000
penguin_df = penguin_df[penguin_df['flipper_length_mm'] < 1000]

# И проверим, что их больше нет
print('Строки с точкой: ', penguin_df[penguin_df['sex']=='.'].shape[0])
print('Cтроки с отрицательным значением длины ласт: ', penguin_df[penguin_df['flipper_length_mm'] <= 0].shape[0])
print('Cтроки с аномальной длиной ласт: ', penguin_df[penguin_df['flipper_length_mm'] >= 1000].shape[0])


kmeans_penguins = penguin_df.copy()

numeric_penguins = kmeans_penguins.select_dtypes(include = 'number')


inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(numeric_penguins)
    inertia.append(kmeans.inertia_)

# Строим график
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Метод локтя")
plt.xlabel("Число кластеров (k)")
plt.ylabel("Сумма внутрикластерных расстояний")
plt.show()

# Оптимальное число кластеров по методу локтя - 4

kmeans = KMeans(n_clusters=4)
kmeans_penguins['labels'] = kmeans.fit_predict(numeric_penguins)


kmeans_penguins


sns.scatterplot(data=kmeans_penguins, x="culmen_length_mm", y="culmen_depth_mm", hue="labels")




sns.scatterplot(data=kmeans_penguins, x="culmen_length_mm", y="body_mass_g", hue="labels")



# !pip install kmodes 

#from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt





df = penguin_df.copy()


num_cols = df.select_dtypes(include = 'number').columns
cat_col = "sex"

df[cat_col] = df[cat_col].astype(str)

# Скейлер позволяет привести все значения к одной шкале
scaler = StandardScaler()
X_num = scaler.fit_transform(df[num_cols])

X = np.concatenate([X_num, df[[cat_col]].values], axis=1)

# Дальше не могу сделать из-за kmodes

