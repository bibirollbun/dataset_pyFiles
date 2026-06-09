import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv')
all_df = pd.concat([df, test_df], axis = 0)
all_df.info()


df.head(10)


# Создаем тепловую карту
sns.heatmap(all_df[['Compartments', 'Weight Capacity (kg)', 'Price']].corr(), annot = True)


# Гистограмма
sns.histplot(df['Price'], kde=True)
plt.title("Гистограмма целевой переменной")
plt.show()

sns.boxplot(x=df['Price'])
plt.title("Boxplot для проверки выбросов")
plt.show()


for col in all_df.select_dtypes(include = 'object'):
    all_df[col] = all_df[col].fillna(all_df[col].mode()[0])
for col in all_df.select_dtypes(include = 'float64'):
    if col != 'Price':
        all_df[col] = all_df[col].fillna(all_df[col].mean())


all_df.info()


OHE_all_df = pd.DataFrame()
OHE_all_df['Waterproof'] = all_df['Waterproof'].apply(lambda x: 1 if x == 'Yes' else 0)
OHE_all_df['Laptop Compartment'] = all_df['Laptop Compartment'].apply(lambda x: 1 if x == 'Yes' else 0)


OHE_all_df = pd.concat(
    [
        all_df.drop(['Waterproof', 'Laptop Compartment'], axis = 1),
        OHE_all_df[['Waterproof', 'Laptop Compartment']]
                    ], axis = 1)


OHE_all_df = pd.get_dummies(OHE_all_df, drop_first = True)


# Применяем Z-score Normalization
scaler = StandardScaler()
OHE_all_df[['Weight Capacity (kg)', 'Compartments']] = scaler.fit_transform(OHE_all_df[['Weight Capacity (kg)', 'Compartments']])


X = OHE_all_df[OHE_all_df['id'] < 300000].drop(['Price', 'id'], axis = 1)
y = OHE_all_df[OHE_all_df['id'] < 300000]['Price']


# OHE_all_df.head()


reg = linear_model.LinearRegression()
reg.fit(X, y)


# reg = RandomForestRegressor(n_estimators=600, max_depth=20, random_state=42)
# reg.fit(X, y)


for coef, col in sorted(zip(abs(reg.coef_), X.columns), key = lambda x: x[0], reverse = True):
    print(f"Переменная: {col}, Коэффициент: {coef}")


# plt.figure(figsize=(20, 16))  # Ширина и высота в дюймах

# # Создаем heatmap
# sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm")


X_t = OHE_all_df[OHE_all_df['id'] >= 300000].drop(['Price', 'id'], axis = 1)



OHE_all_df.loc[OHE_all_df['id'] >= 300000, 'Price'] = reg.predict(X_t)


predicted = OHE_all_df[OHE_all_df['id'] >= 300000][['id', 'Price']]
predicted = predicted.set_index('id')
# predicted.to_csv(r'predict.csv')
predicted.to_csv(r'predict_norm.csv')


df2 = pd.read_csv(r'/kaggle/input/playground-series-s5e2/training_extra.csv')


