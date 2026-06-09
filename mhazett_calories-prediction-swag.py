import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.dropna()
df = df.drop(columns='id')
df = df.rename(columns={'Sex': 'Gender'})
print(df.info())
print('--------------------------------------------------------------------')
print(df.describe())
print('--------------------------------------------------------------------')
print(df.isnull().sum())


plt.figure(figsize=(10,6))
sns.regplot(data=df, x='Duration', y='Calories', line_kws={"color":"red"})
plt.scatter(df['Duration'], df['Calories'])
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.title('Correlation between duration and calories')
plt.grid(True)


plt.figure(figsize=(13,4))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='OrRd')
plt.title('Each data correlation')


plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Gender', y='Weight', palette="Greens", hue='Gender')
plt.grid(True)
plt.title('Data distribution between gender and weight')


df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Age'])

sns.histplot(data=df, x='Age',color='red', kde=True)
plt.grid(True)
plt.title('Age in histogram')


correlation = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation[['Calories']].sort_values(by='Calories', ascending=False), annot=True, cmap='coolwarm')
plt.title('Relationship Between Features and Calories Burned')


X = df[['Duration', 'Heart_Rate', 'Body_Temp']]
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('MSE and R^2 Score for Calorie Prediction Model based on Duration, Heart Rate, and Body Temp:')
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

