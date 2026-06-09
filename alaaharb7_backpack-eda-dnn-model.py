import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid",font_scale=1.3)
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping


train=pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra=pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.shape, train_extra.shape, test.shape


train = pd.concat([train, train_extra], ignore_index=True)


train.shape


train.head()


train.info()


train.isna().sum()


categorical_cols = train.select_dtypes(include = 'O').columns
numerical_cols = train.select_dtypes(include = 'number').columns
print(f'categorical Columns => {categorical_cols}')
print(f'Numerical Columns => {numerical_cols}')


train['Brand'].value_counts()


plt.figure(figsize = (15, 6))
ax = sns.countplot(data = train, x = 'Brand', palette = 'viridis')
for container in ax.containers:
    ax.bar_label(container, fontweight = 'black')

plt.title('Back Brand Distribution', size = 20, pad = 10)
plt.show()


train['Material'].value_counts()


plt.figure(figsize = (15, 6))
plt.subplot(1,2, 1)
sns.countplot(data = train, x = 'Material', palette = 'viridis')
plt.title("Material Distribution", fontweight="black",size=20,pad=20)

plt.subplot(1,2, 2)
plt.title('Back Brand Distribution', size = 20, pad = 10)
plt.pie(train['Material'].value_counts(), autopct = '%1.1f%%', labels = train['Material'].value_counts().index, explode = [0.1,0,0, 0], textprops={"fontweight":"black"})
plt.title("Pie Plot Material",fontweight="black",size=20,pad=20)

plt.show()


train['Size'].value_counts()


plt.figure(figsize = (12, 6))
plt.pie(train['Size'].value_counts(), autopct = '%1.1f%%', labels = train['Size'].value_counts().index, explode = [0.1,0,0], textprops={"fontweight":"black"})
plt.title("Pie Plot Size",fontweight="black",size=20,pad=20)


train['Compartments'].value_counts()


plt.figure(figsize = (15, 8))
train['Compartments'].value_counts().plot(kind = 'bar')
plt.title('Distribution Of Compartments',fontweight="black",size=20,pad=20)
plt.xticks(rotation = 0)
plt.show()


plt.figure(figsize = (15, 6))
plt.subplot(1,2, 1)
plt.pie(train['Laptop Compartment'].value_counts(), autopct = '%1.1f%%', labels = train['Laptop Compartment'].value_counts().index, explode = [0.1,0], textprops={"fontweight":"black"})
plt.title("Pie Plot Laptop Compartment",fontweight="black",size=20,pad=20)

plt.subplot(1,2, 2)
plt.pie(train['Waterproof'].value_counts(), autopct = '%1.1f%%', labels = train['Waterproof'].value_counts().index, explode = [0.1,0], textprops={"fontweight":"black"})
plt.title("Pie Plot of Waterproof",fontweight="black",size=20,pad=20)

plt.show()


train['Style'].value_counts()


train['Color'].value_counts()


plt.figure(figsize = (15, 6))
plt.subplot(1,2, 1)
ax=sns.countplot(data = train, x = 'Color', palette = 'viridis')
for container in ax.containers:
    ax.bar_label(container, fontweight = 'black')
plt.title("Color Distribution", fontweight="black",size=15,pad=20)

plt.subplot(1,2, 2)
plt.pie(train['Style'].value_counts(), autopct = '%1.1f%%', labels = train['Style'].value_counts().index, explode = [0.1,0,0], textprops={"fontweight":"black"})
plt.title("Pie Plot of Style",fontweight="black",size=20,pad=20)

plt.tight_layout()
plt.show()


train['Weight Capacity (kg)'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(train['Weight Capacity (kg)'], fill = True, palette = 'Set2')
plt.title('Weight Capacity (kg) Distribution')
plt.xlabel("Weight Capacity (kg)")
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(15, 8))
sns.histplot(train["Price"], bins=50, kde=True)
plt.title("Price Distribution")
plt.show()


plt.figure(figsize = (15, 10))
for ind, val in enumerate(categorical_cols):
    plt.subplot(2,4, ind + 1)
    sns.boxplot(data = train, x = val, y = 'Price', palette = 'Set2')
    plt.title(f'{val} vs Price')
    plt.xlabel(val)
    plt.xticks(rotation = 45)

plt.tight_layout()
plt.show()


plt.figure(figsize = (15, 8))
sns.kdeplot(data = train, hue = 'Compartments',  x= 'Price', palette = 'viridis')
plt.show()


missing_values = train.isnull().mean() * 100
missing_values


missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(ascending = False)
missing_values


plt.figure(figsize = (15,8))
ax=sns.barplot(x = missing_values.index, y = missing_values.values, palette = 'Set2')
for container in ax.containers:
    ax.bar_label(container, fontweight="black", fmt='%1.2f%%')
plt.title('Distribution Missing Values Mean in Each Column', fontweight="black",pad=20)
plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.xticks(rotation = 45)
plt.show()


numerical_cols = numerical_cols.drop(['id', 'Price'])
numerical_cols


for i in numerical_cols:
    train[i].fillna(train[i].mean(), inplace = True)
    test[i].fillna(test[i].mean(), inplace = True)


for i in categorical_cols:
    train[i].fillna(train[i].mode()[0], inplace = True)
    test[i] = test[i].fillna(test[i].mode()[0])


plt.figure(figsize = (10,6))
missing_values = train.isna().sum()
missing_values = missing_values.values.flatten()
sns.heatmap(missing_values.reshape(1, -1), cbar = False, cmap = 'Set2')
plt.title('Number of Null Values ?', fontweight = 'black', size = 20, pad = 20)
plt.show()



train['Compartments'] = train['Compartments'].astype(int)
test['Compartments'] = test['Compartments'].astype(int)


train['Waterproof'] = train['Waterproof'].map({'Yes': 1, 'No': 0})
train['Laptop Compartment'] = train['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
train['Size'] = train['Size'].map(size_mapping)


test['Waterproof'] = test['Waterproof'].map({'Yes': 1, 'No': 0})
test['Laptop Compartment'] = test['Laptop Compartment'].map({'Yes': 1, 'No': 0})

size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
test['Size'] = test['Size'].map(size_mapping)


cols = ['Brand', 'Material','Style', 'Color']

le = LabelEncoder()

for col in cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.fit_transform(test[col])


train.head()


test_2 = test.copy()
test_2.drop('id', axis = 1, inplace = True)


columns=test_2.columns
scaler = StandardScaler()
train[columns] = scaler.fit_transform(train[columns])
test_2[columns] = scaler.transform(test_2[columns])


train.head()


plt.figure(figsize=(9, 6))
heatmap=sns.heatmap(train.corr(), annot=True, cmap='viridis', fmt=".2f", annot_kws={"size":9})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=70, fontsize=9)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=9)
plt.title('Correlation Heatmap of Train DataSet')
plt.show()


X = train.drop(['id', 'Price'], axis = 1)
y = train['Price']


X.shape


y.shape


train_id, val_id = train_test_split(X.index, test_size=0.2, random_state=42)


X_train, X_val = X.iloc[train_id], X.iloc[val_id]


y_train, y_val = y.iloc[train_id], y.iloc[val_id]


model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(1024, activation = 'relu', input_dim=9),
    Dropout(0.1),

    Dense(1024, activation = 'relu'),
    Dropout(0.1),
    
    Dense(1024, activation = 'relu'),
    Dropout(0.1),
    
    Dense(128, activation = 'relu'),
    Dropout(0.1),
    
    Dense(128, activation = 'relu'),
    Dropout(0.1),
    
    Dense(64, activation = 'relu'),
    Dropout(0.1),

    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error', metrics=[RootMeanSquaredError()])


model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

search = model.fit(X_train, y_train, epochs=50, batch_size=2048, callbacks=[early_stopping],validation_data=(X_val, y_val),verbose=0 )

best_rmse = min(search.history['val_root_mean_squared_error'])  
print("\nBest Val RMSE: ", best_rmse)


y_pred = model.predict(test_2)


y_pred = np.array(y_pred).flatten() 
submission = pd.DataFrame({'id': test['id'], 'Price': y_pred})


submission.to_csv('submission.csv', index=False)
display(submission)

