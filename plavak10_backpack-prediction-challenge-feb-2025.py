import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_df


training_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
training_extra_df


train = pd.concat([train_df,training_extra_df],axis=0)
train


train.info()


train.describe()


train.drop('id',axis=1,inplace=True)
train


categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
categorical_columns


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_columns):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x')

for j in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()   


train['Laptop Compartment'].value_counts()


train.isnull().sum()


train[train['Brand'].isna()]


train['Weight Capacity (kg)'].fillna(train['Weight Capacity (kg)'].mean(), inplace=True)
train['Brand'].fillna(train['Brand'].mode()[0], inplace=True)
train['Material'].fillna(train['Material'].mode()[0], inplace=True)
train['Size'].fillna(train['Size'].mode()[0], inplace=True)
train['Laptop Compartment'].fillna(train['Laptop Compartment'].mode()[0], inplace=True)
train['Waterproof'].fillna(train['Waterproof'].mode()[0], inplace=True)
train['Style'].fillna(train['Style'].mode()[0], inplace=True)
train['Color'].fillna(train['Color'].mode()[0], inplace=True)

train.reset_index(drop=True, inplace=True)
train


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_columns):
    sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x')

for j in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()   


yes_no_mapping = {'Yes':1,'No':0}
train['Laptop Compartment'] = train['Laptop Compartment'].map(yes_no_mapping)
train['Waterproof'] = train['Waterproof'].map(yes_no_mapping)
train


size_mapping = {'Small':0,'Medium':1,'Large':2}
train['Size'] = train['Size'].map(size_mapping)
train


avg_price = train.groupby('Material')['Price'].mean().reset_index()

avg_price


df = pd.get_dummies(train,['Brand','Material','Style','Color'],dtype=int)
df


X = df.drop('Price',axis=1)
y = df['Price']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = StandardScaler()
mm = MinMaxScaler()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



columns_to_pass = X_train.drop('Weight Capacity (kg)',axis=1).columns.tolist()
column_to_scale = ['Weight Capacity (kg)']

preprocessor = ColumnTransformer(
    transformers=[
        ('scale', sc, column_to_scale),
        ('passthrough', 'passthrough', columns_to_pass)
    ])
X_train_transformed = preprocessor.fit_transform(X_train)
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=column_to_scale + columns_to_pass)


X_train_transformed_df


X_test_transformed = preprocessor.transform(X_test)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=column_to_scale + columns_to_pass)


X_test_transformed_df


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


model = Sequential([
    Dense(128, input_shape=(23,)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(64),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(32),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dense(1)
])


model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)



model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])


history = model.fit(
    X_train_transformed_df, y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


test_loss, test_mae, test_mse = model.evaluate(X_test_transformed_df, y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test MAE: {test_mae}")
print(f"Test MSE: {test_mse}")


test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
test_df


test = test_df.copy()
test


test.drop('id',axis=1,inplace=True)
test


test.isnull().sum()


test['Weight Capacity (kg)'].fillna(test['Weight Capacity (kg)'].mean(), inplace=True)
test['Brand'].fillna(test['Brand'].mode()[0], inplace=True)
test['Material'].fillna(test['Material'].mode()[0], inplace=True)
test['Size'].fillna(test['Size'].mode()[0], inplace=True)
test['Laptop Compartment'].fillna(test['Laptop Compartment'].mode()[0], inplace=True)
test['Waterproof'].fillna(test['Waterproof'].mode()[0], inplace=True)
test['Style'].fillna(test['Style'].mode()[0], inplace=True)
test['Color'].fillna(test['Color'].mode()[0], inplace=True)


test


test.isnull().sum()


test['Laptop Compartment'] = test['Laptop Compartment'].map(yes_no_mapping)
test['Waterproof'] = test['Waterproof'].map(yes_no_mapping)
test


size_mapping = {'Small':0,'Medium':1,'Large':2}
test['Size'] = test['Size'].map(size_mapping)

testing_df = pd.get_dummies(test,['Brand','Material','Style','Color'],dtype=int)
testing_df


columns_to_pass = testing_df.drop('Weight Capacity (kg)',axis=1).columns.tolist()
column_to_scale = ['Weight Capacity (kg)']

X_output_transformed = preprocessor.transform(testing_df)
X_output_transformed_df = pd.DataFrame(X_output_transformed, columns=column_to_scale + columns_to_pass)
X_output_transformed_df


y_pred = model.predict(X_output_transformed_df)
y_pred


max(y_pred)


min(y_pred)


y_pred_flat = y_pred.flatten()


output = pd.DataFrame({'id':test_df['id'].tolist(),'Price':y_pred_flat})


output['Price'] = round(output['Price'],2)


output


output.to_csv('/kaggle/working/sample_submission2.csv')




