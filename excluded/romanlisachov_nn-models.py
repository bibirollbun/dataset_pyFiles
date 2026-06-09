import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ReduceLROnPlateau


train = pd.read_csv('../input/playground-series-s5e3/train.csv')


train.head()


 train.info()


target = train.pop('rainfall')


target.head()



train_all = train.copy()
target_all = target.copy()
train, test, target_train, target_test = train_test_split(train_all, target_all, test_size=0.2, random_state=0)


train.info()


 test.info()


train_all = train.copy()
target_all = target_train.copy()
train, valid, target_train, target_valid = train_test_split(train_all, target_all, test_size=0.2, random_state=0)


train.info()


# Creation the dataframe with the resulting score of all models
result = pd.DataFrame({'model' : ['NN 1', 'NN 2', 'NN 3','NN 4'], 
                       'train_accuracy': 0, 'train_loss': 0, 'valid_accuracy': 0, 'valid_loss': 0, 'test_accuracy': 0,'test_loss': 0,})
result['train_loss'] = result['train_loss'].astype(float)
result['train_accuracy'] = result['train_accuracy'].astype(float)
result['valid_loss'] = result['valid_loss'].astype(float)
result['valid_accuracy'] = result['valid_accuracy'].astype(float)
result['test_loss'] = result['test_loss'].astype(float)
result['test_accuracy'] = result['test_accuracy'].astype(float)

result



def build_nn():
    
    model = models.Sequential()
   
    model.add(layers.Dense(16, activation='relu', input_shape=(train.shape[1],)))

    model.add(layers.Dense(8, activation='relu'))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

    return model

nn_model = build_nn()
nn_model.fit(train, target_train, batch_size=32, epochs=50, validation_data=(valid, target_valid), verbose=0)


plt.plot(nn_model.history.history['accuracy'], label='Train Accuracy')
plt.plot(nn_model.history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(nn_model.history.history['loss'],label='loss')
plt.plot(nn_model.history.history['val_loss'],label='val_loss')
plt.title('Loss of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss, accuracy = nn_model.evaluate(train, target_train, verbose=0)
print(f"Train Loss: {loss:.4f}")
print(f"Train Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 1', 'train_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 1', 'train_accuracy'] = round(accuracy, 3)

loss, accuracy = nn_model.evaluate(valid, target_valid, verbose=0)
print(f"Valid Loss: {loss:.4f}")
print(f"Valid Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 1', 'valid_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 1', 'valid_accuracy'] = round(accuracy, 3)

nn_model.fit(test, target_test, batch_size=32, epochs=200, verbose=0)

loss, accuracy = nn_model.evaluate(test, target_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 1', 'test_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 1', 'test_accuracy'] = round(accuracy, 3)

nn_model.summary()


def build_nn():
    
    model = models.Sequential()
      
    model.add(layers.Dense(16, activation='relu', input_shape=(train.shape[1],)))

    model.add(Dropout(0.2))

    model.add(layers.Dense(8, activation='relu'))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    
    return model

nn_model = build_nn()
nn_model.fit(train, target_train, batch_size=32, epochs= 50, validation_data=(valid, target_valid), verbose=0)


plt.plot(nn_model.history.history['accuracy'], label='Train Accuracy')
plt.plot(nn_model.history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(nn_model.history.history['loss'],label='loss')
plt.plot(nn_model.history.history['val_loss'],label='val_loss')
plt.title('Loss of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


loss, accuracy = nn_model.evaluate(train, target_train, verbose=0)
print(f"Train Loss: {loss:.4f}")
print(f"Train Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 2', 'train_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 2', 'train_accuracy'] = round(accuracy, 3)

loss, accuracy = nn_model.evaluate(valid, target_valid, verbose=0)
print(f"Valid Loss: {loss:.4f}")
print(f"Valid Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 2', 'valid_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 2', 'valid_accuracy'] = round(accuracy, 3)

nn_model.fit(test, target_test, batch_size=32, epochs=200, verbose=0)

loss, accuracy = nn_model.evaluate(test, target_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 2', 'test_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 2', 'test_accuracy'] = round(accuracy, 3)

nn_model.summary()


def build_nn():
    
    model = models.Sequential()
    
    model.add(layers.Dense(32, activation='relu', input_shape=(train.shape[1],)))

    model.add(layers.Dense(16, activation='relu'))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    
         
    return model

nn_model = build_nn()
nn_model.fit(train, target_train, batch_size=32, epochs=50, validation_data=(valid, target_valid), verbose=0)

plt.plot(nn_model.history.history['accuracy'], label='Train Accuracy')
plt.plot(nn_model.history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(nn_model.history.history['loss'],label='loss')
plt.plot(nn_model.history.history['val_loss'],label='val_loss')
plt.title('Loss of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss, accuracy = nn_model.evaluate(train, target_train, verbose=1)
print(f"Train Loss: {loss:.4f}")
print(f"Train Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 3', 'train_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 3', 'train_accuracy'] = round(accuracy, 3)

loss, accuracy = nn_model.evaluate(valid, target_valid, verbose=0)
print(f"Valid Loss: {loss:.4f}")
print(f"Valid Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 3', 'valid_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 3', 'valid_accuracy'] = round(accuracy, 3)

nn_model.fit(test, target_test, batch_size=32, epochs=200, verbose=0)

loss, accuracy = nn_model.evaluate(test, target_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 3', 'test_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 3', 'test_accuracy'] = round(accuracy, 3)

nn_model.summary()



def build_nn():
    
    model = models.Sequential()
    
    model.add(layers.Dense(64, activation='relu', input_shape=(train.shape[1],)))
    model.add(Dropout(0.1))
    model.add(layers.Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
           
    return model

nn_model = build_nn()
nn_model.fit(train, target_train, batch_size=32, epochs=50, validation_data=(valid, target_valid), verbose=0)

plt.plot(nn_model.history.history['accuracy'], label='Train Accuracy')
plt.plot(nn_model.history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(nn_model.history.history['loss'],label='loss')
plt.plot(nn_model.history.history['val_loss'],label='val_loss')
plt.title('Loss of NN Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss, accuracy = nn_model.evaluate(train, target_train, verbose=0)
print(f"Train Loss: {loss:.4f}")
print(f"Train Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 4', 'train_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 4', 'train_accuracy'] = round(accuracy, 3)

loss, accuracy = nn_model.evaluate(valid, target_valid, verbose=0)
print(f"Valid Loss: {loss:.4f}")
print(f"Valid Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 4', 'valid_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 4', 'valid_accuracy'] = round(accuracy, 3)

nn_model.fit(test, target_test, batch_size=32, epochs=200, verbose=0)

loss, accuracy = nn_model.evaluate(test, target_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

result.loc[result['model'] == 'NN 4', 'test_loss'] = round(loss, 3)
result.loc[result['model'] == 'NN 4', 'test_accuracy'] = round(accuracy, 3)

nn_model.summary()


mapping = {
    'NN 1': 'NN with 2 hidden layers ',
    'NN 2': 'NN with Dropout',
    'NN 3': 'NN with twice the number of neurons',
    'NN 4': 'NN with 2 Dropout, many neurons and reduce lr'
}

# Rename the values in the model column
result['model'] = result['model'].replace(mapping)

result


# Select models with minimal overfitting
print('Models without retraining are sorted by test_accuracy, train_accuracy')
result_best = result[(result['train_accuracy'] - result['valid_accuracy']).abs() < 5]
result_best.sort_values(by=['test_accuracy', 'train_accuracy'], ascending=False)


best_model_name = result_best.loc[result_best['test_accuracy'].idxmax(), 'model']
print(f'The best model is "{best_model_name}"')


test_df = pd.read_csv('../input/playground-series-s5e3/test.csv')


print(test_df.shape)


output=nn_model.predict(test_df)


preds=output[:,0]


sub_df=pd.DataFrame()
sub_df['id']=test_df['id']
sub_df['rainfall']=preds
sub_df.isnull().sum()
sub_df = sub_df.fillna(0)



sub_df.to_csv("submission.csv",index=False)


sub_df

