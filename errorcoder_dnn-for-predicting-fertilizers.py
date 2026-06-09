import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder , StandardScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


dft=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
dfte=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


dft


datat=pd.DataFrame(dft)
datats=pd.DataFrame(dfte)


datat.info()


from sklearn.preprocessing import LabelEncoder

# for Soil Type
le_soil = LabelEncoder()
datat['Soil Type'] = le_soil.fit_transform(datat['Soil Type'])
datats['Soil Type'] = le_soil.transform(datats['Soil Type'])

# for Crop Type
le_crop = LabelEncoder()
datat['Crop Type'] = le_crop.fit_transform(datat['Crop Type'])
datats['Crop Type'] = le_crop.transform(datats['Crop Type'])

le_fert = LabelEncoder()
datat['Fertilizer Name'] = le_fert.fit_transform(datat['Fertilizer Name'])

y= datat['Fertilizer Name']



from sklearn.model_selection import train_test_split
x=datat.drop(['Fertilizer Name','id'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.nunique(), activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy,
              metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)



history = model.fit(x_train, y_train, epochs=10, batch_size=64,callbacks=[early_stop, reduce_lr], validation_data=(x_test, y_test))


loss, accuracy = model.evaluate(x_test, y_test)
print(f'Validation Accuracy: {accuracy*100:.2f}%')



X_test = datats.drop('id', axis=1)


# Predict probabilities
pred_probs = model.predict(X_test)




# Get top 3 predictions per row
top3_indices = np.argsort(pred_probs, axis=1)[:, ::-1][:, :3]

top3_labels = np.column_stack([
    le_fert.inverse_transform(top3_indices[:, 0]),
    le_fert.inverse_transform(top3_indices[:, 1]),
    le_fert.inverse_transform(top3_indices[:, 2])
])

final_preds = [' '.join(row) for row in top3_labels]

submission = pd.DataFrame({
    'id': datats["id"],
    'Fertilizer Name': final_preds
})

submission.to_csv('submission_tf.csv', index=False)




