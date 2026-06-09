# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Path del directorio donde se encuentran las imágenes y csv
path_base_img="/kaggle/input/computer-vision-xm/images/kaggle/working/Reorganized_Data/images"
path_csv="/kaggle/input/computer-vision-xm"


#CSV de la data de train y test para cargar los datos
import os
train_path=os.path.join(path_csv,"train.csv")
test_path=os.path.join(path_csv,"test.csv")


#Obtengo los dataframes
import pandas as pd
train_df=pd.read_csv(train_path,index_col=0) #index_col los ID's de las imgs
test_df=pd.read_csv(test_path,index_col=0)


train_df.head(5)


train_df["Labels"].value_counts() #Dataset sin desbalance de clases


#Separo la data de train para luego separar la data de val
from sklearn.model_selection import train_test_split

df_train,df_val=train_test_split(train_df,test_size=0.2,stratify=train_df["Labels"],random_state=42,shuffle=True)


df_train.shape,df_val.shape,test_df.shape


#Clases balanceadas en ambos conjuntos
df_train["Labels"].value_counts(),df_val["Labels"].value_counts()


#Cambio el tipo de datos de la columna Labels a STR, pues estoy usando class_mode="binary"
df_train["Labels"]=df_train["Labels"].astype("str")
df_val["Labels"]=df_val["Labels"].astype("str")


df_train["Labels"].dtype,df_val["Labels"].dtype


#Ahora, creo los objetos generadores para cargar los datos y aplicar transformaciones a la data de train (data augmentation)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen=ImageDataGenerator(
    rescale=1/255,
    rotation_range=30,
    fill_mode="nearest",
    zoom_range=0.2,
    brightness_range=[0.5,1.5],
    width_shift_range=0.3
)
    
val_gen=ImageDataGenerator(
    rescale=1/255
)

test_gen=ImageDataGenerator(
    rescale=1/255
)


#Obtengo los datos
train_data=train_gen.flow_from_dataframe(
    dataframe=df_train,
    directory=path_base_img, #Directorio donde se encuentran las imágenes
    x_col="Images",
    y_col="Labels",
    target_size=(128,128),
    color_mode="rgb",
    class_mode="binary",
    batch_size=20,
    shuffle=False #Pues ya se realizó el shuffle al realizar el train_test_split
    
)
val_data=val_gen.flow_from_dataframe(
    dataframe=df_val,
    directory=path_base_img,
    x_col="Images",
    y_col="Labels",
    target_size=(128,128),
    color_mode="rgb",
    class_mode="binary",
    batch_size=20,
    shuffle=False
)


#Defino las clases, según el dataset
clases=["healthy","diseased"]


#Realizo el plotteo para algunas imágenes de la validación
import matplotlib.pyplot as plt

for batch,batch_labels in val_data:
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([]),plt.yticks([])
        plt.title(clases[int(batch_labels[i])])
        plt.imshow(batch[i])
    break


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2


m_s=keras.Input((128,128,3))
x0=layers.Conv2D(filters=32,kernel_size=3,activation="relu")(m_s)
x0=layers.MaxPooling2D(pool_size=(2,2))(x0)


x0=layers.Conv2D(filters=64,kernel_size=3,activation="relu")(x0)
x0=layers.MaxPooling2D(pool_size=(2,2))(x0)


x0=layers.Conv2D(filters=128,kernel_size=3,activation="relu")(x0)
x0=layers.MaxPooling2D(pool_size=(2,2))(x0)


x0=layers.Conv2D(filters=256,kernel_size=3,activation="relu")(x0)
x0=layers.MaxPooling2D(pool_size=(2,2))(x0)


x0=layers.GlobalAveragePooling2D()(x0) #GlobalAveragePooling --> 256, 1d vector
x0=layers.Dense(50,activation="relu")(x0)
x0=layers.Dropout(0.3)(x0)
m_o=layers.Dense(1,activation="sigmoid")(x0)

modelo_scratch=keras.Model(m_s,m_o)


modelo_scratch.summary()


from tensorflow.keras.callbacks import EarlyStopping
ES_scratch=EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True #Para que se 'restablezcan' los mejores pesos luego de que el callback para el training
)


#Compile step
from tensorflow.keras.optimizers import Adam

modelo_scratch.compile(
    optimizer=Adam(learning_rate=0.001), #Depende de cómo van los valores en el training, iré cambiando los valores
    loss="binary_crossentropy", #Pues es clasificación binaria
    metrics=["accuracy"]
    )


#Training step
history_scratch=modelo_scratch.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[ES_scratch]
    )


import matplotlib.pyplot as plt
accuracy_sc=history_scratch.history["accuracy"]
val_accuracy_sc=history_scratch.history["val_accuracy"]
epochs=range(1,len(accuracy_sc)+1)

plt.plot(epochs,accuracy_sc,"red",label="accuracy")
plt.plot(epochs,val_accuracy_sc,"green",label="val_accuracy")
plt.xlabel("epochs")
plt.legend()
plt.show()


loss_sc=history_scratch.history["loss"]
val_loss_sc=history_scratch.history["val_loss"]
epochs=range(1,len(loss_sc)+1)

plt.plot(epochs,loss_sc,"red",label="loss")
plt.plot(epochs,val_loss_sc,"blue",label="val_loss")
plt.xlabel("epochs")
plt.legend()
plt.show()


#Predicciones
val_pred_sc=modelo_scratch.predict(val_data)


val_pred_sc=(val_pred_sc>0.5).astype("int32")


val_pred_n_sc=val_pred_sc.reshape(val_pred_sc.shape[0],)


val_labels_sc=val_data.classes


#Matriz de confusión
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score
c_m_s=confusion_matrix(val_labels_sc,val_pred_n_sc)
c_m_d_s=ConfusionMatrixDisplay(c_m_s)
c_m_d_s.plot()


#Accuracy
acc_sc=accuracy_score(val_labels_sc,val_pred_n_sc)
print(f"Accuracy: {acc_sc}")


#Precision score: la clase positiva es el label "diseased"
precision_sc=precision_score(val_labels_sc,val_pred_n_sc)
print(f"Precision: {precision_sc}")


#Ya que la data fue cargada correctamente, import el modelo preentrenado VGG16
from tensorflow.keras.applications import vgg16
VGG16=vgg16.VGG16(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)


#Establezco que los pesos del modelo de la base convolucional no sean entrenables
VGG16.trainable=False


VGG16.summary()


#Añado las capas densamente conectadas
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import L2

x1=VGG16.output
x1=layers.Flatten()(x1) #Capa flatten para que sea un vector 1-d

x1=layers.Dense(1000,activation="relu")(x1) 
x1=layers.BatchNormalization()(x1)
x1=layers.Dropout(0.3)(x1) #30% de dropout

x1=layers.Dense(100,activation="relu")(x1)
x1=layers.BatchNormalization()(x1)
x1=layers.Dropout(0.3)(x1) #30% de dropout

output=layers.Dense(1,activation="sigmoid")(x1) #1 neurona en la capa de salida: pues es binary classification
modelo=keras.Model(VGG16.inputs,output) #Creación del modelo


modelo.summary()


#Defino el callback EarlyStopping para frenar el training cuando el modelo empieza a sobreajustarse
from tensorflow.keras.callbacks import EarlyStopping
ES=EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True #Para que se 'restablezcan' los mejores pesos luego de que el callback para el training
)


#Compile step
from tensorflow.keras.optimizers import Adam

modelo.compile(
    optimizer=Adam(learning_rate=0.001), #Depende de cómo van los valores en el training, iré cambiando los valores
    loss="binary_crossentropy", #Pues es clasificación binaria
    metrics=["accuracy"]
    )


#Training step
history=modelo.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[ES]
    )


import matplotlib.pyplot as plt
accuracy=history.history["accuracy"]
val_accuracy=history.history["val_accuracy"]
epochs=range(1,len(accuracy)+1)

plt.plot(epochs,accuracy,"red",label="accuracy")
plt.plot(epochs,val_accuracy,"green",label="val_accuracy")
plt.xlabel("epochs")
plt.legend()
plt.show()


loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs=range(1,len(accuracy)+1)

plt.plot(epochs,loss,"red",label="loss")
plt.plot(epochs,val_loss,"blue",label="val_loss")
plt.xlabel("epochs")
plt.legend()
plt.show()


#Predicciones
val_pred=modelo.predict(val_data)


val_pred_n=(val_pred>0.5).astype("int32")


val_pred_n=val_pred_n.reshape(val_pred_n.shape[0],)


val_labels_=val_data.classes


#Matriz de confusión
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score
c_m_1=confusion_matrix(val_labels_,val_pred_n)
c_m_d_1=ConfusionMatrixDisplay(c_m_1)
c_m_d_1.plot()


#Accuracy
acc1=accuracy_score(val_labels_,val_pred_n)
print(f"Accuracy: {acc1}")


#Precision score: la clase positiva es el label "diseased"
precision1=precision_score(val_labels_,val_pred_n)
print(f"Precision: {precision1}")


from tensorflow.keras.applications import vgg16
VGG16_2=vgg16.VGG16(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)


VGG16_2.summary()


VGG16_2.trainable=True


for layer in VGG16_2.layers[:-2]: #Última capa, ya que la de max pooling no se entrena
    layer.trainable=False


VGG16_2.summary()


x2=VGG16_2.output
x2=layers.Flatten()(x2) #Capa flatten para que sea un vector 1-d

x2=layers.Dense(1000,activation="relu")(x2) 
x2=layers.BatchNormalization()(x2)
x2=layers.Dropout(0.3)(x2) #30% de dropout

x2=layers.Dense(100,activation="relu")(x2)
x2=layers.BatchNormalization()(x2)
x2=layers.Dropout(0.3)(x2) #30% de dropout

output2=layers.Dense(1,activation="sigmoid")(x2) #1 neurona en la capa de salida: pues es binary classification
modelo2=keras.Model(VGG16_2.inputs,output2) #Creación del modelo


#Defino el callback EarlyStopping para frenar el training cuando el modelo empieza a sobreajustarse
from tensorflow.keras.callbacks import EarlyStopping
ES2=EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True #Para que se 'restablezcan' los mejores pesos luego de que el callback para el training
)


#Compile step
from tensorflow.keras.optimizers import Adam

modelo2.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss="binary_crossentropy", #Pues es clasificación binaria
    metrics=["accuracy"]
    )


#Training step
history2=modelo2.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[ES2]
    )


accuracy2=history2.history["accuracy"]
val_accuracy2=history2.history["val_accuracy"]
epochs=range(1,len(accuracy2)+1)

plt.plot(epochs,accuracy2,"red",label="accuracy")
plt.plot(epochs,val_accuracy2,"green",label="val_accuracy")
plt.xlabel("epochs")
plt.legend()
plt.show()


loss2=history2.history["loss"]
val_loss2=history2.history["val_loss"]
epochs=range(1,len(loss2)+1)

plt.plot(epochs,loss2,"red",label="loss")
plt.plot(epochs,val_loss2,"blue",label="val_loss")
plt.xlabel("epochs")
plt.legend()
plt.show()


#Predicciones
val_pred2=modelo2.predict(val_data)


val_pred2_n=(val_pred2>0.5).astype("int32")


val_pred2_n=val_pred2_n.reshape(val_pred2_n.shape[0],)


val_labels=val_data.classes


#Matriz de confusión
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score
c_m=confusion_matrix(val_labels,val_pred2_n)
c_m_d=ConfusionMatrixDisplay(c_m)
c_m_d.plot()


#Accuracy
acc=accuracy_score(val_labels,val_pred2_n)
print(f"Accuracy: {acc}")


#Precision score: la clase positiva es el label "diseased"
precision=precision_score(val_labels,val_pred2_n)
print(f"Precision: {precision}")


#Obtengo la data de test
import cv2
img_size= 128

#Cargar y resize a las imágenes
def CargarTest(image_path):
    imagen = cv2.imread(image_path)
    imagen = cv2.resize(imagen, (img_size,img_size))
    imagen = imagen / 255.0  #Normalizando la data
    return imagen


#Preprocesamiento para todas las imágenes
imagenes = []

for i,fila in test_df.iterrows():
    nombre_img=fila["Images"]
    imgs=os.path.join(path_base_img,nombre_img)
    imagenes.append(CargarTest(imgs))


#Convierto a un array
import numpy as np
x_test=np.array(imagenes)


#Predicción
pred_final=modelo2.predict(x_test)


pred_final_labels=(pred_final>0.5).astype("int32")


pred_final_labels=pred_final_labels.reshape(pred_final_labels.shape[0],)


#Creo el dataframe
submission_df = pd.DataFrame({
    'Images': test_df['Images'],
    'Labels': pred_final_labels
})


submission_df.to_csv('submission.csv', index=False)
print('Submission file was created.')

