import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import confusion_matrix,classification_report


image_paths=[]
labels=[]
datapath="/kaggle/input/rsna-bcd-1024x512-preprocessed/train_images"
df=pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
df


     
       
        



image="1864590858.png"
id_part, _ = os.path.splitext(image)#splittext return id in string formate 
df['image_id'] = df['image_id'].astype(str)#so we conver image id in dataframe into string formate to make match correctly
print(id_part)
match = df[df["image_id"] == id_part]#return row that has this image id 
print(match)


print(match["cancer"].values[0])




folds=os.listdir(datapath)#ids folders for patients
for fold in folds:
    folder_path=os.path.join(datapath,fold)#folder path for id patient
    images=os.listdir( folder_path)#images for acertain patient
    for image in images:
        image_path=os.path.join(folder_path,image)
        
        image_paths.append(image_path)
        id_part,_=os.path.splitext(image)
        match = df[df["image_id"] == id_part]
        if not match.empty:
           labels.append(match["cancer"].values[0])
       


df_new=pd.DataFrame({"filepaths":image_paths,"labels":labels})
df_new


df_new["labels"]=df_new['labels'].astype(str)


train_df,dummy_df=train_test_split(df_new,random_state=42,test_size=0.2)
test_df,valid_df=train_test_split(dummy_df,random_state=42,test_size=0.5)



gen=ImageDataGenerator()
train_gen=gen.flow_from_dataframe(train_df,x_col="filepaths",y_col="labels",color_mode="rgb",batch_size=128,class_mode="binary",target_size=(224,224))
test_gen=gen.flow_from_dataframe(test_df,x_col="filepaths",y_col="labels",color_mode="rgb",batch_size=32,class_mode="binary",target_size=(224,224))
valid_gen=gen.flow_from_dataframe(valid_df,x_col="filepaths",y_col="labels",color_mode="rgb",batch_size=32,class_mode="binary",target_size=(224,224))


basemodel=tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3),
    pooling="max"
           
)
basemodel.trainable=True
model=Sequential([
    basemodel,
    Dense(128,activation="relu"),
    Dense(1,activation="sigmoid"),
    
])
model.compile(optimizer=Adam(learning_rate=0.01),loss="binary_crossentropy",metrics=["accuracy"])
model.summary




model.fit(train_gen,validation_data=valid_gen,epochs=1,batch_size=32)



model.save('model.keras')


model.evaluate(test_gen)


from tensorflow.keras.models import load_model
loaded_model=load_model("model.keras")



from PIL import Image
Image_path="/kaggle/input/rsna-bcd-1024x512-preprocessed/train_images/10006/1864590858.png"
image=Image.open(Image_path)#read image and can show it direct without show as plt or cv2
image


image="1864590858.png"
id_part, _ = os.path.splitext(image)#splittext return id in string formate 
df['image_id'] = df['image_id'].astype(str)#so we conver image id in dataframe into string formate to make match correctly
match = df[df["image_id"] == id_part]#return row that has this image id 
print(match["cancer"].values[0])



from PIL import Image
import tensorflow as tf

# Convert the image to RGB
img = image.convert('RGB')

# Resize the image
img = img.resize((224, 224))

# Convert image to array
img_array = tf.keras.preprocessing.image.img_to_array(img)

# Add a batch dimension (1, 224, 224, 3)
img_array = tf.expand_dims(img_array, 0)

# Predict
prediction = loaded_model.predict(img_array)

print(prediction[0])


