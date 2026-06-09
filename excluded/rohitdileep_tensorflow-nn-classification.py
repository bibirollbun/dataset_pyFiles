import numpy as np 
import pandas as pd 



data = pd.read_csv("/kaggle/input/bird-cliff2025-data/df_melfrequencies.csv")
data.head()


data.class_names.value_counts().plot(kind='hist')


## One hot encoding for target variable ###
from sklearn.preprocessing import LabelEncoder

primary_labelencoder = LabelEncoder()

data['class_names'] = primary_labelencoder.fit_transform(data['class_names'])

##X vars ##
cols  = data.columns.difference(['class_names'])


from sklearn.model_selection import train_test_split

trainx , testx , trainy , testy = train_test_split(data[cols] ,data['class_names']  , random_state = 0 , test_size = 0.25)
print(trainx.shape)
print(testx.shape)
print(trainy.shape)
print(testy.shape)


import tensorflow as tf 
from tensorflow.keras import layers , models , callbacks

input_dim = trainx.shape[1]
model = models.Sequential(
    [
        layers.Input(shape = (input_dim , )) , 
        layers.Dense(4096 , activation = 'relu') ,
        layers.Dropout(0.25) ,
        layers.Dense(3072 , activation = 'relu') ,
        layers.Dropout(0.25) ,
        layers.Dense(2048 , activation = 'relu') ,
        layers.Dropout(0.25) ,
        layers.Dense(1024 , activation = 'relu') , 
        layers.Dropout(0.25) ,
        layers.Dense(502 , activation = 'relu') , 
        layers.Dropout(0.25) ,
        layers.Dense(206 , activation = 'softmax')
    ]
)




print(model.summary())


model.compile(optimizer = 'adam' , metrics = ['accuracy'] , loss = 'sparse_categorical_crossentropy')
check_point_callback  =  callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    mode = 'max' ,
    verbose = 1 , 
    monitor = 'val_accuracy' , 
    save_best_only = True , 
)

early_stop_callback =  callbacks.EarlyStopping(monitor = 'val_accuracy' , 
                                               patience = 25 , 
                                               restore_best_weights = True ,
                                              mode = 'max' )


history  = model.fit(x =  trainx , y = trainy , epochs = 1000 , batch_size = 256 , validation_data = (testx , testy) , 
          callbacks = [check_point_callback , early_stop_callback ]  )


from tensorflow.keras.models import load_model


# Load trained model
model = load_model('/kaggle/working/best_model.keras')


proba = model.predict(testx)  # Get probability predictions
pred_indices = np.argmax(proba, axis=1)  


pred_indices


import matplotlib.pyplot as plt

# Assuming you saved your model history object as `history`
plt.figure(figsize=(12, 5))

# ðŸ”¹ Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# ðŸ”¸ Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



## metrics 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Assuming `testy` and `pred_indices` are your actual and predicted labels
print("Accuracy:", accuracy_score(testy, pred_indices))
print("Precision:", precision_score(testy, pred_indices, average='macro'))
print("Recall:", recall_score(testy, pred_indices, average='macro'))
print("F1 Score:", f1_score(testy, pred_indices, average='macro'))

# Detailed report
print("\nðŸ“‹ Classification Report:\n")
print(classification_report(testy, pred_indices, digits=4))






import os
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Load model
model = load_model('/kaggle/working/best_model.keras')

# Load label encoder used during training
primary_labelencoder = LabelEncoder()
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
primary_labelencoder.fit(class_labels)  # Fit encoder on original labels

# Paths
test_dir = '/kaggle/input/birdclef-2025/test_soundscapes'
sample_sub = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')

# Parameters
sr = 32000          # sampling rate
duration = 5        # seconds
n_mfcc = 40         # same as used during training

# Feature extraction
def extract_mfcc(y, sr, n_mfcc):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Final prediction storage
submission_rows = []

# Iterate through soundscapes
for filename in tqdm(sorted(os.listdir(test_dir))):
    if not filename.endswith('.ogg'):
        continue
    try :
        file_path = os.path.join(test_dir, filename)
        y, _ = librosa.load(file_path, sr=sr)
        soundscape_id = filename.replace('.ogg', '')
    
        for i in range(0, 60, duration):
            start_sample = i * sr
            end_sample = (i + duration) * sr
            segment = y[start_sample:end_sample]
    
            # Pad if needed
            if len(segment) < duration * sr:
                segment = np.pad(segment, (0, duration * sr - len(segment)))
    
            # Extract features and reshape
            mfcc = extract_mfcc(segment, sr=sr, n_mfcc=n_mfcc)
            input_features = np.expand_dims(mfcc, axis=0)
    
            # Predict probabilities
            probs = model.predict(input_features, verbose=0)[0]
    
            # Convert indices back to original species IDs
            mapped_labels = primary_labelencoder.inverse_transform(range(len(probs)))
    
            # Create a dictionary of {species_id: probability}
            prob_dict = dict(zip(mapped_labels, probs))
    
            # Ensure submission columns match sample_submission.csv order
            ordered_probs = [prob_dict.get(species, 0) for species in sample_sub.columns[1:]]
    
            # Append row
            row_id = f"{soundscape_id}_{i + duration}"
            submission_rows.append([row_id] + ordered_probs)
    except Exception as e:
        print(e)

# Build submission DataFrame
submission_df = pd.DataFrame(submission_rows, columns=['row_id'] + list(sample_sub.columns[1:]))

# Save CSV
submission_df.to_csv("submission.csv", index=False)
print("submission.csv created")





