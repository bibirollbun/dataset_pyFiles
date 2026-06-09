import os
import json
from tqdm.notebook import tqdm
from glob import glob
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as rf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from concurrent.futures import ThreadPoolExecutor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score
%matplotlib inline


train = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv')


state_labels = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/state_labels.csv')


train['State'].head()


state_labels.head()


train = train.merge(state_labels, left_on='State', right_on='StateID', how='left')
test = test.merge(state_labels, left_on='State', right_on='StateID', how='left')


train[['State', 'StateID', 'StateName']].head()


train.drop('StateID', axis=1, inplace=True)
test.drop('StateID', axis=1, inplace=True)


breed_labels = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/breed_labels.csv')


breed_labels.head()


train = train.merge(breed_labels, left_on='Breed1', right_on='BreedID', how='left')
test = test.merge(breed_labels, left_on='Breed1', right_on='BreedID', how='left')


train.columns


train.drop(['Type_x', 'Type_y', 'BreedID'], axis=1, inplace=True)
test.drop(['Type_x', 'Type_y', 'BreedID'], axis=1, inplace=True)


color_id = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/color_labels.csv')


color_id.head(10)


train = train.merge(color_id, left_on='Color1', right_on='ColorID', how='left')
test = test.merge(color_id, left_on='Color1', right_on='ColorID', how='left')


train.drop('ColorID', axis=1, inplace=True)
test.drop('ColorID', axis=1, inplace=True)


train.info()


train.describe().transpose()


train.isnull().sum()


train.duplicated().sum()


train['Name'].fillna('Unnamed', inplace=True)
test['Name'].fillna('Unnamed', inplace=True)


train['Description'].fillna('No Description', inplace=True)
test['Description'].fillna('No Description', inplace=True)


train['Description_length'] = train['Description'].str.len()
test['Description_length'] = test['Description'].str.len()


train.dropna(subset=['BreedName'], inplace=True)
test.dropna(subset=['BreedName'], inplace=True)


sns.heatmap(train.corr(numeric_only=True), cmap='viridis')


sns.set_style('whitegrid')
plt.figure(figsize=(12,6))
train['AdoptionSpeed'].plot(kind='hist', bins=30, color = 'lightblue')
plt.title('Histogram of Adoption Speed')


plt.figure(figsize=(12,6))
sns.histplot(data=train, x='Age', hue='AdoptionSpeed', bins=30, palette='muted')
plt.title('Histogram of Age of the Pets')


plt.figure(figsize=(12,6))
sns.countplot(data=train, x='Health', hue='AdoptionSpeed', palette='coolwarm')
plt.title('Number of Adopted Pets based off Health Conditions')


plt.figure(figsize=(12,6))
sns.countplot(data=train, x='Vaccinated', hue='AdoptionSpeed', palette='dark:salmon_r')
plt.title('Number of Adopted Pets based off Vaccination')


plt.figure(figsize=(12,6))
sns.countplot(data=train, x='Gender', hue='AdoptionSpeed', palette='icefire')
plt.title('Number of Adopted Pets based off Gender')


plt.figure(figsize=(12, 6))
sns.boxplot(x='AdoptionSpeed', y='PhotoAmt', data=train, palette='coolwarm')
plt.title('PhotoAmt Distribution by AdoptionSpeed')


train[['Description_length', 'AdoptionSpeed']].corr()


train.drop('Description_length', axis=1, inplace=True)
test.drop('Description_length', axis=1, inplace=True)


train['Name_length'] = train['Name'].str.len()
test['Name_length'] = test['Name'].str.len()


train[['Name_length', 'AdoptionSpeed']].corr()


train.drop('Name_length', axis=1, inplace=True)
test.drop('Name_length', axis=1, inplace=True)


train['Mixed_Breed'] = (train['Breed1'] != train['Breed2']).astype(int)
test['Mixed_Breed'] = (test['Breed1'] != test['Breed2']).astype(int)


train['Mixed_Breed'].value_counts()


train['ColorCombo'] = train[['Color1', 'Color2', 'Color3']].astype(str).apply('_'.join, axis=1)
test['ColorCombo'] = test[['Color1', 'Color2', 'Color3']].astype(str).apply('_'.join, axis=1)


plt.figure(figsize=(15, 8))
sns.countplot(y='StateName', hue='AdoptionSpeed', data=train, palette='viridis')
plt.title('AdoptionSpeed by State')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_features(image_path):
    if pd.isna(image_path) or image_path is None:
        return np.zeros(2048)
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x, verbose=0).flatten()
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return np.zeroes(2048)


def find_image_path(pet_id, folder):
    matches = glob(os.path.join(folder, f"{pet_id}-*.jpg"))
    if matches:
        return matches[0]
    else:
        return None


def build_image_dict(folder):
    image_dict = defaultdict(list)
    for f in Path(folder).glob("*-*.jpg"):
        pet_id = f.name.split('-')[0]
        image_dict[pet_id].append(str(f))
    return image_dict


train_dict = build_image_dict('/kaggle/input/petfinder-adoption-prediction/train_images')
train['image_path'] = train['PetID'].map(lambda x: train_dict.get(x, [None])[0])


test_dict = build_image_dict('/kaggle/input/petfinder-adoption-prediction/test_images')
test['image_path'] = test['PetID'].map(lambda x: test_dict.get(x, [None])[0])


image_features_train = np.array([extract_features(path) for path in train['image_path']])


image_features_test = np.array([extract_features(path) for path in test['image_path']])


def load_metadata(pet_id, metadata_folder):
    metadata = []
    json_files = glob(os.path.join(metdata_folder, f"{pet_id}-*.json"))
    for file in json_files:
        with open(file, 'r') as f:
            metadata.append(json.load(F))
    return metadata


def process_file(file):
    try:
        pet_id = Path(file).name.split('-')[0]
        with open(file, 'r', encoding='utf-8') as f:
            return pet_id, json.load(f)
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        return None


def build_metadata_dict(metadata_folder):
    metadata_dict = defaultdict(list)
    files = list(Path(metadata_folder).glob("*-*.json"))
    
    print(f"Found {len(files)} JSON files to process...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(
            executor.map(process_file, files),
            total=len(files),
            desc="Processing JSONs"
        ))
    
    for result in results:
        if result:  # Skip failed files
            pet_id, data = result
            metadata_dict[pet_id].append(data)
    
    print(f"Processed {len(metadata_dict)} unique pets")
    return metadata_dict


train_metadata_dict = build_metadata_dict("/kaggle/input/petfinder-adoption-prediction/train_metadata")
test_metadata_dict = build_metadata_dict("/kaggle/input/petfinder-adoption-prediction/test_metadata")


def extract_animal_info(entries):
    animal_data = []
    for entry in entries:
        for annotation in entry.get('labelAnnotations', []):
            # Case-insensitive check for more animal types
            if annotation['description'].lower() in ['cat', 'dog', 'kitten', 'puppy', 'feline', 'canine']:
                animal_data.append({
                    'detected_animal': annotation['description'],
                    'animal_confidence': annotation['score']
                })
    return animal_data if animal_data else [{'detected_animal': 'unknown', 'animal_confidence': 0}]

def extract_dominant_colors(entries):
    color_info = []
    for entry in entries:
        colors = entry.get('imagePropertiesAnnotation', {}).get('dominantColors', {}).get('colors', [])
        for color in colors[:3]:  # Top 3 colors
            color_info.append({
                'red': color['color'].get('red', 0),
                'green': color['color'].get('green', 0),
                'blue': color['color'].get('blue', 0),
                'color_score': color['score']
            })
    # Return default gray if no colors found
    return color_info if color_info else [{'red': 128, 'green': 128, 'blue': 128, 'color_score': 0}]


def extract_crop_quality(entries):
    crop_scores = []
    for entry in entries:
        crops = entry.get('cropHintsAnnotation', {}).get('cropHints', [])
        if crops:
            crop_scores.append(crops[0].get('confidence', 0))
    return sum(crop_scores)/len(crop_scores) if crop_scores else 0

def extract_all_metadata_features(metadata_dict):
    features = {}
    for pet_id, entries in metadata_dict.items():
        # Skip if no entries exist for this pet
        if not entries:
            features[pet_id] = {
                'primary_animal': 'unknown',
                'avg_animal_confidence': 0,
                'dominant_red': 128,
                'dominant_green': 128,
                'dominant_blue': 128,
                'crop_confidence': 0,
                'num_labels': 0
            }
            continue
            
        try:
            animal_info = extract_animal_info(entries)
            color_info = extract_dominant_colors(entries)
            
            features[pet_id] = {
                # Animal detection
                'primary_animal': animal_info[0]['detected_animal'],
                'avg_animal_confidence': np.nanmean([a['animal_confidence'] for a in animal_info]),
                
                # Color analysis
                'dominant_red': color_info[0]['red'],
                'dominant_green': color_info[0]['green'],
                'dominant_blue': color_info[0]['blue'],
                'avg_color_score': np.nanmean([c['color_score'] for c in color_info]),
                
                # Image quality
                'crop_confidence': extract_crop_quality(entries),
                
                # Additional metrics (safe division)
                'num_labels': sum(len(e.get('labelAnnotations', [])) for e in entries)/max(1, len(entries))
            }
        except Exception as e:
            print(f"Error processing {pet_id}: {str(e)}")
            features[pet_id] = {
                'primary_animal': 'error',
                'avg_animal_confidence': 0,
                'dominant_red': 128,
                'dominant_green': 128,
                'dominant_blue': 128,
                'avg_color_score': 0,
                'crop_confidence': 0,
                'num_labels': 0
            }
    
    return pd.DataFrame.from_dict(features, orient='index').fillna(0)


metadata_features_train = extract_all_metadata_features(train_metadata_dict)


metadata_features_test = extract_all_metadata_features(test_metadata_dict)


image_features_df = pd.DataFrame(image_features_train, index=train['PetID'], columns=[f'img_{i}' for i in range(image_features_train.shape[1])])


image_features_df_test = pd.DataFrame(image_features_test, index=test['PetID'], columns=[f'img_{i}' for i in range(image_features_test.shape[1])])


metadata_features_train = metadata_features_train.reset_index().rename(columns={'index':'PetID'})


metadata_features_test = metadata_features_test.reset_index().rename(columns={'index':'PetID'})


df_train = (train.merge(metadata_features_train, on='PetID', how='left').merge(image_features_df, left_on='PetID', right_index=True, how='left'))


df_test = (test.merge(metadata_features_test, on='PetID', how='left').merge(image_features_df_test, left_on='PetID', right_index=True, how='left'))


X = df_train.drop(['RescuerID', 'image_path', 'Description', 'StateName', 'BreedName', 'ColorName',
                   'AdoptionSpeed', 'Name', 'PetID', 'ColorCombo', 'primary_animal', 'Mixed_Breed'],
                  axis=1)
y = df_train['AdoptionSpeed']

# Drop rows where X has NaN and ensure y aligns
nan_mask = X.isnull().any(axis=1)
X_clean = X[~nan_mask]
y_clean = y[~nan_mask]  

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.25, random_state=42)


xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

print(confusion_matrix(y_test,xgb_predictions))
print('\n')
print(classification_report(y_test,xgb_predictions))


rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print(confusion_matrix(y_test,rf_predictions))
print('\n')
print(classification_report(y_test,rf_predictions))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()

# input layer
model.add(Dense(128, activation='relu', input_shape = (2073,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# first hidden layer
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# second hidden layer
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# third hidden layer
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# fourth hidden layer
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# output layer 
model.add(Dense(units=5, activation='softmax'))

optimizer = Adam(learning_rate=0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25, restore_best_weights=True)

model.fit(x=X_train_scaled, y=y_train, batch_size=128, epochs=600, validation_data=[X_test_scaled,y_test], callbacks=[early_stop])

loss_nn = pd.DataFrame(model.history.history)
loss_nn[['loss', 'val_loss']].plot()
plt.title('Training vs Validation Loss')
plt.show()


ids = df_test['PetID']
X_test_kaggle = df_test.drop(['RescuerID', 'image_path', 'Description', 'StateName', 'BreedName', 'ColorName'
                                , 'Name', 'PetID', 'ColorCombo', 'primary_animal', 'Mixed_Breed'],
                  axis=1)
nan_mask_test = X_test_kaggle.isnull().any(axis=1)
X_test_kaggle = X_test_kaggle[~nan_mask_test]
ids = ids[~nan_mask_test]


xgb_prediction_kaggle = xgb_model.predict(X_test_kaggle)

# submission XGBoost
submission = pd.DataFrame({'PetID': ids, 'AdoptionSpeed':xgb_prediction_kaggle})
submission.to_csv('submission.csv', index = False)


print(os.listdir("/kaggle/working"))




