import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os


# Load and preprocess data
def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]


def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))


def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)


def augment_images(images):
    """Apply data augmentation"""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(images, batch_size=16, shuffle=True)


# Multimodal Model Definition
class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        # Use XLM-Roberta for multilingual capabilities
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.bert_model = TFXLMRobertaModel.from_pretrained('xlm-roberta-base')

    def build_model(self):
        # Image branch (VGG19 with fine-tuning)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)

        for layer in vgg19.layers[:-8]:  # Fine-tune deeper layers
            layer.trainable = False

        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        image_features = LayerNormalization()(x)

        # Text branch (XLM-Roberta)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])
        pooled_output = bert_outputs[1]  # Use the pooled output
        x = Dense(256, activation='relu')(pooled_output)
        x = Dropout(0.4)(x)
        text_features = LayerNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        optimizer = Adam(learning_rate=1e-4)  # Reduce learning rate
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_text(self, texts):
        """Tokenize texts using XLM-Roberta tokenizer"""
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']


# Model Training
def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()

    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    # Data augmentation for images
    train_images_augmented = augment_images(train_images)

    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks
    )

    return model, history


# Main Function
def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )

    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())

    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)

    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])

    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )

    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )

    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])

    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]

    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

    return model, history


if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os
import math


# Load and preprocess data
def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]


def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))


def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)


def augment_images(images):
    """Apply stronger data augmentation"""
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    return datagen.flow(images, batch_size=16, shuffle=True)


# Multimodal Model Definition
class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        # Use XLM-Roberta for multilingual capabilities
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.bert_model = TFXLMRobertaModel.from_pretrained('xlm-roberta-base')

    def build_model(self):
        # Image branch (VGG19 with fine-tuning)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)

        for layer in vgg19.layers[:-8]:  # Fine-tune deeper layers
            layer.trainable = False

        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = Dropout(0.4)(x)
        image_features = BatchNormalization()(x)

        # Text branch (XLM-Roberta)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])
        pooled_output = bert_outputs[1]  # Use the pooled output
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(pooled_output)
        x = Dropout(0.4)(x)
        text_features = BatchNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        optimizer = Adam(learning_rate=1e-4)  # Reduce learning rate
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_text(self, texts):
        """Tokenize texts using XLM-Roberta tokenizer"""
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']


# Cosine Annealing Learning Rate Scheduler
def cosine_decay(epoch):
    initial_lr = 1e-4
    return initial_lr * (1 + math.cos(epoch * math.pi / 20)) / 2


# Model Training
def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()

    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)

    # Compute class weights for imbalanced datasets
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,  # Stop earlier
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.LearningRateScheduler(cosine_decay)
    ]

    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        class_weight=class_weights,
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks
    )

    return model, history


# Main Function
def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )

    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())

    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)

    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])

    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )

    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )

    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])

    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]

    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

    return model, history


if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)
        
        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False
            
        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)
        
        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)
        
        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=16,
        
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)
        
        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False
            
        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)
        
        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)
        
        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=30):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=16,
        
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)
        
        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False
            
        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)
        
        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(512, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)
        
        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(512, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=10):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=8,
        callbacks=callbacks
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)
        
        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False
            
        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)
        
        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(512, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)
        
        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(512, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=30):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=8,
        callbacks=callbacks
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)
        
        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False
            
        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)
        
        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(512, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)
        
        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(512, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=8,
        
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)
        
        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False
            
        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)
        
        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)
        
        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=8,
        
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)
        
        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False
            
        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)
        
        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        
        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)
        
        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )
        
        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=16,
        
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from transformers import TFBertModel, BertTokenizer
import os
from tqdm import tqdm


def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]


def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))


def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)


class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

    def build_model(self):
        # Image branch (VGG19)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)

        # Fine-tune only the top layers
        for layer in vgg19.layers[:-4]:
            layer.trainable = False

        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)

        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']


def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, class_weights, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()

    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]

    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        class_weight=class_weights,
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks
    )

    return model, history


def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )

    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())

    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)

    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )

    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs,
        class_weights_dict
    )

    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])

    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)

    # Evaluate model
    print(classification_report(test_df['Label_Sentiment'].map(label_map), predicted_labels))
    print(confusion_matrix(test_df['Label_Sentiment'].map(label_map), predicted_labels))

    # Save predictions
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")


if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import math
import tensorflow_addons as tfa

# Load and preprocess data
def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]


def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))


def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)


def augment_images(images):
    """Apply stronger data augmentation"""
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    return datagen.flow(images, batch_size=16, shuffle=True)


# Multimodal Model Definition
class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        # Use XLM-Roberta for multilingual capabilities
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.bert_model = TFXLMRobertaModel.from_pretrained('xlm-roberta-base')

    def build_model(self):
        # Image branch (VGG19 with fine-tuning)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)

        for layer in vgg19.layers[:-8]:  # Fine-tune deeper layers
            layer.trainable = False

        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = Dropout(0.4)(x)
        image_features = BatchNormalization()(x)

        # Text branch (XLM-Roberta)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])
        pooled_output = bert_outputs[1]  # Use the pooled output
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(pooled_output)
        x = Dropout(0.4)(x)
        text_features = BatchNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        return model

    def prepare_text(self, texts):
        """Tokenize texts using XLM-Roberta tokenizer"""
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']


# Cosine Annealing Learning Rate Scheduler
def lr_schedule(epoch):
    initial_lr = 1e-4
    if epoch < 5:  # Warm-up phase
        return initial_lr * (epoch + 1) / 5
    else:
        return initial_lr * tf.math.exp(-0.1 * (epoch - 5))


# Model Training
def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()

    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)

    # Compute class weights for imbalanced datasets
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))

    # Focal loss for better handling of class imbalance
    loss = tfa.losses.SigmoidFocalCrossEntropy()

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=loss,
        metrics=['accuracy']
    )

    # Callbacks
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        lr_callback
    ]

    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        class_weight=class_weights,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks
    )

    return model, history


# Main Function
def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )

    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())

    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)

    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])

    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )

    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )

    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])

    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]

    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

    return model, history


if __name__ == "__main__":
    main()



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Concatenate, LayerNormalization, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import math

# Load and preprocess data
def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]


def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))


def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)


def augment_images(images):
    """Apply stronger data augmentation"""
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    return datagen.flow(images, batch_size=16, shuffle=True)


# Multimodal Model Definition
class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        # Use XLM-Roberta for multilingual capabilities
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self.bert_model = TFXLMRobertaModel.from_pretrained('xlm-roberta-base')

    def build_model(self):
        # Image branch (VGG19 with fine-tuning)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        vgg19 = VGG19(weights='imagenet', include_top=False)

        for layer in vgg19.layers[:-8]:  # Fine-tune deeper layers
            layer.trainable = False

        x = vgg19(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = Dropout(0.4)(x)
        image_features = BatchNormalization()(x)

        # Text branch (XLM-Roberta)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])
        pooled_output = bert_outputs[1]  # Use the pooled output
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(pooled_output)
        x = Dropout(0.4)(x)
        text_features = BatchNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        return model

    def prepare_text(self, texts):
        """Tokenize texts using XLM-Roberta tokenizer"""
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']


# Cosine Annealing Learning Rate Scheduler
def lr_schedule(epoch):
    initial_lr = 1e-4
    if epoch < 5:  # Warm-up phase
        return initial_lr * (epoch + 1) / 5
    else:
        return initial_lr * tf.math.exp(-0.1 * (epoch - 5))


# Model Training
def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()

    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)

    # Convert labels to one-hot encoding
    train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=3)
    val_labels_onehot = tf.keras.utils.to_categorical(val_labels, num_classes=3)

    # Compute class weights for imbalanced datasets
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        lr_callback
    ]

    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels_onehot,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels_onehot
        ),
        class_weight=class_weights,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks
    )

    return model, history


# Main Function
def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )

    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())

    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)

    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])

    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )

    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )

    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])

    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)

    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]

    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

    return model, history


if __name__ == "__main__":
    main()



pip install tensorflow numpy pandas transformers vit-keras tqdm scikit-learn pillow



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
from vit_keras import vit
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.imagenet_utils.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (Vision Transformer)
        vit_model = vit.vit_b16(
            image_size=224,
            pretrained=True,
            include_top=False,
            pretrained_top=False
        )
        image_input = Input(shape=(224, 224, 3), name='image_input')
        x = vit_model(image_input)  # Output: (None, 768)
        x = Dense(256, activation='relu')(x)  # Add Dense layer instead of pooling
        x = Dropout(0.3)(x)
        image_features = LayerNormalization()(x)

        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu')(pooled_output)
        x = Dropout(0.3)(x)
        text_features = LayerNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



pip install tensorflow numpy pandas transformers vit-keras tqdm scikit-learn pillow



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
from vit_keras import vit
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.imagenet_utils.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (Vision Transformer)
        vit_model = vit.vit_b16(
            image_size=224,
            pretrained=True,
            include_top=False,
            pretrained_top=False
        )
        image_input = Input(shape=(224, 224, 3), name='image_input')
        x = vit_model(image_input)  # Output: (None, 768)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Add Dense layer with L2 regularization
        x = Dropout(0.4)(x)
        image_features = LayerNormalization()(x)

        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(pooled_output)
        x = Dropout(0.4)(x)
        text_features = LayerNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
        x = Dropout(0.4)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 if epoch > 10 else 2e-5)
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=8,  # Decreased batch size for better generalization
        class_weight=class_weight_dict,  # Add class weights
        callbacks=callbacks
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()




pip install tensorflow pandas numpy tqdm scikit-learn transformers vit-keras



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
from vit_keras import vit
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = tf.keras.applications.imagenet_utils.preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

# Image augmentation
def augment_images(images):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(images, batch_size=len(images), shuffle=False).next()

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (Vision Transformer)
        vit_model = vit.vit_b16(
            image_size=224,
            pretrained=True,
            include_top=False,
            pretrained_top=False
        )
        image_input = Input(shape=(224, 224, 3), name='image_input')
        x = vit_model(image_input)  # Output: (None, 768)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Add Dense layer with L2 regularization
        x = Dropout(0.4)(x)
        image_features = LayerNormalization()(x)

        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(pooled_output)
        x = Dropout(0.4)(x)
        text_features = LayerNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
        x = Dropout(0.4)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Augment training images
    train_images = augment_images(train_images)

    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 if epoch > 10 else 2e-5)
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=8,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



pip install tensorflow pandas numpy tqdm scikit-learn transformers vit-keras langdetect googletrans==4.0.0-rc1



import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, LayerNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from langdetect import detect
from googletrans import Translator

translator = Translator()

# Bengali stopwords list (manually curated or from libraries like bnltk)
BENGALI_STOPWORDS = set(["এবং", "কিন্তু", "যদি", "তবে", "অতএব", "অথচ", "যেমন", "তেমন", "কেন", "কখন", "যা", "তাহলে"])

# Function to detect language and transliterate Banglish to Bengali
def preprocess_text(text):
    try:
        lang = detect(text)
        if lang == "bn":  # Bengali
            text = text
        elif lang == "en":  # English
            text = text.lower()  # Convert to lowercase
        else:  # Banglish or other languages
            text = translator.translate(text, src="en", dest="bn").text

        # Remove special characters and punctuation
        text = ''.join(e for e in text if e.isalnum() or e.isspace())
        # Remove stopwords
        text = ' '.join(word for word in text.split() if word not in BENGALI_STOPWORDS)
    except Exception as e:
        print(f"Error processing text: {text}, {e}")
        return ""
    return text

def load_data(train_path, test_path):
    """Load training and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Preprocess captions
    train_df['Captions'] = train_df['Captions'].apply(preprocess_text)
    test_df['Captions'] = test_df['Captions'].apply(preprocess_text)

    return train_df, test_df

def get_image_paths(directory, image_names):
    """Get full paths for images"""
    image_paths = {img: os.path.join(directory, img) for img in image_names}
    return [image_paths[img] for img in image_names if img in image_paths]

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = vgg_preprocess(img)  # VGG19 preprocessing
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return np.zeros(target_size + (3,))

def process_images(image_paths, target_size=(224, 224)):
    """Process all images with progress bar"""
    images = []
    for path in tqdm(image_paths, desc="Processing images"):
        img = preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)

class MultimodalSentimentModel:
    def __init__(self, num_classes=3, max_length=128):
        self.num_classes = num_classes
        self.max_length = max_length
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        
    def build_model(self):
        # Image branch (VGG19)
        vgg19_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        for layer in vgg19_base.layers:
            layer.trainable = False  # Freeze VGG19 layers

        image_input = Input(shape=(224, 224, 3), name='image_input')
        x = vgg19_base(image_input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.4)(x)
        image_features = LayerNormalization()(x)

        # Text branch (BERT)
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')

        bert_outputs = self.bert_model([input_ids, attention_mask])[0]
        pooled_output = tf.reduce_mean(bert_outputs, axis=1)
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(pooled_output)
        x = Dropout(0.4)(x)
        text_features = LayerNormalization()(x)

        # Combine features
        combined = Concatenate()([image_features, text_features])
        x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(combined)
        x = Dropout(0.4)(x)
        x = LayerNormalization()(x)
        x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(
            inputs=[image_input, input_ids, attention_mask],
            outputs=outputs
        )

        optimizer = Adam(learning_rate=2e-5)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def prepare_text(self, texts):
        """Tokenize texts using BERT tokenizer"""
        encodings = self.bert_tokenizer(
            texts.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

def train_model(train_images, train_texts, train_labels, val_images, val_texts, val_labels, epochs=20):
    # Create model instance
    model_handler = MultimodalSentimentModel()
    model = model_handler.build_model()
    
    # Prepare text data
    train_input_ids, train_attention_mask = model_handler.prepare_text(train_texts)
    val_input_ids, val_attention_mask = model_handler.prepare_text(val_texts)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 if epoch > 10 else 2e-5)
    ]
    
    # Train model
    history = model.fit(
        {
            'image_input': train_images,
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        },
        train_labels,
        validation_data=(
            {
                'image_input': val_images,
                'input_ids': val_input_ids,
                'attention_mask': val_attention_mask
            },
            val_labels
        ),
        epochs=epochs,
        batch_size=8,
        class_weight=class_weight_dict,
        
    )
    
    return model, history

def main():
    # Load data
    train_df, test_df = load_data(
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/train.csv',
        '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/test.csv'
    )
    
    # Get image paths
    memes_folder = '/kaggle/input/multimodal-sentiment-analysis-cuet-nlp/Memes/Memes'
    train_image_paths = get_image_paths(memes_folder, train_df['image_name'].tolist())
    test_image_paths = get_image_paths(memes_folder, test_df['image_name'].tolist())
    
    # Process images
    train_images = process_images(train_image_paths)
    test_images = process_images(test_image_paths)
    
    # Convert labels
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    train_labels = np.array([label_map[label] for label in train_df['Label_Sentiment']])
    
    # Split data
    train_imgs, val_imgs, train_texts, val_texts, train_labs, val_labs = train_test_split(
        train_images, train_df['Captions'],
        train_labels, test_size=0.15,
        random_state=42, stratify=train_labels
    )
    
    # Train model
    model, history = train_model(
        train_imgs, train_texts, train_labs,
        val_imgs, val_texts, val_labs
    )
    
    # Prepare test data
    model_handler = MultimodalSentimentModel()
    test_input_ids, test_attention_mask = model_handler.prepare_text(test_df['Captions'])
    
    # Make predictions
    predictions = model.predict({
        'image_input': test_images,
        'input_ids': test_input_ids,
        'attention_mask': test_attention_mask
    })
    predicted_labels = np.argmax(predictions, axis=1)
    # Convert predictions to labels
    reverse_label_map = {v: k for k, v in label_map.items()}
    test_df['Label'] = [reverse_label_map[label] for label in predicted_labels]
    
    # Save predictions
    test_df[['Id', 'Label']].to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    return model, history

if __name__ == "__main__":
    main()



import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Image Preprocessing and Augmentation
image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for ImageNet
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define Dataset
class MultimodalDataset(Dataset):
    def __init__(self, image_paths, text_data, labels, transform=None, tokenizer=None):
        self.image_paths = image_paths
        self.text_data = text_data
        self.labels = labels
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize text
        text = self.text_data[idx]
        encoded_text = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )

        # Extract inputs and attention mask
        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)

        label = torch.tensor(self.labels[idx])
        return image, input_ids, attention_mask, label

# Define Image Model (ResNet)
class ImageModel(nn.Module):
    def __init__(self, output_dim):
        super(ImageModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

# Define Text Model (BERT)
class TextModel(nn.Module):
    def __init__(self, output_dim):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls_token_output)

# Define Combined Model
class MultimodalModel(nn.Module):
    def __init__(self, img_output_dim, text_output_dim, final_output_dim):
        super(MultimodalModel, self).__init__()
        self.image_model = ImageModel(img_output_dim)
        self.text_model = TextModel(text_output_dim)
        self.fc = nn.Linear(img_output_dim + text_output_dim, final_output_dim)

    def forward(self, image, input_ids, attention_mask):
        img_features = self.image_model(image)
        text_features = self.text_model(input_ids, attention_mask)
        combined_features = torch.cat((img_features, text_features), dim=1)
        return self.fc(combined_features)

# Training Function
def train(model, dataloaders, criterion, optimizer, scheduler, device, epochs=10):
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            preds, targets = [], []

            for images, input_ids, attention_mask, labels in dataloaders[phase]:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images, input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    _, predictions = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                        optimizer.step()

                # Track metrics
                running_loss += loss.item() * images.size(0)
                preds.extend(predictions.cpu().numpy())
                targets.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(targets, preds)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Track the best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model.pth")
        scheduler.step()

    print(f"Best Validation Accuracy: {best_acc:.4f}")

# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(images, input_ids, attention_mask)
            _, predictions = torch.max(outputs, 1)
            preds.extend(predictions.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Compute Metrics
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="weighted")
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Main Function
if __name__ == "__main__":
    # Load dataset (replace with your dataset)
    from PIL import Image
    import os

    # Example: Replace with actual data
    image_paths = ["path_to_images/image1.jpg", "path_to_images/image2.jpg"]  # Add your image paths
    text_data = ["Sample text 1", "Sample text 2"]  # Add your text data
    labels = [0, 1]  # Add your labels

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = {
        "train": MultimodalDataset(image_paths, text_data, labels, image_transforms["train"], tokenizer),
        "val": MultimodalDataset(image_paths, text_data, labels, image_transforms["val"], tokenizer),
    }
    dataloaders = {
        phase: DataLoader(dataset[phase], batch_size=4, shuffle=(phase == "train")) for phase in ["train", "val"]
    }

    # Initialize model, loss, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalModel(img_output_dim=128, text_output_dim=128, final_output_dim=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train and Evaluate
    train(model, dataloaders, criterion, optimizer, scheduler, device, epochs=10)
    evaluate(model, dataloaders["val"], device)


