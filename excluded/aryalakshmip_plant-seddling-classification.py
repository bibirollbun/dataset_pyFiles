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


# Install required packages
!pip install joblib

# Import libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import joblib

# Check GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Set up directories
INPUT_DIR = '/kaggle/input/plant-seedlings-classification/'
OUTPUT_DIR = '/kaggle/working/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class PlantSeedlingClassifier:
    def __init__(self, data_dir, target_size=(224, 224), model_name='resnet50'):
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        self.data_dir = data_dir
        self.target_size = target_size
        self.model_name = model_name.lower()
        self.label_encoder = LabelEncoder()
        self.data_augmentation = self.create_data_augmentation()

    def create_data_augmentation(self):
        return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

    def load_and_preprocess_data(self):
        images = []
        labels = []
        print("\nLoading images from directories...")
        
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                print(f"Processing class: {class_dir}")
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = load_img(img_path, target_size=self.target_size)
                        img_array = img_to_array(img)
                        img_array = img_array / 255.0
                        images.append(img_array)
                        labels.append(class_dir)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {str(e)}")
        
        X = np.array(images)
        y = self.label_encoder.fit_transform(labels)
        y = to_categorical(y)
        
        print(f"\nLoaded {len(images)} images across {len(np.unique(labels))} classes")
        return X, y

    def build_model(self, num_classes):
        try:
            print(f"\nBuilding model with {self.model_name} architecture...")
            if self.model_name == 'vgg16':
                base_model = VGG16(weights='imagenet', include_top=False, 
                                input_shape=self.target_size + (3,))
            elif self.model_name == 'resnet50':
                base_model = ResNet50(weights='imagenet', include_top=False,
                                    input_shape=self.target_size + (3,))
            elif self.model_name == 'inceptionv3':
                base_model = InceptionV3(weights='imagenet', include_top=False,
                                    input_shape=self.target_size + (3,))
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            for layer in base_model.layers:
                layer.trainable = False
                
            return model
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def train_and_evaluate_model(self, model_name, X_train, X_val, X_test, y_train, y_val, y_test, class_names, output_dir):
        try:
            print(f"\n=== Training {model_name.upper()} ===")
            self.model_name = model_name
            
            model = self.build_model(y_train.shape[1])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            model_path = os.path.join(output_dir, f'best_model_{model_name}.keras')
            callbacks = [
                ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            batch_size = 32
            steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
            
            train_generator = self.data_augmentation.flow(
                X_train, y_train,
                batch_size=batch_size,
                shuffle=True
            )
            
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            report = classification_report(y_test_classes, y_pred_classes, 
                                        target_names=class_names)
            with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
            
            self.plot_confusion_matrix(y_test_classes, y_pred_classes, 
                                     class_names, model_dir, model_name)
            self.plot_training_history(history, model_dir, model_name)
            
            return {
                'model_name': model_name,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'history': history.history,
                'report': report
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            raise

    def plot_confusion_matrix(self, y_true, y_pred, class_names, output_dir, model_name):
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_training_history(self, history, output_dir, model_name):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{model_name.upper()} - Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name.upper()} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()

    def compare_models(self, models_results, output_dir):
        comparison_dir = os.path.join(output_dir, 'model_comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        for result in models_results:
            plt.plot(result['history']['val_accuracy'], 
                    label=f"{result['model_name']} (Test Acc: {result['test_accuracy']:.4f})")
        plt.title('Model Comparison - Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(comparison_dir, 'accuracy_comparison.png'))
        plt.close()
        
        comparison_table = "\nModel Comparison Results:\n"
        comparison_table += "-" * 50 + "\n"
        comparison_table += "Model         Test Accuracy     Test Loss\n"
        comparison_table += "-" * 50 + "\n"
        
        for result in models_results:
            comparison_table += f"{result['model_name']:<12} {result['test_accuracy']:>13.4f} {result['test_loss']:>12.4f}\n"
        
        with open(os.path.join(comparison_dir, 'comparison_results.txt'), 'w') as f:
            f.write(comparison_table)
        
        return comparison_table


def main():
    try:
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Setup directories
        output_dir = '/kaggle/working/outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize classifier
        data_dir = '/kaggle/input/plant-seedlings-classification/train'
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory not found: {data_dir}")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        classifier = PlantSeedlingClassifier(data_dir)
        X, y = classifier.load_and_preprocess_data()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Train and evaluate each model
        models_to_evaluate = ['resnet50', 'vgg16', 'inceptionv3']
        models_results = []
        
        for model_name in models_to_evaluate:
            result = classifier.train_and_evaluate_model(
                model_name, X_train, X_val, X_test,
                y_train, y_val, y_test,
                classifier.label_encoder.classes_,
                output_dir
            )
            models_results.append(result)
        
        # Compare models
        comparison_results = classifier.compare_models(models_results, output_dir)
        print(comparison_results)
        
        print(f"\nAll models trained and evaluated successfully!")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        return 1

if __name__ == "__main__":
    main()











import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys

class PlantSeedlingClassifier:
    def __init__(self, data_dir, target_size=(224, 224), model_name='resnet50'):
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        self.data_dir = data_dir
        self.target_size = target_size
        self.model_name = model_name.lower()
        self.label_encoder = LabelEncoder()
        self.data_augmentation = self.create_data_augmentation()
    
    def load_and_preprocess_data(self):
        images = []
        labels = []
        print("\nLoading images from directories...")
        
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                print(f"Processing class: {class_dir}")
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = load_img(img_path, target_size=self.target_size)
                        img_array = img_to_array(img)
                        img_array = img_array / 255.0
                        images.append(img_array)
                        labels.append(class_dir)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {str(e)}")
        
        X = np.array(images)
        y = self.label_encoder.fit_transform(labels)
        y = to_categorical(y)
        
        print(f"\nLoaded {len(images)} images across {len(np.unique(labels))} classes")
        return X, y

    def create_data_augmentation(self):
        return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    
    def build_model(self, num_classes):
        try:
            print(f"\nBuilding model with {self.model_name} architecture...")
            if self.model_name == 'vgg16':
                base_model = VGG16(weights='imagenet', include_top=False, 
                                input_shape=self.target_size + (3,))
            elif self.model_name == 'resnet50':
                base_model = ResNet50(weights='imagenet', include_top=False,
                                    input_shape=self.target_size + (3,))
            elif self.model_name == 'inceptionv3':
                base_model = InceptionV3(weights='imagenet', include_top=False,
                                    input_shape=self.target_size + (3,))
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")
            
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=base_model.input, outputs=predictions)
            
            for layer in base_model.layers:
                layer.trainable = False
                
            return model
        except Exception as e:
            print(f"Error building model: {str(e)}")
            raise

    def plot_sample_images(self, X, y, output_dir, num_samples=10):
        """Plot sample images from the dataset"""
        plt.figure(figsize=(20, 4))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(X[i])
            plt.title(f'Class: {self.label_encoder.inverse_transform([np.argmax(y[i])])[0]}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_images.png'))
        plt.close()

    def plot_data_distribution(self, y, output_dir):
        """Plot class distribution in the dataset"""
        class_counts = np.sum(y, axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(self.label_encoder.classes_, class_counts)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
        plt.close()

def main():
    try:
        # Set random seed
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Setup directories
        output_dir = os.path.join('/kaggle/working', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize classifier
        data_dir = '/kaggle/input/plant-seedlings-classification/train'
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory not found: {data_dir}")
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        classifier = PlantSeedlingClassifier(data_dir)
        X, y = classifier.load_and_preprocess_data()
        
        # Plot dataset visualizations
        print("\nGenerating dataset visualizations...")
        classifier.plot_sample_images(X, y, output_dir)
        classifier.plot_data_distribution(y, output_dir)
        
        # Split data
        print("\nSplitting dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Print dataset split information
        print(f"\nDataset splits:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train and evaluate models
        models_to_evaluate = ['resnet50', 'vgg16', 'inceptionv3']
        models_results = []
        
        for model_name in models_to_evaluate:
            try:
                print(f"\n{'='*50}")
                print(f"Training {model_name.upper()}")
                print(f"{'='*50}")
                
                # Calculate batch size and steps
                batch_size = 32
                steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
                
                # Create data generator
                train_datagen = ImageDataGenerator(
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest'
                )
                
                # Configure generator
                train_generator = train_datagen.flow(
                    X_train, y_train,
                    batch_size=batch_size,
                    shuffle=True
                )
                
                # Build and compile model
                model = classifier.build_model(y.shape[1])
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Setup callbacks
                model_path = os.path.join(output_dir, f'best_model_{model_name}.keras')
                callbacks = [
                    ModelCheckpoint(
                        filepath=model_path,
                        monitor='val_accuracy',
                        save_best_only=True,
                        mode='max'
                    ),
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
                
                # Train model
                history = model.fit(
                    train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate model
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
                print(f"\n{model_name.upper()} Test accuracy: {test_accuracy:.4f}")
                
                # Generate predictions
                y_pred = model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_test_classes = np.argmax(y_test, axis=1)
                
                # Save results
                model_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)
                
                # Save classification report
                report = classification_report(y_test_classes, y_pred_classes,
                                            target_names=classifier.label_encoder.classes_)
                with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
                    f.write(report)
                
                # Plot confusion matrix
                plt.figure(figsize=(12, 10))
                cm = confusion_matrix(y_test_classes, y_pred_classes)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=classifier.label_encoder.classes_,
                           yticklabels=classifier.label_encoder.classes_)
                plt.title(f'Confusion Matrix - {model_name.upper()}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
                plt.close()
                
                # Plot training history
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title(f'{model_name.upper()} - Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'])
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title(f'{model_name.upper()} - Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'])
                plt.tight_layout()
                plt.savefig(os.path.join(model_dir, 'training_history.png'))
                plt.close()
                
                # Store results
                models_results.append({
                    'model_name': model_name,
                    'test_accuracy': test_accuracy,
                    'test_loss': test_loss,
                    'history': history.history,
                    'report': report
                })
                
            except Exception as e:
                print(f"Error processing {model_name}: {str(e)}")
                continue
        
        # Generate model comparison if we have results
        if models_results:
            comparison_dir = os.path.join(output_dir, 'model_comparison')
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Plot accuracy comparison
            plt.figure(figsize=(12, 6))
            for result in models_results:
                plt.plot(result['history']['val_accuracy'],
                        label=f"{result['model_name']} (Test Acc: {result['test_accuracy']:.4f})")
            plt.title('Model Comparison - Validation Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(comparison_dir, 'accuracy_comparison.png'))
            plt.close()
            
            # Create and save comparison table
            comparison_table = "\nModel Comparison Results:\n"
            comparison_table += "-" * 50 + "\n"
            comparison_table += "Model         Test Accuracy     Test Loss\n"
            comparison_table += "-" * 50 + "\n"
            
            for result in models_results:
                comparison_table += f"{result['model_name']:<12} {result['test_accuracy']:>13.4f} {result['test_loss']:>12.4f}\n"
            
            print("\nFinal Results:")
            print(comparison_table)
            
            with open(os.path.join(comparison_dir, 'comparison_results.txt'), 'w') as f:
                f.write(comparison_table)
        
        print(f"\nAll results saved in: {output_dir}")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        return 1

if __name__ == "__main__":
    main()

