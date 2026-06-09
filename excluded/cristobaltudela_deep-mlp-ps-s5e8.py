import cudf
import cupy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')


class BankDepositPredictor:
    """
    Neural Network model for predicting bank term deposit subscriptions using Rapids
    """
    
    def __init__(self, config=None):
        """
        Initialize the model with configurable parameters
        """
        # Default configuration
        self.config = {
            'hidden_layers': [2048, 512, 128, 32, 8],  # Hidden layer sizes
            'dropout_rate': 0.15,            # Dropout rate for regularization
            'learning_rate': 0.007,         # Learning rate for optimizer
            'batch_size': 32,                # Batch size for training
            'epochs': 70,                    # Number of training epochs
            'validation_split': 0.1,         # Validation split ratio
            'early_stopping_patience': 10,    # Early stopping patience
            'activation': 'relu',            # Activation function for hidden layers
            'optimizer': 'adamw',             # Optimizer type
            'random_state': 42               # Random state for reproducibility
        }
        
        # Update config if provided
        if config:
            self.config.update(config)
            
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the training data using cuDF
        """
        print("Loading data with cuDF...")
        
        # Load data using cuDF for GPU acceleration
        try:
            df = cudf.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
        except:
            # Fallback to pandas if cuDF fails
            print("cuDF failed, using pandas...")
            df = pd.read_csv(file_path)
            df = cudf.from_pandas(df)
        
        # Display basic info about the dataset
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Remove 'id' column if it exists
        if 'id' in df.columns:
            df = df.drop(['id'], axis=1)
            print("Removed 'id' column")
        
        # Separate target variable
        if 'y' not in df.columns:
            raise ValueError("Target column 'y' not found in dataset")
            
        y = df['y'].copy()
        X = df.drop(['y'], axis=1)
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Convert to pandas for preprocessing (Rapids sklearn preprocessing)
        X_pd = X.to_pandas()
        y_pd = y.to_pandas()
        
        # Identify categorical and numerical columns
        categorical_columns = X_pd.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = X_pd.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Categorical columns: {categorical_columns}")
        print(f"Numerical columns: {numerical_columns}")
        
        # Encode categorical variables
        X_encoded = X_pd.copy()
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_pd[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded column '{col}': {len(le.classes_)} unique values")
        
        # Encode target variable (y)
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y_pd.astype(str))
        self.label_encoders['target'] = target_encoder
        
        print(f"Target classes: {target_encoder.classes_}")
        print(f"Target distribution: {np.bincount(y_encoded)}")
        
        # Store feature columns
        self.feature_columns = X_encoded.columns.tolist()
        
        return X_encoded.values, y_encoded
    
    def build_model(self, input_dim):
        """
        Build the neural network model with sigmoid output
        """
        print(f"Building neural network model with input dimension: {input_dim}")
        
        # Importar inicializadores y activaciones avanzadas
        from tensorflow.keras import initializers, regularizers
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.activations import swish
        
        model = keras.Sequential()
        
        # Input layer con inicialización HeNormal
        model.add(layers.Dense(
            self.config['hidden_layers'][0], 
            kernel_initializer=initializers.HeNormal(seed=self.config['random_state']),
            use_bias=False,  # BatchNorm incluye bias
            input_shape=(input_dim,)
        ))
        model.add(layers.BatchNormalization())
        model.add(Activation(swish))
        model.add(layers.Dropout(self.config['dropout_rate']))
        
        # Hidden layers con BatchNorm y Swish
        for i, units in enumerate(self.config['hidden_layers'][1:], 1):
            model.add(layers.Dense(
                units, 
                kernel_initializer=initializers.HeNormal(seed=self.config['random_state']),
                use_bias=False
            ))
            model.add(layers.BatchNormalization())
            model.add(Activation(swish))
            model.add(layers.Dropout(self.config['dropout_rate']))
        
        # Output layer con sigmoid
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model con nuevo learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train(self, X, y):
        """
        Train the neural network model
        """
        print("Starting model training...")
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config['validation_split'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_precision, val_recall, val_auc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nValidation Results:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"AUC ROC: {val_auc:.4f}")
        
        return history
    
    def predict(self, X):
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get probability predictions
        probabilities = self.model.predict(X)
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        predictions = (probabilities > 0.5).astype(int).flatten()
        #predictions = probabilities.flatten()
        
        return predictions, probabilities.flatten()
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        """
        predictions, probabilities = self.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        auc_roc = roc_auc_score(y, probabilities)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC ROC: {auc_roc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y, predictions, 
                                  target_names=self.label_encoders['target'].classes_))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, predictions))
        
        return accuracy, predictions, probabilities, auc_roc
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_test_data(self, test_file_path, submission_file_path):
        """
        Load test data, make predictions, and generate submission file
        """
        print("Loading test data...")
        
        # Load test data using cuDF
        try:
            test_df = cudf.read_csv(test_file_path)
            print(f"Test data loaded successfully. Shape: {test_df.shape}")
        except:
            # Fallback to pandas if cuDF fails
            print("cuDF failed, using pandas...")
            test_df = pd.read_csv(test_file_path)
            test_df = cudf.from_pandas(test_df)
        
        # Store the IDs for submission
        test_ids = test_df['id'].to_pandas().values
        
        # Remove 'id' column for prediction
        test_features = test_df.drop(['id'], axis=1)
        
        print(f"Test features shape: {test_features.shape}")
        print(f"Test columns: {list(test_features.columns)}")
        
        # Convert to pandas for preprocessing
        test_features_pd = test_features.to_pandas()
        
        # Apply the same preprocessing as training data
        test_encoded = test_features_pd.copy()
        
        # Encode categorical variables using the same encoders from training
        for col in test_features_pd.columns:
            if col in self.label_encoders:
                # Use the encoder fitted during training
                le = self.label_encoders[col]
                # Handle unseen categories by using the most frequent class
                test_encoded[col] = test_features_pd[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
                )
                print(f"Encoded test column '{col}' using training encoder")
            elif test_features_pd[col].dtype == 'object':
                # If it's a categorical column not seen in training, create a simple encoding
                print(f"Warning: Column '{col}' not found in training encoders, creating new encoding")
                unique_vals = test_features_pd[col].unique()
                encoding_map = {val: i for i, val in enumerate(unique_vals)}
                test_encoded[col] = test_features_pd[col].map(encoding_map)
        
        # Ensure the feature order matches training
        if hasattr(self, 'feature_columns') and self.feature_columns:
            # Reorder columns to match training
            missing_cols = set(self.feature_columns) - set(test_encoded.columns)
            extra_cols = set(test_encoded.columns) - set(self.feature_columns)
            
            if missing_cols:
                print(f"Warning: Missing columns in test data: {missing_cols}")
                # Add missing columns with default value 0
                for col in missing_cols:
                    test_encoded[col] = 0
            
            if extra_cols:
                print(f"Warning: Extra columns in test data: {extra_cols}")
                # Remove extra columns
                test_encoded = test_encoded.drop(columns=list(extra_cols))
            
            # Reorder to match training
            test_encoded = test_encoded[self.feature_columns]
        
        print(f"Final test features shape: {test_encoded.shape}")
        
        # Make predictions
        if self.model is None:
            raise ValueError("Model not trained yet. Train the model first.")
        
        print("Making predictions on test data...")
        test_probabilities = self.model.predict(test_encoded.values)
        #test_predictions = (test_probabilities > 0.5).astype(int).flatten()
        test_predictions = test_probabilities.flatten()
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'id': test_ids,
            'y': test_predictions
        })
        
        # Save submission file
        submission_df.to_csv(submission_file_path, index=False)
        print(f"Submission file saved to: {submission_file_path}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Prediction distribution: {np.bincount(test_predictions)}")
        
        return submission_df, test_probabilities



predictor = BankDepositPredictor(config=None)

# Load and preprocess data
data_path = '/kaggle/input/playground-series-s5e8/train.csv'
X, y = predictor.load_and_preprocess_data(data_path)
    
# Train the model
history = predictor.train(X, y)


accuracy, predictions, probabilities, auc_roc = predictor.evaluate(X, y)
# Save the model
# predictor.save_model('/kaggle/working/bank_deposit_model.h5')


print("\n" + "="*50)
print("GENERATING TEST PREDICTIONS...")
print("="*50)
test_data_path = '/kaggle/input/playground-series-s5e8/test.csv'
submission_path = '/kaggle/working/submission.csv'
    
try:
    submission_df, test_probabilities = predictor.predict_test_data(test_data_path, submission_path)
    print(f"\nTest predictions completed successfully!")
    print(f"Submission file created: {submission_path}")
except Exception as e:
    print(f"Error generating test predictions: {e}")
    print("Continuing without test predictions...")


print("\n" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Final AUC ROC: {auc_roc:.4f}")
print("="*50)

