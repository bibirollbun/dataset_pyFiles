# ==========================
# Kaggle Notebook: Learnable KNN for Reddit Moderation - FIXED VERSION
# ==========================

import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

import numpy as np
import json
import os
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder

# === Label Encoder ===
class LabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}

    def fit(self, labels):
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def transform(self, labels):
        return [self.label_to_index.get(label, -1) for label in labels]  # Handle unseen labels

    def inverse_transform(self, indices):
        return [self.index_to_label.get(idx, "Unknown") for idx in indices]  # Handle unknown indices

    def to_dict(self):
        return {
            "label_to_index": self.label_to_index,
            "index_to_label": self.index_to_label
        }

    def from_dict(self, d):
        self.label_to_index = d["label_to_index"]
        self.index_to_label = {int(k): v for k, v in d["index_to_label"].items()}


# === Learnable KNN Model ===
class LibertyLearnableKNN:
    def __init__(
        self,
        model_path, 
        feature_size: int,
        num_classes: int,
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 16,
        weight_decay: float = 0.001,
        early_stopping_rounds: int = 50
    ):
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.weight_decay = weight_decay
        self.early_stopping_rounds = early_stopping_rounds

        self.weights = np.ones(self.feature_size)
        self.class_prototypes = np.zeros((self.num_classes, self.feature_size))

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.weights

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def _one_hot(self, label: int) -> np.ndarray:
        vec = np.zeros(self.num_classes)
        if 0 <= label < self.num_classes:  # Handle invalid labels
            vec[label] = 1
        return vec

    def fit(self, data: List[Tuple[np.ndarray, int]], encoder: LabelEncoder):
        print(f"Training {self.model_path} model...")
        best_accuracy = 0
        no_improve_rounds = 0

        np.random.shuffle(data)
        split_idx = int(0.9 * len(data))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        for epoch in range(self.epochs):
            np.random.shuffle(train_data)
            total_loss = 0
            correct = 0

            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i:i + self.batch_size]
                grad_batch = np.zeros_like(self.weights)

                for x, true_label in batch:
                    transformed_x = self._transform(x)
                    logits = np.dot(self.class_prototypes, transformed_x)
                    probs = self._softmax(logits)
                    pred_label = int(np.argmax(probs))

                    loss = -np.sum(self._one_hot(true_label) * np.log(probs + 1e-10))
                    total_loss += loss

                    if pred_label == true_label:
                        correct += 1

                    error = probs - self._one_hot(true_label)
                    grad_batch += np.dot(error, self.class_prototypes) * x

                grad_batch /= len(batch)
                grad_batch = np.clip(grad_batch, -10, 10)
                grad_batch += self.weight_decay * self.weights
                self.weights -= self.lr * grad_batch

            self._update_class_prototypes(train_data)

            train_accuracy = correct / len(train_data) * 100
            val_accuracy = self._evaluate(val_data)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                no_improve_rounds = 0
                self.save_model(encoder)
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= self.early_stopping_rounds:
                    print("⏹️ Early stopping triggered.")
                    break

    def _update_class_prototypes(self, data: List[Tuple[np.ndarray, int]]):
        sums = np.zeros((self.num_classes, self.feature_size))
        counts = np.zeros(self.num_classes)
        for x, label in data:
            if 0 <= label < self.num_classes:  # Only process valid labels
                sums[label] += self._transform(x)
                counts[label] += 1
        for i in range(self.num_classes):
            if counts[i] > 0:
                self.class_prototypes[i] = sums[i] / counts[i]

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        transformed_x = self._transform(x)
        logits = np.dot(self.class_prototypes, transformed_x)
        probs = self._softmax(logits)
        pred_label = int(np.argmax(probs))
        return pred_label, probs, logits

    def _evaluate(self, data: List[Tuple[np.ndarray, int]]) -> float:
        correct = 0
        for x, label in data:
            if 0 <= label < self.num_classes:  # Only evaluate valid labels
                pred, _, _ = self.predict(x)
                if pred == label:
                    correct += 1
        return correct / len(data) * 100 if len(data) > 0 else 0.0

    def save_model(self, encoder: LabelEncoder = None):
        model_data = {
            'weights': self.weights.tolist(),
            'prototypes': self.class_prototypes.tolist()
        }
        
        # save vectorizer vocab if available
        if hasattr(self, "vectorizer") and hasattr(self.vectorizer, "vocabulary_"):
            model_data["vectorizer_vocab"] = self.vectorizer.vocabulary_
        if encoder:
            model_data['encoder'] = encoder.to_dict()

        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)
        print(f"✅ model saved to {self.model_path}")

    def load_model(self) -> LabelEncoder:
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
                self.weights = np.array(model_data['weights'])
                self.class_prototypes = np.array(model_data['prototypes'])
                
                # rebuild vectorizer from saved vocab
                if "vectorizer_vocab" in model_data:
                    self.vectorizer = TfidfVectorizer(vocabulary=model_data["vectorizer_vocab"])
                    print("✅ Vectorizer loaded with saved vocabulary.")
                    
                if "encoder" in model_data:
                    encoder = LabelEncoder()
                    encoder.from_dict(model_data["encoder"])
                    print("✅ Encoder loaded.")
                    return encoder
                else:
                    print("⚠️ Encoder not found in saved model.")
        else:
            print("⚠️ model file not found.")
        return None


class NLPDataPipeline:
    def __init__(self, filepath: str,
                 text_cols=None,
                 target_col: str = "rule",
                 max_features: int = 3000,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        A full pipeline for loading, cleaning, encoding, vectorizing,
        and splitting NLP datasets.
        """
        self.filepath = filepath
        self.text_cols = text_cols or ['body', 'positive_example_1',
                                       'positive_example_2',
                                       'negative_example_1',
                                       'negative_example_2']
        self.target_col = target_col
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state

        self.data = None
        self.vectorizers = {}
        self.encoders = {}

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic cleaning for text."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # remove URLs
        text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove special chars
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess(self):
        if self.data is None:
            raise ValueError("Load data first with load_data().")

        # Fill missing values
        self.data.fillna({col: '' for col in self.text_cols}, inplace=True)
        
        if self.target_col in self.data.columns:
            if self.target_col == "rule_violation":
                self.data.fillna({self.target_col: 0}, inplace=True)
                self.data[self.target_col] = self.data[self.target_col].astype(int)
            else:
                self.data.fillna({self.target_col: "Unknown"}, inplace=True)

        # Clean text columns
        for col in self.text_cols:
            self.data[col] = self.data[col].apply(self.clean_text)

        return self.data

    def encode_categorical(self, col: str):
        le = SklearnLabelEncoder()
        le.fit(self.data[col])  
        self.data[col] = le.transform(self.data[col])
        self.encoders[col] = le

    def vectorize_text(self, col: str):
        tfidf = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))
        X = tfidf.fit_transform(self.data[col])
        self.vectorizers[col] = tfidf
        return X
        
    def transform_features(self, fitted_vectorizers, fitted_encoders=None):
        # Use fitted vectorizers from training - handle missing columns gracefully
        X_text_parts = []
        for col in self.text_cols:
            if col in self.data.columns and col in fitted_vectorizers:
                X_text_parts.append(fitted_vectorizers[col].transform(self.data[col]))
            else:
                # Create empty matrix if column is missing
                empty_matrix = csr_matrix((len(self.data), fitted_vectorizers[list(fitted_vectorizers.keys())[0]].vocabulary_.get('dummy', self.max_features)))
                X_text_parts.append(empty_matrix)
        
        X = hstack(X_text_parts) if X_text_parts else csr_matrix((len(self.data), 0))
        
        # Handle categorical if encoders given
        if fitted_encoders:
            cat_features = []
            for col in fitted_encoders:
                if col in self.data.columns:
                    try:
                        encoded_vals = fitted_encoders[col].transform(self.data[col])
                        cat_features.append(encoded_vals)
                    except ValueError:
                        # Handle unseen categories by using a default value
                        default_val = 0  # or len(fitted_encoders[col].classes_) for "unknown" category
                        encoded_vals = np.full(len(self.data), default_val)
                        cat_features.append(encoded_vals)
                else:
                    # If column is missing, use default values
                    encoded_vals = np.zeros(len(self.data))
                    cat_features.append(encoded_vals)
            
            if cat_features:
                X_num = np.array(cat_features).T
                X = hstack([X, csr_matrix(X_num)])
        
        if self.target_col in self.data.columns:
            y = self.data[self.target_col].values
        else:
            y = None
        return X, y
    
    def prepare_features(self):
        # Encode categorical columns
        cat_cols = []
        for col in ['subreddit', 'rule']:
            if col in self.data.columns:
                self.encode_categorical(col)
                cat_cols.append(col)

        # Vectorize text columns
        X_text_parts = []
        for col in self.text_cols:
            if col in self.data.columns:
                X_text_parts.append(self.vectorize_text(col))

        # Combine TF-IDF features
        X = hstack(X_text_parts) if X_text_parts else csr_matrix((len(self.data), 0))

        # Add encoded categorical features
        if cat_cols:
            X_num = np.array(self.data[cat_cols])
            X = hstack([X, csr_matrix(X_num)])

        if self.target_col in self.data.columns:
            y = self.data[self.target_col].values
        else:
            y = None  
            
        return X, y

    def split(self):
        X, y = self.prepare_features()
        if y is not None:
            return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        else:
            # For test data without labels
            return X, None

# ==========================
# MAIN EXECUTION - FIXED
# ==========================

try:
    # Train pipeline
    print("Loading and preprocessing training data...")
    train_pipeline = NLPDataPipeline("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    train_data_df = train_pipeline.load_data()
    print(f"Training data shape: {train_data_df.shape}")
    
    train_pipeline.preprocess()
    X_train, X_test, y_train, y_test = train_pipeline.split()

    # Convert to dense
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Train label encoder
    encoder = LabelEncoder()
    all_labels = y_train.tolist() + y_test.tolist()
    encoder.fit(all_labels)
    y_train_enc = encoder.transform(y_train.tolist())
    y_test_enc = encoder.transform(y_test.tolist())

    # Prepare data
    train_data = [(X_train[i], y_train_enc[i]) for i in range(len(X_train))]
    test_data = [(X_test[i], y_test_enc[i]) for i in range(len(X_test))]

    # Train model
    feature_size = X_train.shape[1]
    num_classes = len(set(y_train_enc))

    print(f"Feature size: {feature_size}, Number of classes: {num_classes}")

    model = LibertyLearnableKNN(
        model_path="/kaggle/working/reddit_model.json",
        feature_size=feature_size,
        num_classes=num_classes,
        lr=0.01,
        epochs=20,
        batch_size=24,
        weight_decay=0.001,
        early_stopping_rounds=5
    )
    model.fit(train_data, encoder)

    # Test pipeline using *trained* vectorizers
    print("Loading and preprocessing test data...")
    test_pipeline = NLPDataPipeline("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    test_data_df = test_pipeline.load_data()
    print(f"Test data shape: {test_data_df.shape}")
    
    test_pipeline.preprocess()
    X_test_final, _ = test_pipeline.transform_features(train_pipeline.vectorizers, train_pipeline.encoders)
    X_test_final = X_test_final.toarray()

    print(f"Final test features shape: {X_test_final.shape}")

    # Predict
    preds = []
    for i in range(len(X_test_final)):
        _, probs, _ = model.predict(X_test_final[i])
        # Find the index for violation class (assuming binary classification)
        if 1 in encoder.label_to_index.values():
            violation_idx = list(encoder.label_to_index.values()).index(1)
            preds.append(round(probs[violation_idx], 6))
        else:
            # If no violation class found, use the highest probability
            preds.append(round(np.max(probs), 6))

    # Create submission
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    submission = pd.DataFrame({
        "row_id": test_df["row_id"],
        "rule_violation": preds
    })
    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"Predictions: {preds[:10]}")  # Show first 10 predictions
    print("✅ submission.csv ready for Kaggle upload with probabilities!")

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()


