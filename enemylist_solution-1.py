import pandas as pd

data = pd.read_csv('/kaggle/input/test111/trained_soundscape_bird_clef_2025.csv')

sample_submission = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')


data.head()


from sklearn.preprocessing import LabelEncoder

# Encode the 'prediction' column
label_encoder = LabelEncoder()
data['encoded_prediction'] = label_encoder.fit_transform(data['prediction'])


# Setup X and y

X = data[['confidence']].values  # Features
y = data['encoded_prediction'].values  # Target


from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize y for multi-class ROC AUC
y_bin = label_binarize(y, classes=np.unique(y))


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Define K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize arrays to store all predictions and true labels
all_predictions = []
all_true_labels = []

# Perform K-Fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_bin[train_index], y_bin[test_index]
    
    # Standardize the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the neural network model architecture
    def build_model(input_dim, num_classes):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))  # Output layer for multi-class classification
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    # Build and train the model
    model = build_model(input_dim=X_train_scaled.shape[1], num_classes=y_train.shape[1])
    
    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
    
    # Train the model
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stop], verbose=1)
    
    # Predict probabilities on the test set
    y_pred_proba = model.predict(X_test_scaled)
    
    # Store the predictions and true labels
    all_predictions.append(y_pred_proba)
    all_true_labels.append(y_test)

# Combine all predictions and true labels
y_pred_proba_all = np.vstack(all_predictions)
y_true_all = np.vstack(all_true_labels)

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_true_all, y_pred_proba_all, multi_class='ovr')
roc_auc


y_pred_proba_corrected = y_pred_proba[:len(sample_submission)]


sample_submission.head()


# Save the submission to CSV
sample_submission.to_csv('submission.csv', index=False)

