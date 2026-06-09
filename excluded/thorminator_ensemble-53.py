import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import SGD, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from sklearn.utils.class_weight import compute_class_weight



# Data loading and preprocessing
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')





class NNTuner(kt.HyperModel):
    def build(self, hp):
        # PCA hyperparameter
        pca_components = hp.Int('pca_components', 500, 1000, step=100)
        
        # Architecture hyperparameters
        hp_noise = hp.Float('input_noise', 0.01, 0.1)
        hp_units1 = hp.Int('units1', 300, 600, step=50)
        hp_units2 = hp.Int('units2', 160, 224, step=32)
        hp_dropout1 = hp.Float('dropout1', 0.4, 0.7)
        hp_dropout2 = hp.Float('dropout2', 0.3, 0.6)
        
        # Regularization
        l1_reg = hp.Float('l1_reg', 1e-4, 1e-3)
        l2_reg = hp.Float('l2_reg', 1e-3, 1e-2)
        
        # Build model
        inputs = layers.Input(shape=(pca_components,))
        x = layers.GaussianNoise(hp_noise)(inputs)
        x = layers.Dense(hp_units1, activation='gelu', 
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp_dropout1)(x)
        x = layers.Dense(hp_units2, activation='gelu', 
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp_dropout2)(x)
        outputs = layers.Dense(20, activation='softmax')(x)
        
        return Model(inputs, outputs)

    def fit(self, hp, model, x, y, **kwargs):
        # Split and preprocess
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.18, random_state=42)
        
        # Scale data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        
        # Apply PCA
        pca = PCA(n_components=hp.get('pca_components'))
        x_train_pca = pca.fit_transform(x_train_scaled)
        x_val_pca = pca.transform(x_val_scaled)
        
        # Optimizer parameters
        sgd_lr = hp.Float('sgd_lr', 0.01, 0.05)
        adamw_lr = hp.Float('adamw_lr', 0.0001, 0.001)
        batch_size = hp.Int('batch_size', 128, 256, step=64)
        
        # Phase 1: SGD
        model.compile(
            optimizer=SGD(learning_rate=sgd_lr, momentum=0.95, nesterov=True, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history1 = model.fit(
            x_train_pca, y_train,
            validation_data=(x_val_pca, y_val),
            epochs=50,
            batch_size=batch_size,
            verbose=0,
            class_weight=kwargs.get('class_weight')
        )
        
        # Phase 2: AdamW
        model.compile(
            optimizer=AdamW(learning_rate=adamw_lr, weight_decay=0.005, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        history2 = model.fit(
            x_train_pca, y_train,
            validation_data=(x_val_pca, y_val),
            epochs=30,
            batch_size=batch_size,
            verbose=0,
            class_weight=kwargs.get('class_weight')
        )
        
        # Return combined history
        combined_history = {
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy']
        }
        return combined_history

# Configure and run tuner
tuner = kt.BayesianOptimization(
    NNTuner(),
    objective='val_accuracy',
    max_trials=50,  # Increased exploration
    num_initial_points=10,  # Initial random exploration
    alpha=0.0001,
    beta=2.6,
    directory='tuning',
    project_name='pca_nn_bayesian'
)




# Compute class weights
classes, counts = np.unique(y_train, return_counts=True)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train.flatten())
class_weights = dict(enumerate(1.5 * class_weights))

# Run hyperparameter search
tuner.search(x_train, y_train, class_weight=class_weights)

# Build ensemble of top 3 models
best_hps = tuner.get_best_hyperparameters(num_trials=5)
ensemble = []

for hp in best_hps:
    # Full preprocessing with best PCA
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    pca = PCA(n_components=hp.get('pca_components'))
    x_train_pca = pca.fit_transform(x_train_scaled)
    
    # Build and train model
    model = tuner.hypermodel.build(hp)
    
    # Phase 1: SGD
    model.compile(
        optimizer=SGD(
            learning_rate=hp.get('sgd_lr'),
            momentum=0.95,
            nesterov=True,
            clipnorm=1.0
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x_train_pca, y_train,
        epochs=120,
        batch_size=hp.get('batch_size'),
        verbose=0,
        class_weight=class_weights
    )
    
    # Phase 2: AdamW
    model.compile(
        optimizer=AdamW(
            learning_rate=hp.get('adamw_lr'),
            weight_decay=0.005,
            clipnorm=1.0
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        x_train_pca, y_train,
        epochs=60,
        batch_size=hp.get('batch_size'),
        verbose=0,
        class_weight=class_weights
    )
    
    ensemble.append(model)

# Ensemble predictions with TTA
def ensemble_predict(models, x_test, scalers_pcas, tta_steps=5):
    all_preds = []
    for model, (scaler, pca) in zip(models, scalers_pcas):
        x_test_scaled = scaler.transform(x_test)
        x_test_pca = pca.transform(x_test_scaled)
        preds = [model.predict(x_test_pca, verbose=0) for _ in range(tta_steps)]
        all_preds.append(np.mean(preds, axis=0))
    return np.argmax(np.mean(all_preds, axis=0), axis=1)

# Prepare scalers and PCAs for each model
scalers_pcas = []
for hp in best_hps:
    scaler = StandardScaler().fit(x_train)
    pca = PCA(n_components=hp.get('pca_components')).fit(scaler.transform(x_train))
    scalers_pcas.append((scaler, pca))


# Generate final predictions
y_test_hat = ensemble_predict(ensemble, x_test, scalers_pcas)


# Save ensemble predictions
pd.DataFrame({'Id': np.arange(len(y_test_hat)), 'Predicted': y_test_hat}).to_csv('ensemble2_submission.csv', index=False)

