import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping


# Load data
x_train = np.load('X_train.npy')
x_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')

# Normalize data 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# I'll cheat and load answers (so I can show performance in notebook)
y_test = np.load('y_test.npy')


# Alternative loss function that allows for label smoothing.
# Modern TensorFlow version support this directby through 
# loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)

def sparse_categorical_crossentropy_with_label_smoothing(label_smoothing=0.0):
    def loss(y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]

        y_true = tf.reshape(y_true, [-1])
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)

        y_true_smoothed = y_true_one_hot * (1 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)

        return tf.keras.losses.categorical_crossentropy(y_true_smoothed, y_pred)

    return loss



def build_and_train_model(
        # Model
        base_units: int,
        contraction: float,
        num_blocks: int,
        activation: str,
        l2_reg: float,
        dropout_rate: float,
        # Train settings
        epochs: int,
        batch_size: int,
        initial_lr: float,
        cosine_alpha: float,
        # Data
        x_train,
        y_train,
        # Train, logs
        train: bool,
        verbose: int,
        #
        loss = 'sparse_categorical_crossentropy',
        validation_data = None,
        ):
    inputs = Input(shape=(x_train.shape[1],))
    x = inputs

    for block in range(num_blocks):
        units = int(base_units * (contraction ** block))

        x = Dense(units, activation=activation, kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(20, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compute decay_steps from the number of training samples (using 90% split) and batch size
    train_samples = int(len(x_train) * 0.9)
    decay_steps = epochs * (train_samples // batch_size)

    lr_schedule = CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=cosine_alpha,
    )

    # Use a single optimizer (Adam) with our cosine decay
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    # Use early stopping to prevent overtraining & to speed up optimization
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    if not train:
        return model

    if verbose > 0:
        model.summary()

    train_args = {
        'x': x_train,
        'y': y_train,
        'epochs': epochs,
        'batch_size': batch_size,
        'callbacks': [early_stop],
        'verbose': verbose,
    }
    
    if validation_data is not None:
        train_args['validation_data'] = validation_data
    else:
        train_args['validation_split'] = 0.1

    history = model.fit(**train_args)

    return model, history


model, history = build_and_train_model(
    base_units=1024,
    contraction=0.9, # We contract 10% at each layer
    num_blocks=4, # Four hidden layers
    activation='swish',
    l2_reg=0.0015,
    dropout_rate=0.4,
    epochs=390,
    batch_size=128,
    initial_lr=0.0001,
    cosine_alpha=0.001,
    x_train=x_train,
    y_train=y_train,
    train=True,
    verbose=1,
)


plt.plot(history.history['val_accuracy'])


# You cannot run this now, but I'll show how I get this below
hyperparam_study = pd.read_csv('hyperparam_study.csv')
hyperparam_study.sort_values('value', ascending=False).head(3)


model_opt, history_opt = build_and_train_model(
    base_units=2048,
    contraction=0.875840	,
    num_blocks=4,
    activation='relu',
    l2_reg=0.000579,
    dropout_rate=0.413789,
    epochs=390,
    batch_size=256,
    initial_lr=0.000100,
    cosine_alpha=0.000439,
    x_train=x_train,
    y_train=y_train,
    train=True,
    verbose=1,
    loss=sparse_categorical_crossentropy_with_label_smoothing(0.000447),
)


plt.plot(history_opt.history['val_accuracy'])


model_opt.evaluate(x_test, y_test)


import optuna # You may have to install this


def objective(trial):
    # Hyperparameters for the model architecture
    base_units = trial.suggest_categorical('base_units', [1024, 2048]) # Suggest either 1024 or 2048
    contraction = trial.suggest_float('contraction', 0.8, 1.0) # Suggest a value between 0.8 and 1
    num_blocks = trial.suggest_int('num_blocks', 4, 8) # Suggest integer between 4 and 8
    activation = trial.suggest_categorical('activation', ['swish', 'relu'])
    l2_reg = trial.suggest_float('l2_reg', 0.0005, 0.005, log=True) # Suggest value between 0.0005 and 0.005, log spaced
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    # Training hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    initial_lr = trial.suggest_float('initial_lr', 1e-4, 1e-2, log=True)
    cosine_alpha = trial.suggest_float('cosine_alpha', 1e-5, 0.5, log=True)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.3)

    # This is very common, but will turn out not to be hugely impactful
    # in this case
    loss_fn = sparse_categorical_crossentropy_with_label_smoothing(
        label_smoothing=label_smoothing,
    )

    # Build and train the model using our function with the Optuna suggested
    # values
    model, history = build_and_train_model(
        base_units=base_units,
        contraction=contraction,
        num_blocks=num_blocks,
        activation=activation,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        epochs=390,
        batch_size=batch_size,
        initial_lr=initial_lr,
        cosine_alpha=cosine_alpha,
        x_train=x_train,
        y_train=y_train,
        train=True,
        verbose=0,
        loss=loss_fn,
    )

    best_val_acc = max(history.history['val_accuracy'])
    tf.keras.backend.clear_session()

    return best_val_acc


# Create an Optuna study
study = optuna.create_study(
    direction='maximize',
    storage='sqlite:///optuna_study.db', # We'll save our results to a db file...
    study_name='ml-ef-assignment-02', # ... using this name as ID
    load_if_exists=True, # If I stop after 10 tries and then rerun, it will pick up from 11
)

# Run optimization - specify how many models to train in total
study.optimize(objective, n_trials=XXX)


# Save as .csv (the one I loaded earlier)
study.trials_dataframe().to_csv('hyperparam_study.csv')


def bootstrap_sample(X, y):
    n = X.shape[0]
    indices = np.random.choice(n, size=n, replace=True)
    return X[indices], y[indices]


from sklearn.model_selection import train_test_split

x_train_for_ensemble, x_val, y_train_for_ensemble, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42,
)


n_ensemble = 25

ensemble_models = []
ensemble_histories = []


hyperparam_study = hyperparam_study.sort_values('value', ascending=False).reset_index(drop=True)


for i in range(n_ensemble):
    print(f'\nTraining ensemble model {i + 1}/{n_ensemble} ...')

    tf.random.set_seed(i)
    
    # Create a bootstrap sample from the training data
    # X_boot, y_boot = bootstrap_sample(x_train_for_ensemble, y_train_for_ensemble)
    X_boot, y_boot = bootstrap_sample(x_train, y_train)
    
    hyperparams = hyperparam_study.iloc[i]

    base_units = hyperparams.params_base_units
    contraction = hyperparams.params_contraction
    num_blocks = hyperparams.params_num_blocks
    activation = hyperparams.params_activation
    l2_reg = hyperparams.params_l2_reg
    dropout_rate = hyperparams.params_dropout_rate
    epochs = 390
    batch_size = hyperparams.params_batch_size
    initial_lr = hyperparams.params_initial_lr
    cosine_alpha = hyperparams.params_cosine_alpha
    loss_fn = sparse_categorical_crossentropy_with_label_smoothing(hyperparams.params_label_smoothing)
    
    model, history = build_and_train_model(
        base_units=base_units,
        contraction=contraction,
        num_blocks=num_blocks,
        activation=activation,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        epochs=epochs,
        batch_size=batch_size,
        initial_lr=initial_lr,
        cosine_alpha=cosine_alpha,
        x_train=X_boot,
        y_train=y_boot,
        train=True,
        verbose=0,
        loss=loss_fn,
        # validation_data=(x_val, y_val),
    )
    
    ensemble_models.append(model)
    ensemble_histories.append(history)


def ensemble_predict(models, x):
    preds = [model.predict(x) for model in models]
    avg_preds = np.mean(preds, axis=0)

    return np.argmax(avg_preds, axis=1)


from sklearn.metrics import accuracy_score

ensemble_preds = ensemble_predict(ensemble_models, x_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)

print(f'\nEnsemble Test Accuracy: {ensemble_accuracy:.4f}')

