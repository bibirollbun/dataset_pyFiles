import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc, os,random
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.utils import to_categorical
print('TensorFlow version =',tf.__version__)

# USE MULTIPLE GPUS
gpus = tf.config.list_physical_devices('GPU')
if len(gpus)<=1: 
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:# Invalid device or cannot modify virtual devices once initialized.
        pass    
    print(f'Using {len(gpus)} GPU')
    
else: 
    strategy = tf.distribute.MirroredStrategy()
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except:# Invalid device or cannot modify virtual devices once initialized.
        pass    
    print(f'Using {len(gpus)} GPUs')


class CFG:
    SEED    = 42
    FOLDS   = 10
    verbose = 0
    epochs  = 100
    
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(seed=CFG.SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.rename(columns={'Temparature':'Temperature','Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer_Name'}, inplace=True)
test.rename(columns={'Temparature':'Temperature','Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer_Name'}, inplace=True)
original.rename(columns={'Temparature':'Temperature','Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer_Name'}, inplace=True)

test.drop(['id'], axis=1, inplace=True)
train.drop(['id'], axis=1, inplace=True)

train_ori_len=train.shape[0]
print(train.shape)


for k in range(4):
    train = pd.concat([train,original],axis=0)

train=train.reset_index(drop=True)
train.head()


features=['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type','Nitrogen', 'Potassium', 'Phosphorous']
target=['Fertilizer_Name']

Fertilizer_Name_dict_inv=dict(enumerate(train.Fertilizer_Name.unique()))
Fertilizer_Name_dict= {v: k for k, v in Fertilizer_Name_dict_inv.items()}
train['Fertilizer_Name']=train['Fertilizer_Name'].map(Fertilizer_Name_dict)
print(Fertilizer_Name_dict)

Soil_Type_dict=dict(enumerate(train.Soil_Type.unique()))
Soil_Type_dict= {v: k for k, v in Soil_Type_dict.items()}
print('Soil_Type_dict :',Soil_Type_dict)

Crop_Type_dict=dict(enumerate(train.Crop_Type.unique()))
Crop_Type_dict= {v: k for k, v in Crop_Type_dict.items()}
print('Crop_Type_dict :',Crop_Type_dict)


import pandas as pd
import numpy as np

# Combine train and test data
df_combi = pd.concat((train, test)).reset_index(drop=True)

# Check for missing values
print("Missing values before processing:")
print(df_combi.isnull().sum())

# Map categorical variables
df_combi['Soil_Type'] = df_combi['Soil_Type'].map(Soil_Type_dict)
df_combi['Crop_Type'] = df_combi['Crop_Type'].map(Crop_Type_dict)

# Handle missing values in categorical columns after mapping
print("\nMissing values after categorical mapping:")
print(df_combi[['Soil_Type', 'Crop_Type']].isnull().sum())

# Fill missing values in categorical columns if any
if df_combi['Soil_Type'].isnull().any():
    # Fill with mode or a default value
    df_combi['Soil_Type'] = df_combi['Soil_Type'].fillna(df_combi['Soil_Type'].mode()[0])
    
if df_combi['Crop_Type'].isnull().any():
    df_combi['Crop_Type'] = df_combi['Crop_Type'].fillna(df_combi['Crop_Type'].mode()[0])

# Convert categorical to int32
df_combi['Soil_Type'] = df_combi['Soil_Type'].astype('int32')
df_combi['Crop_Type'] = df_combi['Crop_Type'].astype('int32')

# Handle numerical columns
numerical_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

print("\nMissing values in numerical columns:")
for col in numerical_cols:
    print(f"{col}: {df_combi[col].isnull().sum()}")

# Method 1: Fill missing values with median/mean before scaling
for col in numerical_cols:
    if df_combi[col].isnull().any():
        # Fill with median (more robust to outliers)
        median_val = df_combi[col].median()
        df_combi[col] = df_combi[col].fillna(median_val)
        print(f"Filled {col} missing values with median: {median_val}")

# Now perform min-max scaling and convert to int32
for col in numerical_cols:
    # Check for infinite values
    if np.isinf(df_combi[col]).any():
        print(f"Warning: {col} contains infinite values")
        df_combi[col] = df_combi[col].replace([np.inf, -np.inf], np.nan)
        df_combi[col] = df_combi[col].fillna(df_combi[col].median())
    
    # Min-max scaling
    min_val = df_combi[col].min()
    max_val = df_combi[col].max()
    
    if min_val == max_val:
        # Handle case where all values are the same
        df_combi[col] = 0
    else:
        df_combi[col] = df_combi[col] - min_val
    
    # Convert to int32
    df_combi[col] = df_combi[col].astype('int32')

print("\nData types after processing:")
print(df_combi.dtypes)

print("\nFinal check for missing values:")
print(df_combi.isnull().sum())

# Split back to train and test
train = df_combi[:len(train)]
test = df_combi[len(train):]

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Optional: More sophisticated missing value handling
def advanced_missing_value_handling(df_combi, numerical_cols):
    """
    Advanced missing value handling with multiple strategies
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler
    
    # Create a copy for advanced processing
    df_advanced = df_combi.copy()
    
    # Strategy 1: KNN Imputation for numerical columns
    if any(df_advanced[numerical_cols].isnull().sum() > 0):
        print("Using KNN imputation for missing values...")
        
        # Separate numerical and categorical for imputation
        numerical_data = df_advanced[numerical_cols]
        
        # KNN imputation
        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
        numerical_imputed = knn_imputer.fit_transform(numerical_data)
        
        # Replace the columns
        df_advanced[numerical_cols] = numerical_imputed
    
    # Strategy 2: Feature engineering based imputation
    # Create interaction features that might help with imputation
    if 'Temperature' in df_advanced.columns and 'Humidity' in df_advanced.columns:
        df_advanced['Temp_Humidity_Interaction'] = df_advanced['Temperature'] * df_advanced['Humidity']
    
    if 'Nitrogen' in df_advanced.columns and 'Phosphorous' in df_advanced.columns:
        df_advanced['N_P_Ratio'] = df_advanced['Nitrogen'] / (df_advanced['Phosphorous'] + 1e-8)
    
    return df_advanced

# Alternative: Use the advanced handling if needed
# df_combi_advanced = advanced_missing_value_handling(df_combi, numerical_cols)

# Verification function
def verify_data_quality(df, name="Dataset"):
    """
    Verify data quality after preprocessing
    """
    print(f"\n=== {name} Quality Check ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"Data types:\n{df.dtypes}")
    
    # Check for any remaining issues
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            
            if df[col].isnull().any():
                print(f"Warning: {col} still has missing values")
            if np.isinf(df[col]).any():
                print(f"Warning: {col} still has infinite values")



# Run verification
verify_data_quality(train, "Train")
verify_data_quality(test, "Test")


features=['Temperature', 'Moisture', 'Soil_Type', 'Crop_Type','Nitrogen', 'Potassium', 'Phosphorous']


train.head()


test.drop(['Humidity '], axis=1, inplace=True)


test.head()


humidity_cols = [col for col in train.columns if 'humidity' in col.lower()]
if humidity_cols:
    train.drop(humidity_cols, axis=1, inplace=True)
    print(f"Dropped columns: {humidity_cols}")

train.head()



y_ori=train.Fertilizer_Name.values
y = to_categorical(train.Fertilizer_Name)
Fertilizer_Name_classes = [key for key in Fertilizer_Name_dict]
N_CLASSES = len(Fertilizer_Name_classes)
train.drop(['Fertilizer_Name'], axis=1, inplace=True)



def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []    
    
    categoricals=['int32']
    #the cols to be embedded: rescaling to range [0, # values)
    for c in X_train.select_dtypes(include=categoricals):
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i       
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)
     

    return input_list_train, input_list_val, input_list_test 


def top_3_accuracy(y_true, y_pred):
    dd=tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)*.5
    dd+=tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)*.17
    dd+=tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)*.33
    return dd

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K

def top_3_accuracy(y_true, y_pred):
    """Custom metric for top-3 accuracy"""
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

def attention_layer(inputs, attention_dim=128):
    """Self-attention mechanism"""
    attention = layers.Dense(attention_dim, activation='tanh')(inputs)
    attention = layers.Dense(1)(attention)
    attention_weights = layers.Activation('softmax')(attention)
    attended = layers.Multiply()([inputs, attention_weights])
    return attended

def residual_block(x, units, dropout_rate=0.3):
    """Residual block with skip connections"""
    residual = x
    
    # First layer
    x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second layer
    x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection (adjust dimensions if needed)
    if residual.shape[-1] != units:
        residual = layers.Dense(units, activation='linear')(residual)
    
    x = layers.Add()([x, residual])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    return x

def create_embedding_network(features, train, name_suffix=""):
    """Create embedding network for categorical features"""
    input_models = []
    output_embeddings = []
    
    for categorical_var in features:
        # Name of the categorical variable for embedding layer
        cat_emb_name = categorical_var.replace(" ", "") + f'_Embedding_{name_suffix}'
        
        # Calculate embedding size with improved formula
        no_of_unique_cat = train[categorical_var].nunique()
        embedding_size = int(min(np.ceil((no_of_unique_cat**0.25) * 8), 128))
        
        # Input layer
        input_model = layers.Input(shape=(1,), name=f"{categorical_var}_{name_suffix}")
        
        # Embedding with regularization
        output_model = layers.Embedding(
            no_of_unique_cat, 
            embedding_size, 
            name=cat_emb_name,
            embeddings_regularizer=regularizers.l2(1e-6)
        )(input_model)
        
        # Advanced dropout strategy
        if embedding_size > 32:
            output_model = layers.SpatialDropout1D(0.3)(output_model)
        elif embedding_size > 16:
            output_model = layers.SpatialDropout1D(0.2)(output_model)
        
        output_model = layers.Reshape(target_shape=(embedding_size,))(output_model)
        
        input_models.append(input_model)
        output_embeddings.append(output_model)
    
    return input_models, output_embeddings

def network_branch_1(concatenated_embeddings, name="branch1"):
    """First neural network branch - Deep and Wide"""
    # Initial processing
    x = layers.BatchNormalization(name=f'{name}_bn_0')(concatenated_embeddings)
    x = layers.Dropout(0.3, name=f'{name}_dropout_0')(x)
    
    # Deep path with residual connections
    x = layers.Dense(2048, activation='relu', kernel_initializer='he_normal', name=f'{name}_dense_1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn_1')(x)
    x = layers.Dropout(0.4, name=f'{name}_dropout_1')(x)
    
    # Residual blocks
    x = residual_block(x, 1024, dropout_rate=0.3)
    x = residual_block(x, 512, dropout_rate=0.3)
    
    # Attention mechanism
    x = attention_layer(x, attention_dim=256)
    
    # Final layers
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal', name=f'{name}_dense_final')(x)
    x = layers.BatchNormalization(name=f'{name}_bn_final')(x)
    x = layers.Dropout(0.3, name=f'{name}_dropout_final')(x)
    
    return x

def network_branch_2(concatenated_embeddings, name="branch2"):
    """Second neural network branch - Wide and Shallow with different architecture"""
    # Initial processing
    x = layers.BatchNormalization(name=f'{name}_bn_0')(concatenated_embeddings)
    x = layers.Dropout(0.2, name=f'{name}_dropout_0')(x)
    
    # Wide layers with different activation functions
    x1 = layers.Dense(1536, activation='relu', kernel_initializer='he_normal', name=f'{name}_dense_1a')(x)
    x2 = layers.Dense(1536, activation='swish', kernel_initializer='he_normal', name=f'{name}_dense_1b')(x)
    x3 = layers.Dense(1536, activation='gelu', kernel_initializer='he_normal', name=f'{name}_dense_1c')(x)
    
    # Combine different activations
    x = layers.Concatenate(name=f'{name}_concat_1')([x1, x2, x3])
    x = layers.BatchNormalization(name=f'{name}_bn_1')(x)
    x = layers.Dropout(0.4, name=f'{name}_dropout_1')(x)
    
    # Feature interaction layer
    x = layers.Dense(1024, activation='relu', kernel_initializer='he_normal', name=f'{name}_dense_2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn_2')(x)
    x = layers.Dropout(0.3, name=f'{name}_dropout_2')(x)
    
    # Highway connection
    highway_gate = layers.Dense(1024, activation='sigmoid', name=f'{name}_highway_gate')(x)
    highway_transform = layers.Dense(1024, activation='relu', name=f'{name}_highway_transform')(x)
    x = layers.Multiply(name=f'{name}_highway_mul1')([highway_gate, highway_transform])
    carry_gate = layers.Lambda(lambda x: 1.0 - x, name=f'{name}_carry_gate')(highway_gate)
    carry = layers.Multiply(name=f'{name}_highway_mul2')([carry_gate, x])
    x = layers.Add(name=f'{name}_highway_add')([x, carry])
    
    # Final processing
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal', name=f'{name}_dense_final')(x)
    x = layers.BatchNormalization(name=f'{name}_bn_final')(x)
    x = layers.Dropout(0.2, name=f'{name}_dropout_final')(x)
    
    return x

def enhanced_dual_network(features, train, N_CLASSES):
    """
    Enhanced dual neural network with two different architectures
    that are concatenated before final prediction
    """
    
    # Create embeddings (shared between both networks)
    input_models, output_embeddings = create_embedding_network(features, train)
    
    # Concatenate all embeddings
    if len(output_embeddings) > 1:
        concatenated = layers.Concatenate(name='embedding_concat')(output_embeddings)
    else:
        concatenated = output_embeddings[0]
    
    # Branch 1: Deep and Wide with Residual Connections
    branch1_output = network_branch_1(concatenated, name="branch1")
    
    # Branch 2: Wide and Shallow with Highway Networks
    branch2_output = network_branch_2(concatenated, name="branch2")
    
    # Concatenate both branches
    combined = layers.Concatenate(name='branches_concat')([branch1_output, branch2_output])
    
    # Final fusion layers
    fusion = layers.Dense(512, activation='relu', kernel_initializer='he_normal', name='fusion_1')(combined)
    fusion = layers.BatchNormalization(name='fusion_bn_1')(fusion)
    fusion = layers.Dropout(0.4, name='fusion_dropout_1')(fusion)
    
    fusion = layers.Dense(256, activation='relu', kernel_initializer='he_normal', name='fusion_2')(fusion)
    fusion = layers.BatchNormalization(name='fusion_bn_2')(fusion)
    fusion = layers.Dropout(0.3, name='fusion_dropout_2')(fusion)
    
    # Multi-head output for ensemble effect
    head1 = layers.Dense(128, activation='relu', name='head1')(fusion)
    head2 = layers.Dense(128, activation='swish', name='head2')(fusion)
    head3 = layers.Dense(128, activation='gelu', name='head3')(fusion)
    
    multi_head = layers.Concatenate(name='multi_head_concat')([head1, head2, head3])
    multi_head = layers.BatchNormalization(name='multi_head_bn')(multi_head)
    multi_head = layers.Dropout(0.2, name='multi_head_dropout')(multi_head)
    
    # Final output layer with multiple regularization techniques
    output = layers.Dense(
        N_CLASSES, 
        activation='softmax',
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-5),
        name='final_output'
    )(multi_head)
    
    # Create and compile model
    model = Model(inputs=input_models, outputs=output, name='enhanced_dual_fertilizer_net')
    
    # Advanced optimizer with learning rate scheduling
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', top_3_accuracy, 'top_k_categorical_accuracy']
    )
    
    return model

def get_advanced_callbacks():
    """Get advanced callbacks for training"""
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_top_3_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_fertilizer_model.h5',
            monitor='val_top_3_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    return callbacks

# Example usage:
def create_and_train_model(features, train, N_CLASSES, X_train, y_train, X_val, y_val):
    """
    Create and train the enhanced dual network model
    """
    # Create the model
    model = enhanced_dual_network(features, train, N_CLASSES)
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    # Get callbacks
    callbacks = get_advanced_callbacks()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Additional utility function for ensemble predictions
def ensemble_predict(models, X_test):
    """
    Make ensemble predictions from multiple models
    """
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
# model=base_network()
# model.summary()


%%time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

print("="*50)
print("Training Enhanced Dual Neural Network")
print("="*50)

# Split data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    train[features], 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_ori
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(test)}")

# Preprocess data
X_train_list, X_val_list, X_test_list = preproc(X_train, X_val, test[features])

# Clear session
tf.keras.backend.clear_session()

# Create the enhanced dual network model
print("\nCreating Enhanced Dual Neural Network...")
with strategy.scope():
    model = enhanced_dual_network(features, train, N_CLASSES)

print(f"Model created successfully!")
print(f"Total parameters: {model.count_params():,}")

# Display model architecture (optional)
if CFG.verbose > 0:
    model.summary()

# Setup callbacks
print("\nSetting up training callbacks...")
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_top_3_accuracy',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-7,
        mode="max"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_top_3_accuracy',
        patience=10,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_enhanced_fertilizer_model.h5',
        monitor='val_top_3_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Train the model
print("\nStarting training...")
print("-" * 30)

history = model.fit(
    X_train_list, 
    y_train,
    validation_data=(X_val_list, y_val),
    epochs=CFG.epochs,
    batch_size=512,
    callbacks=callbacks,
    verbose=CFG.verbose
)

print("\nTraining completed!")

# Evaluate on validation set
print("\nEvaluating model performance...")
val_predictions = model.predict(X_val_list, verbose=0, batch_size=1024)
val_score = top_3_accuracy(
    tf.convert_to_tensor(y_val, dtype=tf.float32),
    tf.convert_to_tensor(val_predictions, dtype=tf.float32)
).numpy().mean()

print(f"Validation top_3_accuracy: {val_score:.6f}")

# Training history summary
if len(history.history['val_top_3_accuracy']) > 0:
    best_epoch = np.argmax(history.history['val_top_3_accuracy']) + 1
    best_val_score = max(history.history['val_top_3_accuracy'])
    final_train_score = history.history['top_3_accuracy'][-1]
    
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation score: {best_val_score:.6f}")
    print(f"Final training score: {final_train_score:.6f}")

# Make predictions on test set
print("\nMaking predictions on test set...")
test_predictions = model.predict(X_test_list, verbose=0, batch_size=1024)

# Generate submission
print("Generating submission file...")
top_3_preds = pd.DataFrame(
    np.argsort(test_predictions, axis=1)[:, -3:][:, ::-1]
).replace(Fertilizer_Name_dict_inv)

sample_submission['Fertilizer Name'] = top_3_preds.agg(' '.join, axis=1)
sample_submission.to_csv('enhanced_dual_network_submission.csv', index=False)

print("Submission saved as 'enhanced_dual_network_submission.csv'")

# Display results
print("\n" + "="*50)
print("TRAINING RESULTS SUMMARY")
print("="*50)
print(f"Model: Enhanced Dual Neural Network")
print(f"Parameters: {model.count_params():,}")
print(f"Validation Score: {val_score:.6f}")
print(f"Training Epochs: {len(history.history['loss'])}")
print("="*50)

print("\nFirst few predictions:")
print(sample_submission.head(10))

# Optional: Save model and predictions
print("\nSaving additional files...")
model.save('enhanced_dual_fertilizer_model.h5')
np.save('validation_predictions.npy', val_predictions)
np.save('test_predictions.npy', test_predictions)

print("Model and predictions saved successfully!")
print("\nTraining complete! ðŸŽ‰")


print("Submission saved as 'enhanced_dual_network_submission.csv'")
print("\nFirst few predictions:")
print(sample_submission.head())



top_3_preds = pd.DataFrame(np.argsort(test_pred, axis=1)[:, -3:][:, ::-1] ).replace(Fertilizer_Name_dict_inv)
sample_submission['Fertilizer Name']=top_3_preds.agg(' '.join, axis=1)
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()

