import pandas as pd
import numpy as np
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
orignal = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=";")


testf = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


import tensorflow as tf
tf.random.set_seed(2200)
from tensorflow.keras import layers, models, metrics, regularizers
from sklearn.model_selection import train_test_split
from sklearn.compose        import ColumnTransformer
from sklearn.preprocessing  import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline       import Pipeline
from tensorflow.keras.callbacks import EarlyStopping


# train and test data
train = train.drop('id', axis=1)
test  = test.drop('id', axis=1)
# orignal["y"] = train["y"].apply(lambda x: 0 if x == "yes" else 1)

# train= pd.concat([train,orignal], axis=0)

def prepare_data(df, target_col, test_size=0.2, random_state=42):
    # 1) Split off X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2) Auto-detect numeric vs. categorical
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols     = X.select_dtypes(include=['object','category']).columns.tolist()
    
    # 3) Build sub-pipelines
    numeric_pipe = Pipeline([('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
    
    # 4) Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numeric_pipe,     numeric_cols),
        ('cat', categorical_pipe, cat_cols),
    ], remainder='drop')
    
    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 6) Fit on train, transform both
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)
    
    return X_train_p, X_test_p, y_train, y_test, preprocessor

X_train, X_test, y_train, y_test, preproc = prepare_data(train, target_col='y')



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def preprocess_bank_data(train_df, test_df, target_col='y', test_size=0.2, random_state=42):
    def feature_engineering(df, job_categories):
        df = df.copy()

        # Month features
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        
        df['month'] = df['month'].str.lower()  # Ensure lowercase for mapping
        df['month_num'] = df['month'].map(month_map)
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

        # Encode categoricals
        
        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes

        # Cross features
        df['jobxloanxpoutcome'] = (df['job'].astype(str) + '_' +
                                   df['loan'].astype(str) + '_' +
                                   df['poutcome'].astype(str)).astype('category').cat.codes

        df['jobxmonthxbalancexduration'] = (df['job'].astype(str) + '_' +
                                            df['month'].astype(str) + '_' +
                                            df['balance'].astype(str) + '_' +
                                            df['duration'].astype(str)).astype('category').cat.codes

        # Derived features
        df['is_contacted'] = (df['contact'] != -1).astype(int)
        df['has_previous_contact'] = ((df['previous'] > 0) & (df['pdays'] != -1)).astype(int)
        df['balance_to_duration_ratio'] = df['balance'] / (df['duration'] + 1)
        df['is_working_age'] = df['age'].between(25, 60).astype(int)
        df['is_married'] = (df['marital'] == 1).astype(int)
        df['duration_bin'] = pd.cut(df['duration'], bins=5, labels=False)
        df['loan_and_default_risk'] = ((df['loan'] == 1) & (df['default'] == 1)).astype(int)

        # Employment risk using original job names
        job_map = dict(enumerate(job_categories))
        df['employment_risk'] = df['job'].map(job_map).isin(['unemployed', 'unknown', 'housemaid']).astype(int)

        # Drop raw cols
        df.drop(columns=['month', 'month_num'], inplace=True, errors='ignore')
        
        return df

    # Separate X and y
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Store job category map for consistent employment_risk feature
    job_categories = X['job'].astype('category').cat.categories

    # Split train into train/val
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Apply feature engineering
    X_train_fe = feature_engineering(X_train_raw, job_categories)
    X_val_fe = feature_engineering(X_val_raw, job_categories)
    X_test_fe = feature_engineering(test_df, job_categories)

    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_fe), columns=X_train_fe.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_fe), columns=X_val_fe.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_fe), columns=X_test_fe.columns)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train.reset_index(drop=True), y_val.reset_index(drop=True)






import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Data preprocessing check
# -------------------------
# Example: Replace these with your real preprocessed data
# X_train_final, X_val_final, y_train, y_val = preprocess_bank_data(train, test)

# Convert to numpy (if pandas DataFrame)
X_train_final = np.array(X_train_final)
X_val_final   = np.array(X_val_final)
y_train       = np.array(y_train)
y_val         = np.array(y_val)

# Check for NaN / Inf
for name, arr in [
    ("X_train", X_train_final),
    ("X_val", X_val_final),
    ("y_train", y_train),
    ("y_val", y_val)
]:
    print(f"{name}: NaNs={np.isnan(arr).sum()} Infs={np.isinf(arr).sum()} MaxAbs={np.max(np.abs(arr))}")

# Ensure no NaNs/Infs remain
if np.isnan(X_train_final).any() or np.isinf(X_train_final).any():
    raise ValueError("NaN or Inf found in X_train_final")
if np.isnan(X_val_final).any() or np.isinf(X_val_final).any():
    raise ValueError("NaN or Inf found in X_val_final")
if np.isnan(y_train).any() or np.isinf(y_train).any():
    raise ValueError("NaN or Inf found in y_train")
if np.isnan(y_val).any() or np.isinf(y_val).any():
    raise ValueError("NaN or Inf found in y_val")

# Convert to float32
X_train_final = X_train_final.astype(np.float32)
X_val_final   = X_val_final.astype(np.float32)
y_train       = y_train.astype(np.float32)
y_val         = y_val.astype(np.float32)

# Convert to tensors
tensorX  = tf.convert_to_tensor(X_train_final)
tensorXt = tf.convert_to_tensor(X_val_final)
tensorY  = tf.convert_to_tensor(y_train)
tensorYt = tf.convert_to_tensor(y_val)

# -------------------------
# Multi-GPU setup
# -------------------------
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus)

strategy = tf.distribute.MirroredStrategy()
print("Number of devices:", strategy.num_replicas_in_sync)

# -------------------------
# Model build and train
# -------------------------
with strategy.scope():
    # Learning rate schedule (safer starting LR)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )

    # Optimizer with gradient clipping
    opt = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9, beta_2=0.999, epsilon=1e-6,
        clipnorm=1.0
    )

    # Model architecture
    model = models.Sequential([
        tf.keras.Input(shape=(tensorX.shape[1],)),

        layers.Dense(512)

        layers.Dense(512, use_bias=False,
                     kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
        layers.BatchNormalization(epsilon=1e-5),
        layers.Activation('swish'),
        layers.Dropout(0.3),

        layers.Dense(512, use_bias=False,
                     kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
        layers.BatchNormalization(epsilon=1e-5),
        layers.Activation('swish'),
        layers.Dropout(0.4),

        layers.Dense(1024, use_bias=False,
                     kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
        layers.BatchNormalization(epsilon=1e-5),
        layers.Activation('swish'),

        layers.Dense(512, use_bias=False,
                     kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
        layers.BatchNormalization(epsilon=1e-5),
        layers.Activation('swish'),

        layers.Dense(1, activation='sigmoid')
    ])

    # Loss and metrics (correct metric objects)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [
        tf.keras.metrics.AUC(name='AUC'),
        tf.keras.metrics.Recall(name='Recall'),
        tf.keras.metrics.Accuracy(name='Accuracy'),
        tf.keras.metrics.Precision(name='Precision')
    ]

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_AUC',
        patience=10,
        restore_best_weights=True
    )

    # Train
    model.fit(
        tensorX, tensorY,
        epochs=100,
        batch_size=1024,
        class_weight={0: 0.45, 1: 4.54},
        validation_data=(tensorXt, tensorYt),
        callbacks=[early_stop]
    )



# binary_preds = np.round(predicted_test).astype(int)
predicted_test = model.predict(X_test_final, batch_size=2048).flatten()
subvc = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

submission = pd.DataFrame({
    'id': subvc.id, 
    'y': predicted_test})


submission.to_csv("submission.csv", index=False)







