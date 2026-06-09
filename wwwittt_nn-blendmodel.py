


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

# Load data
train = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/train.csv')
test = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv')

# EDA Visualizations
# Correlation matrix of the 300 features
features_all = train.drop(['id', 'target'], axis=1)
corr_matrix = features_all.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of 300 Features')
plt.show()

# Correlation between Top 20 Features and Target
corr_with_target = features_all.corrwith(train['target']).abs().sort_values(ascending=False)
top_20_corr = corr_with_target.head(20)
plt.figure(figsize=(10, 6))
top_20_corr.plot(kind='bar')
plt.title('Correlation between Top 20 Features and Target')
plt.ylabel('Absolute Correlation')
plt.xlabel('Features')
plt.show()

# features selected by RFECV with lasso
features = ['16', '33', '43', '45', '52', '63', '65', '73', '90', '91', 
            '117', '133', '134', '149', '189', '199', '217', '237', '258', '295']

X = train[features].values
y = train['target'].values
X_test = test[features].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print(f"Data shape: {X.shape}, Test: {X_test.shape}")

def create_model_v1(input_dim):
    """Simple model"""
    inp = Input(shape=(input_dim,))
    x = Dense(3, activation='sigmoid')(inp)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def create_model_v2(input_dim):
    """Slightly enhanced model"""
    inp = Input(shape=(input_dim,))
    x = Dense(4, activation='sigmoid')(inp)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.005), 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def create_model_v3(input_dim):
    """Model with batch normalization"""
    inp = Input(shape=(input_dim,))
    x = Dense(5, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.003), 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def train_model(create_model_fn, model_name, epochs=200):
    """Train model with 10-fold CV"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print('='*60)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
        
        model = create_model_fn(X.shape[1])
        
        cb_checkpoint = ModelCheckpoint(
            f'weights_{model_name}_fold{fold}.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
        
        cb_early = EarlyStopping(
            monitor='val_auc',
            patience=50,
            restore_best_weights=True,
            mode='max',
            verbose=0
        )
        
        model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=32,
            callbacks=[cb_checkpoint, cb_early],
            verbose=0
        )
        
        model.load_weights(f'weights_{model_name}_fold{fold}.h5')
        
        oof_preds[val_idx] = model.predict(X_val_fold, verbose=0).ravel()
        test_preds += model.predict(X_test, verbose=0).ravel()
        
        fold_score = roc_auc_score(y_val_fold, oof_preds[val_idx])
        fold_scores.append(fold_score)
        print(f"Fold {fold}: AUC = {fold_score:.6f}")
        
        del model
        tf.keras.backend.clear_session()
    
    test_preds /= 10
    oof_score = roc_auc_score(y, oof_preds)
    
    print(f"\n{model_name} - Mean Fold AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
    print(f"{model_name} - OOF AUC: {oof_score:.6f}")
    
    return oof_preds, test_preds, oof_score

# Train multiple model variants
results = {}
results['v1'] = train_model(create_model_v1, 'simple', epochs=200)
results['v2'] = train_model(create_model_v2, 'dropout', epochs=200)
results['v3'] = train_model(create_model_v3, 'batchnorm', epochs=250)

# Find best single model
best_model = max(results.items(), key=lambda x: x[1][2])
print(f"\n{'='*60}")
print(f"Best single model: {best_model[0]} with OOF AUC = {best_model[1][2]:.6f}")
print('='*60)

# Ensemble optimization
print("\n" + "="*60)
print("ENSEMBLE OPTIMIZATION")
print("="*60)

best_score = 0
best_weights = None

for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < -0.001:
            continue
        
        ensemble_oof = (w1 * results['v1'][0] + 
                       w2 * results['v2'][0] + 
                       w3 * results['v3'][0])
        score = roc_auc_score(y, ensemble_oof)
        
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

w1, w2, w3 = best_weights
print(f"Best weights: V1={w1:.3f}, V2={w2:.3f}, V3={w3:.3f}")
print(f"Ensemble OOF AUC: {best_score:.6f}")

# Final predictions
ensemble_test = (w1 * results['v1'][1] + 
                w2 * results['v2'][1] + 
                w3 * results['v3'][1])

# Create submission
submission = pd.DataFrame({
    'id': test['id'].values,
    'target': ensemble_test
})
submission.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"V1 (Simple) OOF:     {results['v1'][2]:.6f}")
print(f"V2 (Dropout) OOF:    {results['v2'][2]:.6f}")
print(f"V3 (BatchNorm) OOF:  {results['v3'][2]:.6f}")
print(f"Ensemble OOF:        {best_score:.6f}")
print(f"\nSubmission saved: submission.csv")


import matplotlib.pyplot as plt

# Score visualization
models = ['V1 (Simple)', 'V2 (Dropout)', 'V3 (BatchNorm)', 'Ensemble']
scores = [results['v1'][2], results['v2'][2], results['v3'][2], best_score]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
plt.figure(figsize=(8, 6))
bars = plt.bar(models, scores, color=colors)
plt.ylabel('OOF AUC')
plt.title('Model Performance Comparison')
plt.ylim(0.7, 1.0)
plt.tight_layout()
plt.show()




