import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
print(train.shape,test.shape)


train.head()


train.info()


train.isnull().sum()


test.isnull().sum()


# Filling One Null value in test set
test.fillna(test.mean(), inplace=True)


numerical_variables = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = ['winddirection']


num_cols = numerical_variables
n = len(num_cols)
rows = (n + 2) // 3

fig, axes = plt.subplots(rows, 3, figsize=(12, rows * 3))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(train[col], kde=True, ax=axes[i], color='red')
    axes[i].set_title(f'{col} Distribution')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()




plt.figure(figsize=(8, 6))
corr = train[numerical_variables + [target_variable]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
avg_rainfall = train.groupby('winddirection')['rainfall'].mean().sort_values()

sns.barplot(x=avg_rainfall.index, y=avg_rainfall.values, palette='coolwarm')
plt.title('Average Rainfall by Wind Direction')
plt.xlabel('Wind Direction')
plt.ylabel('Average Rainfall')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(numerical_variables):
    sns.boxplot(data=train, y=col, ax=axes[i], color='lightgreen')
    axes[i].set_title(f'{col} Boxplot')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd

def feature_engineering(df):
    df = df.copy()
    
    df['hci'] = df['humidity'] * df['cloud']
    df['hsi'] = df['humidity'] * df['sunshine']
    df['csr'] = df['cloud'] / (df['sunshine'] + 1e-5)
    df['rd'] = 100 - df['humidity']
    df['sp'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['wi'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    df['humidity_cloud_ratio'] = df['humidity'] / (df['cloud'] + 1e-3)
    df['sunshine_cloud_ratio'] = df['sunshine'] / (df['cloud'] + 1e-3)
    df['pressure_wind_interaction'] = df['pressure'] * df['winddirection']
    df['temp_pressure_ratio'] = df['temparature'] / (df['pressure'] + 1e-3)
    df['wind_pressure_ratio'] = df['windspeed'] / (df['pressure'] + 1e-3)
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)


train


xgb_params = {
    'n_estimators': 2407,
    'eta': 0.009462133032592785,
    'gamma': 0.2865859948765318,
    'max_depth': 31,
    'min_child_weight': 47,
    'subsample': 0.6956431754146083,
    'colsample_bytree': 0.3670732604094118,
    'grow_policy': 'lossguide',
    'max_leaves': 73,
    'enable_categorical': True,
    'n_jobs': -1,
    'device': 'cuda',
    'tree_method': 'hist'
}

lgbm_params = {
    'n_estimators': 2500,
    'random_state': 42,
    'max_bin': 1024,
    'colsample_bytree': 0.6,
    'reg_lambda': 80,
    'verbosity': -1
}


X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']
test=test.drop(['id'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Set up models with pre-defined parameters
xgb_model = XGBClassifier(**xgb_params)  # XGBoost model
lgbm_model = LGBMClassifier(**lgbm_params)  # LightGBM model

# Function to train a model using Stratified K-Fold cross-validation
def train_model(model, X, y, n_splits=5, random_state=42):
    # Convert DataFrame/Series to NumPy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Stratified K-Fold ensures each fold has similar class distribution
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Lists to store results from each fold
    all_probs = []       # predicted probabilities
    all_mccs = []        # MCC score per fold
    all_accuracies = []  # Accuracy per fold

    # Print model name
    print("+" * 80)
    print(f"Training model: {model.__class__.__name__}")
    print("+" * 80)

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data for this fold
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Train model
        model.fit(X_train, y_train)

        # Predict class labels and probabilities
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        mcc = matthews_corrcoef(y_val, y_pred)

        # Save results
        all_probs.append(y_prob)
        all_mccs.append(mcc)
        all_accuracies.append(acc)

        # Print metrics for this fold
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}, MCC = {mcc:.4f}")

    # Print overall performance
    print("\nOverall Performance:")
    print(f"Mean Accuracy: {np.mean(all_accuracies):.4f} Â± {np.std(all_accuracies):.4f}")
    print(f"Mean MCC:      {np.mean(all_mccs):.4f} Â± {np.std(all_mccs):.4f}")

    return all_probs, all_mccs, all_accuracies

# ==== Train both models ====

# Train XGBoost model on training data
oof_probs_xgb, oof_mccs_xgb, oof_accuracies_xgb = train_model(
    xgb_model, X_train_scaled, y_train, random_state=42)

# Train LightGBM model on training data
oof_probs_lgbm, oof_mccs_lgbm, oof_accuracies_lgbm = train_model(
    lgbm_model, X_train_scaled, y_train, random_state=42)

# ==== Make predictions on the validation set ====

# Predict class labels (0 or 1)
y_val_pred_xgb = xgb_model.predict(X_val_scaled)
y_val_pred_lgbm = lgbm_model.predict(X_val_scaled)

# Predict probabilities of class 1 (used for ROC, thresholds, etc.)
y_val_prob_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
y_val_prob_lgbm = lgbm_model.predict_proba(X_val_scaled)[:, 1]


# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------
# ROC Curve and AUC for XGBoost
# ---------------------------------------------------------------------

# Generate False Positive Rate, True Positive Rate, and Thresholds
# These are calculated based on the predicted probabilities for the positive class (class 1)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_val, y_val_prob_xgb)

# Compute the Area Under the Curve (AUC) for XGBoost
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# ---------------------------------------------------------------------
# ROC Curve and AUC for LightGBM
# ---------------------------------------------------------------------

# Same as above, but for LightGBM
fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(y_val, y_val_prob_lgbm)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

# ---------------------------------------------------------------------
# Plotting the ROC Curves
# ---------------------------------------------------------------------

# Set the size of the plot
plt.figure(figsize=(10, 6))

# Plot the ROC curve for XGBoost
plt.plot(fpr_xgb, tpr_xgb, color='blue', linewidth=2,
         label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')

# Plot the ROC curve for LightGBM
plt.plot(fpr_lgbm, tpr_lgbm, color='red', linewidth=2,
         label=f'LightGBM (AUC = {roc_auc_lgbm:.2f})')

# Plot a diagonal dashed line to show what random guessing would look like
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)

# Add title and labels to the plot
plt.title('ROC Curve Comparison: XGBoost vs LightGBM', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)

# Show legend in the bottom right
plt.legend(loc='lower right')

# Add grid for better readability
plt.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()



# Import required libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC

# -----------------------------------------------------------------------------
# 1. Standardize the feature data
# -----------------------------------------------------------------------------

# Initialize the standard scaler
scaler = StandardScaler()

# Fit the scaler on the full feature set X and transform it to scaled values
X_scaled = scaler.fit_transform(X)

# Scale the test set in the same way (don't fit again!)
X_test_scaled = scaler.transform(test.drop([], axis=1))  # If no columns to drop, replace [] with actual list

# -----------------------------------------------------------------------------
# 2. Split data into training and validation sets
# -----------------------------------------------------------------------------

# Use train_test_split to split data into 80% training and 20% validation
# stratify=y keeps the class distribution balanced in both sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# 3. Reshape data for Conv1D input
# -----------------------------------------------------------------------------

# Conv1D expects input of shape (samples, time_steps, features)
# So we reshape the data by adding a third dimension

# Reshape training data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Reshape validation data
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

# Reshape test data
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))



X_train.shape


# -----------------------------------------------------------------------------
# 1. Define the CNN Model Architecture
# -----------------------------------------------------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Create a Sequential model (layer-by-layer)
model = Sequential([

    # First convolutional layer: 64 filters of size 3, ReLU activation
    Conv1D(filters=64, kernel_size=3, activation='relu', 
           input_shape=(X_train.shape[1], X_train.shape[2])),

    # Downsample the output using max pooling (pool size = 2)
    MaxPooling1D(pool_size=2),

    # Second convolutional layer: 32 filters
    Conv1D(filters=32, kernel_size=3, activation='relu'),

    # Another max pooling layer
    MaxPooling1D(pool_size=2),

    # Flatten the 3D output to 1D before passing to dense layers
    Flatten(),

    # First fully connected (dense) layer
    Dense(64, activation='relu'),

    # Dropout layer to prevent overfitting (30% neurons randomly turned off)
    Dropout(0.3),

    # Second dense layer
    Dense(32, activation='relu'),

    # Output layer with 1 neuron (for binary classification), sigmoid gives probability
    Dense(1, activation='sigmoid')
])

# -----------------------------------------------------------------------------
# 2. Compile the Model
# -----------------------------------------------------------------------------

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC

# Use SGD optimizer (Stochastic Gradient Descent) with learning rate and momentum
optimizer = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6)

# Compile the model with:
# - Binary crossentropy loss (for binary classification)
# - AUC metric to evaluate classification performance
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[AUC(name='auc')])

# -----------------------------------------------------------------------------
# 3. Set Up Training Callbacks
# -----------------------------------------------------------------------------

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Stop training if validation loss doesn't improve for 20 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate if validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,        # Reduce LR by 10x
    patience=10,       # Wait 10 epochs before reducing
    min_lr=1e-5,       # Minimum LR limit
    verbose=1
)

# -----------------------------------------------------------------------------
# 4. Train the Model
# -----------------------------------------------------------------------------

# Fit the model to the training data
history = model.fit(
    X_train, y_train,                 # Training data
    epochs=200,                       # Maximum number of epochs
    batch_size=32,                    # How many samples per update
    validation_data=(X_val, y_val),  # Validation data for monitoring
    callbacks=[early_stopping, reduce_lr],  # Use callbacks for smart training
    verbose=1                         # Show training output
)



train_auc = history.history['auc']
val_auc = history.history['val_auc']

plt.figure(figsize=(10, 6))
plt.plot(train_auc, label='Training AUC', color='b', lw=2)
plt.plot(val_auc, label='Validation AUC', color='r', lw=2)

plt.title('Training and Validation AUC vs Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()




