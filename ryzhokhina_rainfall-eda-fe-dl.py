import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col = 0)
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv", index_col = 0)


print(train.shape)
train.head()


train.describe()


train.isna().sum()


def plot_avg(dataset, column):
    # Aggregate data by day (average temperature per day across years)
    if 'day' not in dataset.columns or column not in dataset.columns:
        raise ValueError(f"Dataset must contain 'day' and {column} columns")
    
        # Aggregate data by day (mean and standard deviation)
    daily_avg = dataset.groupby('day')[column].mean()
    daily_std = dataset.groupby('day')[column].std()

    # Set fancy style
    plt.figure(figsize=(12, 6))
    #plt.style.use("seaborn-darkgrid")

    # Plot smoothed temperature curve
    plt.plot(daily_avg.index, daily_avg.values, color='darkblue', linewidth=2, label=f"Smoothed Avg {column}")

    # Fill between ±1 std deviation
    plt.fill_between(daily_avg.index, daily_avg - daily_std, daily_avg + daily_std,
                     color='lightblue', alpha=0.3, label="±1 Std Dev")

    # Highlight seasons
    plt.axvspan(1, 80, color='cyan', alpha=0.1, label="Winter")  # Winter
    plt.axvspan(81, 172, color='lightgreen', alpha=0.1, label="Spring")  # Spring
    plt.axvspan(173, 265, color='orange', alpha=0.1, label="Summer")  # Summer
    plt.axvspan(266, 365, color='brown', alpha=0.1, label="Fall")  # Fall

    # Titles and labels
    plt.xlabel("Day of the Year (1-365)", fontsize=12, fontweight='bold')
    plt.ylabel(f"{column}", fontsize=12, fontweight='bold')
    plt.title(f"Trend of {column} Over a Year", fontsize=14, fontweight='bold', color="darkblue")

    # Grid and legend
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)

    # Show plot
    plt.show()


# for col in train.columns:
#     if col in ['day', 'rainfall']:
#         continue
#     plot_avg(train, col)


import seaborn as sns


def plot_correlation_matrix(dataset):
    """
    Plots a correlation heatmap to see relationships between numerical features and rainfall.

    Parameters:
    dataset (pd.DataFrame): DataFrame with numerical features including 'rainfall'.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    corr = dataset.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

    plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold', color="darkblue")
    plt.show()


#plot_correlation_matrix(train)


def plot_rainfall_vs_feature(dataset, feature):
    """
    Creates a boxplot of a numerical feature split by rainfall (0 or 1).

    Parameters:
    dataset (pd.DataFrame): DataFrame with a numerical feature and 'rainfall' column.
    feature (str): Name of the numerical feature.

    Returns:
    None
    """
    if feature not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError(f"Dataset must contain '{feature}' and 'rainfall' columns")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=dataset['rainfall'], y=dataset[feature], hue=dataset['rainfall'], 
                palette=['orange', 'skyblue'])

    plt.xlabel("Rainfall (0 = No Rain, 1 = Rain)", fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.title(f"{feature} Distribution When It Rains vs. No Rain", fontsize=14, fontweight='bold', color="darkblue")

    plt.show()


# for col in train.columns:
#     if col in ['day', 'rainfall']:
#         continue
#     plot_rainfall_vs_feature(train, col)


# Create the bar plot for target variable (rainfall)
plt.figure(figsize=(7, 4))
ax = sns.countplot(x=train['rainfall'])

# Add labels on top of the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=12, fontweight='bold', padding=3)

# Labels and title
plt.xlabel("Rainfall (0 = No Rain, 1 = Rain)", fontsize=12, fontweight='bold')
plt.ylabel("Count", fontsize=12, fontweight='bold')
plt.title("Rainfall Distribution", fontsize=14, fontweight='bold')

# Show the plot
plt.show()


train.day.plot()


train.day.value_counts().value_counts()


len(train)/365


full_days = pd.DataFrame({'day': np.arange(1, 366)})
all_missing_days = []
for i in range(6):
    data = train[i * 365:365 * (i + 1)].day.values
    missing_days = full_days[~full_days['day'].isin(data)]
    all_missing_days.extend(list(missing_days.day.values))
    print(f"Missing Days Count in {i + 1} year: {len(missing_days)}")
    print(list(missing_days.day.values))
print(f"All missing Days Count :  {len(all_missing_days)}")


import matplotlib.pyplot as plt
import pandas as pd

def plot_rainfall_seasonality(dataset):
    """
    Plots the probability of rainfall (Rainfall = 1) over the days of the year to check for seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).

    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError("Dataset must contain 'day' and 'rainfall' columns")

    # Compute probability of rainfall per day
    rainfall_prob = dataset.groupby('day')['rainfall'].mean()

    # Plot probability of rainfall over days
    plt.figure(figsize=(12, 6))
    plt.plot(rainfall_prob.index, rainfall_prob.values, marker='', linestyle='-', color='blue', linewidth=2)

    plt.xlabel("Day of the Year (1-365)", fontsize=12, fontweight='bold')
    plt.ylabel("Probability of Rain", fontsize=12, fontweight='bold')
    plt.title("Rainfall Seasonality Check", fontsize=14, fontweight='bold', color="blue")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()


def plot_rolling_rainfall(dataset, window=30):
    """
    Plots a rolling mean of rainfall probability over days to visualize seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).
    window (int): Rolling window size (default = 30 days).

    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError("Dataset must contain 'day' and 'rainfall' columns")

    # Compute rolling mean of rainfall probability
    daily_rainfall = dataset.groupby('day')['rainfall'].mean()
    rolling_mean = daily_rainfall.rolling(window=window, center=False).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_rainfall.index, daily_rainfall.values, linestyle='--', alpha=0.5, label="Daily Probability", color='gray')
    plt.plot(rolling_mean.index, rolling_mean.values, color='blue', linewidth=2, label=f"{window}-Day Rolling Mean")

    # Highlight seasons
    plt.axvspan(1, 80, color='cyan', alpha=0.1, label="Winter")  # Winter
    plt.axvspan(81, 172, color='lightgreen', alpha=0.1, label="Spring")  # Spring
    plt.axvspan(173, 265, color='orange', alpha=0.1, label="Summer")  # Summer
    plt.axvspan(266, 365, color='brown', alpha=0.1, label="Fall")  # Fall


    plt.xlabel("Day of the Year (1-365)", fontsize=12, fontweight='bold')
    plt.ylabel("Probability of Rain", fontsize=12, fontweight='bold')
    plt.title("Rolling Mean of Rainfall Probability", fontsize=14, fontweight='bold', color="darkblue")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()


import numpy as np

def plot_fft_rainfall(dataset):
    """
    Performs a Fast Fourier Transform (FFT) on rainfall probability data to check for seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).

    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError("Dataset must contain 'day' and 'rainfall' columns")

    # Compute rainfall probability per day
    daily_rainfall = dataset.groupby('day')['rainfall'].mean()

    # Apply FFT
    fft_vals = np.fft.fft(daily_rainfall.values)
    fft_freqs = np.fft.fftfreq(len(daily_rainfall))

    # Plot FFT spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(fft_freqs[1:len(fft_freqs)//2], np.abs(fft_vals[1:len(fft_vals)//2]), color='lightblue')

    plt.xlabel("Frequency", fontsize=12, fontweight='bold')
    plt.ylabel("Magnitude", fontsize=12, fontweight='bold')
    plt.title("FFT Spectrum of Rainfall Data", fontsize=14, fontweight='bold', color="darkblue")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


import statsmodels.api as sm

def plot_acf_rainfall(dataset, column, lags = 180):
    """
    Plots the Autocorrelation Function (ACF) for column probability to check for seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) 
    Returns:
    None
    """
    if 'day' not in dataset.columns or 'rainfall' not in dataset.columns:
        raise ValueError(f"Dataset must contain 'day' and {column} columns")

    # Compute daily rainfall probability
    daily_rainfall = dataset.groupby('day')[column].mean()

    # Plot ACF
    sm.graphics.tsa.plot_acf(daily_rainfall, lags=lags, alpha=0.05)
    plt.title(f"Autocorrelation of {column} Probability")
    plt.xlabel("Lag (Days)")
    plt.ylabel("ACF")
    plt.show()


#Mean Analysis
plot_avg(train, 'rainfall')

#Rolling Mean Analysis
plot_rolling_rainfall(train, window = 30)

#Fourier Transform Analysis
plot_fft_rainfall(train)

#Autocorrelation Check
plot_acf_rainfall(train, 'rainfall')


def plot_rolling_mean(dataset, column, window=30):
    """
    Plots a rolling mean of rainfall probability over days to visualize seasonality.

    Parameters:
    dataset (pd.DataFrame): DataFrame with 'day' (1-365) and 'rainfall' (0 or 1).
    window (int): Rolling window size (default = 30 days).

    Returns:
    None
    """
    if column not in dataset.columns or 'day' not in dataset.columns:
        raise ValueError(f"Dataset must contain 'day' and {column} columns")

    # Compute rolling mean of rainfall probability
    daily_rainfall = dataset.groupby('day')[column].mean()
    rolling_mean = daily_rainfall.rolling(window=window, center=False).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(daily_rainfall.index, daily_rainfall.values, linestyle='--', alpha=0.5, label="Daily Probability", color='gray')
    plt.plot(rolling_mean.index, rolling_mean.values, color='blue', linewidth=2, label=f"Rolling Mean")

    # Highlight seasons
    plt.axvspan(1, 80, color='cyan', alpha=0.1, label="Winter")  # Winter
    plt.axvspan(81, 172, color='lightgreen', alpha=0.1, label="Spring")  # Spring
    plt.axvspan(173, 265, color='orange', alpha=0.1, label="Summer")  # Summer
    plt.axvspan(266, 365, color='brown', alpha=0.1, label="Fall")  # Fall


    plt.xlabel("Day of the Year (1-365)", fontsize=12, fontweight='bold')
    plt.ylabel(column, fontsize=12, fontweight='bold')
    plt.title(f"Rolling Mean of {column}", fontsize=14, fontweight='bold', color="darkblue")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()


# for col in train.columns:
#     if col in ['day', 'rainfall']:
#         continue
#     plot_rolling_mean(train, col)


# plt.figure(figsize=(12, 6))
# col = 'dewpoint'
# train[col].plot()
# temp_rolling = train[col].rolling(window=21, center=False).mean()
# temp_rolling = temp_rolling.bfill()
# temp_rolling.plot()


# plt.figure(figsize=(12, 6))
# col = 'dewpoint'
# temp_sin = np.sin(2 * np.pi * train[col] / 365)
# temp_cos = np.cos(2 * np.pi * train[col] / 365)
# temp_sin.plot()
# temp_cos.plot()


for col in train.columns:
    if col in ['day', 'rainfall']:
        continue
    plot_acf_rainfall(train, col, lags = 180)


train.day.plot()


def fix_days(data, target_years):
    df = data.copy()
    # remove extra days
    df = df.groupby('day').apply(lambda x: x.sample(n=target_years, random_state=42) if len(x) >= target_years else x)
    df.drop('day', axis= 1, inplace = True)
    df.reset_index(drop=False, inplace = True)
    # remove misplaced days
    df = df.sort_values(by=['id', 'day'])
    df = df[(df['day'].diff().fillna(0) >= 0) | (df['day'] == 1)]  # Ensure day is non-decreasing
    df.set_index('id', inplace=True)
    df = df.sort_index()
    return df    


train.day.value_counts().value_counts()


updated_train = fix_days(train, 6)
updated_train.day.plot()


updated_train.day.value_counts().value_counts()


def cyclic_transformation(data, columns):
    data_copy = data.copy()
    for column in columns:
        sin_column = f'{column}_sin'
        cos_column = f'{column}_cos'
        data_copy[sin_column] = np.sin(2 * np.pi * data[column] / 365)
        data_copy[cos_column] = np.cos(2 * np.pi * data[column] / 365)
    return data_copy


def rolling_mean(data, columns, window = 14):
    data_copy = data.copy()
    for column in columns:
        rolling_feature = f'{column}_rolling'
        data_copy[rolling_feature] = data_copy[column].rolling(window=window).mean()
        # handle nan in rolling features
        data_copy[rolling_feature]  = data_copy[rolling_feature].bfill()
    return data_copy


from sklearn.preprocessing import StandardScaler

def standart_scale(data, features_to_scale, scaler = None):
    data_copy = data.copy()
    if scaler is None:
        scaler = StandardScaler()
        data_copy[features_to_scale] = scaler.fit_transform(data_copy[features_to_scale])
        return scaler, data_copy
    data_copy[features_to_scale] = scaler.transform(data_copy[features_to_scale])
    return None, data_copy


train.columns


plt.scatter(y=updated_train.day, x= range(len(updated_train)))


cyclic_features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'winddirection']
rolling_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
       'winddirection']
rolled_features = [ft+'_rolling' for ft in rolling_features]

features_for_scaling =['humidity', 'cloud', 'sunshine', 'windspeed'] + rolled_features
target_feature = ['rainfall']


train_transform = cyclic_transformation(updated_train, cyclic_features)
train_transform = rolling_mean(train_transform, rolling_features, window = 14)
scaler,train_transform = standart_scale(train_transform, features_for_scaling, scaler = None)


train_transform.head()


train_transform.describe()


# for col in train_transform.columns:
#     plt.subplot()
#     train_transform[col].plot()
#     plt.title(col)
#     plt.show()


train_transform.columns


train_columns = rolled_features + \
                 [ft+'_cos' for ft in cyclic_features] + \
                 [ft+'_sin' for ft in cyclic_features]+ \
                 ['humidity', 'cloud', 'sunshine', 'windspeed']
train_columns               


X = train_transform[train_columns]
y = train_transform[target_feature]
X.shape, y.shape


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col = 0)
print(test.shape)
test.head()


test.isna().sum()


mode_per_day = train.groupby('day')['winddirection'].agg(lambda x: x.mode()[0]).reset_index()
mode_per_day.columns = ['day', 'winddirection_mode']
mode_per_day.head()


# Merge mode information into test dataset
test_data = test.merge(mode_per_day, on='day', how='left')

# Fill missing values in test_data using the mode from train_data
test_data['winddirection'] = test_data['winddirection'].fillna(test_data['winddirection_mode'])

# Drop the extra mode column after filling
test_data.drop(columns=['winddirection_mode'], inplace=True)
test_data.head()


test_transform = cyclic_transformation(test_data, cyclic_features)
test_transform = rolling_mean(test_transform, rolling_features, window = 14)
_,test_transform = standart_scale(test_transform, features_for_scaling, scaler = scaler)



test_transform


X_test = test_transform[train_columns]
print(X_test.shape)
X_test.head(20)


# for col in X_test.columns:
#     plt.subplot()
#     X_test[col].plot()
#     plt.title(col)
#     plt.show()


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt


from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve



def plot_roc_curve(y_true, y_pred):
    # Evaluate Model
    print("Model Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_pred):.4f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Rainfall Prediction")
    plt.legend()
    plt.grid()
    plt.show()



from sklearn.model_selection import train_test_split


x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123, shuffle = True)


def plot_training_history(history):
    # Extract values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()


def create_nn_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),  # Input Layer
        Dropout(0.5),  # Dropout for regularization
        Dense(128, activation='relu'),  # Input Layer
        Dropout(0.5),  # Dropout for regularization
        Dense(64, activation='relu'),  # Hidden Layer
        Dropout(0.5),
        Dense(32, activation='relu'),  # Hidden Layer
        Dropout(0.2),
        Dense(16, activation='relu'),  # Another Hidden Layer
        Dense(1, activation='sigmoid')  # Output Layer for Binary Classification
    ])
    return model


INPUT_SHARE = (x_train.shape[1],)
INPUT_SHARE


y.value_counts()


# Define the Neural Network Model
model = create_nn_model(INPUT_SHARE)


# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=35,  # Wait for 5 epochs after the last improvement
    restore_best_weights=True,  # Restore the best weights from the epoch with the lowest validation loss
    verbose=1  # Print information about early stopping
)

# Compile the Model
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(
    x_train,
    y_train,
    epochs=250, batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose = 0
)


plot_training_history(history)


# Predict Probabilities & Convert to Binary Labels
y_prob = model.predict(x_val)
# Evaluate Model
y_pred = (y_prob > 0.5).astype(int)
plot_roc_curve(y_val, y_pred)


x_train = x_train.values
y_train = y_train.values
x_val = x_val.values
y_val = y_val.values


np.unique(y_train)


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y.rainfall), y=y.rainfall)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
class_weights_dict


model2 = create_nn_model(INPUT_SHARE)

# Compile the Model
#sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model2.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=35,  # Wait for 5 epochs after the last improvement
    restore_best_weights=True,  # Restore the best weights from the epoch with the lowest validation loss
    verbose=1  # Print information about early stopping
)

# Train the Model
history2 = model2.fit(
    x_train,
    y_train,
    epochs=250, batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weights_dict,
    verbose=0)



plot_training_history(history2)


# Predict Probabilities & Convert to Binary Labels
y_prob2 = model2.predict(x_val)
# Evaluate Model
y_pred2 = (y_prob2 > 0.5).astype(int)
plot_roc_curve(y_val, y_pred2)


model3 = create_nn_model(INPUT_SHARE)

# Compile the Model
#sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

focal_loss = tf.keras.losses.BinaryFocalCrossentropy()
model3.compile(optimizer=adam_optimizer, loss=focal_loss, metrics=['accuracy'])


# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=35,  # Wait for 5 epochs after the last improvement
    restore_best_weights=True,  # Restore the best weights from the epoch with the lowest validation loss
    verbose=1  # Print information about early stopping
)


# Train the Model
history3 = model3.fit(
    x_train,
    y_train,
    epochs=250, batch_size=128,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weights_dict,
    verbose=0)


plot_training_history(history3)


# Predict Probabilities & Convert to Binary Labels
y_prob3 = model3.predict(x_val)
# Evaluate Model
y_pred3 = (y_prob3 > 0.5).astype(int)
plot_roc_curve(y_val, y_pred3)


from sklearn.model_selection import KFold
import numpy as np

def train_and_evaluate_with_cv(x_data, y_data, gamma_values, n_splits=5, epochs=250, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for gamma in gamma_values:
        fold_results = []
        for fold, (train_index, val_index) in enumerate(kf.split(x_data, y_data)):
            X_train, X_val = x_data[train_index], x_data[val_index]
            y_train, y_val = y_data[train_index], y_data[val_index]
            
            # Early Stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',  # Monitor the validation loss
                patience=35,  # Wait for 5 epochs after the last improvement
                restore_best_weights=True,  # Restore the best weights from the epoch with the lowest validation loss
                verbose=1  # Print information about early stopping
            )

            model = create_nn_model(INPUT_SHARE) 
            model.compile(optimizer='adam', loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=gamma), metrics=['accuracy'])
            model.fit(
                X_train, 
                y_train, 
                epochs=epochs, 
                callbacks=[early_stopping],
                class_weight=class_weights_dict,
                batch_size=batch_size, 
                validation_data=(X_val, y_val),
                verbose=0)
            _, accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_results.append(accuracy)
        results[gamma] = np.mean(fold_results)
    return results


# gamma_values = [0, 1, 2, 3, 4, 5]

# model_exp = create_nn_model(INPUT_SHARE)

# cv_results = train_and_evaluate_with_cv(X.values, y.values, gamma_values)

# best_gamma = max(cv_results, key=cv_results.get)
# print(f"Best gamma: {best_gamma} with accuracy: {cv_results[best_gamma]}")



model4 = create_nn_model(INPUT_SHARE)

# Compile the Model
model4.compile(optimizer='adam', loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=5), metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=35,  # Wait for 5 epochs after the last improvement
    restore_best_weights=True,  # Restore the best weights from the epoch with the lowest validation loss
    verbose=1  # Print information about early stopping
)

# Train the Model
history4 = model4.fit(
    x_train,
    y_train,
    epochs=250, batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weights_dict,
    verbose=0)


plot_training_history(history4)


# Predict Probabilities & Convert to Binary Labels
y_prob4 = model4.predict(x_val)
# Evaluate Model
y_pred4 = (y_prob4 > 0.5).astype(int)
plot_roc_curve(y_val, y_pred4)


predict = model2.predict(X_test)


predict = predict.squeeze()


submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.head() 


submission['rainfall'] = predict
submission.to_csv("/kaggle/working/submission.csv", index=False)


submission

