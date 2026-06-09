%ls ../input/playground-series-s5e5/


train_path = '../input/playground-series-s5e5/train.csv'
test_path = '../input/playground-series-s5e5/test.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import random



train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_df.head(5)


train_df.info()


train_df.describe()


train_df.isna().sum()


target_column = 'Calories'


train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#plotting distribution of all columns
def plot_distribution(df, column):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
columns_to_plot = ['Calories', 'Duration', 'Heart_Rate', 'Body_Temp','Weight','Height','Age','Sex']
for column in columns_to_plot:
    plot_distribution(train_df, column)



# Plotting the distribution of the 'Sex' column
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Sex', palette='pastel')
plt.title('Distribution of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


#plotting correlation matrix
def plot_correlation_matrix(df):
    #exclude id and sex columns
    df = df.drop(columns=['id'])
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.show()
plot_correlation_matrix(train_df)


plot_correlation_matrix(test_df)


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Function to calculate VIF
def calculate_vif(df, features):
    numerical_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    vif_data = pd.DataFrame()
    vif_data['Feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data

# Calculate VIF for numerical features
numerical_features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Height', 'Weight']
vif_df = calculate_vif(train_df, numerical_features)
print(vif_df)



train_df.drop(columns=['Weight'], inplace=True)
numerical_features = ['Duration', 'Heart_Rate', 'Body_Temp', 'Age', 'Height']
vif_df = calculate_vif(train_df, numerical_features)
print(vif_df)


train_df.drop(columns=['Duration'], inplace=True)
numerical_features = ['Heart_Rate', 'Body_Temp', 'Age', 'Height']
vif_df = calculate_vif(train_df, numerical_features)
print(vif_df)


def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, encoding categorical variables,
    and scaling numerical features.
    
    Parameters:
        df (pd.DataFrame): The input dataframe to preprocess.
    
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Handle missing values (if any)
    # Since there are no missing values in the dataset, this step is skipped.

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    columns_to_drop = ['Weight', 'Duration']
    df.drop(columns=columns_to_drop, inplace=True)
    # Scale numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numerical_features = ['Age', 'Height', 'Heart_Rate', 'Body_Temp']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

# Apply preprocessing to the training set
train_df = pd.read_csv(train_path)
train_df = preprocess_data(train_df)


#plot correlation matrix after preprocessing
plot_correlation_matrix(train_df)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
# Split the data into features and target
X_train = train_df.drop(columns=['id', 'Calories'])
y_train = train_df['Calories']
# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Define the input shape explicitly
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='linear')  # Linear activation for regression
])




import tensorflow as tf

# Define RMSLE loss function
def rmsle(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=rmsle, metrics=['mae'])


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stopping])


#export model
model.save('model.h5')


#load model
# from tensorflow.keras.models import load_model
# model = load_model('model.h5', custom_objects={'rmsle': rmsle})


# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#make predictions on test set
test_df = pd.read_csv(test_path)
test_df = preprocess_data(test_df)
ids = test_df['id']
X_test = test_df.drop(columns=['id'])
y_pred = model.predict(X_test)
# Create submission DataFrame
submission_df = pd.DataFrame({'id': ids, 'Calories': y_pred.flatten()})
submission_df.to_csv('submission.csv', index=False)
submission_df.head(5)

