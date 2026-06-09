# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization, LeakyReLU, Input
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


print(train.shape)
train.head()


print(test.shape)
test.head()


print(submission.shape)
submission.head()


train.describe()


train.info()


train.isna().sum()



# Set up the number of subplots (rows and columns)
num_cols = 2  # Number of columns in the subplot grid
num_rows = int(np.ceil(len(train.columns[2:-1]) / num_cols))  # Calculate the required number of rows

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()  # Flatten the axes array for easier indexing

# Loop through each column (excluding the first one)
for idx, col in enumerate(train.columns[2:-1]):
    # Calculate skewness
    skewness = train[col].skew()

    # Plotting on the corresponding subplot
    ax = axes[idx]
    ax.hist(train[col], bins=40, edgecolor="black", color='skyblue')
    ax.set_title(f"Distribution of {col}\nSkewness: {skewness:.2f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

# Hide any extra subplots (in case the grid is larger than needed)
for i in range(idx + 1, len(axes)):
    axes[i].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



def plot_features_by_day(df, day_column):
    numeric_cols = df.select_dtypes(include="number").columns[2:-1]
    num_plots = len(numeric_cols)

    rows = (num_plots // 2) + (num_plots % 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 6 * rows))

    axes = axes.flatten() if num_plots > 1 else [axes]

    colors = plt.cm.get_cmap("tab10", len(numeric_cols))

    for ax, col, color in zip(axes, numeric_cols, colors.colors):
        ax.scatter(df[day_column], df[col], color=color, marker='o', linestyle='-', label=col)
        ax.set_title(f"{col} over Days")
        ax.set_xlabel("Day")
        ax.set_ylabel(col)
        ax.legend()
        ax.grid()
        ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

plot_features_by_day(train, "day")




def handle_outliers(df, columns):
    """
    This function clips outliers based on the IQR (Interquartile Range) method for specific columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to clean.
    columns (list): The list of column names on which to apply outlier clipping.
    
    Returns:
    pd.DataFrame: The DataFrame with outliers clipped to the IQR boundaries.
    """
    clipped_count = 0
    
    # Loop through each specified column
    for col in columns:
        # Calculate Q1, Q3, and IQR for the current column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        clipped_count += outliers
        
        # Clip the outliers to the bounds
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    print("Number of outliers clipped: ", clipped_count)
    return df



def scale_data(df,columns_to_scale,scaler="robust"):
    if(scaler=="standard"):
        scaler=StandardScaler()
    elif(scaler=="minmax"):
        scaler=MinMaxScaler()
    elif(scaler=="robust"):
        scaler=RobustScaler()
        
    remaining_columns = df.columns.difference(columns_to_scale)
    
    # Scale the selected columns
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[columns_to_scale]), columns=columns_to_scale)
    
    # Combine scaled and unscaled columns
    df_final = pd.concat([df_scaled, df[remaining_columns].reset_index(drop=True)], axis=1)
    
    return df_final


def transformation(df,feature):
    transformer = PowerTransformer(method='yeo-johnson')  # Initialize Yeo-Johnson transformer
    df[feature] = transformer.fit_transform(df[[feature]])

    return df



# Function to extract features
def extract_features_from_day(df, day_column):
    # 1. Day of the week (approximate, assuming year starts on a Monday)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    df['day_of_week'] = ((df[day_column] - 1) % 7) + 1

    # 2. Week number
    df['week_number'] = ((df[day_column] - 1) // 7) + 1
    
    
    # 3. Season
    def get_season(day):
        if day <= 79 or day >= 355:
            return 'winter'
        elif day <= 171:
            return 'spring'
        elif day <= 264:
            return 'summer'
        else:
            return 'fall'

    df['season'] = df[day_column].apply(get_season)

    # 4. Half of the year
    df['half_of_year'] = df[day_column].apply(lambda d: 1 if d <= 182 else 2)

    
    window_size = 7
    df['temp_roll_mean'] = df['temparature'].rolling(window=window_size, min_periods=1).mean()
    df['temp_roll_std'] = df['temparature'].rolling(window=window_size, min_periods=1).std()

    df['dew_roll_mean'] = df['dewpoint'].rolling(window=window_size, min_periods=1).mean()
    df['dew_roll_std'] = df['dewpoint'].rolling(window=window_size, min_periods=1).std()

    # df['cloud_roll_mean'] = df['cloud'].rolling(window=window_size, min_periods=1).mean()
    # df['cloud_roll_std'] = df['cloud'].rolling(window=window_size, min_periods=1).std()
    
    df['press_roll_mean'] = df['pressure'].rolling(window=window_size, min_periods=1).mean()
    df['press_roll_std'] = df['pressure'].rolling(window=window_size, min_periods=1).std()

    df['average_temp']=(df['mintemp']+df['maxtemp'])/2

    df['temp_diff'] = df['temparature'].diff().fillna(0)

    df['pressure_diff']=df['pressure'].diff().fillna(0)

    #df['dewpoint_diff']=df['dewpoint'].diff().fillna(0)

    df['average_temp_diff']=df['average_temp'].diff().fillna(0)

#    fft_vals = fft(df['pressure'].values)
 #   df['fft_real'] = np.real(fft_vals)
  #  df['fft_imag'] = np.imag(fft_vals)

    # One-hot encode the season
    season_dummies = pd.get_dummies(df['season'], prefix='season').astype(float)
    
   


    # Combine the original DataFrame with the one-hot encoded season columns
    df = pd.concat([df, season_dummies], axis=1).drop("season",axis=1)

    return df


def feature_selection(df):
    
    pca=PCA(n_components=0.99)
    reduced=pca.fit_transform(df)
    n_components=reduced.shape[1]
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    reduced = pd.DataFrame(reduced, columns=pca_columns)
    print('Number of reduced features: ',df.shape[1]-n_components)
    return reduced,pca



def impute(df):
    # Separate categorical and numerical columns
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    except:
        print("Error in Categorical imputing")
        pass 
    try:
        numerical_cols = df.select_dtypes(exclude=['object']).columns
    # Impute numerical features with the mean value
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    except:
        print("Error in numerical imputing")
        pass
    
    return df


def preprocessing_pipeline(train,test,handle_outliers=False):
    print("train set shape before preprocessing: ",train.shape)
    print("test set shape before preprocessing: ",test.shape)
    target_column='rainfall'
    X_test=test.drop('id',axis=1)
    X_train=train.drop('id',axis=1)
    X_train=X_train.drop(target_column,axis=1)
    
    y_train=train[target_column]
    
    X_train=X_train.drop_duplicates()
    
    X_train=extract_features_from_day(X_train, "day")
    X_test=extract_features_from_day(X_test, "day")
    
    
    

    X_train=impute(X_train)
    X_test=impute(X_test)
    # for feature in X_train.columns:
    #     if X_train[feature].dtype in ['int64', 'float64']:
    #         skewness = X_train[feature].skew()
    #         if abs(skewness) > 0.4:
    #             X_train=transformation(X_train,feature)
    #             X_test=transformation(X_test,feature)
            

    if(handle_outliers==True):
        X_train=handle_outliers(X_train,X_train.columns)
    
    not_to_scale=['season_fall','season_spring','season_summer','season_winter']
    to_scale = X_train.columns.difference(not_to_scale)
   

    X_train=scale_data(X_train,to_scale)
    X_test=scale_data(X_test,to_scale)
    print("train shape after preprocessing: ",X_train.shape)
    print("test shape after Preprocessing: ",X_test.shape)


    return X_train,y_train,X_test

    


X,y,test_set=preprocessing_pipeline(train,test)



test_set.sample(5)


data=X.copy()
data['rainfall']=y
minority=data[data['rainfall']==0]
majority=data[data['rainfall']==1]



visualisation_initial = pd.concat([minority, majority])
features, labels = visualisation_initial.drop('rainfall', axis=1).values, \
                   visualisation_initial['rainfall'].values


def tsne_scatter(features, labels, dimensions=2):
    if dimensions not in (2, 3):
        raise ValueError('tsne_scatter can only plot in 2d or 3d')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=42).fit_transform(features)
    
    # initialising the plot
    fig, ax = plt.subplots(figsize=(8,8))
    
    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels==0)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Minority'
    )
    ax.scatter(
        *zip(*features_embedded[np.where(labels==1)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Majority'
    )

    plt.legend(loc='best')
    plt.show;


tsne_scatter(features, labels, dimensions=2)



y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.show()


smote=SMOTE(sampling_strategy='minority')
X_resample,y_resample=smote.fit_resample(X,y)



y_resample.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.show()


np.isinf(X_resample).sum()



X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

test_accuracy=accuracy_score(y_test,y_pred)
train_accuracy=accuracy_score(y_train,y_train_pred)

print("Model accuracy on test set: ",test_accuracy)
print("Model accuracy on train set: ",train_accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


svm_model=SVC(kernel='rbf',C=3,gamma="scale",random_state=42)
svm_model.fit(X_train,y_train)
y_pred=svm_model.predict(X_test)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
y_train_pred=svm_model.predict(X_train)
train_accuracy=accuracy_score(y_train,y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




# Initialize the XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Use 'multi:softmax' for multi-class classification
    eval_metric='logloss',
    n_estimators=150,  # Number of boosting rounds
    learning_rate=0.2,  # Step size shrinkage
    max_depth=7,  # Depth of each tree
    subsample=0.9,  # Randomly select 80% of data for training each tree
    colsample_bytree=0.9,  # Randomly select 80% of features for training each tree
    reg_alpha=10,
    reg_lambda=100
    ,random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)
# Predict on test set
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
y_train_pred=xgb_model.predict(X_train)
train_accuracy=accuracy_score(y_train,y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f}")
# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Define the model
model = Sequential([
    Input(shape=(X_resample.shape[1],)),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dense(16, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])

# Train the model
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_val, y_val), 
                    callbacks=[early_stopping], verbose=1)



test_loss, test_accuracy = model.evaluate(X_test,y_test)
print(f"Test Loss : {test_loss}")
print(f"Test auc: {test_accuracy}")

