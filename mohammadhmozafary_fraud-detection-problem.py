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



# # Function to extract features
def extract_features_from_day(df, day_column):
    
    # df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    # df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    # df['day_of_week'] = ((df[day_column] - 1) % 7) + 1


    # df['week_number'] = ((df[day_column] - 1) // 7) + 1
    
    
    
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

    
    # df['half_of_year'] = df[day_column].apply(lambda d: 1 if d <= 182 else 2)

    
    # window_size = 7
    # df['temp_roll_mean'] = df['temparature'].rolling(window=window_size, min_periods=1).mean()
    # df['temp_roll_std'] = df['temparature'].rolling(window=window_size, min_periods=1).std()

    # df['dew_roll_mean'] = df['dewpoint'].rolling(window=window_size, min_periods=1).mean()
    # df['dew_roll_std'] = df['dewpoint'].rolling(window=window_size, min_periods=1).std()

    # df['cloud_roll_mean'] = df['cloud'].rolling(window=window_size, min_periods=1).mean()
    # df['cloud_roll_std'] = df['cloud'].rolling(window=window_size, min_periods=1).std()
    
    # df['press_roll_mean'] = df['pressure'].rolling(window=window_size, min_periods=1).mean()
    # df['press_roll_std'] = df['pressure'].rolling(window=window_size, min_periods=1).std()

    # df['average_temp']=(df['mintemp']+df['maxtemp'])/2

    # df['temp_diff'] = df['temparature'].diff().fillna(0)

#     df['pressure_diff']=df['pressure'].diff().fillna(0)

    #df['dewpoint_diff']=df['dewpoint'].diff().fillna(0)

#     df['average_temp_diff']=df['average_temp'].diff().fillna(0)

# #    fft_vals = fft(df['pressure'].values)
#  #   df['fft_real'] = np.real(fft_vals)
#   #  df['fft_imag'] = np.imag(fft_vals)

#     # One-hot encode the season
    season_dummies = pd.get_dummies(df['season'], prefix='season').astype(float)
    
   


#     # Combine the original DataFrame with the one-hot encoded season columns
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
    for feature in X_train.columns:
        if X_train[feature].dtype in ['int64', 'float64']:
            skewness = X_train[feature].skew()
            if abs(skewness) > 0.4:
                X_train=transformation(X_train,feature)
                X_test=transformation(X_test,feature)
            

    if(handle_outliers==True):
        X_train=handle_outliers(X_train,X_train.columns)
    
    not_to_scale=['season_fall','season_spring','season_summer','season_winter']
    to_scale = X_train.columns.difference(not_to_scale)
    # to_scale=X_train.columns

    X_train=scale_data(X_train,to_scale,'minmax')
    X_test=scale_data(X_test,to_scale,'minmax')
    print("train shape after preprocessing: ",X_train.shape)
    print("test shape after Preprocessing: ",X_test.shape)


    return X_train,y_train,X_test

    


X,y,test_set=preprocessing_pipeline(train,test)



test_set.sample(5)


y.value_counts()


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


data=X.copy()
data['rainfall']=y
minority=data[data['rainfall']==0]
majority=data[data['rainfall']==1]

visualisation_initial = pd.concat([minority, majority])
features, labels = visualisation_initial.drop('rainfall', axis=1).values, \
                   visualisation_initial['rainfall'].values
tsne_scatter(features, labels, dimensions=2)



tsne_scatter(features, labels, dimensions=3)



majority = majority.sample(frac=1).reset_index(drop=True)
X_train = majority.iloc[:int((len(majority))*0.8)].drop('rainfall', axis=1)

# testing  set: the remaining non-fraud + all the fraud 
X_test = pd.concat([majority.iloc[int((len(majority))*0.8):],minority]).sample(frac=1)

# train // validate - no labels since they're all clean anyway
X_train, X_validate = train_test_split(X_train, 
                                       test_size=0.2, 
                                       random_state=42)

# manually splitting the labels from the test df
X_test, y_test = X_test.drop('rainfall', axis=1).values, X_test['rainfall'].values


input_dim=X_train.shape[1]


# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True,min_delta=0.0005,
    mode='min',)

# Define the model
autoencoder = Sequential([
    Input(shape=(input_dim,)),
    Dense(64,activation='elu'),
    Dense(32,activation='elu'),
    Dense(16,activation='elu'),
    Dense(8,activation='elu'),
    Dense(4, activation='elu', ),
    

    Dense(2, activation='elu', ),

    
    Dense(4, activation='elu', ),
    Dense(8,activation='elu'),
    Dense(16,activation='elu'),
    Dense(32,activation='elu'),
    
    Dense(input_dim, activation='linear', ),
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['acc'])

# Train the model
history = autoencoder.fit(X_train, X_train, epochs=200, batch_size=32, validation_data=(X_validate, X_validate), 
                    callbacks=[early_stopping], verbose=1,shuffle=True)



reconstructions = autoencoder.predict(X_test)



mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)


clean = mse[y_test==0]
fraud = mse[y_test==1]

fig, ax = plt.subplots(figsize=(6,6))

ax.hist(clean, bins=50, density=True, label="clean", alpha=.6, color="green")
ax.hist(fraud, bins=50, density=True, label="fraud", alpha=.6, color="red")

plt.title("(Normalized) Distribution of the Reconstruction Loss")
plt.legend()
plt.show()


THRESHOLD = 2.5

def mad_score(points):
    """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)
    
    return 0.6745 * ad / mad

z_scores = mad_score(mse)
outliers = z_scores > THRESHOLD


print(f"Detected {np.sum(outliers):,} outliers in a total of {np.size(z_scores):,} transactions [{np.sum(outliers)/np.size(z_scores):.2%}].")


from sklearn.metrics import (confusion_matrix, 
                             precision_recall_curve)

# get (mis)classification
cm = confusion_matrix(y_test, outliers)

# true/false positives/negatives
(tn, fp, 
 fn, tp) = cm.flatten()


print(f"""The classifications using the MAD method with threshold={THRESHOLD} are as follows:
{cm}

% of transactions labeled as fraud that were correct (precision): {tp}/({fp}+{tp}) = {tp/(fp+tp):.2%}
% of fraudulent transactions were caught succesfully (recall):    {tp}/({fn}+{tp}) = {tp/(fn+tp):.2%}""")

