!pip install -q ipyplot


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set(style="darkgrid")


!pip install tensorflow


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Dropout
from tensorflow.keras.regularizers import L1
from tensorflow.keras.callbacks import ModelCheckpoint


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train_df.head(3)


train_df.shape


train_df.isna().sum()


train_df.describe()


test_df =pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_df.head(3)


id = test_df['id']


test_df.info() 


fig, ax = plt.subplots(1,2, figsize=(8,4))
sns.histplot(train_df['Calories'],bins=60, kde=True, ax= ax[0], color='green')
ax[0].set_title("Calories (Raw scale)")

sns.histplot(np.log1p(train_df['Calories']), bins=60, kde=True,
            ax=ax[1], color="tomato")
ax[1].set_title("Calories (log1p scale)")
plt.show();


print("Skewness :", train_df['Calories'].skew().round(3))
print("Kurtosis :", train_df["Calories"].kurt().round(3))



def preprocess_features(df):
    df = df.copy()

    # BMI calculation
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 45, 80], labels=['Youth', 'Adult', 'Senior'])

    # Intensity Score
    df['Intensity_Score'] = df['Heart_Rate'] / df['Duration']

    # Temperature deviation from 37°C
    df['Temp_Deviation'] = np.abs(df['Body_Temp'] - 37)

    # Height-Weight Ratio
    df['Height_Weight_Ratio'] = df['Height'] / df['Weight']

    return df



train_df_processed = preprocess_features(train_df)
train_df_processed.head(3)


test_df_processed = preprocess_features(test_df)
test_df_processed.head(3)


test_df_processed.drop(columns='id', axis=1, inplace=True)
train_df_processed.drop(columns='id',axis=1, inplace=True)
train_df_processed.columns


cat_cols =["Sex","Age_Group"]


num_cols = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp',
            'BMI','Intensity_Score','Temp_Deviation','Height_Weight_Ratio']

# Create subplots
fig, axis = plt.subplots(len(num_cols) // 3 + 1, 3, figsize=(15, 4 * (len(num_cols) // 3)))

# Flatten the axis array to avoid indexing issues
axis = axis.flatten()

for i, col in enumerate(num_cols):
    sns.kdeplot(x=train_df_processed[col], ax=axis[i], color="red", fill=True)
    axis[i].set_title(f"Train Dataset - {col}")

# Hide any unused subplots
for j in range(i + 1, len(axis)):
    axis[j].set_visible(False)

plt.tight_layout()
plt.show()



for col in num_cols:
    plt.figure(figsize = (4,3))
    sns.scatterplot(x = train_df_processed[col], y = train_df_processed["Calories"], alpha = 0.2)
    sns.regplot(x = train_df_processed[col], y = train_df_processed["Calories"], scatter = False, color = "red")
    plt.title(f"{col} vs Calories")
    plt.show();


x = train_df_processed.drop(columns=['Calories'])
y = train_df_processed['Calories'] 


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


X_train.shape, X_test.shape, y_train.shape, y_train.shape


most_frequent_age_group = X_train['Age_Group'].mode()[0]  # Get the most common value

X_train['Age_Group'].fillna(most_frequent_age_group, inplace=True)

X_test['Age_Group'].fillna(most_frequent_age_group, inplace=True)

test_df_processed['Age_Group'].fillna(most_frequent_age_group, inplace=True)



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


numerical_transformer = Pipeline(steps=[
    ('scaling', StandardScaler())
])
categorial_transformer = Pipeline(steps=[
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])


# Combining transformers in a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, num_cols),
    ('cat', categorial_transformer, cat_cols) 
])


# Pre-process the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
test_df_transformed = preprocessor.transform(test_df_processed)


# Get categorical feature names after transformation
cat_transformer = preprocessor.named_transformers_['cat']  # Extract the categorical transformer
cat_feature_names = cat_transformer.get_feature_names_out(cat_cols)  # Get the new feature names

# Convert to a list for easier use
cat_feature_names = list(cat_feature_names)
print(cat_feature_names)


feature_names = list(num_cols) + cat_feature_names
feature_names


# Recreate DataFrame with transformed data and original column names
X_train = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test = pd.DataFrame(X_test_transformed, columns=feature_names)
test_df = pd.DataFrame(test_df_transformed, columns=feature_names)


X_train.head(3)


X_train.shape


from tensorflow.keras.layers import Activation


model_seq = Sequential()
model_seq.add(Dense(30, input_dim=15, activation='relu',kernel_regularizer=L1(0.01)))
model_seq.add(BatchNormalization())
model_seq.add(Dropout(0.3))

# model_seq.add(Dense(20, activation='leaky_relu', kernel_regularizer=L1(0.01)))
# model_seq.add(BatchNormalization())
# model_seq.add(Dropout(0.3))

model_seq.add(Dense(10, activation='relu', kernel_regularizer= L1(0.01)))
model_seq.add(BatchNormalization())
model_seq.add(Dropout(0.3)) 

model_seq.add(Dense(1, activation='relu')) 
# Or alternatively, use softplus (always positive, smoother than relu)
# model_seq.add(Activation('softplus'))


# custom RMSLE Loss
import tensorflow as tf
import keras.backend as K

def rmsle(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0, tf.reduce_max(y_true))  # Ensure non-negative
    y_pred = tf.clip_by_value(y_pred, 0, tf.reduce_max(y_pred))
    
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))))

# Compile model with this loss function
model_seq.compile(optimizer='adam', loss=rmsle, metrics=[rmsle, 'mae'] )


checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True) 


history = model_seq.fit(
    X_train, y_train,
   validation_split=0.2,  # Use 20% of training data for validation
    epochs=20,
    batch_size=32,
    callbacks=[checkpoint]
)


y_test_pred = model_seq.predict(X_test)
y_test_pred[:10] 


final_submission_pred = model_seq.predict(test_df)
final_submission_pred[:10]



# Convert to 1D arrays
id = np.array(id).flatten()
final_submission_pred = np.array(final_submission_pred).flatten()

# Create DataFrame
submission = pd.DataFrame({
    'id': id,
    'Calories': final_submission_pred
})

print(submission.head(10))



# submission.to_csv('submission.csv', index=False)
# print("submission.csv created successfully!")


!pip install keras-tuner


y_train.min(), y_train.max()


import tensorflow as tf
import keras.backend as K

# Custom RMSLE metric

def root_mean_squared_log_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.math.log(y_true + 1) - tf.math.log(y_pred + 1))))



import keras_tuner as kt

# Define the model with tunable parameters
def build_model(hp):
    model = keras.Sequential()
    
    # Tunable number of units in the first layer
    model.add(Dense(hp.Int('units_1', min_value=16, max_value=128, step=16), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

    # Second dense layer
    model.add(Dense(hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='softplus'))  # activation= linear was giving nan values

    # Tunable optimizer learning rate
    optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001]))

    # Compile model
    model.compile(optimizer=optimizer, loss=root_mean_squared_log_error, metrics=[root_mean_squared_log_error])
    
    return model


import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt  

tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_root_mean_squared_log_error", direction="min"),  # Explicitly minimizing RMSLE
    max_trials=7,
    executions_per_trial=1,
    directory='keras_tuner_logs',
    project_name='hyperparameter_tuning'
)

tuner.search(X_train, y_train, epochs=20, validation_split=0.2)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Units Layer 1: {best_hps.get('units_1')}")
print(f"Best Dropout Layer 1: {best_hps.get('dropout_1')}")
print(f"Best Learning Rate: {best_hps.get('learning_rate')}")


best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=10, validation_split=0.2)


y_test_pred = best_model.predict(X_test)


test_df



final_sub = best_model.predict(test_df)


# Convert to 1D arrays
final_sub = np.array(final_sub).flatten()


submission = pd.DataFrame({
    'id': id,
    'Calories': final_sub
})


submission.head()


submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully!")




