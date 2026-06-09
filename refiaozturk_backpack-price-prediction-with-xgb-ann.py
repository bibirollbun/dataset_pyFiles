import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import models
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

pd.options.display.float_format = '{:.3f}'.format

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


df0 = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_train = df0.copy()
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# Store our ID for easy access
test_ids = df_test['id']


df_train.head()


df_train.tail()


train_extra.head()


df_test.head()


df_train.columns == train_extra.columns


df = pd.concat([df_train, train_extra], axis=0, ignore_index=True)
df.head()


df.info(show_counts=True)


df_test.info()


df.describe().T


df.describe(include="object").T


df.duplicated().sum()


df.isnull().sum()


msno.bar(df, color="mediumseagreen");


# Numerical features
numeric_columns = df.select_dtypes(include=['number']).columns

# Categorical features
categoric_features = df.select_dtypes(include=['object', 'category']).columns.tolist()


# Checking out unique values in categorical features
for col in categoric_features:
    print(f"{col}")
    print("-" * 20)
    print("\n".join(df[col].unique().astype(str)))
    print("\n")


# Calculate the number of rows based on the specified number of columns
num_columns = 4
num_rows = (len(numeric_columns) // num_columns) + (1 if len(numeric_columns) % num_columns != 0 else 0)

# Create a figure with specified size
fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot each numeric column
for x, col in enumerate(numeric_columns):
    sns.boxplot(data=df[col], color='mediumseagreen', ax=axes[x])
    axes[x].set_title(col)

# Hide any unused axes (if there are any)
for i in range(x + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout() 
plt.show()


columns = numeric_columns

# Calculate the number of rows based on the specified number of columns
num_columns = 2
num_rows = (len(numeric_columns) // num_columns) + (1 if len(numeric_columns) % num_columns != 0 else 0)

# Create a figure with specified size
fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))
axes = axes.flatten()

for i, column in enumerate(columns):
    sns.histplot(df[column], kde=True, ax=axes[i], color='mediumseagreen')
    axes[i].set_title(column)
    
for i in range(len(columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


df.isnull().sum()


df_test.isnull().sum()


df.Brand.isnull().sum()


df.Brand.value_counts()


df.groupby(["Brand"])["Price"].mean()


df['Brand'].fillna('Unknown', inplace=True)


df.Brand.isnull().sum()


df_test['Brand'].fillna('Unknown', inplace=True)


df_test.Brand.isnull().sum()


df.Material.isnull().sum()


df.Material.value_counts()


df['Material'] = df.groupby('Brand')['Material'].transform(lambda x: x.fillna(x.mode()[0]))


df.Material.isnull().sum()


df_test['Material'] = df_test.groupby('Brand')['Material'].transform(lambda x: x.fillna(x.mode()[0]))


df_test.Material.isnull().sum()


df.Size.isnull().sum()


df['Size'] = df.groupby('Brand')['Size'].transform(lambda x: x.fillna(x.mode()[0]))


df.Size.isnull().sum()


df_test['Size'] = df_test.groupby('Brand')['Size'].transform(lambda x: x.fillna(x.mode()[0]))


df_test.Size.isnull().sum()


df["Laptop Compartment"].isnull().sum()


df.groupby(["Brand", "Compartments"])["Laptop Compartment"].agg(lambda x: x.mode()[0])


df["Laptop Compartment"] = df.groupby(["Brand", "Compartments"])["Laptop Compartment"].transform(lambda x: x.fillna(x.mode()[0]))


df["Laptop Compartment"].isnull().sum()


df_test["Laptop Compartment"] = df_test.groupby(["Brand", "Compartments"])["Laptop Compartment"].transform(lambda x: x.fillna(x.mode()[0]))


df_test["Laptop Compartment"].isnull().sum()


df.Waterproof.isnull().sum()


df.groupby(["Brand", "Material"])["Waterproof"].value_counts()


def fill_with_proportions(group):
    yes_count = (group == "Yes").sum()
    no_count = (group == "No").sum()
    total_count = len(group)
    
    # Calculate probabilities
    yes_prob = yes_count / total_count
    no_prob = no_count / total_count
    
    # Normalize probabilities to ensure they sum to 1
    prob_sum = yes_prob + no_prob
    yes_prob /= prob_sum
    no_prob /= prob_sum

    # Randomly fill NaN values based on the normalized probabilities
    return group.apply(lambda x: np.random.choice(["Yes", "No"], p=[yes_prob, no_prob]) if pd.isna(x) else x)

# Apply the function to each group
df["Waterproof"] = df.groupby(["Brand", "Material"])["Waterproof"].transform(fill_with_proportions)


df.Waterproof.isnull().sum()


df_test["Waterproof"] = df_test.groupby(["Brand", "Material"])["Waterproof"].transform(fill_with_proportions)


df_test.Waterproof.isnull().sum()


df.Style.isnull().sum()


df.groupby(["Brand", "Material"])["Style"].value_counts()


# Group by Brand and Material, then apply filling with proportionate distribution
def fill_style_with_proportions(group):
    style_counts = group.value_counts(normalize=True)  # Get the distribution of 'Style' values
    
    # Randomly fill NaN values based on the calculated proportions
    return group.apply(lambda x: np.random.choice(style_counts.index, p=style_counts.values) if pd.isna(x) else x)

# Apply the function to each group
df["Style"] = df.groupby(["Brand", "Material"])["Style"].transform(fill_style_with_proportions)


df.Style.isnull().sum()


df_test["Style"] = df_test.groupby(["Brand", "Material"])["Style"].transform(fill_style_with_proportions)


df_test.Style.isnull().sum()


df.Color.isnull().sum()


df.groupby(["Brand", "Material"])["Color"].value_counts()


# Group by Brand and Material, then apply filling with proportionate distribution
def fill_color_with_proportions(group):
    color_counts = group.value_counts(normalize=True)  # Get the distribution of 'Color' values
    
    # Randomly fill NaN values based on the calculated proportions
    return group.apply(lambda x: np.random.choice(color_counts.index, p=color_counts.values) if pd.isna(x) else x)

# Apply the function to each group
df["Color"] = df.groupby(["Brand", "Material"])["Color"].transform(fill_color_with_proportions)


df.Color.isnull().sum()


df_test["Color"] = df_test.groupby(["Brand", "Material"])["Color"].transform(fill_color_with_proportions)


df_test.Color.isnull().sum()


df["Weight Capacity (kg)"].isnull().sum()


df.groupby(["Brand", "Size"])["Weight Capacity (kg)"].mean()


df['Weight Capacity (kg)'] = df.groupby(['Brand', 'Size'])['Weight Capacity (kg)'].transform(lambda x: x.fillna(x.median()))


df["Weight Capacity (kg)"].isnull().sum()


df_test['Weight Capacity (kg)'] = df_test.groupby(['Brand', 'Size'])['Weight Capacity (kg)'].transform(lambda x: x.fillna(x.median()))


df_test["Weight Capacity (kg)"].isnull().sum()


# last check
df.isnull().sum().sum()


# last check
df_test.isnull().sum().sum()


# saving clean data
df.to_csv("train_clean.csv", index=False)


# saving clean data
df_test.to_csv("test_clean.csv", index=False)


def detect_outliers(df, col_name,tukey=1.5):
    ''' 
    this function detects outliers based on 1.5 time IQR and
    returns the number of lower and uper limit and number of outliers respectively
    '''
    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
                      
    upper_limit = third_quartile+(tukey*IQR)
    lower_limit = first_quartile-(tukey*IQR)
    outlier_count = 0
                      
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count +=1
    return lower_limit, upper_limit, outlier_count


threshold = 1.5
out_cols = []

for col in numeric_columns:
    print(
        f"{col}\nlower:{detect_outliers(df, col,threshold)[0]} \nupper:{detect_outliers(df, col,threshold)[1]}\
        \noutlier:{detect_outliers(df, col,threshold)[2]}\n*-*-*-*-*-*-*"
    )
    if detect_outliers(df, col,threshold)[2] > 0 :
        out_cols.append(col)
print(out_cols)  


df = pd.read_csv("/kaggle/input/s5e2-clean-datasets/train_clean.csv")
df.head()


# dropping unnecessary features from df
df.drop(columns="id", inplace=True)


X = df.drop(columns="Price")
y = df.Price


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


categoric_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categoric_features


from sklearn.preprocessing import OrdinalEncoder

size_order = ['Small', 'Medium', 'Large']
encoder = OrdinalEncoder(categories=[size_order])

# Fitting train set only
X_train['Size'] = encoder.fit_transform(X_train[['Size']])

# Transforming test data
X_test['Size'] = encoder.transform(X_test[['Size']])


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

cat_onehot = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_onehot),
    remainder='passthrough'
)


X_train = pd.DataFrame(column_trans.fit_transform(X_train), columns=column_trans.get_feature_names_out())
X_test = pd.DataFrame(column_trans.transform(X_test), columns=column_trans.get_feature_names_out())


X_train.head()


X_test.head()


scaler = MinMaxScaler() 


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV

model_xgb = XGBRegressor(random_state=42)
model_xgb.fit(X_train, y_train)


scores = cross_validate(model_xgb, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv= 10,
                       return_train_score=True)
pd.DataFrame(scores).iloc[:, 2:].mean()


xgb_model = XGBRegressor(
    n_estimators=1000,     
    learning_rate=0.05,   
    max_depth=6,          
    subsample=0.8, 
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

xgb_model.fit(
    X_train, y_train,
    early_stopping_rounds=50,              
    eval_set=[(X_test, y_test)],          
    verbose=100                           
)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_val(model, X_train, y_train, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)


y_pred = xgb_model.predict(X_test)


train_val(xgb_model, X_train, y_train, X_test, y_test)


xgb_model2 = XGBRegressor(
    device="cuda",
    enable_categorical=True,
    early_stopping_rounds=25,
    max_depth=3,  
    n_estimators=1649,  
    learning_rate=0.11223067797668791,  
    subsample=0.9289161715595431,  
    colsample_bytree=0.9309622394094211,  
    reg_alpha=0.2808737551552198,  
    reg_lambda=2.391197795322899,  
    min_child_weight=7
)

xgb_model2.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  
    verbose=200 
)


y_pred = xgb_model2.predict(X_test)


train_val(xgb_model2, X_train, y_train, X_test, y_test)


xgb_model4 = XGBRegressor(
    n_estimators=2000,     
    learning_rate=0.01,      
    max_depth=8,          
    subsample=0.7,           
    colsample_bytree=0.7,   
    gamma=1,                   
    reg_alpha=0.5,            
    reg_lambda=1.5,        
    random_state=42,
    objective='reg:squarederror'
)

xgb_model4.fit(
    X_train, y_train,
    early_stopping_rounds=100,
    eval_set=[(X_test, y_test)],
    verbose=100
)


y_pred = xgb_model4.predict(X_test)


train_val(xgb_model4, X_train, y_train, X_test, y_test)


residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()


from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz

visualizer = RadViz(size=(720, 600))
model = xgb_model4
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();


X_train.shape


from tensorflow.keras.metrics import RootMeanSquaredError

rmse = RootMeanSquaredError()


from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score


def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    score = r2_score(actual, pred)
    return print("r2_score:", score, "\nmae:", mae, "\nmse:", mse, "\nrmse:", rmse)


model1 = Sequential()

model1.add(Dense(1024, activation="relu", input_dim=X_train.shape[1]))
model1.add(Dropout(0.2))

model1.add(Dense(512, activation="relu"))
model1.add(Dropout(0.3))

model1.add(Dense(256, activation="relu"))

model1.add(Dense(128, activation="relu"))
model1.add(Dropout(0.2))

model1.add(Dense(1, activation="linear"))

model1.compile(optimizer='adam', loss='mse', metrics=[rmse])


model1.summary()


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


model1.fit(X_train, y_train, batch_size = 3000, epochs = 100, validation_data=(X_test, y_test), callbacks=[early_stopping])


model1.save("ann-1.keras")


summary = pd.DataFrame(model1.history.history)
summary.head()


plt.figure(figsize=(15, 6))
plt.plot(summary.loss, label="loss")
plt.plot(summary.val_loss, label="val_loss")
plt.legend(loc="upper right")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


plt.figure(figsize=(15, 6))
plt.plot(summary.root_mean_squared_error, label="rmse")
plt.plot(summary.val_root_mean_squared_error, label="val_rmse")
plt.legend(loc="upper left")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.show()


model1.evaluate(X_test, y_test, verbose=0)


y_pred = model1.predict(X_test)


eval_metric(y_test, y_pred)


from tensorflow.keras.layers import BatchNormalization

model2 = Sequential()

model2.add(Dense(512, activation="relu", input_dim=X_train.shape[1]))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))

model2.add(Dense(128, activation="relu"))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))

model2.add(Dense(64, activation="relu"))
model2.add(BatchNormalization())
model2.add(Dropout(0.1))

model2.add(Dense(1, activation="linear"))

opt = Adam(learning_rate = 0.001)
model2.compile(optimizer= opt, loss= 'mse', metrics= [rmse])


model2.summary()


model2.fit(X_train, y_train, batch_size = 3000, epochs = 100, validation_data=(X_test, y_test), callbacks=[early_stopping])


model2.save("ann-2.keras")


summary = pd.DataFrame(model2.history.history)
summary.head()


plt.figure(figsize=(15, 6))
plt.plot(summary.loss, label="loss")
plt.plot(summary.val_loss, label="val_loss")
plt.legend(loc="upper right")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


plt.figure(figsize=(15, 6))
plt.plot(summary.root_mean_squared_error, label="rmse")
plt.plot(summary.val_root_mean_squared_error, label="val_rmse")
plt.legend(loc="upper left")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.show()


model2.evaluate(X_test, y_test, verbose=0)


y_pred = model2.predict(X_test)


eval_metric(y_test, y_pred)


from tensorflow.keras import regularizers

model3 = Sequential()

model3.add(Dense(1024, activation="relu", input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(0.01)))
model3.add(BatchNormalization())
model3.add(Dropout(0.2))

model3.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)))  
model3.add(BatchNormalization()) 
model3.add(Dropout(0.3))

model3.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)))  
model3.add(BatchNormalization())

model3.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))) 
model3.add(BatchNormalization())
model3.add(Dropout(0.2))

model3.add(Dense(1, activation="linear"))

opt = Adam(learning_rate = 0.0005)
model3.compile(optimizer= opt, loss='mse', metrics=[rmse])


model3.summary()


model3.fit(X_train, y_train, batch_size=3000, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])


model3.save("ann-3.keras")


summary = pd.DataFrame(model3.history.history)
summary.head()


plt.figure(figsize=(15, 6))
plt.plot(summary.loss, label="loss")
plt.plot(summary.val_loss, label="val_loss")
plt.legend(loc="upper right")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


plt.figure(figsize=(15, 6))
plt.plot(summary.root_mean_squared_error, label="rmse")
plt.plot(summary.val_root_mean_squared_error, label="val_rmse")
plt.legend(loc="upper left")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.show()


model3.evaluate(X_test, y_test, verbose=0)


y_pred = model3.predict(X_test)


eval_metric(y_test, y_pred)


from tensorflow.keras.layers import LeakyReLU

model4 = Sequential()

model4.add(Dense(512, input_dim=X_train.shape[1], kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
model4.add(LeakyReLU(alpha=0.1))
model4.add(BatchNormalization())
model4.add(Dropout(0.2))

model4.add(Dense(512, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))  
model4.add(LeakyReLU(alpha=0.1))
model4.add(BatchNormalization())
model4.add(Dropout(0.3))

model4.add(Dense(256, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))  
model4.add(LeakyReLU(alpha=0.1))
model4.add(BatchNormalization())

model4.add(Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))) 
model4.add(LeakyReLU(alpha=0.1))
model4.add(BatchNormalization())
model4.add(Dropout(0.2))

model4.add(Dense(1, activation="linear"))

opt = Adam(learning_rate = 0.0001)
model4.compile(optimizer= opt, loss='mse', metrics=[rmse])


model4.summary()


model4.fit(X_train, y_train, batch_size=3000, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])


model4.save("ann-4.keras")


summary = pd.DataFrame(model4.history.history)
summary.head()


plt.figure(figsize=(15, 6))
plt.plot(summary.loss, label="loss")
plt.plot(summary.val_loss, label="val_loss")
plt.legend(loc="upper right")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


plt.figure(figsize=(15, 6))
plt.plot(summary.root_mean_squared_error, label="rmse")
plt.plot(summary.val_root_mean_squared_error, label="val_rmse")
plt.legend(loc="upper left")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.show()


model4.evaluate(X_test, y_test, verbose=0)


y_pred = model4.predict(X_test)


eval_metric(y_test, y_pred)


model5 = Sequential()

model5.add(Dense(512, activation="relu", input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(0.01)))
model5.add(BatchNormalization())
model5.add(Dropout(0.2))

model5.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)))  
model5.add(BatchNormalization()) 
model5.add(Dropout(0.3))

model5.add(Dense(64, activation="relu"))

model5.add(Dense(1, activation="linear"))

opt = Adam(learning_rate = 0.0001)
model5.compile(optimizer= opt, loss='mse', metrics=[rmse])


model5.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=15)


model5.fit(X_train, y_train, batch_size=2048, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping])


model5.save("ann-5.keras")


summary = pd.DataFrame(model5.history.history)
summary.head()


plt.figure(figsize=(15, 6))
plt.plot(summary.loss, label="loss")
plt.plot(summary.val_loss, label="val_loss")
plt.legend(loc="upper right")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


plt.figure(figsize=(15, 6))
plt.plot(summary.root_mean_squared_error, label="rmse")
plt.plot(summary.val_root_mean_squared_error, label="val_rmse")
plt.legend(loc="upper left")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.show()


model5.evaluate(X_test, y_test, verbose=0)


y_pred = model5.predict(X_test)


eval_metric(y_test, y_pred)


from tensorflow.keras import regularizers

model6 = Sequential()

model6.add(Dense(1024, activation="relu", input_dim=X_train.shape[1], kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
model6.add(BatchNormalization())
model6.add(Dropout(0.2))

model6.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))  
model6.add(BatchNormalization()) 
model6.add(Dropout(0.3))

model6.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))  
model6.add(BatchNormalization())

model6.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))) 
model6.add(BatchNormalization())
model6.add(Dropout(0.2))

model6.add(Dense(1, activation="linear"))

opt = Adam(learning_rate = 0.0001)
model6.compile(optimizer= opt, loss='mse', metrics=[rmse])


model6.summary()


model6.fit(X_train, y_train, batch_size=3000, epochs=150, validation_data=(X_test, y_test), callbacks=[early_stopping])


model6.save("ann-6.keras")


summary = pd.DataFrame(model6.history.history)
summary.head()


plt.figure(figsize=(15, 6))
plt.plot(summary.loss, label="loss")
plt.plot(summary.val_loss, label="val_loss")
plt.legend(loc="upper right")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()


plt.figure(figsize=(15, 6))
plt.plot(summary.root_mean_squared_error, label="rmse")
plt.plot(summary.val_root_mean_squared_error, label="val_rmse")
plt.legend(loc="upper left")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.show()


model6.evaluate(X_test, y_test, verbose=0)


y_pred = model6.predict(X_test)


eval_metric(y_test, y_pred)


from tensorflow.keras.models import load_model

new_model = load_model('/kaggle/working/ann-2.keras', custom_objects={'rmse': rmse})


df_test = pd.read_csv("/kaggle/input/s5e2-clean-datasets/test_clean.csv")


# dropping unnecessary features from df_test
df_test.drop(columns="id", inplace=True)


df_test.head()


from sklearn.preprocessing import OrdinalEncoder

size_order = ['Small', 'Medium', 'Large']
encoder = OrdinalEncoder(categories=[size_order])

df_test['Size'] = encoder.fit_transform(df_test[['Size']])


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

cat_onehot = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_onehot),
    remainder='passthrough'
)


df_test = pd.DataFrame(column_trans.fit_transform(df_test), columns=column_trans.get_feature_names_out())


df_test.head()


scaler = MinMaxScaler()
df_test = scaler.fit_transform(df_test)


prediction = xgb_model2.predict(df_test)


# Saving predictions according to Kaggle competition format
submit_df = pd.DataFrame({'id': test_ids, 'Price': prediction})
submit_df.head()


submit_df.to_csv("xgb_submit.csv", index=False)  


# let's make predictions with test data
prediction = new_model.predict(df_test)


# If prediction is 2D, convert to 1D
prediction = prediction.flatten()


# Saving predictions according to Kaggle competition format
submit_df = pd.DataFrame({'id': test_ids, 'Price': prediction})
submit_df.head()


submit_df.to_csv("backpackprice_submit2.csv", index=False)  

