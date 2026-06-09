TRAIN_URL = "/kaggle/input/innovative-ai-challenge-2024/train.csv"
TEST_URL = "/kaggle/input/innovative-ai-challenge-2024/test.csv"
SUBMISSION_URL = "/kaggle/input/innovative-ai-challenge-2024/sample_submission.csv"


import pandas as pd


train_df = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

submission_df = pd.read_csv(SUBMISSION_URL)


train_df.head()


test_df.head()


submission_df.head()


train_df.info()


train_df.columns


len(train_df.columns)


train_df.shape


train_df.describe()


train_df["Year"].unique()


train_df["Year"].value_counts()


train_df["State"].unique()


train_df["State"].value_counts()


train_df["Crop_Type"].unique()


train_df["Crop_Type"].value_counts()


train_df["Soil_Type"].unique()


train_df["Soil_Type"].value_counts()


train_df.isna().sum()


(train_df["Crop_Yield (kg/ha)"] == 0).sum()


(train_df["Crop_Yield (kg/ha)"] <= 100).sum()


train_df


train_df = train_df.drop(train_df[train_df["Crop_Yield (kg/ha)"] < 100].index)
# data after cleaning null values
train_df


train_df.info()


for k, v in train_df.items():
    print(k,"column has datatype of:", v.dtype)


# print columns with datatype as object
for k, v in train_df.items():
    if v.dtype == "object":
        print(k,"column has datatype of:", v.dtype)


# Label Encoder to deal with object datatypes
from sklearn.preprocessing import LabelEncoder


train_cp_df = train_df.copy()
train_cp_df.head()


label_encoder = LabelEncoder()

# Encoding labels in columns
train_cp_df["State"] = label_encoder.fit_transform(train_cp_df["State"])
train_cp_df["Crop_Type"] = label_encoder.fit_transform(train_cp_df["Crop_Type"])
train_cp_df["Soil_Type"] = label_encoder.fit_transform(train_cp_df["Soil_Type"])


train_cp_df.head()


for k, v in train_cp_df.items():
    print(k,"column has datatype of:", v.dtype)


# old dataset info
for k, v in train_df.items():
    print(k,"column has datatype of:", v.dtype)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

train_cp_df.plot(figsize=(16, 6), cmap="YlGn")
plt.title("Before Feature Scaling", fontsize=16)
plt.xlabel("X Scale")
plt.ylabel("Y Scale")

plt.show()


train_cp_df.plot(figsize=(16, 6), cmap="YlGn", kind="hist", alpha=0.45, bins=15)
plt.title("Before Feature Scaling", fontsize=16)
plt.xlabel("X Scale")
plt.ylabel("Y Scale")

plt.show()


train_cp_df.hist(figsize=(16, 8), bins=20, color=["Green"], alpha=0.5)
plt.show()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

# we will apply feature scaling on train_cp_df dataset
train_cp_df = scaler.fit_transform(train_cp_df)


type(train_cp_df)


train_cp_df = pd.DataFrame(data=train_cp_df, columns=train_df.columns)
train_cp_df.head()


# plot after feature scaling
train_cp_df.plot(figsize=(16, 6), cmap="YlGn")
plt.title("After Feature Scaling", fontsize=16)
plt.xlabel("X Scale")
plt.ylabel("Y Scale")

plt.show()


train_cp_df.plot(figsize=(16, 6), cmap="YlGn", kind="hist", alpha=0.45, bins=15)
plt.title("After Feature Scaling", fontsize=16)
plt.xlabel("X Scale")
plt.ylabel("Y Scale")

plt.show()


train_cp_df.hist(figsize=(16, 8), bins=20, color="Green", alpha=0.5)
plt.show()


correlation_matrix = train_cp_df.corr()
correlation_matrix


# here is the correlation matrix including all feature's and our labels
correlation_matrix.plot(kind="bar", figsize=(16, 6), cmap="Greens")
plt.title("Correlation Matrix (including labels)")
plt.show()


# let's visualize the correlation matrix more better using heatmap
plt.figure(figsize=(16, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=sns.color_palette("YlGn"))
plt.title("Correlation Matrix Heatmap (including labels)", fontsize=16)
plt.show();


correlation_matrix["Crop_Yield (kg/ha)"].sort_values(ascending=False)


# in visualization form
correlation_matrix["Crop_Yield (kg/ha)"].sort_values(ascending=False).plot(kind="bar", figsize=(16, 6), color="green", alpha=0.5)
plt.title("Correlation Matrix (including labels)")
plt.xlabel("Features")
plt.ylabel("Feature Scores with respect to Label: Crop_Yield (kg/ha)")
plt.show()


from sklearn.ensemble import RandomForestRegressor


# using train_cp_df for feature importance (already scaled features)
train_cp_df.head()


# split into X and y 
X = train_cp_df.drop("Crop_Yield (kg/ha)", axis=1)
y = train_cp_df["Crop_Yield (kg/ha)"]

# Applying RandomForestRegressor
feature_extract = RandomForestRegressor(n_estimators=1000, random_state=42)
feature_extract.fit(X, y)


feature_extract.feature_importances_


# Displaying feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': feature_extract.feature_importances_}, index=X.columns)
print(feature_importance[1:].sort_values(by='Importance', ascending=False))


# Bar graph visualization
feature_importance[1:].sort_values(by='Importance', ascending=False).plot(kind="bar", color="green", alpha=0.5, figsize=(16, 6))
plt.title("Feature Importance using Random Forest Regression")
plt.show()


# before processing the training data
train_df.plot(cmap="YlGn", figsize=(16, 6))
plt.show()


# before processing the training data
fig, axs = plt.subplots(2, 4, layout="constrained", figsize=(16, 8))

ax = axs[0][0]
ax.plot(train_df.index, train_df["Year"], color="g")
ax.set_title("Year")

ax = axs[0, 1]
ax.plot(train_df["State"], color="lawngreen")
ax.set_title("State")

ax = axs[0, 2]
ax.plot(train_df["Crop_Type"], color="lightgreen")
ax.set_title("Crop_Type")

ax = axs[0, 3]
ax.scatter(train_df.index, train_df["Rainfall"], color="yellowgreen")
ax.set_title("Rainfall")

ax = axs[1, 0]
ax.plot(train_df["Soil_Type"], color="olivedrab")
ax.set_title("Soil_Type")

ax = axs[1, 1]
ax.scatter(train_df.index, train_df["Irrigation_Area"], color="greenyellow")
ax.set_title("Irrigation_Area")

ax = axs[1, 2]
ax.plot(train_df["Crop_Yield (kg/ha)"], color="chartreuse")
ax.set_title("Crop_Yield")

plt.show()


# before processing the training data
train_df.plot(figsize=(16, 6), cmap="YlGn", kind="hist", alpha=0.45, bins=15)
plt.title("Before Feature Scaling", fontsize=16)
plt.xlabel("X Scale")
plt.ylabel("Y Scale")

plt.show()


# before processing the training data
train_df.hist(figsize=(16, 8), bins=20, color="Green", alpha=0.5)
plt.show()


# after processing the training data
train_cp_df.plot(cmap="YlGn", figsize=(16, 6))
plt.show()


# before processing the training data
fig, axs = plt.subplots(2, 4, layout="constrained", figsize=(16, 8))

ax = axs[0][0]
ax.plot(train_cp_df.index, train_cp_df["Year"], color="g")
ax.set_title("Year")

ax = axs[0, 1]
ax.plot(train_cp_df["State"], color="lawngreen")
ax.set_title("State")

ax = axs[0, 2]
ax.plot(train_cp_df["Crop_Type"], color="lightgreen")
ax.set_title("Crop_Type")

ax = axs[0, 3]
ax.scatter(train_cp_df.index, train_cp_df["Rainfall"], color="yellowgreen")
ax.set_title("Rainfall")

ax = axs[1, 0]
ax.plot(train_cp_df["Soil_Type"], color="olivedrab")
ax.set_title("Soil_Type")

ax = axs[1, 1]
ax.scatter(train_cp_df.index, train_cp_df["Irrigation_Area"], color="greenyellow")
ax.set_title("Irrigation_Area")

ax = axs[1, 2]
ax.plot(train_cp_df["Crop_Yield (kg/ha)"], color="chartreuse")
ax.set_title("Crop_Yield")

plt.show()


# after processing the training data
train_cp_df.plot(figsize=(16, 6), cmap="YlGn", kind="hist", alpha=0.45, bins=15)
plt.title("Before Feature Scaling", fontsize=16)
plt.xlabel("X Scale")
plt.ylabel("Y Scale")

plt.show()


# after processing the training data
train_cp_df.hist(figsize=(16, 8), bins=20, color="Green", alpha=0.5)
plt.show()


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score


import numpy as np
np.random.seed(42)

selected_features = ["Year", "Crop_Type", "Irrigation_Area"]
X_train = train_df[selected_features]
y_train = train_df["Crop_Yield (kg/ha)"]

def make_model(model):
    # Custom transformer for label encoding
    class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, column):
            self.column = column
            self.label_encoder = LabelEncoder()
    
        def fit(self, X, y=None):
            self.label_encoder.fit(X[self.column])
            return self
    
        def transform(self, X):
            X = X.copy()
            X[self.column] = self.label_encoder.transform(X[self.column])
            return X
    
        def inverse_transform(self, X):
            X = X.copy()
            X[self.column] = self.label_encoder.inverse_transform(X[self.column])
            return X
    
    pipeline = make_pipeline(
        LabelEncoderTransformer(column='Crop_Type'),  # Custom Label Encoder for Crop_Type
        StandardScaler(),
        model,
    )
    
    pipeline.fit(X_train, y_train)

    

    print("model name:", type(model).__name__)
    
    score = pipeline.score(X_train, y_train)
    print("model score on X_train and y_train dataset:", score)

    y_pred = pipeline.predict(X_train)
    mse_error = mean_squared_error(y_train, y_pred)
    print("mean squared error:", mse_error)

    r2 = r2_score(y_train, y_pred)
    print("r2 score:", r2)

    return {
        "model_name": type(model).__name__, 
        "mean_squared_error": mse_error,
        "r2_score": r2
    }


from sklearn.linear_model import LinearRegression


np.random.seed(42)
linear_model = LinearRegression()
make_model(linear_model)


from sklearn.svm import SVR


np.random.seed(42)
svm_model = SVR()
make_model(svm_model)


from sklearn.neighbors import KNeighborsRegressor


np.random.seed(42)
knn_model = KNeighborsRegressor()
make_model(knn_model)


from sklearn.ensemble import RandomForestRegressor


rf_model = RandomForestRegressor(random_state=42)
make_model(rf_model)


from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor # also using decision tree for regression 


np.random.seed(42)
ensemble_model_voting = VotingRegressor([("lr", LinearRegression()), ('rf', RandomForestRegressor()), ('knn', KNeighborsRegressor()), ('dc', DecisionTreeRegressor())], n_jobs=-1)
make_model(ensemble_model_voting)


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor # using decision tree for regression


bagging_model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=500, n_jobs=-1, random_state=42)
make_model(bagging_model)


from sklearn.ensemble import GradientBoostingRegressor


gbr_model = GradientBoostingRegressor(random_state=42)
make_model(gbr_model)


from sklearn.neural_network import MLPRegressor


mlp_model = MLPRegressor(hidden_layer_sizes=[3, 100, 100, 100, 1], activation="relu", solver="lbfgs", learning_rate_init=0.001, random_state=42, max_iter=1000)

make_model(mlp_model)


#MLP
from sklearn.neural_network import MLPRegressor

X = train_cp_df.drop(["Crop_Yield (kg/ha)", "State", "id", "Rainfall", "Soil_Type"], axis=1) 

model = MLPRegressor(hidden_layer_sizes=[3, 100, 100, 100, 1], activation="relu", solver="lbfgs", learning_rate_init=0.001, random_state=42, max_iter=1000)
model.fit(X, y)


model.score(X, y)


y_pred = model.predict(X)


from sklearn.metrics import mean_squared_error, r2_score

mean_squared_error(y, y_pred)


r2_score(y, y_pred)


import tensorflow as tf
from sklearn.model_selection import train_test_split


tf.random.set_seed(42)

selected_features = ["Year", "Crop_Type", "Irrigation_Area"]
X = train_df[selected_features]
y = train_df["Crop_Yield (kg/ha)"]


label_encoder = LabelEncoder()

# Encoding labels in columns
X["Crop_Type"] = label_encoder.fit_transform(X["Crop_Type"])
# X["Soil_Type"] = label_encoder.fit_transform(X["Soil_Type"])

X = np.array(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)


X_train.shape


tf.random.set_seed(42)

norm_layer = tf.keras.layers.Normalization()
tf_model = tf.keras.Sequential([
    tf.keras.layers.Input(X_train.shape[1:]),
    norm_layer,
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1000, activation="relu"),
    # tf.keras.layers.Dense(1000, activation="relu"),
    # # tf.keras.layers.Dense(5000, activation="relu"),
    # tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(1)
])


tf_model.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

tf_model.compile(loss="mse", optimizer=optimizer, metrics=["R2Score"])


# X_train_array = np.array(X_train)
norm_layer.adapt(X_train)


from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=5,
                                        restore_best_weights=True)

history = tf_model.fit(X_train, y_train, epochs=50000, validation_data=(X_val, y_val), verbose=0, callbacks=[earlystopping])


mse_train, r2_train = tf_model.evaluate(X_train, y_train)


mse_train, r2_train = tf_model.evaluate(X_val, y_val)


X_val[:3]


y_val[:3]


y_pred = tf_model.predict(X_val[:3])
y_pred


mean_squared_error(y_val[:3], y_pred)


# Final Neural Network Model for Regression considering all the training data

tf.random.set_seed(42)

selected_features = ["Year", "Crop_Type", "Irrigation_Area"]
X_train = train_df[selected_features]
y_train = train_df["Crop_Yield (kg/ha)"]


label_encoder = LabelEncoder()

# Encoding labels in columns
X_train["Crop_Type"] = label_encoder.fit_transform(X_train["Crop_Type"])

X_train = np.array(X_train)


tf.random.set_seed(42)

norm_layer = tf.keras.layers.Normalization()
nn_model = tf.keras.Sequential([
    tf.keras.layers.Input(X_train.shape[1:]),
    norm_layer,
    tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(500, activation="relu"),
    tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(1)
])


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

nn_model.compile(loss="mse", optimizer=optimizer, metrics=["R2Score"])

# X_train_array = np.array(X_train)
norm_layer.adapt(X_train)


%%time
from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=300,
                                        restore_best_weights=True)

history = nn_model.fit(X_train, y_train, epochs=50000, validation_data=(X_val, y_val), verbose=0, callbacks=[earlystopping])


nn_model.evaluate(X_train, y_train)


test_df


from sklearn.preprocessing  import LabelEncoder
from sklearn.preprocessing import StandardScaler


np.random.seed(42)

selected_features = ["Year", "Crop_Type", "Irrigation_Area"]

X_test = test_df[selected_features]

label_encoder = LabelEncoder()

# Encoding labels in columns
X_test["Crop_Type"] = label_encoder.fit_transform(X_test["Crop_Type"])

scaler = StandardScaler()

X_test_new = scaler.fit_transform(X_test)

X_test_new


import numpy as np
np.random.seed(42)

selected_features = ["Year", "Crop_Type", "Irrigation_Area"]
X_test = test_df[selected_features]

def make_test_data(X_data):
    # Custom transformer for label encoding
    class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, column):
            self.column = column
            self.label_encoder = LabelEncoder()
    
        def fit(self, X, y=None):
            self.label_encoder.fit(X[self.column])
            return self
    
        def transform(self, X):
            X = X.copy()
            X[self.column] = self.label_encoder.transform(X[self.column])
            return X
    
        def inverse_transform(self, X):
            X = X.copy()
            X[self.column] = self.label_encoder.inverse_transform(X[self.column])
            return X
    
    pipeline = make_pipeline(
        LabelEncoderTransformer(column='Crop_Type'),  # Custom Label Encoder for Crop_Type
        StandardScaler()
    )
    
    return pipeline.fit_transform(X_data)

X_test_new = make_test_data(X_test)


X_test_new


gbr_predict = gbr_model.predict(X_test_new)
gbr_predict


test_data_temp = test_df
test_df_cp = pd.DataFrame(test_data_temp['id'])
test_df_cp["Crop_Yield (kg/ha)"] = gbr_predict
test_df_cp.to_csv("Gradient_Boost_Regression_Prediction.csv", index=None)


test_df_cp


tf.random.set_seed(42)

selected_features = ["Year", "Crop_Type", "Irrigation_Area"]
X_test_tf = test_df[selected_features]


label_encoder = LabelEncoder()

# Encoding labels in columns
X_test_tf["Crop_Type"] = label_encoder.fit_transform(X_test_tf["Crop_Type"])

X_test_tf = np.array(X_test_tf)
X_test_tf


nn_model_prediction = nn_model.predict(X_test_tf)
nn_model_prediction


test_data_temp = test_df
test_df_cp = pd.DataFrame(test_data_temp['id'])
test_df_cp["Crop_Yield (kg/ha)"] = nn_model_prediction
test_df_cp.to_csv("TensorFlow_Neural_Net_Prediction.csv", index=None)


test_df_cp


bagging_model_prediction = bagging_model.predict(X_test_new)
bagging_model_prediction


test_data_temp = test_df
test_df_cp = pd.DataFrame(test_data_temp['id'])
test_df_cp["Crop_Yield (kg/ha)"] = bagging_model_prediction
test_df_cp.to_csv("Bagging_Model_Prediction.csv", index=None)


test_df_cp


rf_model_prediction = rf_model.predict(X_test_new)
rf_model_prediction


test_data_temp = test_df
test_df_cp = pd.DataFrame(test_data_temp['id'])
test_df_cp["Crop_Yield (kg/ha)"] = rf_model_prediction
test_df_cp.to_csv("Random_Forest_Prediction.csv", index=None)


test_df_cp


voting_prediction = ensemble_model_voting.predict(X_test_new)
voting_prediction


test_data_temp = test_df
test_df_cp = pd.DataFrame(test_data_temp['id'])
test_df_cp["Crop_Yield (kg/ha)"] = voting_prediction
test_df_cp.to_csv("Voting_Model_Prediction.csv", index=None)


test_df_cp


mlp_model.predict(X_test_new)




