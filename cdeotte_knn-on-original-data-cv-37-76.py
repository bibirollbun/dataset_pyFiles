import pandas as pd, numpy as np
df = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
print("Original Data shape",df.shape)
df.head()


# LABEL ENCODE AND PRESERVE NANS
COLS = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
for c in COLS:
    nans = df[c].isna()
    df[c],_ = pd.factorize(df[c])
    df[c] = df[c].astype("float32")
    df.loc[nans,c] = np.nan
COLS2 = ['Compartments', 'Weight Capacity (kg)', 'Price']
for c in COLS2:
    df[c] = df[c].astype("float32")
print("Data after label encoding...")
df.head()


# SPLIT INTO TRAIN AND VALID FOR EXPERIMENTS BELOW
train = df.iloc[:50_000]
valid = df.iloc[50_000:]
print("Original subset train shape",train.shape)
print("Original subset valid shape",valid.shape)


# MAKE PREDICTIONS
train_mean = train.Price.mean()
true = valid.Price.values
pred = np.ones(len(valid))*train_mean
print("First 10 predictions:", pred[:10] )


# COMPUTE METRIC
m = np.sqrt(np.nanmean( (true-pred)**2 ))
print("Using Constant Prediction - Validation score =",m)


# CONVERT TRAIN ANDS VALID TO NUMPY ARRAYS
X_train = train[COLS+['Compartments', 'Weight Capacity (kg)']].values
X_train[:,-2] /= 2.0
X_train[:,-1] /= 2.0
y_train = train['Price'].values

X_valid = valid[COLS+['Compartments', 'Weight Capacity (kg)']].values
X_valid[:,-2] /= 2.0
X_valid[:,-1] /= 2.0
y_valid = valid['Price'].values

print(X_train.shape, X_valid.shape)


# SUBTRACT EVERY (2500) VALIDATION VECTOR FROM EVERY (50000) TRAIN VECTOR
diff = X_train[:, :, np.newaxis] - X_valid.T[np.newaxis, :, :]
nan_count = np.isnan(diff).sum(axis=1)

# NOW SQUARE DIFFERENCES AND SUM TO GET DISTANCE SQUARED OF EVERY VAL TO EVERY TRAIN
result = np.nansum(diff**2, axis=1) + nan_count * 1.0

# FOR EACH VALID, WE HAVE THE INDEX OF TRAIN OF THE SHORTEST DISTANCE
distances = np.min(result,axis=0)
pred_index = np.argmin(result,axis=0)
distances.shape


# MAKE PREDICTIONS
pred = y_train[pred_index].copy()
pred[distances>=1] = np.nanmean(y_train)
print("First 10 predictions:", pred[:10] )


# COMPUTE METRIC
m = np.sqrt(np.nanmean( (true-pred)**2 ))
print("Using KNN Regressor - Validation score =",m)

