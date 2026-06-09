# importing
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
te=pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
print(df.shape,te.shape)


df


te


# importing
import seaborn as sns
import matplotlib.pyplot as plt


# We have no null values
# Let us drop duplicates and then drop the Id column
df.drop_duplicates(inplace=True)
df.info()


df.fillna(df.mean(), inplace=True)
te.fillna(te.mean(), inplace=True)


fig,ax=plt.subplots(4,3,figsize=(20,10))
ax=ax.flatten()
i=0
for cols in df.columns:
    if i<len(ax):
        sns.boxplot(data=df,x=df[cols],ax=ax[i])
        plt.title(cols)
        i+=1
plt.tight_layout()
plt.show()


# We do have some outliers let us see how this will affect the model
# No duplicates
# No null values
# Let us visualize these features


# Let us use kde plots to check the distribution of the features as well as our target variable
fig,ax=plt.subplots(4,3,figsize=(20,20))
ax=ax.flatten()
i=0
for col in df.columns:
    if i <len(ax):
        sns.kdeplot(data=df,x=df[col],ax=ax[i])
        i+=1
plt.tight_layout()
plt.show()


# let us compare train and test data and look at that distributions

fig, ax = plt.subplots(4, 3, figsize=(20, 20))
ax = ax.flatten()
i = 0
for col in df.columns:
    if col != 'rainfall':
        sns.kdeplot(data=df, x=col, ax=ax[i], label="Train", fill=True)
        sns.kdeplot(data=te, x=col, ax=ax[i], label="Test", fill=True)
        ax[i].set_title(col)
        ax[i].legend()
        i += 1
plt.tight_layout()

for j in range(i, len(ax)):
    ax[j].axis("off")

plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(4, 3, figsize=(20, 20))
ax = ax.flatten()

for i, col in enumerate(df.columns):
    if col not in ['id', 'rainfall']:
        sns.violinplot(data=df, y=col, x='rainfall', ax=ax[i], inner="quartile", palette="coolwarm")
        ax[i].set_title(f'Violin Plot of {col} (Hue: Rainfall)', fontsize=14)

ax[0].axis('off')
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import itertools

features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
pairs = list(itertools.combinations(features, 2))  
n_cols = 3
n_rows = -(-len(pairs) // n_cols)  

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()
for i, (x, y) in enumerate(pairs):
    sns.scatterplot(data=df, x=x, y=y, hue='rainfall', palette='coolwarm', ax=axes[i])
    axes[i].set_title(f'{x} vs. {y} (Hue: Rainfall)', fontsize=14)
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



# Let us visualize correlation values using a heatmap
plt.figure(figsize=(20,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


# Calculating Mutual Information
# importing
from sklearn.feature_selection import mutual_info_regression

X = df.drop(columns=['id', 'rainfall'])
y = df['rainfall']
mi=mutual_info_regression(X,y)
mi_df=pd.DataFrame({"Cols":X.columns,'MI':mi})
mi_df.sort_values(ascending=False,inplace=True,by='MI')

plt.figure(figsize=(20,8))
sns.barplot(data=mi_df,x='MI',y='Cols')
plt.show()


# some new features
df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
df['humidity_sunshine_interaction'] = df['humidity'] * df['sunshine']
df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1e-5)
df['relative_dryness'] = 100 - df['humidity']
df['sunshine_percentage'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
df['weather_index'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])

te['humidity_cloud_interaction'] = te['humidity'] * te['cloud']
te['humidity_sunshine_interaction'] = te['humidity'] * te['sunshine']
te['cloud_sunshine_ratio'] = te['cloud'] / (te['sunshine'] + 1e-5)
te['relative_dryness'] = 100 - te['humidity']
te['sunshine_percentage'] = te['sunshine'] / (te['sunshine'] + te['cloud'] + 1e-5)
te['weather_index'] = (0.4 * te['humidity']) + (0.3 * te['cloud']) - (0.3 * te['sunshine'])


plt.figure(figsize=(20,8))
sns.heatmap(df.corr(),annot=True)
plt.show()

X = df.drop(columns=['id', 'rainfall'])
y = df['rainfall']
mi=mutual_info_regression(X,y)
mi_df=pd.DataFrame({"Cols":X.columns,'MI':mi})
mi_df.sort_values(ascending=False,inplace=True,by='MI')

plt.figure(figsize=(20,8))
sns.barplot(data=mi_df,x='MI',y='Cols')
plt.show()


X = df.drop(columns=['id', 'rainfall'])
y = df['rainfall']
X_test = te.drop(columns=['id'])


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Reshape Input for CNN (adding a channel dimension)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') 
])


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


test_preds = model.predict(X_test_scaled).flatten()

if np.isnan(test_preds).sum() > 0:
    print(f"Found {np.isnan(test_preds).sum()} NaN values in predictions. Fixing them...")
    test_preds = np.nan_to_num(test_preds)  


submission = pd.DataFrame({"id": te['id'], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)

print("Successfully Saved Raaaaaaahhhhhh!!!")

