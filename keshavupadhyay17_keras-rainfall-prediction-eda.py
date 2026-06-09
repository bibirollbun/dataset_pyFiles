import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import optuna

import warnings
warnings.filterwarnings('ignore')


df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


custom_colors = ["#4d3f2e","#a25818","#dfa14c","#f3b91f","#f7d383"]
customPalette = sns.set_palette(sns.color_palette(custom_colors))
sns.palplot(sns.color_palette(custom_colors),size=0.8)
plt.tick_params(axis='both', labelsize=0, length = 0)
custom_cmap = colors.LinearSegmentedColormap.from_list("custom", custom_colors)


print("\n There are {:,} rows and {} columns in the Playground Series Train Data.".format(df_train.shape[0], 
                                                                           df_train.shape[1]))
print("\n There are {:,} rows and {} columns in the Playground Series Test Data ".format(df_test.shape[0], 
                                                                           df_test.shape[1]))
print("\n There are {} missing values in the Playground Series Train Data.".format(df_train.isna().sum().sum()))
print("\n There are {} missing values in the Playground Series Test Data.".format(df_test.isna().sum().sum()))


df_subm.shape


df_train.head(5)


df_train.describe().round(2).style.background_gradient(cmap=custom_cmap)


feat_float = df_train.select_dtypes(float).columns
feat_int = df_train.select_dtypes(int).columns
feat_object = df_train.select_dtypes(object).columns
print("Float Features:",feat_float)
print("Integer Features:",feat_int)
print("Object Features:",feat_object)


test_float = df_test.select_dtypes(float).columns
test_int = df_test.select_dtypes(int).columns
test_object = df_test.select_dtypes(object).columns


labels_1=['Float', 'Integer','Object',]
values_1= [len(feat_float), len(feat_int),len(feat_object)]
labels_2=['Float', 'Integer','Object',]
values_2= [len(test_float), len(test_int),len(test_object)]

fig, ax = plt.subplots(1,2, figsize = (8,8))
((ax1, ax2)) = ax

labels = labels_1
values = values_1
ax1.pie(x=values, labels=labels, autopct="%1.1f%%",colors=["#f3b91f","#a25818"],shadow=True, 
        startangle=45, explode=[0.065, 0.07,0.065])
ax1.set_title("Train Features", fontdict={'fontsize': 14},fontweight ='bold',color="#3a0ca3")

labels = labels_2
values = values_2
ax2.pie(x=values, labels=labels, autopct="%1.1f%%",colors=["#f3b91f","#a25818"],shadow=True, 
        startangle=45, explode=[0.06, 0.07,0.065])
ax2.set_title("Test Features", fontdict={'fontsize': 14},fontweight ='bold',color="#3a0ca3")


target_class = pd.DataFrame({'count': df_train.rainfall.value_counts(),
                             'percentage': df_train['rainfall'].value_counts() / df_train.shape[0] * 100
})
target_class


sns.countplot(x= df_train["rainfall"], palette= ["#f3b91f","#a25818"])


print ("Unique values are:\n",df_train.nunique())


fig, axes = plt.subplots(3, 3, figsize = (35,35))
for i, ax in enumerate(axes.reshape(-1)):
    if i < len(feat_float) - 1:
        sns.kdeplot(x = feat_float[i], hue='rainfall', data = df_train, fill = True, ax = ax, palette = ["#d62828","#ffb703"])
        ax.tick_params()
        ax.xaxis.get_label().set_fontsize(20)
        ax.set_ylabel('')
fig.suptitle('Distribution of Float features', color="#3a0ca3",fontsize = 35, x = 0.5, y = 1)
plt.tight_layout()
plt.show()


df_train.drop(columns = 'id', inplace = True)
df_test.drop(columns = 'id', inplace = True)


plt.figure(figsize=(11,11))
corr=df_train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=custom_colors, robust=True, center=0,square=True, linewidths=.6)
plt.title('Correlation')
plt.show()


df_test['winddirection'].fillna(df_test['winddirection'].median(), inplace=True)


# define dataset
X = df_train.drop(['rainfall'], axis=1)
y = df_train['rainfall']


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

X_train = df_train.drop(columns=['day', 'rainfall'])
y_train = df_train['rainfall']
X_test = df_test.drop(columns=['day'])

# Scaling the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Initialize Neural Network
model = Sequential([
    Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(16, activation='relu', kernel_initializer='he_normal'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, 
                    callbacks=[early_stopping], verbose=1)

# Make Predictions
y_pred_keras = model.predict(X_test_scaled).flatten()

# Save Submission
df_subm['rainfall'] = y_pred_keras
df_subm.to_csv('subm_keras.csv', index=False)
df_subm.head()


