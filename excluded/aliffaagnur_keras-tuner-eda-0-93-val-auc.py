import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import shap
import keras_tuner as kt
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score, auc, roc_curve , accuracy_score , confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# SEED

import random

seed_value = 2125

'''NOTE : If u want try different Fine-Tuning, first change that seed_value'''

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# LOAD DATASET

train_data = pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test_data  = pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e3/sample_submission.csv')

train_data.shape , test_data.shape , submission.shape


print('Train data : ')
display(train_data.head(5))

print('Test data : ')
display(test_data.head(5))


train_data.info()


test_data.info()


train_data.isna().sum()


test_data.isna().sum()


# CHECK DUPLICATE
train_data.duplicated().sum() ,  test_data.duplicated().sum()


# CHECK DISTRIBUTION

numeric_feature = train_data.select_dtypes(include='number')
numeric_feature = numeric_feature.drop(labels=['id','rainfall'], axis=1)   # DROP NON INDEPENDENT VARIABLE

# DEFINE FIGURE
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20,10))

for i, feature in enumerate(numeric_feature):
    sns.histplot(data= train_data, x=feature, color='skyblue', ax = axes[i % 3, i//3])  # DISTRIBUTION FOR TRAINING DATA. MARKED IN BLUE
    sns.histplot(data= test_data, x= feature, color='deepskyblue', ax= axes[i % 3, i//3])    # DISTRIBUTION FOR TEST DATA. MARKED IN RED

plt.suptitle('Comparison Distribution of Train data and Test data', fontsize=20)
plt.tight_layout()
plt.show()


# CHECK DISTRIBUTION FOR TARGET FEATURE (rainfall)

sns.countplot(data= train_data, x='rainfall', color='deepskyblue')
plt.title('Rainfall Distribution')


# DISTRIBUTION RAINFALL PERCENTAGES

class_percentages = (train_data['rainfall'].value_counts(normalize=True) * 100).round(2)

print(f'Class Percentages : {class_percentages}')


# QQ-PLOT

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14,7))

for i, feature in enumerate(numeric_feature):
    stats.probplot(x= train_data[feature], dist='norm', rvalue= True, plot= axes[i % 3, i//3])

    ax = axes[i % 3, i//3]
    ax.set_title(feature)

    line = ax.get_lines()[0]  
    line.set_color('deepskyblue')

plt.tight_layout()
plt.show()


# CHECK CORRELATION

corr_matrix = train_data.corr(method='pearson')

plt.figure(figsize=(10,8))

sns.heatmap(data= corr_matrix, vmax=1, vmin=-1, cmap='Blues', annot=True, fmt='.2f', linewidths=1)
plt.title('Check Multicolinearity')
plt.show()


# PAIRPLOT

#cols = train_data.drop(labels='id', axis=1)
#list_cols = cols.columns.tolist()

# CREATE PAIR PLOT
#pairplot = sns.pairplot(cols, height=2, aspect=1.2)

# ADD TITLE FOR EVERY SUBPLOT
#for i in range(len(list_cols)):
#    for j in range(len(list_cols)):
#        pairplot.axes[i, j].set_title(f'{list_cols[i]} vs {list_cols[j]}', fontsize=12)

#plt.tight_layout()
#plt.show()


# DEPENDENT VARIABLE CORRELATION

corr_dependent = train_data.corr(method='spearman')
corr_dependent = corr_dependent[['rainfall']]

sns.heatmap(corr_dependent, vmin=-1, vmax=1, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation')
plt.show()


# CEK DATA DISTRIBUTION FOR RAINFALL FEATURE

cols = train_data.drop(labels=['id', 'rainfall'], axis=1)

plt.figure(figsize=(14,9))
for i , feature in enumerate(cols):
    plt.subplot(3,4,i+1)
    sns.kdeplot(data= train_data, x= feature, hue='rainfall', fill=True)
    plt.title(f'KDE plot for {feature}')

plt.tight_layout()



# VISUALIZE SCATTER PLOT

x = train_data.drop(labels=['id', 'rainfall'], axis=1)
y = train_data['rainfall']

tsne = TSNE(n_components=2)
train_data_reduced = tsne.fit_transform(x, y)

plt.figure(figsize=(10,6))
plt.scatter(x= train_data_reduced[:, 0], y = train_data_reduced[:,1], c= y)
plt.title('Exploratory Rainfall Patterns')


# CHECKING FEATURE IMPORTANCES USING TREE-BASED MODEL

# DEFINE X AND Y
x = train_data.drop(labels=['id', 'rainfall'], axis=1)
y = train_data['rainfall']

forest = RandomForestClassifier(n_estimators=1000)
forest.fit(x, y)


indices = np.argsort(forest.feature_importances_)

# SORTING COLUMNS AND FEATURE IMPORTANCES BASED ON THE SORTED INDEX
sorted_columns = x.columns[indices]
sorted_importances = forest.feature_importances_[indices]

# PLOT
plt.figure(figsize=(10, 10))
plt.barh(y=sorted_columns, width=sorted_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Random Forest')
plt.show()


# CHECK OUTLIER

plt.figure(figsize=(12,8))
for i, feature in enumerate(numeric_feature):
    plt.subplot(3,4,i+1)
    sns.boxplot(data= train_data, y=feature)
    plt.title(feature)
    plt.ylabel('')


# DROP UNNECESSARY COLUMN

id = test_data['id']

test_data = test_data.drop(labels=['id'], axis=1) 


# HANDLING MISSING VALUES

# REPLACE NULL VALUES WITH MEDIAN
test_data['winddirection'] = test_data['winddirection'].fillna(value= test_data['winddirection'].median())  

test_data['winddirection'].isna().sum()


# FEATURE ENGINEERING . CREATE NEW FEATURE

combined_data = pd.concat((train_data, test_data))  # MERGE TRAIN AND TEST DATA

combined_data['temp_range'] = combined_data['maxtemp'] - combined_data['mintemp']

# TO REFLECT THE RELATIVE HUMIDITY OF THE AIR
combined_data['temp_dew_spread'] = combined_data['temparature'] - combined_data['dewpoint']

'''combines temperature, dew point, and humidity to measure the â€œfeltâ€� temperature. High values â€‹â€‹indicate hot, humid conditions, 
   which are often associated with the potential for convective precipitation.''' 
combined_data['heat_index'] = 0.5 * (combined_data['temparature'] + combined_data['dewpoint']) + 0.1 * combined_data['humidity'] - 10

# CREATE A NEW DIRECTION FEATURE SO THAT THE MODEL UNDERSTANDS 'WINNDDIRECTION' PATTERNS
combined_data['wind_east'] = combined_data['windspeed'] * np.cos(combined_data['winddirection'])
combined_data['wind_north']= combined_data['windspeed'] * np.sin(combined_data['winddirection'])

# INTERACTION BETWEEN HUMIDITY AND CLOUD
combined_data['humidity_cloud'] = combined_data['humidity'] * combined_data['cloud']

# THIS FEATURE MEASURES THE RATIO OF HOW MUCH SUN LIGHT IS BLOCKED BY CLOUDS
combined_data['sunshine_cloud_ratio'] = combined_data['sunshine'] / (combined_data['cloud'] + 1e-5)

# THE COMBINATION OF DEWPOINT AND HUMIDITY REFLECTS THE WATER VAPOR CONTENT IN THE AIR
combined_data['moisture_index'] = combined_data['dewpoint'] * combined_data['humidity']

# DESCRIBING THE TEMPERATURE FELT DUE TO THE COMBINATION OF TEMPERATURE AND WIND
combined_data['wind_chill'] = 13.12 + 0.6215 * combined_data['temparature'] - 11.37 * (combined_data['windspeed']**0.16) + 0.3965 * combined_data['temparature'] * (combined_data['windspeed']**0.16)  


def set_high_rain(row):
   if row['humidity'] > 80 and row['cloud'] > 80:
      return 1
   else:
      return 0

combined_data['high_rain_risk'] = combined_data.apply(set_high_rain, axis=1)

# RE-SEPARATE TRAIN AND TEST DATA
TRAIN_SIZE = 2190
new_train_data = combined_data[:TRAIN_SIZE]
new_test_data  = combined_data[TRAIN_SIZE:]


# DISPLAY INFORMATION
print('Train data : ')
display(new_train_data)

new_test_data = new_test_data.drop(labels=['id', 'rainfall'], axis=1)
print('Test data  : ')
display(new_test_data)


# BOXPLOT FOR NEW FEATURES

#new_features = ['temp_range', 'temp_dew_spread', 'heat_index', 'wind_east', 'wind_north', 'humidity_cloud', 'sunshine_cloud_ratio', 'moisture_index', 'wind_chill', 'high_rain_risk']
numeric_features = new_train_data.select_dtypes(include='number').drop(labels=['id', 'rainfall'], axis=1)

plt.figure(figsize=(17,8))
for i, feature in enumerate(numeric_features):
    plt.subplot(3,7,i+1)
    sns.boxplot(data= new_train_data, y= feature)
    plt.title(feature)

plt.tight_layout()


# WE CAN HANDLE OUTLIER VALUES USING WINSORIZING TECHNIQUES

# SELECT FEATURE
#numeric_feature = train_data.select_dtypes(include='number')
#numeric_feature = numeric_feature.drop(columns=['id', 'rainfall'])

''' winsorize is a technique for handling outliers without removing them by replacing the extreme values â€‹â€‹with the maximum/minimum threshold values. '''


# DO WINSORIZING FOR TRAIN DATA
#train_data['pressure'] = winsorize(a = train_data['pressure'], limits=[0.05, 0.05])
#train_data['humidity'] = winsorize(a = train_data['humidity'], limits=[0.15, 0])
#train_data['cloud'] = winsorize(a = train_data['cloud'], limits=[0.1, 0])
#train_data['windspeed'] = winsorize(a = train_data['windspeed'], limits=[0, 0.08])
new_train_data['sunshine_cloud_ratio'] = winsorize(a = new_train_data['sunshine_cloud_ratio'], limits=[0, 0.01])

# DO WINSORIZING FOR TEST DATA
#test_data['pressure'] = winsorize(a = test_data['pressure'], limits=[0.05, 0.05])
#test_data['humidity'] = winsorize(a = test_data['humidity'], limits=[0.15, 0])
#test_data['cloud'] = winsorize(a = test_data['cloud'], limits=[0.1, 0])
#test_data['windspeed'] = winsorize(a = test_data['windspeed'], limits=[0, 0.08])
new_test_data['sunshine_cloud_ratio'] = winsorize(a = new_test_data['sunshine_cloud_ratio'], limits=[0, 0.01])


# VISUALIZE AFTER WINSORIZE
plt.figure(figsize=(3,3))
sns.boxplot(data= new_train_data, y = 'sunshine_cloud_ratio')
plt.title('sunshine_cloud_ratio')


# SPLIT DATASET

x = new_train_data.drop(labels=['id', 'rainfall'], axis=1)
y = new_train_data['rainfall']

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.20, random_state= seed_value, shuffle=True)

print(f'Type : {type(x_train)}')
print(f'x_train shape : {x_train.shape}')
print(f'x_test shape  : {x_val.shape}')
print(f'y_train shape : {y_train.shape}')
print(f'y_test shape  : {y_val.shape}')


# FEATURE SCALING

cols_to_robust = ['pressure', 'dewpoint', 'humidity', 'cloud', 'windspeed', 'temp_range', 'temp_dew_spread', 'humidity_cloud', 'sunshine_cloud_ratio', 'moisture_index']
cols_to_zscore = ['day','maxtemp','temparature','mintemp','sunshine','winddirection', 'heat_index', 'wind_east', 'wind_north', 'wind_chill']

# DEFINE NORMALIZATION TECHNIQUE
robust = RobustScaler()
zscore = StandardScaler()

# FIT AND TRANSFORM TRAIN DATA
x_train[cols_to_robust] = robust.fit_transform(x_train[cols_to_robust])
x_train[cols_to_zscore] = zscore.fit_transform(x_train[cols_to_zscore])

# TRANSFORM VALIDATION DATA
x_val[cols_to_robust] = robust.transform(x_val[cols_to_robust])
x_val[cols_to_zscore] = zscore.transform(x_val[cols_to_zscore])

# TRANSFORM TEST DATA
#new_test_data = copy.deepcopy(test_data)
new_test_data[cols_to_robust] = robust.transform(new_test_data[cols_to_robust])
new_test_data[cols_to_zscore] = zscore.transform(new_test_data[cols_to_zscore])


# DISPLAY INFORMATION
type(x_train) , x_train.shape, x_val.shape, type(test_data), test_data.shape , x_train.columns


# SHOW NORMALIZED DATA
x_val


# BUILD FEEDFORWARD ARCHITECTURE

def build_model(hp):
    model = tf.keras.Sequential()

    # DEFINE INPUT LAYER
    model.add(tf.keras.layers.Input(shape= (x_train.shape[1],)))

    # FIRST HIDDEN LAYERS
    model.add(tf.keras.layers.Dense(units= hp.Int( name = 'hidden_layer_1', min_value= 150, max_value = 300, step= 15 ), 
                                    activation='relu', 
                                    kernel_initializer='HeNormal',
                                    kernel_regularizer= tf.keras.regularizers.L2( hp.Float(name= 'L2_Regularizer', 
                                                                                           min_value = 0.001, 
                                                                                           max_value= 0.015, 
                                                                                           default= 0.005))))
    
    # CREATE 1 - 3 MORE HIDDEN LAYERS (KERAS-TUNER WILL CHOOSE THE BEST COMBINATION)
    for i in range(hp.Int(name= 'num_layers', min_value = 1, max_value = 3)):
        
        model.add(tf.keras.layers.Dense(units= hp.Int( name = 'hidden_layer_' + str(i + 2), 
                                                      min_value = 35, 
                                                      max_value=200, 
                                                      step = 15 ),
                                                      
                                        activation='relu', 
                                        kernel_initializer='HeNormal'))
        # DROPOUT LAYERS
        model.add(tf.keras.layers.Dropout(rate= hp.Float( name = 'dropout_' + str(i+2), min_value=0.35, max_value=0.6, step=0.07)))
        

    # OUTPUT LAYERS
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal'))

    # DEFINE COMPILER
    model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

    return model


# DEFINE TUNER
tuner = kt.RandomSearch(hypermodel= build_model, 
                        objective= kt.Objective('val_auc', direction='max'), 
                        max_trials= 30, 
                        seed= seed_value, 
                        tune_new_entries= True, 
                        allow_new_entries= True, 
                        max_retries_per_trial= 3, 
                        max_consecutive_failed_trials=2,
                        overwrite=True)

# DEFINE CALLBACKS
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=14, verbose=0, restore_best_weights=True) 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', patience=4, verbose=1, min_lr= 0.000001, factor=0.4)

# SEARCH ALL COMBINATION BEST HYPERATAMETERS
tuner.search(x_train, y_train, batch_size= 32, epochs=50, validation_data= (x_val, y_val), callbacks=[reduce_lr, early_stopping])


# BEST HYPERPARAMETERS

# GET BEST HYPERPARAMETERS
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]


'''Best Hyperparameters 2 : Best val_auc So Far: 0.9204849004745483
{'hidden_layer_1': 225,
 'L2_Regularizer': 0.0015722925341434806,
 'num_layers': 2,
 'dropout_1': 0.49,
 'hidden_layer_2': 200,
 'dropout_2': 0.42,
 'hidden_layer_3': 50,
 'dropout_3': 0.49
 }
'''

''' Best Hyperparameters 3 : val_auc : 0.9205234050750732
{'hidden_layer_1': 240,
 'L2_Regularizer': 0.0036815133592687675,
 'num_layers': 2,
 'hidden_layer_2': 65,
 'dropout_2': 0.56,
 'hidden_layer_3': 185,
 'dropout_3': 0.42,
 'hidden_layer_4': 155,
 'dropout_4': 0.35}
'''

''' Best Hyperparameters 4 :  val_auc : 0.924605131149292
{'hidden_layer_1': 180,
 'L2_Regularizer': 0.014568082066387083,
 'num_layers': 2,
 'hidden_layer_2': 140,
 'dropout_2': 0.42,
 'hidden_layer_3': 185,
 'dropout_3': 0.56,
 'hidden_layer_4': 95,
 'dropout_4': 0.42}
 '''
print('Best Hyperparameters : ')
best_trial.hyperparameters.values



# TAKE THE BEST MODEL
best_model = tuner.get_best_models(num_models=1)[0]

# PREDICT WITH BEST MODEL
y_pred_prob = best_model.predict(x_val)

# ROC-AUC SCORE
roc_auc = roc_auc_score(y_val, y_pred_prob)

print(f'ROC AUC Score: {roc_auc}')



# EVALUATION VALIDATION DATA

# PREDICT VALIDATION DATA
y_pred = best_model.predict(x_val)

# COMPUTE ROC_CURVE
fpr, tpr, threshold = roc_curve(y_val, y_pred)

# COMPUTE AUC SCORE
roc_auc_score = auc(x= fpr, y= tpr)

# PLOT ROC CURVE
plt.figure(figsize=(10,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Roc-Auc curve : {roc_auc_score:.2f}')
plt.plot([0,1],[0,1], color='red', lw=2, linestyle='--')    # RANDOM LINE (Random Classification)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()



# TAKE ONLY POSITIVE CLASS PROBABILITY
if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] == 2:
    y_pred_prob = y_pred_prob[:, 1]  


# ENSURE SIZE OF THRESHOLD == TPR AND FPR
print(f"fpr shape: {fpr.shape}, tpr shape: {tpr.shape}, thresholds shape: {threshold.shape}")

# Plot TPR & FPR 
plt.figure(figsize=(8,6))
plt.plot(threshold, tpr, 'g-', label='True Positive Rate')
plt.plot(threshold, fpr, 'r-', label='False Positive Rate')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('TPR and FPR at Different Thresholds')
plt.legend(loc='best')
plt.xlim(0, 1)  

plt.show()



# DISPLAY DISTRIBUTION PROBABILITY PREDICTION

plt.figure(figsize=(15,5))

# DISPLAY PROBABILITY PREDICTED
plt.subplot(1,2,1)
sns.histplot(y_pred, bins=10, color='lightblue', legend=False)
plt.xlabel('Distribution Probability')
plt.ylabel('Frequency')
plt.title('Prediction Probability Distribution')

# DISPLAY RAINFALL CLASS DISTRIBUTION AT VALIDATION DATA
plt.subplot(1,2,2)
sns.countplot(x = y_val, color='skyblue')
plt.xticks(ticks=[0, 1], labels=['No Rain', 'Rain'])
plt.title('Rainfall Class Distribution')
plt.show()


# PCA VISUALIZATION FOR VALIDATION DATA

# PREDICT VALIDATION DATA
y_pred = best_model.predict(x_val)

# CONVERT TO BINARY NUMBER (0 and 1)
y_pred_class = [1 if score >= 0.5 else 0 for score in y_pred]


pca = PCA(n_components=2)
data_reduced = pca.fit_transform(x_val)

# SCATTER PLOT 
plt.figure(figsize=(20,6))

# SCATTER PLOT FOR TRUE LABEL
plt.subplot(1,2,1)
scatter_true = plt.scatter(x= data_reduced[:,0], y = data_reduced[:,1], c = y_val)
plt.title('True Label')
plt.colorbar(scatter_true)
  # LEGEND

handles_true, labels_true = scatter_true.legend_elements()
plt.legend(handles_true, labels_true, title="True Label Classes")

# SCATTER PLOT FOR PREDICTED LABEL
plt.subplot(1,2,2)
scatter_predicted = plt.scatter(x= data_reduced[:,0], y = data_reduced[:,1], c = y_pred_class)  
plt.title('Predicted Label')
plt.colorbar(scatter_predicted)

  # LEGEND
handles_pred, labels_pred = scatter_predicted.legend_elements()
plt.legend(handles_pred, labels_pred, title='Predicted Label Classes')

plt.suptitle('Model Comparison', fontsize=23)


# PLOT DECISION BOUNDARY

# WE NEED USE PCA TO VISUALIZE DECISION BOUNDARY
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(x_val)


# DEFINE X-AXIS AND Y-AXIS
x_min, x_max = data_reduced[:, 0].min() - 1, data_reduced[:, 0].max() + 1
y_min, y_max = data_reduced[:, 1].min() - 1, data_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


# RESTORE FEATURE DIMENSION TO ORIGINAL STATE
grid_points = np.c_[xx.ravel(), yy.ravel()]
X_grid = pca.inverse_transform(grid_points)  # TRANSFORM PCA TO NORMAL DIMENSION

# PREDICT 
Z = best_model.predict(X_grid)
Z = Z.reshape(xx.shape)

# PLOT CONTOUR
plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)

# PLOT SCATTER
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=y_val, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.title('Decision Boundary Neural Networks')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# CHECK FEATURE IMPORTANCES USING SHAP


# DEFINE SHAP
masker = shap.maskers.Independent(x_train)
explainer = shap.Explainer(best_model, masker)

shap_values = explainer.shap_values(x_val)


plt.title("Feature Importance Analysis for Rainfall Prediction", fontsize=14)

# PLOT SHAP
shap.summary_plot(shap_values, x_val, feature_names= x_train.columns.tolist())




# SAVE SUBMISSION


# PREDICT TEST DATA
y_pred = best_model.predict(new_test_data, verbose=0)
y_pred_flatten = y_pred.flatten()


submission = pd.read_csv(r'/kaggle/input/playground-series-s5e3/sample_submission.csv')

# SAVE PREDICTED RAINFALL
submission['rainfall'] = y_pred_flatten

submission.to_csv('submission.csv', index=False)

print("Submission file has been saved as 'submission.csv'.")



submission

