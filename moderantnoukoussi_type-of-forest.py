#Import the Library
import pandas as pd
import numpy as np
import xgboost as xgb
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')

#print(train) 
#print(test)  

#Divide training data into features and objective variables
train_x = train.drop(['Cover_Type'], axis=1)
train_y = train['Cover_Type']

# Since the test data is only feature quantity, it is good as it is
test_x = test.copy()

# Exclude variable Ids
train_x = train_x.drop(['Id'], axis=1)
test_x = test_x.drop(['Id'], axis=1)


train_x['Wilderness_Area_cat'] = train_x[[f'Wilderness_Area{i}' for i in range(1, 5)]].idxmax(axis=1).str.extract('(\d)').astype(int)
# Display the distribution（0〜3）
sns.countplot(data=train_x.join(train_y), x='Wilderness_Area_cat', hue='Cover_Type')
plt.title("Distribution of Cover_Type against Wilderness Area")
plt.show()
# Delete variable
train_x = train_x.drop(['Wilderness_Area_cat'], axis=1)



#Normalize Data
cols_to_normalize = ['Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
                     'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points',
                    ]
train_x[cols_to_normalize] = normalize(train_x[cols_to_normalize])
test_x[cols_to_normalize]  = normalize(test_x[cols_to_normalize])



#Feature creation
# Elevation binning 
train_x['Elevation_bin'] = [math.floor(v/50.0) for v in train_x['Elevation']]
test_x['Elevation_bin']  = [math.floor(v/50.0) for v in test_x['Elevation']]

#Horizontal_Distance_To_Roadways
train_x['Horizontal_Distance_To_Roadways_Log'] = np.log1p(train_x['Horizontal_Distance_To_Roadways'])
test_x['Horizontal_Distance_To_Roadways_Log']  = np.log1p(test_x['Horizontal_Distance_To_Roadways'])

#Soil_Type
train_x['Soil_Type12_32'] = train_x['Soil_Type32'] + train_x['Soil_Type12']
test_x['Soil_Type12_32']  = test_x['Soil_Type32'] + test_x['Soil_Type12']
train_x['Soil_Type23_22_32_33'] = train_x['Soil_Type23'] + train_x['Soil_Type22'] + train_x['Soil_Type32'] + train_x['Soil_Type33']
test_x['Soil_Type23_22_32_33']  = test_x['Soil_Type23'] + test_x['Soil_Type22'] + test_x['Soil_Type32'] + test_x['Soil_Type33']


print("List of the features：")
print(train_x.columns.tolist())


tsize = 0.2
rseed = 71
x, x_val, y, y_val = train_test_split(train_x, train_y, test_size=tsize, random_state=rseed)

# Learning the Model
model_main = RandomForestClassifier(n_estimators=100, random_state=rseed)
model_main.fit(x, y)

# Making predicitons
y_pred_main = model_main.predict(x_val)
accuracy_main = accuracy_score(y_val, y_pred_main)
print(f'Main Model Accuracy：{accuracy_main:.4f}') 


mask_train_bin = y.isin([1, 2])
x_bin = x[mask_train_bin].copy()
y_bin = y[mask_train_bin].copy()

# For verification
mask_val_bin = y_val.isin([1, 2])
x_val_bin = x_val[mask_val_bin].copy()
y_val_bin = y_val[mask_val_bin].copy()

# Model Learning
model_bin = RandomForestClassifier(n_estimators=100, random_state=rseed)
model_bin.fit(x_bin, y_bin)
y_pred_bin = model_bin.predict(x_val_bin)
accuracy_bin = accuracy_score(y_val_bin, y_pred_bin)
print(f"Sub Model Accuracy for Class1&2：{accuracy_bin:.4f}")


suspect_idx = np.where((y_pred_main == 1) | (y_pred_main == 2))[0]
x_val_suspect = x_val.iloc[suspect_idx]
y_pred_bin_suspect = model_bin.predict(x_val_suspect)

# Overwrites only those with high confidence
proba = model_bin.predict_proba(x_val_suspect)
conf_mask = np.max(proba, axis=1) > 0.85  
y_pred_bin_confident = model_bin.predict(x_val_suspect[conf_mask])
# Get index and replace
replace_idx = suspect_idx[conf_mask]
y_pred_main[replace_idx] = y_pred_bin_confident

# Accuracy after overwriting
final_accuracy = accuracy_score(y_val, y_pred_main)
print(f"Final Accuracy After Overwrite：{final_accuracy:.4f}")


test_ids = test['Id']

# Main Model Prediction for Test Data
y_pred_test = model_main.predict(test_x)

# Extract indexes with 1 or 2 main predictions
suspect_idx_test = np.where((y_pred_test == 1) | (y_pred_test == 2))[0]
# Re-predict target data with submodels
x_suspect_test = test_x.iloc[suspect_idx_test]
y_pred_test_bin = model_bin.predict(x_suspect_test)

# Overwrites only those with high confidence
proba = model_bin.predict_proba(x_suspect_test)
conf_mask = np.max(proba, axis=1) > 0.85 
pred_bin_confident = model_bin.predict(x_suspect_test[conf_mask])
# Get index and replace
replace_idx = suspect_idx_test[conf_mask]
y_pred_test[replace_idx] = pred_bin_confident

#print('test_x : ' + int(len(test_x)) + ', y_pred : ' + int(len(y_pred_test)))

submission = pd.DataFrame({
    "Id": test_ids, 
    "Cover_Type": y_pred_test
})
submission.to_csv("submission.csv", index=False)
print('-- Submission is completed! --') 

