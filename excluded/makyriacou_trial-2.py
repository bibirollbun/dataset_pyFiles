import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


def columns_encoding(cols_encode, df): 
    encoder = LabelEncoder()
    for col in cols_encode: 
        df[str(col)] = encoder.fit_transform(df[col])    
    return df


def model_performance(y_true, y_pred): 
    return{'R2': round(r2_score(y_true, y_pred),5), 
           'MAE': round(mean_absolute_error(y_true, y_pred),5), 
           'MSE': round(mean_squared_error(y_true, y_pred),5)
          }





train_path = '/kaggle/input/innovative-ai-challenge-2024/train.csv'
test_path  = '/kaggle/input/innovative-ai-challenge-2024/test.csv'
cols_enc = ['Year', 'State', 'Crop_Type', 'Soil_Type']


train_df = pd.read_csv(train_path)
train_df = train_df.drop(columns='id', inplace=False)


train_df = columns_encoding(cols_encode=cols_enc, df=train_df)

# Separate features and labels
features = train_df.drop(columns=["Crop_Yield (kg/ha)"])
labels = train_df["Crop_Yield (kg/ha)"]

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(train_df)

# Combine standardized features with the labels
standardized_df = pd.DataFrame(standardized_features, columns=train_df.columns)
#standardized_df["Crop_Yield (kg/ha)"] = labels.values

# Shuffle the DataFrame
shuffled_df = shuffle(standardized_df, random_state=42)



X_shap = shuffled_df.drop(columns=['Crop_Yield (kg/ha)'])  
y_shap = shuffled_df['Crop_Yield (kg/ha)'] 

X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(X_shap, y_shap, test_size=0.2, random_state=42)
                                                    
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_shap, y_train_shap)

# Explain the predictions using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train_shap)
# Plot feature importance
shap.summary_plot(shap_values, X_train_shap, feature_names=list(shuffled_df.columns ))


shuffled_df = shuffled_df.drop(columns=["State"])


X = shuffled_df.drop(columns=['Crop_Yield (kg/ha)'])  
y = shuffled_df['Crop_Yield (kg/ha)'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
rf_model = ExtraTreesRegressor(n_estimators=80) # RandomForestRegressor(max_depth=5, n_estimators=100)


kf = KFold(n_splits=15, shuffle=True) # 15
fold = 1
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train model on training set
    rf_model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = rf_model.predict(X_test)
    
    # Compute RMSE for the fold
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Fold {fold} RMSE: {rmse:.4f}")
    fold += 1


y_pred_test = rf_model.predict(X_test)
model_perf = model_performance(y_true=y_test, y_pred=y_pred_test)
model_perf


test_df = pd.read_csv(test_path)
test_id = test_df.id


test_df = columns_encoding(cols_encode=cols_enc, df=test_df)
test_df = test_df.drop(columns=['id', 'State'], inplace=False)


# Standardize the features
scaler = StandardScaler()
standardized = scaler.fit_transform(test_df)

# Combine standardized features with the labels
standardized_df = pd.DataFrame(standardized, columns=test_df.columns)


unseen_pred = rf_model.predict(standardized_df)
submit = pd.DataFrame({'id':test_id, 'Target_standarize':unseen_pred})


target_mean = train_df['Crop_Yield (kg/ha)'].mean()
target_std = train_df['Crop_Yield (kg/ha)'].std()


submit['Target'] = submit['Target_standarize'].apply(lambda x: x * target_std + target_mean)
submit['Target'] = submit['Target'].astype(int)
submit = submit.drop(columns=['Target_standarize'], inplace=False)





submit.to_csv("submission.csv", index = False)

