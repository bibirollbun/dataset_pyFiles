
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')






    train_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv')
    test_df = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv')
   
    test_ids = test_df['Unique ID'].copy()



train_df['Penalty'] = train_df['Penalty'].fillna("No Penalty")
test_df['Penalty'] = test_df['Penalty'].fillna("No Penalty")



low_card_cols = ['category_x', 'Track_Condition', 'Tire_Compound_Front', 
                'Tire_Compound_Rear', 'Penalty', 'Session', 'weather', 'track']

train_df = pd.get_dummies(train_df, columns=low_card_cols, drop_first=True)
test_df = pd.get_dummies(test_df, columns=low_card_cols, drop_first=True)

high_card_cols = ['shortname', 'circuit_name', 'rider_name', 'team_name', 'bike_name']
le = LabelEncoder()
for col in high_card_cols:
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))



train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)
viz_df = train_df



plt.figure(figsize=(10, 6))
sns.histplot(train_df['Lap_Time_Seconds'].to_pandas() if hasattr(train_df, 'to_pandas') else train_df['Lap_Time_Seconds'], 
             bins=50, kde=True)
plt.title("Distribution of Lap Times")
plt.xlabel("Lap Time (seconds)")
plt.ylabel("Frequency")
plt.show()


corr_matrix = train_df.to_pandas().corr() if hasattr(train_df, 'to_pandas') else train_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix[['Lap_Time_Seconds']].sort_values(by='Lap_Time_Seconds', ascending=False), 
            annot=True, cmap='coolwarm')
plt.title("Features Correlation with Lap Time")
plt.show()



X = train_df.drop(columns=['Lap_Time_Seconds', 'Unique ID'])
y = train_df['Lap_Time_Seconds']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



corr_matrix = train_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix[['Lap_Time_Seconds']].sort_values(by='Lap_Time_Seconds', ascending=False), 
            annot=True, cmap='coolwarm')
plt.title("Features Correlation with Lap Time")
plt.show()



    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                   random_state=42, n_jobs=-1)
   
    xgb_model = XGBRegressor(n_estimators=500, max_depth=8,
                           learning_rate=0.05, random_state=42, n_jobs=-1)
    
   
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    X_train_pd, y_train_pd, X_val_pd, y_val_pd = X_train, y_train, X_val, y_val



rf_model.fit(X_train_pd, y_train_pd)
xgb_model.fit(X_train_pd, y_train_pd)



def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    return y_pred

print("Random Forest Performance:")
rf_pred = evaluate_model(rf_model, X_val_pd, y_val_pd)

print("\nXGBoost Performance:")
xgb_pred = evaluate_model(xgb_model, X_val_pd, y_val_pd)


print("Random Forest Performance:")
rf_pred = evaluate_model(rf_model, X_val, y_val)

print("\nXGBoost Performance:")
xgb_pred = evaluate_model(xgb_model, X_val, y_val)



rf_feature_imp = pd.DataFrame({
    'Feature': X.columns.to_list() if hasattr(X.columns, 'to_list') else X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_feature_imp.head(20))
plt.title("Random Forest Feature Importance")
plt.show()


xgb_feature_imp = pd.DataFrame({
    'Feature': X.columns.to_list() if hasattr(X.columns, 'to_list') else X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_feature_imp.head(20))
plt.title("XGBoost Feature Importance")
plt.show()




test_features = test_df[X.columns] 



test_features_pd = test_features

final_predictions = xgb_model.predict(test_features_pd)


submission = pd.DataFrame({
    'Unique ID': test_ids.to_pandas() if hasattr(test_ids, 'to_pandas') else test_ids,
    'Lap_Time_Seconds': final_predictions
})

submission.to_csv('sub.csv', index=False)


