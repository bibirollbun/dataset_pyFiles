import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


print(train.shape)
print(test.shape)


train.info()


train.describe().T


train.isnull().sum()


test.isnull().sum()


import pandas as pd
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

class TemporalFeatureEngineering:
    
    def __init__(self, df):
        self.df = df
        self._prepare_date_column()

    def _prepare_date_column(self):
        """Helper method to prepare the 'day' column as a datetime object."""
        self.df['date'] = pd.to_datetime(self.df['day'], format='%j')

    def extract_month_day_of_week(self):
        """
        Extracts month and day of the week from the 'day' column.
        
        Adds columns 'month' and 'day_of_week' to the dataframe.
        """
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_week'] = self.df['date'].dt.dayofweek

    def extract_season_and_quarter(self):
        """
        Extracts season and quarter information based on the 'month'.
        
        Adds columns 'season' and 'quarter' to the dataframe.
        """
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'

        self.df['season'] = self.df['month'].apply(get_season)
        self.df = pd.get_dummies(self.df, columns=['season'], drop_first=True)
        # self.df['quarter'] = self.df['month'].apply(lambda x: (x - 1) // 3 + 1)

    def cyclic_encoding(self):
        """
        Applies cyclic encoding to 'month' and 'day_of_week'.
        
        Adds columns 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos' to the dataframe.
        """
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)

    def extract_week_of_year(self):
        """
        Extracts the week of the year based on the 'date' column.
        
        Adds column 'week_of_year' to the dataframe.
        """
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week

    def weekend_flag(self):
        """
        Adds a flag for weekends (1 = weekend, 0 = weekday) based on 'day_of_week'.
        
        Adds column 'is_weekend' to the dataframe.
        """
        self.df['is_weekend'] = self.df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


    def create_lag_features(self, lag=1):
        """
        Creates lag features for continuous weather variables like temperature, humidity, etc.
        
        Parameters:
        lag (int): Number of days to shift for lag features. Default is 1.
        
        Adds lag features like 'temperature_lag1', 'humidity_lag1', etc. to the dataframe.
        """
        self.df['temperature_lag1'] = self.df['temparature'].shift(lag)
        self.df['humidity_lag1'] = self.df['humidity'].shift(lag)
        self.df['pressure_lag1'] = self.df['pressure'].shift(lag)
        # self.df.dropna(inplace=True)
        self.df.bfill(inplace=True)
        
    def create_expanding_window_features(self):
        """
        Creates expanding window features like cumulative sum of rainfall and temperature.
        
        Adds columns like 'cumulative_temperature' to the dataframe.
        """
        self.df['cumulative_temperature'] = self.df['temparature'].expanding().mean()


    def apply_fft(self, feature='temparature'):
        """
        Applies Fourier Transform to a given feature to capture frequency components.
        
        Parameters:
        feature (str): The feature (column) to apply FFT on. Default is 'temparature'.
        
        Adds a column with FFT result to the dataframe.
        """
        fft_values = np.abs(fft(self.df[feature]))
        self.df[f'{feature}_fft'] = fft_values

    def visualize_fft(self, feature='temparature'):
        """
        Visualizes the FFT of a given feature.
        
        Parameters:
        feature (str): The feature to visualize the FFT. Default is 'temparature'.
        """
        fft_values = np.abs(fft(self.df[feature]))
        plt.plot(fft_values)
        plt.title(f"FFT of {feature} Data")
        plt.show()

    def get_transformed_df(self):
        """
        Returns the transformed DataFrame with all the engineered features.
        
        Returns:
        pd.DataFrame: The DataFrame with temporal features added.
        """
        self.df = self.df.drop('date', axis=1)
        return self.df




feature_engineer = TemporalFeatureEngineering(train)
feature_engineer.extract_month_day_of_week()
feature_engineer.extract_season_and_quarter()
feature_engineer.cyclic_encoding()
# feature_engineer.extract_week_of_year()
# feature_engineer.weekend_flag()
feature_engineer.create_lag_features()
feature_engineer.create_expanding_window_features()
feature_engineer.apply_fft()
train = feature_engineer.get_transformed_df()


feature_engineer = TemporalFeatureEngineering(test)
feature_engineer.extract_month_day_of_week()
feature_engineer.extract_season_and_quarter()
feature_engineer.cyclic_encoding()
# feature_engineer.extract_week_of_year()
# feature_engineer.weekend_flag()
feature_engineer.create_lag_features()
feature_engineer.create_expanding_window_features()
feature_engineer.apply_fft()
test = feature_engineer.get_transformed_df()


class InteractionFeatureEngineering:
    
    def __init__(self, df):
        self.df = df
    def heat_index(self):
         self.df['heat_index'] = self.df['temparature'] + 0.5555 * (self.df['humidity'] - 50)
    def dew_temp(self):
        self.df['temp_new_diff'] = self.df['temparature'] - self.df['dewpoint']
    def cloud_sunshine_interaction(self):
        self.df['cloud_sunshine'] = self.df['cloud'] * (1 - self.df['sunshine'] / 100)
    def pressure_cloud_interaction(self):
        self.df['pressure_cloud'] = self.df['pressure'] * self.df['cloud']
    def temp_diff(self):
        self.df['temp_diff'] = self.df['maxtemp'] - self.df['mintemp']
    def dew_humidity_interaction(self):
        self.df['dew_humidity'] = self.df['dewpoint'] * self.df['humidity']
    


feature_engineer = InteractionFeatureEngineering(train)
feature_engineer.heat_index()
feature_engineer.dew_temp()
feature_engineer.cloud_sunshine_interaction()
feature_engineer.pressure_cloud_interaction()
feature_engineer.temp_diff()
feature_engineer.dew_humidity_interaction()


feature_engineer = InteractionFeatureEngineering(test)
feature_engineer.heat_index()
feature_engineer.dew_temp()
feature_engineer.cloud_sunshine_interaction()
feature_engineer.pressure_cloud_interaction()
feature_engineer.temp_diff()
feature_engineer.dew_humidity_interaction()


train.columns


test.columns


train.head()


from sklearn.metrics import roc_auc_score, roc_curve, auc
test = test.drop(columns=['id'])
X_train = train.drop(columns=['rainfall', 'id'])

y_train = train['rainfall']

# params = {
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'learning_rate': 0.01,  # Low learning rate for better generalization
#     'max_depth': 8,        
#     'subsample': 0.8,        # Random sampling of data for each tree
#     'colsample_bytree': 0.8, # Subsampling of features
#     'scale_pos_weight': 0.2465753424657534,   # Balance between classes (use with imbalanced classes like in our data)
#     'n_jobs': -1,            # Use all CPU threads
#     'tree_method': 'hist',  # Use GPU for training if available
#     'predictor': 'cpu_predictor'  # Use CPU for prediction
# }

# Set up Cross-Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])  
test_preds = np.zeros(test.shape[0]) 


for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"Training fold {fold + 1}...")
    
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    rfc = RandomForestClassifier(random_state=42)
    model = rfc.fit(X_train_fold, y_train_fold)
    
    val_preds = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_preds
    
    test_preds += model.predict_proba(test)[:, 1]/cv.get_n_splits()

roc_auc_value = roc_auc_score(y_train, oof_preds)  # Calculate ROC-AUC score
print(f"Overall ROC-AUC Score: {roc_auc_value}")




# Visualize the ROC Curve
fpr, tpr, thresholds = roc_curve(y_train, oof_preds)
roc_auc = auc(fpr, tpr)  

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# save our predictions
submission = pd.DataFrame({
    'id': [len(train) + i for i in range(len(test))],  
    'rainfall': test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission saved as 'submission.csv'")


submission


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, random_state=42, test_size=0.2)
rfc = RandomForestClassifier(random_state=42)
param_grid = {
    'max_depth': [3, 5, 10, 15, 20], 
    'min_samples_split': [2, 5, 10], 
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    rfc, param_grid, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1
)

grid_search.fit(X_train2, y_train2)

print("best_params:", grid_search.best_params_)
print("best_score:", grid_search.best_score_)

best_model = grid_search.best_estimator_

test_accuracy = best_model.score(X_test2, y_test2)
print("test_accuracy:", test_accuracy)


# Set up Cross-Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])  
test_preds = np.zeros(test.shape[0]) 


for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"Training fold {fold + 1}...")
    
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    rfc = RandomForestClassifier(criterion='entropy',
                                 max_depth=10, 
                                 min_samples_leaf=5, 
                                 min_samples_split=2,
                                 n_jobs=-1,
                                 random_state=42)
    model = rfc.fit(X_train_fold, y_train_fold)
    
    val_preds = model.predict_proba(X_val_fold)[:, 1]
    oof_preds[val_idx] = val_preds
    
    test_preds += model.predict_proba(test)[:, 1]/cv.get_n_splits()

roc_auc_value = roc_auc_score(y_train, oof_preds)  # Calculate ROC-AUC score
print(f"Overall ROC-AUC Score: {roc_auc_value}")

# Visualize the ROC Curve
fpr, tpr, thresholds = roc_curve(y_train, oof_preds)
roc_auc = auc(fpr, tpr)  

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# save our predictions
submission = pd.DataFrame({
    'id': [len(train) + i for i in range(len(test))],  
    'rainfall': test_preds
})

submission.to_csv("submission1.csv", index=False)
print("Submission saved as 'submission1.csv'")


submission

