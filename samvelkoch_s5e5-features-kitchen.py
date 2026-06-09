import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class EnhancedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Enhanced class for creating informative features
    for calorie expenditure prediction task based on data statistics
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Basic anthropometric features
        # Body Mass Index (BMI)
        X['BMI'] = X['Weight'] / ((X['Height'] / 100) ** 2)
        
        # Body Surface Area (DuBois formula)
        X['BSA'] = 0.007184 * (X['Height'])**0.725 * X['Weight']**0.425
        
        # Ponderal Index (weight / height^3)
        X['Ponderal_Index'] = X['Weight'] / ((X['Height'] / 100) ** 3)
        
        # 2. Body composition (adjusted for actual ranges)
        # Fat-Free Mass (FFM)
        X['FFM'] = np.where(X['Sex'] == 'male',
                           X['Weight'] * 0.80,  # Adjustment for 36-132 kg range
                           X['Weight'] * 0.70)
        
        # Fat Mass estimation
        X['Fat_Mass'] = X['Weight'] - X['FFM']
        
        # Body Fat Percentage
        X['Body_Fat_Percentage'] = (X['Fat_Mass'] / X['Weight']) * 100
        
        # 3. Metabolism-related features
        # Basal Metabolic Rate (BMR) - Mifflin-St Jeor formula
        # Adjusted for age range 20-79 years
        X['BMR'] = np.where(X['Sex'] == 'male',
                           (10 * X['Weight']) + (6.25 * X['Height']) - (5 * X['Age']) + 5,
                           (10 * X['Weight']) + (6.25 * X['Height']) - (5 * X['Age']) - 161)
        
        # 4. Exercise and intensity features
        # Based on Heart_Rate range (67-128) and Duration range (1-30)
        
        # Intensity (Heart Rate / Duration)
        X['Intensity'] = X['Heart_Rate'] / X['Duration']
        
        # Maximum heart rate adjusted for age range 20-79
        X['Max_HR'] = 220 - X['Age']
        
        # Relative intensity normalized by maximum heart rate
        X['Relative_Intensity'] = X['Heart_Rate'] / X['Max_HR']
        
        # Training intensity zones (based on actual HR metrics)
        X['HR_Zone'] = pd.cut(X['Relative_Intensity'], 
                           bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0], 
                           labels=[1, 2, 3, 4, 5])
        
        # Heart work (considering duration range)
        X['Heart_Work'] = X['Heart_Rate'] * X['Duration']
        
        # 5. Thermodynamic features (based on temperature range 37.1-41.5)
        # Body temperature to heart rate ratio
        X['Temp_HR_Ratio'] = X['Body_Temp'] / X['Heart_Rate']
        
        # Temperature deviation from normal
        X['Temp_Deviation'] = X['Body_Temp'] - 37.0
        
        # Thermal index considering duration
        X['Thermal_Index'] = X['Temp_Deviation'] * X['Duration']
        
        # 6. Age-related features (based on range 20-79 years)
        # Age groups adapted to actual distribution
        X['Age_Group'] = pd.cut(X['Age'], 
                              bins=[19, 30, 40, 50, 65, 80], 
                              labels=[0, 1, 2, 3, 4])
        
        # Age metabolism factor (exponential decrease with age)
        X['Age_Metabolism_Factor'] = np.exp(-(X['Age'] - 20) / 50)
        
        # 7. Weight-related features (based on range 36-132 kg)
        # Weight categories
        X['Weight_Category'] = pd.cut(X['Weight'], 
                                   bins=[35, 60, 75, 90, 105, 133], 
                                   labels=[0, 1, 2, 3, 4])
        
        # Weight to height ratio
        X['Weight_Height_Ratio'] = X['Weight'] / X['Height']
        
        # 8. Exercise analysis features
        # Caloric coefficient (normalized by parameter ranges)
        X['Caloric_Coefficient'] = (X['Heart_Rate'] - 67) / (128 - 67) * \
                                  (X['Duration']) / 30 * \
                                  (X['Body_Temp'] - 37) / (41.5 - 37)
        
        # Energy index considering ranges
        X['Energy_Index'] = X['Weight'] * X['Duration'] * X['Relative_Intensity']
        
        # 9. Feature interactions
        # Age and intensity interaction
        X['Age_Intensity_Interaction'] = X['Age'] * X['Intensity'] / 100
        
        # BMI and heart rate interaction
        X['BMI_HR_Interaction'] = X['BMI'] * X['Heart_Rate'] / 100
        
        # Weight and duration interaction
        X['Weight_Duration_Interaction'] = X['Weight'] * X['Duration'] / 100
        
        # 10. Physiological indices
        # Oxygen pulse (estimate of oxygen consumption)
        X['Oxygen_Pulse'] = (X['Heart_Rate'] * X['Duration']) / X['Weight']
        
        # Recovery rate (inverse ratio of HR to duration)
        X['Recovery_Rate'] = X['Duration'] / X['Heart_Rate']
        
        # 11. Metabolic estimates
        # MET estimate (Metabolic Equivalent of Task)
        X['MET_Estimate'] = 3.5 + (X['Heart_Rate'] - 60) * 0.1 + (X['Body_Temp'] - 37) * 2
        
        # Potential calorie burn (based on MET)
        X['Potential_Calorie_Burn'] = X['MET_Estimate'] * X['Weight'] * X['Duration'] / 60
        
        # 12. Normalized features for more stable training
        # Normalized heart rate (range 0-1)
        X['Normalized_HR'] = (X['Heart_Rate'] - 67) / (128 - 67)
        
        # Normalized duration
        X['Normalized_Duration'] = X['Duration'] / 30
        
        # Normalized temperature
        X['Normalized_Temp'] = (X['Body_Temp'] - 37.1) / (41.5 - 37.1)
        
        # 13. Polynomial features
        # Squares of key features
        X['Duration_Squared'] = X['Duration'] ** 2
        X['HR_Squared'] = X['Heart_Rate'] ** 2
        X['Weight_Squared'] = X['Weight'] ** 2
        
        # 14. Logarithmic transformations
        # Log of duration
        X['Log_Duration'] = np.log1p(X['Duration'])
        
        # Log of weight-duration load
        X['Log_Weight_Duration'] = np.log1p(X['Weight'] * X['Duration'])
        
        # 15. Combined indices
        # Training intensity index
        X['Training_Intensity_Index'] = (X['Heart_Rate'] / X['Max_HR']) * \
                                       (X['Body_Temp'] - 37) * X['Duration']
        
        # Composite energy expenditure index
        X['Energy_Expenditure_Index'] = X['Weight'] * X['Duration'] * \
                                      (X['Heart_Rate'] / X['Max_HR']) * \
                                      (X['Body_Temp'] / 37)
                                      
        # 16. Sex as numeric variable
        X['Sex_numeric'] = np.where(X['Sex'] == 'male', 1, 0)
        
        # 17. Additional complex features
        # Recovery index (estimate of return to normal state)
        X['Recovery_Index'] = X['Duration'] * (X['Body_Temp'] - 37) / X['Heart_Rate']
        
        # Workload to age ratio
        X['Workload_Age_Ratio'] = (X['Heart_Rate'] * X['Duration']) / X['Age']
        
        # Fitness index
        X['Fitness_Index'] = X['Max_HR'] / X['Heart_Rate'] * X['Duration']
        
        return X


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print(f"Original train size: {train.shape}")
print(f"Original test size: {test.shape}")

feature_creator = EnhancedFeatureCreator()

# Apply transformation to training set
train_enhanced = feature_creator.transform(train)
print(f"Train size after feature creation: {train_enhanced.shape}")
print(f"Number of added features: {train_enhanced.shape[1] - train.shape[1]}")

test_enhanced = feature_creator.transform(test)
print(f"Test size after feature creation: {test_enhanced.shape}")


pd.set_option('display.max_columns', None)
train_enhanced


train_enhanced.describe().T


train_enhanced.to_parquet('s5e5_train_enhanced.parquet')
test_enhanced.to_parquet('s5e5_test_enhanced.parquet')

