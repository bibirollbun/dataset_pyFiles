!pip install catboost


!pip install optuna


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import warnings
import joblib
import time

warnings.filterwarnings('ignore')

class AdvancedCalorieFeatureEngineering:
    """
    Advanced feature engineering for calorie expenditure prediction
    Uses sports physiology and metabolism knowledge
    """
    def __init__(self):
        # Sports physiology constants
        self.MET_VALUES = {
            'rest': 1.0, 'light': 3.0, 'moderate': 5.0, 'vigorous': 8.0, 'extreme': 12.0
        }
        self.MAX_HR_FORMULAS = {
            'tanaka': lambda age: 208 - (0.7 * age),
            'classic': lambda age: 220 - age,
            'gulati': lambda age: 206 - (0.88 * age)
        }
        self.HR_ZONES = {
            'recovery': (0.5, 0.6),
            'aerobic': (0.6, 0.7),
            'threshold': (0.7, 0.8),
            'vo2max': (0.8, 0.9),
            'neuromuscular': (0.9, 1.0)
        }
        self.gender_map = {'Male': 1, 'Female': 0}
        print("âœ… AdvancedCalorieFeatureEngineering initialized")
        print(f"   - {len(self.MET_VALUES)} MET values defined")
        print(f"   - {len(self.MAX_HR_FORMULAS)} HR formulas ready")
        print(f"   - {len(self.HR_ZONES)} HR zones defined")

    def load_and_explore(self, train_path='train.csv', test_path='test.csv'):
        """Load and explore data"""
        print("ğŸ“Š === DATA LOADING AND EXPLORATION ===")
        try:
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            print(f"Train shape: {train.shape}")
            print(f"Test shape: {test.shape}")
            print(f"\nColumns: {list(train.columns)}")

            # Basic statistics for Calories
            if 'Calories' in train.columns:
                print("\nCalories statistics:")
                print(train['Calories'].describe())
                Q1 = train['Calories'].quantile(0.25)
                Q3 = train['Calories'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = train[(train['Calories'] < lower_bound) | (train['Calories'] > upper_bound)]
                print(f"Outlier count: {len(outliers)} ({len(outliers)/len(train)*100:.2f}%)")

            # Missing values
            print("\nMissing values:")
            print("Train:")
            print(train.isnull().sum())
            print("\nTest:")
            print(test.isnull().sum())

            # Data types
            print("\nData types:")
            print(train.dtypes)

            # Gender encoding
            if 'Gender' in train.columns:
                print("\nGender distribution:")
                print(train['Gender'].value_counts())
                train['Gender'] = train['Gender'].map(self.gender_map)
                test['Gender'] = test['Gender'].map(self.gender_map)
                print("Gender encoded")

            return train, test
        except FileNotFoundError as e:
            print(f"â�Œ File not found: {e}")
            print("Please ensure 'train.csv' and 'test.csv' are in the current directory")
            return None, None
        except Exception as e:
            print(f"â�Œ Data loading error: {e}")
            return None, None

    def visualize_data(self, train_df):
        """Visualize data"""
        print("ğŸ“ˆ === DATA VISUALIZATION ===")
        plt.figure(figsize=(15, 12))

        # Calories distribution
        plt.subplot(2, 3, 1)
        plt.hist(train_df['Calories'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Calories Distribution')
        plt.xlabel('Calories')
        plt.ylabel('Frequency')

        # Correlation matrix
        plt.subplot(2, 3, 2)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        corr_matrix = train_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
        plt.title('Correlation Matrix')

        # Heart Rate vs Calories
        if 'Heart_Rate' in train_df.columns:
            plt.subplot(2, 3, 3)
            plt.scatter(train_df['Heart_Rate'], train_df['Calories'], alpha=0.6)
            plt.xlabel('Heart Rate')
            plt.ylabel('Calories')
            plt.title('Heart Rate vs Calories')

        # Duration vs Calories
        if 'Duration' in train_df.columns:
            plt.subplot(2, 3, 4)
            plt.scatter(train_df['Duration'], train_df['Calories'], alpha=0.6)
            plt.xlabel('Duration')
            plt.ylabel('Calories')
            plt.title('Duration vs Calories')

        # Age distribution
        if 'Age' in train_df.columns:
            plt.subplot(2, 3, 5)
            plt.hist(train_df['Age'], bins=30, alpha=0.7, color='lightgreen')
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Frequency')

        # Gender vs Calories
        if 'Gender' in train_df.columns:
            plt.subplot(2, 3, 6)
            gender_calories = train_df.groupby('Gender')['Calories'].mean()
            plt.bar(['Female', 'Male'], gender_calories.values, color=['pink', 'lightblue'])
            plt.title('Average Calories by Gender')
            plt.ylabel('Average Calories')

        plt.tight_layout()
        plt.show()
        print("âœ… Visualization completed")

    def create_physiological_features(self, df):
        """Create physiological features based on sports science"""
        print("ğŸ§¬ === PHYSIOLOGICAL FEATURES ===")
        # BMI and body composition
        if 'Height' in df.columns and 'Weight' in df.columns:
            df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
            df['BMI_underweight'] = (df['BMI'] < 18.5).astype(int)
            df['BMI_normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
            df['BMI_overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
            df['BMI_obese'] = (df['BMI'] >= 30).astype(int)
            df['Ponderal_Index'] = df['Weight'] / ((df['Height']/100) ** 3)
            df['Body_Surface_Area'] = 0.007184 * (df['Weight'] ** 0.425) * (df['Height'] ** 0.725)
            df['Ideal_Weight_Male'] = 52 + 1.9 * ((df['Height'] - 152.4) / 2.54)
            df['Ideal_Weight_Female'] = 53.1 + 1.36 * ((df['Height'] - 152.4) / 2.54)
            if 'Gender' in df.columns:
                df['Weight_Deviation'] = np.where(df['Gender'] == 1,
                                                df['Weight'] - df['Ideal_Weight_Male'],
                                                df['Weight'] - df['Ideal_Weight_Female'])
            else:
                df['Weight_Deviation'] = df['Weight'] - ((df['Ideal_Weight_Male'] + df['Ideal_Weight_Female']) / 2)
            print("âœ… BMI, Ponderal Index, Body Surface Area, Ideal Weight created")

        # Basal Metabolic Rate (BMR) calculations
        if all(col in df.columns for col in ['Age', 'Weight', 'Height']):
            df['BMR_Harris_Male'] = 88.362 + (13.397 * df['Weight']) + (4.799 * df['Height']) - (5.677 * df['Age'])
            df['BMR_Harris_Female'] = 447.593 + (9.247 * df['Weight']) + (3.098 * df['Height']) - (4.330 * df['Age'])
            df['BMR_Mifflin_Male'] = (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) + 5
            df['BMR_Mifflin_Female'] = (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) - 161
            lean_body_mass = df['Weight'] * 0.8
            df['BMR_Katch'] = 370 + (21.6 * lean_body_mass)
            if 'Gender' in df.columns:
                df['BMR_Harris'] = np.where(df['Gender'] == 1, df['BMR_Harris_Male'], df['BMR_Harris_Female'])
                df['BMR_Mifflin'] = np.where(df['Gender'] == 1, df['BMR_Mifflin_Male'], df['BMR_Mifflin_Female'])
            else:
                df['BMR_Harris'] = (df['BMR_Harris_Male'] + df['BMR_Harris_Female']) / 2
                df['BMR_Mifflin'] = (df['BMR_Mifflin_Male'] + df['BMR_Mifflin_Female']) / 2
            df['BMR_Average'] = (df['BMR_Harris'] + df['BMR_Mifflin'] + df['BMR_Katch']) / 3
            df['BMR_per_kg'] = df['BMR_Average'] / df['Weight']
            print("âœ… BMR calculations (Harris-Benedict, Mifflin-St Jeor, Katch-McArdle) created")

        # Age-related physiological changes
        if 'Age' in df.columns:
            df['Age_Child'] = (df['Age'] < 18).astype(int)
            df['Age_YoungAdult'] = ((df['Age'] >= 18) & (df['Age'] < 30)).astype(int)
            df['Age_Adult'] = ((df['Age'] >= 30) & (df['Age'] < 50)).astype(int)
            df['Age_MiddleAge'] = ((df['Age'] >= 50) & (df['Age'] < 65)).astype(int)
            df['Age_Senior'] = (df['Age'] >= 65).astype(int)
            df['Metabolism_Decline'] = np.maximum(0, (df['Age'] - 30) * 0.015)
            df['VO2_Age_Decline'] = np.maximum(20, 60 - (df['Age'] - 20) * 0.5)
            df['Age_Weight_Category'] = pd.cut(df['Age'], bins=[0, 25, 40, 55, 100],
                                             labels=['young', 'adult', 'middle', 'senior'])
            age_weight_dummies = pd.get_dummies(df['Age_Weight_Category'], prefix='age_cat')
            df = pd.concat([df, age_weight_dummies], axis=1)
            df.drop('Age_Weight_Category', axis=1, inplace=True)
            print("âœ… Age groups, metabolism decline, VO2 age effect created")

        # Anthropometric ratios
        if 'Height' in df.columns and 'Weight' in df.columns:
            df['WHR_Estimated'] = df['BMI'] / 25
            df['Body_Type_Ectomorph'] = (df['BMI'] < 20).astype(int)
            df['Body_Type_Mesomorph'] = ((df['BMI'] >= 20) & (df['BMI'] < 25)).astype(int)
            df['Body_Type_Endomorph'] = (df['BMI'] >= 25).astype(int)
            print("âœ… Anthropometric ratios and body type categories created")

        return df

    def create_cardiovascular_features(self, df):
        """Create cardiovascular features"""
        print("â�¤ï¸� === CARDIOVASCULAR FEATURES ===")
        if 'Heart_Rate' in df.columns and 'Age' in df.columns:
            for name, formula in self.MAX_HR_FORMULAS.items():
                df[f'Max_HR_{name}'] = formula(df['Age'])
                df[f'HR_Reserve_{name}'] = df['Heart_Rate'] / df[f'Max_HR_{name}']
                df[f'HR_Reserve_Abs_{name}'] = df['Heart_Rate'] - (df[f'Max_HR_{name}'] * 0.6)
            df['HR_Reserve_Avg'] = (df['HR_Reserve_tanaka'] + df['HR_Reserve_classic'] + df['HR_Reserve_gulati']) / 3
            df['Max_HR_Avg'] = (df['Max_HR_tanaka'] + df['Max_HR_classic'] + df['Max_HR_gulati']) / 3
            for zone_name, (lower, upper) in self.HR_ZONES.items():
                df[f'HR_Zone_{zone_name}'] = ((df['HR_Reserve_Avg'] >= lower) &
                                             (df['HR_Reserve_Avg'] < upper)).astype(int)
            hr_zone_cols = [f'HR_Zone_{zone}' for zone in self.HR_ZONES.keys()]
            df['Dominant_HR_Zone'] = df[hr_zone_cols].idxmax(axis=1)
            zone_dummies = pd.get_dummies(df['Dominant_HR_Zone'], prefix='dominant_zone')
            df = pd.concat([df, zone_dummies], axis=1)
            df.drop('Dominant_HR_Zone', axis=1, inplace=True)
            df['HR_Efficiency'] = df['Heart_Rate'] / df['Age']
            df['HR_Intensity'] = df['Heart_Rate'] / df['Max_HR_Avg']
            df['HR_Stress_Level'] = np.where(df['HR_Reserve_Avg'] > 0.85, 1, 0)
            df['HR_Variability_Est'] = np.abs(df['Heart_Rate'] - df['Heart_Rate'].rolling(window=5, min_periods=1).mean())
            print("âœ… Max HR, HR Reserve, HR Zones created")

        if 'Heart_Rate' in df.columns:
            if 'Age' in df.columns:
                expected_resting_hr = 60 + (df['Age'] * 0.1)
                df['HR_Above_Expected_Rest'] = df['Heart_Rate'] - expected_resting_hr
                df['Fitness_Level_HR'] = np.where(df['Heart_Rate'] < 60, 'high',
                                        np.where(df['Heart_Rate'] < 70, 'good',
                                        np.where(df['Heart_Rate'] < 80, 'average', 'poor')))
                fitness_dummies = pd.get_dummies(df['Fitness_Level_HR'], prefix='fitness_hr')
                df = pd.concat([df, fitness_dummies], axis=1)
                df.drop('Fitness_Level_HR', axis=1, inplace=True)
            else:
                expected_resting_hr = 70
                df['HR_Above_Expected_Rest'] = df['Heart_Rate'] - expected_resting_hr
            df['HR_Bradycardia'] = (df['Heart_Rate'] < 60).astype(int)
            df['HR_Normal'] = ((df['Heart_Rate'] >= 60) & (df['Heart_Rate'] <= 100)).astype(int)
            df['HR_Tachycardia'] = (df['Heart_Rate'] > 100).astype(int)
            df['HR_Severe_Tachy'] = (df['Heart_Rate'] > 120).astype(int)
            df['HR_Percentile'] = df['Heart_Rate'].rank(pct=True)
            df['HR_Low_Intensity'] = (df['HR_Percentile'] < 0.33).astype(int)
            df['HR_Med_Intensity'] = ((df['HR_Percentile'] >= 0.33) & (df['HR_Percentile'] < 0.67)).astype(int)
            df['HR_High_Intensity'] = (df['HR_Percentile'] >= 0.67).astype(int)
            print("âœ… Heart rate categories and intensity levels created")

        if all(col in df.columns for col in ['Age', 'BMI', 'Heart_Rate']):
            risk_score = 0
            risk_score += np.where(df['Age'] > 45, 1, 0)
            risk_score += np.where(df['BMI'] > 30, 1, 0)
            risk_score += np.where(df['Heart_Rate'] > 100, 1, 0)
            df['CV_Risk_Score'] = risk_score
            df['CV_High_Risk'] = (risk_score >= 2).astype(int)
            df['MetS_Risk_BMI'] = (df['BMI'] > 30).astype(int)
            df['MetS_Risk_Age'] = (df['Age'] > 40).astype(int)
            print("âœ… Cardiovascular risk scores created")

        if all(col in df.columns for col in ['Heart_Rate', 'Age', 'HR_Reserve_Avg']):
            df['Functional_Capacity'] = np.where(df['HR_Reserve_Avg'] < 0.5, 4,
                                       np.where(df['HR_Reserve_Avg'] < 0.6, 6,
                                       np.where(df['HR_Reserve_Avg'] < 0.7, 8,
                                       np.where(df['HR_Reserve_Avg'] < 0.8, 10, 12))))
            df['Exercise_Tolerance'] = np.where(df['Functional_Capacity'] < 6, 'poor',
                                      np.where(df['Functional_Capacity'] < 9, 'fair', 'good'))
            tolerance_dummies = pd.get_dummies(df['Exercise_Tolerance'], prefix='ex_tolerance')
            df = pd.concat([df, tolerance_dummies], axis=1)
            df.drop('Exercise_Tolerance', axis=1, inplace=True)
            print("âœ… Exercise capacity and tolerance indicators created")

        return df

    def create_metabolic_features(self, df):
        """Create metabolic features focused on energy expenditure"""
        print("ğŸ”¥ === METABOLIC FEATURES ===")
        if all(col in df.columns for col in ['Heart_Rate', 'Age', 'Duration']):
            if 'HR_Reserve_Avg' in df.columns:
                conditions = [
                    (df['HR_Reserve_Avg'] < 0.5),
                    (df['HR_Reserve_Avg'] < 0.6),
                    (df['HR_Reserve_Avg'] < 0.7),
                    (df['HR_Reserve_Avg'] < 0.8),
                    (df['HR_Reserve_Avg'] < 0.9),
                    (df['HR_Reserve_Avg'] >= 0.9)
                ]
                choices = [2.0, 3.5, 5.0, 7.0, 9.0, 12.0]
                df['MET_Estimated'] = np.select(conditions, choices, default=8.0)
                df['Activity_Sedentary'] = (df['MET_Estimated'] < 1.5).astype(int)
                df['Activity_Light'] = ((df['MET_Estimated'] >= 1.5) & (df['MET_Estimated'] < 3)).astype(int)
                df['Activity_Moderate'] = ((df['MET_Estimated'] >= 3) & (df['MET_Estimated'] < 6)).astype(int)
                df['Activity_Vigorous'] = ((df['MET_Estimated'] >= 6) & (df['MET_Estimated'] < 9)).astype(int)
                df['Activity_Very_Vigorous'] = (df['MET_Estimated'] >= 9).astype(int)
                age_factor = np.where(df['Age'] < 30, 1.0,
                            np.where(df['Age'] < 50, 0.95,
                            np.where(df['Age'] < 65, 0.90, 0.85)))
                df['MET_Age_Adjusted'] = df['MET_Estimated'] * age_factor
                print("âœ… MET estimates and activity levels created")
            df['MET_Duration_Product'] = df['MET_Estimated'] * df['Duration']
            df['MET_Duration_Intensity'] = df['MET_Duration_Product'] / 100
            df['Short_High_Intensity'] = ((df['Duration'] < 30) & (df['MET_Estimated'] > 6)).astype(int)
            df['Long_Moderate_Intensity'] = ((df['Duration'] >= 60) & (df['MET_Estimated'] >= 3) & (df['MET_Estimated'] <= 6)).astype(int)
            df['Long_Low_Intensity'] = ((df['Duration'] >= 60) & (df['MET_Estimated'] < 3)).astype(int)

        if 'Body_Temp' in df.columns:
            df['Temp_Deviation'] = abs(df['Body_Temp'] - 37.0)
            df['Temp_Deviation_Squared'] = df['Temp_Deviation'] ** 2
            df['Temp_Hypothermia'] = (df['Body_Temp'] < 36.0).astype(int)
            df['Temp_Normal'] = ((df['Body_Temp'] >= 36.0) & (df['Body_Temp'] <= 37.5)).astype(int)
            df['Temp_Mild_Fever'] = ((df['Body_Temp'] > 37.5) & (df['Body_Temp'] <= 38.0)).astype(int)
            df['Temp_Moderate_Fever'] = ((df['Body_Temp'] > 38.0) & (df['Body_Temp'] <= 39.0)).astype(int)
            df['Temp_High_Fever'] = (df['Body_Temp'] > 39.0).astype(int)
            if 'Heart_Rate' in df.columns:
                expected_hr_from_temp = 60 + ((df['Body_Temp'] - 37.0) * 10)
                df['HR_Temp_Correlation'] = df['Heart_Rate'] - expected_hr_from_temp
                df['HR_Temp_Sync'] = np.abs(df['HR_Temp_Correlation'])
                df['Thermoregulation_Efficiency'] = 1 / (1 + df['Temp_Deviation'] + df['HR_Temp_Sync']/10)
            print("âœ… Thermoregulation features created")

        if all(col in df.columns for col in ['Weight', 'Duration', 'MET_Estimated']):
            df['Theoretical_Calories'] = df['MET_Estimated'] * df['Weight'] * (df['Duration'] / 60)
            df['Duration_per_kg'] = df['Duration'] / df['Weight']
            df['MET_per_kg'] = df['MET_Estimated'] / df['Weight'] * 100
            if 'Heart_Rate' in df.columns:
                df['HR_per_kg'] = df['Heart_Rate'] / df['Weight']
                df['CV_Efficiency'] = df['MET_Estimated'] / (df['Heart_Rate'] / 100)
            df['Metabolic_Power'] = df['MET_Estimated'] * df['Weight']
            print("âœ… Energy expenditure features created")

        return df

    def create_interaction_features(self, df):
        """Create interaction features"""
        print("ğŸ”— === INTERACTION FEATURES ===")
        if all(col in df.columns for col in ['Age', 'Weight', 'Duration', 'Heart_Rate']):
            df['Age_Weight_Interaction'] = (df['Age'] * df['Weight']) / 1000
            df['HR_Duration_Stress'] = (df['Heart_Rate'] * df['Duration']) / 1000
            df['Age_HR_Efficiency'] = df['Heart_Rate'] / (df['Age'] + 1)
            df['Weight_Duration_Load'] = (df['Weight'] * df['Duration']) / 1000
            df['Age_Weight_Duration'] = (df['Age'] * df['Weight'] * df['Duration']) / 100000
            print("âœ… Basic interactions created")

        if 'BMI' in df.columns:
            if 'Duration' in df.columns:
                df['BMI_Duration'] = (df['BMI'] * df['Duration']) / 100
            if 'MET_Estimated' in df.columns:
                df['BMI_MET'] = df['BMI'] * df['MET_Estimated']
            if 'HR_Reserve_Avg' in df.columns:
                df['BMI_HRReserve'] = df['BMI'] * df['HR_Reserve_Avg']
            if 'Age' in df.columns:
                df['BMI_Age_Factor'] = df['BMI'] * (df['Age'] / 40)
            if 'Heart_Rate' in df.columns:
                df['BMI_HR_Load'] = (df['BMI'] * df['Heart_Rate']) / 1000
            print("âœ… BMI-based interactions created")

        if all(col in df.columns for col in ['BMR_Average', 'MET_Estimated', 'Duration']):
            df['BMR_MET_Ratio'] = df['BMR_Average'] / (df['MET_Estimated'] + 1)
            df['Total_Energy_Expenditure'] = df['BMR_Average'] + (df['MET_Estimated'] * df['Duration'])
            df['Activity_vs_BMR'] = (df['MET_Estimated'] * df['Duration']) / df['BMR_Average']
            print("âœ… Physiological system interactions created")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            high_corr_cols = ['Heart_Rate', 'Duration', 'MET_Estimated', 'BMI']
            available_high_corr = [col for col in high_corr_cols if col in df.columns]
            for col in available_high_corr:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
            print("âœ… Statistical features created")

        return df

    def create_basic_features(self, df):
        """Create basic features (placeholder for additional basic transformations)"""
        print("ğŸ› ï¸� === BASIC FEATURES ===")
        if 'Duration' in df.columns:
            df['Duration_Hours'] = df['Duration'] / 60
        if 'Height' in df.columns:
            df['Height_m'] = df['Height'] / 100
        print("âœ… Basic features created")
        return df

    def create_temporal_features(self, df):
        """Create temporal features (placeholder for time-based features)"""
        print("â�° === TEMPORAL FEATURES ===")
        if 'Duration' in df.columns:
            df['Duration_Squared'] = df['Duration'] ** 2
            df['Duration_Log'] = np.log1p(df['Duration'].clip(lower=0))
        print("âœ… Temporal features created")
        return df

    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("ğŸ�·ï¸� === CATEGORICAL ENCODING ===")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
        print("âœ… Categorical features encoded")
        return df

    def create_statistical_features(self, df):
        """Create statistical features"""
        print("ğŸ“ˆ === STATISTICAL FEATURES ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[f'{col}_rank'] = df[col].rank(pct=True)
        print("âœ… Statistical features created")
        return df

    def handle_outliers(self, df):
        """Handle outliers using IQR method"""
        print("ğŸ›¡ï¸� === OUTLIER HANDLING ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        print("âœ… Outliers handled")
        return df

    def optimize_hyperparameters(self, X_train, y_train):
        """Hyperparameter optimization for regression models using Optuna"""
        print("ğŸ�¯ === HYPERPARAMETER OPTIMIZATION ===")
        optimized_models = {}

        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
            return -scores.mean()

        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
            }
            model = GradientBoostingRegressor(**params, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
            return -scores.mean()

        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 2)
            }
            model = xgb.XGBRegressor(**params, random_state=42, eval_metric='mae')
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
            return -scores.mean()

        def objective_catboost(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1)
            }
            model = CatBoostRegressor(**params, random_state=42, verbose=0)
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
            return -scores.mean()

        # Optimize RandomForest
        print("ğŸŒ² Optimizing RandomForest...")
        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(objective_rf, n_trials=20)
        optimized_models['rf_optimized'] = RandomForestRegressor(**study_rf.best_params, random_state=42, n_jobs=-1)
        print(f"âœ… Best RF params: {study_rf.best_params}, MAE: {study_rf.best_value:.4f}")

        # Optimize GradientBoosting
        print("âš¡ Optimizing GradientBoosting...")
        study_gb = optuna.create_study(direction='minimize')
        study_gb.optimize(objective_gb, n_trials=20)
        optimized_models['gb_optimized'] = GradientBoostingRegressor(**study_gb.best_params, random_state=42)
        print(f"âœ… Best GB params: {study_gb.best_params}, MAE: {study_gb.best_value:.4f}")

        # Optimize XGBoost
        print("ğŸš€ Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(objective_xgb, n_trials=20)
        optimized_models['xgb_optimized'] = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, eval_metric='mae')
        print(f"âœ… Best XGB params: {study_xgb.best_params}, MAE: {study_xgb.best_value:.4f}")

        # Optimize CatBoost
        print("ğŸ�± Optimizing CatBoost...")
        study_cat = optuna.create_study(direction='minimize')
        study_cat.optimize(objective_catboost, n_trials=20)
        optimized_models['cat_optimized'] = CatBoostRegressor(**study_cat.best_params, random_state=42, verbose=0)
        print(f"âœ… Best CatBoost params: {study_cat.best_params}, MAE: {study_cat.best_value:.4f}")

        return optimized_models

    def validate_models(self, models, X_train, y_train, X_val, y_val):
        """Validate regression models"""
        print("ğŸ”� === MODEL VALIDATION ===")
        validation_results = {}
        for name, model in models.items():
            print(f"\nğŸ”„ Validating {name}...")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            overfitting = train_r2 - val_r2
            validation_results[name] = {
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'overfitting': overfitting,
                'is_overfitting': overfitting > 0.05
            }
            print(f"ğŸ“Š {name} Results:")
            print(f"   Train MAE: {train_mae:.4f}")
            print(f"   Val MAE:   {val_mae:.4f}")
            print(f"   Train RMSE: {train_rmse:.4f}")
            print(f"   Val RMSE:   {val_rmse:.4f}")
            print(f"   Train RÂ²:  {train_r2:.4f}")
            print(f"   Val RÂ²:    {val_r2:.4f}")
            print(f"   Overfitting: {overfitting:.4f} {'âš ï¸� HIGH' if overfitting > 0.05 else 'âœ… LOW'}")

            # Residual plot
            plt.figure(figsize=(8, 6))
            plt.scatter(y_val_pred, y_val - y_val_pred, alpha=0.6)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Predicted Calories')
            plt.ylabel('Residuals')
            plt.title(f'{name} - Residual Plot')
            plt.show()

        print("\nğŸ�† === MODEL COMPARISON SUMMARY ===")
        comparison_df = pd.DataFrame(validation_results).T
        comparison_df = comparison_df.round(4)
        print(comparison_df)
        best_model_name = comparison_df['val_r2'].idxmax()
        print(f"\nğŸ¥‡ Best model: {best_model_name} (RÂ²: {comparison_df.loc[best_model_name, 'val_r2']:.4f})")
        return validation_results

    def analyze_feature_importance(self, models, feature_names, X_val, y_val):
        """Analyze feature importance for regression models"""
        print("ğŸ“Š === FEATURE IMPORTANCE ANALYSIS ===")
        feature_importance_df = pd.DataFrame()
        for name, model in models.items():
            print(f"\nğŸ”� Analyzing features for {name}...")
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                temp_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance,
                    'model': name,
                    'method': 'builtin'
                })
                feature_importance_df = pd.concat([feature_importance_df, temp_df])
                print(f"âœ… Built-in importance calculated")
        if not feature_importance_df.empty:
            avg_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
            print("\nğŸ�† TOP 10 IMPORTANT FEATURES:")
            top_features = avg_importance.head(10)
            for i, (feature, score) in enumerate(top_features.items(), 1):
                print(f"{i:2d}. {feature:<30} : {score:.6f}")
            print("\nâ¬‡ï¸� BOTTOM 10 IMPORTANT FEATURES:")
            bottom_features = avg_importance.tail(10)
            for feature, score in bottom_features.items():
                print(f"    {feature:<30} : {score:.6f}")
            plt.figure(figsize=(12, 8))
            top_20_features = avg_importance.head(20)
            top_20_features.plot(kind='barh', color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title('Top 20 Most Important Features', fontsize=16)
            plt.xlabel('Average Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.show()
            return {'avg_importance': avg_importance}
        else:
            print("âš ï¸� No feature importance scores calculated!")
            return None

    def create_final_pipeline(self, train_path='train.csv', test_path='test.csv', output_path='submission.csv'):
        """Complete pipeline for calorie prediction"""
        print("ğŸš€ === FINAL PIPELINE ===")
        start_time = time.time()
        # Load data
        print("ğŸ“Š Loading data...")
        train, test = self.load_and_explore(train_path, test_path)
        if train is None or test is None:
            print("â�Œ Data loading failed!")
            return None
        # Visualize data
        self.visualize_data(train)
        # Feature engineering
        print("\nğŸ”§ === FEATURE ENGINEERING ===")
        train_processed = train.copy()
        test_processed = test.copy()
        for method in [self.create_basic_features, self.create_physiological_features,
                       self.create_cardiovascular_features, self.create_metabolic_features,
                       self.create_temporal_features, self.encode_categorical_features,
                       self.create_statistical_features, self.handle_outliers,
                       self.create_interaction_features]:
            train_processed = method(train_processed)
            test_processed = method(test_processed)
        print(f"âœ… Feature engineering completed!")
        print(f"Train processed shape: {train_processed.shape}")
        print(f"Test processed shape: {test_processed.shape}")
        # Prepare features and target
        target_col = 'Calories'
        exclude_cols = ['id', target_col] if 'id' in train_processed.columns else [target_col]
        feature_cols = [col for col in train_processed.columns
                       if col in test_processed.columns and col not in exclude_cols]
        X_train = train_processed[feature_cols].fillna(0)
        y_train = train_processed[target_col]
        X_test = test_processed[feature_cols].fillna(0)
        print(f"âœ… Features selected: {len(feature_cols)}")
        # Train-validation split
        print("\nâœ‚ï¸� Train-Validation split...")
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        print(f"âœ… Train: {X_tr.shape}, Validation: {X_val.shape}")
        # Feature scaling
        print("\nâš–ï¸� Feature scaling...")
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print("âœ… Feature scaling completed")
        # Hyperparameter optimization
        print("\nğŸ�¯ Hyperparameter optimization...")
        optimized_models = self.optimize_hyperparameters(X_tr_scaled, y_tr)
        # Model training and cross-validation
        print("\nğŸ�‹ï¸� Model training and cross-validation...")
        cv_folds = KFold(n_splits=5, shuffle=True, random_state=42)
        for model_name, model in optimized_models.items():
            print(f"ğŸ”„ {model_name} cross-validation...")
            scores = cross_val_score(model, X_tr_scaled, y_tr, cv=cv_folds, scoring='neg_mean_absolute_error')
            print(f"âœ… {model_name}: MAE = {-scores.mean():.4f} (+/- {scores.std()*2:.4f})")
            model.fit(X_tr_scaled, y_tr)
        # Model validation
        print("\nğŸ”� Model validation...")
        validation_results = self.validate_models(optimized_models, X_tr_scaled, y_tr, X_val_scaled, y_val)
        # Feature importance analysis
        print("\nğŸ“Š Feature importance analysis...")
        importance_results = self.analyze_feature_importance(optimized_models, feature_cols, X_val_scaled, y_val)
        # Ensemble predictions
        print("\nğŸ�­ Creating ensemble...")
        model_weights = {}
        total_r2 = sum([validation_results[name]['val_r2'] for name in optimized_models.keys()])
        for name in optimized_models.keys():
            model_weights[name] = validation_results[name]['val_r2'] / total_r2 if total_r2 > 0 else 1/len(optimized_models)
        print("Model weights:")
        for name, weight in model_weights.items():
            print(f"  {name}: {weight:.4f}")
        ensemble_predictions = np.zeros(len(X_test_scaled))
        for name, model in optimized_models.items():
            pred = model.predict(X_test_scaled)
            ensemble_predictions += model_weights[name] * pred
        # Save submission
        print("\nğŸ’¾ Saving submission...")
        submission = pd.DataFrame({'id': test['id'] if 'id' in test.columns else range(len(test)),
                                 'Calories': ensemble_predictions})
        submission.to_csv(output_path, index=False)
        print(f"âœ… Submission saved to {output_path}")
        print(f"Total runtime: {(time.time() - start_time)/60:.2f} minutes")
        return optimized_models, submission

# Generate synthetic data for testing
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    train = pd.DataFrame({
        'id': range(n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Weight': np.random.uniform(50, 100, n_samples),
        'Height': np.random.uniform(150, 200, n_samples),
        'Heart_Rate': np.random.uniform(60, 180, n_samples),
        'Duration': np.random.uniform(10, 120, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Body_Temp': np.random.uniform(36, 39, n_samples),
        'Calories': np.random.uniform(100, 600, n_samples)  # Simplified target
    })
    test = train.drop('Calories', axis=1).copy()
    test['id'] = range(n_samples, n_samples + len(test))
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    print("âœ… Synthetic data generated and saved as train.csv and test.csv")

# Run the pipeline
if __name__ == "__main__":
    # Generate synthetic data for testing
    generate_synthetic_data()
    # Initialize the class
    fe = AdvancedCalorieFeatureEngineering()
    # Run the pipeline
    models, submission = fe.create_final_pipeline()

