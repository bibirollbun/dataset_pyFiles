from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import lightgbm as lgb
import pandas as pd
import optuna


def feature(df: pd.DataFrame) -> pd.DataFrame:
    current_year = datetime.now().year

    # Vehicle age calculation
    df['Vehicle_Age'] = current_year - df['model_year'] + 1

    # Mileage per year calculation
    df['Mileage_per_Year'] = df['milage'] / df['Vehicle_Age']

    # Extract horsepower, engine size, and cylinder count from engine column
    def extract_horsepower(engine: str) -> float | None:
        try:
            return float(engine.split('HP')[0].strip())
        except:
            return None

    def extract_engine_size(engine: str) -> float | None:
        try:
            parts = engine.split(' ')
            for part in parts:
                if 'L' in part:
                    return float(part.replace('L', '').strip())
            return None
        except:
            return None

    def extract_cylinders(engine: str) -> int | None:
        try:
            cylinder_index = engine.lower().find(' cylinder')
            if cylinder_index == -1:
                return None

            number_str = engine[cylinder_index - 2:cylinder_index].strip()
            if len(number_str) == 2 and number_str[0].isdigit() and number_str[1].isdigit():
                return int(number_str)
            if len(number_str) == 2 and not number_str[0].isdigit() and number_str[1].isdigit():
                return int(number_str[1])
            if len(number_str) == 1 and number_str.isdigit():
                return int(number_str)
            return None
        except:
            return None

    df['Horsepower'] = df['engine'].apply(extract_horsepower)
    df['Engine_Size'] = df['engine'].apply(extract_engine_size)
    df['Cylinders'] = df['engine'].apply(extract_cylinders)

    # Power-to-size ratio calculation
    df['Power_to_Size_Ratio'] = df['Horsepower'] / df['Engine_Size']

    # Define luxury and supercar brands
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                      'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                      'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']

    supercar_brands = ['Ferrari', 'Lamborghini', 'McLaren', 'Aston', 'Bugatti']
    
    # Define a list of unpopular brands
    unpopular_brands = ['Pontiac', 'Saturn', 'Saab', 'Plymouth', 'Mercury', 'Karma']

    # Create Is_Luxury_Brand, Is_Supercar and Is_Unpopular columns
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    df['Is_Supercar'] = df['brand'].apply(lambda x: 1 if x in supercar_brands else 0)
    df['Is_Unpopular'] = df['brand'].apply(lambda x: 1 if x in unpopular_brands else 0)

    # Accident impact feature
    df['Accident_Impact'] = df.apply(lambda x: 1 if x['accident'] == 1 and x['clean_title'] == 0 else 0, axis=1)

    # Map brand to country of origin
    brand_country_dict = {
        'MINI': 'United Kingdom',
        'Lincoln': 'United States',
        'Chevrolet': 'United States',
        'Genesis': 'South Korea',
        'Mercedes-Benz': 'Germany',
        'Audi': 'Germany',
        'Ford': 'United States',
        'BMW': 'Germany',
        'Tesla': 'United States',
        'Cadillac': 'United States',
        'Land': 'United Kingdom',
        'GMC': 'United States',
        'Toyota': 'Japan',
        'Hyundai': 'South Korea',
        'Volvo': 'Sweden',
        'Volkswagen': 'Germany',
        'Buick': 'United States',
        'Rivian': 'United States',
        'RAM': 'United States',
        'Hummer': 'United States',
        'Alfa': 'Italy',
        'INFINITI': 'Japan',
        'Jeep': 'United States',
        'Porsche': 'Germany',
        'McLaren': 'United Kingdom',
        'Honda': 'Japan',
        'Lexus': 'Japan',
        'Dodge': 'United States',
        'Nissan': 'Japan',
        'Jaguar': 'United Kingdom',
        'Acura': 'Japan',
        'Kia': 'South Korea',
        'Mitsubishi': 'Japan',
        'Rolls-Royce': 'United Kingdom',
        'Maserati': 'Italy',
        'Pontiac': 'United States',
        'Saturn': 'United States',
        'Bentley': 'United Kingdom',
        'Mazda': 'Japan',
        'Subaru': 'Japan',
        'Ferrari': 'Italy',
        'Aston': 'United Kingdom',
        'Lamborghini': 'Italy',
        'Chrysler': 'United States',
        'Lucid': 'United States',
        'Lotus': 'United Kingdom',
        'Scion': 'Japan',
        'smart': 'Germany',
        'Karma': 'United States',
        'Plymouth': 'United States',
        'Suzuki': 'Japan',
        'FIAT': 'Italy',
        'Saab': 'Sweden',
        'Bugatti': 'France',
        'Mercury': 'United States',
        'Polestar': 'Sweden',
        'Maybach': 'Germany'
    }

    df['Country_Origin'] = df['brand'].map(brand_country_dict)

    # Define a dictionary to map countries to continents
    country_to_continent = {
        'United States': 'North America',
        'Germany': 'Europe',
        'United Kingdom': 'Europe',
        'South Korea': 'Asia',
        'Japan': 'Asia',
        'Sweden': 'Europe',
        'Italy': 'Europe',
        'France': 'Europe',
    }

    # Function to map country to continent
    def map_to_continent(country: str) -> str:
        return country_to_continent.get(country, 'Other')

    # Map country to continent in the DataFrame
    df['Continent_Origin'] = df['Country_Origin'].apply(map_to_continent)

    # Perform one-hot encoding for continents
    continent_dummies = pd.get_dummies(df['Continent_Origin'], prefix='Continent')
    df = pd.concat([df, continent_dummies], axis=1)

    # Transmission one-hot encoding for 'Fixed', 'Hybrid', 'Automatic', 'Dual-Clutch', 'Manual', and 'CVT'
    def map_transmission(transmission: str) -> str:
        transmission_lower = transmission.lower()
        if any(x in transmission_lower for x in ['fixed gear']):
            return 'Fixed'
        if any(x in transmission_lower for x in ['auto-shift', 'override', 'variable', 'at/mt', 'steptronic']):
            return 'Hybrid'
        if any(x in transmission_lower for x in ['dual', 'dct', 'cmdshft']):
            return 'Dual'
        if any(x in transmission_lower for x in ['cvt']):
            return 'CVT'
        if any(x in transmission_lower for x in ['a/t', 'automatic', '8-speed at']) or len(transmission_lower) <= 1:
            return 'Automatic'
        return 'Manual'

    df['Transmission_Type'] = df['transmission'].apply(map_transmission)

    # One-hot encode transmission types
    transmission_dummies = pd.get_dummies(df['Transmission_Type'], prefix='Transmission')
    df = pd.concat([df, transmission_dummies], axis=1)

    # Extract speed number from transmission (e.g., "6-Speed")
    def extract_speed(transmission: str) -> int:
        import re
        match = re.search(r'(\d+)-Speed', transmission)
        if match:
            return int(match.group(1))
        return None

    df['Speed_Number'] = df['transmission'].apply(extract_speed)

    return df


# Train a linear regression model to predict horsepower using cylinders, vehicle age, and engine size
def train_hp_regression_with_size(df: pd.DataFrame) -> LinearRegression:
    # Drop rows with missing values in the relevant columns for this model
    df_hp_notna = df[['Cylinders', 'Engine_Size', 'Horsepower']].dropna()

    # Separate the features (X) and target (y)
    X_hp_with_size = df_hp_notna[['Cylinders', 'Engine_Size']]
    y_hp_with_size = df_hp_notna['Horsepower']

    # Train the model
    reg_hp_with_size = LinearRegression()
    reg_hp_with_size.fit(X_hp_with_size, y_hp_with_size)

    # Print the equation of the linear regression model
    print(f"HP Model with Engine_Size: Horsepower = ({reg_hp_with_size.coef_[0]} * Cylinders) + ({reg_hp_with_size.coef_[1]} * Engine_Size) + {reg_hp_with_size.intercept_}")

    return reg_hp_with_size

# Train a linear regression model to predict horsepower using only cylinders and vehicle age
def train_hp_regression_without_size(df: pd.DataFrame) -> LinearRegression:
    # Drop rows with missing values in the relevant columns for this model
    df_hp_notna = df[['Cylinders', 'Horsepower']].dropna()

    # Separate the features (X) and target (y)
    X_hp_without_size = df_hp_notna[['Cylinders']]
    y_hp_without_size = df_hp_notna['Horsepower']

    # Train the model
    reg_hp_without_size = LinearRegression()
    reg_hp_without_size.fit(X_hp_without_size, y_hp_without_size)

    # Print the equation of the linear regression model
    print(f"HP Model without Engine_Size: Horsepower = ({reg_hp_without_size.coef_[0]} * Cylinders) + {reg_hp_without_size.intercept_}")

    return reg_hp_without_size

# Train a linear regression model to predict engine size using horsepower, cylinders, and vehicle age
def train_size_regression(df: pd.DataFrame) -> LinearRegression:
    # Drop rows with missing values in the relevant columns for this model
    df_size_notna = df[['Horsepower', 'Cylinders', 'Engine_Size']].dropna()

    # Separate the features (X) and target (y)
    X_size = df_size_notna[['Horsepower', 'Cylinders']]
    y_size = df_size_notna['Engine_Size']

    # Train the model
    reg_size = LinearRegression()
    reg_size.fit(X_size, y_size)

    # Print the equation of the linear regression model
    print(f"Engine Size Model: Engine_Size = ({reg_size.coef_[0]} * Horsepower) + ({reg_size.coef_[1]} * Cylinders) + {reg_size.intercept_}")

    return reg_size


def impute_missing_values(df: pd.DataFrame, avg_speed_number: float, avg_cylinders: float, hp_regressor_with_size: LinearRegression, hp_regressor_without_size: LinearRegression, size_regressor: LinearRegression) -> pd.DataFrame:
    # Impute missing cylinders with the average number of cylinders
    df['Cylinders'] = df['Cylinders'].fillna(avg_cylinders)

    # Impute missing horsepower
    def impute_horsepower(row: pd.Series) -> float:
        if pd.isna(row['Horsepower']):
            if pd.notna(row['Engine_Size']):
                # Use the regressor that includes engine size
                X_test = pd.DataFrame({'Cylinders': [row['Cylinders']], 'Engine_Size': [row['Engine_Size']]})
                return hp_regressor_with_size.predict(X_test)[0]
            else:
                # Use the regressor that only uses cylinders and vehicle age
                X_test = pd.DataFrame({'Cylinders': [row['Cylinders']]})
                return hp_regressor_without_size.predict(X_test)[0]
        return row['Horsepower']

    # Impute missing engine size
    def impute_engine_size(row: pd.Series) -> float:
        if pd.isna(row['Engine_Size']):
            X_test = pd.DataFrame({'Horsepower': [row['Horsepower']], 'Cylinders': [row['Cylinders']]})
            return size_regressor.predict(X_test)[0]
        return row['Engine_Size']

    # Impute missing speed number using the average speed number
    df['Speed_Number'] = df['Speed_Number'].fillna(avg_speed_number)

    # Apply imputation
    df['Horsepower'] = df.apply(impute_horsepower, axis=1)
    df['Engine_Size'] = df.apply(impute_engine_size, axis=1)

    # Recalculate power-to-weight ratio
    df['Power_to_Size_Ratio'] = df['Horsepower'] / df['Engine_Size']

    return df


def objective(trial: optuna.Trial) -> float:
    # Define the parameter space for tuning
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_loguniform('learning_rate', 2e-2, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 7, 11),
        'num_leaves': trial.suggest_int('num_leaves', 220, 320),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 140, 240),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 2e-1),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1e-3),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.15, 0.5),
        'subsample': trial.suggest_uniform('subsample', 0.82, 1.0),
        'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', 1e-3, 1.0),
        'force_row_wise': True,
    }

    # Perform 5-fold cross-validation with LightGBM
    cv_results = lgb.cv(
        params,
        lgb.Dataset(X_train, label=y_train),
        nfold=25,
        stratified=False,
        metrics='rmse',
        seed=42
    )
    
    # Return the best RMSE for this trial
    return cv_results['valid rmse-mean'][-1]


# File paths
train_file_path = '/kaggle/input/playground-series-s4e9/train.csv'
test_file_path = '/kaggle/input/playground-series-s4e9/test.csv'
extended_data_path = '/kaggle/input/extended-dataset-for-used-car-prices-regression/extended_data.csv'

# Reading the datasets
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Load the extended data
extended_df = pd.read_csv(extended_data_path)

# Merge the train and test data with the extended data
train_df = train_df.merge(extended_df[['model_year', 'brand', 'model', 'type', 'miles_per_gallon', 'msrp', 'premium_version', 'collection_car']], 
                          how='left', 
                          on=['model_year', 'brand', 'model'])

test_df = test_df.merge(extended_df[['model_year', 'brand', 'model', 'type', 'miles_per_gallon', 'msrp', 'premium_version', 'collection_car']], 
                        how='left', 
                        on=['model_year', 'brand', 'model'])

# Apply feature engineering to training data
train_df = feature(train_df)

# Drop 'engine', 'transmission', and 'model_year' columns after feature engineering
train_df.drop(columns=['engine', 'transmission', 'model_year'], inplace=True)

# Calculate the average speed number
avg_speed_number = train_df['Speed_Number'].mean()

# Train linear regression models to predict horsepower and engine size
hp_regressor_with_size = train_hp_regression_with_size(train_df)
hp_regressor_without_size = train_hp_regression_without_size(train_df)
size_regressor = train_size_regression(train_df)

# Calculate the average number of cylinders
avg_cylinders = train_df['Cylinders'].mean()

# Apply imputation to the training data using the new models
train_df = impute_missing_values(train_df, avg_speed_number, avg_cylinders, hp_regressor_with_size, hp_regressor_without_size, size_regressor)

# Apply feature engineering to the test data
test_df = feature(test_df)

# Drop 'engine', 'transmission', and 'model_year' columns after feature engineering
test_df.drop(columns=['engine', 'transmission', 'model_year'], inplace=True)

# Apply imputation to the test data using the same averages
test_df = impute_missing_values(test_df, avg_speed_number, avg_cylinders, hp_regressor_with_size, hp_regressor_without_size, size_regressor)

# Encoding categorical variables
categorical_cols = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'accident', 'clean_title', 'Country_Origin', 'Continent_Origin', 'Transmission_Type', 'type']
label_encoders = {col: LabelEncoder() for col in categorical_cols}

# Apply Label Encoding to the categorical columns in the training data
for col in categorical_cols:
    train_df[col] = label_encoders[col].fit_transform(train_df[col].astype(str))

# Handle unseen labels in the test data
for col in categorical_cols:
    # Map the test data labels to the training data labels
    test_df[col] = test_df[col].map(lambda s: label_encoders[col].classes_.tolist().index(s) if s in label_encoders[col].classes_ else -1)

# Scaling the numerical features
numerical_cols = ['milage', 'Vehicle_Age', 'Mileage_per_Year', 'Horsepower', 'Engine_Size', 'Power_to_Size_Ratio', 'miles_per_gallon', 'msrp']
scaler = StandardScaler()

train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# Defining the target and features
X_train = train_df.drop(columns=['id', 'price'])
y_train = train_df['price'].values


# Create the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=256)

# Print the best parameters found by Optuna
print(f"Best params: {study.best_params}")
print(f"Best RMSE: {study.best_value}")

# Use the best parameters to train the final model with the entire dataset
best_params = study.best_params
best_params['metric'] = 'rmse'
best_model = lgb.train(best_params, lgb.Dataset(X_train, label=y_train))

# Make predictions on the test set
predictions = best_model.predict(test_df.drop(columns=['id']).values, num_iteration=best_model.best_iteration)

# Save the predictions
test_df['price'] = predictions
test_df[['id', 'price']].to_csv('/kaggle/working/submission.csv', index=False)
print("Predictions saved as submission.csv")

