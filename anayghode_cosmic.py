# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def generate_mock_data(n_train=10000, n_test=3000):
    """
    Generates a synthetic dataset based on the competition description.
    In a real scenario, you would use:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    """
    np.random.seed(42)
    
    def create_features(n, start_id):
        data = {
            'id': np.arange(start_id, start_id + n),
            'star_luminosity': np.random.uniform(0.1, 10, n),
            'cosmic_radiation': np.random.uniform(50, 500, n),
            'orbital_velocity': np.random.uniform(5, 52, n),
            'plasma_density': np.random.uniform(0.1, 20, n),
            'solar_wind_pressure': np.random.uniform(0.5, 10, n),
            'hull_temperature': np.random.uniform(200, 400, n),
            'magnetic_field_strength': np.random.uniform(1, 100, n),
            'dark_matter_flux': np.random.uniform(-5, 5, n),
            'nebula_density': np.random.uniform(0.001, 0.1, n),
            'photon_noise_level': np.random.uniform(10, 102, n),
            'engine_thrust': np.random.uniform(100, 1000, n),
            'gravity_well_depth': np.random.uniform(1, 20, n)
        }
        return pd.DataFrame(data)

    train = create_features(n_train, 0)
    test = create_features(n_test, 10000)

    # Creating a synthetic target with some non-linear logic
    # (Stability decreases with radiation/noise, increases with thrust/magnetic field)
    y = (
        (train['engine_thrust'] * 0.05) - 
        (train['cosmic_radiation'] * 0.02) + 
        (train['magnetic_field_strength'] * 0.1) - 
        (train['photon_noise_level'] * 0.05) +
        (train['dark_matter_flux'] ** 2) * 0.5 + 
        np.random.normal(0, 2, n_train) # Add noise
    )
    # Scale to 0-100+ range
    train['cosmic_stability_index'] = (y - y.min()) / (y.max() - y.min()) * 100
    
    return train, test

def main():
    # 1. Load Data
    print("ğŸš€ Initializing Cosmic Stability Model...")
    # Swap generate_mock_data() with pd.read_csv in your local environment
    train, test = generate_mock_data()
    
    X = train.drop(['id', 'cosmic_stability_index'], axis=1)
    y = train['cosmic_stability_index']
    X_test = test.drop(['id'], axis=1)

    # 2. Feature Engineering Pipeline
    # Using PolynomialFeatures to capture interactions like Radiation * Temp
    # Scaling is crucial for many regression models
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42
        ))
    ])

    # 3. Model Evaluation (Cross-Validation)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, X, y, cv=kf, scoring='r2')
    
    print(f"ğŸ“Š CV RÂ² Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # 4. Final Training and Prediction
    model_pipeline.fit(X, y)
    predictions = model_pipeline.predict(X_test)

    # 5. Create Submission File
    submission = pd.DataFrame({
        'id': test['id'],
        'cosmic_stability_index': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("âœ… Submission file 'submission.csv' generated successfully!")
    
    # Feature Importance Visualization (Optional but helpful for hints!)
    feature_names = model_pipeline.named_steps['poly'].get_feature_names_out(X.columns)
    importances = model_pipeline.named_steps['regressor'].feature_importances_
    
    # Show top 10 important features/interactions
    indices = np.argsort(importances)[-10:]
    print("\nğŸ”� Top 10 Stability Drivers:")
    for i in reversed(indices):
        print(f"- {feature_names[i]}: {importances[i]:.4f}")

if __name__ == "__main__":
    main()

