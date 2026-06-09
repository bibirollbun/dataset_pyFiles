import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


class RoadAccidentPredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def create_enhanced_features(self, df):
        """Create engineered features for better prediction"""
        df_enhanced = df.copy()
        
        # Basic feature engineering
        df_enhanced['curvature_speed_interaction'] = df_enhanced['curvature'] * df_enhanced['speed_limit']
        df_enhanced['complexity_score'] = df_enhanced['num_lanes'] * df_enhanced['curvature']
        df_enhanced['visibility_score'] = df_enhanced['lighting'].map({'daylight': 0, 'dim': 1, 'night': 2}) + \
                                         df_enhanced['weather'].map({'clear': 0, 'rainy': 1, 'foggy': 2})
        
        # Risk interaction features
        df_enhanced['night_foggy_risk'] = ((df_enhanced['lighting'] == 'night') & 
                                          (df_enhanced['weather'] == 'foggy')).astype(int)
        df_enhanced['rainy_high_curvature'] = ((df_enhanced['weather'] == 'rainy') & 
                                              (df_enhanced['curvature'] > 0.5)).astype(int)
        df_enhanced['high_speed_night'] = ((df_enhanced['speed_limit'] > 50) & 
                                          (df_enhanced['lighting'] == 'night')).astype(int)
        
        # Road type risk encoding
        road_type_risk = {'urban': 1, 'rural': 2, 'highway': 3}
        df_enhanced['road_type_risk'] = df_enhanced['road_type'].map(road_type_risk)
        
        # Time of day risk
        time_risk = {'morning': 1, 'afternoon': 2, 'evening': 3}
        df_enhanced['time_risk'] = df_enhanced['time_of_day'].map(time_risk)
        
        return df_enhanced
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data with encoding and feature preparation"""
        df_processed = df.copy()
        
        # Convert boolean columns
        bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
        for col in bool_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        # Label encode categorical variables
        categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if is_training:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le is not None:
                        # Handle unseen labels
                        mask = ~df_processed[col].isin(le.classes_)
                        if mask.any():
                            df_processed.loc[mask, col] = le.classes_[0]
                        df_processed[col] = le.transform(df_processed[col])
        
        return df_processed
    
    def prepare_features(self, df):
        """Prepare final feature set for modeling"""
        base_features = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 
                        'weather', 'road_signs_present', 'public_road', 'time_of_day', 
                        'holiday', 'school_season', 'num_reported_accidents']
        
        enhanced_features = ['curvature_speed_interaction', 'complexity_score', 'visibility_score',
                           'night_foggy_risk', 'rainy_high_curvature', 'high_speed_night',
                           'road_type_risk', 'time_risk']
        
        all_features = base_features + enhanced_features
        
        # Ensure all features exist
        for feature in all_features:
            if feature not in df.columns:
                df[feature] = 0
        
        return df[all_features], all_features
    
    def train(self, train_df, test_df=None, use_cross_validation=True):
        """Train the model with enhanced features and optional cross-validation"""
        print("Step 1: Creating enhanced features...")
        train_enhanced = self.create_enhanced_features(train_df)
        
        print("Step 2: Preprocessing data...")
        train_processed = self.preprocess_data(train_enhanced, is_training=True)
        
        print("Step 3: Preparing features...")
        X, self.feature_columns = self.prepare_features(train_processed)
        y = train_processed['accident_risk']
        
        # Initialize optimized models
        self.models = {
            'hgb_main': HistGradientBoostingRegressor(
                max_iter=500,
                max_depth=8,
                learning_rate=0.08,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42
            ),
            'hgb_backup': HistGradientBoostingRegressor(
                max_iter=400,
                max_depth=10,
                learning_rate=0.1,
                min_samples_leaf=15,
                l2_regularization=0.8,
                random_state=123
            )
        }
        
        if use_cross_validation:
            print("Step 4: Training with Cross-Validation...")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for model_name, model in self.models.items():
                print(f"  CV for {model_name}...")
                model_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    model_scores.append(rmse)
                
                avg_rmse = np.mean(model_scores)
                cv_scores.append(avg_rmse)
                print(f"  {model_name} - Average CV RMSE: {avg_rmse:.5f}")
            
            # Select best model based on CV
            best_model_idx = np.argmin(cv_scores)
            best_model_name = list(self.models.keys())[best_model_idx]
            self.best_model = self.models[best_model_name]
            print(f"\nBest model: {best_model_name} with RMSE: {cv_scores[best_model_idx]:.5f}")
        
        else:
            print("Step 4: Training without Cross-Validation...")
            self.best_model = self.models['hgb_main']
        
        # Final training on full data
        print("Step 5: Final training on full dataset...")
        self.best_model.fit(X, y)
        self.is_trained = True
        
        # Quick validation score
        y_pred_full = self.best_model.predict(X)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred_full))
        print(f"Final Training RMSE: {final_rmse:.5f}")
        
        return self
    
    def predict(self, test_df):
        """Generate predictions for test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        print("Generating predictions...")
        
        # Apply same feature engineering and preprocessing
        test_enhanced = self.create_enhanced_features(test_df)
        test_processed = self.preprocess_data(test_enhanced, is_training=False)
        X_test, _ = self.prepare_features(test_processed)
        
        # Generate predictions
        predictions = self.best_model.predict(X_test)
        predictions = np.clip(predictions, 0, 1)  # Ensure within [0,1] range
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': test_df['id'],
            'accident_risk': [round(pred, 3) for pred in predictions]
        })
        
        return submission
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print("Feature importance not available for this model")
            return None


def main():
    """Main execution function"""
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Initialize and train predictor
    predictor = RoadAccidentPredictor()
    
    # Train with cross-validation (recommended for better performance)
    predictor.train(train_df, use_cross_validation=True)
    
    # Get feature importance
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
    
    # Generate predictions
    submission = predictor.predict(test_df)
    
    # Save submission
    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission saved as: {submission_file}")
    
    # Show sample predictions
    print("\nSample predictions:")
    print(submission.head(10))
    
    # Summary statistics
    print(f"\nPrediction Statistics:")
    print(f"Min risk: {submission['accident_risk'].min():.3f}")
    print(f"Max risk: {submission['accident_risk'].max():.3f}")
    print(f"Mean risk: {submission['accident_risk'].mean():.3f}")


if __name__ == "__main__":
    # Run the main pipeline
    main()

