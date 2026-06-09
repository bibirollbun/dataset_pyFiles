

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom feature transformer with different feature engineering versions"""

    def __init__(self, create_version=1):
        self.create_version = create_version
        self.medians = {}
        self.quantile_transformers = {}
        self.kmeans = None
        self.quantiles = {}
        self.scaler = RobustScaler()
        self.feature_params = {}

    def fit(self, X, y=None):
        X = X.copy()

        if self.create_version == 1:
            # Handle bimodal features
            bimodal_features = ['AcousticQuality',
                                'InstrumentalScore', 'LivePerformanceLikelihood']
            for feature in bimodal_features:
                self.medians[feature] = X[feature].median()
                self.quantiles[feature] = np.quantile(
                    X[feature], [0, 0.2, 0.4, 0.6, 0.8, 1.0])

            # Apply quantile transformation
            self.quantile_transformers['quantile'] = QuantileTransformer(
                output_distribution='normal')
            quantile_features = ['RhythmScore', 'Energy', 'AudioLoudness']
            self.quantile_transformers['quantile'].fit(X[quantile_features])

            # Apply clustering
            cluster_features = ['RhythmScore', 'Energy', 'AudioLoudness']
            if len(X) > 100:
                self.kmeans = KMeans(n_clusters=5, n_init=10, random_state=3)
                self.kmeans.fit(X[cluster_features])

        elif self.create_version == 2:
            # Apply quantile transformation to all columns except BPM
            columns = [col for col in X.columns if col != 'BeatsPerMinute']
            self.quantile_transformers['quantile'] = QuantileTransformer(
                output_distribution='normal')
            self.quantile_transformers['quantile'].fit(X[columns])

        elif self.create_version == 3:
            # Calculate quantiles for specific features
            self.quantiles['Energy'] = np.quantile(
                X['Energy'], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            self.quantiles['RhythmScore'] = np.quantile(
                X['RhythmScore'], [0, 0.2, 0.4, 0.6, 0.8, 1.0])

        elif self.create_version == 4:
            # Apply quantile transformation to selected features
            features = ['MoodScore', 'TrackDurationMs',
                        'RhythmScore', 'Energy']
            self.quantile_transformers['quantile'] = QuantileTransformer(
                output_distribution='normal')
            self.quantile_transformers['quantile'].fit(X[features])

        return self

    def transform(self, X):
        X = X.copy()

        if self.create_version == 1:
            # Transform bimodal features
            bimodal_features = ['AcousticQuality',
                                'InstrumentalScore', 'LivePerformanceLikelihood']
            for feature in bimodal_features:
                X[f'binary_{feature}'] = (
                    X[feature] > self.medians[feature]).astype(int)
                X[f'{feature}_cat'] = np.digitize(
                    X[feature], self.quantiles[feature][1:-1]) - 1

            # Apply quantile transformation
            quantile_features = ['RhythmScore', 'Energy', 'AudioLoudness']
            quantile_transformed = self.quantile_transformers['quantile'].transform(
                X[quantile_features])
            for i, feature in enumerate(quantile_features):
                X[f'{feature}_quantile'] = quantile_transformed[:, i]

            # Add cluster features
            cluster_features = ['RhythmScore', 'Energy', 'AudioLoudness']
            if self.kmeans is not None:
                X['cluster'] = self.kmeans.predict(X[cluster_features])

            # Add polynomial and sqrt features
            X['Energy_squared'] = X['Energy'] ** 2
            X['Rhythm_squared'] = X['RhythmScore'] ** 2
            X['Energy_cube'] = X['Energy'] ** 3
            X['Rhythm_cube'] = X['RhythmScore'] ** 3
            X['AudioLoudness_sqrt'] = np.sqrt(np.abs(X['AudioLoudness']))
            X['RhythmScore_sqrt'] = np.sqrt(np.abs(X['RhythmScore']))
            X['Energy_sqrt'] = np.sqrt(np.abs(X['Energy']))

        elif self.create_version == 2:
            columns = [col for col in X.columns if col != 'BeatsPerMinute']

            # Apply logarithmic transformation
            for col in columns:
                X[f'log_{col}'] = np.log1p(X[col] - X[col].min())

            # Apply quantile transformation
            quantile_transformed = self.quantile_transformers['quantile'].transform(
                X[columns])
            for i, feature in enumerate(columns):
                X[f'{feature}_quantile'] = quantile_transformed[:, i]

            # Add polynomial and sqrt features
            for col in columns:
                X[f'squared_{col}'] = X[col]**2
                X[f'cube_{col}'] = X[col]**3
                X[f'sqrt_{col}'] = np.sqrt(np.abs(X[col]))

            # Add z-score features
            for col in columns:
                X[f'zscore_{col}'] = (X[col] - X[col].mean()) / X[col].std()

            # Add interaction features
            X['mean_rhythm_energy'] = (X['RhythmScore'] + X['Energy']) / 2

        elif self.create_version == 3:
            # Create interaction features
            X['RhythmEnergyProduct'] = X['RhythmScore'] * X['Energy']
            X['RhythmEnergyRatio'] = X['RhythmScore'] / (X['Energy'] + 1e-8)

            X['LoudnessEnergyProduct'] = X['AudioLoudness'] * X['Energy']
            X['VocalInstrumentalRatio'] = X['VocalContent'] / \
                (X['InstrumentalScore'] + 1e-8)

            X['DurationMoodProduct'] = X['TrackDurationMs'] * X['MoodScore']

            X['QualityPerformanceProduct'] = X['AcousticQuality'] * \
                X['LivePerformanceLikelihood']

            # Add transformations for top features
            top_3_features = ['RhythmScore', 'MoodScore', 'TrackDurationMs']
            for feature in top_3_features:
                X[f'{feature}_squared'] = X[feature] ** 2
                X[f'{feature}_sqrt'] = np.sqrt(np.abs(X[feature]))

            # Bin features
            X['EnergyBin'] = np.digitize(
                X['Energy'], self.quantiles['Energy'][1:-1]) - 1
            X['RhythmBin'] = np.digitize(
                X['RhythmScore'], self.quantiles['RhythmScore'][1:-1]) - 1

            X['RhythmDurationInteraction'] = X['RhythmScore'] * X['TrackDurationMs']

        elif self.create_version == 4:
            features = ['MoodScore', 'TrackDurationMs',
                        'RhythmScore', 'Energy']

            # Apply quantile transformation
            quantile_transformed = self.quantile_transformers['quantile'].transform(
                X[features])
            for i, feature in enumerate(features):
                X[f'{feature}_quantile'] = quantile_transformed[:, i]

            # Add polynomial and sqrt features
            for col in features:
                X[f'squared_{col}'] = X[col]**2
                X[f'cube_{col}'] = X[col]**3
                X[f'sqrt_{col}'] = np.sqrt(np.abs(X[col]))

            # Add z-score features
            for col in features:
                X[f'zscore_{col}'] = (X[col] - X[col].mean()) / X[col].std()

            # Apply logarithmic transformation
            for col in features:
                X[f'log_{col}'] = np.log1p(X[col] - X[col].min())

            X['Mood_quantile_log'] = np.log1p(
                X['MoodScore_quantile'] - X['MoodScore_quantile'].min())

        return X


def load_data():
    """Load and preprocess training and test data"""
    print("\nData loaded")
    train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv', index_col='id')

    # Convert milliseconds to minutes
    train['TrackDurationMs'] = train['TrackDurationMs']/60000
    test['TrackDurationMs'] = test['TrackDurationMs']/60000

    print(f"\nTrain size: {train.shape}\nTest size: {test.shape}")

    return train, test


def choose_feature_engin(train, test, create):
    """Apply feature engineering based on selected version"""
    transformer = FeatureTransformer(create_version=create)
    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)

    print(f"\nApply create Features #{create} ")
    return train_transformed, test_transformed


def evaluate_model(model, X, y, model_name, n_folds=12):
    """Evaluate model using K-Fold cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    fold_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if model_name in ['Ridge', 'Elastic Net']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        score = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_scores.append(score)

    avg_score = np.mean(fold_scores)
    result = {
        'model_name': model_name,
        'RMSE_score': avg_score,
        'model': model
    }

    return result


def model_params():
    """Return optimized parameters for different models"""
    lgbm_params = {
        'n_estimators': 100,
        'learning_rate': 0.003,
        'num_leaves': 35,
        'max_depth': 10,
        'subsample': 0.8,
        'min_child_sample': 92,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.8,
        'reg_lambda': 0.2
    }

    cat_params = {
        'min_data_in_leaf': 65,
        'iterations': 100,
        'depth': 5,
        'learning_rate': 0.003,
        'l2_leaf_reg': 9,
        'random_strength': 0.2,
        'bagging_temperature': 0.9
    }

    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.003,
        'max_depth': 9,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.2,
        'reg_lambda': 0.9
    }

    return lgbm_params, cat_params, xgb_params


def create_model(X, y, n_folds):
    """Train and evaluate multiple models"""
    results = []
    lgbm_params, cat_params, xgb_params = model_params()

    models = {
        'XGBoost': XGBRegressor(
            **xgb_params,
            random_state=1,
            n_jobs=1,
            device='gpu'
        ),
        'LightGBM': LGBMRegressor(
            **lgbm_params,
            random_state=1,
            n_jobs=1,
            verbose=-1,
            device='gpu'
        ),
        'CatBoost': CatBoostRegressor(
            **cat_params,
            random_state=1,
            verbose=False,
            task_type='GPU'
        ),
        'Ridge': Ridge(alpha=3.0),
        'Elastic Net': ElasticNet(alpha=3.0, random_state=1)
    }

    print(f"\nTrain model with {len(X.columns)} features.")
    for name, model in models.items():
        print(f"\nTraining model: {name}")
        result = evaluate_model(model, X, y, name, n_folds=n_folds)
        print(f"RMSE: {name} {result['RMSE_score']:.5f} ")
        results.append(result)

    return results


def correlation_df(X, create_version):
    """Analyze feature correlations with target"""
    target_col = 'BeatsPerMinute'

    X = X.copy()
    corr_matrix = X.corr()
    target_corr = corr_matrix[target_col].drop(
        target_col).abs().sort_values(ascending=False)

    # Select top 15 features
    top_features = target_corr.head(15)

    # Create correlation DataFrame
    result_df = pd.DataFrame({'Correlation': top_features})

    # Add outlier percentage information
    outlier_percentages = []
    for feature in top_features.index:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR * 1.5
        upper_bound = Q3 + IQR * 1.5

        outlier_count = ((X[feature] < lower_bound) |
                         (X[feature] > upper_bound)).sum()
        outlier_percentage = outlier_count / len(X) * 100
        outlier_percentages.append(outlier_percentage)

    result_df['Outlier_Percentage'] = outlier_percentages

    # Add transformation version information
    result_df['Create_Version'] = create_version

    return result_df


def quick_stacking(X_train, y_train, X_val, y_val, results_df, cv=3):
    """Create stacking ensemble model"""
    models = []
    for _, row in results_df.iterrows():
        model = row['model']
        params = model.get_params()
        model_name = row['model_name']

        new_model = model.__class__(**params)
        models.append((model_name, new_model))

    stack = StackingRegressor(
        estimators=models,
        final_estimator=RandomForestRegressor(
            n_estimators=20,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=5,
            random_state=3,
            n_jobs=-1),
        cv=cv,
        n_jobs=1
    )
    print("\nStacking...")
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_val)

    return np.sqrt(mean_squared_error(y_val, y_pred)), stack, y_pred


def weighted_blending(X_train, y_train, X_val, results_df):
    """Blend models using weighted averaging"""
    weights = 1 / results_df['RMSE_score'].values
    weights = weights / weights.sum()

    predictions = []
    for _, row in results_df.iterrows():
        model = row['model']
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_val))

    weighted_pred = np.average(np.column_stack(
        predictions), axis=1, weights=weights)
    return weighted_pred


def create_submission(test_index, final_pred, sub_name='submission_stacking_new_1'):
    """Create submission file"""
    submission = pd.DataFrame({
        'id': test_index,
        'BeatsPerMinute': final_pred
    })
    submission.to_csv(f'{sub_name}.csv', index=False)
    print(f"Submission file saved as {sub_name}.csv")
    print(f"Submission shape: {submission.shape}")


def main(create, n_folds, feat_engineering, train_mode, stacking,
         get_correlation, train_individual, submission=False):
    """Main pipeline function"""
    train, test = load_data()
    target_col = 'BeatsPerMinute'

    # Save test index for submission
    test_index = test.index

    train_engi, test_engi = choose_feature_engin(train, test, create)

    # Initialize matrix to None
    matrix = None

    if get_correlation:
        matrix = correlation_df(train_engi, create)

        corr_matrix = train_engi.corr()
        features_corr = corr_matrix[target_col].drop(
            target_col).abs().sort_values(ascending=False)

        print(f"\nTop 15 Features by Correlation with BPM:")
        for i, (feature, corr) in enumerate(features_corr.head(15).items(), 1):
            print(f"{i:2d}. {feature:<30} | Correlation: {corr:.4f}")

    col_names = [col for col in train_engi.columns if col != target_col]

    X = train_engi[col_names]
    y = train_engi[target_col]

    X_test = test_engi[col_names]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=1)

    results_df = None
    best_individual_rmse = None
    stack_model = None
    blend_pred = None

    if train_individual:
        results = create_model(X_train, y_train, n_folds)
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE_score')
        print("\n" + "="*50)
        print("INDIVIDUAL MODELS RESULTS:")
        print("="*50)
        for _, row in results_df.iterrows():
            print(f"{row['model_name']:<15} | RMSE: {row['RMSE_score']:.5f}")

        # Evaluate best model on validation set
        best_model = results_df.iloc[0]['model']
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_val)
        best_individual_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        print(
            f"\nBest individual model validation RMSE: {best_individual_rmse:.5f}")

    if stacking:
        if results_df is None:
            print("\nTraining individual models for stacking...")
            results = create_model(X_train, y_train, n_folds)
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('RMSE_score')

        mse_stack, stack, y_pred = quick_stacking(
            X_train, y_train, X_val, y_val, results_df, cv=5)
        print(f"\nStacking RMSE: {mse_stack:.5f}")
        stack_model = stack

        if best_individual_rmse is not None:
            improvement = best_individual_rmse - mse_stack
            print(f"Best individual model RMSE: {best_individual_rmse:.5f}")
            print(
                f"Improvement from stacking: {improvement:.5f} ({improvement/best_individual_rmse*100:.5f}%)")

    # Blending
    if results_df is not None:
        print("\nBlending...")
        y_pred_weighted = weighted_blending(
            X_train, y_train, X_val, results_df)
        mse_weighted = np.sqrt(mean_squared_error(y_val, y_pred_weighted))
        blend_pred = y_pred_weighted

        if best_individual_rmse is not None:
            improvement_blend = best_individual_rmse - mse_weighted
            print(f"Blending RMSE: {mse_weighted:.5f}")
            print(
                f"Improvement from blending: {improvement_blend:.5f} ({improvement_blend/best_individual_rmse*100:.5f}%)")

    # Create submission
    if submission:
        print("Creating submission...")
        if stack_model is not None:
            # Use stacking model
            test_predictions = stack_model.predict(X_test)
            create_submission(test_index, test_predictions,
                              f'submission_stacking_{create}')
        elif results_df is not None and not results_df.empty:
            # Use best individual model
            best_model = results_df.iloc[0]['model']
            test_predictions = best_model.predict(X_test)
            create_submission(test_index, test_predictions,
                              f'submission_individual_{create}')
        else:
            print("No trained models available for submission")

    return matrix, train_engi


if __name__ == "__main__":
    matrices = {}  # Dictionary to store all correlation matrices
    dfs = {}

    for i in (1, 2, 3, 4):
        CREATE = i
        N_FOLDS = 5
        GET_CORRELATION = True
        TRAIN_INDIVIDUAL = True
        STACKING = True
        SUBMISSION = False

        if not TRAIN_INDIVIDUAL and not STACKING:
            TRAIN_INDIVIDUAL = False
            STACKING = False

        start = datetime.now()
        print(f"Start time:\n{start}")
        print(f"\nCV: {N_FOLDS} folds")

        matrix, df = main(
            create=CREATE,
            n_folds=N_FOLDS,
            feat_engineering=True,
            train_mode=True,
            stacking=STACKING,
            get_correlation=GET_CORRELATION,
            train_individual=TRAIN_INDIVIDUAL,
            submission=SUBMISSION
        )
        dfs[f'df_{i}'] = df
        matrices[f'matrix_{i}'] = matrix  # Store result in dictionary
        print(f"\nTime spent\n{datetime.now()-start}")

    # Now you have 4 separate DataFrames:
    matrix_1 = matrices['matrix_1']
    matrix_2 = matrices['matrix_2']
    matrix_3 = matrices['matrix_3']
    matrix_4 = matrices['matrix_4']

    df_1 = dfs['df_1']
    df_2 = dfs['df_2']
    df_3 = dfs['df_3']
    df_4 = dfs['df_4']


