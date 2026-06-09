import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

class GeologyPredictor:
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = {}
        self.feature_columns = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = df.copy()

        input_cols = [col for col in df.columns if col.lstrip('-').isdigit() and int(col) <= 0]
        input_cols = sorted(input_cols, key=lambda x: int(x))

        for idx, row in df.iterrows():
            available_data = []
            positions = []

            for col in input_cols:
                if pd.notna(row[col]):
                    available_data.append(row[col])
                    positions.append(int(col))

            if len(available_data) > 1:
                available_data = np.array(available_data)
                positions = np.array(positions)

                features_df.loc[idx, 'mean_z'] = np.mean(available_data)
                features_df.loc[idx, 'std_z'] = np.std(available_data)
                features_df.loc[idx, 'min_z'] = np.min(available_data)
                features_df.loc[idx, 'max_z'] = np.max(available_data)
                features_df.loc[idx, 'range_z'] = np.max(available_data) - np.min(available_data)

                if len(available_data) > 2:
                    trend_coef = np.polyfit(positions, available_data, 1)[0]
                    features_df.loc[idx, 'trend_slope'] = trend_coef

                    last_5 = available_data[-5:] if len(available_data) >= 5 else available_data
                    last_5_pos = positions[-5:] if len(positions) >= 5 else positions
                    if len(last_5) > 1:
                        recent_trend = np.polyfit(last_5_pos, last_5, 1)[0]
                        features_df.loc[idx, 'recent_trend'] = recent_trend

                    changes = np.diff(available_data)
                    features_df.loc[idx, 'volatility'] = np.std(changes) if len(changes) > 0 else 0
                    features_df.loc[idx, 'mean_change'] = np.mean(changes) if len(changes) > 0 else 0

                features_df.loc[idx, 'last_value'] = available_data[-1]
                if len(available_data) > 1:
                    features_df.loc[idx, 'second_last_value'] = available_data[-2]
                if len(available_data) > 2:
                    features_df.loc[idx, 'third_last_value'] = available_data[-3]

               
                features_df.loc[idx, 'data_length'] = len(available_data)
                features_df.loc[idx, 'first_position'] = positions[0]
                features_df.loc[idx, 'last_position'] = positions[-1]

        feature_cols = ['mean_z', 'std_z', 'min_z', 'max_z', 'range_z', 'trend_slope',
                        'recent_trend', 'volatility', 'mean_change', 'last_value',
                        'second_last_value', 'third_last_value', 'data_length',
                        'first_position', 'last_position']

       
        for col in feature_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0)

        return features_df[['geology_id'] + feature_cols]

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Eğitim verisi hazırlama"""
        features_df = self.create_features(df)

       
        self.feature_columns = [col for col in features_df.columns if col != 'geology_id']

        X = features_df[self.feature_columns].values

        target_cols = [str(i) for i in range(1, 301)]
        y = df[target_cols].values

        return X, y

    def train(self, df: pd.DataFrame):
        print("Extracting features...")
        X, y = self.prepare_training_data(df)

        print(X.shape, y.shape)

        print("Training model...")
        for pos in range(y.shape[1]):
            valid_mask = ~np.isnan(y[:, pos])
            if np.sum(valid_mask) > 10:  
                X_pos = X[valid_mask]
                y_pos = y[valid_mask, pos]

                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_pos, y_pos)
                self.models[pos] = model

        print(f"Training finished")

    def predict_realizations(self, df: pd.DataFrame, n_realizations=10) -> np.ndarray:
        features_df = self.create_features(df)
        X = features_df[self.feature_columns].values

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, 300, n_realizations))

        for pos in range(300):
            if pos in self.models:
                base_pred = self.models[pos].predict(X)

                for real in range(n_realizations):
                    noise_scale = np.std(base_pred) * 0.1 
                    noise = np.random.normal(0, noise_scale, len(base_pred))
                    predictions[:, pos, real] = base_pred + noise
            else:
                for i in range(n_samples):
                    last_known = features_df.iloc[i]['last_value']
                    trend = features_df.iloc[i]['recent_trend']

                    for real in range(n_realizations):
                        pred_value = last_known + trend * (pos + 1)
                        noise = np.random.normal(0, abs(pred_value) * 0.1)
                        predictions[i, pos, real] = pred_value + noise

        return predictions

    def create_submission(self, test_df: pd.DataFrame, n_realizations=10) -> pd.DataFrame:
        predictions = self.predict_realizations(test_df, n_realizations)

        submission = test_df[['geology_id']].copy()

        for pos in range(300):
            submission[str(pos + 1)] = predictions[:, pos, 0]

        for real in range(1, n_realizations):
            for pos in range(300):
                col_name = f"r_{real}_pos_{pos + 1}"
                submission[col_name] = predictions[:, pos, real]

        return submission


def load_and_analyze_data(train_path: str):
    train_df = pd.read_csv(train_path)

    input_cols = [col for col in train_df.columns if col.lstrip('-').isdigit() and int(col) <= 0]

    target_cols = [col for col in train_df.columns if col.isdigit() and int(col) > 0]

    return train_df


def plot_sample_geology(df: pd.DataFrame, sample_idx: int = 0):
    row = df.iloc[sample_idx]

    input_cols = [col for col in df.columns if col.lstrip('-').isdigit() and int(col) <= 0]
    input_positions = []
    input_values = []

    for col in sorted(input_cols, key=lambda x: int(x)):
        if pd.notna(row[col]):
            input_positions.append(int(col))
            input_values.append(row[col])

    target_cols = [str(i) for i in range(1, 301)]
    target_positions = list(range(1, 301))
    target_values = [row[col] if pd.notna(row[col]) else None for col in target_cols]

    plt.figure(figsize=(15, 6))
    plt.plot(input_positions, input_values, 'b-o', label='Bilinen veri', markersize=3)

 
    valid_targets = [(pos, val) for pos, val in zip(target_positions, target_values) if val is not None]
    if valid_targets:
        target_pos, target_val = zip(*valid_targets)
        plt.plot(target_pos, target_val, 'r-o', label='Hedef veri', markersize=3)

    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Mevcut pozisyon')
    plt.xlabel('Pozisyon (feet)')
    plt.ylabel('Z koordinatı (feet)')
    plt.title(f'Jeoloji Profili - Örnek {sample_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_df = load_and_analyze_data("/kaggle/input/geology-forecast-challenge-open/data/train.csv")
  
    plot_sample_geology(train_df, sample_idx=0)
   
    predictor = GeologyPredictor(n_estimators=50, random_state=42)
    predictor.train(train_df)

   
    test_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv")
    submission = predictor.create_submission(test_df)
    submission.to_csv("submission.csv", index=False)

    print("Submission file saved")

