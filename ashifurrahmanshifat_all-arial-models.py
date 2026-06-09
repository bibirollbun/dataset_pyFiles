import os
import logging
from typing import Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Force unbuffered output for real-time logging in Kaggle
os.environ['PYTHONUNBUFFERED'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    import pandas as pd
    import numpy as np
    import tqdm
    if not tf.__version__.startswith('2.'):
        logger.warning(f"TensorFlow version {tf.__version__} detected. Version 2.17.0+ recommended.")
    if not tfp.__version__.startswith('0.'):
        logger.warning(f"TensorFlow Probability version {tfp.__version__} detected. Version 0.24.0+ recommended.")
    logger.info(f"Using TensorFlow {tf.__version__} and TensorFlow Probability {tfp.__version__}")
except ImportError as e:
    logger.error(f"Missing dependency: {e}. Install: tensorflow==2.17.0, tensorflow-probability==0.24.0, pandas, numpy, tqdm")
    raise

# Configuration
class Config:
    """Configuration for the Ariel Data Challenge 2025."""
    data_path: str = '/kaggle/input/ariel-data-challenge-2025'
    train_spec_file: str = 'train.csv'
    wavelengths_file: str = 'wavelengths.csv'
    test_ids_file: str = 'sample_submission.csv'
    train_dir: str = 'train'
    test_dir: str = 'test'
    batch_size: int = 32
    epochs: int = 20
    cache_dir: str = '/kaggle/working/cache'
    checkpoint_dir: str = '/kaggle/working/checkpoints'
    chunk_size: int = 100
    mc_dropout_samples: int = 5
    n_outputs: int = 0
    default_n_outputs: int = 283

# Custom Attention Layer
class AttentionLayer(layers.Layer):
    """Custom attention layer to focus on important spectral features."""
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1, activation=None)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        scores = self.dense(inputs)
        attention_weights = tf.nn.softmax(scores, axis=1)
        return inputs * attention_weights

# Physics-Informed Loss Layer
class PhysicsLossLayer(layers.Layer):
    """Enforces smoothness and energy variance constraints for spectra."""
    def __init__(self, wavelengths: np.ndarray, phys_reg_weight: float, n_outputs: int):
        super().__init__()
        if n_outputs < 3:
            raise ValueError(f"n_outputs must be >= 3 for Conv1D with kernel_size=3, got {n_outputs}")
        self.wavelengths = tf.convert_to_tensor(wavelengths, dtype=tf.float32)
        self.phys_reg_weight = phys_reg_weight
        self.n_outputs = n_outputs
        kernel = np.array([1., -2., 1.], dtype=np.float32).reshape(3, 1, 1)
        self.conv = layers.Conv1D(1, 3, padding='valid', use_bias=False, trainable=False)
        self.conv.build((None, self.n_outputs, 1))
        self.conv.set_weights([kernel])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mu = inputs
        mu_expanded = tf.expand_dims(mu, axis=-1)
        d2 = self.conv(mu_expanded)
        smooth_loss = tf.reduce_mean(tf.square(d2), axis=[1, 2])
        E = tf.reduce_sum(mu / tf.math.pow(self.wavelengths, 4), axis=1)
        energy_var = tfp.stats.variance(E)
        phys_loss = self.phys_reg_weight * (smooth_loss + 0.1 * energy_var)
        return phys_loss

    def get_config(self) -> dict:
        return {
            'wavelengths': self.wavelengths.numpy(),
            'phys_reg_weight': self.phys_reg_weight,
            'n_outputs': self.n_outputs
        }

# Model Definition
class ArielModel(keras.Model):
    """CNN-based model with attention for spectral prediction and uncertainties."""
    def __init__(
        self,
        input_shape: tuple,
        output_shape: int,
        wavelengths: np.ndarray,
        phys_reg_weight: float,
        dropout_rate: float
    ):
        super().__init__()
        self.conv1 = layers.Conv1D(64, 5, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(32, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.attention = AttentionLayer()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.mean_output = layers.Dense(output_shape, name='mean_output')
        self.var_output = layers.Dense(output_shape, activation='softplus', name='var_output')
        self.phys_loss = PhysicsLossLayer(wavelengths, phys_reg_weight, output_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> dict:
        x = tf.expand_dims(inputs, axis=-1)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        mean = self.mean_output(x)
        var = self.var_output(x)
        phys = self.phys_loss(mean)
        return {'mean_output': mean, 'var_output': var, 'phys_loss': phys}

# Data Pipeline
class ArielDataPipeline:
    """Handles data loading and preprocessing for all planets."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            os.makedirs(cfg.cache_dir, exist_ok=True)
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories {cfg.cache_dir} or {cfg.checkpoint_dir}: {e}")
            raise

    def _load_observation(self, pid: str, is_train: bool = True) -> np.ndarray:
        """Load and average observations for a planet."""
        pattern = 'AIRS-CH0_signal_{}.parquet'
        obs_dir = os.path.join(self.cfg.data_path, self.cfg.train_dir if is_train else self.cfg.test_dir, pid)
        obs = []
        n_outputs = self.cfg.n_outputs if self.cfg.n_outputs > 0 else self.cfg.default_n_outputs
        for i in range(2):
            obs_file = os.path.join(obs_dir, pattern.format(i))
            if not os.path.exists(obs_file):
                logger.warning(f"File not found: {obs_file}, using zeros")
                obs.append(np.zeros(n_outputs, dtype=np.float32))
                continue
            try:
                obs_df = pd.read_parquet(obs_file)
                arr = obs_df.values.astype(np.float32).mean(axis=0)[:n_outputs]
                if np.any(np.isnan(arr)) or np.any(arr < 0):
                    logger.warning(f"Invalid data in {obs_file}: NaNs or negative values")
                    mask = np.isnan(arr) | (arr < 0)
                    if np.all(mask):
                        logger.warning(f"All values invalid in {obs_file}, using zeros")
                        arr = np.zeros_like(arr)
                    else:
                        arr[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], arr[~mask])
                obs.append(arr)
            except Exception as e:
                logger.error(f"Error loading {obs_file}: {e}")
                obs.append(np.zeros(n_outputs, dtype=np.float32))
        return np.mean(obs, axis=0)

    def _worker(self, args: Tuple[str, bool]) -> np.ndarray:
        """Worker function for parallel processing."""
        pid, is_train = args
        return self._load_observation(pid, is_train)

    def _prepare_cache(self, planet_ids: List[str], is_train: bool = True) -> np.ndarray:
        """Cache observations in parallel with chunked processing."""
        cache_file = os.path.join(self.cfg.cache_dir, f'cached_obs_{"train" if is_train else "test"}.npy')
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            try:
                cached_obs = np.load(cache_file)
                n_outputs = self.cfg.n_outputs if self.cfg.n_outputs > 0 else self.cfg.default_n_outputs
                if cached_obs.shape[0] == len(planet_ids) and cached_obs.shape[1] == n_outputs:
                    return cached_obs
                logger.warning(f"Cached data shape mismatch, regenerating cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}, regenerating")

        logger.info(f"Caching observations for {len(planet_ids)} planets...")
        cached_obs = []
        for chunk_start in range(0, len(planet_ids), self.cfg.chunk_size):
            chunk_end = min(chunk_start + self.cfg.chunk_size, len(planet_ids))
            chunk_ids = planet_ids[chunk_start:chunk_end]
            logger.info(f"Processing chunk {chunk_start // self.cfg.chunk_size + 1}/{(len(planet_ids) - 1) // self.cfg.chunk_size + 1}")
            with Pool(processes=cpu_count()) as pool:
                chunk_obs = np.array(list(tqdm.tqdm(
                    pool.imap(self._worker, [(pid, is_train) for pid in chunk_ids]),
                    total=len(chunk_ids),
                    desc=f"Caching chunk {chunk_start // self.cfg.chunk_size + 1}"
                )), dtype=np.float32)
            cached_obs.append(chunk_obs)
        cached_obs = np.concatenate(cached_obs, axis=0)
        try:
            np.save(cache_file, cached_obs)
            logger.info(f"Saved cached data to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
        return cached_obs

    def prepare_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, List[str]]:
        """Prepare datasets for training and testing."""
        # Validate file paths
        for file_path in [
            os.path.join(self.cfg.data_path, self.cfg.train_spec_file),
            os.path.join(self.cfg.data_path, self.cfg.wavelengths_file),
            os.path.join(self.cfg.data_path, self.cfg.test_ids_file)
        ]:
            if not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load wavelengths
        logger.info("Loading wavelengths...")
        try:
            wavelengths_df = pd.read_csv(os.path.join(self.cfg.data_path, self.cfg.wavelengths_file))
            logger.info(f"Loaded wavelengths_df with shape: {wavelengths_df.shape}")
            logger.info(f"Wavelengths_df columns: {wavelengths_df.columns.tolist()}")
            logger.info(f"First 5 rows of wavelengths_df:\n{wavelengths_df.head().to_string()}")
            self.cfg.n_outputs = len(wavelengths_df)
            if self.cfg.n_outputs < 3:
                logger.warning(f"Number of wavelengths ({self.cfg.n_outputs}) is too small. Using default n_outputs={self.cfg.default_n_outputs}.")
                self.cfg.n_outputs = self.cfg.default_n_outputs
                wavelengths = np.linspace(0.5, 5.0, self.cfg.n_outputs).astype(np.float32)
            else:
                wavelengths = wavelengths_df.values.flatten().astype(np.float32)
            logger.info(f"Set n_outputs to {self.cfg.n_outputs}")
        except Exception as e:
            logger.error(f"Failed to load wavelengths: {e}. Using default n_outputs={self.cfg.default_n_outputs}.")
            self.cfg.n_outputs = self.cfg.default_n_outputs
            wavelengths = np.linspace(0.5, 5.0, self.cfg.n_outputs).astype(np.float32)

        # Load training data
        logger.info("Loading training data...")
        try:
            train_spec_df = pd.read_csv(os.path.join(self.cfg.data_path, self.cfg.train_spec_file))
            train_ids = train_spec_df['planet_id'].astype(str).values
            train_spec = train_spec_df.filter(regex='wl_').values[:, :self.cfg.n_outputs].astype(np.float32)
            logger.info(f"Loaded {len(train_ids)} training samples")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

        if len(train_ids) != len(train_spec):
            raise ValueError(f"Train IDs ({len(train_ids)}) and train spectra ({len(train_spec)}) mismatch")
        if train_spec.shape[1] != self.cfg.n_outputs:
            raise ValueError(f"Train spectra columns ({train_spec.shape[1]}) mismatch with n_outputs ({self.cfg.n_outputs})")

        # Load test data
        logger.info("Loading test data...")
        try:
            test_ids_df = pd.read_csv(os.path.join(self.cfg.data_path, self.cfg.test_ids_file))
            test_ids = test_ids_df['planet_id'].astype(str).values
            logger.info(f"Loaded {len(test_ids)} test samples")
        except Exception as e:
            logger.error(f"Failed to load test IDs: {e}")
            raise

        # Cache observations
        logger.info("Caching training observations...")
        train_obs = self._prepare_cache(train_ids, is_train=True)
        logger.info("Caching test observations...")
        test_obs = self._prepare_cache(test_ids, is_train=False)

        # Validate shapes
        if train_obs.shape[0] != len(train_ids) or train_obs.shape[1] != self.cfg.n_outputs:
            raise ValueError(f"Train observations shape {train_obs.shape} mismatch with expected ({len(train_ids)}, {self.cfg.n_outputs})")
        if test_obs.shape[0] != len(test_ids) or test_obs.shape[1] != self.cfg.n_outputs:
            raise ValueError(f"Test observations shape {test_obs.shape} mismatch with expected ({len(test_ids)}, {self.cfg.n_outputs})")

        # Create datasets with dictionary output for y_true
        ds_train = tf.data.Dataset.from_tensor_slices((
            train_obs,
            {'mean_output': train_spec, 'var_output': tf.zeros_like(train_spec), 'phys_loss': tf.zeros((len(train_ids),))}
        )).batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        ds_test = tf.data.Dataset.from_tensor_slices(test_obs).batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        logger.info(f"Prepared datasets: {len(train_ids)} train samples, {len(test_ids)} test samples")
        return ds_train, ds_test, wavelengths, test_ids

# Training and Prediction
def train_and_predict(
    cfg: Config,
    ds_train: tf.data.Dataset,
    ds_test: tf.data.Dataset,
    wavelengths: np.ndarray,
    test_ids: List[str]
) -> Tuple[keras.Model, pd.DataFrame]:
    """Train the model and generate predictions."""
    logger.info("Initializing model...")
    try:
        model = ArielModel(
            input_shape=(cfg.n_outputs,),
            output_shape=cfg.n_outputs,
            wavelengths=wavelengths,
            phys_reg_weight=0.5,
            dropout_rate=0.3
        )
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
            loss={
                'mean_output': 'mse',
                'var_output': 'mse',
                'phys_loss': lambda y_true, y_pred: y_pred
            },
            loss_weights={'mean_output': 1.0, 'var_output': 0.1, 'phys_loss': 0.5},
            metrics={'mean_output': 'mse', 'var_output': 'mse'}
        )
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

    logger.info("Starting training...")
    try:
        model.fit(
            ds_train,
            epochs=cfg.epochs,
            verbose=1,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    f'{cfg.checkpoint_dir}/model.h5',
                    save_best_only=True,
                    monitor='loss',
                    mode='min'
                ),
                keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info("Generating predictions with MC-Dropout...")
    mus, stds = [], []
    try:
        for x in tqdm.tqdm(ds_test, desc="Test batches"):
            preds = [model(x, training=True)['mean_output'] for _ in range(cfg.mc_dropout_samples)]
            preds = np.array([p.numpy() for p in preds])
            mus.append(np.mean(preds, axis=0))
            stds.append(np.std(preds, axis=0))
        mu_ens = np.concatenate(mus, axis=0)
        sig_ens = np.concatenate(stds, axis=0)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

    # Validate predictions
    if mu_ens.shape != (len(test_ids), cfg.n_outputs) or sig_ens.shape != (len(test_ids), cfg.n_outputs):
        logger.error(f"Prediction shape mismatch: expected ({len(test_ids)}, {cfg.n_outputs}), got mu_ens {mu_ens.shape}, sig_ens {sig_ens.shape}")
        raise ValueError(f"Prediction shape mismatch")
    if np.all(mu_ens == 0) or np.all(sig_ens == 0):
        logger.error(f"All prediction values are zero: mu_ens mean={np.mean(mu_ens)}, sig_ens mean={np.mean(sig_ens)}")
        raise ValueError("Prediction values are all zero")
    if np.any(np.isnan(mu_ens)) or np.any(np.isnan(sig_ens)):
        logger.warning("NaN values in predictions, replacing with zeros")
        mu_ens = np.nan_to_num(mu_ens, nan=0.0)
        sig_ens = np.nan_to_num(sig_ens, nan=0.0)
    logger.info(f"Prediction stats - mu_ens mean: {np.mean(mu_ens):.4f}, sig_ens mean: {np.mean(sig_ens):.4f}")

    # Create submission DataFrame
    logger.info("Creating submission DataFrame...")
    try:
        sample_submission = pd.read_csv(os.path.join(cfg.data_path, cfg.test_ids_file))
        expected_columns = sample_submission.columns.tolist()
        logger.info(f"Expected columns: {expected_columns}")

        if len(expected_columns) != 1 + 2 * cfg.n_outputs:
            logger.error(f"Expected {1 + 2 * cfg.n_outputs} columns (1 planet_id + {2 * cfg.n_outputs} predictions), got {len(expected_columns)}")
            raise ValueError(f"Column count mismatch in sample_submission.csv")

        prediction_cols = expected_columns[1:]
        logger.info(f"Using prediction columns: {prediction_cols[:5]}... (total {len(prediction_cols)})")

        df_data = np.hstack((np.array(test_ids).reshape(-1, 1), mu_ens, sig_ens))
        if df_data.shape[1] != len(expected_columns):
            logger.error(f"Data shape {df_data.shape} does not match expected columns {len(expected_columns)}")
            raise ValueError(f"Data column count {df_data.shape[1]} does not match {len(expected_columns)}")

        df = pd.DataFrame(df_data, columns=expected_columns)
        df['planet_id'] = df['planet_id'].astype(str)
        for col in prediction_cols:
            df[col] = df[col].astype(float)

        if list(df.columns) != expected_columns:
            logger.error(f"Submission columns mismatch: expected {expected_columns[:5]}..., got {list(df.columns)[:5]}...")
            raise ValueError(f"Submission columns mismatch")
        if df.iloc[0].isna().all():
            logger.error("All values in DataFrame are NaN")
            raise ValueError("DataFrame contains only NaN values")

        logger.info(f"Sample values - First row {prediction_cols[0]}: {df.iloc[0][prediction_cols[0]]:.4f}, {prediction_cols[cfg.n_outputs]}: {df.iloc[0][prediction_cols[cfg.n_outputs]]:.4f}")

    except Exception as e:
        logger.error(f"Failed to create submission DataFrame: {e}")
        raise

    # Save submission
    try:
        df.to_csv('/kaggle/working/submission.csv', index=False)
        logger.info("Submission saved to /kaggle/working/submission.csv")
        logger.info(f"Submission preview (first 5 rows):\n{df.head().to_string()}")
    except Exception as e:
        logger.error(f"Failed to save submission: {e}")
        raise

    return model, df

# Main Execution
def main():
    """Execute the pipeline for all planets."""
    logger.info("Starting execution for all planets...")
    tf.keras.utils.set_random_seed(42)
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    cfg = Config()
    data_pipeline = ArielDataPipeline(cfg)

    try:
        ds_train, ds_test, wavelengths, test_ids = data_pipeline.prepare_datasets()
        logger.info(f"Model will be initialized with n_outputs={cfg.n_outputs}")
        model, submission_df = train_and_predict(cfg, ds_train, ds_test, wavelengths, test_ids)
        logger.info("Execution completed successfully!")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == '__main__':
    main()




