import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
RANDOM_STATE = 42
TEST_SIZE = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.0005
L2_REG = 0.001
DROPOUT_RATES = [0.4, 0.3, 0.2]
HIDDEN_UNITS = [256, 128, 64]
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
SUBMISSION_PATH = 'submission.csv'


def load_data(train_path: str, test_path: str) -> tuple:
    """
    加载训练和测试数据集
    参数:
        train_path: 训练数据文件路径
        test_path: 测试数据文件路径
    返回:
        train_df, test_df: 加载的数据框
    """
    logger.info(f"Loading data from {train_path} and {test_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Training data shape: {train_df.shape}, Test data shape: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    执行特征工程
    参数:
        df: 输入数据框
    返回:
        包含新特征的数据框
    """
    logger.info("Performing feature engineering")

    # 计算养分平衡
    for nutrient in ['Nitrogen', 'Phosphorous', 'Potassium']:
        if nutrient in df.columns:
            df[f'{nutrient[0]}_balance'] = df[nutrient] - df[nutrient].mean()

    # 环境交互特征
    if 'Temparature' in df.columns and 'Humidity' in df.columns:
        df['temp_humidity'] = df['Temparature'] * df['Humidity'] / 100

    if 'Moisture' in df.columns and all(n in df.columns for n in ['Nitrogen', 'Phosphorous', 'Potassium']):
        df['moisture_nutrient'] = df['Moisture'] * (
                df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
        )

    return df


def preprocess_data(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = 'Fertilizer Name'
) -> tuple:
    """
    数据预处理流程
    参数:
        train_df: 训练数据
        test_df: 测试数据
        target_col: 目标列名
    返回:
        X_train, X_val, y_train, y_val, preprocessor, label_encoder: 预处理后的数据和转换器
    """
    logger.info("Preprocessing data")

    # 定义特征类型
    categorical_features = ['Soil Type', 'Crop Type']
    numeric_features = [
        'Temparature', 'Humidity', 'Moisture',
        'Nitrogen', 'Potassium', 'Phosphorous'
    ]

    # 添加工程特征
    for feature in ['N_balance', 'P_balance', 'K_balance', 'temp_humidity', 'moisture_nutrient']:
        if feature in train_df.columns:
            numeric_features.append(feature)

    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numeric features: {numeric_features}")

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ],
        remainder='drop'  # 明确处理未指定的列
    )

    # 分离特征和目标
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # 预处理训练数据
    X_processed = preprocessor.fit_transform(X_train)

    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
    logger.info(f"Number of classes: {len(label_encoder.classes_)}")

    return X_train, X_val, y_train, y_val, preprocessor, label_encoder


def build_model(input_dim: int, num_classes: int) -> Sequential:
    """
    构建神经网络模型
    参数:
        input_dim: 输入特征维度
        num_classes: 类别数量
    返回:
        编译好的Keras模型
    """
    logger.info(f"Building model with input_dim={input_dim}, num_classes={num_classes}")

    model = Sequential()

    # 添加隐藏层
    for i, units in enumerate(HIDDEN_UNITS):
        # 第一层需要指定input_shape
        if i == 0:
            model.add(Dense(
                units,
                activation='relu',
                input_shape=(input_dim,),
                kernel_regularizer=l2(L2_REG)
            ))
        else:
            model.add(Dense(
                units,
                activation='relu',
                kernel_regularizer=l2(L2_REG)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(DROPOUT_RATES[i]))

            # 输出层
            model.add(Dense(num_classes, activation='softmax'))

            # 编译模型
            optimizer = Adam(learning_rate=LEARNING_RATE)
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

            model.summary()
    return model


def train_model(
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
) -> Sequential:
    """
    训练模型
    参数:
        model: 要训练的模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
    返回:
        训练好的模型
    """
    logger.info("Training model")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # 评估最终模型
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Final validation accuracy: {val_acc:.4f}, loss: {val_loss:.4f}")

    return model


def create_submission(
        model: Sequential,
        test_df: pd.DataFrame,
        preprocessor: ColumnTransformer,
        label_encoder: LabelEncoder,
        submission_path: str = SUBMISSION_PATH
) -> None:
    """
    生成预测提交文件
    参数:
        model: 训练好的模型
        test_df: 测试数据
        preprocessor: 数据预处理器
        label_encoder: 标签编码器
        submission_path: 提交文件路径
    """
    logger.info("Creating submission file")

    try:
        # 确保id列存在
        if 'id' not in test_df.columns:
            test_df['id'] = range(1, len(test_df) + 1)
            logger.warning("Added 'id' column to test data")

            # 预处理测试数据
            X_test = preprocessor.transform(test_df)

            # 预测概率
            test_probs = model.predict(X_test, verbose=0)

            # 获取Top3预测结果
            top3_indices = np.argsort(-test_probs, axis=1)[:, :3]
            top3_fertilizers = label_encoder.inverse_transform(top3_indices.reshape(-1))
            top3_fertilizers = top3_fertilizers.reshape(len(test_df), 3)

            # 创建提交文件
            submission = pd.DataFrame({
                'id': test_df['id'],
                'Fertilizer Name': [' '.join(row) for row in top3_fertilizers]
            })

            # 保存结果
            submission.to_csv(submission_path, index=False)
            logger.info(f"Submission file created successfully at {submission_path}")

    except NotFittedError as e:
        logger.error(f"Preprocessor not fitted: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating submission: {str(e)}")
        raise


def main():
    try:
        # 加载数据
        train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

        # 特征工程
        train_df = feature_engineering(train_df)
        test_df = feature_engineering(test_df)

        # 预处理数据
        X_train, X_val, y_train, y_val, preprocessor, label_encoder = preprocess_data(
            train_df, test_df
        )

        # 构建和训练模型
        model = build_model(X_train.shape[1], len(label_encoder.classes_))
        trained_model = train_model(model, X_train, y_train, X_val, y_val)

        # 生成提交文件
        create_submission(trained_model, test_df, preprocessor, label_encoder)

        logger.info("Process completed successfully!")

    except Exception as e:
        logger.exception(f"Process failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

