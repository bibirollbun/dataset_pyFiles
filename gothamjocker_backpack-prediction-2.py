import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import os  # For checking device availability and cpu_count

##########
# Step 0: Data Import
##########
def import_data(train_file1="/kaggle/input/playground-series-s5e2/train.csv", train_file2="/kaggle/input/playground-series-s5e2/training_extra.csv", test_file="/kaggle/input/playground-series-s5e2/test.csv"):
    """Imports and returns the training and test datasets."""
    print("Step 0: 导入数据...")
    try:
        train_df1 = pd.read_csv(train_file1)
        train_df2 = pd.read_csv(train_file2)
        test_df = pd.read_csv(test_file)
    except FileNotFoundError as e:
        print(f"Error: File not found.  Check file paths: {e}")
        return None, None, None
    print("数据导入完成！\n")
    return train_df1, train_df2, test_df

##########
# Step 1: Data Cleaning
##########
def fill_missing_with_mode(df):
    """Fills missing values in the DataFrame with the mode of each column."""
    for col in df.columns:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    return df

def preprocess_data(train_df1, train_df2, test_df):
    """Preprocesses the data: fills missing values, concatenates training sets."""
    print("Step 1: 数据清洗...")
    if train_df1 is None or train_df2 is None or test_df is None:
        print("Error: Cannot preprocess. DataFrames are None.")
        return None, None
    train_df1 = fill_missing_with_mode(train_df1)
    train_df2 = fill_missing_with_mode(train_df2)
    test_df = fill_missing_with_mode(test_df)
    train_df = pd.concat([train_df1, train_df2], ignore_index=True)
    print("数据清洗完成！\n")
    return train_df, test_df

##########
# Step 2: Feature Engineering
##########
# Modified to include 'Weight Capacity (kg)' in the features for Random Forest
def encode_discrete_features(train_df, test_df):
    """
    对离散特征和数值特征进行预处理, 准备 Random Forest 的输入。
    """
    print("Step 2: 特征工程...")

    # Features for Random Forest
    rf_features = ['Brand', 'Material', 'Size', 'Compartments',
                   'Laptop Compartment', 'Waterproof', 'Weight Capacity (kg)']

    # Separate features and target variable
    X_train_df = train_df[rf_features]
    y_train_series = train_df['Price']
    X_test_df = test_df[rf_features]

   # Convert 'Weight Capacity (kg)' to numeric, handling errors
    X_train_df['Weight Capacity (kg)'] = pd.to_numeric(X_train_df['Weight Capacity (kg)'], errors='coerce').fillna(0)
    X_test_df['Weight Capacity (kg)'] = pd.to_numeric(X_test_df['Weight Capacity (kg)'], errors='coerce').fillna(0)


    # Identify categorical columns for one-hot encoding
    categorical_cols = [col for col in rf_features if X_train_df[col].dtype == 'object']

    # One-hot encode categorical features
    X_train_encoded_df = pd.get_dummies(X_train_df, columns=categorical_cols, dummy_na=False)
    X_test_encoded_df = pd.get_dummies(X_test_df, columns=categorical_cols, dummy_na=False)

    # Align columns (important for consistent feature sets)
    X_test_encoded_df = X_test_encoded_df.reindex(columns=X_train_encoded_df.columns, fill_value=0)

    print("训练集编码后形状：", X_train_encoded_df.shape)
    print("测试集编码后形状：", X_test_encoded_df.shape)
    print("特征工程完成！\n")
    return X_train_encoded_df, y_train_series, X_test_encoded_df

def convert_bool_to_int(df):
    """
    将布尔类型的列转换为 int 类型 (int32)。
    """
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(np.int32)
    return df

##########
# Step 3: Random Forest Model (Using sklearn + sklearnex + INC)
##########
def discretize_price(price):
    """
    将价格离散化为 Level。
    """
    levels = [(15 + 2*i, 15 + 2*(i+1)) for i in range((155 - 15) // 2)]
    for i, (lower, upper) in enumerate(levels):
        if lower <= price <= upper:
            return i + 1
    return len(levels)

def discretize_weight(weight):
    """
    将重量容量离散化为 Level_2。
    """
    levels = [(5 + i, 5 + (i + 1)) for i in range(30 - 5)]  # 5 to 30, interval of 1
    for i, (lower, upper) in enumerate(levels):
        if lower <= weight <= upper:
            return i + 1
    return len(levels)

def train_random_forest_parallel(X_train, y_train, n_estimators=100, random_state=42, n_splits=5):
    """
    使用并行处理和 sklearnex/标准 sklearn 训练随机森林。

    Args:
        X_train (pd.DataFrame): 训练特征。
        y_train (pd.Series): 训练目标变量 (Price).
        n_estimators (int): 树的数量。
        random_state (int): 随机种子。
        n_splits (int): 数据分割数，用于并行训练。

    Returns:
        list: 训练好的 RandomForestRegressor 模型列表。
    """
    print("Step 3: 训练随机森林模型 (并行)...")

    # Get the number of CPU cores
    n_jobs = os.cpu_count() or 1  # Use 1 if os.cpu_count() returns None

    # Split data for parallel training
    X_splits = np.array_split(X_train, n_splits)
    y_splits = np.array_split(y_train, n_splits)

    def train_subset(X_sub, y_sub):
        model = RandomForestRegressor(n_estimators=n_estimators // n_splits,  # Adjust estimators per split
                                      random_state=random_state,
                                      n_jobs=n_jobs)  # Use n_jobs here
        model.fit(X_sub, y_sub)
        return model

    # Use ThreadPoolExecutor for parallelism
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        models = list(executor.map(train_subset, X_splits, y_splits))

    print("随机森林模型训练完成 (并行)。")
    return models

def predict_random_forest_ensemble(models, X_test):
    """
    使用随机    使用随机森林模型集合进行预测 (平均预测值)。

    Args:
        models (list): RandomForestRegressor 模型列表。
        X_test (pd.DataFrame): 测试特征。

    Returns:
        np.ndarray: 预测的平均价格。
    """
    print("Step 3: 使用随机森林模型集合进行预测...")
    predictions = np.mean([model.predict(X_test) for model in models], axis=0)
    print("预测完成。")
    return predictions

def evaluate_random_forest_parallel(models, X_train, y_train, n_splits=5):
    """
        评估随机森林模型 (RMSE on discretized levels)，并打印每个切片的预测结果。

        Args:
            models (list): 训练好的 RandomForestRegressor 模型列表。
            X_train (pd.DataFrame): 训练特征数据（用于分割和评估）。
            y_train (pd.Series): 训练目标变量（用于分割和评估）。
            n_splits (int): 数据分割数，用于评估。
    """
    print("Step 3: 评估随机森林 (并行)...")

    # Split data for evaluation
    X_splits = np.array_split(X_train, n_splits)
    y_splits = np.array_split(y_train, n_splits)


    for i, (X_test_fold, y_true_fold) in enumerate(zip(X_splits, y_splits)):
        # Predict using the ensemble
        y_pred_fold = predict_random_forest_ensemble(models, X_test_fold)

        # Discretize true and predicted prices
        y_true_levels = y_true_fold.apply(discretize_price)
        y_pred_levels = np.array([discretize_price(p) for p in y_pred_fold])

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true_levels, y_pred_levels))

        # Output results for this fold
        print(f"切片 {i+1} 训练完成，样本数: {len(X_test_fold)}，RMSE: {rmse:.4f}")
        df_compare = pd.DataFrame({
            "Actual Level": y_true_levels.values[:5],
            "Predicted Level": y_pred_levels[:5],
            "Level Residual": y_true_levels.values[:5] - y_pred_levels[:5]
        })
        print(f"切片 {i+1} 预测 vs 真实前5行：")
        print(df_compare)


##########
# Step 4:  Nonlinear Model
##########
def train_nonlinear_model(train_data, trained_forest, X_train_discrete_data):
    """
    训练非线性模型。
    """
    print("开始训练非线性模型...")
    train_with_levels = train_data.copy()
    train_with_levels['Level'] = train_with_levels['Price'].apply(discretize_price)
    train_with_levels['Level_2'] = train_with_levels['Weight Capacity (kg)'].apply(discretize_weight)

    levels = [(15 + 2*i, 15 + 2*(i+1)) for i in range((155 - 15) // 2)]
    levels_2 = [(5 + i, 5 + (i + 1)) for i in range(30 - 5)]


    print("  准备非线性模型的输入特征...")
    # Convert necessary columns to numeric and handle missing values
    train_with_levels['wcap'] = pd.to_numeric(train_with_levels['Weight Capacity (kg)'], errors='coerce').fillna(0)
    train_with_levels['size'] = train_with_levels['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3}).fillna(0)
    train_with_levels['comp'] = pd.to_numeric(train_with_levels['Compartments'], errors='coerce').fillna(0)
    train_with_levels['lap_comp'] = train_with_levels['Laptop Compartment'].map({'No': 0, 'Yes': 1}).fillna(0)
    train_with_levels['waterproof'] = train_with_levels['Waterproof'].map({'No': 0, 'Yes': 1}).fillna(0)
    train_with_levels['brand'] = train_with_levels['Brand'].astype('category').cat.codes
    train_with_levels['style'] = train_with_levels['Style'].astype('category').cat.codes
    train_with_levels['color'] = train_with_levels['Color'].astype('category').cat.codes

    def calculate_level_input(row, w1, w2, w3, w4, w5, w6, w7):
      level_index = int(row['Level']) - 1
      level_2_index = int(row['Level_2']) - 1

      # Basic level inputs (price and weight)
      lower_limit_price = levels[level_index][0] if 0 <= level_index < len(levels) else 0
      lower_limit_weight = levels_2[level_2_index][0] if 0 <= level_2_index < len(levels_2) else 0

      # Combined level input incorporating various features
      return (lower_limit_price + lower_limit_weight +
              w1 * row['size'] +
              w2 * row['comp'] +
              w3 * row['lap_comp'] +
              w4 * row['waterproof'] +
              w5 * row['brand'] +
              w6 * row['style'] +
              w7 * row['color'])

    # Define coefficients
    w1, w2, w3, w4, w5, w6, w7 = (0.1, 0.2, 0.05, 0.1, 0.05, 0.03, 0.02)
    train_with_levels['level_input'] = train_with_levels.apply(calculate_level_input, axis=1, args=(w1, w2, w3, w4, w5, w6, w7))
    print("  输入特征准备完成。")

    print("  构建设计矩阵 X 和目标向量 y...")
    X = train_with_levels[['size', 'comp', 'lap_comp', 'waterproof', 'wcap', 'level_input','brand','style','color']].values
    y = train_with_levels['Price'].values

    print("  训练多项式回归模型...")
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    print("非线性模型训练完成。")
    return model, poly

##########
# Step 5: Prediction and Output
##########
def predict_with_nonlinear_model(test_data, X_test_discrete_data, trained_nonlinear_model, poly_transform, trained_forest):
    """
    使用非线性模型进行预测。
    """
    print("开始使用非线性模型进行预测...")
    test_with_levels = test_data.copy()


    # Predict Levels using Random Forest (discretized price and weight)
    print("  使用随机森林预测 Level (先预测连续价格，再离散化)...")
    price_predictions = predict_random_forest_ensemble(trained_forest, X_test_discrete_data)
    test_with_levels['Level'] = [discretize_price(p) for p in price_predictions]
    print("  Level 预测完成。")
    print("  使用随机森林预测 Level_2 (先预测连续重量，再离散化)...")
    # Need to predict 'Weight Capacity (kg)' using Random Forest as well
    weight_predictions = predict_random_forest_ensemble(trained_forest, X_test_discrete_data)
    test_with_levels['Level_2'] = [discretize_weight(w) for w in weight_predictions]
    print("  Level_2 预测完成")

    levels = [(15 + 2*i, 15 + 2*(i+1)) for i in range((155 - 15) // 2)]
    levels_2 = [(5 + i, 5 + (i + 1)) for i in range(30 - 5)] # Weight levels

    print("  准备非线性模型的输入特征...")
    # Convert and map just like in training
    test_with_levels['wcap'] = pd.to_numeric(test_with_levels['Weight Capacity (kg)'], errors='coerce').fillna(0)
    test_with_levels['size'] = test_with_levels['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3}).fillna(0)
    test_with_levels['comp'] = pd.to_numeric(test_with_levels['Compartments'], errors='coerce').fillna(0)
    test_with_levels['lap_comp'] = test_with_levels['Laptop Compartment'].map({'No': 0, 'Yes': 1}).fillna(0)
    test_with_levels['waterproof'] = test_with_levels['Waterproof'].map({'No': 0, 'Yes': 1}).fillna(0)
    test_with_levels['brand'] = test_with_levels['Brand'].astype('category').cat.codes
    test_with_levels['style'] = test_with_levels['Style'].astype('category').cat.codes
    test_with_levels['color'] = test_with_levels['Color'].astype('category').cat.codes


    # Use the SAME calculate_level_input function (important for consistency)
    def calculate_level_input(row, w1, w2, w3, w4, w5, w6, w7):
      level_index = int(row['Level']) - 1
      level_2_index = int(row['Level_2']) - 1

      lower_limit_price = levels[level_index][0] if 0 <= level_index < len(levels) else 0
      lower_limit_weight = levels_2[level_2_index][0] if 0 <= level_2_index < len(levels_2) else 0

      return (lower_limit_price + lower_limit_weight +
              w1 * row['size'] +
              w2 * row['comp'] +
              w3 * row['lap_comp'] +
              w4 * row['waterproof'] +
              w5 * row['brand'] +
              w6 * row['style'] +
              w7 * row['color'])


    w1, w2, w3, w4, w5, w6, w7 = (0.1, 0.2, 0.05, 0.1, 0.05, 0.03, 0.02)  # Same coefficients as training
    test_with_levels['level_input'] = test_with_levels.apply(calculate_level_input, axis=1, args=(w1, w2, w3, w4, w5, w6, w7))
    print("  输入特征准备完成。")

    print("  构建设计矩阵...")
    X_test = test_with_levels[['size', 'comp', 'lap_comp', 'waterproof', 'wcap', 'level_input','brand','style','color']].values
    X_test_poly = poly_transform.transform(X_test)  # Use the trained poly object
    print("  设计矩阵构建完成。")

    print("  进行价格预测...")
    final_preds = trained_nonlinear_model.predict(X_test_poly)
    print("  预测完成。")

    return final_preds

def output_results(test_data, final_predictions):
    """
    输出预测结果到 CSV 文件。
    """
    test_ids = test_data.iloc[:, 0]
    output = pd.DataFrame({
        "id": test_ids,
        "Price": final_predictions
    })
    print("预测结果示例：")
    print(output.head())
    output.to_csv("/kaggle/working/submission.csv", index=False)
    print("预测结果已保存至 submission.csv")

# --- Main Execution Block ---
if __name__ == "__main__":
    train1, train2, test = import_data()
    if train1 is not None and train2 is not None and test is not None:
        train, test = preprocess_data(train1, train2, test)
        X_train_discrete, y_train, X_test_discrete = encode_discrete_features(train, test)
        X_train_discrete = convert_bool_to_int(X_train_discrete)
        X_test_discrete = convert_bool_to_int(X_test_discrete)

        # --- Random Forest (Parallel Training) ---
        n_splits = 10  # Number of splits for parallel training.  Adjust as needed.
        trained_forest = train_random_forest_parallel(X_train_discrete, y_train, n_splits=n_splits)
        evaluate_random_forest_parallel(trained_forest, X_train_discrete, y_train, n_splits=n_splits)

        # --- Nonlinear Model ---
        nonlinear_model, poly = train_nonlinear_model(train, trained_forest, X_train_discrete)
        final_preds = predict_with_nonlinear_model(test, X_test_discrete, nonlinear_model, poly, trained_forest)
        output_results(test, final_preds)

