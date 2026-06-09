import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import os
import re
import joblib

# 设置字体
plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 数据加载（适配Kaggle路径）
def load_data():
    print("开始准备数据...")
    
    # Kaggle数据集路径（无需解压，直接读取）
    data_dir = r"/kaggle/input/playground-series-s5e6"
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    print(f"数据目录: {data_dir}")
    
    # 列出目录中的所有文件
    file_list = os.listdir(data_dir)
    print("\n目录内文件列表:")
    for file in file_list:
        print(file)
    
    # 定义数据文件名
    train_file = "train.csv"
    test_file = "test.csv"
    submission_file = "sample_submission.csv"
    
    # 构建文件路径（使用os.path.join确保路径分隔符正确）
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)
    submission_path = os.path.join(data_dir, submission_file) if submission_file in file_list else None
    
    # 验证文件存在性
    print("\n验证文件存在性:")
    print(f"训练集路径: {train_path}")
    print(f"训练集是否存在: {os.path.exists(train_path)}")
    print(f"测试集路径: {test_path}")
    print(f"测试集是否存在: {os.path.exists(test_path)}")
    
    if not os.path.exists(train_path):
        print("\n错误提示:")
        print(f"1. 请确认文件是否存在于目录: {data_dir}")
        print(f"2. 请确认文件名拼写正确: {train_file}")
        print(f"3. 请确认文件未被其他程序占用")
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试集文件不存在: {test_path}")
    
    # 打印实际读取路径
    print(f"\n实际读取路径:")
    print(f"训练集: {train_path}")
    print(f"测试集: {test_path}")
    print(f"提交示例: {submission_path if submission_path else '无'}")
    
    # 读取数据
    print(f"\n正在读取训练集: {train_path}")
    train = pd.read_csv(train_path)
    print(f"正在读取测试集: {test_path}")
    test = pd.read_csv(test_path)
    
    if submission_path:
        print(f"正在读取提交示例: {submission_path}")
        submission = pd.read_csv(submission_path)
    else:
        print("未找到提交示例文件，将自动生成")
        submission = pd.DataFrame({'id': test['id'], 'Fertilizer Name': ''})
    
    # 打印数据基本信息
    print("\n数据加载成功！")
    print("\n训练集基本信息：")
    train.info()
    print("\n训练集列名:")
    print(train.columns.tolist())
    
    # 识别目标列
    possible_target_cols = [col for col in train.columns if re.search(r'fertilizer|target|label', col.lower())]
    if 'Fertilizer Name' in train.columns:
        target_col = 'Fertilizer Name'
        print(f"已确认目标列为: {target_col}")
    elif possible_target_cols:
        target_col = possible_target_cols[0]
        print(f"自动识别目标列为: {target_col}")
        if target_col != 'Fertilizer Name':
            print(f"警告: 目标列名与预期不符，使用 {target_col} 代替 'Fertilizer Name'")
    else:
        print("\n警告: 无法自动识别目标列")
        target_col = input("请输入实际目标列名: ")
    
    if target_col not in train.columns:
        raise KeyError(f"列 '{target_col}' 不存在于训练集中")
    
    # Kaggle输出目录
    output_dir = "/kaggle/working"
    os.makedirs(output_dir, exist_ok=True)
    
    return train, test, submission, target_col, output_dir

# 2. 数据预处理与特征工程（保持不变）
def preprocess_data(train, test, target_col):
    print("\n正在进行数据预处理...")

    def preprocess(df):
        df = df.copy()
        numerical_cols = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Moisture']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
            else:
                print(f"警告: 列 {col} 不存在，跳过填充")

        categorical_cols = ['Soil Type', 'Crop Type']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                print(f"已对类别特征 {col} 进行编码")
            else:
                print(f"警告: 列 {col} 不存在，跳过编码")

        if 'Nitrogen' in df.columns and 'Phosphorous' in df.columns and 'Potassium' in df.columns:
            df['NPK_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + df['Potassium'] + 1e-8)
            df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-8)
            df['N_K_Ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-8)
            df['PK_Ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-8)

        if 'Temperature' in df.columns and 'Humidity' in df.columns:
            df['Temp_Humidity_Interaction'] = df['Temperature'] * df['Humidity']

        if 'Moisture' in df.columns:
            df['Moisture_Level'] = pd.cut(df['Moisture'], bins=5, labels=[1, 2, 3, 4, 5])

        return df

    train_processed = preprocess(train)
    test_processed = preprocess(test)

    print("\n预处理后的特征:")
    print(train_processed.columns.tolist())

    return train_processed, test_processed

# 3. MAP@5 评估指标（保持不变）
def map_at_k(y_true, y_pred_proba, k=5):
    map_scores = []
    for i in range(len(y_true)):
        probs = y_pred_proba[i]
        top_k_indices = np.argsort(probs)[::-1][:k]
        true_label = y_true[i]
        ap_score = 0.0
        num_hits = 0
        for j, pred_idx in enumerate(top_k_indices):
            if pred_idx == true_label:
                num_hits += 1
                ap_score += num_hits / (j + 1)
        if num_hits > 0:
            map_scores.append(ap_score / num_hits)
        else:
            map_scores.append(0.0)
    return np.mean(map_scores)

# 4. 训练与预测（保持不变）
def train_and_predict(train_processed, test_processed, target_col, data_dir):
    print("\n正在准备训练数据...")
    X = train_processed.drop(columns=['id', target_col])
    y = train_processed[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n{target_col} 编码映射:")
    for i, class_name in enumerate(le.classes_):
        print(f"{i}: {class_name}")
    print("\n预测结果的取值范围（训练集中出现的肥料类型）：")
    print(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print(f"\n训练集形状: {X_train.shape}")
    print(f"验证集形状: {X_val.shape}")

    print("\n开始训练模型...")
    model = LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=40,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=0.2,
        random_state=42
    )

    # 自定义训练进度打印
    from lightgbm.callback import _format_eval_result
    
    def print_evaluation(period=1, show_stdv=True):
        def callback(env):
            if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
                result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
                print(f"[Iteration {env.iteration + 1}] {result}")
        callback.order = 10
        return callback

    # 模型训练
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[
            early_stopping(100, verbose=True),
            print_evaluation(period=10)
        ]
    )

    # 验证集评估
    val_preds = model.predict(X_val)
    val_preds_proba = model.predict_proba(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    print(f"\n验证集准确率: {accuracy:.4f}")
    map5_score = map_at_k(y_val, val_preds_proba, k=5)
    print(f"验证集 MAP@5: {map5_score:.4f}")

    # 训练集损失
    train_preds_proba = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_preds_proba)
    print(f"训练集 multi_logloss: {train_loss:.4f}")
    print(f"验证集 multi_logloss: {model.best_score_['valid_0']['multi_logloss']:.4f}")


    # 测试集预测
    print("\n正在预测测试集数据...")
    test_features = test_processed.drop(columns=['id'])
    missing_test_cols = [col for col in X_train.columns if col not in test_features.columns]
    if missing_test_cols:
        print(f"\n警告: 测试集缺少以下特征: {missing_test_cols}，可能影响预测结果")

    test_preds_proba = model.predict_proba(test_features)
    top5_indices = np.argsort(test_preds_proba, axis=1)[:, -5:][:, ::-1]
    top5_fertilizers = []
    for row in top5_indices:
        top5_fertilizers.append(le.inverse_transform(row))

    # 打印部分预测结果
    num_samples_to_display = 20
    print(f"\n===== Top 5预测结果（显示前{num_samples_to_display}个样本） =====")
    for i in range(min(num_samples_to_display, len(top5_fertilizers))):
        sample_id = test_processed['id'].iloc[i]
        top5_classes = top5_fertilizers[i]
        probs = [test_preds_proba[i][idx] for idx in top5_indices[i]]
        probs = [f"{p:.4f}" for p in probs]
        result = ", ".join([f"{cls}({prob})" for cls, prob in zip(top5_classes, probs)])
        print(f"样本 {i+1} (id={sample_id}): {result}")

    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_processed['id'],
        'Fertilizer Name': [' '.join(row) for row in top5_fertilizers]
    })
    submission_path = os.path.join(data_dir, "submission_top5.csv")
    submission.to_csv(submission_path, index=False)
    print(f"\nTop 5预测的提交文件已生成: {submission_path}")

    # 保存模型和编码器
    model_path = os.path.join(data_dir, 'fertilizer_model.pkl')
    encoder_path = os.path.join(data_dir, 'label_encoder.pkl')
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    print(f"\n模型已保存到: {model_path}")
    print(f"标签编码器已保存到: {encoder_path}")

    return model, le, map5_score

# 5. 农学结果解释（保持不变）
def explain_results(model, X_train, le, map5_score):
    print("\n===== 农学结果解释 =====")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X_train.columns
        df_importance = pd.DataFrame({
            "特征": feature_names,
            "重要性": importances
        }).sort_values("重要性", ascending=False)

        print("\n特征重要性排名:")
        print(df_importance.head(10))

        top_features = df_importance['特征'].tolist()[:5]
        print("\n农学解释:")
        if 'Nitrogen' in top_features:
            print("- 氮含量(Nitrogen)是最重要的特征之一，说明氮肥是最常用的肥料类型。")
            print("  高氮含量的土壤可能需要较少的氮肥，而低氮土壤需要补充。")
        if 'Phosphorous' in top_features:
            print("- 磷含量(Phosphorous)重要性高，表明磷肥对作物生长至关重要，尤其是在早期阶段。")
        if 'Soil Type' in top_features:
            print("- 土壤类型(Soil Type)是关键因素，不同土壤保肥能力不同。例如：")
            print("  - 砂质土壤需要更多的有机肥和缓释肥；")
            print("  - 粘质土壤保肥能力强，但透气性差。")
        if 'Crop Type' in top_features:
            print("- 作物类型(Crop Type)影响显著，不同作物对肥料需求差异大。例如：")
            print("  - 叶菜类作物对氮肥需求较高；")
            print("  - 瓜果类作物需要更多的钾肥和磷肥。")
        if 'NPK_Ratio' in top_features:
            print("- NPK比例特征重要，说明合理的肥料配比比单一元素更重要。")

        print(f"\n模型性能: MAP@5 = {map5_score:.4f}")
        print(f"这意味着模型在Top5预测中平均有 {map5_score*100:.2f}% 的准确率。")
        print("对于农业应用，这个分数表明模型能够为大多数情况提供合理的肥料推荐。")

        print("\n改进建议:")
        print("1. 收集更多样化的训练数据，尤其是罕见的土壤-作物组合；")
        print("2. 考虑添加更多环境因素，如降雨量、光照时间等；")
        print("3. 尝试集成多个模型，提高预测稳定性；")
        print("4. 针对特定作物类型开发专用模型。")

# 6. 主函数（保持不变）
def main():
    # 加载数据
    train, test, submission, target_col, data_dir = load_data()
    
    # 预处理数据
    train_processed, test_processed = preprocess_data(train, test, target_col)
    
    # 训练模型并预测
    model, le, map5_score = train_and_predict(train_processed, test_processed, target_col, data_dir)
    
    # 结果解释
    explain_results(
        model, 
        train_processed.drop(columns=['id', target_col]), 
        le, 
        map5_score
    )
    
    print("\n全部处理完成！")

if __name__ == "__main__":
    main()

