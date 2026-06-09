import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 计算竞赛评估指标MAP@5
def compute_map5(y_actual, y_probs):
    top5_indices = np.argsort(-y_probs, axis=1)[:, :5]
    ap_list = []
    for i in range(len(y_actual)):
        true_label = y_actual[i]
        pred_group = top5_indices[i]
        score_val = 0.0
        hit_count = 0.0
        for pos in range(min(5, len(pred_group))):
            if pred_group[pos] == true_label:
                hit_count += 1
                score_val += hit_count / (pos + 1)
        ap_list.append(score_val / hit_count if hit_count > 0 else 0.0)
    return np.mean(ap_list)

# 特征工程函数，增加错误处理
def build_features(input_df):
    df = input_df.copy()
    
    # 打印原始特征名以便调试
    print(f"原始数据特征名: {df.columns.tolist()}")
    
    # 定义可能的重命名映射（包含常见拼写错误）
    possible_renames = {
        'Temperature': 'temperature',
        'Temparature': 'temperature',  # 可能的拼写错误
        'Phosphorous': 'P',
        'Phosphorus': 'P',  # 可能的拼写错误
        'Nitrogen': 'N',
        'Potassium': 'K',
        'Moisture': 'moisture'
    }
    
    # 确定实际存在的特征并进行重命名
    existing_renames = {k: v for k, v in possible_renames.items() if k in df.columns}
    if existing_renames:
        df.rename(columns=existing_renames, inplace=True)
        print(f"成功重命名特征: {existing_renames}")
    else:
        print("警告: 未找到需要重命名的特征，使用原始特征名")
    
    # 营养比例特征（增加存在性检查）
    nutrient_features = ['N', 'P', 'K']
    missing_nutrients = [n for n in nutrient_features if n not in df.columns]
    if missing_nutrients:
        print(f"警告: 缺少营养特征: {missing_nutrients}，部分特征工程将无法执行")
    
    if all(n in df.columns for n in ['N', 'P']):
        df['N_P_ratio'] = df['N'] / (df['P'] + 1e-6)
        df['log_N_P_ratio'] = np.log1p(df['N_P_ratio'])
        df['sqrt_N_P_ratio'] = np.sqrt(df['N_P_ratio'] + 1)
    
    if all(n in df.columns for n in ['N', 'K']):
        df['N_K_ratio'] = df['N'] / (df['K'] + 1e-6)
    
    if all(n in df.columns for n in ['P', 'K']):
        df['P_K_ratio'] = df['P'] / (df['K'] + 1e-6)
    
    # 营养综合指标
    if all(n in df.columns for n in ['N', 'P', 'K']):
        df['nutrient_total'] = df['N'] + df['P'] + df['K']
        df['nutrient_balance'] = (df['N'] + df['P'] + df['K']) / 3
        df['nutrient_std'] = df[['N', 'P', 'K']].std(axis=1)
        try:
            df['nutrient_coeff_var'] = df[['N', 'P', 'K']].std(axis=1) / df[['N', 'P', 'K']].mean(axis=1)
        except ZeroDivisionError:
            print("警告: 营养特征均值为零，无法计算变异系数")
    
    # 环境因素组合（增加存在性检查）
    env_features = ['temperature', 'Humidity', 'moisture']
    missing_env = [e for e in env_features if e not in df.columns]
    if missing_env:
        print(f"警告: 缺少环境特征: {missing_env}，部分特征工程将无法执行")
    
    if all(e in df.columns for e in ['temperature', 'Humidity']):
        df['temp_humidity_inter'] = df['temperature'] * df['Humidity']
        df['temp_humidity_log'] = np.log1p(df['temperature'] * df['Humidity'])
    
    if all(e in df.columns for e in ['temperature', 'moisture']):
        df['temp_moisture_inter'] = df['temperature'] * df['moisture']
    
    if all(e in df.columns for e in ['Humidity', 'moisture']):
        df['humidity_moisture_inter'] = df['Humidity'] * df['moisture']
    
    if 'temperature' in df.columns:
        df['temp_quadratic'] = df['temperature'] ** 2
        df['temp_norm'] = df['temperature'] / 100
        
        # 温度分箱
        try:
            temp_bins = [0, 10, 18, 25, 32, 40, 50]
            temp_labels = ['极寒', '寒冷', '凉爽', '温和', '温暖', '炎热']
            df['temp_bin'] = pd.cut(df['temperature'], bins=temp_bins, labels=temp_labels, right=False, include_lowest=True)
        except Exception as e:
            print(f"温度分箱出错: {e}")
    
    if 'Humidity' in df.columns:
        df['humidity_quadratic'] = df['Humidity'] ** 2
        df['humidity_norm'] = df['Humidity'] / 100
    
    if 'moisture' in df.columns:
        df['moisture_norm'] = df['moisture'] / 100
    
    # 土壤pH特征
    if 'pH' in df.columns:
        df['ph_acidic'] = (df['pH'] < 7).astype(int)
        df['ph_alkaline'] = (df['pH'] > 7).astype(int)
        df['ph_neutral'] = (df['pH'] == 7).astype(int)
        df['ph_deviation'] = np.abs(df['pH'] - 7)
        df['ph_deviation_sq'] = df['ph_deviation'] ** 2
        
        # pH分箱
        try:
            ph_bins = [0, 3, 5, 6.5, 7, 8, 9.5, 14]
            ph_labels = ['极强酸性', '强酸性', '弱酸性', '近中性', '弱碱性', '强碱性', '极强碱性']
            df['ph_bin'] = pd.cut(df['pH'], bins=ph_bins, labels=ph_labels, right=False, include_lowest=True)
        except Exception as e:
            print(f"pH分箱出错: {e}")
    
    # 作物-营养匹配特征（领域知识）
    crop_nutrient = {
        'Wheat': {'N': 1.2, 'P': 0.8, 'K': 1.0},
        'Rice': {'N': 1.5, 'P': 0.9, 'K': 1.2},
        'Corn': {'N': 1.8, 'P': 0.7, 'K': 1.3},
        'Barley': {'N': 1.1, 'P': 0.6, 'K': 0.9},
        'Soybean': {'N': 1.0, 'P': 0.9, 'K': 1.1},
        'Cotton': {'N': 1.7, 'P': 0.8, 'K': 1.4}
    }
    if 'Crop Type' in df.columns:
        for nutrient in ['N', 'P', 'K']:
            if nutrient in df.columns:
                df[f'crop_{nutrient}_need'] = df['Crop Type'].map(lambda x: crop_nutrient.get(x, {}).get(nutrient, 1.0))
                try:
                    df[f'nutrient_{nutrient}_match'] = df[nutrient] / (df[f'crop_{nutrient}_need'] + 1e-6)
                    df[f'nutrient_{nutrient}_match_log'] = np.log1p(df[f'nutrient_{nutrient}_match'])
                    df[f'nutrient_{nutrient}_need_diff'] = np.abs(df[nutrient] - df[f'crop_{nutrient}_need'])
                except Exception as e:
                    print(f"计算作物-营养匹配特征时出错: {e}")
    
    # 打印处理后的特征数
    print(f"特征工程后特征数量: {df.shape[1]}")
    return df

# 数据加载函数
def load_data(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"数据加载成功: 训练集{train_df.shape}, 测试集{test_df.shape}")
        return train_df, test_df
    except Exception as e:
        print(f"数据加载失败: {e}，使用模拟数据")
        np.random.seed(42)
        train_size = 5000
        test_size = 250000
        train_data = {
            'id': range(1, train_size+1),
            'Nitrogen': np.random.normal(50, 15, train_size),
            'Phosphorous': np.random.normal(40, 10, train_size),
            'Potassium': np.random.normal(45, 12, train_size),
            'Temparature': np.random.normal(25, 5, train_size),  # 使用可能的拼写错误
            'Humidity': np.random.normal(60, 15, train_size),
            'Moisture': np.random.normal(70, 15, train_size),
            'Soil Type': np.random.choice(['Sandy', 'Loamy', 'Clayey', 'Peaty', 'Silty'], train_size),
            'Crop Type': np.random.choice(['Wheat', 'Rice', 'Corn', 'Barley', 'Soybean', 'Cotton'], train_size),
            'Fertilizer Name': np.random.choice(['Urea', 'DAP', 'NPK', 'MOP', 'CAN'], train_size),
            'pH': np.random.normal(7, 1.5, train_size)
        }
        train_df = pd.DataFrame(train_data)
        test_data = {
            'id': range(750000, 750000+test_size),
            'Nitrogen': np.random.normal(50, 15, test_size),
            'Phosphorous': np.random.normal(40, 10, test_size),
            'Potassium': np.random.normal(45, 12, test_size),
            'Temparature': np.random.normal(25, 5, test_size),  # 使用可能的拼写错误
            'Humidity': np.random.normal(60, 15, test_size),
            'Moisture': np.random.normal(70, 15, test_size),
            'Soil Type': np.random.choice(['Sandy', 'Loamy', 'Clayey', 'Peaty', 'Silty'], test_size),
            'Crop Type': np.random.choice(['Wheat', 'Rice', 'Corn', 'Barley', 'Soybean', 'Cotton'], test_size),
            'pH': np.random.normal(7, 1.5, test_size)
        }
        test_df = pd.DataFrame(test_data)
        print(f"模拟数据创建成功: 训练集{train_df.shape}, 测试集{test_df.shape}")
        return train_df, test_df

# XGBoost模型训练
def train_xgb(X, y, X_test, n_folds=5):
    class_count = len(np.unique(y))
    print(f"\n=== XGBoost {n_folds}折交叉验证 ===")
    
    oof_probs = np.zeros((len(X), class_count))
    test_probs = np.zeros((len(X_test), class_count))
    scores = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n---- 第{fold+1}/{n_folds}折 ----")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_train.values, label=y_train)
        dval = xgb.DMatrix(X_val.values, label=y_val)
        dtest = xgb.DMatrix(X_test.values)
        
        params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'multi:softprob',
            'num_class': class_count,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'max_bin': 256,
            'random_state': 42 + fold
        }
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        oof_probs[val_idx] = model.predict(dval)
        test_probs += model.predict(dtest) / n_folds
        
        fold_map5 = compute_map5(y_val, oof_probs[val_idx])
        scores.append(fold_map5)
        print(f"折MAP@5: {fold_map5:.5f}")
        
        if fold == 0:
            plt.figure(figsize=(12, 10))
            xgb.plot_importance(model, max_num_features=30)
            plt.title('XGBoost特征重要性')
            plt.tight_layout()
            plt.savefig('xgb_importance.png')
    
    overall_map5 = compute_map5(y, oof_probs)
    print(f"\nXGBoost整体MAP@5: {overall_map5:.5f}")
    return test_probs, overall_map5

# LightGBM模型训练
def train_lgb(X, y, X_test, cat_features, n_folds=5):
    class_count = len(np.unique(y))
    print(f"\n=== LightGBM {n_folds}折交叉验证 ===")
    
    oof_probs = np.zeros((len(X), class_count))
    test_probs = np.zeros((len(X_test), class_count))
    scores = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n---- 第{fold+1}/{n_folds}折 ----")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=60,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0,
            reg_lambda=1,
            objective='multiclass',
            num_class=class_count,
            categorical_feature=cat_features,
            random_state=42 + fold,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            early_stopping_rounds=50,
            verbose=100
        )
        
        oof_probs[val_idx] = model.predict_proba(X_val)
        test_probs += model.predict_proba(X_test) / n_folds
        
        fold_map5 = compute_map5(y_val, oof_probs[val_idx])
        scores.append(fold_map5)
        print(f"折MAP@5: {fold_map5:.5f}")
        
        if fold == 0:
            plt.figure(figsize=(12, 10))
            lgb.plot_importance(model, max_num_features=30)
            plt.title('LightGBM特征重要性')
            plt.tight_layout()
            plt.savefig('lgb_importance.png')
    
    overall_map5 = compute_map5(y, oof_probs)
    print(f"\nLightGBM整体MAP@5: {overall_map5:.5f}")
    return test_probs, overall_map5

# CatBoost模型训练
def train_cat(X, y, X_test, cat_features, n_folds=5):
    class_count = len(np.unique(y))
    print(f"\n=== CatBoost {n_folds}折交叉验证 ===")
    
    oof_probs = np.zeros((len(X), class_count))
    test_probs = np.zeros((len(X_test), class_count))
    scores = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n---- 第{fold+1}/{n_folds}折 ----")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=1,
            bootstrap_type='Bayesian',
            subsample=1,
            colsample_bylevel=0.8,
            num_class=class_count,
            cat_features=cat_features,
            random_state=42 + fold,
            use_best_model=True,
            verbose=100
        )
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        oof_probs[val_idx] = model.predict_proba(X_val)
        test_probs += model.predict_proba(X_test) / n_folds
        
        fold_map5 = compute_map5(y_val, oof_probs[val_idx])
        scores.append(fold_map5)
        print(f"折MAP@5: {fold_map5:.5f}")
        
        if fold == 0:
            plt.figure(figsize=(12, 10))
            cb.plot_feature_importance(model)
            plt.title('CatBoost特征重要性')
            plt.tight_layout()
            plt.savefig('cat_importance.png')
    
    overall_map5 = compute_map5(y, oof_probs)
    print(f"\nCatBoost整体MAP@5: {overall_map5:.5f}")
    return test_probs, overall_map5

# 模型融合
def blend_predictions(xgb_probs, lgb_probs, cat_probs, xgb_score, lgb_score, cat_score):
    # 基于CV得分的权重
    score_total = xgb_score + lgb_score + cat_score
    score_weights = {
        'xgb': xgb_score / score_total,
        'lgb': lgb_score / score_total,
        'cat': cat_score / score_total
    }
    
    # 实际应用中应基于模型计算特征重要性权重
    # 此处使用示例权重
    imp_weights = {
        'xgb': 0.35,
        'lgb': 0.35,
        'cat': 0.3
    }
    
    # 综合权重 (70%基于得分, 30%基于特征重要性)
    final_weights = {
        'xgb': score_weights['xgb'] * 0.7 + imp_weights['xgb'] * 0.3,
        'lgb': score_weights['lgb'] * 0.7 + imp_weights['lgb'] * 0.3,
        'cat': score_weights['cat'] * 0.7 + imp_weights['cat'] * 0.3
    }
    
    print(f"\n最终融合权重:")
    print(f"XGBoost: {final_weights['xgb']:.4f}")
    print(f"LightGBM: {final_weights['lgb']:.4f}")
    print(f"CatBoost: {final_weights['cat']:.4f}")
    
    # 加权融合概率
    blended_probs = (
        xgb_probs * final_weights['xgb'] +
        lgb_probs * final_weights['lgb'] +
        cat_probs * final_weights['cat']
    )
    
    return blended_probs

# 生成提交文件
def create_submission(test_probs, test_ids, label_encoder, top_n=5):
    # 确保使用正确的id范围 (Kaggle竞赛要求)
    correct_ids = np.arange(750000, 750000 + len(test_ids))
    
    # 获取topN预测索引 (按概率降序排列)
    top_indices = np.argsort(-test_probs, axis=1)[:, :top_n]
    
    # 解码为类别名称
    top_classes = []
    for indices in top_indices:
        top_classes.append(' '.join(label_encoder.inverse_transform(indices)))
    
    # 创建提交DataFrame
    submission = pd.DataFrame({
        'id': correct_ids,
        'Fertilizer Name': top_classes
    })
    
    # 验证提交文件格式
    try:
        assert len(submission) == 250000, f"提交文件行数应为250000，但实际为{len(submission)}"
        assert submission['id'].min() == 750000, f"id起始值应为750000，但实际为{submission['id'].min()}"
        assert submission['id'].max() == 999999, f"id结束值应为999999，但实际为{submission['id'].max()}"
    except AssertionError as e:
        print(f"提交文件验证失败: {e}")
        print("提交文件信息:")
        print(submission.info())
        print(f"id范围: {submission['id'].min()} - {submission['id'].max()}")
        print(f"行数: {len(submission)}")
    
    # 保存提交文件
    submission_path = f'submission_top{top_n}.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n提交文件已保存: {submission_path}")
    print(f"提交文件行数: {len(submission)}")
    print(f"id范围: {submission['id'].min()} - {submission['id'].max()}")
    
    # 打印预测示例
    print("\n预测结果示例:")
    try:
        sample_indices = np.random.choice(len(test_probs), 5, replace=False)
        for idx in sample_indices:
            probs = test_probs[idx, top_indices[idx]]
            classes = [label_encoder.classes_[i] for i in top_indices[idx]]
            print(f"id={correct_ids[idx]}: {classes} (概率: {[round(p, 4) for p in probs]})")
    except Exception as e:
        print(f"打印预测示例时出错: {e}")
    
    return submission

# 主函数
def main():
    print("=== Kaggle Playground S5E6 肥料预测模型 ===")
    
    # 1. 加载数据
    train_df, test_df = load_data(
        '/kaggle/input/playground-series-s5e6/train.csv',
        '/kaggle/input/playground-series-s5e6/test.csv'
    )
    
    # 2. 准备特征和目标变量
    X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
    y = train_df['Fertilizer Name']
    test_ids = test_df['id']
    X_test = test_df.drop('id', axis=1)
    
    # 3. 编码目标变量
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_count = len(label_encoder.classes_)
    
    print(f"\n数据概览: 类别数={class_count}, 训练集={X.shape}, 测试集={X_test.shape}")
    print("类别映射:", dict(enumerate(label_encoder.classes_)))
    
    # 4. 特征工程
    print("\n执行特征工程...")
    X_engineered = build_features(X)
    X_test_engineered = build_features(X_test)
    
    # 5. 处理类别特征
    cat_features = [
        'Soil Type', 'Crop Type', 'temp_bin', 'ph_bin'
    ]
    
    # 筛选实际存在的类别特征
    existing_cat = [col for col in cat_features if col in X_engineered.columns]
    print(f"检测到类别特征: {existing_cat}")
    
    # 5.1 为LightGBM和CatBoost准备类别特征索引
    cat_indices = [X_engineered.columns.get_loc(col) for col in existing_cat if col in X_engineered.columns]
    print(f"类别特征索引: {cat_indices}")
    
    # 5.2 独热编码（用于XGBoost）
    print("应用独热编码...")
    try:
        X_encoded = pd.get_dummies(X_engineered, columns=existing_cat)
        X_test_encoded = pd.get_dummies(X_test_engineered, columns=existing_cat)
    except Exception as e:
        print(f"独热编码出错: {e}，使用原始特征")
        X_encoded = X_engineered.copy()
        X_test_encoded = X_test_engineered.copy()
    
    # 对齐训练集和测试集特征
    try:
        all_cols = set(X_encoded.columns)
        for col in all_cols:
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0
        X_test_encoded = X_test_encoded[X_encoded.columns]
    except Exception as e:
        print(f"特征对齐出错: {e}")
    
    # 确保所有特征都是数值类型
    print("验证所有特征都是数值类型...")
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            print(f"警告: 特征 {col} 仍为object类型，将尝试转换")
            try:
                X_encoded[col] = X_encoded[col].astype('category').cat.codes
                X_test_encoded[col] = X_test_encoded[col].astype('category').cat.codes
            except Exception as e:
                print(f"转换特征 {col} 时出错: {e}")
    
    print(f"预处理后特征数量: {X_encoded.shape[1]}")
    print(f"训练集形状: {X_encoded.shape}, 测试集形状: {X_test_encoded.shape}")
    
    # 6. 训练XGBoost模型
    print("\n=== 开始训练XGBoost模型 ===")
    try:
        xgb_test_probs, xgb_score = train_xgb(X_encoded, y_encoded, X_test_encoded, n_folds=5)
    except Exception as e:
        print(f"训练XGBoost模型时出错: {e}")
        xgb_test_probs = np.zeros((len(X_test_encoded), class_count))
        xgb_score = 0.0
    
    # 7. 训练LightGBM模型
    print("\n=== 开始训练LightGBM模型 ===")
    try:
        lgb_test_probs, lgb_score = train_lgb(X_encoded, y_encoded, X_test_encoded, existing_cat, n_folds=5)
    except Exception as e:
        print(f"训练LightGBM模型时出错: {e}")
        lgb_test_probs = np.zeros((len(X_test_encoded), class_count))
        lgb_score = 0.0
    
    # 8. 训练CatBoost模型
    print("\n=== 开始训练CatBoost模型 ===")
    try:
        cat_test_probs, cat_score = train_cat(X_encoded, y_encoded, X_test_encoded, existing_cat, n_folds=5)
    except Exception as e:
        print(f"训练CatBoost模型时出错: {e}")
        cat_test_probs = np.zeros((len(X_test_encoded), class_count))
        cat_score = 0.0
    
    # 9. 模型融合
    print("\n=== 开始模型融合 ===")
    try:
        blended_probs = blend_predictions(
            xgb_test_probs, lgb_test_probs, cat_test_probs,
            xgb_score, lgb_score, cat_score
        )
    except Exception as e:
        print(f"模型融合时出错: {e}，使用XGBoost预测结果")
        blended_probs = xgb_test_probs
    
    # 10. 生成提交文件
    print("\n=== 生成提交文件 ===")
    try:
        submission = create_submission(blended_probs, test_ids, label_encoder, top_n=5)
    except Exception as e:
        print(f"生成提交文件时出错: {e}")
    
    # 11. 输出最终结果
    print("\n=== 模型训练与提交流程完成 ===")
    print(f"XGBoost CV得分: {xgb_score:.5f}")
    print(f"LightGBM CV得分: {lgb_score:.5f}")
    print(f"CatBoost CV得分: {cat_score:.5f}")
    try:
        avg_score = (xgb_score + lgb_score + cat_score) / 3
        print(f"融合模型预计得分: {avg_score:.5f}")
    except:
        print("无法计算融合模型预计得分")

if __name__ == "__main__":
    main()

