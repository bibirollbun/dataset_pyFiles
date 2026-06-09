# 学号: 2024423310124, 姓名: 孙瑞茜
#代码文件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import warnings
import re

warnings.filterwarnings('ignore')


# 自定义MAP@5评估函数
def map5_score(y_true, y_pred):
    U = len(y_true)
    map5_sum = 0.0
    top5_preds = np.argsort(y_pred, axis=1)[:, -5:][:, ::-1]
    for i in range(U):
        correct_label = y_true[i]
        ap = 0.0
        correct_count = 0
        for k in range(min(5, len(y_pred[i]))):
            if top5_preds[i, k] == correct_label:
                correct_count += 1
                precision_at_k = correct_count / (k + 1)
                ap += precision_at_k
        if correct_count > 0:
            ap /= correct_count
        map5_sum += ap
    return map5_sum / U


# 肥料名称到NPK格式的转换函数
def convert_to_npk(fertilizer_name):
    fertilizer_to_npk = {
        'Urea': '46-0-0', 'DAP': '18-46-0', 'MOP': '0-0-60', 'SSP': '0-16-0',
        'TSP': '0-46-0', 'Ammonium Sulphate': '21-0-0', '10-26-26': '10-26-26',
        '14-35-14': '14-35-14', '17-17-17': '17-17-17', '20-20': '20-20-0',
        '28-28': '28-28-0', '14-28-14': '14-28-14', '19-19-19': '19-19-19', '20-10-10': '20-10-10'
    }
    if fertilizer_name in fertilizer_to_npk:
        return fertilizer_to_npk[fertilizer_name]
    if re.match(r'^\d+-\d+$', fertilizer_name):
        parts = fertilizer_name.split('-')
        return f"{parts[0]}-{parts[1]}-0"
    elif re.match(r'^\d+$', fertilizer_name):
        return f"{fertilizer_name}-0-0"
    elif re.match(r'^\d+-\d+-\d+$', fertilizer_name):
        return fertilizer_name
    warnings.warn(f"无法识别的肥料名称格式: {fertilizer_name}")
    return fertilizer_name


# 数据加载和预处理
def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    
    # 缺失值处理
    if train.isnull().sum().sum() > 0:
        train = train.fillna(train.mean())
    if test.isnull().sum().sum() > 0:
        test = test.fillna(test.mean())

    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train, test], axis=0, ignore_index=True)

    # 特征工程
    all_data['N_P_ratio'] = all_data['Nitrogen'] / (all_data['Phosphorous'] + 1e-5)
    all_data['nutrient_index'] = all_data['Nitrogen'] * 0.4 + all_data['Phosphorous'] * 0.3 + all_data['Potassium'] * 0.3
    all_data['temp_humidity'] = all_data['Temparature'] * all_data['Humidity']
    all_data['humidity_moisture'] = all_data['Humidity'] * all_data['Moisture']
    
    # 简化分组特征
    all_data['temp_group'] = pd.cut(all_data['Temparature'], bins=[-10, 20, 40, 50],
                                    labels=['cold', 'moderate', 'hot'])
    all_data['humidity_group'] = pd.cut(all_data['Humidity'], bins=[0, 40, 80, 100],
                                        labels=['dry', 'moderate', 'humid'])

    train = all_data[all_data['is_train'] == 1].drop('is_train', axis=1)
    test = all_data[all_data['is_train'] == 0].drop('is_train', axis=1)
    return train, test


# 特征分析和可视化
def analyze_features(train):
    num_features = ['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(num_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(train[feature], kde=True)
        plt.title(f'{feature} Distribution')
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=train, y='Fertilizer Name', order=train['Fertilizer Name'].value_counts().index)
    plt.title('Fertilizer Type Distribution')
    plt.tight_layout()
    plt.savefig('fertilizer_distribution.png')
    plt.close()


# 模型训练和预测（包含双模型对比和集成学习）
def train_and_predict():
    train, test = load_data()
    train['Fertilizer Name'] = train['Fertilizer Name'].apply(convert_to_npk)
    analyze_features(train)

    features = [
        'Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
        'Nitrogen', 'Potassium', 'Phosphorous', 'N_P_ratio', 
        'nutrient_index', 'temp_humidity', 'humidity_moisture',
        'temp_group', 'humidity_group'
    ]

    # 处理分类特征
    cat_features = ['Soil Type', 'Crop Type', 'temp_group', 'humidity_group']
    for col in cat_features:
        le = LabelEncoder()
        all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(all_values)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # 编码目标变量
    le_fertilizer = LabelEncoder()
    y = train['Fertilizer Name']
    y_encoded = le_fertilizer.fit_transform(y)
    X_train = train[features]
    X_test = test[features]

    # 交叉验证设置
    n_folds = 3
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 定义模型字典
    models = {
        'LightGBM': {
            'oof_preds': np.zeros((X_train.shape[0], len(le_fertilizer.classes_))),
            'test_preds': np.zeros((X_test.shape[0], len(le_fertilizer.classes_))),
            'fold_scores': [],
            'feature_importances': pd.DataFrame(index=features)
        },
        'XGBoost': {
            'oof_preds': np.zeros((X_train.shape[0], len(le_fertilizer.classes_))),
            'test_preds': np.zeros((X_test.shape[0], len(le_fertilizer.classes_))),
            'fold_scores': [],
            'feature_importances': pd.DataFrame(index=features)
        },
        'Ensemble': {
            'oof_preds': np.zeros((X_train.shape[0], len(le_fertilizer.classes_))),
            'test_preds': np.zeros((X_test.shape[0], len(le_fertilizer.classes_))),
            'fold_scores': []
        }
    }

    # 训练LightGBM和XGBoost
    for model_name in ['LightGBM', 'XGBoost']:
        print(f"\n===== 训练 {model_name} 模型 =====")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_encoded)):
            print(f"\nTraining Fold {fold + 1}/{n_folds}")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

            if model_name == 'LightGBM':
                # LightGBM训练
                train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_features)
                params = {
                    'objective': 'multiclass', 'num_class': len(le_fertilizer.classes_),
                    'metric': 'multi_logloss', 'boosting_type': 'gbdt',
                    'num_leaves': 31, 'learning_rate': 0.1,
                    'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                    'bagging_freq': 5, 'min_child_samples': 20,
                    'verbose': -1, 'seed': 42 + fold, 'n_jobs': -1
                }
                model = lgb.train(
                    params, train_data, num_boost_round=800,
                    valid_sets=[train_data, val_data],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)]
                )
                val_preds = model.predict(X_val, num_iteration=model.best_iteration)
                test_preds = model.predict(X_test, num_iteration=model.best_iteration)
                fold_importance = pd.Series(model.feature_importance(), index=features)

            elif model_name == 'XGBoost':
                # XGBoost训练
                params = {
                    'objective': 'multi:softprob', 'num_class': len(le_fertilizer.classes_),
                    'eval_metric': 'mlogloss', 'max_depth': 5,
                    'learning_rate': 0.1, 'subsample': 0.8,
                    'colsample_bytree': 0.8, 'min_child_weight': 3,
                    'seed': 42 + fold, 'n_jobs': -1
                }
                model = xgb.XGBClassifier(**params, n_estimators=800, early_stopping_rounds=30, verbose=0)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                val_preds = model.predict_proba(X_val)
                test_preds = model.predict_proba(X_test)
                fold_importance = pd.Series(model.feature_importances_, index=features)

            # 保存预测结果和特征重要性
            models[model_name]['oof_preds'][val_idx] = val_preds
            fold_map5 = map5_score(y_val, val_preds)
            models[model_name]['fold_scores'].append(fold_map5)
            print(f"{model_name} Fold {fold + 1} MAP@5: {fold_map5:.4f}")
            models[model_name]['test_preds'] += test_preds / n_folds
            models[model_name]['feature_importances'][f'fold_{fold + 1}'] = fold_importance

        # 模型评估和特征重要性
        oof_map5 = map5_score(y_encoded, models[model_name]['oof_preds'])
        print(f"\n{model_name} Overall OOF MAP@5: {oof_map5:.4f}")
        print(f"{model_name} 平均MAP@5: {np.mean(models[model_name]['fold_scores']):.4f}")
        models[model_name]['feature_importances']['mean'] = models[model_name]['feature_importances'].mean(axis=1)
        models[model_name]['feature_importances'] = models[model_name]['feature_importances'].sort_values('mean', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models[model_name]['feature_importances']['mean'], y=models[model_name]['feature_importances'].index)
        plt.title(f'{model_name} Feature Importances')
        plt.savefig(f'{model_name.lower()}_feature_importances.png')
        plt.close()

    # 双模型集成
    print("\n===== 训练集成模型 =====")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_encoded)):
        ensemble_val_preds = models['LightGBM']['oof_preds'][val_idx] * 0.5 + models['XGBoost']['oof_preds'][val_idx] * 0.5
        models['Ensemble']['oof_preds'][val_idx] = ensemble_val_preds
        fold_map5 = map5_score(y_encoded[val_idx], ensemble_val_preds)
        models['Ensemble']['fold_scores'].append(fold_map5)
        print(f"Ensemble Fold {fold + 1} MAP@5: {fold_map5:.4f}")

    # 集成模型测试集预测
    models['Ensemble']['test_preds'] = models['LightGBM']['test_preds'] * 0.5 + models['XGBoost']['test_preds'] * 0.5
    ensemble_oof_map5 = map5_score(y_encoded, models['Ensemble']['oof_preds'])
    print(f"\n集成模型 Overall OOF MAP@5: {ensemble_oof_map5:.4f}")
    print(f"集成模型 平均MAP@5: {np.mean(models['Ensemble']['fold_scores']):.4f}")

    # 模型对比
    model_comparison = pd.DataFrame({
        'Model': ['LightGBM', 'XGBoost', 'Ensemble'],
        'MAP@5': [
            np.mean(models['LightGBM']['fold_scores']),
            np.mean(models['XGBoost']['fold_scores']),
            np.mean(models['Ensemble']['fold_scores'])
        ]
    }).sort_values('MAP@5', ascending=False)
    print("\n模型对比:")
    print(model_comparison)
    model_comparison.to_csv('model_comparison.csv', index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='MAP@5', data=model_comparison)
    plt.title('模型性能对比 (MAP@5)')
    plt.savefig('model_comparison.png')
    plt.close()

    # 选择最佳模型生成预测
    best_model = model_comparison.iloc[0]['Model']
    top5_indices = np.argsort(models[best_model]['test_preds'], axis=1)[:, -5:][:, ::-1]
    top5_fertilizers = np.array([le_fertilizer.inverse_transform(top5_indices[:, i]) for i in range(5)]).T
    fertilizer_strings = [' '.join(row) for row in top5_fertilizers]
    submission = pd.DataFrame({'id': test['id'], 'Fertilizer Name': fertilizer_strings})
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已生成（基于{best_model}模型）: submission.csv")

    # 农学合理性分析
    analyze_agronomic_reasoning(X_train, y_encoded, le_fertilizer, models[best_model]['oof_preds'])
    return model_comparison


# 农学合理性分析
def analyze_agronomic_reasoning(X_train, y_encoded, le_fertilizer, oof_preds):
    analysis = X_train.copy()
    analysis['true_fertilizer'] = le_fertilizer.inverse_transform(y_encoded)
    top1_preds = np.argmax(oof_preds, axis=1)
    analysis['predicted_fertilizer'] = le_fertilizer.inverse_transform(top1_preds)
    analysis['true_n'] = analysis['true_fertilizer'].apply(lambda x: int(x.split('-')[0]))
    analysis['true_p'] = analysis['true_fertilizer'].apply(lambda x: int(x.split('-')[1]))
    analysis['true_k'] = analysis['true_fertilizer'].apply(lambda x: int(x.split('-')[2]))

    crop_groups = analysis.groupby('Crop Type')
    with open('agronomic_analysis.txt', 'w') as f:
        f.write("===== 农学合理性分析 =====")
        for crop_type, group in crop_groups:
            f.write(f"\n\n=== 作物类型: {crop_type} ===\n")
            f.write("\n最常使用的肥料:\n")
            fertilizer_counts = group['true_fertilizer'].value_counts().head(3)
            for fertilizer, count in fertilizer_counts.items():
                f.write(f"- {fertilizer}: {count}次 ({count/len(group):.2%})\n")
            avg_nutrients = group[['Nitrogen', 'Phosphorous', 'Potassium']].mean()
            f.write(f"\n平均土壤养分:\n- 氮: {avg_nutrients['Nitrogen']:.2f}\n- 磷: {avg_nutrients['Phosphorous']:.2f}\n- 钾: {avg_nutrients['Potassium']:.2f}\n")
            f.write("\n最常使用肥料的NPK特点:\n")
            for fertilizer in fertilizer_counts.index:
                f_subset = group[group['true_fertilizer'] == fertilizer]
                avg_soil_n = f_subset['Nitrogen'].mean()
                avg_soil_p = f_subset['Phosphorous'].mean()
                avg_soil_k = f_subset['Potassium'].mean()
                f.write(f"- {fertilizer}:\n  土壤养分: N={avg_soil_n:.2f}, P={avg_soil_p:.2f}, K={avg_soil_k:.2f}\n")
                f.write(f"  肥料成分: N={int(fertilizer.split('-')[0])}, P={int(fertilizer.split('-')[1])}, K={int(fertilizer.split('-')[2])}\n")
                f.write(f"  匹配分析: {'氮肥补充' if avg_soil_n < 50 else '氮肥充足'},\n           {'磷肥补充' if avg_soil_p < 30 else '磷肥充足'},\n           {'钾肥补充' if avg_soil_k < 40 else '钾肥充足'}\n")


if __name__ == "__main__":
    model_comparison = train_and_predict()
    print("\n" + "=" * 50)
    print("最终模型性能对比:")
    print(model_comparison)
    print("=" * 50)

