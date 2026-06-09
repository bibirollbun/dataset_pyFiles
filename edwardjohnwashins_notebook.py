# å­¦å�·: 2024423310228, å§“å��: å¼ è±ª
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


# å­¦å�·: 2024423310228, å§“å��: å¼ è±ª
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re
import subprocess
import time
import os
import warnings
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def robust_column_standardization(df):

    df.columns = [re.sub(r'[^\w]', '', col).strip().lower() for col in df.columns]
    rename_map = {
        'fertilizername': 'fertilizer', 'temparature': 'temperature',
        'n': 'nitrogen', 'p': 'phosphorous', 'k': 'potassium'
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

def transform_categorical_features(df, encoder=None):

    cat_cols = [col for col in df.columns if df[col].dtype == 'object' 
                and col not in ['id', 'fertilizer']]
    
    if not cat_cols:
        return df, encoder
    
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(df[cat_cols])
    
    df[cat_cols] = encoder.transform(df[cat_cols])
    
    
    for col in cat_cols:
        df[col] = df[col].astype('int32')
    
    return df, encoder

def create_agri_features(df):

    required_cols = {'temperature', 'nitrogen', 'phosphorous', 'potassium'}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"å†œä¸šæ•°æ�®å…³é”®åˆ—ç¼ºå¤±: {missing}")

    df['n_p_ratio'] = df['nitrogen'] / (df['phosphorous'] + 1e-6)
    df['nutrient_balance'] = (df['nitrogen'] + df['phosphorous'] + df['potassium']) / 3
    df['temperature_effect'] = np.log1p(np.abs(df['temperature'] - 25))
    
    return df

def auto_detect_hardware():

    try:
        gpu_status = subprocess.run('nvidia-smi', capture_output=True)
        if gpu_status.returncode == 0:
            print("âœ… GPUåŠ é€Ÿå·²å�¯ç”¨")
            return {'device': 'cuda:0', 'tree_method': 'hist'}
    except:
        pass
    print("âš ï¸� ä½¿ç”¨CPUä¼˜åŒ–æ¨¡å¼�")
    return {'device': 'cpu', 'tree_method': 'hist'}

def plot_feature_importance(model, feature_names):
  
    plt.figure(figsize=(12, 8))
 
    importance_types = ['weight', 'gain', 'cover']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, imp_type in enumerate(importance_types):
    
        importance = model.get_booster().get_score(importance_type=imp_type)

        imp_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values('Importance', ascending=True).tail(15)

        axes[i].barh(imp_df['Feature'], imp_df['Importance'], color='skyblue')
        axes[i].set_title(f'{imp_type.capitalize()}é‡�è¦�æ€§')
        axes[i].set_xlabel('é‡�è¦�æ€§åˆ†æ•°')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

def plot_training_history(history):
  
    results = history.evals_result()
    epochs = len(results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, results['validation_0']['mlogloss'], label='è®­ç»ƒé›†')
    plt.plot(x_axis, results['validation_1']['mlogloss'], label='éªŒè¯�é›†')
    plt.legend()
    plt.ylabel('å¤šåˆ†ç±»å¯¹æ•°æ�Ÿå¤±')
    plt.xlabel('è®­ç»ƒè½®æ¬¡')
    plt.title('XGBoostè®­ç»ƒè¿‡ç¨‹ç›‘æ�§')
    plt.grid(True)
    plt.savefig('training_history.png', dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('æ··æ·†çŸ©é˜µ')
    plt.ylabel('çœŸå®�æ ‡ç­¾')
    plt.xlabel('é¢„æµ‹æ ‡ç­¾')
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

def safe_shap_analysis(model, X, feature_names):
    
    print("ğŸ”� å�¯åŠ¨å®‰å…¨æ¨¡å¼�SHAPåˆ†æ��...")
    
 
    try:
        explainer = shap.TreeExplainer(
            model,
            model_output="raw",
            feature_perturbation="interventional",  
            data=shap.sample(X, 100) 
        )
        shap_values = explainer.shap_values(
            X, 
            tree_limit=min(getattr(model, 'best_iteration_', 100), 100),
            check_additivity=False  
        )
        print("âœ… SHAPè®¡ç®—æˆ�åŠŸ(æ–¹æ³•1)")
        return shap_values, explainer
    except Exception as e:
        print(f"âš ï¸� æ–¹æ³•1å¤±è´¥: {str(e)[:100]}...")


    try:
      
        X = X[model.get_booster().feature_names]
        explainer = shap.Explainer(
            model, 
            masker=shap.sample(X, 100)
        )
        shap_values = explainer(X).values
        print("âœ… SHAPè®¡ç®—æˆ�åŠŸ(æ–¹æ³•2)")
        return shap_values, explainer
    except Exception as e:
        print(f"âš ï¸� æ–¹æ³•2å¤±è´¥: {str(e)[:100]}...")
    
  
    print("ğŸ”„ é™�çº§åˆ°ç‰¹å¾�é‡�è¦�æ€§åˆ†æ��")
    return None, None

def plot_shap_summary(model, X, feature_names):

    shap_values, explainer = safe_shap_analysis(model, X, feature_names)
    
    if shap_values is None:
      
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=15)
        plt.title('ç‰¹å¾�é‡�è¦�æ€§(æ›¿ä»£æ–¹æ¡ˆ)')
        plt.savefig('shap_fallback.png', dpi=300)
        plt.show()
        return
    
    try:
       
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=feature_names,
            show=False,
            plot_type="dot" 
        )
        plt.title('SHAPç‰¹å¾�é‡�è¦�æ€§(å…¨å±€)')
        plt.tight_layout()
        plt.savefig('shap_global.png', dpi=300)
        plt.show()
        
        if len(X) > 10:
            sample_idx = np.random.randint(0, len(X))
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[sample_idx], 
                X.iloc[sample_idx],
                feature_names=feature_names,
                matplotlib=True,
                text_rotation=15 
            )
            plt.title(f'SHAPè§£é‡Š(æ ·æœ¬#{sample_idx})')
            plt.tight_layout()
            plt.savefig('shap_single.png', dpi=300)
            plt.show()
            
    except Exception as e:
        print(f"âš ï¸� å�¯è§†åŒ–å¤±è´¥: {str(e)[:100]}...")
        print("ğŸ“Š ç”Ÿæˆ�ç‰¹å¾�é‡�è¦�æ€§æ›¿ä»£å›¾...")
        plot_feature_importance(model, feature_names)

def main():
    start_time = time.time()
    print("â�³ åŠ è½½æ•°æ�®...")

    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    print(f"æ•°æ�®åŠ è½½å®Œæˆ�ï¼Œè€—æ—¶: {time.time()-start_time:.2f}s")

    train = robust_column_standardization(train)
    test = robust_column_standardization(test)

    print("ğŸ”„ å¤„ç�†ç±»åˆ«ç‰¹å¾�...")
    train, encoder = transform_categorical_features(train)
    test, _ = transform_categorical_features(test, encoder)
    
    print("ğŸ› ï¸� æ‰§è¡Œç‰¹å¾�å·¥ç¨‹...")
    train = create_agri_features(train)
    test = create_agri_features(test)

    target_col = next((col for col in train.columns if 'fert' in col), None)
    if not target_col:
        raise KeyError(f"æœªæ‰¾åˆ°è‚¥æ–™åˆ—! å�¯ç”¨åˆ—: {train.columns.tolist()}")
    print(f"ç›®æ ‡åˆ—å·²è¯†åˆ«: {target_col}")

    le = LabelEncoder()
    y = le.fit_transform(train[target_col])
    feature_columns = [col for col in train.columns if col != target_col and col != 'id']
    X = train[feature_columns]

    hw_config = auto_detect_hardware()

    params = {
        'objective': 'multi:softprob',
        'num_class': len(le.classes_),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 20,
        'enable_categorical': True,
        'verbosity': 1,
        **hw_config
    }
 
    print("ğŸš€ å¼€å§‹è®­ç»ƒæ¨¡å�‹...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(**params)
    history = model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=10
    )
    print(f"æ¨¡å�‹è®­ç»ƒå®Œæˆ�ï¼Œè€—æ—¶: {time.time()-start_time:.2f}s")
    
    print("ğŸ“Š æ¨¡å�‹è¯„ä¼°...")
    y_pred = model.predict(X_val)
   
    print("\nåˆ†ç±»æŠ¥å‘Š:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    
    print("ğŸ�¨ ç”Ÿæˆ�å�¯è§†åŒ–...")

    plot_feature_importance(model, feature_columns)
    
    plot_training_history(history)

    plot_confusion_matrix(y_val, y_pred, le.classes_)

    sample_size = min(100, len(X_val))
    print(f"ğŸ”� æ‰§è¡ŒSHAPåˆ†æ��ï¼ˆå®‰å…¨æ¨¡å¼�ï¼Œæ ·æœ¬é‡�={sample_size}ï¼‰...")
    sample_idx = np.random.choice(X_val.index, size=sample_size, replace=False)
    plot_shap_summary(model, X_val.loc[sample_idx], feature_columns)
 
    print("ğŸ”® ç”Ÿæˆ�é¢„æµ‹...")
    test_data = test[feature_columns]
    test_probs = model.predict_proba(test_data)

    top5_idx = np.argsort(-test_probs, axis=1)[:, :5]
    top5_preds = [' '.join(le.inverse_transform(idx_row)) for idx_row in top5_idx]
    submission = pd.DataFrame({'id': test['id'], 'predictions': top5_preds})
    submission.to_csv('submission.csv', index=False)
    print(f"âœ… ä»»åŠ¡å®Œæˆ�! æ€»è€—æ—¶: {time.time()-start_time:.2f}s")

if __name__ == '__main__':
    main()

