import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

# ----------------------------------------------------
import sklearn
import scipy
import statsmodels.api as sm 
from scipy.stats import shapiro

# ----------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# ----------------------------------------------------
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,StackingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ----------------------------------------------------
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from collections import Counter
import time

# ----------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin


import warnings
warnings.filterwarnings("ignore")


def calculate_woe_iv(df, feature, target):
    """Tính WOE và IV cho một biến phân loại."""
    lst = []
    # Fill NaN tạm thời để tính toán
    temp_df = df[[feature, target]].copy()
    temp_df[feature] = temp_df[feature].fillna("Unknown")

    for val in temp_df[feature].unique():
        lst.append({
            'Value': val,
            'All': temp_df[temp_df[feature] == val].count()[feature],
            'Good': temp_df[(temp_df[feature] == val) & (temp_df[target] == 0)].count()[feature],
            'Bad': temp_df[(temp_df[feature] == val) & (temp_df[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    
    # Smoothing để tránh chia cho 0
    dset['Distr_Good'] = dset['Distr_Good'].replace(0, 0.0001)
    dset['Distr_Bad'] = dset['Distr_Bad'].replace(0, 0.0001)
    
    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    
    return dict(zip(dset['Value'], dset['WoE']))

def outlier_treatment(df, cols):
    """Xử lý outliers bằng phương pháp IQR (kẹp giá trị)."""
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip giá trị về ngưỡng trên và dưới
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

def outlier_detect(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    return df[((df[col] < (q1_col - 1.5 * iqr_col)) |(df[col] > (q3_col + 1.5 * iqr_col)))]

# ----------------------------------------------------------
def lower_outlier(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    lower = df[(df[col] < (q1_col - 1.5 * iqr_col))]
    return lower

# ----------------------------------------------------------
def upper_outlier(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    upper = df[(df[col] > (q3_col + 1.5 * iqr_col))]
    return upper

# ----------------------------------------------------------
def preprocess(df, col):
    print("*********************** {} ***********************\n".format(col))
    print("lower outlier: {} ****** upper outlier: {}\n".format(lower_outlier(df,col).shape[0], upper_outlier(df,col).shape[0]))
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    df[col].plot(kind='box', subplots=True, sharex=False, vert=False)
    plt.subplot(2,1,2)
    df[col].plot(kind='density', subplots=True, sharex=False)
    plt.show()

# ----------------------------------------------------------
def preprocess_cat(df, col):
    print("******************** {} ********************\n".format(col))
    df[col].value_counts().plot(kind='bar')
    plt.xticks(rotation='vertical')
    plt.show()
    
# ----------------------------------------------------------
def replace_upper(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    tmp = 9999999
    upper = q3_col + 1.5 * iqr_col
    df[col] = df[col].where(lambda x: (x < (upper)), tmp)
    df[col] = df[col].replace(tmp, upper)

# ----------------------------------------------------------
def replace_lower(df, col):
    q1_col = Q1[col]
    iqr_col = IQR[col]
    q3_col = Q3[col]
    tmp = 1111111
    lower = q1_col - 1.5 * iqr_col
    df[col] = df[col].where(lambda x: (x > (lower)), tmp)
    df[col] = df[col].replace(tmp, lower)

# ----------------------------------------------------------
def replace_mode(df, col):
    df[col] = df[col].fillna(df[col].mode()[0])
    print("NaN in {} raplaced with {}".format(col, df[col].mode()[0]))

# ----------------------------------------------------------
def replace_mean(df, col):
    df[col] = df[col].fillna(df[col].mean())
    print("NaN in {} raplaced with {}".format(col, df[col].mean()))
    

def replace_median(df, col):
    df[col] = df[col].fillna(df[col].median())
    print("NaN in {} raplaced with {}".format(col, df[col].median()))

# ----------------------------------------------------------
kfold = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)

def preprocess_car_info(df):

    if 'OWN_CAR_AGE' in df.columns:
        df['OWN_CAR_AGE_MISSING'] = df['OWN_CAR_AGE'].isna().astype(int)
        df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)
    return df

def cross_validation(x, y, model):
    result= cross_val_score(model, x, y, cv=kfold, scoring="roc_auc", n_jobs=-1)
    print("Score: %f" % result.mean())
    
# ----------------------------------------------------------
def RndSrch_Tune(model, X, y, params):
    
    clf = RandomizedSearchCV(model, params, scoring ='roc_auc', cv = kfold, n_jobs=-1, random_state=100)
    clf.fit(X, y)
    print("best score is :" , clf.best_score_)
    print("best estimator is :" , clf.best_estimator_)
    print("best Params is :" , clf.best_params_)
    return (clf.best_score_)

def plot_roc_curves(tuned_models, X, y):
    """Vẽ biểu đồ ROC cho tất cả các mô hình đã huấn luyện."""
    plt.figure(figsize=(10, 8))
    
    for name, model in tuned_models.items():
        try:
            # Dự đoán xác suất
            y_proba = model.predict_proba(X)[:, 1]
            
            # Tính toán False Positive Rate (FPR), True Positive Rate (TPR)
            fpr, tpr, _ = roc_curve(y, y_proba)
            auc = roc_auc_score(y, y_proba)
            
            # đường cong ROC
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
        except Exception as e:
            print(f"Lỗi khi vẽ ROC cho {name}: {e}")
            
    # đường chéo ngẫu nhiên
    plt.plot([0, 1], [0, 1], 'k--', label='Ngẫu nhiên (AUC = 0.50)')
    
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Sensitivity')
    plt.title('Biểu đồ ROC so sánh các mô hình')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def ks_statistic(y_true, y_proba):

    df = pd.DataFrame({'true': y_true, 'proba': y_proba})
    df = df.sort_values('proba', ascending=False)
    
    df['cdf1'] = df['true'].cumsum() / df['true'].sum()
    df['cdf0'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()
    
    # KS là giá trị tuyệt đối lớn nhất của hiệu hai CDF
    return np.max(np.abs(df['cdf1'] - df['cdf0']))

def gini_coefficient(auc):
    return 2 * auc - 1

def evaluate_model(model, X, y):

    model_name = model.__class__.__name__
    start_time = time.time()
    
    # Huấn luyện mô hình
    print(f"Bắt đầu train {model_name}...")
    
    try:
        if model_name != 'StackingClassifier':
            model.fit(X, y)
    except Exception as e:
        print(f"Lỗi khi train {model_name}: {e}")
        return None
        
    end_time = time.time()
    
    # 
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    
    # Metrics
    auc = roc_auc_score(y, y_proba)
    gini = gini_coefficient(auc)
    ks = ks_statistic(y, y_proba)

    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    cm = confusion_matrix(y, y_pred)
    
    return {
        'Model': model_name,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC': auc,
        'Gini': gini,
        'KS-Statistic': ks,
        'Training_Time (s)': round(end_time - start_time, 2),
        'Hyperparameters': str(model.get_params()),
        'True Positives (TP)': cm[1, 1],
        'False Negatives (FN)': cm[1, 0]
    }

def perform_tuning(model, X, y, param_dist, n_iter=10):

    model_name = model.__class__.__name__
    print(f"\n--- Bắt đầu điều chỉnh siêu tham số cho {model_name} (n_iter={n_iter}) ---")
    
    # Sử dụng StratifiedKFold để đảm bảo tỷ lệ class được giữ nguyên trong các fold
    cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
    
    # Random SearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter, 
        scoring='roc_auc', 
        cv=cv_folds,
        verbose=1,
        random_state=100,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    
    print(f"Hoàn thành điều chỉnh. AUC tốt nhất: {random_search.best_score_:.4f}")
    print(f"Tham số tốt nhất: {random_search.best_params_}")
    
    return random_search.best_estimator_

class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, learning_rate=0.001, epochs=50, batch_size=512):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        self.classes_ = [0, 1] 

    def _build_model(self):
        # Kiến trúc mạng nơ-ron đơn giản cho dữ liệu dạng bảng
        model = Sequential()
        
        # Input Layer + Hidden Layer 1
        model.add(Dense(128, input_dim=self.input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden Layer 2
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Hidden Layer 3
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        # Output Layer (Sigmoid cho Binary Classification)
        model.add(Dense(1, activation='sigmoid'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])
        return model

    def fit(self, X, y):
        # Xử lý class weight để cân bằng dữ liệu (tương tự scale_pos_weight của XGBoost)
        # Bạn đã tính biến 'estimate' ở cell 69, ta có thể tính lại ở đây hoặc truyền vào
        count_0 = len(y) - y.sum()
        count_1 = y.sum()
        # Trọng số cho lớp 1 (thiểu số) sẽ cao hơn
        class_weight = {0: 1.0, 1: count_0 / count_1}
        
        # Callbacks để dừng sớm nếu không cải thiện
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.00001)

        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr],
            verbose=1 
        )
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32").flatten()

    def predict_proba(self, X):
        # Hàm này quan trọng để tương thích với roc_auc_score và hàm evaluate_model của bạn
        probs = self.model.predict(X, verbose=0)
        # Scikit-learn yêu cầu array (N_samples, 2) cho binary classification
        # Cột 0: xác suất lớp 0, Cột 1: xác suất lớp 1
        return np.hstack((1 - probs, probs))

    def get_params(self, deep=True):
        return {"input_dim": self.input_dim, 
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size}


# def create_domain_features(df):
#     # 1. INCOME_CREDIT_PERCENT: Income relative to credit amount
#     # (Higher is better: you earn more relative to what you borrow)
#     df['INCOME_CREDIT_PERCENT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    
#     # 2. ANNUITY_INCOME_PERCENT: Loan annuity relative to income
#     # (Lower is better: smaller portion of income goes to loan repayment)
#     df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
#     # 3. CREDIT_TERM: Payment length in months (approximate)
#     # (Annuity is monthly payment, Credit is total loan)
#     df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
#     # 4. INCOME_PER_PERSON: Income per family member
#     df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    
#     # 5. CNT_ADULT_FAM_MEMBER: Number of adults in family
#     df['CNT_ADULT_FAM_MEMBER'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    
#     # 6. RATIO_CHILDREN_TO_ADULTS: Dependency ratio within family
#     # Handle potential division by zero if no adults (unlikely but safe to handle)
#     df['RATIO_CHILDREN_TO_ADULTS'] = df['CNT_CHILDREN'] / df['CNT_ADULT_FAM_MEMBER'].replace(0, 1) 
    
    # # 7. RATIO_AMT_CREDIT_TO_CNT_FAM_MEMBERS: Credit load per person
    # df['RATIO_AMT_CREDIT_TO_CNT_FAM_MEMBERS'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    
    # # 8. RATIO_AMT_CREDIT_TO_CNT_ADULT_FAM_MEMBER: Credit load per adult
    # df['RATIO_AMT_CREDIT_TO_CNT_ADULT_FAM_MEMBER'] = df['AMT_CREDIT'] / df['CNT_ADULT_FAM_MEMBER'].replace(0, 1)
    
    # # 9. AMT_INCOME_TOTAL_PER_ADULT_FAM_MEMBER: Income per adult
    # df['AMT_INCOME_TOTAL_PER_ADULT_FAM_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_ADULT_FAM_MEMBER'].replace(0, 1)

    # # 10. (Optional from previous list) GOODS_CREDIT_RATIO
    # df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']

    # # 11. Based on EXT_SOURCES features: calculate mean, max, sum, min, median
    # df['EXT_SOURCE_mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    # df['EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    # df['EXT_SOURCES_SUM'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    # df['EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    # df['EXT_SOURCES_MEDIAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)
    
    
    # return df

def get_apps_processed(apps):
    """
    feature engineering for apps
    """

    # 1.EXT_SOURCE_X FEATURE 
    apps['APPS_EXT_SOURCE_MEAN'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    apps['APPS_EXT_SOURCE_STD'] = apps[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    apps['APPS_EXT_SOURCE_STD'] = apps['APPS_EXT_SOURCE_STD'].fillna(apps['APPS_EXT_SOURCE_STD'].mean())
    
    # AMT_CREDIT 
    apps['APPS_ANNUITY_CREDIT_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_CREDIT']
    apps['APPS_GOODS_CREDIT_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_CREDIT']
    
    # AMT_INCOME_TOTAL 
    apps['APPS_ANNUITY_INCOME_RATIO'] = apps['AMT_ANNUITY']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CREDIT_INCOME_RATIO'] = apps['AMT_CREDIT']/apps['AMT_INCOME_TOTAL']
    apps['APPS_GOODS_INCOME_RATIO'] = apps['AMT_GOODS_PRICE']/apps['AMT_INCOME_TOTAL']
    apps['APPS_CNT_FAM_INCOME_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['CNT_FAM_MEMBERS']
    
    # DAYS_BIRTH, DAYS_EMPLOYED 
    apps['APPS_EMPLOYED_BIRTH_RATIO'] = apps['DAYS_EMPLOYED']/apps['DAYS_BIRTH']
    apps['APPS_INCOME_EMPLOYED_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_EMPLOYED']
    apps['APPS_INCOME_BIRTH_RATIO'] = apps['AMT_INCOME_TOTAL']/apps['DAYS_BIRTH']
    apps['APPS_CAR_BIRTH_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_BIRTH']
    apps['APPS_CAR_EMPLOYED_RATIO'] = apps['OWN_CAR_AGE'] / apps['DAYS_EMPLOYED']
    
    return apps

def get_prev_processed(prev):
    """
    feature engineering 
    for previouse application credit history
    """
    prev['PREV_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['PREV_CREDIT_APPL_RATIO'] = prev['AMT_CREDIT']/prev['AMT_APPLICATION']
    # prev['PREV_ANNUITY_APPL_RATIO'] = prev['AMT_ANNUITY']/prev['AMT_APPLICATION']
    prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE']/prev['AMT_APPLICATION']

    # Data Cleansing
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # substraction between DAYS_LAST_DUE_1ST_VERSION and DAYS_LAST_DUE
    prev['PREV_DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

    # 1.Calculate the interest rate
    all_pay = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['PREV_INTERESTS_RATE'] = (all_pay/prev['AMT_CREDIT'] - 1)/prev['CNT_PAYMENT']

    agg_dict = {
      'AMT_CREDIT':['mean', 'max', 'sum'],
      'AMT_ANNUITY':['mean', 'max', 'sum'], 
      'AMT_APPLICATION':['mean', 'max', 'sum'],
      'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
      'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
      'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
      'DAYS_DECISION': ['min', 'max', 'mean'],
      'CNT_PAYMENT': ['mean', 'sum'],
        
      'PREV_CREDIT_DIFF':['mean', 'max', 'sum'], 
      'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
      'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
      'PREV_GOODS_APPL_RATIO':['mean', 'max'],
      'PREV_DAYS_LAST_DUE_DIFF':['mean', 'max', 'sum'],
      'PREV_INTERESTS_RATE':['mean', 'max']
    }

    prev_group = prev.groupby('SK_ID_CURR')
    prev_amt_agg = prev_group.agg(agg_dict)

    # multi index 
    prev_amt_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_amt_agg.columns.ravel()]

    return prev_amt_agg

def get_bureau_processed(bureau):
    bureau['BUREAU_ENDDATE_FACT_DIFF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_FACT_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
  
    bureau['BUREAU_CREDIT_DEBT_RATIO']=bureau['AMT_CREDIT_SUM_DEBT']/bureau['AMT_CREDIT_SUM']
    #bureau['BUREAU_CREDIT_DEBT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['BUREAU_CREDIT_DEBT_DIFF'] = bureau['AMT_CREDIT_SUM_DEBT'] - bureau['AMT_CREDIT_SUM']
    
    bureau['BUREAU_IS_DPD'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)
    bureau['BUREAU_IS_DPD_OVER120'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x >120 else 0)
    bureau = bureau.groupby('SK_ID_CURR').mean()
    bureau_agg_dict = {
    'DAYS_CREDIT':['min', 'max', 'mean'],
    'CREDIT_DAY_OVERDUE':['min', 'max', 'mean'],
    'DAYS_CREDIT_ENDDATE':['min', 'max', 'mean'],
    'DAYS_ENDDATE_FACT':['min', 'max', 'mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
    'AMT_ANNUITY': ['max', 'mean', 'sum'],

    'BUREAU_ENDDATE_FACT_DIFF':['min', 'max', 'mean'],
    'BUREAU_CREDIT_FACT_DIFF':['min', 'max', 'mean'],
    'BUREAU_CREDIT_ENDDATE_DIFF':['min', 'max', 'mean'],
    'BUREAU_CREDIT_DEBT_RATIO':['min', 'max', 'mean'],
    'BUREAU_CREDIT_DEBT_DIFF':['min', 'max', 'mean'],
    'BUREAU_IS_DPD':['mean', 'sum'],
    'BUREAU_IS_DPD_OVER120':['mean', 'sum']
    }
    bureau_grp = bureau.groupby('SK_ID_CURR')
    bureau_day_amt_agg = bureau_grp.agg(bureau_agg_dict)
    bureau_day_amt_agg.columns = ['BUREAU_'+('_').join(column).upper() for column in bureau_day_amt_agg.columns.ravel()]
    # SK_ID_CURR reset_index()
    bureau_day_amt_agg = bureau_day_amt_agg.reset_index()
    #print('bureau_day_amt_agg shape:', bureau_day_amt_agg.shape)
    return bureau_day_amt_agg
    return bureau

def get_pos_bal_procressed(pos_bal):
    # (SK_DPD) 0 , 0~ 100 , 100
    cond_over_0 = pos_bal['SK_DPD'] > 0
    cond_100 = (pos_bal['SK_DPD'] < 100) & (pos_bal['SK_DPD'] > 0)
    cond_over_100 = (pos_bal['SK_DPD'] >= 100)

    # 0~ 120 120
    pos_bal['POS_IS_DPD'] = pos_bal['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    pos_bal['POS_IS_DPD_UNDER_120'] = pos_bal['SK_DPD'].apply(lambda x:1 if (x > 0) & (x <120) else 0 )
    pos_bal['POS_IS_DPD_OVER_120'] = pos_bal['SK_DPD'].apply(lambda x:1 if x >= 120 else 0)

    pos_bal_grp = pos_bal.groupby('SK_ID_CURR')
    pos_bal_agg_dict = {
        'MONTHS_BALANCE':['min', 'mean', 'max'], 
        'SK_DPD':['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT':['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE':['min', 'max', 'mean', 'sum'],
        
        'POS_IS_DPD':['mean', 'sum'],
        'POS_IS_DPD_UNDER_120':['mean', 'sum'],
        'POS_IS_DPD_OVER_120':['mean', 'sum']
    }

    pos_bal_agg = pos_bal_grp.agg(pos_bal_agg_dict)

    pos_bal_agg.columns = [('POS_')+('_').join(column).upper() for column in pos_bal_agg.columns.ravel()]
    
    return pos_bal_agg
def get_install_processed(install):
    # DPD  
    install['AMT_DIFF'] = install['AMT_INSTALMENT'] - install['AMT_PAYMENT']
    install['AMT_RATIO'] =  (install['AMT_PAYMENT'] +1)/ (install['AMT_INSTALMENT'] + 1)
    install['SK_DPD'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']

    # 30~ 120 100
    install['INS_IS_DPD'] = install['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    install['INS_IS_DPD_UNDER_120'] = install['SK_DPD'].apply(lambda x:1 if (x > 0) & (x <120) else 0 )
    install['INS_IS_DPD_OVER_120'] = install['SK_DPD'].apply(lambda x:1 if x >= 120 else 0)

    # SK_ID_CURR aggregation
    install_grp = install.groupby('SK_ID_CURR')

    install_agg_dict = {
        'NUM_INSTALMENT_VERSION':['nunique'], 
        'DAYS_ENTRY_PAYMENT':['mean', 'max', 'sum'],
        'DAYS_INSTALMENT':['mean', 'max', 'sum'],
        'AMT_INSTALMENT':['mean', 'max', 'sum'],
        'AMT_PAYMENT':['mean', 'max','sum'],

        'AMT_DIFF':['mean','min', 'max','sum'],
        'AMT_RATIO':['mean', 'max'],
        'SK_DPD':['mean', 'min', 'max'],
        'INS_IS_DPD':['mean', 'sum'],
        'INS_IS_DPD_UNDER_120':['mean', 'sum'],
        'INS_IS_DPD_OVER_120':['mean', 'sum']    
    }

    install_agg = install_grp.agg(install_agg_dict)
    install_agg.columns = ['INS_'+('_').join(column).upper() for column in install_agg.columns.ravel()]
    return install_agg

def get_card_bal_processed(card_bal):
    card_bal['BALANCE_LIMIT_RATIO'] = card_bal['AMT_BALANCE']/card_bal['AMT_CREDIT_LIMIT_ACTUAL']
    card_bal['DRAWING_LIMIT_RATIO'] = card_bal['AMT_DRAWINGS_CURRENT'] / card_bal['AMT_CREDIT_LIMIT_ACTUAL']

    # DPD
    card_bal['CARD_IS_DPD'] = card_bal['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    card_bal['CARD_IS_DPD_UNDER_120'] = card_bal['SK_DPD'].apply(lambda x:1 if (x > 0) & (x <120) else 0 )
    card_bal['CARD_IS_DPD_OVER_120'] = card_bal['SK_DPD'].apply(lambda x:1 if x >= 120 else 0)
    # SK_ID_CURR aggregation
    card_bal_grp = card_bal.groupby('SK_ID_CURR')
    card_bal_agg_dict = {
         #'MONTHS_BALANCE':['min', 'max', 'mean'],
        'AMT_BALANCE':['max'],
        'AMT_CREDIT_LIMIT_ACTUAL':['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max','sum'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max','sum'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean'],
        'SK_DPD': ['mean', 'max', 'sum'],

        'BALANCE_LIMIT_RATIO':['min','max'],
        'DRAWING_LIMIT_RATIO':['min', 'max'],
        'CARD_IS_DPD':['mean', 'sum'],
        'CARD_IS_DPD_UNDER_120':['mean', 'sum'],
        'CARD_IS_DPD_OVER_120':['mean', 'sum']    
    }
    card_bal_agg = card_bal_grp.agg(card_bal_agg_dict)
    card_bal_agg.columns = ['CARD_'+('_').join(column).upper() for column in card_bal_agg.columns.ravel()]

    card_bal_agg = card_bal_agg.reset_index()
    return card_bal_agg


def get_balance_data():
    pos_dtype = {
        'SK_ID_PREV':np.uint32, 'SK_ID_CURR':np.uint32, 'MONTHS_BALANCE':np.int32, 'SK_DPD':np.int32,
        'SK_DPD_DEF':np.int32, 'CNT_INSTALMENT':np.float32,'CNT_INSTALMENT_FUTURE':np.float32
    }

    install_dtype = {
        'SK_ID_PREV':np.uint32, 'SK_ID_CURR':np.uint32, 'NUM_INSTALMENT_NUMBER':np.int32, 'NUM_INSTALMENT_VERSION':np.float32,
        'DAYS_INSTALMENT':np.float32, 'DAYS_ENTRY_PAYMENT':np.float32, 'AMT_INSTALMENT':np.float32, 'AMT_PAYMENT':np.float32
    }

    card_dtype = {
        'SK_ID_PREV':np.uint32, 'SK_ID_CURR':np.uint32, 'MONTHS_BALANCE':np.int16,
        'AMT_CREDIT_LIMIT_ACTUAL':np.int32, 'CNT_DRAWINGS_CURRENT':np.int32, 'SK_DPD':np.int32,'SK_DPD_DEF':np.int32,
        'AMT_BALANCE':np.float32, 'AMT_DRAWINGS_ATM_CURRENT':np.float32, 'AMT_DRAWINGS_CURRENT':np.float32,
        'AMT_DRAWINGS_OTHER_CURRENT':np.float32, 'AMT_DRAWINGS_POS_CURRENT':np.float32, 'AMT_INST_MIN_REGULARITY':np.float32,
        'AMT_PAYMENT_CURRENT':np.float32, 'AMT_PAYMENT_TOTAL_CURRENT':np.float32, 'AMT_RECEIVABLE_PRINCIPAL':np.float32,
        'AMT_RECIVABLE':np.float32, 'AMT_TOTAL_RECEIVABLE':np.float32, 'CNT_DRAWINGS_ATM_CURRENT':np.float32,
        'CNT_DRAWINGS_OTHER_CURRENT':np.float32, 'CNT_DRAWINGS_POS_CURRENT':np.float32, 'CNT_INSTALMENT_MATURE_CUM':np.float32
    }

    pos_bal = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv', dtype=pos_dtype)
    install = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv', dtype=install_dtype)
    card_bal = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv', dtype=card_dtype)

    return pos_bal, install, card_bal


train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
prev = pd.read_csv("/kaggle/input/home-credit-default-risk/previous_application.csv")
pos_bal, install, card_bal = get_balance_data()


apps = train.append(test)

card_bal = get_card_bal_processed(card_bal)
install = get_install_processed(install)
apps = get_apps_processed(apps)
bureau = get_bureau_processed(bureau)
prev = get_prev_processed(prev)

all_ = pd.merge(apps, bureau, how='left', on='SK_ID_CURR')
all_ = all_.merge(card_bal, how='left', on='SK_ID_CURR')
all_ = all_.merge(install, how='left', on='SK_ID_CURR')
all_ = all_.merge(prev, how='left', on='SK_ID_CURR')
all_ = preprocess_car_info(all_)
print(all_.shape)

train = all_[all_['SK_ID_CURR'].isin(train.SK_ID_CURR)]
test = all_[all_.SK_ID_CURR.isin(test.SK_ID_CURR)]
test.drop('TARGET', axis = 1, inplace = True)


sns.countplot(x = "TARGET", data = train)
train.loc[:, 'TARGET'].value_counts()


print(train.info())
print("*******************************")
print(test.info())


pd.set_option('display.max_rows', train.shape[0])
train.describe().T


pd.DataFrame(train.isnull().sum().sort_values(ascending = False))


pd.DataFrame(test.isnull().sum().sort_values(ascending = False))


threshold_train = len(train) * 0.60
int(threshold_train)


threshold_test = len(test) * 0.60
int(threshold_test)


print(f"In train data: {len(train.columns[train.isnull().sum() > int(threshold_train)])}\n")
print(train.columns[train.isnull().sum() > int(threshold_train)])
print("******************************************")
print(f"In test data: {len(test.columns[test.isnull().sum() > int(threshold_test)])}\n")
print(test.columns[test.isnull().sum() > int(threshold_test)])


train.dropna(axis=1, thresh=threshold_train).shape
to_drop = train.columns[train.isnull().sum() > int(threshold_train)]


train_new = train.drop(columns = to_drop)
print(train_new.shape)
test_new = test.drop(columns = to_drop)
print(test_new.shape)


# train_new = train.dropna(axis=1, thresh=threshold_train)
# print(train_new.shape)
# print("******************************************")
# test_new = test.dropna(axis=1, thresh=threshold_test)
# print(test_new.shape)


numeric_feature = train_new.dtypes!=object
final_numeric_feature = train_new.columns[numeric_feature].tolist()

#----------------------------------------------------
numeric_feature_test = test_new.dtypes!=object
final_numeric_feature_test = test_new.columns[numeric_feature_test].tolist()


numeric = train_new[final_numeric_feature]

#-------------------------------------------
numeric_test = test_new[final_numeric_feature_test]
numeric.head()


inf_summary = np.isinf(numeric).sum()
inf_summary = inf_summary[inf_summary > 0]
inf_summary


numeric.replace([np.inf, -np.inf], np.nan, inplace=True)


pd.DataFrame(numeric.isnull().sum().sort_values(ascending = False))


pd.DataFrame(numeric_test.isnull().sum().sort_values(ascending = False))


discrete_features = numeric.dtypes==int
final_discrete_feature = numeric.columns[discrete_features].tolist()
discrete = numeric[final_discrete_feature]

#-------------------------------------------
discrete_features_test = numeric_test.dtypes==int
final_discrete_feature_test = numeric_test.columns[discrete_features_test].tolist()
discrete_test = numeric_test[final_discrete_feature_test]

discrete.head()


pd.DataFrame(discrete.isnull().sum().sort_values(ascending = False))


pd.DataFrame(discrete_test.isnull().sum().sort_values(ascending = False))


continuous_features = numeric.dtypes==float
final_continuous_feature = numeric.columns[continuous_features].tolist()
continuous = numeric[final_continuous_feature]

#-------------------------------------------
continuous_features_test = numeric_test.dtypes==float
final_continuous_feature_test = numeric_test.columns[continuous_features_test].tolist()
continuous_test = numeric_test[final_continuous_feature_test]

continuous.head()


pd.DataFrame(continuous.isnull().sum().sort_values(ascending = False))


inf_summary = np.isinf(numeric).sum()
inf_summary = inf_summary[inf_summary > 0]
inf_summary


pd.DataFrame(continuous_test.isnull().sum().sort_values(ascending = False))


Q1 = train_new.quantile(0.25)
Q3 = train_new.quantile(0.75)
IQR = Q3 - Q1


continuous_is_null = continuous.isnull().sum() != 0
final_continuous_feature = continuous.columns[continuous_is_null].tolist()
print("In train: \n",final_continuous_feature)

print("****************************************")
continuous_is_null_test = continuous_test.isnull().sum() != 0
final_continuous_feature_test = continuous_test.columns[continuous_is_null_test].tolist()
print("In test: \n",final_continuous_feature_test)


print("In train:\n")
for i in range(len(final_continuous_feature)):
    replace_median(continuous, final_continuous_feature[i])

print("************************************")
print("In test:\n")
for i in range(len(final_continuous_feature_test)):
    replace_median(continuous_test, final_continuous_feature_test[i])


pd.DataFrame(continuous.isnull().sum().sort_values(ascending = False))


pd.DataFrame(continuous_test.isnull().sum().sort_values(ascending = False))


pd.DataFrame(continuous_test.isnull().sum().sort_values(ascending = False))


# numeric[continuous_col] = continuous[continuous_col]

# # ----------------------------------------------

# continuous_col_test = [c for c in continuous_col if c in continuous_test.columns and c != 'TARGET']
# numeric_test[continuous_col_test] = continuous_test[continuous_col_test]


continuous_col = continuous.columns
for i in range(len(continuous_col)):
    preprocess(continuous[continuous_col], continuous_col[i])


col_names = numeric.columns

# ------------------------------------
col_names_test = numeric_test.columns


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(outlier_detect(numeric,col_names[i]).shape[0])))
    
print("\n\n***************************************\n")
print("In test:\n")
for i in range(len(col_names_test)):
    print("{}: {}".format(col_names_test[i],(outlier_detect(numeric_test,col_names_test[i]).shape[0])))


outlier = []
for i in range(len(final_numeric_feature)):
    if outlier_detect(numeric[final_numeric_feature],final_numeric_feature[i]).shape[0] !=0:
        outlier.append(final_numeric_feature[i])

outlier_test = []
for i in range(len(final_numeric_feature_test)):
    if outlier_detect(numeric_test[final_numeric_feature_test],final_numeric_feature_test[i]).shape[0] !=0:
        outlier_test.append(final_numeric_feature_test[i])


# without TARGET field
col_names = outlier_test


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric_test,col_names[i]).shape[0])))


for i in range(len(col_names)):
    replace_upper(numeric, col_names[i])   
    
#------------------------------------------------------
for i in range(len(col_names)):
    replace_upper(numeric_test, col_names[i])   


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(upper_outlier(numeric_test,col_names[i]).shape[0])))


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric_test,col_names[i]).shape[0])))


for i in range(len(col_names)):
    replace_lower(numeric, col_names[i])
    
# #--------------------------------------------------
for i in range(len(col_names)):
    replace_lower(numeric_test, col_names[i])


print("In train:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric,col_names[i]).shape[0])))
    
print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names)):
    print("{}: {}".format(col_names[i],(lower_outlier(numeric_test,col_names[i]).shape[0])))


categorical_feature = train_new.dtypes==object
final_categorical_feature = train_new.columns[categorical_feature].tolist()

#----------------------------------------------------
categorical_feature_test = test_new.dtypes==object
final_categorical_feature_test = test_new.columns[categorical_feature_test].tolist()


categorical = train_new[final_categorical_feature]

#---------------------------------------------
categorical_test = test_new[final_categorical_feature_test]
categorical.head()


pd.DataFrame(categorical.isnull().sum().sort_values(ascending = False))


pd.DataFrame(categorical_test.isnull().sum().sort_values(ascending = False))


col_names_cat = categorical.columns


for i in range(len(col_names_cat)):
    preprocess_cat(categorical, col_names_cat[i])


print("unique number is = {}\nunique values are: \n{} ".format(len(train_new['ORGANIZATION_TYPE'].unique()), train_new['ORGANIZATION_TYPE'].unique()))


print("In train:\n")
for i in range(len(col_names_cat)):
    replace_mode(categorical, col_names_cat[i])

print("\n\n****************************************\n")
print("In test:\n")
for i in range(len(col_names_cat)):
    replace_mode(categorical_test, col_names_cat[i])


pd.DataFrame(categorical.isnull().sum().sort_values(ascending = False))


pd.DataFrame(categorical_test.isnull().sum().sort_values(ascending = False))


categorical.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)
# ---------------------------------------------
categorical_test.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)


le = LabelEncoder() 
categorical = categorical.apply(lambda col_names_cat: le.fit_transform(col_names_cat)) 
categorical_test = categorical_test.apply(lambda col_names_cat: le.fit_transform(col_names_cat)) 
categorical.head()


print("In train: ",categorical.shape)
print("In test: ",categorical_test.shape)


col_names_cat = categorical.columns
col_names = numeric_test.columns


train_new[col_names_cat] = categorical[col_names_cat]
train_new[col_names] = numeric[col_names]

# ----------------------------------------------------
test_new[col_names] = numeric_test[col_names]
test_new[col_names_cat] = categorical_test[col_names_cat]


train_new.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)
test_new.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)


print("In train: ",train_new.loc[train.duplicated()].shape)
#--------------------------------------------------
print("In test: ",test_new.loc[test.duplicated()].shape)


y = train_new['TARGET']


x_train = train_new.drop("TARGET", axis = 1)
# x_train = create_domain_features(x_train)
# test_new = create_domain_features(test_new)


num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_cols = [c for c in num_cols if c != 'TARGET']


scaler=MinMaxScaler()

x_train[num_cols] = pd.DataFrame(scaler.fit_transform(x_train[num_cols]))
test_new[num_cols] = pd.DataFrame(scaler.transform(test_new[num_cols]))


# Xử lý mất cân bằng class
counter = Counter(y)
estimate = counter[0] / counter[1]


tuned_models = {}
all_results = []
cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=100) # CV cho Stacking


# 1. LGBMClassifier Tuning
# lgbm_param_dist = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'n_estimators': [200, 300, 500],
#     'num_leaves': [20, 31, 50, 70],
#     'max_depth': [3, 5, 8],
#     'min_child_samples': [500, 800, 1000],
#     'reg_alpha': [0.1, 0.2, 0.5],
#     'reg_lambda': [0.1, 0.5, 1.0],
# }
# lgbm_base = LGBMClassifier(objective='binary', n_jobs=-1, random_state=100, class_weight='balanced', verbose=-1)
# lgbm_tuned = perform_tuning(lgbm_base, x_train, y, lgbm_param_dist, n_iter=tuning_iterations)
# tuned_models['LGBMClassifier'] = lgbm_tuned

# Tham số tốt nhất: {'reg_lambda': 0.1, 'reg_alpha': 0.1, 'num_leaves': 50, 'n_estimators': 500, 
                    # 'min_child_samples': 1000, 'max_depth': 5, 'learning_rate': 0.05}

lgb = LGBMClassifier(**{'reg_lambda': 0.1, 
                        'reg_alpha': 0.1, 
                        'num_leaves': 75, 
                        'n_estimators': 500, 
                        'min_child_samples': 1000, 
                        'max_depth': 5, 
                        'learning_rate': 0.05,
                        'class_weight':'balanced',
                        'random_state':100})

scores = cross_validation(x_train, y, lgb)
print(scores)


# 2. XGBClassifier Tuning
# xgb_param_dist = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'n_estimators': [200, 300, 500],
#     'max_depth': [3, 5, 7],
#     'min_child_weight': [1, 5, 10],
#     'gamma': [0, 0.1, 0.3],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }
# # Sử dụng scale_pos_weight để xử lý mất cân bằng
# xgb_base = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, 
#                          scale_pos_weight=estimate, n_jobs=-1, random_state=100)
# xgb_tuned = perform_tuning(xgb_base, x_train, y, xgb_param_dist, n_iter=tuning_iterations)
# tuned_models['XGBClassifier'] = xgb_tuned
# Tham số tốt nhất: {'subsample': 1.0, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 3, 
#                    'learning_rate': 0.2, 'gamma': 0.3, 'colsample_bytree': 1.0}
xgbc = XGBClassifier(
    subsample=1.0,
    n_estimators=300,
    min_child_weight=1,
    max_depth=3,
    learning_rate=0.1,
    colsample_bytree=0.8,
    random_state=100,
    scale_pos_weight=estimate,
    n_jobs=-1
)

scores_xgbc = cross_validation(x_train, y, xgbc)
print(scores_xgbc)


# 3. CatBoostClassifier Tuning
# cat_param_dist = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'iterations': [200, 300, 500],
#     'depth': [4, 6, 8],
#     'l2_leaf_reg': [1, 3, 5, 7]
# }
# # Sử dụng class_weights để xử lý mất cân bằng
# cat_base = CatBoostClassifier(verbose=0, random_state=100, class_weights=[1, estimate])
# cat_tuned = perform_tuning(cat_base, x_train, y, cat_param_dist, n_iter=tuning_iterations)
# tuned_models['CatBoostClassifier'] = cat_tuned

# Tham số tốt nhất: {'learning_rate': 0.2, 'l2_leaf_reg': 5, 'iterations': 200, 'depth': 4}
cat_model = CatBoostClassifier(
    learning_rate=0.2,
    l2_leaf_reg=5,
    iterations=200,
    depth=4,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False,
    random_state=100
)
scores_xgbc = cross_validation(x_train, y, cat_model)
print(scores_xgbc)


# 4. RandomForestClassifier Tuning
# rf_param_dist = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 10, 15, 20, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# rf_base = RandomForestClassifier(random_state=100, n_jobs=-1, class_weight='balanced')
# rf_tuned = perform_tuning(rf_base, x_train, y, rf_param_dist, n_iter=tuning_iterations)
# tuned_models['RandomForestClassifier'] = rf_tuned
# Tham số tốt nhất: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_depth': None}
rf_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=4,
    max_depth=None,
    random_state=100,
    n_jobs=-1
)
scores_xgbc = cross_validation(x_train, y, rf_model)
print(scores_xgbc)



# 5. LogisticRegression Tuning
# logreg_param_dist = {
#     'C': [0.001, 0.01, 0.1, 1, 10],
#     'solver': ['liblinear', 'saga'], # Sử dụng solver phù hợp với dataset lớn và penalty L1/L2
#     'penalty': ['l1', 'l2']
# }
# logreg_base = LogisticRegression(random_state=100, max_iter=500, class_weight='balanced')
# logreg_tuned = perform_tuning(logreg_base, x_train, y, logreg_param_dist, n_iter=tuning_iterations)
# tuned_models['LogisticRegression'] = logreg_tuned
# Tham số tốt nhất: {'solver': 'liblinear', 'penalty': 'l1', 'C': 10}
logit_model = LogisticRegression(
    solver='liblinear',
    penalty='l1',
    C=10,
    max_iter=1000
)
scores_xgbc = cross_validation(x_train, y, logit_model)
print(scores_xgbc)


# 6. AdaBoostClassifier Tuning
# ada_param_dist = {
#     'n_estimators': [50, 100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
# }
# ada_base = AdaBoostClassifier(random_state=100, algorithm='SAMME.R')
# ada_tuned = perform_tuning(ada_base, x_train, y, ada_param_dist, n_iter=tuning_iterations)
# tuned_models['AdaBoostClassifier'] = ada_tuned
# Tham số tốt nhất: {'n_estimators': 200, 'learning_rate': 1.0}

ada = AdaBoostClassifier(n_estimators= 200, learning_rate= 1.0, 
                         algorithm = 'SAMME.R',
                         random_state=100)
scores_ada = cross_validation(x_train, y, ada)
print(scores_ada)


models_to_run = {
    "AdaBoost": ada,
    "LogisticRegression": logit_model,
    "RandomForest": rf_model,
    "CatBoost": cat_model,
    "XGBoost": xgbc,
    "LightGBM": lgb,
}


# 7. StackingClassifier
stack_model = StackingClassifier(
    estimators=[
        ('logit', logit_model),
        ('lgb', lgb)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=100, class_weight='balanced'),
    stack_method='predict_proba',
    n_jobs=-1,
    cv=cv_folds,
    passthrough=False, 
)
print("\n--- Huấn luyện Stacking Model ---")
# Stacking không cần tuning ở bước này, ta huấn luyện trực tiếp
stack_model.fit(x_train, y) 
models_to_run["Stacking"] = stack_model
scores_stacking = cross_validation(x_train, y, stack_model)
print(scores_stacking)


# 8. Neural Network (Deep Learning)
# ---------------------------------------------------------
print("\n--- Huấn luyện Neural Network (Keras/Tensorflow) ---")

# Khởi tạo mô hình
nn_clf = KerasBinaryClassifier(
    input_dim=x_train.shape[1], 
    learning_rate=0.001, 
    epochs=20, 
    batch_size=1024
)

# Huấn luyện
try:
    nn_clf.fit(x_train, y)
    # Thêm vào dictionary tuned_models để so sánh sau này
    models_to_run['NeuralNetwork'] = nn_clf
    print("Neural Network đã được huấn luyện và thêm vào danh sách.")
except Exception as e:
    print(f"Lỗi khi huấn luyện Neural Network: {e}")


# --- 4. Thu thập và so sánh kết quả ---
all_results = []

for name, model in models_to_run.items():
    print(f"\nĐang chạy: {name}")
    result = evaluate_model(model, x_train, y)
    result["Model"] = name
    all_results.append(result)

comparison_df = pd.DataFrame(all_results)
comparison_df = comparison_df.sort_values(by="AUC", ascending=False)

print("\n" + "="*90)
print("BẢNG TỔNG HỢP SO SÁNH KẾT QUẢ CÁC MÔ HÌNH (TRAINING DATA)")
print("="*90)

display_cols = [
    "Model","AUC","Gini","KS-Statistic",
    "Precision","Recall","F1-Score",
    "Training_Time (s)",
    "True Positives (TP)", "False Negatives (FN)"
]

print(comparison_df[display_cols].to_markdown(index=False, floatfmt=".4f"))



plot_roc_curves(models_to_run, x_train, y)


comparison_df


# Lấy tên của mô hình có AUC cao nhất
best_model_name = comparison_df.iloc[0]['Model']

# Lấy đối tượng mô hình đã được huấn luyện
best_model = tuned_models[best_model_name]

print(f"\n--- Mô hình tốt nhất để dự đoán là: {best_model_name} ---")

# Dự đoán xác suất rủi ro (lấy xác suất của lớp 1) trên tập test
# Lưu ý: 'x_test' phải là dữ liệu test đã được tiền xử lý và scale (như trong phần 2. Data Preprocessing của script)
try:
    test_probabilities = best_model.predict_proba(x_test)[:, 1]
    sk_id_curr = test['SK_ID_CURR']
    submission = pd.DataFrame({
        'SK_ID_CURR': sk_id_curr,
        'TARGET': test_probabilities
    })
    
    # Lưu file submission
    submission_filename = 'my_submission.csv'
    submission.to_csv(submission_filename, index=False)


