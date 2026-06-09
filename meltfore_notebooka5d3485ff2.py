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


# å¯¼å…¥å¿…è¦�çš„åº“
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# æœºå™¨å­¦ä¹ åº“
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, precision_recall_curve, auc)



import pandas as pd
# æ��å‰�å®šä¹‰df_clean
df_clean = pd.read_csv("/kaggle/input/customer-churn-prediction-2020/sampleSubmission.csv")  # éœ€æ›¿æ�¢ä¸ºå®�é™…æ•°æ�®æ–‡ä»¶è·¯å¾„


import pandas as pd
# æ��å‰�å®šä¹‰df_clean
df_clean = pd.read_csv("/kaggle/input/customer-churn-prediction-2020/test.csv")  # éœ€æ›¿æ�¢ä¸ºå®�é™…æ•°æ�®æ–‡ä»¶è·¯å¾„


import pandas as pd
# æ��å‰�å®šä¹‰df_clean
df_clean = pd.read_csv("/kaggle/input/customer-churn-prediction-2020/train.csv")  # éœ€æ›¿æ�¢ä¸ºå®�é™…æ•°æ�®æ–‡ä»¶è·¯å¾„


# è®¾ç½®ä¸­æ–‡å­—ä½“å’Œå›¾å½¢æ ·å¼�
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

import warnings
warnings.filterwarnings('ignore')

print("æ‰€æœ‰åº“å¯¼å…¥å®Œæˆ�ï¼�")
print("æ­£åœ¨åŠ è½½æ•°æ�®...")


# æ–¹æ³•1ï¼šå°�è¯•æ­£ç¡®çš„Kaggleè·¯å¾„
try:
    df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("âœ“ æ•°æ�®ä»�Kaggleè·¯å¾„åŠ è½½æˆ�åŠŸï¼�")
except FileNotFoundError:
    # è¡¥å……exceptå�—çš„æ‰§è¡Œä»£ç �
    print("Ã— é”™è¯¯ï¼šæœªæ‰¾åˆ°Kaggleè·¯å¾„ä¸‹çš„æ–‡ä»¶")



# æ–¹æ³•2ï¼šå°�è¯•å…¶ä»–å�¯èƒ½çš„è·¯å¾„
try:  # å�–æ¶ˆè¯¥è¡Œçš„ç¼©è¿›ï¼Œä¸�æ³¨é‡Šè¡Œå¯¹é½�
    df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("âœ“ æ•°æ�®ä»�ç›¸å¯¹è·¯å¾„åŠ è½½æˆ�åŠŸï¼�")
except FileNotFoundError:
    # æ–¹æ³•3ï¼šæ£€æŸ¥å�¯ç”¨æ–‡ä»¶
    print("æ£€æŸ¥å�¯ç”¨æ•°æ�®é›†æ–‡ä»¶...")
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(f"æ‰¾åˆ°æ–‡ä»¶: {os.path.join(dirname, filename)}")


# æ��å‰�å®šä¹‰dfï¼ˆä»¥è¯»å�–æ–‡ä»¶ä¸ºä¾‹ï¼‰
import pandas as pd
df = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/sampleSubmission.csv')  # æ›¿æ�¢ä¸ºå®�é™…æ•°æ�®è·¯å¾„


# æ��å‰�å®šä¹‰dfï¼ˆä»¥è¯»å�–æ–‡ä»¶ä¸ºä¾‹ï¼‰
import pandas as pd
df = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/test.csv')  # æ›¿æ�¢ä¸ºå®�é™…æ•°æ�®è·¯å¾„


# æ��å‰�å®šä¹‰dfï¼ˆä»¥è¯»å�–æ–‡ä»¶ä¸ºä¾‹ï¼‰
import pandas as pd
# ä»…ä¿�ç•™æ­£ç¡®çš„æ–‡ä»¶è·¯å¾„å�‚æ•°ï¼Œé—­å�ˆå­—ç¬¦ä¸²å¼•å�·
df = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/train.csv')  # æ›¿æ�¢ä¸ºå®�é™…æ•°æ�®è·¯å¾„


# æ˜¾ç¤ºæ•°æ�®åŸºæœ¬ä¿¡æ�¯
print("=" * 50)
print("æ•°æ�®åŸºæœ¬ä¿¡æ�¯:")
print("=" * 50)
print(f"æ•°æ�®é›†å½¢çŠ¶: {df.shape}")
print(f"è¡Œæ•°: {df.shape[0]}, åˆ—æ•°: {df.shape[1]}")

print("\nå‰�5è¡Œæ•°æ�®é¢„è§ˆ:")
display(df.head())

print("\næ•°æ�®é›†åŸºæœ¬ä¿¡æ�¯:")
df.info()



# æ•°æ�®æ¸…æ´—
# =============================================================================
print("å¼€å§‹æ•°æ�®æ¸…æ´—...")

# å¤�åˆ¶å�Ÿå§‹æ•°æ�®
df_clean = df.copy()

print(f"æ•°æ�®æ¸…æ´—å‰�å½¢çŠ¶: {df_clean.shape}")

# æ£€æŸ¥ç¼ºå¤±å€¼
print("\nç¼ºå¤±å€¼ç»Ÿè®¡:")
missing_data = df_clean.isnull().sum()
missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
missing_info = pd.DataFrame({
    'ç¼ºå¤±æ•°é‡�': missing_data,
    'ç¼ºå¤±æ¯”ä¾‹%': missing_percent
})
display(missing_info[missing_info['ç¼ºå¤±æ•°é‡�'] > 0])

# æ£€æŸ¥TotalChargesåˆ—çš„ç‰¹æ®Šæƒ…å†µ
if 'TotalCharges' in df_clean.columns:
    print(f"\nTotalChargesåˆ—æ•°æ�®ç±»å�‹: {df_clean['TotalCharges'].dtype}")
    print(f"TotalChargesåˆ—å”¯ä¸€å€¼ç¤ºä¾‹: {df_clean['TotalCharges'].unique()[:10]}")
    
    # å°†TotalChargesè½¬æ�¢ä¸ºæ•°å€¼å�‹ï¼Œæ— æ³•è½¬æ�¢çš„è®¾ä¸ºNaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # å†�æ¬¡æ£€æŸ¥ç¼ºå¤±å€¼
    missing_after = df_clean.isnull().sum()
    if missing_after['TotalCharges'] > 0:
        print(f"TotalChargesç¼ºå¤±å€¼æ•°é‡�: {missing_after['TotalCharges']}")
        
        # å¤„ç�†ç¼ºå¤±å€¼ - ç”¨0å¡«å……ï¼ˆå› ä¸ºtenure=0è¡¨ç¤ºæ–°å®¢æˆ·ï¼‰
        df_clean['TotalCharges'].fillna(0, inplace=True)
        print("âœ“ TotalChargesç¼ºå¤±å€¼å·²ç”¨0å¡«å……")

# åˆ é™¤å®¢æˆ·ID - å¯¹é¢„æµ‹æ— å¸®åŠ©
if 'customerID' in df_clean.columns:
    df_clean.drop('customerID', axis=1, inplace=True)
    print("âœ“ å·²åˆ é™¤customerIDåˆ—")

# æ£€æŸ¥é‡�å¤�å€¼
duplicates = df_clean.duplicated().sum()
print(f"\né‡�å¤�è¡Œæ•°é‡�: {duplicates}")

print(f"\næ•°æ�®æ¸…æ´—å��å½¢çŠ¶: {df_clean.shape}")
print("âœ“ æ•°æ�®æ¸…æ´—å®Œæˆ�ï¼�")



# ç›®æ ‡å�˜é‡�åˆ†æ��
# =============================================================================
print("ç›®æ ‡å�˜é‡�åˆ†å¸ƒåˆ†æ��:")

if 'Churn' in df_clean.columns:
    churn_distribution = df_clean['Churn'].value_counts()
    churn_percentage = df_clean['Churn'].value_counts(normalize=True) * 100

    print("æµ�å¤±å®¢æˆ·åˆ†å¸ƒ:")
    for status, count, percent in zip(churn_distribution.index, churn_distribution.values, churn_percentage.values):
        print(f"  {status}: {count} ({percent:.2f}%)")

    # å�¯è§†åŒ–ç›®æ ‡å�˜é‡�åˆ†å¸ƒ
    plt.figure(figsize=(10, 6))
    colors = ['#2E8B57', '#CD5C5C']
    plt.subplot(1, 2, 1)
    plt.pie(churn_distribution.values, labels=churn_distribution.index, autopct='%1.1f%%', 
            startangle=90, colors=colors, explode=(0.05, 0))
    plt.title('å®¢æˆ·æµ�å¤±åˆ†å¸ƒ')

    plt.subplot(1, 2, 2)
    sns.countplot(data=df_clean, x='Churn', palette=colors)
    plt.title('å®¢æˆ·æµ�å¤±æ•°é‡�åˆ†å¸ƒ')
    plt.xlabel('æ˜¯å�¦æµ�å¤±')
    plt.ylabel('å®¢æˆ·æ•°é‡�')

    plt.tight_layout()
    plt.show()
else:
    print("âš  æ•°æ�®é›†ä¸­æ²¡æœ‰æ‰¾åˆ°'Churn'åˆ—")



# ç›®æ ‡å�˜é‡�åˆ†å¸ƒç»Ÿè®¡
churn_distribution = df_clean['churn'].value_counts()
churn_percentage = df_clean['churn'].value_counts(normalize=True) * 100

# ç»˜åˆ¶é¥¼å›¾
plt.figure(figsize=(10, 6))
colors = ['#2E8B57', '#CD5C5C']
plt.pie(churn_distribution.values, labels=churn_distribution.index, autopct='%1.1f%%',
        startangle=90, colors=colors, explode=(0.05, 0))
plt.title('å®¢æˆ·æµ�å¤±åˆ†å¸ƒ')
plt.show()


# æ�¢ç´¢æ€§æ•°æ�®åˆ†æ��
# =============================================================================
print("å¼€å§‹æ�¢ç´¢æ€§æ•°æ�®åˆ†æ��...")

# é€‰æ‹©æ•°å€¼å�‹å�˜é‡�è¿›è¡Œåˆ†æ��
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
available_numeric = [col for col in numeric_cols if col in df_clean.columns]

if available_numeric:
    print(f"å�¯ç”¨çš„æ•°å€¼å�‹å�˜é‡�: {available_numeric}")
    
    # æ•°å€¼å�‹å�˜é‡�åˆ†å¸ƒ
    fig, axes = plt.subplots(2, len(available_numeric), figsize=(5*len(available_numeric), 10))
    
    for i, col in enumerate(available_numeric):
        # æ•´ä½“åˆ†å¸ƒ
        axes[0, i].hist(df_clean[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, i].set_title(f'{col} - æ•´ä½“åˆ†å¸ƒ')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('é¢‘æ•°')
        
        # æŒ‰æµ�å¤±çŠ¶æ€�åˆ†å¸ƒï¼ˆå¦‚æ�œChurnå­˜åœ¨ï¼‰
        if 'Churn' in df_clean.columns:
            for churn_status in ['Yes', 'No']:
                data = df_clean[df_clean['Churn'] == churn_status][col]
                axes[1, i].hist(data, bins=30, alpha=0.6, label=churn_status, density=True)
            
            axes[1, i].set_title(f'{col} - æŒ‰æµ�å¤±çŠ¶æ€�åˆ†å¸ƒ')
            axes[1, i].set_xlabel(col)
            axes[1, i].set_ylabel('å¯†åº¦')
            axes[1, i].legend()
        else:
            axes[1, i].text(0.5, 0.5, 'æ— Churnæ•°æ�®', ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    plt.show()



# åˆ†ç±»å�˜é‡�åˆ†æ��
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                   'PhoneService', 'InternetService', 'Contract', 'PaymentMethod']
available_categorical = [col for col in categorical_cols if col in df_clean.columns]

if available_categorical and 'Churn' in df_clean.columns:
    print(f"å�¯ç”¨çš„åˆ†ç±»å�˜é‡�: {available_categorical}")
    
    # åˆ›å»ºåˆ†ç±»å�˜é‡�å�¯è§†åŒ–
    n_cols = min(3, len(available_categorical))
    n_rows = (len(available_categorical) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(available_categorical):
        row = i // n_cols
        col_idx = i % n_cols
        
        # è®¡ç®—æµ�å¤±ç�‡
        cross_tab = pd.crosstab(df_clean[col], df_clean['Churn'], normalize='index') * 100
        cross_tab = cross_tab.sort_values('Yes', ascending=False)
        
        # ç»˜åˆ¶æ�¡å½¢å›¾
        bars = cross_tab.plot(kind='bar', ax=axes[row, col_idx], color=['#2E8B57', '#CD5C5C'])
        axes[row, col_idx].set_title(f'{col} - æµ�å¤±ç�‡åˆ†æ��')
        axes[row, col_idx].set_xlabel(col)
        axes[row, col_idx].set_ylabel('ç™¾åˆ†æ¯” (%)')
        axes[row, col_idx].tick_params(axis='x', rotation=45)
        axes[row, col_idx].legend(['æœªæµ�å¤±', 'æµ�å¤±'])
    
    # éš�è—�å¤šä½™çš„å­�å›¾
    for i in range(len(available_categorical), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# %%



# ç‰¹å¾�å·¥ç¨‹
# =============================================================================
print("å¼€å§‹ç‰¹å¾�å·¥ç¨‹...")

# å¤�åˆ¶æ¸…æ´—å��çš„æ•°æ�®
df_featured = df_clean.copy()

# åˆ›å»ºæ–°ç‰¹å¾�
# 1. å®¢æˆ·ä»·å€¼åˆ†æ®µï¼ˆå¦‚æ�œMonthlyChargeså­˜åœ¨ï¼‰
if 'MonthlyCharges' in df_featured.columns:
    df_featured['CustomerValue'] = pd.cut(df_featured['MonthlyCharges'], 
                                         bins=[0, 35, 70, 100, df_featured['MonthlyCharges'].max()], 
                                         labels=['ä½�ä»·å€¼', 'ä¸­ä»·å€¼', 'é«˜ä»·å€¼', 'è¶…é«˜ä»·å€¼'])

# 2. åœ¨ç½‘æ—¶é•¿åˆ†æ®µï¼ˆå¦‚æ�œtenureå­˜åœ¨ï¼‰
if 'tenure' in df_featured.columns:
    df_featured['TenureGroup'] = pd.cut(df_featured['tenure'], 
                                       bins=[0, 12, 24, 48, df_featured['tenure'].max()], 
                                       labels=['æ–°å®¢æˆ·(<1å¹´)', 'ç¨³å®šå®¢æˆ·(1-2å¹´)', 'å¿ è¯šå®¢æˆ·(2-4å¹´)', 'èµ„æ·±å®¢æˆ·(>4å¹´)'])

# 3. æœ�åŠ¡æ•°é‡�è®¡æ•°
service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                  'TechSupport', 'StreamingTV', 'StreamingMovies']
available_services = [col for col in service_columns if col in df_featured.columns]

if available_services:
    df_featured['TotalServices'] = df_featured[available_services].apply(
        lambda x: (x == 'Yes').sum(), axis=1)
    print(f"âœ“ åˆ›å»ºæ–°ç‰¹å¾�: TotalServices (åŸºäº�{len(available_services)}ä¸ªæœ�åŠ¡)")

print("æ–°åˆ›å»ºçš„ç‰¹å¾�:")
new_features = []
if 'CustomerValue' in df_featured.columns:
    new_features.append('CustomerValue')
if 'TenureGroup' in df_featured.columns:
    new_features.append('TenureGroup')
if 'TotalServices' in df_featured.columns:
    new_features.append('TotalServices')

for feature in new_features:
    print(f"- {feature}: {df_featured[feature].nunique()} ä¸ªç±»åˆ«")



# æ•°æ�®é¢„å¤„ç�†
# =============================================================================
print("æ•°æ�®é¢„å¤„ç�†...")

if 'Churn' not in df_featured.columns:
    print("âš  è­¦å‘Š: æ•°æ�®é›†ä¸­æ²¡æœ‰æ‰¾åˆ°ç›®æ ‡å�˜é‡�'Churn'ï¼Œæ— æ³•è¿›è¡Œå»ºæ¨¡")
    print("å°†å±•ç¤ºæ•°æ�®æ�¢ç´¢å’Œé¢„å¤„ç�†éƒ¨åˆ†...")
else:
    # åˆ†ç¦»ç‰¹å¾�å’Œç›®æ ‡å�˜é‡�
    X = df_featured.drop('Churn', axis=1)
    y = df_featured['Churn']

    # ç¼–ç �ç›®æ ‡å�˜é‡�
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"ç›®æ ‡å�˜é‡�ç¼–ç �: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # åˆ†ç¦»æ•°å€¼å�‹å’Œåˆ†ç±»ç‰¹å¾�
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    available_numeric = [col for col in numeric_features if col in X.columns]
    
    if 'TotalServices' in X.columns:
        available_numeric.append('TotalServices')
    
    categorical_features = [col for col in X.columns if col not in available_numeric]

    print(f"æ•°å€¼å�‹ç‰¹å¾� ({len(available_numeric)}): {available_numeric}")
    print(f"åˆ†ç±»ç‰¹å¾� ({len(categorical_features)}): {categorical_features}")

    # é¢„å¤„ç�†ç®¡é�“
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    # æ•°å€¼å�‹ç‰¹å¾�é¢„å¤„ç�†
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # åˆ†ç±»ç‰¹å¾�é¢„å¤„ç�†
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # ç»„å�ˆé¢„å¤„ç�†å™¨
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, available_numeric),
            ('cat', categorical_transformer, categorical_features)
        ])

    # åº”ç”¨é¢„å¤„ç�†
    X_processed = preprocessor.fit_transform(X)
    
    # è�·å�–ç‰¹å¾�å��ç§°
    numeric_feature_names = available_numeric
    categorical_feature_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
    feature_names = numeric_feature_names + categorical_feature_names

    print(f"é¢„å¤„ç�†å��ç‰¹å¾�å½¢çŠ¶: {X_processed.shape}")
    print(f"ç‰¹å¾�æ•°é‡�: {len(feature_names)}")

    # %%



# 1. å¯¼å…¥æ‰€éœ€æ¨¡å�—ï¼ˆå®Œæ•´å¯¼å…¥ï¼Œé�¿å…�æœªå®šä¹‰é”™è¯¯ï¼‰
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 2. è¯»å�–æ•°æ�®ï¼ˆæ”¯æŒ�æœ¬åœ°/ Kaggle è·¯å¾„ï¼Œæ·»åŠ å¼‚å¸¸å¤„ç�†ï¼‰
try:
    # ä¼˜å…ˆå°�è¯• Kaggle è·¯å¾„ï¼ˆæ ¹æ�®å®�é™…æ•°æ�®é›†è°ƒæ•´æ–‡ä»¶å��ï¼‰
    df = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/train.csv')
    print("âœ… æˆ�åŠŸä»� Kaggle è·¯å¾„è¯»å�–æ•°æ�®")
except FileNotFoundError:
    # è‹¥ Kaggle è·¯å¾„å¤±è´¥ï¼Œå°�è¯•æœ¬åœ°è·¯å¾„ï¼ˆæ›¿æ�¢ä¸ºä½ çš„æœ¬åœ°æ–‡ä»¶è·¯å¾„ï¼‰
    df = pd.read_csv('train.csv')  # æœ¬åœ°æ–‡ä»¶éœ€å’Œä»£ç �å�Œç›®å½•ï¼Œæˆ–å†™ç»�å¯¹è·¯å¾„
    print("âœ… æˆ�åŠŸä»�æœ¬åœ°è·¯å¾„è¯»å�–æ•°æ�®")

# 3. æ£€æµ‹æ•°æ�®åˆ—å��ï¼Œç¡®è®¤æ ‡ç­¾åˆ—ï¼ˆè§£å†³ KeyError é—®é¢˜ï¼‰
print("\nğŸ“Š æ•°æ�®æ‰€æœ‰åˆ—å��ï¼š", df.columns.tolist())
# è‡ªåŠ¨åŒ¹é…�å�¯èƒ½çš„æ ‡ç­¾åˆ—ï¼ˆå¸¸è§�æ ‡ç­¾åˆ—å��ï¼šChurnã€�churnã€�ChurnStatusã€�æ˜¯å�¦æµ�å¤±ç­‰ï¼‰
possible_label_cols = ['Churn', 'churn', 'ChurnStatus', 'æ˜¯å�¦æµ�å¤±', 'å®¢æˆ·æµ�å¤±']
label_col = None
for col in possible_label_cols:
    if col in df.columns:
        label_col = col
        break

# è‹¥æœªè‡ªåŠ¨åŒ¹é…�åˆ°ï¼Œæ‰‹åŠ¨æŒ‡å®šï¼ˆè¯·æ ¹æ�®å®�é™…åˆ—å��ä¿®æ”¹ï¼�ï¼‰
if label_col is None:
    print("\nâ�Œ æœªè‡ªåŠ¨è¯†åˆ«æ ‡ç­¾åˆ—ï¼Œè¯·æ‰‹åŠ¨ä¿®æ”¹ä»£ç �ä¸­çš„ label_col å�˜é‡�")
    label_col = 'ä½ çš„æ ‡ç­¾åˆ—å��'  # ï¼�ï¼�ï¼�å…³é”®ï¼šæ›¿æ�¢ä¸ºä½ æ•°æ�®ä¸­å®�é™…çš„æ ‡ç­¾åˆ—å��ï¼�ï¼�ï¼�
print(f"\nğŸ�¯ æœ€ç»ˆä½¿ç”¨çš„æ ‡ç­¾åˆ—ï¼š{label_col}")

# 4. æ•°æ�®é¢„å¤„ç�†ï¼ˆæ·»åŠ ç©ºå€¼å¤„ç�†ï¼Œé�¿å…�å��ç»­æŠ¥é”™ï¼‰
# åˆ†ç¦»ç‰¹å¾�å’Œæ ‡ç­¾
X = df.drop(label_col, axis=1)
y = df[label_col]

# å¤„ç�†æ ‡ç­¾ï¼ˆå­—ç¬¦ä¸²è½¬æ•°å€¼ï¼Œå…¼å®¹äºŒåˆ†ç±»/å¤šåˆ†ç±»ï¼‰
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nğŸ�·ï¸�  æ ‡ç­¾ç¼–ç �æ˜ å°„ï¼š{dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# ç‰¹å¾�é¢„å¤„ç�†ï¼ˆä»…ä¿�ç•™æ•°å€¼ç‰¹å¾�ï¼Œå¡«å……ç©ºå€¼ï¼‰
X_processed = X.select_dtypes(include=['int64', 'float64']).fillna(X.select_dtypes(include=['int64', 'float64']).mean())
print(f"\nğŸ“ˆ é¢„å¤„ç�†å��ç‰¹å¾�å½¢çŠ¶ï¼š{X_processed.shape}ï¼ˆä»…ä¿�ç•™æ•°å€¼ç‰¹å¾�å¹¶å¡«å……ç©ºå€¼ï¼‰")

# 5. è®­ç»ƒæµ‹è¯•é›†åˆ’åˆ†ï¼ˆç¡®ä¿� stratify æœ‰æ•ˆï¼Œé�¿å…�æ•°æ�®ä¸�å¹³è¡¡é—®é¢˜ï¼‰
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, 
    test_size=0.2,  # æµ‹è¯•é›†å� æ¯” 20%
    random_state=42,  # å›ºå®šéš�æœºç§�å­�ï¼Œç»“æ�œå�¯å¤�ç�°
    stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None  # åˆ†å±‚æŠ½æ ·ï¼ˆé�¿å…�å�•æ ‡ç­¾æŠ¥é”™ï¼‰
)

# 6. è¾“å‡ºå…³é”®ä¿¡æ�¯ï¼ˆéªŒè¯�åˆ’åˆ†ç»“æ�œï¼‰
print("\n" + "="*50)
print("ğŸ“‹ æ•°æ�®åˆ’åˆ†ç»“æ�œï¼š")
print(f"è®­ç»ƒé›†å¤§å°�ï¼š{X_train.shape}ï¼ˆç‰¹å¾�æ•°ï¼š{X_train.shape[1]}ï¼Œæ ·æœ¬æ•°ï¼š{X_train.shape[0]}ï¼‰")
print(f"æµ‹è¯•é›†å¤§å°�ï¼š{X_test.shape}ï¼ˆç‰¹å¾�æ•°ï¼š{X_test.shape[1]}ï¼Œæ ·æœ¬æ•°ï¼š{X_test.shape[0]}ï¼‰")
print(f"è®­ç»ƒé›†æ ‡ç­¾åˆ†å¸ƒï¼š{np.bincount(y_train)}ï¼ˆå¯¹åº”ç¼–ç �ï¼š{label_encoder.classes_}ï¼‰")
print(f"æµ‹è¯•é›†æ ‡ç­¾åˆ†å¸ƒï¼š{np.bincount(y_test)}ï¼ˆå¯¹åº”ç¼–ç �ï¼š{label_encoder.classes_}ï¼‰")
print("="*50)


 # æ¨¡å�‹è®­ç»ƒä¸�è¯„ä¼°
    # =============================================================================
 print("å¼€å§‹æ¨¡å�‹è®­ç»ƒ...")

    # å®šä¹‰è¦�æ¯”è¾ƒçš„æ¨¡å�‹
models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    # å­˜å‚¨ç»“æ�œ
 results = {}



# éœ€æ��å‰�å¯¼å…¥è¯„ä¼°æŒ‡æ ‡æ¨¡å�—ï¼ˆç¡®ä¿�å¯¼å…¥å®Œæ•´ï¼Œæ— é�—æ¼�ï¼‰
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# è®­ç»ƒå’Œè¯„ä¼°æ¯�ä¸ªæ¨¡å�‹
# åˆ�å§‹åŒ–ç»“æ�œå­˜å‚¨å­—å…¸
results = {}
for name, model in models.items():
    print(f"è®­ç»ƒ {name}...")
    
    # è®­ç»ƒæ¨¡å�‹ï¼ˆç»Ÿä¸€4ä¸ªç©ºæ ¼ç¼©è¿›ï¼‰
    model.fit(X_train, y_train)
    
    # é¢„æµ‹ï¼ˆç»Ÿä¸€ç¼©è¿›ï¼Œé�¿å…�æ··å�ˆç©ºæ ¼/åˆ¶è¡¨ç¬¦ï¼‰
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # å�–æ­£ç±»æ¦‚ç�‡
    
    # è®¡ç®—è¯„ä¼°æŒ‡æ ‡ï¼ˆè¡¥å……averageå�‚æ•°ï¼Œé�¿å…�äºŒåˆ†ç±»/å¤šåˆ†ç±»æ­§ä¹‰ï¼‰
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')  # äºŒåˆ†ç±»ç”¨binaryï¼Œå¤šåˆ†ç±»éœ€è°ƒæ•´
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # å­˜å‚¨ç»“æ�œï¼ˆå­—å…¸æ ¼å¼�å®Œæ•´ï¼Œæ— ç¼ºå¤±é€—å�·ï¼‰
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # è¾“å‡ºä¿¡æ�¯ï¼ˆå­—ç¬¦ä¸²æ ¼å¼�åŒ–æ— è¯­æ³•é”™è¯¯ï¼‰
    print(f"  {name} - AUC: {auc_score:.4f}, F1: {f1:.4f}, å‡†ç¡®ç�‡: {accuracy:.4f}")
    print("-" * 50)  # ç¬¬39è¡Œï¼šä¿®æ­£æ½œåœ¨çš„æ ¼å¼�é”™è¯¯ï¼Œç¡®ä¿�ä»…ç”¨ç©ºæ ¼ç¼©è¿›


# total_day_minutesç›´æ–¹å›¾ï¼ˆæŒ‰æµ�å¤±æ ‡ç­¾ç�€è‰²ï¼‰
plt.figure(figsize=(10, 6))
# æœªæµ�å¤±å®¢æˆ·
plt.hist(df_clean[df_clean['churn']=='no']['total_day_minutes'], 
         bins=30, alpha=0.7, color='skyblue', label='æœªæµ�å¤±')
# æµ�å¤±å®¢æˆ·
plt.hist(df_clean[df_clean['churn']=='yes']['total_day_minutes'], 
         bins=30, alpha=0.7, color='salmon', label='æµ�å¤±')
plt.xlabel('ç™½å¤©é€šè¯�æ—¶é•¿ï¼ˆåˆ†é’Ÿï¼‰')
plt.ylabel('å®¢æˆ·æ•°é‡�')
plt.title('ç™½å¤©é€šè¯�æ—¶é•¿åˆ†å¸ƒä¸�æµ�å¤±å…³è�”')
plt.legend()
plt.show()

# é€šè¯�æ—¶é•¿ä¸�è´¹ç”¨ç›¸å…³æ€§
print(df_clean[['total_day_minutes', 'total_day_charge']].corr())


# æ¨¡å�‹æ€§èƒ½æ¯”è¾ƒ
# =============================================================================
print("æ¨¡å�‹æ€§èƒ½æ¯”è¾ƒ:")

# åˆ›å»ºæ¯”è¾ƒè¡¨æ ¼ï¼ˆä¿®æ­£ç¼©è¿›ï¼šä¸�å¤–å±‚ä»£ç �å·¦å¯¹é½�ï¼‰
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'Precision': [results[name]['precision'] for name in results.keys()],
    'Recall': [results[name]['recall'] for name in results.keys()],
    'F1-Score': [results[name]['f1_score'] for name in results.keys()],
    'AUC': [results[name]['auc'] for name in results.keys()]
}).sort_values('AUC', ascending=False)

display(comparison_df)


# éœ€æ��å‰�å¯¼å…¥ç»˜å›¾æ¨¡å�—å’Œpandasï¼ˆè‹¥æœªå¯¼å…¥ï¼‰
import pandas as pd
import matplotlib.pyplot as plt

# å�¯è§†åŒ–æ¨¡å�‹æ¯”è¾ƒï¼ˆä¿®æ­£ç¼©è¿›ï¼šä¸�å¤–å±‚ä»£ç �å·¦å¯¹é½�ï¼‰
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# AUCæ¯”è¾ƒ
models_sorted = comparison_df['Model'].values
auc_scores = comparison_df['AUC'].values
axes[0, 0].barh(models_sorted, auc_scores, color='skyblue')
axes[0, 0].set_xlabel('AUC Score')
axes[0, 0].set_title('æ¨¡å�‹AUCæ€§èƒ½æ¯”è¾ƒ')
axes[0, 0].set_xlim(0, 1)
# åœ¨æ�¡å½¢å›¾ä¸Šæ·»åŠ æ•°å€¼æ ‡ç­¾ï¼ˆä¼˜åŒ–ï¼‰
for i, v in enumerate(auc_scores):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center')

# F1-Scoreæ¯”è¾ƒ
f1_scores = comparison_df['F1-Score'].values
axes[0, 1].barh(models_sorted, f1_scores, color='lightcoral')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_title('æ¨¡å�‹F1-Scoreæ€§èƒ½æ¯”è¾ƒ')
axes[0, 1].set_xlim(0, 1)
# æ·»åŠ æ•°å€¼æ ‡ç­¾
for i, v in enumerate(f1_scores):
    axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center')

# è¡¥å……å‰©ä½™2ä¸ªå­�å›¾ï¼ˆå‡†ç¡®ç�‡ã€�ç²¾ç¡®ç�‡+å�¬å›�ç�‡å¯¹æ¯”ï¼Œå®Œå–„2x2å¸ƒå±€ï¼‰
# å‡†ç¡®ç�‡æ¯”è¾ƒ
acc_scores = comparison_df['Accuracy'].values
axes[1, 0].barh(models_sorted, acc_scores, color='lightgreen')
axes[1, 0].set_xlabel('Accuracy')
axes[1, 0].set_title('æ¨¡å�‹å‡†ç¡®ç�‡æ€§èƒ½æ¯”è¾ƒ')
axes[1, 0].set_xlim(0, 1)
for i, v in enumerate(acc_scores):
    axes[1, 0].text(v + 0.01, i, f'{v:.4f}', va='center')

# ç²¾ç¡®ç�‡+å�¬å›�ç�‡å¯¹æ¯”ï¼ˆå�Œæ�¡å½¢å›¾ï¼‰
x = range(len(models_sorted))
width = 0.35
axes[1, 1].bar([i - width/2 for i in x], comparison_df['Precision'], width, label='Precision', color='gold')
axes[1, 1].bar([i + width/2 for i in x], comparison_df['Recall'], width, label='Recall', color='orange')
axes[1, 1].set_xlabel('æ¨¡å�‹')
axes[1, 1].set_ylabel('åˆ†æ•°')
axes[1, 1].set_title('æ¨¡å�‹ç²¾ç¡®ç�‡vså�¬å›�ç�‡å¯¹æ¯”')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models_sorted, rotation=45, ha='right')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()
# æ·»åŠ æ•°å€¼æ ‡ç­¾
for i, (p, r) in enumerate(zip(comparison_df['Precision'], comparison_df['Recall'])):
    axes[1, 1].text(i - width/2, p + 0.01, f'{p:.4f}', ha='center', va='bottom', fontsize=9)
    axes[1, 1].text(i + width/2, r + 0.01, f'{r:.4f}', ha='center', va='bottom', fontsize=9)

# è°ƒæ•´å­�å›¾é—´è·�ï¼Œé�¿å…�æ ‡ç­¾é‡�å� 
plt.tight_layout()
# æ˜¾ç¤ºå›¾ç‰‡ï¼ˆå¿…è¦�æ—¶ä¿�å­˜ï¼‰
plt.show()
# plt.savefig('æ¨¡å�‹æ€§èƒ½å¯¹æ¯”å›¾.png', dpi=300, bbox_inches='tight')  # ä¿�å­˜å›¾ç‰‡


# éœ€æ��å‰�å¯¼å…¥ç¼ºå¤±æ¨¡å�—
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# å‡†ç¡®ç�‡å’Œå�¬å›�ç�‡æ¯”è¾ƒï¼ˆä¿®æ­£ç¼©è¿›ï¼šä¸�å¤–å±‚ä»£ç �å·¦å¯¹é½�ï¼‰
width = 0.35
x = np.arange(len(models_sorted))
axes[1, 0].bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy', alpha=0.7)
axes[1, 0].bar(x + width/2, comparison_df['Recall'], width, label='Recall', alpha=0.7)
axes[1, 0].set_xlabel('Models')
axes[1, 0].set_ylabel('Scores')
axes[1, 0].set_title('å‡†ç¡®ç�‡ vs å�¬å›�ç�‡')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(models_sorted, rotation=45, ha='right')  # æ–°å¢�ha='right'é�¿å…�æ ‡ç­¾é‡�å� 
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1)  # å›ºå®šyè½´èŒƒå›´ï¼Œç»Ÿä¸€å¯¹æ¯”æ ‡å‡†

# ROCæ›²çº¿ï¼ˆä¿®æ­£ç¼©è¿›ï¼‰
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='éš�æœºåˆ†ç±»å™¨')
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    auc_score = result['auc']
    axes[1, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

axes[1, 1].set_xlabel('å�‡æ­£ç�‡')
axes[1, 1].set_ylabel('çœŸæ­£ç�‡')
axes[1, 1].set_title('ROCæ›²çº¿æ¯”è¾ƒ')
axes[1, 1].legend(loc='lower right')  # è°ƒæ•´å›¾ä¾‹ä½�ç½®ï¼Œé�¿å…�é�®æŒ¡æ›²çº¿
axes[1, 1].grid(True, alpha=0.3)  # é™�ä½�ç½‘æ ¼é€�æ˜�åº¦ï¼Œæ��å�‡å�¯è¯»æ€§
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1.05)

plt.tight_layout()
plt.show()



  # æœ€ä½³æ¨¡å�‹åˆ†æ��
    # =============================================================================
    # é€‰æ‹©æœ€ä½³æ¨¡å�‹
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']
y_pred_proba_best = results[best_model_name]['y_pred_proba']

print(f"æœ€ä½³æ¨¡å�‹: {best_model_name}")
print(f"æœ€ä½³æ¨¡å�‹AUC: {results[best_model_name]['auc']:.4f}")

    # æ··æ·†çŸ©é˜µ
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['é¢„æµ‹æœªæµ�å¤±', 'é¢„æµ‹æµ�å¤±'], 
            yticklabels=['å®�é™…æœªæµ�å¤±', 'å®�é™…æµ�å¤±'])
plt.title(f'{best_model_name} - æ··æ·†çŸ©é˜µ')
plt.ylabel('çœŸå®�æ ‡ç­¾')
plt.xlabel('é¢„æµ‹æ ‡ç­¾')
plt.show()




# éœ€æ��å‰�å¯¼å…¥ç¼ºå¤±æ¨¡å�—å’Œå®šä¹‰æœªæ˜�ç¡®å�˜é‡�
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

# 1. å®šä¹‰å…³é”®å�˜é‡�ï¼ˆå‰�æ–‡æœªæ˜�ç¡®ï¼Œéœ€è¡¥å……ï¼‰
# ä»�resultså­—å…¸ä¸­ç­›é€‰AUCæœ€é«˜çš„æœ€ä¼˜æ¨¡å�‹
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']  # æœ€ä¼˜æ¨¡å�‹çš„é¢„æµ‹ç»“æ�œ
feature_names = X_processed.columns.tolist()  # ç‰¹å¾�å��ç§°ï¼ˆæ�¥è‡ªé¢„å¤„ç�†å��çš„ç‰¹å¾�æ•°æ�®ï¼‰

# åˆ†ç±»æŠ¥å‘Šï¼ˆä¿®æ­£ç¼©è¿›ï¼Œç¡®ä¿�è¯­æ³•æ­£ç¡®ï¼‰
print(f"\n{best_model_name} åˆ†ç±»æŠ¥å‘Š:")
print(classification_report(
    y_test, y_pred_best,
    target_names=['æœªæµ�å¤±', 'æµ�å¤±'],  # é€‚é…�äºŒåˆ†ç±»åœºæ™¯çš„æ ‡ç­¾å��ç§°
    digits=4  # ä¿�ç•™4ä½�å°�æ•°ï¼Œæ��å�‡ç²¾åº¦å�¯è¯»æ€§
))

# ç‰¹å¾�é‡�è¦�æ€§ï¼ˆå¦‚æ�œæ¨¡å�‹æ”¯æŒ�ï¼‰
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)  # å�–Top15é‡�è¦�ç‰¹å¾�
    
    # ç»˜åˆ¶ç‰¹å¾�é‡�è¦�æ€§å›¾ï¼ˆä¿®æ­£ç¼©è¿›ï¼Œå®Œå–„æ˜¾ç¤ºç»†èŠ‚ï¼‰
    plt.figure(figsize=(10, 8))
    bars = plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
    plt.xlabel('ç‰¹å¾�é‡�è¦�æ€§', fontsize=12)
    plt.title(f'{best_model_name} - å‰�15ä¸ªæœ€é‡�è¦�ç‰¹å¾�', fontsize=14, pad=20)
    plt.gca().invert_yaxis()  # ä»�ä¸Šåˆ°ä¸‹æŒ‰é‡�è¦�æ€§é™�åº�æ�’åˆ—
    plt.grid(axis='x', alpha=0.3)  # æ·»åŠ xè½´ç½‘æ ¼ï¼Œæ��å�‡å�¯è¯»æ€§
    
    # åœ¨æ�¡å½¢å›¾ä¸Šæ·»åŠ æ•°å€¼æ ‡ç­¾
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
else:
    print(f"\n{best_model_name} ä¸�æ”¯æŒ�ç‰¹å¾�é‡�è¦�æ€§è®¡ç®—ï¼ˆè¯¥æ¨¡å�‹æ— feature_importances_å±�æ€§ï¼‰")



# é¡¹ç›®æ€»ç»“
# =============================================================================
print("=" * 60)
print("é¡¹ç›®å®Œæˆ�æ€»ç»“")
print("=" * 60)

if 'Churn' in df_clean.columns:
    achievements = [
        ("æ•°æ�®åŠ è½½ä¸�æ¸…æ´—", "âœ“ å®Œæˆ�"),
        ("æ�¢ç´¢æ€§æ•°æ�®åˆ†æ��", "âœ“ å®Œæˆ�"),
        ("ç‰¹å¾�å·¥ç¨‹", "âœ“ å®Œæˆ�"),
        ("æ¨¡å�‹è®­ç»ƒ", "âœ“ å®Œæˆ�"),
        ("æ¨¡å�‹è¯„ä¼°", "âœ“ å®Œæˆ�"),
        ("ç»“æ�œåˆ†æ��", "âœ“ å®Œæˆ�")
    ]
    
    if 'results' in locals():
        best_auc = results[best_model_name]['auc']
        best_f1 = results[best_model_name]['f1_score']
        print(f"æœ€ä½³æ¨¡å�‹: {best_model_name}")
        print(f"æœ€ä½³AUC: {best_auc:.4f}")
        print(f"æœ€ä½³F1-Score: {best_f1:.4f}")
else:
    achievements = [
        ("æ•°æ�®åŠ è½½ä¸�æ¸…æ´—", "âœ“ å®Œæˆ�"),
        ("æ�¢ç´¢æ€§æ•°æ�®åˆ†æ��", "âœ“ å®Œæˆ�"),
        ("ç‰¹å¾�å·¥ç¨‹", "âœ“ å®Œæˆ�"),
        ("æ¨¡å�‹è®­ç»ƒ", "âš  ç¼ºå°‘ç›®æ ‡å�˜é‡�"),
        ("å®Œæ•´åˆ†æ��æŠ¥å‘Š", "âœ“ å®Œæˆ�")
    ]

for achievement, status in achievements:
    print(f"{achievement:.<30} {status}")



# è®¡ç®—international_planæµ�å¤±ç�‡
intl_churn = pd.crosstab(df_clean['international_plan'], df_clean['churn'], normalize='index') * 100

plt.figure(figsize=(8, 5))
intl_churn['yes'].plot(kind='bar', color=['lightcoral', 'lightgreen'])
plt.xlabel('æ˜¯å�¦å¼€é€šå›½é™…å¥—é¤�')
plt.ylabel('æµ�å¤±ç�‡ï¼ˆ%ï¼‰')
plt.title('å›½é™…å¥—é¤�ä¸�å®¢æˆ·æµ�å¤±ç�‡å…³è�”')
plt.xticks(rotation=0)
plt.show()


# ========== ç”Ÿæˆ�ç«�èµ›è¾“å‡ºæ–‡ä»¶ ==========
# 1. è¯»å�–æµ‹è¯•é›†ï¼ˆæ›¿æ�¢ä¸ºä½ çš„æµ‹è¯•é›†è·¯å¾„ï¼‰
test_df = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/test.csv')

# 2. æµ‹è¯•é›†é¢„å¤„ç�†ï¼ˆä¸�è®­ç»ƒé›†å®Œå…¨ä¸€è‡´ï¼Œä¸”åˆ é™¤idåˆ—ï¼‰
test_id = test_df['id']  # æ��å�–æ��äº¤ç”¨çš„id
test_features = test_df.drop('id', axis=1)  # æ�’é™¤idåˆ—ï¼ˆé��ç‰¹å¾�ï¼‰
# è¿™é‡Œçš„é¢„å¤„ç�†é€»è¾‘å¿…é¡»å’Œè®­ç»ƒé›†X_processedå®Œå…¨ç›¸å�Œ
test_df_processed = test_features.select_dtypes(include=['int64', 'float64']).fillna(
    test_features.select_dtypes(include=['int64', 'float64']).mean()
)

# 3. ç”¨æœ€ä¼˜æ¨¡å�‹é¢„æµ‹
best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
best_model = results[best_model_name]['model']
test_pred = best_model.predict(test_df_processed)

# 4. ç”Ÿæˆ�æ��äº¤æ–‡ä»¶ï¼ˆåˆ—å��éœ€åŒ¹é…�ç«�èµ›è¦�æ±‚ï¼‰
submission = pd.DataFrame({
    'id': test_id,
    'Churn': label_encoder.inverse_transform(test_pred)  # è¿˜å�Ÿä¸ºå�Ÿå§‹æ ‡ç­¾ï¼ˆå¦‚"Yes"/"No"ï¼‰
})
submission.to_csv('submission.csv', index=False)  # ä¿�å­˜ä¸ºCSV

print("âœ… è¾“å‡ºæ–‡ä»¶å·²ç”Ÿæˆ�ï¼šsubmission.csv")

