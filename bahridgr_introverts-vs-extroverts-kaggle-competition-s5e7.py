import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import math
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna
from sklearn.model_selection import cross_val_score


warnings.filterwarnings("ignore")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

train_id = train_df['id']
train_df.drop('id', axis=1, inplace=True)

test_id = test_df['id']
test_df.drop('id', axis=1, inplace=True)


# Checking the basic properties of data 
def check_data(dataframe):
    print("########################## HEAD ##########################")
    print(dataframe.head(3))
    print("########################## ISNULL(?) ##########################")
    print(dataframe.isna().sum())
    print("########################## INFO ##########################")
    print(dataframe.info())
    print("########################## SHAPE ##########################")
    print(dataframe.shape)
    print("########################## DESCRÄ°BE ##########################")
    print(dataframe.describe([0.1, 0.25, 0.5, 0.75, 0.90, 0.99]).T)


check_data(train_df)


def explore_dataset(df, title):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=False)
    fig.suptitle(f"{title} Veri Seti - Kategorik ve SayÄ±sal GÃ¶rselleÅŸtirme", fontsize=16)
    fig.subplots_adjust(top=0.85)
    
    # Kategorik deÄŸiÅŸkenler
    if cat_cols:
        sns.countplot(data=df[cat_cols].melt(var_name='variable', value_name='value'),
                      x='value', hue='variable', ax=axes[0])
        axes[0].set_title('Kategorik DeÄŸiÅŸken DaÄŸÄ±lÄ±mÄ±')
        axes[0].tick_params(axis='x', rotation=45)
    else:
        axes[0].axis('off')
    
    # SayÄ±sal deÄŸiÅŸkenler
    if num_cols:
        df[num_cols].hist(bins=20, ax=axes[1])
        axes[1].set_title('SayÄ±sal DeÄŸiÅŸken HistogramlarÄ±')
    else:
        axes[1].axis('off')
    
    plt.show()



explore_dataset(train_df, "Train")


explore_dataset(test_df, "Test")


train_df.isnull().sum()


# SayÄ±sal deÄŸiÅŸkenleri seÃ§iyoruz (Ã§Ã¼nkÃ¼ KNN sadece sayÄ±sal ile Ã§alÄ±ÅŸÄ±r)
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# KNN ile doldurulacak DataFrame
imputer = KNNImputer(n_neighbors=5)
train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])


# Kategorik sÃ¼tunlarÄ± alalÄ±m (object tipli)
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

# EÄŸer varsa, mod (en sÄ±k deÄŸer) ile doldur
if cat_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])



# Eksik deÄŸer var mÄ± tekrar kontrol edelim
train_df.isnull().sum()


def plot_boxplots_grid(df, numeric_cols, cols_per_row=3):
    n_cols = len(numeric_cols)
    n_rows = math.ceil(n_cols / cols_per_row)

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(cols_per_row * 5, n_rows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x=col, ax=axes[idx], color='lightblue')
        axes[idx].set_title(f'Boxplot - {col}', fontsize=12)
        axes[idx].tick_params(axis='x', labelrotation=15)

    # BoÅŸ kalan subplotlarÄ± gizle
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('TÃ¼m SayÄ±sal DeÄŸiÅŸkenler Ä°Ã§in Boxplotlar', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


plot_boxplots_grid(df=train_df, numeric_cols=numeric_cols)


def outliers_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def check_outliers(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in numeric_cols:
    print(col, check_outliers(train_df, col))
    if check_outliers(train_df, col) == False:
        replace_with_threshold(train_df, col)


train_df.head()


train_df.describe().T


def categorize_aloneness(x):
    if x <= 3:
        return "Social"
    elif x <= 7:
        return "Balanced"
    else:
        return "Introvert"
train_df['NEW_Alone_Level'] = train_df['Time_spent_Alone'].apply(categorize_aloneness)


train_df['NEW_Social_event_attendance_Level'] = pd.cut(train_df['Social_event_attendance'],
                                          bins=[-1, 3, 7, 10],
                                          labels=['Low', 'Medium', 'High'])



train_df['NEW_Friend_Group'] = pd.cut(train_df['Friends_circle_size'],
                                  bins=[-1, 4, 8, 15],
                                  labels=['Small', 'Medium', 'Large'])


train_df['NEW_Post_Freq_Level'] = pd.cut(train_df['Post_frequency'],
                                     bins=[-1, 2, 5, 10],
                                     labels=['Low', 'Medium', 'High'])


social_cols = [
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Post_frequency'
]

binary_cols = ['Stage_fear', 'Drained_after_socializing']

# Her sÃ¼tun iÃ§in ayrÄ± LabelEncoder (genel alÄ±ÅŸkanlÄ±ktÄ±r, tek encoder bazÄ± modellerde ters dÃ¶nÃ¼ÅŸÃ¼mde sorun yaratabilir)
for col in binary_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])

train_df['NEW_Social_Score'] = train_df[social_cols].mean(axis=1)

train_df.head()


# One-hot encoding yapÄ±lacak kategorik sÃ¼tunlar
categorical_columns = [
    'NEW_Alone_Level',
    'NEW_Social_event_attendance_Level',
    'NEW_Friend_Group',
    'NEW_Post_Freq_Level'
]

# 1. OneHotEncoder nesnesi
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')

# 2. Sadece kategorik sÃ¼tunlarÄ± encode et
ohe_array = ohe.fit_transform(train_df[categorical_columns])

# 3. Yeni sÃ¼tun isimlerini al
ohe_columns = ohe.get_feature_names_out(categorical_columns)

# 4. DataFrame olarak dÃ¶nÃ¼ÅŸtÃ¼r (index'e dikkat!)
ohe_df = pd.DataFrame(ohe_array, columns=ohe_columns, index=train_df.index)

# 5. Orijinal train_df'ten eski kategorik sÃ¼tunlarÄ± Ã§Ä±kar, yenileri ekle
train_df = train_df.drop(columns=categorical_columns)
train_df = pd.concat([train_df, ohe_df], axis=1)


le = LabelEncoder()
train_df['Personality'] = le.fit_transform(train_df['Personality'])


train_df.head()


X = train_df.drop('Personality', axis=1)
y = train_df['Personality'] # Extrovert:1, Introvert:0 gibi

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=7)

# Models list
models = {
    'XGBoost (GPU)': XGBClassifier(random_state=7, tree_method='gpu_hist', predictor='gpu_predictor', verbosity=0),
    'CatBoost (GPU)': CatBoostClassifier(random_state=7, task_type='GPU', devices='0', verbose=False),
    'LightGBM (GPU)': LGBMClassifier(random_state=7, device='gpu', verbose=-1),
    'RandomForest': RandomForestClassifier(random_state=7),
    'GradientBoosting': GradientBoostingClassifier(random_state=7),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=7),
    'SVC': SVC(probability=True, random_state=7),
    'KNN': KNeighborsClassifier()
}
# 4. EÄŸitim ve deÄŸerlendirme
for name, model in models.items():
    print(f'------------ {name} ----------------')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    print('############################################')


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 0,
        'random_seed': 7,
        'allow_writing_files': False
    }

    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X_train, y_train, scoring='f1_macro', cv=3, n_jobs=1).mean()
    print(f"Trial {trial.number}: F1_macro = {score:.5f} with params = {params}")

    return score


study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50, show_progress_bar=True)


# print("Best F1 Score:", study.best_value)
# print("Best Params:", study.best_params)


# En iyi parametreleri al

best_params={
            'iterations': 760,
            'depth': 6,
            'learning_rate': 0.015057252897842722,
            'l2_leaf_reg': 3.199155644798831,
            'border_count': 168,
            'random_strength': 0.002006600084079399,
            'bagging_temperature': 0.9692067817097408,
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 0,
            'random_seed': 7,
            'allow_writing_files': False}


# En iyi CatBoost modelini oluÅŸtur
final_model = CatBoostClassifier(**best_params)
final_model.fit(X, y)


import joblib

joblib.dump(final_model, "catboost_best_model.pkl")


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
plot_importance(final_model, X_train)


def preprocessing(data, ohe):
    print("Preprocessing started...")

    # ====================
    # 1. MISSING VALUE IMPUTATION
    # ====================
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    imputer = KNNImputer(n_neighbors=5)
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

    # ====================
    # 2. OUTLIER HANDLING
    # ====================
    def outliers_thresholds(df, variable, q1=0.10, q3=0.90):
        quartile1 = df[variable].quantile(q1)
        quartile3 = df[variable].quantile(q3)
        iqr = quartile3 - quartile1
        up = quartile3 + 1.5 * iqr
        low = quartile1 - 1.5 * iqr
        return low, up

    def replace_with_threshold(df, variable):
        low, up = outliers_thresholds(df, variable)
        df.loc[df[variable] < low, variable] = low
        df.loc[df[variable] > up, variable] = up

    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            replace_with_threshold(data, col)

    # ====================
    # 3. FEATURE ENGINEERING
    # ====================
    data['NEW_Alone_Level'] = data['Time_spent_Alone'].apply(lambda x: "Social" if x <= 3 else "Balanced" if x <= 7 else "Introvert")

    data['NEW_Social_event_attendance_Level'] = pd.cut(data['Social_event_attendance'],
                                                       bins=[-1, 3, 7, 10],
                                                       labels=['Low', 'Medium', 'High'])

    data['NEW_Friend_Group'] = pd.cut(data['Friends_circle_size'],
                                      bins=[-1, 4, 8, 15],
                                      labels=['Small', 'Medium', 'Large'])

    data['NEW_Post_Freq_Level'] = pd.cut(data['Post_frequency'],
                                         bins=[-1, 2, 5, 10],
                                         labels=['Low', 'Medium', 'High'])

    # Binary kategorik deÄŸiÅŸkenler (Yes/No)
    binary_cols = ['Stage_fear', 'Drained_after_socializing']
    for col in binary_cols:
        data[col] = data[col].map({'Yes': 1, 'No': 0})

    social_cols = [
        'Social_event_attendance',
        'Going_outside',
        'Drained_after_socializing',
        'Friends_circle_size',
        'Post_frequency'
    ]
    data['NEW_Social_Score'] = data[social_cols].mean(axis=1)

    # ====================
    # 4. ONE-HOT ENCODING (Scikit-learn ile)
    # ====================
    categorical_columns = [col for col in data.columns if data[col].dtype in ['object', 'category']]
    ohe_array = ohe.transform(data[categorical_columns])
    ohe_columns = ohe.get_feature_names_out(categorical_columns)
    ohe_df = pd.DataFrame(ohe_array, columns=ohe_columns, index=data.index)

    data = data.drop(columns=categorical_columns)
    data = pd.concat([data, ohe_df], axis=1)

    print("Preprocessing finished.")
    return data


test_processed = preprocessing(test_df.copy(), ohe)


y_pred = final_model.predict(test_processed)
test_preds_labels = le.inverse_transform(y_pred)


assert len(test_id) == len(test_processed), "ID ile veri seti uzunluÄŸu uyuÅŸmuyor!"


submission = pd.DataFrame({
    "id": test_id,
    "Personality": test_preds_labels
})

submission.to_csv("submission.csv", index=False)


