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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


df_delay_79 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
df_not_delay_79 = pd.read_csv('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')


data_79 = pd.concat([df_delay_79, df_not_delay_79], ignore_index=True)


def plot_missing_values(df):
    cols = df.columns
    count = [df[col].isnull().sum() for col in cols]
    percent = [i / len(df) for i in count]
    
    missing = pd.DataFrame({'proportion': percent}, index=cols)
    # Chá»‰ giá»¯ láº¡i cÃ¡c cá»™t cÃ³ giÃ¡ trá»‹ thiáº¿u
    missing = missing[missing['proportion'] > 0]
    
    if missing.empty:
        print("KhÃ´ng cÃ³ cá»™t nÃ o bá»‹ thiáº¿u giÃ¡ trá»‹.")
        return missing, []

    # Sáº¯p xáº¿p giáº£m dáº§n theo tá»‰ lá»‡ missing
    missing = missing.sort_values(by='proportion', ascending=False)
    
    plt.figure(figsize=(12, 0.5 * len(missing)))  # chiá»�u cao Ä‘á»™ng theo sá»‘ cá»™t
    plt.title('Missing values in columns', fontsize=14)
    ax = sns.barplot(x=missing['proportion'], y=missing.index, palette='viridis')

    for i, p in enumerate(ax.patches):
        ax.text(p.get_x() + p.get_width() + 0.01, 
                p.get_y() + p.get_height() / 2, 
                f"{missing.iloc[i]['proportion']*100:.2f}%", 
                va='center')

    mean = np.mean(missing['proportion'])
    std = np.std(missing['proportion'])

    plt.xlabel('Proportion of Missing Values')
    plt.ylabel('Columns')
    plt.plot([], [], ' ', label=f'Average missing: {mean:.2f} Â± {std:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return missing, missing.index.tolist()


missing_train_79, train_cols_79 = plot_missing_values(data_79)


# Xá»­ lÃ½ thay tháº¿ cÃ¡c vá»‹ trÃ­ trá»‘ng thÃ nh nan
data_79 = data_79.replace(r'^\s*$', np.nan, regex=True)


eliminated_cols_by_missing = ['QTUF_RCV_NO', 'SOUF_RCV_NO', 'REASON_CD', 'SHIP DECISION NO']


data_79.drop(columns = eliminated_cols_by_missing, inplace=True)


data_79 = data_79.dropna(subset=['Ship Mode','SUPPLIER_DIV'])


data_79['OTHER AREA SHIP DIV'] = data_79['OTHER AREA SHIP DIV'].fillna('0')


attribute_types = {
    "Order date": "datetime",
    'SUBSIDIARY_CD': "category",
    "GLOBAL_NO": "category",
    "CLASSIFY_CD": "category",
    "BRAND_CD": "category",
    "INNER_CD": "category",
    "CUST_CD": "category",
    "SUPPLIER_CD": "category",
    "Sales order line number": "numerical",
    "Stock class": "category",
    "Consider count hodiday Saturday": "numerical",
    "SO QTY": "numerical",
    "OTHER AREA SHIP DIV": "category",
    "ALLOCATION QTY": "numerical",
    "SUPPLIER INV AMOUNT": "numerical",
    "PACKING RANK": "category",
    "PRODUCT_CD": "category",
    "PRODUCT ATTRIBUTION": "category",
    "SPECIAL DIV": "category",
    "LOGICAL PLANT": "category",
    "PURCHASE AMOUNT": "numerical",
    "VSD": "datetime",
    "DIRECT SHIP FLG": "category",
    "DELI_DIV": "category",
    "label": "category",
    "Ship Mode": "category",
    "PACK QTY": "numerical",
    "WEIGHT PER PIECE": "numerical",
    "SUPPLIER_DIV": "category",
    "SPECIAL_DIV": "category",
    "SO_TIME": "numerical",
    "SO_DAY_OF_MONTH": "numerical",
    "SO_DAY_OF_WEEK": "numerical"
}


class DataTypeConverter:
    def __init__(self, type_mapping):
        self.type_mapping = type_mapping

    def convert(self, df):
        df = df.copy()
        for col, dtype in self.type_mapping.items():
            try:
                if dtype == 'datetime':
                    # Convert mixed formats and normalize (drop time part)
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.normalize()
                elif dtype == 'category':
                    df[col] = df[col].astype('category')
                elif dtype == 'numerical':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"â�Œ Failed to convert column {col} to {dtype}: {e}")

        return df


converter = DataTypeConverter(attribute_types)


df_79 = converter.convert(data_79)


df_79.drop(columns = 'SUBSIDIARY_CD', inplace=True)


import warnings

def iv_woe(data, target, bins=10, show_woe=False, plot_iv=True):
    # áº¨n cÃ¡c cáº£nh bÃ¡o FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning)

    data = data.copy()
    data[target] = data[target].astype(int)  # Fix: Ensure target is numeric

    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns
    strong_features = []

    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})

        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)

        total_iv = d['IV'].sum()
        print("Information value of " + ivars + " is " + str(round(total_iv, 6)))

        temp = pd.DataFrame({"Variable": [ivars], "IV": [total_iv]})
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        if total_iv >= 0.3:
            strong_features.append(ivars)

        if show_woe:
            print(d)

    newDF.reset_index(drop=True, inplace=True)

    if plot_iv:
        plt.figure(figsize=(10, 6))
        sorted_df = newDF.sort_values(by='IV', ascending=False)
        plt.barh(sorted_df['Variable'], sorted_df['IV'], color='skyblue')
        plt.xlabel("Information Value (IV)")
        plt.title("Information Value of Features")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    if strong_features:
        print("\nğŸ’ª Strong predictive features (IV >= 0.3):")
        for feat in strong_features:
            print(f" - {feat}")
    else:
        print("\nâš ï¸� No strong features found (IV >= 0.3).")

    return newDF, woeDF


iv_79, woe_79 = iv_woe(df_79, target='label', bins=20, show_woe=False, plot_iv=True)


def split_columns_by_type(df):
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [col for col in categorical_cols if col not in ['label']]
    return cat_cols, numerical_cols


useful_cols = [
    'GLOBAL_NO',
    'CLASSIFY_CD',
    'Order date',
    'BRAND_CD',
    'Sales order line number',
    'VSD',
    'DIRECT SHIP FLG',
    'PRODUCT ATTRIBUTION',
    'DELI_DIV',
    'OTHER AREA SHIP DIV',
    'SUPPLIER_DIV',
    'SPECIAL DIV',
    'SO_TIME',
    'SUPPLIER INV AMOUNT',
    'SO_DAY_OF_MONTH',
    'SO_DAY_OF_WEEK',
    'label'
]


df_train_on_79 = df_79[useful_cols]


datetime_cols = ['Order date', 'VSD']


df_train_on_79['range_day'] = (df_train_on_79['VSD'] - df_train_on_79['Order date']).dt.days.astype(int)
df_train_on_79['order month'] = df_train_on_79['Order date'].dt.month.astype(int)
df_train_on_79 = df_train_on_79.drop(columns=datetime_cols)


df_train_on_79['SUPPLIER_DIV'] = df_train_on_79['SUPPLIER_DIV'].astype(int)
df_train_on_79['SUPPLIER_DIV'] = df_train_on_79['SUPPLIER_DIV'].astype('category')
df_train_on_79['SUPPLIER_DIV'] = df_train_on_79['SUPPLIER_DIV'].astype(int)
df_train_on_79['SUPPLIER_DIV'] = df_train_on_79['SUPPLIER_DIV'].astype('category')


df_train_on_79['OTHER AREA SHIP DIV'] = df_train_on_79['OTHER AREA SHIP DIV'].astype(int)
df_train_on_79['OTHER AREA SHIP DIV'] = df_train_on_79['OTHER AREA SHIP DIV'].astype('category')
df_train_on_79['OTHER AREA SHIP DIV'] = df_train_on_79['OTHER AREA SHIP DIV'].astype(int)
df_train_on_79['OTHER AREA SHIP DIV'] = df_train_on_79['OTHER AREA SHIP DIV'].astype('category')


cat_col, num_col = split_columns_by_type(df_train_on_79)
cat_col, num_col


import pandas as pd

def undersample_to_ratio(df, label_col='label', minority_class=1, ratio=1/20, random_state=24):
    """
    Undersample lá»›p Ä‘a sá»‘ Ä‘á»ƒ Ä‘áº¡t tá»· lá»‡ mong muá»‘n giá»¯a lá»›p thiá»ƒu sá»‘ vÃ  Ä‘a sá»‘.
    
    Parameters:
        df (pd.DataFrame): Dá»¯ liá»‡u Ä‘áº§u vÃ o.
        label_col (str): TÃªn cá»™t label.
        minority_class (int or str): GiÃ¡ trá»‹ Ä‘áº¡i diá»‡n lá»›p thiá»ƒu sá»‘.
        ratio (float): Tá»· lá»‡ lá»›p thiá»ƒu sá»‘ / lá»›p Ä‘a sá»‘ (vd: 1/20).
        random_state (int): Seed cho viá»‡c láº¥y máº«u ngáº«u nhiÃªn.
        
    Returns:
        pd.DataFrame: Dataset sau khi undersample.
    """
    # TÃ¡ch lá»›p thiá»ƒu sá»‘ vÃ  Ä‘a sá»‘
    minority_df = df[df[label_col] == minority_class]
    majority_df = df[df[label_col] != minority_class]
    
    # Sá»‘ máº«u cáº§n tá»« lá»›p Ä‘a sá»‘
    n_minority = len(minority_df)
    n_majority = int(n_minority / ratio)
    
    # Kiá»ƒm tra náº¿u thiáº¿u dá»¯ liá»‡u
    if n_majority > len(majority_df):
        raise ValueError("KhÃ´ng Ä‘á»§ máº«u trong lá»›p Ä‘a sá»‘ Ä‘á»ƒ thá»±c hiá»‡n undersampling theo tá»· lá»‡ yÃªu cáº§u.")
    
    majority_sampled = majority_df.sample(n=n_majority, random_state=24)
    
    # Gá»™p láº¡i vÃ  shuffle
    df_balanced = pd.concat([minority_df, majority_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=24).reset_index(drop=True)
    
    return df_balanced



X_79 = df_train_on_79.drop(columns=['label'])
y_79 = df_train_on_79['label']


from sklearn.model_selection import train_test_split


x_train_79, x_test_79, y_train_79, y_test_79 = train_test_split(X_79, y_79, test_size=0.2, random_state=1009)


df_train = pd.concat([x_train_79, y_train_79], axis=1)
df_train = undersample_to_ratio(df_train, label_col='label', minority_class=1, ratio=1/10, random_state=24)
x_train_79 = df_train.drop(columns=['label'])
y_train_79 = df_train['label']


from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix


def plot_catboost_importance(model, feature_names, top_n=20):
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).head(top_n)
    # print the importance_df' feature names and importance values
    print(importance_df)
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='lightcoral')
    plt.title('Top Feature Importances (CatBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


model_catboost_1=CatBoostClassifier(
iterations=1000,          # Sá»‘ vÃ²ng láº·p (cÃ³ thá»ƒ Ä‘iá»�u chá»‰nh)
learning_rate=0.1,        # Tá»· lá»‡ há»�c
depth=5,                  # Ä�á»™ sÃ¢u cÃ¢y
loss_function='Logloss',  # HÃ m máº¥t mÃ¡t cho bÃ i toÃ¡n phÃ¢n loáº¡i nhá»‹ phÃ¢n
verbose=100,              # In thÃ´ng tin sau má»—i 100 vÃ²ng
random_seed=42,
cat_features=cat_col
)   


model_catboost_1.fit(x_train_79, y_train_79)
y_pred_79 = model_catboost_1.predict(x_test_79)
print(classification_report(y_test_79, y_pred_79))


plot_catboost_importance(model_catboost_1, X_79.columns, top_n=20)


final_test_path = r'/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv'


test_data = pd.read_csv(final_test_path)


x_test_cols = useful_cols.copy()
if 'label' in x_test_cols:
    x_test_cols.remove('label')


x_test_cols


x_test_final = test_data[x_test_cols]


x_test_final['OTHER AREA SHIP DIV'] = x_test_final['OTHER AREA SHIP DIV'].fillna('0')


x_test_final['SUPPLIER_DIV'] = x_test_final['SUPPLIER_DIV'].fillna(1.0)


x_test_final = converter.convert(x_test_final)


x_test_final['range_day'] = (x_test_final['VSD'] - x_test_final['Order date']).dt.days.astype(int)
x_test_final['order month'] = x_test_final['Order date'].dt.month.astype(int)
x_test_final = x_test_final.drop(columns=datetime_cols)


x_test_final['OTHER AREA SHIP DIV'] = x_test_final['OTHER AREA SHIP DIV'].astype(int)
x_test_final['OTHER AREA SHIP DIV'] = x_test_final['OTHER AREA SHIP DIV'].astype('category')
x_test_final['SUPPLIER_DIV'] = x_test_final['SUPPLIER_DIV'].astype(int)
x_test_final['SUPPLIER_DIV'] = x_test_final['SUPPLIER_DIV'].astype('category')


y_final_test_pred_cat_79 = model_catboost_1.predict(x_test_final)


submission = pd.DataFrame({
    "ID": x_test_final.index+1,       # hoáº·c index náº¿u khÃ´ng cÃ³ cá»™t id
    "label": y_final_test_pred_cat_79       # nhÃ£n dá»± Ä‘oÃ¡n
})

submission.to_csv("submission_catboost_train_79.csv", index=False)

