import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from datetime import datetime


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from copy import deepcopy


# Mute warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
print('*'*100)
print('DATA TYPES OF df_train:')
print(df_train.dtypes)
print('-'*100)
print('DATA TYPES OF df_test:')
print(df_test.dtypes)
print('*'*100)
print('SHAPE OF df_train:')
print(df_train.shape)
print('-'*100)
print('SHAPE OF df_test:')
print(df_test.shape)
print('*'*100)
print('HEAD OF df_train:')
df_train.head()


#----------------------------------------------------------------------------------------------------------------------------
def grab_col_names(df, target=None, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtype in ["O", "category", "bool"]]
    num_but_cat = [col for col in df.columns 
                   if df[col].nunique() < cat_th and df[col].dtype in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns 
                   if df[col].nunique() > car_th and df[col].dtype in ["O", "category"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    if target:
        for col_list in [cat_cols, num_cols, cat_but_car, num_but_cat]:
            if target in col_list:
                col_list.remove(target)
    cat_cols = [col for col in cat_cols if col not in num_but_cat]
    print("-" * 20)
    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print("-" * 20)
    print('Cat_cols:\n',cat_cols)
    print('num_cols:\n',num_cols)
    print('cat_but_car:\n',cat_but_car)
    print('num_but_cat:\n',num_but_cat)
    print("-" * 20)    
    return cat_cols, num_cols, cat_but_car, num_but_cat
#----------------------------------------------------------------------------------------------------------------------------
def sub_plot(col):
    n_cols = 5
    n_rows = len(col) // n_cols
    if len(col) % n_cols != 0:
        n_rows+=1
    fig, axes=plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes=axes.flatten()
    return fig, axes
#----------------------------------------------------------------------------------------------------------------------------
def corr_matrix(df, cmap="coolwarm", factorize_categorical=True):
    df = df.copy()
    
    if factorize_categorical:
        df = df.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'object' or str(col.dtypes) == 'category' else col)
    
    corr_matrix = df.corr(numeric_only=True)
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Ajuste automÃ¡tico del tamaÃ±o segÃºn nÃºmero de columnas
    n_cols = len(corr_matrix.columns)
    figsize = (n_cols * 0.8, n_cols * 0.6)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 10}
    )
    plt.title("Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
#----------------------------------------------------------------------------------------------------------------------------
from sklearn.feature_selection import mutual_info_regression
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()
#----------------------------------------------------------------------------------------------------------------------------
# check the outliers of the dataset
def outliers(df, df_cols):
    df_copy = df.copy()
    # Box plots for outlier visualization
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df_cols):
        plt.subplot(len(df_cols) // 5 + 1, 5, i + 1)
        sns.boxplot(y=df_copy[col])
        plt.title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate IQR and identify outliers for each numerical column
    outlier_info = {}  # Store outlier information for each column
    
    for col in df_cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    
        outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)]
    
        outlier_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'num_outliers': len(outliers),
        }
    
        print(f"Column: {col}")
        print(f"Number of outliers: {len(outliers)}")
        print("-" * 20)
#----------------------------------------------------------------------------------------------------------------------------


cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df=df_train, target='loan_paid_back')


for col in [col for col in df_train.columns if col not in num_cols]:
    print(f"ðŸ”¸ {col}:")
    print(df_train[col].unique())
    print("-" * 40)


print('*'*100)
print('SHAPE OF df_train:')
print(df_train.shape)
print('NULL DATA:')
print(df_train.isnull().sum())
print('*'*100)
print('DUPLICATED DATA:')
print(df_train.duplicated().sum())
print('*'*100)

# df_train = df_train.dropna()
# print('*'*100)
# print('SHAPE OF df_train:')
# print(df_train.shape)
# print('NULL DATA:')
# print(df_train.isnull().sum())


fig, axes = sub_plot(num_cols)
for i, col in enumerate(num_cols):
    sns.histplot(df_train[col], bins=25, kde=True, ax=axes[i], color=sns.color_palette("Set2")[2])
    axes[i].set_title(f"Histogram - {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()


fig, axes = sub_plot(cat_cols)
for i, col in enumerate(cat_cols):
    sns.countplot(x=col, data=df_train, ax=axes[i], palette="Set2")
    axes[i].set_title(f"Count Plot - {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()


rare_cats = {}
for col in cat_cols:
    freqs = df_train[col].value_counts(normalize=True)
    rare = freqs[freqs < 0.025]  # <2,5%
    if not rare.empty:
        rare_cats[col] = rare

# Show infrequent categories with their percentages
for col, rares in rare_cats.items():
    print(f"\nðŸ“Š {col}:")
    for cat, freq in rares.items():
        print(f"   - {cat} ({freq:.2%})")


df_plot = df_train.copy()
print('*'*50)
print('Data Types of df_train:')
print(df_plot.dtypes)
print('*'*50)
df_plot.columns


grade_order = [f"{g}{n}" for g in "ABCDEF" for n in range(1, 6)]   
education_order = ["Other", "High School", "Bachelor's", "Master's", "PhD"]    
def add_on(df):
    # Sorting the grade_subgrade for better view in the plots
 
    df["grade_subgrade"] = pd.Categorical(df["grade_subgrade"],
                                               categories=grade_order,
                                               ordered=True)
    # -----------------------------------------------------------------------
    # Binning credit Score in new column 'credit_score_bin':
    df["credit_score_bin"] = pd.cut(
        df["credit_score"],
        bins=[300, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
    )
    # -----------------------------------------------------------------------
    # Ordinal education level:
    df["education_level"] = pd.Categorical(
        df["education_level"],
        categories=education_order,
        ordered=True
    )
    return df

df_plot=add_on(df_plot)


plt.figure(figsize=(14,6))
sns.boxplot(
    data=df_plot, 
    x='loan_purpose', 
    y='interest_rate', 
    palette="Set2"
)
plt.title("Distribution of loan_purpose vs interest_rate")
plt.xlabel("Loan Purpose")
plt.ylabel("Mean Interest Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.barplot(
    data=df_plot,
    x='loan_purpose',
    y='interest_rate',
    palette="Set2",
    ci=None
)

plt.title("Mean of Interest Rate by Loan Purpose", fontsize=14)
plt.xlabel("Loan Purpose")
plt.ylabel("Interest Rate (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14,6))
sns.boxplot(
    data=df_plot,
    x='grade_subgrade',
    y='annual_income',
    palette="Set2"
)

plt.title("Distribution of grade_subgrade vs annual_income", fontsize=14)
plt.xlabel("Grade / Subgrade")
plt.ylabel("Annual Income")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.barplot(
    data=df_plot,
    x='grade_subgrade',
    y='annual_income',
    palette="Set2",
    ci=None
)

plt.title("Mean of Anual income by Grade Subgrade", fontsize=14)
plt.xlabel("Grade / Subgrade")
plt.ylabel("Annual Income")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14,6))
sns.boxplot(
    data=df_plot,
    x='grade_subgrade',
    y='interest_rate',
    order=grade_order,
    palette="Set2"
)

plt.title("Distribution of grade_subgrade vs interest_rate", fontsize=14)
plt.xlabel("Grade / Subgrade")
plt.ylabel("Interest Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.barplot(
    data=df_plot,
    x='grade_subgrade',
    y='interest_rate',
    order=grade_order,
    palette="Set2",
    ci=None
)

plt.title("Average Interest rate by Grade/Subgrade", fontsize=14)
plt.xlabel("Grade / Subgrade")
plt.ylabel("Interest Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



plt.figure(figsize=(14,6))
sns.boxplot(
    data=df_plot,
    x='credit_score_bin',
    y='debt_to_income_ratio',
    palette="Set2"
)

plt.title("Distribution of Debt-to-Income Ratio vs Credit Score", fontsize=14)
plt.xlabel("Credit Score")
plt.ylabel("Debt-to-Income Ratio")
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.barplot(
    data=df_plot,
    x='credit_score_bin',
    y='debt_to_income_ratio',
    palette="Set2",
    ci=None
)

plt.title("Mean of Debt-to-Income Ratio vs Credit Score", fontsize=14)
plt.xlabel("Credit Score")
plt.ylabel("Debt-to-Income Ratio (Mean)")
plt.tight_layout()
plt.show()



plt.figure(figsize=(14,6))
sns.boxplot(
    data=df_plot,
    x="loan_paid_back",
    y="interest_rate",
    palette="Set2"
)

plt.title("Interest Rate vs Loan Paid Back", fontsize=14)
plt.xlabel("Loan Paid Back (1 = Yes, 0 = No)")
plt.ylabel("Interest Rate (%)")
plt.xticks([0, 1], ["No", "Yes"])
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.violinplot(
    data=df_plot,
    x="loan_paid_back",
    y="interest_rate",
    palette="Set2"
)
plt.title("Interest Rate Distribution by Loan Repayment Status")
plt.xlabel("Loan Paid Back (1 = Yes, 0 = No)")
plt.ylabel("Interest Rate (%)")
plt.show()



plt.figure(figsize=(14,6))
sns.countplot(
    data=df_plot,
    x="grade_subgrade",
    hue="loan_paid_back",
    palette="Set2"
)

plt.title("Count of Loans Paid Back by Grade/Subgrade", fontsize=14)
plt.xlabel("Grade / Subgrade")
plt.ylabel("Count of Loans")
plt.legend(title="Loan Paid Back", labels=["No", "Yes"])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.barplot(
    data=df_plot,
    x="grade_subgrade",
    y="loan_paid_back",
    palette="Set2",
    ci=None
)

plt.title("Proportion of Loans Paid Back by Grade/Subgrade", fontsize=14)
plt.xlabel("Grade / Subgrade")
plt.ylabel("Proportion of Paid Loans")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



print('*'*100)
print('Loan Paid Back Distribution: ')
print((df_plot["loan_paid_back"]
 .value_counts(normalize=True)
 .mul(100)
 .round(1)))
print('*'*100)
# -----------------------------------------------------------------------
plt.figure(figsize=(14,6))
sns.countplot(data=df_plot, x='loan_paid_back', palette='Set2')
plt.title("Distribution of the Target Variable: loan_paid_back", fontsize=14)
plt.xlabel("Loan Paid Back (1 = Yes, 0 = No)")
plt.ylabel("Number of Loans")
plt.xticks([0, 1], ["Did Not Pay", "Paid"])
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------
plt.figure(figsize=(10,10))
df_plot["loan_paid_back"].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['#fc8d62', '#66c2a5'],
    labels=["Paid", "Did Not Pay"],
    startangle=90
)
plt.title("Percentage Distribution of Loan Paid Back", fontsize=14)
plt.ylabel("")  # Hide Y-axis label
plt.show()


outliers(df_plot, num_cols)


corr_matrix(df_plot)


X_mi = df_plot.drop(['loan_paid_back'],axis=1)
y_mi = df_plot['loan_paid_back']

scores = make_mi_scores(X_mi, y_mi)
plot_mi_scores(scores)


df_train=add_on(df_train)
for col in [col for col in df_train.columns if col not in ['loan_paid_back']+num_cols]:
    print(f"ðŸ”¸ {col}:")
    print(df_train[col].unique())
    print("-" * 40)


le_coder_cols = ['employment_status', 'loan_purpose']
one_hot_cols  = ['gender', 'marital_status']
ordinal_cols  = ['education_level', 'grade_subgrade', 'credit_score_bin']
TARGET='loan_paid_back'

def df_coder(df, ordinal_cols, le_coder_cols, one_hot_cols, num_cols):
    # Ordinal encoder
    for col in ordinal_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    df[ordinal_cols] = df[ordinal_cols].apply(lambda x: x.cat.codes)
    # Label encoder
    for col in le_coder_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    # One-Hot encoder
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    # Coding boolean columns
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    # StandarScaler
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("âœ… Encodeded Dataframe")
    # print(df.head())
    return df

df_train_encode = df_coder(df_train, ordinal_cols=ordinal_cols, le_coder_cols=le_coder_cols, one_hot_cols=one_hot_cols, num_cols=num_cols)

df_test=add_on(df_test)
df_test_encode  = df_coder(df_test, ordinal_cols=ordinal_cols, le_coder_cols=le_coder_cols, one_hot_cols=one_hot_cols, num_cols=num_cols)


X = df_train_encode.drop([TARGET],axis=1)
y = df_train_encode[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      
    random_state=42,   
    stratify=y          
)


xgb_model = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.01,
    reg_lambda=1.0,
    reg_alpha=0.3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


models = []
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(df_test_encode))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"ðŸ”¹ Fold {fold+1}")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Nuevo modelo por fold
    model = deepcopy(xgb_model)
    model.fit(X_tr, y_tr)
    
    # Guardar predicciones out-of-fold (validaciÃ³n)
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    
    # Predicciones en test
    test_preds += model.predict_proba(df_test_encode)[:, 1] / kf.n_splits
    
    models.append(model)
    print(f"   ROC AUC (fold {fold+1}): {roc_auc_score(y_val, oof_preds[val_idx]):.4f}\n")

# MÃ©trica final
print(f"ðŸ“Š ROC AUC promedio: {roc_auc_score(y_train, oof_preds):.4f}")

# Predicciones promedio finales sobre test
print("âœ… Predicciones promedio (ensemble) generadas para df_test_encode")
print(test_preds[:10])


# AsegÃºrate de que df_test_encode tenga los mismos Ã­ndices o una columna 'id'
submission = pd.DataFrame({
    "id": df_test_encode.index,         # o df_test["id"] si tienes la columna explÃ­cita
    "loan_paid_back": test_preds        # tus probabilidades predichas
})

# Guarda en CSV
submission.to_csv("submission.csv", index=False)

print("âœ… Archivo 'submission.csv' creado con Ã©xito!")
print(submission.head())

