# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import gaussian_kde
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from lightgbm import LGBMClassifier
from sklearn.feature_selection import f_classif

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Importando os dados
test_data = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
train_data = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')


print(train_data.info())


for coluna in train_data.select_dtypes(include=['object']).columns:
    print(coluna)
    print(train_data[coluna].unique())


obesity_order = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

ax = sns.countplot(
    data=train_data,
    x='NObeyesdad',
    palette='viridis',
    order=obesity_order
)

total = len(train_data)
for p in ax.patches:
    height = p.get_height()
    percentage = f'{100 * height / total:.1f}%'
    ax.annotate(
        percentage,
        (p.get_x() + p.get_width() / 2., height),
        ha='center',
        va='bottom',
        xytext=(0, 5),
        textcoords='offset points',
        fontsize=10
    )

plt.xticks([])
plt.xlabel('')
plt.ylabel('Contagem')
plt.title('Distribuição de NObeyesdad')

handles = [plt.Rectangle((0, 0), 1, 1, color=color) 
           for color in sns.color_palette('viridis', n_colors=len(obesity_order))]
plt.legend(
    handles,
    obesity_order,
    title='Categorias',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()


warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

obesity_order = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

palette = sns.color_palette('viridis', len(obesity_order))
color_map = dict(zip(obesity_order, palette))

cat_cols = train_data.select_dtypes(include=['object']).columns.drop('NObeyesdad')

sns.set(style="whitegrid")

for col in cat_cols:
    category_order = train_data[col].value_counts().index
    
    g = sns.FacetGrid(
        data=train_data,
        col=col,
        col_wrap=3,
        height=4,
        aspect=1.2,
        sharey=False,
        col_order=category_order
    )
    
    g.map_dataframe(
        sns.countplot,
        x='NObeyesdad',
        order=obesity_order,
        palette=color_map
    )
    
    g.set_titles("{col_name}")
    g.set_axis_labels("", "Contagem")
    g.set_xticklabels([])
    
    handles = [plt.Rectangle((0,0),1,1, color=color_map[cls]) for cls in obesity_order]
    g.fig.legend(
        handles=handles,
        labels=obesity_order,
        title='Categorias de Obesidade',
        bbox_to_anchor=(1.05, 0.5),
        loc='center left'
    )
    
    plt.suptitle(f'Distribuição de NObeyesdad por {col}', y=1.02)
    plt.show()

    category_percent = (train_data[col].value_counts(normalize=True) * 100).loc[category_order]
    
    print(f"\n=== Porcentagens por categoria ({col}) ===")
    
    cross_tab = pd.crosstab(
        train_data[col], 
        train_data['NObeyesdad'], 
        normalize='index'
    ).loc[category_order, obesity_order] * 100
    
    for category in cross_tab.index:
        print(f"\n** {category} **  —  {category_percent[category]:.1f}% do total")
        for obesity_class in cross_tab.columns:
            percent = cross_tab.loc[category, obesity_class]
            print(f"{obesity_class}: {percent:.1f}%")
    
    print("\n" + "="*50 + "\n")


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) 
    phi2corr = max(0, phi2 - ((k-1)*(r-1)/(n-1)))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

cat_vars = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

cramers_matrix = pd.DataFrame(index=cat_vars, columns=cat_vars)

for var1 in cat_vars:
    for var2 in cat_vars:
        cramers_matrix.loc[var1, var2] = cramers_v(train_data[var1], train_data[var2])

cramers_matrix = cramers_matrix.astype(float)

plt.figure(figsize=(12, 10))
sns.heatmap(cramers_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='Blues', 
            vmin=0, 
            vmax=1,
            linewidths=0.5)

plt.title("Matriz de Associação (V de Cramér - Variáveis Categóricas)", pad=20, fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


warnings.filterwarnings("ignore", 
                      category=FutureWarning,
                      module="seaborn._oldcore")

def kde_bins(data, bw_multiplier=1):
    """
    Calcula o número de bins usando a largura de banda do KDE.
    
    Parâmetros:
    - data: Array com os dados
    - bw_multiplier: Fator de ajuste (padrão=1)
    
    Retorna:
    - Número recomendado de bins
    """
    kde = gaussian_kde(data.dropna())
    bw = kde.scotts_factor() * np.std(data, ddof=1)  # Largura de banda de Scott
    data_range = np.max(data) - np.min(data)
    bins = int(np.ceil(data_range / (bw * bw_multiplier)))
    return max(bins, 5)  # Mínimo de 5 bins

num_cols = train_data.select_dtypes(include=['number']).drop(columns='id')
for col in num_cols:
    bins = kde_bins(train_data[col], bw_multiplier=1.2)  # Ajuste fino com bw_multiplier
    
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=train_data,
        x=col,
        bins=bins,
        kde=True,  # Opcional: sobrepor a curva KDE
        color='skyblue',
        edgecolor='white'
    )
    plt.title(f'Histograma de {col} (Bins: {bins}, calculados via KDE)')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.show()


numeric_vars = [col for col in train_data.select_dtypes(include=['int64', 'float64']).columns 
                if col.lower() != 'id' and col != 'NObeyesdad']
target = 'NObeyesdad'

for var in numeric_vars:
    plt.figure(figsize=(12, 6))
    
    # Boxplot
    sns.boxplot(x=target, y=var, data=train_data, palette='viridis')
    
    plt.title(f'Distribuição de {var} por classe de {target}', fontsize=14)
    plt.xlabel(target, fontsize=12)
    plt.ylabel(var, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


X_num = train_data.select_dtypes(include=['number']).drop(columns='id')
y = train_data['NObeyesdad']

f_values, p_values = f_classif(X_num, y)

anova_results = pd.DataFrame({
    'Variável': X_num.columns,
    'F_value': f_values,
    'p_value': p_values
}).sort_values('F_value', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=anova_results, y='Variável', x='F_value', palette='rocket')
plt.title("Valores F (ANOVA) - Relação com o Target", fontsize=14)
plt.show()


X = train_data.drop(columns=['NObeyesdad', 'id'])
y = train_data['NObeyesdad']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

model = LGBMClassifier(
    objective='multiclass',
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

param_dist = {
    'num_leaves': np.arange(15, 150, 10),
    'max_depth': [-1, 3, 5, 7, 9, 12],
    'learning_rate': np.linspace(0.01, 0.3, 20),
    'n_estimators': np.arange(100, 1000, 100),
    'min_child_samples': np.arange(5, 50, 5),
    'subsample': np.linspace(0.6, 1.0, 5),
    'colsample_bytree': np.linspace(0.6, 1.0, 5)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search_f1 = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  
    scoring='f1_weighted',
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search_acc = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Buscando melhores hiperparâmetros por F1-weighted...")
random_search_f1.fit(X_train, y_train, categorical_feature=categorical_cols)

print("\nBuscando melhores hiperparâmetros por Accuracy...")
random_search_acc.fit(X_train, y_train, categorical_feature=categorical_cols)

best_model_f1 = random_search_f1.best_estimator_
best_model_acc = random_search_acc.best_estimator_


y_pred_f1 = best_model_f1.predict(X_test)
print("\nAvaliação - Melhor modelo por F1:")
print("Accuracy:", accuracy_score(y_test, y_pred_f1))
print("F1 weighted:", f1_score(y_test, y_pred_f1, average='weighted'))
print(classification_report(y_test, y_pred_f1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_f1))

y_pred_acc = best_model_acc.predict(X_test)
print("\nAvaliação - Melhor modelo por Accuracy:")
print("Accuracy:", accuracy_score(y_test, y_pred_acc))
print("F1 weighted:", f1_score(y_test, y_pred_acc, average='weighted'))
print(classification_report(y_test, y_pred_acc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_acc))

print("\nMelhores parâmetros por F1:", random_search_f1.best_params_)
print("Melhores parâmetros por Accuracy:", random_search_acc.best_params_)



X_train = train_data.drop(columns=['id','NObeyesdad'])
y_train = train_data['NObeyesdad']

test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

model = LGBMClassifier(
    objective='multiclass',
    class_weight='balanced',
    subsample=1.0,
    num_leaves=55,
    n_estimators=400,
    min_child_samples=45,
    max_depth=5,
    learning_rate=0.04052631578947368,
    colsample_bytree=0.6,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train, categorical_feature=categorical_cols)

y_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': test_ids,
    'NObeyesdad': y_pred
})

submission.to_csv('/kaggle/working/submission.csv', index=False)

