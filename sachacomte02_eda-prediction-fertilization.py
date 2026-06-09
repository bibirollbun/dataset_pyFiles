# ===================================================================
#âš™ï¸� Standard Library
# ===================================================================
from IPython.core.display import display, HTML # print html notebook
import markdown                                # print html notebook
import warnings                                # manag warning
import numpy as np                             # manag table
import pandas as pd                            # manag table with key    

# ===================================================================
#ğŸ“ˆ Graph Library
# ===================================================================
import matplotlib.pyplot as plt      # graphique classique
import seaborn as sns                # graphique with pandas

# ===================================================================
# ğŸ¤– Machine Learning 
# ===================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose  import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection   import train_test_split
from sklearn.metrics import (
    mutual_info_score,
    confusion_matrix,
    average_precision_score,
)
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder,
    MultiLabelBinarizer, 
    LabelEncoder
)

# ===================================================================
# Model 
# ===================================================================
import xgboost as xgb
from sklearn.tree         import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble     import (
    AdaBoostClassifier,
    BaggingClassifier,
)


# ===================================================================
# ğŸ“Š Evaluation Metrics
# ===================================================================
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    silhouette_score
)

# ===================================================================
# RÃ©seau de Neuronne
# ===================================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# ----  Formatting charts
%matplotlib inline
from IPython.core.pylabtools import figsize
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['axes.edgecolor']  = "#bcbcbc"
mpl.rcParams['patch.linewidth'] = 0.5
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['axes.facecolor']  = "#eeeeee"
mpl.rcParams['axes.labelsize']  = "large"
mpl.rcParams['axes.grid']       = True
mpl.rcParams['grid.linestyle']  = "--"
mpl.rcParams['patch.edgecolor'] = "#eeeeee"
mpl.rcParams['axes.titlesize']  = "x-large"


# ---- Suppression des warning
warnings.filterwarnings("ignore")


def is_notebook()-> bool:
    """
    Permet savoir si nous somme dans un notebook.

    Parameters
    ----------
    None

    Returns
    -------
    Bool :
        - True  : Alors le code est lancÃ© dans un notebook
        - False : Alors le code n'est pas lancÃ© dans un notebook
    """
    # ---- Test si on est dans un notebook
    try:
        # VÃ©rifie si nous sommes dans un environnement Jupyter
        get_ipython
        return True
    except NameError:
        # Nous ne sommes pas dans un notebook
        return False


def print_head(text:str) -> None:
    """
    Print a text between two lines "#". Ou si dans un notebook affiche en markdown.

    Parameters
    ----------
    text : str
        Text that printing.
        
    Returns
    -------
    None
        Print in the console.

    Examples
    --------
    >>> print_head("Titre")
    ##################################################
    Titre
    ##################################################
    """
    if is_notebook():
        # ---- Si nous sommes dans un notebook
        display(HTML(markdown.markdown(f"**{text}**")))
    else :
        # ---- Si nous ne sommes pas dans un notebook
        print(50*"#")
        print(text)
        print(50*"#")


def print_columns(index: np.array, *columns: np.array) -> None:
    """
    Displays the columns of data associated with a given index.

    This function takes an array of indices and a variable number of columns,
    then prints the values of each column for each index.

    Parameters:
    -----------
    index : np.array
        A NumPy array containing the indices to display.
    
    *columns : np.array
        One or more NumPy arrays containing the data to display.
        Each array must have the same length as the index.

    Exceptions:
    -----------
    ValueError : 
        If the length of any of the columns does not match the length of the index,
        a ValueError is raised with the message:
        "All columns must have the same length as the index."

    Example usage:
    -----------------------
    >>> index = np.array([0, 1, 2])
    >>> column1 = np.array([10, 20, 30])
    >>> column2 = np.array([0.1, 0.2, 0.3])
    >>> print_columns(index, column1, column2)
    0                    10        0.1      
    1                    20        0.2      
    2                    30        0.3      
    """
    # ---- Verification
    for col in columns:
        if len(col) != len(index):
            raise ValueError("All columns must have the same length as the index.")

    # ---- Affichage
    # Remarque le end="" permet de ne pas revenir Ã  la ligne
    for k in index:
        print(f"{k:<20}", end="")   
        for col in columns:
            print(f"{col[k]:<10}", end="")
        print() # New line


def print_count(data:pd.DataFrame, column:str) -> None :
    """
    Print the number and percentage of values.

    Parameters
    ----------
    data : pandas.DataFrame
        Data.
    column : str
        Column's name.

    Returns
    -------
    None
        Print in the console.

    Examples
    --------
    >>> print_count(train, "Brand")
    Adidas                   60077     20.03 %
    Under Armour             59992     20.00 %
    Nike                     57336     19.11 %
    Puma                     56814     18.94 %
    Jansport                 56076     18.69 %
    None                      9705      3.23 %
    """
    valueCount         = data[column].value_counts()
    valueCountPourcent = data[column].value_counts() / data.shape[0] * 100

    # Start
    for k, i, j in zip(valueCount.index, valueCount, valueCountPourcent):
            print(f"{k:<20}{i:10}{j:10.2f} %")


def print_value_nan(data:pd.DataFrame) -> None :
    """
    Print the number and percentage of the nan value.

    Parameters
    ----------
    data : pandas.DataFrame
        Data. 

    Returns
    -------
    None
        Print in the console.

    Examples
    --------
    >>> data.head()
    id 	Brand 	Material 	Size 	Compartments 	Laptop Compartment 	Waterproof 	Style 	Color 	Weight Capacity (kg) 	Price
    0 	0 	Jansport 	Leather 	Medium 	7.0 	Yes 	No 	Tote 	Black 	11.611723 	112.15875
    1 	1 	Jansport 	Canvas 	Small 	10.0 	Yes 	Yes 	Messenger 	Green 	27.078537 	68.88056
    2 	2 	Under Armour 	Leather 	Small 	2.0 	Yes 	No 	Messenger 	Red 	16.643760 	39.17320
    ...
    >>> print_value_nan(data)
    id                           0      0.00 %
    Brand                     9705      3.23 %
    Material                  8347      2.78 %
    Size                      6595      2.20 %
    Compartments                 0      0.00 %
    Laptop Compartment        7444      2.48 %
    Waterproof                7050      2.35 %
    Style                     7970      2.66 %
    Color                     9950      3.32 %
    Weight Capacity (kg)       138      0.05 %
    Price                        0      0.00 %
    """
    # ---- Initialisation
    valueNan = data.isnull().sum()
    valueNanPourcent = valueNan/data.shape[0] * 100

    # ---- Affichage
    print_columns(data.columns, valueNan, valueNanPourcent)


def create_wind_rose(ax, data, dataset_name, color):
    wind_direction_radians = np.radians(data['winddirection'].dropna())

    bins = np.linspace(0, 2*np.pi, 37)  
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), edgecolor='black',color=color, alpha=0.8)

    ax.set_theta_zero_location("N") 
    ax.set_theta_direction(-1) 
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))  
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([]) 
    ax.set_title(f"Wind Direction ({dataset_name})", fontsize=12, fontweight='bold', pad=10)


def plot_categorical_feature_distribution(
    ser: pd.Series, 
    palette: str="Set2",
    explode_value=None,
) -> None:
    nunique = ser.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(15 + nunique*0.01, 5 + nunique*0.1))
    axes = axes.flatten()
    value_counts = ser.value_counts(ascending=True)
    labels = value_counts.index.tolist()
    colors = sns.color_palette(palette, len(labels)).as_hex()  # we borrow colors from a seaborn color palette
    # Donut Chart
    explodes=None if explode_value is None else [0.1 if i == explode_value else 0 for i in value_counts.index]
    axes[0].pie(
        value_counts, 
        autopct='%1.1f%%', 
        textprops={'size': 8, 'color': 'black'}, 
        colors=colors,
        wedgeprops=dict(width=0.4),  # donut wedge width
        startangle=80, 
        pctdistance=0.85,  # have percentage displayed within wedge
        explode=explodes,
        labels=labels,
    )
    # Count Plot 
    for i, v in enumerate(value_counts):
        axes[1].barh(y=i, width=v, color='none', edgecolor=colors[i], hatch='////')
        axes[1].text(x=v + 1, y=i, s=str(v), color='black', fontsize=10, va='center')
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels)
    sns.despine(left=True, bottom=True)  # remove default spines (borders) from plot
    axes[1].set_xticks([])
    fig.suptitle(f'{ser.name} Distribution', fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


# ---- Data for EDA
data : pd.DataFrame = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")   # importation data
data.head()    


# ---- Data for train
train : pd.DataFrame = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")  # importation data
train.head()   


# ---- Data for test
test  : pd.DataFrame = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")   # importation data
test.head()     


# ---- Print shape data
print_head("Shape Data")
print(f"Number of colones : {data.shape[1]:_}")
print(f"Nomber of lines   : {data.shape[0]:_}")


# ---- Print name columes
print_head("Name columns")
for i in data.columns:
    print(f"     - {i}")


# ---- Print type columns
print_head("Type")
print(data.info())


tab20c = plt.color_sequences["tab20c"]
inner_colors = [tab20c[i] for i in [1, 5, 10]]
plt.figure()

plt.title("Data type in the dataset")
plt.pie(data.dtypes.value_counts(), 
        autopct='%1.1f%%', 
        startangle=140,
        colors=inner_colors,
        labels=data.dtypes.value_counts().index)
plt.show()


col_numerique = [
    "Temparature",
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous",
]
col_object = [
    "Soil Type",
    "Crop Type",
]
col_target = ["Fertilizer Name"]


for col in col_object:
    data[col] = data[col].astype('category') 


print_head("Number of missing data train")
print_value_nan(data)


plt.figure()
plt.title("Value Miss Data")
sns.heatmap(data[::500].isna(), cbar=False)
plt.grid(visible=False)


print_count(data, col_target[0])


plot_categorical_feature_distribution(ser=data[col_target[0]])


for variable in col_object :
    print_head(variable)
    print(f"    - Number of variables : {len(data[variable].unique())}")
    print_count(data, variable)


# ---- Plot bar charts for each categorical feature
for feature in col_object:
    plt.figure(figsize=(10, 6))
    
    # For features with many unique values, plot top 10 categories
    top_categories = data[feature].value_counts().nlargest(10)
    sns.barplot(x=top_categories.index, 
                y=top_categories.values,
                hue=top_categories.values,
                palette="viridis",
    )
    plt.title(f"Top 10 {feature} Categories")

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

    # Print the number of unique values
    #print(f"Number of Unique {feature}: {data[feature].nunique()}")
    #print(f"Missing Values in {feature}: {data[feature].isnull().sum()}")


for feature in col_object:
    plot_categorical_feature_distribution(ser=data[feature])


contingency_table = pd.crosstab(data[col_object[0]], data[col_object[1]])


plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table,  cmap='YlGnBu') #annot=True, fmt='d',
plt.title(f'Contingence of table : {col_object[0]} vs {col_object[1]}')
plt.xlabel(f'{col_object[0]}')
plt.ylabel(f'{col_object[1]}')
plt.show()


data.describe().T


number_value = (len(col_numerique) + 1)//2 
palette = sns.color_palette("muted", n_colors=len(col_numerique))

fig, axs = plt.subplots(number_value, 2, figsize=(12, 12))

for i, j in enumerate(col_numerique):
    # Utilisez divmod pour obtenir les indices de ligne et de colonne
    row, col = divmod(i, 2)
    sns.histplot(
        data, x=j, kde=True, 
        color=palette[i], 
        ax=axs[row, col])

plt.tight_layout()  # Pour Ã©viter le chevauchement des sous-graphes
plt.show()


number_value = (len(col_numerique) + 1)//2 
palette = sns.color_palette("pastel", n_colors=len(col_numerique))

fig, axs = plt.subplots(number_value, 2, figsize=(12, 12))

for i, j in enumerate(col_numerique):
    # Utilisez divmod pour obtenir les indices de ligne et de colonne
    row, col = divmod(i, 2)
    sns.boxplot(
        data, x=j, 
        color=palette[i], 
        ax=axs[row, col])

plt.tight_layout()  # Pour Ã©viter le chevauchement des sous-graphes
plt.show()


# Grid scatterplots in pairs to visualize correlations
pd.plotting.scatter_matrix(data[col_numerique][::500], 
                           figsize=(12, 12), 
                           diagonal='kde', 
                           alpha=0.7)

# Ajouter un titre gÃ©nÃ©ral Ã  la grille
plt.suptitle("Two by two scatterplots grid", 
             fontsize=16, 
             y=1.02)

# Afficher le graphique
plt.tight_layout()
plt.show()


correlation = data[col_numerique].corr(method="pearson")


mask = np.triu(np.ones_like(correlation, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
plt.figure(figsize=(7, 7))
plt.title("Correlation with the Pearson method")
sns.heatmap(correlation, mask=mask, cmap=cmap,  
            cbar_kws={"shrink": .5}, square=True, 
            vmax=1,  vmin=-1, center=0, annot=True,
            fmt=".2f", cbar=False)
plt.grid(False)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 1, figsize=(7, 7))
plt.title("Correlation with the Spearman method")
sns.heatmap(data[col_numerique].corr(method='spearman'), annot=True, cmap="coolwarm", fmt=".2f", ax=axes, cbar=False)
plt.tight_layout()
plt.grid(False)
plt.show()


X = data.dropna()[col_numerique+col_object]
y = data.dropna()[col_target[0]]


mi_scores = mutual_info_classif(X, y, discrete_features=True)
feature_importance = pd.Series(
    mi_scores, 
    index=X.columns).sort_values(ascending=False)

sns.heatmap(
    pd.DataFrame(feature_importance), 
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    cbar=False)
plt.title("Mutual Information Score Heatmap")
plt.grid()
plt.show()


# Boucle sur chaque variable numÃ©rique
for col in col_numerique:
    plt.figure(figsize=(8, 5))
    num_bins = len(data[col].unique())
    ax = sns.histplot(
        data=data, x=col, hue=col_target[0], bins = num_bins,
        alpha=0.7, palette="Set2", multiple="fill", stat="proportion")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f"{col} distribution by {col_target[0]}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


for col in col_numerique:
    plt.figure(figsize=(14,6))
    sns.boxplot(
        data=data,
        x=col_target[0], y=col,
        hue=col_target[0]
    )
    plt.title(f"{col} by Fertilizer")
    plt.xticks(rotation=45)
    plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer,     col_numerique),
        ('cat', categorical_transformer, col_object),
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])


# EntraÃ®ner le modÃ¨le
pipeline.fit(X_train, y_train)


# Faire des prÃ©dictions
tree_decision = pipeline.predict(X_test)


conf_matrix_tree_decision = confusion_matrix(y_test, tree_decision)


# View the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree_decision, 
            annot=True, fmt='d', cmap='Blues', 
            xticklabels=col_numerique+col_target, 
            yticklabels=col_numerique+col_target)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')
plt.show()


mlb = MultiLabelBinarizer()
y_test_bin = mlb.fit_transform(y_test)
tree_decision_bin = mlb.transform(tree_decision)

average_precision_tree_decision = average_precision_score(y_test_bin, tree_decision_bin)
print(f"Result : {average_precision_tree_decision*100:.2f}%")


pipeline2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)


# EntraÃ®ner le modÃ¨le
pipeline2.fit(X_train, y_encoded)


xgb_predict = label_encoder.inverse_transform(pipeline2.predict(X_test))


conf_matrix_xgb_predict = confusion_matrix(y_test, xgb_predict)


# View the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb_predict, 
            annot=True, fmt='d', cmap='Blues', 
            xticklabels=col_numerique+col_target, 
            yticklabels=col_numerique+col_target)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')
plt.show()


mlb = MultiLabelBinarizer()
y_test_bin = mlb.fit_transform(y_test)
xgb_predict_bin = mlb.transform(xgb_predict)

average_precision_tree_decision = average_precision_score(y_test_bin, xgb_predict_bin)
print(f"Result : {average_precision_tree_decision*100:.2f}%")


# CrÃ©er un DataFrame pour les valeurs rÃ©elles et les prÃ©dictions
data = pd.DataFrame({
    'XGB Boost': label_encoder.transform(xgb_predict), 
    'Tree Decision': label_encoder.transform(tree_decision)})

# Calculer la matrice de corrÃ©lation
corr_matrix = data.corr()


fig, axes = plt.subplots(1, 1, figsize=(7, 7))
plt.title("Correlation Tree Decision & XGB")
sns.heatmap(corr_matrix*100, annot=True, cmap="coolwarm", fmt=".2f", ax=axes, cbar=False)
plt.grid(False)
plt.show()


print(f"Mutial Information Score : {mutual_info_score(y_test, xgb_predict):.3e}")


base_estimator = DecisionTreeClassifier(max_depth=5)

pipeline3 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', AdaBoostClassifier(
        estimator=base_estimator, 
        n_estimators=50, 
        random_state=42))
])


# EntraÃ®ner le modÃ¨le
pipeline3.fit(X_train, y_train)


# Faire des prÃ©dictions
ADA_decision = pipeline3.predict(X_test)


conf_matrix_ADA_decision = confusion_matrix(y_test, ADA_decision)


# View the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ADA_decision, 
            annot=True, fmt='d', cmap='Blues', 
            xticklabels=col_numerique+col_target, 
            yticklabels=col_numerique+col_target)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')
plt.show()


mlb = MultiLabelBinarizer()
y_test_bin = mlb.fit_transform(y_test)
ADA_decision_bin = mlb.transform(ADA_decision)

average_precision_ADA_decision = average_precision_score(y_test_bin, ADA_decision_bin)
print(f"Result : {average_precision_ADA_decision*100:.2f}%")


# CrÃ©er un DataFrame pour les valeurs rÃ©elles et les prÃ©dictions
data = pd.DataFrame({
    'XGB Boost': label_encoder.transform(xgb_predict), 
    'Tree Decision': label_encoder.transform(ADA_decision)})

# Calculer la matrice de corrÃ©lation
corr_matrix = data.corr()


fig, axes = plt.subplots(1, 1, figsize=(7, 7))
plt.title("Correlation Tree Decision & XGB")
sns.heatmap(corr_matrix*100, annot=True, cmap="coolwarm", fmt=".2f", ax=axes, cbar=False)
plt.grid(False)
plt.show()


print(f"Mutial Information Score : {mutual_info_score(y_test, ADA_decision):.3e}")


base_estimator = RidgeClassifier()

pipeline5 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', BaggingClassifier(
    estimator=base_estimator, n_estimators=50,
    max_samples=0.5, max_features=0.5))
])


pipeline5.fit(X_train, y_train)


# Faire des prÃ©dictions
bag_decision = pipeline5.predict(X_test)


conf_matrix_Bag_decision = confusion_matrix(y_test, bag_decision)


# View the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_Bag_decision, 
            annot=True, fmt='d', cmap='Blues', 
            xticklabels=col_numerique+col_target, 
            yticklabels=col_numerique+col_target)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')
plt.show()


y_train_encoded = label_encoder.transform(y_train)


pipeline4 = Pipeline(steps=[
    ('preprocessor', preprocessor),
])


X_train_transformed = pipeline4.fit_transform(X_train)


# Fonction pour crÃ©er le modÃ¨le DNN
model = Sequential([
    Dense(16, activation='relu', input_shape=(22,)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


model.summary()


# EntraÃ®ner le modÃ¨le
model.fit(X_train_transformed, y_train_encoded, epochs=5, batch_size=128, verbose=1)


X_test_transformed = pipeline4.fit_transform(X_test)
predict_DNN = model.predict(X_test_transformed)


predict_DNN_bon = label_encoder.inverse_transform(np.argmax(predict_DNN, axis=-1))


conf_matrix_xgb_predict = confusion_matrix(y_test, predict_DNN_bon)


# View the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_xgb_predict, 
            annot=True, fmt='d', cmap='Blues', 
            xticklabels=col_numerique+col_target, 
            yticklabels=col_numerique+col_target)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')
plt.show()


# Dans la suite

