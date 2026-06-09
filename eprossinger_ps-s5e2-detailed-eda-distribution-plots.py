%%HTML
<style type="text/css">
h1 {
  %   background-color: lightblue; %
     padding: 12px; 
     padding-right: 300px; 
     font-size: 28px; 
     max-width: 1500px; 
     margin-top: 50px;
     margin-bottom: 8px;
     border-radius: 7px; 
}
h2 {
     background-color: lightblue; %
%     background-color: #DCDCDC; 
     padding: 8px; 
     padding-right: 300px; 
     font-size: 24px; 
     max-width: 1500px; 
     margin-top: 50px;
     margin-bottom: 4px;
     border-radius: 7px;
}
p, li {
    font-size: 15px;
}
</style>



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
# Matplotlib config
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
%config InlineBackend.rc = {'figure.figsize': (5.0, 3.0)}



# functions

# Berechnet den maximalen Wert einer Spalte, wenn sie numerisch ist
def max_value(column):
    if pd.api.types.is_numeric_dtype(column):  # ÃœberprÃ¼fe, ob der Datentyp numerisch ist
        return column.dropna().max() if not column.dropna().empty else np.nan
    return ""


# Gibt die einzigartigen Werte einer Spalte zurÃ¼ck, oder eine Range (falls es eine gibt)
def get_unique_values(column):
    if pd.api.types.is_integer_dtype(column):  # ÃœberprÃ¼fe, ob der Datentyp eine Ganzzahl ist
        unique_vals = sorted(set(column.dropna()))
        min_val, max_val = column.min(), column.max()
        if unique_vals == list(range(min_val, max_val + 1)):
#          return f"range({min_val},{max_val + 1})"
            return f"(range:{min_val}-{max_val})"
        return unique_vals
#        return f"between {min_val} and {max_val}"
    return sorted(set(column.dropna()))


def summary(df=df):
    summary_df = pd.DataFrame({
#        '1 id': range(1,df.shape[1]+1),
        'dtypes': df.dtypes.astype(str),
        'missing count': df.isna().sum(),
        'missing ratio': (df.isnull().mean()*100).round(0).astype(int).astype(str) + ' %',
        'values': [get_unique_values(df[col]) for col in df.columns],
        'values max': [max_value(df[col]) for col in df.columns],
        'unique': df.nunique() #datatype float if NaN in df
    })
    return summary_df


def prepare_and_merge(base_df, additional_df, name_df):
    """Bereitet den Merge zwischen base_df und additional_df vor und gibt das gemergte DataFrame zurÃ¼ck."""
    
    summary_additional = summary(additional_df)

    # Fehlende Spalten in additional_df hinzufÃ¼gen
    missing_columns = set(base_df.index.tolist()) - set(additional_df.columns)
    for col in missing_columns:
#        summary_additional.loc[col] = np.nan
        summary_additional.loc[col] = ""
      
    # ÃœberzÃ¤hlige Spalten in summary_additional entfernen  # for example column ID
    too_much_columns = set(additional_df.columns) - set(base_df.index.tolist())
 #  summary_additional = summary_additional.drop(index=too_much_columns)
    summary_additional = summary_additional.drop(too_much_columns, axis=0)
    
    # Spaltennamen umbenennen damit keine Verwechslung mit base_df kommt
    summary_additional.columns = [f"{col} ({name_df})" for col in summary_additional.columns]
 
    # ZusammenfÃ¼hren der Zusammenfassung-DataFrames
    summary_df = pd.concat([base_df, summary_additional], axis=1)

    # Spalten alphabetisch sortieren damit nicht base_df vorne und additional_df hinten
    summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)

    return summary_df


##############################################################################################################
####
##################################################################################################################



import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def plot_quantitative_var_distributions(df, var_list, bin_dict=None):
    """
    Plottet Histogramm, Boxplot und KDE fÃ¼r jede Variable in der Ã¼bergebenen Liste.
    
    Parameters:
    df (pd.DataFrame): Der DataFrame, der die Daten enthÃ¤lt.
    var_list (list): Liste der numerischen Variablen, die geplottet werden sollen.
    bin_dict (dict): Dictionary mit benutzerdefinierten Bins fÃ¼r spezifische Variablen. 
                     Falls nicht angegeben, wird die Anzahl irgendwie festgelegt
    """
    
    for col in var_list:
        plt.figure(figsize=(12, 3))
    
    
        # Histogramm
        plt.subplot(1, 3, 1)
        if col in bin_dict:
            df[col].plot.hist(color='skyblue', bins=bin_dict[col])
        else:    
            df[col].plot.hist(color='skyblue')
        plt.title(f'Histogram of {col}')
        
        
        # Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f'Boxplot of {col}')
        plt.xlabel('')

        # KDE
        plt.subplot(1, 3, 3)
        sns.kdeplot(df[col], fill=True)
        plt.title(f'KDE of {col}')
        plt.xlabel('')

        plt.tight_layout()
        plt.show()


##############################################################################################################
####
##################################################################################################################




path = '/kaggle/input/playground-series-s5e2/'
df = pd.read_csv(path + 'train.csv')
df_test  = pd.read_csv(path + 'test.csv')
df_sub = pd.read_csv(path + 'sample_submission.csv')
df_org = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')

############################
# delete unnecessary columns
df = df.drop('id',axis=1)

###########################
# change data-type to Boolean
for dataset in [df, df_test, df_org]:
    dataset['Laptop Compartment'] = dataset['Laptop Compartment'].map({'Yes': 1, 'No': 0})
    dataset['Waterproof'] = dataset['Waterproof'].map({'Yes': 1, 'No': 0}) 

############################
# dataset shape
for id_df, dataset in enumerate([df, df_test, df_sub, df_org]):
    print(f"Dataset {id_df} contains {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
print("="*80)

############################

missing_columns = set(df.columns) - set(df_test.columns)
print("These columns are in df but are not in df_test:", missing_columns)
too_much_columns = set(df_test.columns) - set(df.columns) 
print("These columns are in df_test but are not in df:", too_much_columns)
missing_columns = set(df.columns) - set(df_org.columns)
print("These columns are in df but are not in df_org:", missing_columns)
too_much_columns = set(df_org.columns) - set(df.columns) 
print("These columns are in df_org but are not in df:", too_much_columns)

############################
# change display settings 
pd.set_option('display.max_columns', None) #to show all columns


df.head()


ordinal_variable_order = {
    'Size': ['Small', 'Medium', 'Large']
}
for column, value_ordering in ordinal_variable_order.items():
    for dataset in [df, df_test, df_org]:
        dataset[column] = pd.Categorical(dataset[column], categories=value_ordering, ordered=True) 





desc = df.describe(include=['object', 'category', 'bool', np.number,'datetime']).round(2).fillna('').T.drop(columns=['unique', 'count'])

pd.concat([summary(df) , desc], axis=1)


merged_test_summary  = prepare_and_merge(summary(df), df_test,'test')
summary_df = prepare_and_merge(merged_test_summary, df_org,'org')

summary_df#.sort_values(by=['dtypes'])


variables = df.select_dtypes(include='number').columns.tolist()

col = ['Compartments', 'Laptop Compartment', 'Waterproof']
custom_bins = df[col].nunique().to_dict() #anzahl der einzigartiges Values


for dataset in [df, df_org, df_test]:
    if dataset is df_org:
        print("\n","="*90,"\nOriginal Data\n","="*90)
    elif dataset is df_test:
        print("\n","="*90,"\nTest Data\n","="*90)
        variables.remove('Price')
        
    plot_quantitative_var_distributions(dataset, variables, custom_bins)


import matplotlib.pyplot as plt

def plot_density_relationship(feature, target):
    clean_data = df.dropna(subset=[feature])

    plt.figure(figsize=(10, 5)) 
    plt.hist2d(x=clean_data[feature], y=clean_data[target], cmap='Blues')
    plt.colorbar()
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"frequency of Data Points: {feature} vs {target}")
    plt.show()

plot_density_relationship('Weight Capacity (kg)', 'Price')
plot_density_relationship('Compartments', 'Price')



import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def plot_quantitative_qualitative_distributions(df, numerical_var, categorical_var):
    """
    Erstellt 6 verschiedene Visualisierungen zur Analyse der Beziehung zwischen einer quantitativen und einer qualitativen Variable.
    Die Funktion unterstÃ¼tzt zwei AnwendungsfÃ¤lle:

    Fall 1: Analyse numerischer Variablen nach Kategorien (Classification)
        - numerical_var: Quantitativ/Numerisch (z.B. 'Age', 'Income')
        - categorical_var: Qualitativ/Kategorisch (z.B. 'PurchaseStatus')
        Beispiel:
        >>> target = 'PurchaseStatus'
        >>> for num_var in ['Age', 'Income']:
        >>>     plot_quantitative_qualitative_distributions(df, num_var, target, {})

    Fall 2: Analyse kategorischer Features fÃ¼r numerisches Target (Regression)
        - numerical_var: Quantitativ/Numerisch (z.B. 'Price', 'CarbonEmission')
        - categorical_var: Qualitativ/Kategorisch (z.B. 'Gender', 'Vehicle_Type')
        Beispiel:
        >>> target = 'Price'
        >>> for cat_var in ['Gender', 'Vehicle_Type']:
        >>>     plot_quantitative_qualitative_distributions(df, target, cat_var, ordinal_variable_order)

    Parameter:
    ----------
    df (pd.DataFrame): Der DataFrame, der die Daten enthÃ¤lt
    numerical_var (str): numerische Variable auf der y-Achse
        - Fall 1: Name der numerischen Variable (z.B. 'Age')
        - Fall 2: Name der numerischen Zielvariable (z.B. 'Price')
    categorical_var (str): kategoriale Variable auf der x-Achse
        - Fall 1: Name der kategorialen Zielvariable (z.B. 'PurchaseStatus')
        - Fall 2: Name eines kategorialen Features (z.B. 'Gender')
    ordinal_variable_order (dict): Ordnung der ordinalen Variablen
    """
        
    # Erstelle eine groÃŸe Figure fÃ¼r alle 6 Subplots
    plt.figure(figsize=(15, 4))
    
    # Dictionary fÃ¼r Plot-Konfigurationen
    plot_configs = {
        (0, 0): {
            'func': sns.countplot,
            'kwargs': {'data': df, 'x': categorical_var, 'palette': "Set2"},
            'title': f'Frequency Distribution: {categorical_var}'
        },
        (0, 1): {
            'func': sns.violinplot,
            'kwargs': {'x': categorical_var, 'y': numerical_var, 'data': df, 'palette': "Set2"},
            'title': f'Violinplot: {categorical_var} vs. {numerical_var}'
        },
        (0, 2): {
            'func': sns.pointplot,
            'kwargs': {'x': categorical_var, 'y': numerical_var, 'data': df, 'palette': "Set2",  'capsize': .2},
            'title': f'Mean with Standard Deviation: {categorical_var} vs. {numerical_var}'
        }
    }
    
    # Erstelle alle Plots
    for (row, col), config in plot_configs.items():
        plt.subplot(1, 3, col + 1)
        config['func'](**config['kwargs'])
        plt.title(config['title'])
        plt.xticks(rotation=45, ha='right')
    
    # Adjustiere das Layout
    plt.tight_layout()
    plt.show()



target = 'Price'

categorical_variables = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist() + ["Compartments",'Laptop Compartment', 'Waterproof']

for cat_var in categorical_variables:
    plot_quantitative_qualitative_distributions(df, target, cat_var)


def plot_categorical_distributions_dftest(df: pd.DataFrame,   list_categorical_var: list, ordinal_variable_order: dict = None):
    """
    Plots the count of categorical features against a target variable.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    list_categorical_var (list): List of categorical variables to plot.
    ordinal_variable_order (dict): Dictionary indicating the order of ordinal variables. 
                                    If not provided, variables are treated as nominal.
    """
    for feature in list_categorical_var:
        if ordinal_variable_order and feature in ordinal_variable_order:
            order = ordinal_variable_order[feature]
        else:
            order = None

        plt.figure(figsize=(16, 4))

        # Plot 0: Count plot
        plt.subplot(1, 3, 1)
        sns.countplot(data=df, x=feature, order=order)
#        plt.title(f'Count of {feature}')
        plt.xticks(rotation=0)
        

        plt.tight_layout()
        plt.show()


ordinal_variable_order = {
    'Size': ['Small', 'Medium', 'Large']
}

variables = df.select_dtypes(include=['object', 'category','bool']).columns.tolist() + ["Compartments",'Laptop Compartment', 'Waterproof']


plot_categorical_distributions_dftest(df_test, variables, ordinal_variable_order)


df_corr_ordinal = pd.get_dummies(df, columns=['Brand', 'Material', 'Style', 'Color']) 


# encoding for ordinal variables based on defined order
ordinal_variable_order = {
    'Size': ['Small', 'Medium', 'Large']
}
for column, column_ordering in ordinal_variable_order.items():
   mapping = {category: idx for idx, category in enumerate(column_ordering)}
   df_corr_ordinal[column] = df[column].map(mapping)

# delete upper diagonal matrix
mask = np.triu(np.ones_like(df_corr_ordinal.corr(), dtype=bool))

plt.figure(figsize=(24, 9)) 
sns.heatmap(df_corr_ordinal.corr(),fmt = '.2f', cmap="magma", annot=True, mask=mask)


plt.title('Pearson Correlation Matrix')
plt.show()


df_corr_ordinal = pd.get_dummies(df_org, columns=['Brand', 'Material', 'Style', 'Color']) 


# encoding for ordinal variables based on defined order
ordinal_variable_order = {
    'Size': ['Small', 'Medium', 'Large']
}
for column, column_ordering in ordinal_variable_order.items():
   mapping = {category: idx for idx, category in enumerate(column_ordering)}
   df_corr_ordinal[column] = df[column].map(mapping)

# delete upper diagonal matrix
mask = np.triu(np.ones_like(df_corr_ordinal.corr(), dtype=bool))

plt.figure(figsize=(16, 8)) 
#sns.heatmap(df_corr_ordinal.corr(),fmt = '.1f', cmap="coolwarm", annot=True)
sns.heatmap(df_corr_ordinal.corr(),fmt = '.2f', cmap="seismic", annot=True, mask=mask,vmax=1,vmin=-1) #cmap="coolwarm"  #cmap="seismic"
plt.title('Pearson Correlation Matrix for original Data')
plt.show()


for id_df, dataset in enumerate([df, df_test, df_org]):
    print(f"Number of Duplicates in dataframe {id_df}: {dataset.duplicated().sum()}")


df_org.loc[df_org.duplicated()]


df_org[(df_org['Brand']=='Adidas') & (df_org['Size']=='Large') & ((df_org['Compartments']==4) | (df_org['Compartments']==9)) & (df_org['Waterproof']==1) & ((df_org['Color']=='Pink') | (df_org['Color']=='Red')) & (df_org['Weight Capacity (kg)']==30) & (df_org['Price'].isna())].sort_values(by=['Style'])


print("The shape of the DataFrame is:", df_org.shape)

#Drop Duplicates
df_org.drop_duplicates(inplace = True)

#Checking if delete was successful 
print("Number of Duplicates:",df_org.duplicated().sum())
print("The shape of the DataFrame is:", df_org.shape)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


X = df.drop(["Price"], axis=1)
y = df["Price"]

bool_features= ['Laptop Compartment', 'Waterproof']
#num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_features = list(set(X.select_dtypes(include=['int64', 'float64']).columns) - set(bool_features))
cat_features = X.select_dtypes(include=['object','category']).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ]), num_features),
    ("bool", SimpleImputer(strategy="most_frequent"), bool_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first"))
    ]), cat_features)
],remainder="passthrough")


X_transformed = preprocessor.fit_transform(X)
X_pred= df_test[X.columns.tolist()]
X_pred = preprocessor.transform(X_pred)

preprocessor


def get_onehot_encoded_columns(column_transformer, begin_index, end_index):
    """
    Extracts the OneHotEncoded feature names for the specified transformers in the ColumnTransformer.

    Args:
        column_transformer (ColumnTransformer): Fitted ColumnTransformer object.
        begin_index (int): Starting index of the ordinal encoders.
        end_index (int): Ending index (exclusive) of the ordinal encoders.

    Returns:
        List[str]: List of one-hot encoded feature names.
    """
    ohe_feature_names = []
    for col_name, _, col_list in column_transformer.transformers_[begin_index:end_index]:
        ohe_features = column_transformer.named_transformers_[col_name].get_feature_names_out(col_list).tolist()
        ohe_feature_names.extend(ohe_features)
    return ohe_feature_names


dummy_categorical_features = get_onehot_encoded_columns(preprocessor, 2, 3)  

transformed_feature_names = num_features + bool_features + dummy_categorical_features 

X_transformed = pd.DataFrame(X_transformed, columns= transformed_feature_names)

X_transformed


X_pred = pd.DataFrame(X_pred, columns= transformed_feature_names)
X_pred




