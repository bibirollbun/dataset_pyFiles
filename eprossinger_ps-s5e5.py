import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
# Matplotlib config
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
%config InlineBackend.rc = {'figure.figsize': (5.0, 3.0)}


path = '/kaggle/input/playground-series-s5e5/'
df = pd.read_csv(path + 'train.csv')
df_test  = pd.read_csv(path + 'test.csv')
df_sub = pd.read_csv(path + 'sample_submission.csv')
df_org=pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')

############################
# delete unnecessary columns
df=df.drop('id',axis=1)
df_org=df_org.drop('User_ID',axis=1)

############################
# dataset shape
for id_df, dataset in enumerate([df, df_test, df_sub, df_org]):
#    print(f"dataset shape of dataframe {id_df} is: {dataset.shape}")
    print(f"Dataset {id_df} contains {dataset.shape[0]} rows and {dataset.shape[1]} columns.")
print("="*80)

############################
# Rename Columns

df_org.rename(columns= {'Gender':'gender_male'}, inplace = True)
for dataset in [df, df_test]:
    dataset.rename(columns= {'Sex':'gender_male'}, inplace = True)



    # Replace spaces with underscores & make all column names lowercase
#for dataset in [df, df_test, df_org]:
#    dataset.columns = dataset.columns.str.lower()


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
# change data-type to Boolean
for dataset in [df, df_test, df_org]:
    dataset['gender_male'] = dataset['gender_male'].map({'male': 1, 'female': 0}) 


############################
# change display settings 
pd.set_option('display.max_columns', None) #to show all columns

#pd.reset_option(â€œmax_columnsâ€�) #RÃ¼ckgÃ¤ngig machen
pd.set_option('max_colwidth', 75) #Set the Column Width #You can increase the width by passing an int. Or put at the max passing None:
#pd.set_option('max_colwidth', None) #Set the Column Width #You can increase the width by passing an int. Or put at the max passing None:
#pd.reset_option('max_colwidth') #RÃ¼ckgÃ¤ngig machen
#pd.set_option(â€˜precisionâ€™, 2) #number of places after the decimal in DataFrame is 2
# change display settings if there are many columns in df
#pd.set_option("display.max_rows", number_rows+1)


df.head()



df_org


# DataFrame Core Statistics Functions ğŸ”¬<a id="basic"></a>

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
        '1 id': range(1,df.shape[1]+1),
        'dtypes': df.dtypes.astype(str),
        'missing count': df.isna().sum(),
        'missing ratio': (df.isnull().mean()*100).round(0).astype(int).astype(str) + ' %',
#        'missing ratio': round(df.isna().sum() / df.shape[0],2),
 #       'missing ratio2': (df.isna().sum() / df.shape[0] * 100).round(0).astype(int).astype(str) + ' %',
        'values': [get_unique_values(df[col]) for col in df.columns],
        'values max': [max_value(df[col]) for col in df.columns],
#        'z Cardinality': df.nunique() #datatype float if NaN in df
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


desc = df.describe(include=['object', 'category', 'bool', np.number,'datetime']).round(2).fillna('').T

pd.concat([summary(df) , desc], axis=1)


merged_test_summary  = prepare_and_merge(summary(df), df_test,'test')
summary_df = prepare_and_merge(merged_test_summary, df_org,'org')
summary_df.sort_values(by=['dtypes', 'unique','missing count'], ascending = False)




