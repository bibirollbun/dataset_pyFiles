import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Statistical methods
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency

# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings

# Warnings
warnings.filterwarnings('ignore')



df = pd.read_csv('/kaggle/input/santander-product-recommendation/train_ver2.csv.zip')


df.head()



df.shape


df.isna().sum()


df['ncodpers'].nunique()


df[df['ncodpers']==1050678]


(13647309/956645)


print(len(df.select_dtypes(include = 'object').columns))
print(len(df.select_dtypes(include = 'number').columns))


df['age'].unique()


df["age"]   = pd.to_numeric(df["age"], errors="coerce")
sns.set_style("whitegrid")
sns.histplot(
    df["age"].dropna(),  # Drop NaN values
    bins=80,             # Number of bins
    stat='count',        # Show actual counts
    color="tomato"
)
plt.title("Age Distribution")
plt.ylabel("Count")
plt.xlabel("Age")
plt.ticklabel_format(style='plain', axis='y')

plt.show()


df['sexo'].value_counts()


df['tiprel_1mes'].value_counts()


df['ind_actividad_cliente'].value_counts()


df['segmento'].value_counts()


df['renta'].describe()


df.columns.get_loc('ind_ahor_fin_ult1')


product = df.iloc[:,24:]
product.head()


# Count the occurrences of '1' in each column
count_ones = (product == 1).sum()

# Sort the counts in descending order
sorted_counts = pd.DataFrame(count_ones.sort_values(ascending=False)).reset_index()
sorted_counts.columns = ['product','count']
# Display the sorted counts
print(sorted_counts)


sorted_counts['proportion'] =  sorted_counts['count']/13647309


sorted_counts

