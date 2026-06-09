import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as snb


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings
warnings.filterwarnings('ignore')


DROP_COLS = ['id']
TARTGET_COL = 'Fertilizer Name'
TRAIN_PATH = "/kaggle/input/playground-series-s5e6/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e6/test.csv"


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df.head()


test_df.head()


train_df = train_df.drop(DROP_COLS, axis='columns') if 'id' in train_df.columns else train_df
train_df.info()


class DataProcessing:
    def __init__(self, cat_cols:list, num_cols:list, target_cols:str):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_cols = target_cols


    def catColsProcessing(self, df:pd.DataFrame, typ:str, is_target:bool) -> pd.DataFrame:
        """
        Args:
            df: dataframe
            typ: choose from ('label','onehot')
;
        Return: 
            processed dataframe in pd.DataFrame format
        """
        df_copy = df.copy()
        
        if typ == 'label':
            encoder = LabelEncoder()

        elif typ == 'onehot':
            if self.target_cols not in df_copy.columns or self.target_cols is not None:
                df_copy = df_copy.drop([self.target_cols], axis='columns')
                
                for col in [self.target_cols]:
                    if col in self.cat_cols:
                        self.cat_cols.remove(col)  
                print(self.target_cols, self.cat_cols)
                
            df_encoded = pd.get_dummies(df_copy, columns=self.cat_cols)
            df_temp = df_copy.drop(self.cat_cols, axis='columns')
            final_df = pd.concat([df_temp, df_encoded], axis='columns')
            return final_df

        else:
            raise ValueError(typ + " is not valid choose from ('label', 'onehot')")


        if is_target:
            for col in self.cat_cols:
                df_copy[col] = encoder.fit_transform(df[col])
        else:
            self.cat_cols.remove(self.target_cols)
            for col in self.cat_cols:
                df_copy[col] = encoder.fit_transform(df[col])
            
        return df_copy


    
    def numColsProcessing(self, df:pd.DataFrame, typ:str) -> pd.DataFrame:
        """
        Args:
            df: dataframe
            typ: choose from ('std', 'minmax')

        Return:
            processed dataframe in pd.DataFrame format
        """
        if typ == 'minmax':
            scaler = MinMaxScaler()

        elif typ == 'std':
            scaler = StandardScaler()

        else:
            raise ValueError(typ + " is not valid choose from ('std', 'minmax')")

        
        scaled_data = scaler.fit_transform(df[self.num_cols])
        scaled_data = pd.DataFrame(scaled_data, columns=self.num_cols)
        df_copy = df.drop(self.num_cols, axis='columns')
        final_df = pd.concat([df_copy, scaled_data], axis='columns')
        
        return final_df



    def catNumProcessing(self, df:pd.DataFrame, num_typ:str, cat_typ:str, is_target:bool=True) -> pd.DataFrame:
        """
        Args:
            df: dataframe
            num_typ: choose from ('std', 'minmax')
            cat_typ: choose from ('label', 'onehot')

        Return:
            processed dataframe in pd.DataFrame format
        """

        df = self.numColsProcessing(df, num_typ)
        df = self.catColsProcessing(df, cat_typ, is_target=is_target)
        
        return df



cat_cols = list(train_df.select_dtypes(exclude=np.number).columns)
num_cols = list(train_df.select_dtypes(include=np.number).columns)
target_cols='Fertilizer Name'

processor = DataProcessing(cat_cols, num_cols, target_cols=target_cols)
train_processed = processor.catNumProcessing(df=train_df, num_typ='minmax', cat_typ='label',is_target=True)
test_processed = processor.catNumProcessing(df=test_df, num_typ='minmax', cat_typ='label', is_target=False)
train_processed.head()


test_processed.head()


import plotly.graph_objects as go

target_counts = train_processed['Fertilizer Name'].value_counts().sort_index()

# Create pie chart
fig = go.Figure(data=[go.Pie(
    labels=target_counts.index.astype(str),
    values=target_counts.values,
    hole=0.3,  # Optional: for donut chart
    textinfo='percent+label'
)])

# Customize layout
fig.update_layout(
    title_text="Distribution of Fertilizer ",
    legend_title="Classes"
)

# Show plot
fig.show()





numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Plot distribution for each numerical column
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 2, i)
    snb.histplot(train_processed[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()






correlation_df = pd.DataFrame()
correlation_col = []
correlation_vals = []
for col in train_processed.columns:
    corr_valu = train_processed['Fertilizer Name'].corr(train_processed[col])
    correlation_col.append(col)
    correlation_vals.append(corr_valu)

correlation_df['Column_Name'] = correlation_col
correlation_df['Correlation_With (Fertilizer Name)'] = correlation_vals
correlation_df


from sklearn.model_selection import train_test_split
X, y = train_processed.drop(['Fertilizer Name'], axis=1), train_processed['Fertilizer Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape, y_test.shape


models_and_parameters = {
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)

}


%pip install -q dagshub mlflow
%pip install mlflow


import dagshub
dagshub.init(repo_owner='vijaytakbhate2002', repo_name='fertilizer-prediction-experiment-tracking', mlflow=True)


import mlflow

def validation(y_test, y_pred):
    recall = recall_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)
    
    return {'precision':precision, 'recall':recall, 'f1_score': f1, 'accuracy':accuracy}


mlflow.set_experiment("Model Selection Experiment")

# for name, model in models_and_parameters.items():
#     with mlflow.start_run(run_name=name):
#         print(name, " training initiated")
#         model = model.fit(X_train, y_train)
#         print(name, " training is done...")
#         y_pred = model.predict(X_test)
#         metrics = validation(y_test, y_pred)

#         mlflow.log_metrics(metrics)



runs = mlflow.search_runs(
    # experiment_ids=[experiment.experiment_id],
    filter_string="metrics.accuracy > 0.15",
    order_by=["metrics.f1_score DESC"],
    max_results=5
)


runs


model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred


cm = confusion_matrix(y_test, y_pred)  
snb.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()







