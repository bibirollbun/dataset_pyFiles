import numpy as np
import pandas as pd
import seaborn as sns
import random as rm


import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot  as plt
import plotly.graph_objs as go
import plotly.subplots as sp
from sklearn.metrics import mean_squared_error


train = pd.read_csv(r"/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv(r"/kaggle/input/playground-series-s5e4/test.csv")
sub = pd.read_csv(r"/kaggle/input/playground-series-s5e4/sample_submission.csv")


print(f"Size of Training Dataset :{train.shape}")
print(f"Size of Testing Dataset :{test.shape}")


display(train.head(5))
print("-"*50, "Train vs Test", "-"*50)
display(test.head(5))


train.drop(columns=['id'],inplace=True,errors='ignore')
test.drop(columns=['id'],inplace=True,errors='ignore')


print(f"No of null Value on train dataset: {train.isna().sum()/train.shape[0]*100}")


train.info()


train.describe(include = "all")


sns.heatmap(train.corr(numeric_only = True),annot = True,cmap = "coolwarm")


def kde(df):
    numeric_col = df.select_dtypes(include=[np.number]).columns
    num_plots = len(numeric_col)

    num_cols = 2  
    num_rows = (len(numeric_col) + num_cols - 1) // num_cols 

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    fig.suptitle('Distribution of Numerical Features', fontsize=16)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

     # Iterate over each categorical column and create a bar plot
    for i, col in enumerate(numeric_col):
        sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue", element="step", stat="density")
        # Setting titles and labels
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].tick_params(axis='x')

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the main title space
    plt.show()




def visualize_categorical_distributions_plotly(df, target):
    categorical_columns = df.select_dtypes(include=['object']).columns

    
    num_cols = 2
    num_rows = (len(categorical_columns) + num_cols - 1) // num_cols

    fig = sp.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f'Distribution of {col} vs {target}' for col in categorical_columns])

    for idx, col in enumerate(categorical_columns):
        counts = df.groupby(col)[target].sum().reset_index().sort_values(by=col)
 
        row = idx // num_cols + 1
        col_pos = idx % num_cols + 1

        
        fig.add_trace(
            go.Bar(x=counts[col], y=counts[target], name=col),
            row=row, col=col_pos
        )

    fig.update_layout(
        height=num_rows * 400,
        width=1100,
        title_text="Distribution of Categorical Features",
        showlegend=False
    )
    fig.show()



visualize_categorical_distributions_plotly(train,target ="Listening_Time_minutes" )


kde(train)


# For train
train['Number_of_Ads'].fillna(train['Number_of_Ads'].mean(),inplace=True)
train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean(),inplace=True)
train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(),inplace=True)
# For test
test['Number_of_Ads'].fillna(test['Number_of_Ads'].mean(),inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].mean(),inplace=True)
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(),inplace=True)
 


from sklearn.model_selection import cross_val_score,train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,r2_score
target='Listening_Time_minutes'
x=train.drop(target,axis=1)
y=train[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_cols=x.select_dtypes(include=(['int64','float64'])).columns.tolist()
cat_cols=x.select_dtypes(include=(['object'])).columns.tolist()
num_pipeline=Pipeline([('Impute',SimpleImputer(strategy='mean')),
                       ('scaler',StandardScaler())])

cat_pipeline=Pipeline([('scaler',OneHotEncoder(handle_unknown='ignore',drop='first',
                                              sparse_output=False))])

col_transformer=ColumnTransformer([('num',num_pipeline,num_cols),
                        ('cat',cat_pipeline,cat_cols)])

lg=lgb.LGBMRegressor(
        n_iter=3000,
        max_depth=100,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
        random_state=69,
        subsample=0.8,
        subsample_freq=1)


    
model=Pipeline([('pre',col_transformer),
               ('lg',lg)])

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(f'MSE: {mean_squared_error(y_test,y_pred) :.2f}')
print(f'R2 score {r2_score(y_test,y_pred) * 100 :.2f}')
rmsc=np.sqrt(mean_squared_error(y_test,y_pred))
print(f'RMSC = {rmsc :.4f}')
for actual,pred in zip(y_test[:10],y_pred[:10]):
    print(f'Actual: {actual :.2f}   | Predicted: {pred :.2f}')


# pred = model.predict(test)


# pred.shape


# sub['Listening_Time_minutes'] = pred


# sub.to_csv("Podcast Listening Pred-2.csv",index = False)

