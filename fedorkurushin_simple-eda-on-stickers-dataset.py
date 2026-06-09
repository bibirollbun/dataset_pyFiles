import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


train.head()


train.shape, test.shape


train.nunique()


train.isnull().sum()


missing_data = train.loc[train['num_sold'].isnull(), ['country', 'store', 'product', 'num_sold']]
missing_data.groupby(by=['country', 'store', 'product'], as_index=False).size()


for col_name in ['country', 'store', 'product']:
    grouped_train = train[['country', 'store', 'product', 'num_sold']].groupby(by=[col_name], as_index=False).sum()    

    plt.figure(figsize=(5, 3))
    plt.bar(grouped_train[col_name], grouped_train['num_sold'], color='skyblue')
    plt.xlabel(col_name.capitalize())
    plt.ylabel('Count')
    plt.title(f'Total solds by {col_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def aggregate_and_plot_by(col_name, train):
    unique_col_values = train[col_name].unique()
    grouped_train = train.groupby(by=['date', 'country', col_name], as_index=False).sum()
    
    for country in train['country'].unique():
        country_df = grouped_train[grouped_train['country'] == country]
        
        plt.figure(figsize=(15, 5))
        for val in unique_col_values:
            sub_country_df = country_df[country_df[col_name] == val]
            plt.plot(sub_country_df.date, sub_country_df.num_sold, label=val)
            
        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title(f'Num Sold in {country} by {col_name}')
        plt.xticks(rotation=45, ha='right')
    
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=30))
        plt.legend()
        plt.tight_layout()
        plt.show()


aggregate_and_plot_by('store', train)


aggregate_and_plot_by('product', train)




