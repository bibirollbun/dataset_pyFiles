import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import duckdb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_file_path = '/kaggle/input/neo-bank-non-sub-churn-prediction/train_*.parquet' 


df_train = pd.concat([pd.read_parquet(file) for file in (glob.glob(train_file_path))], ignore_index=True)


display(df_train)


test_file_path = '/kaggle/input/neo-bank-non-sub-churn-prediction/test.parquet' 


df_test = pd.concat([pd.read_parquet(file) for file in (glob.glob(test_file_path))], ignore_index=True)


display(df_test)


df_train['data_type'] = 'train'
df_test['data_type'] = 'test'


df = pd.concat([df_train, df_test], ignore_index=True)


def bar_plot(data, figure_size, x_label, y_label, fig_title, bar_color):

    plt.figure(figsize=figure_size)
    plt.bar(data.index, data.values, color=bar_color, alpha=0.8, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)

    plt.xticks(ticks=data.index)

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def line_plot(data, value, figure_size, x_label, y_label, fig_title):
    # Filtra os dados para a data mínima
    agg_data = data.groupby(['date', 'data_type'])[value].sum().reset_index()
    agg_data = agg_data[agg_data['date'] >= pd.to_datetime('2008-01-17')]

    # Configura a figura
    plt.figure(figsize=figure_size)

    # Plota as linhas para cada tipo de dado
    for data_type, group_data in agg_data.groupby('data_type'):
        color = '#BFD641' if data_type == 'test' else '#7DDA58'
        plt.plot(group_data['date'], group_data[value], color=color, linewidth=1, label=data_type)

    # Preenchimento entre o eixo x e os valores
    plt.fill_between(
        agg_data['date'], 
        agg_data.groupby('date')[value].sum(), 
        color='gray', 
        alpha=0.2
    )

    # Configurações dos eixos e título
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Configuração dos ticks do eixo x
    years = agg_data['date'].dt.year.unique()
    plt.xticks(ticks=pd.to_datetime([f"{year}-01-01" for year in years]), labels=years)
    
    # Adiciona legenda
    plt.legend(title='Data Type')
    
    plt.tight_layout()
    plt.show()



def histogram_plot(data, figure_size, bins_value, x_label, y_label, fig_title, color='blue', alpha=0.8):

    plt.figure(figsize=figure_size)
    plt.hist(data, bins=bins_value, color=color, alpha=alpha, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_title)

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()



df.info()


print('columns on dataset:')
df.columns


print('number of rows years of train dataset:')

duckdb.query("""
    SELECT 
        DISTINCT YEAR(date) AS year,
        COUNT(customer_id)
    FROM df
    GROUP BY year
    ORDER BY year ASC
""")

# in python
# df.groupby(df['date'].dt.year)['customer_id'].count()


print('number of customers:')

duckdb.query("""
    SELECT 
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM df
""")

# in python
# len(df['customer_id'].unique())


print('number of churn:')

duckdb.query("""
    SELECT 
        churn_due_to_fraud, 
        COUNT(*) AS count
    FROM 
        df
    GROUP BY 
        churn_due_to_fraud
    ORDER BY 
        count DESC
""")

# in python
# df['churn_due_to_fraud'].value_counts()


print('number of model predicted_fraud:')

duckdb.query("""
    SELECT
        model_predicted_fraud,
        COUNT(*) AS count
    FROM
        df
    GROUP BY
        model_predicted_fraud
    ORDER BY
        count DESC             
""")

# in python
# df['model_predicted_fraud'].value_counts()


print('customers per countries:')

duckdb.query("""
    SELECT 
        country, 
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM df
    GROUP BY country
    ORDER BY unique_customers DESC
""")

# in python
# df.groupby('country')['customer_id'].nunique().sort_values(ascending=False)


bar_plot(df['date'].dt.year.value_counts().sort_index(),
        (10,3),
        'Years',
        'Number',
        'Number of data records per year',
        '#5DE2E7')


line_plot(df,
         'atm_transfer_in',
        (10,3),
        'Years',
        'Transactions',
        'Number of ATM pay-ins')


line_plot(df,
         'atm_transfer_out',
        (10,3),
        'Years',
        'Transactions',
        'Number of ATM withdrawals')


line_plot(df,
         'bank_transfer_in',
        (10,3),
        'Years',
        'Transactions',
        'Number of in-going transactions')


line_plot(df,
         'bank_transfer_out',
        (10,3),
        'Years',
        'Transactions',
        'Number of out-going transactions')


line_plot(df,
         'bank_transfer_in_volume',
        (10,3),
        'Years',
        'Volume',
        'Total volume of in-going transactions')


line_plot(df,
         'bank_transfer_out_volume',
        (10,3),
        'Years',
        'Volume',
        'Total volume of out-going transactions')


line_plot(df,
         'crypto_in',
        (10,3),
        'Years',
        'Transactions',
        'Number of buying-crypto transactions')


line_plot(df,
         'crypto_out',
        (10,3),
        'Years',
        'Transactions',
        'Number of selling-crypto transactions')


line_plot(df,
         'crypto_in_volume',
        (10,3),
        'Years',
        'Volume',
        'Total volume of buying-crypto transactions')


line_plot(df,
         'crypto_out_volume',
        (10,3),
        'Years',
        'Volume',
        'Total volume of selling-crypto transactions')


histogram_plot(df['interest_rate'], 
               (5,3), 
               10, 
               'Value ranges', 
               'Frequency', 
               'Histogram of bank account interest rate')


print('Sorting df by customer id and date:')

df = df.sort_values(by=['customer_id', 'date'])


df_analytics = (df[['Id', 'customer_id', 'date']]).copy()

df_analytics['last_activity'] = df_analytics.groupby('customer_id')['date'].shift(1)

df_analytics['days_inactive'] = (df_analytics['date'] - df_analytics['last_activity']).dt.days

df_analytics['days_inactive'] = df_analytics['days_inactive'].fillna(0)


df_analytics


df_analytics.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])


histogram_plot(df_analytics['days_inactive'], 
               (5,3), 
               1000, 
               'Value ranges', 
               'Frequency', 
               'Histogram of bank account interest rate')


df_analytics = pd.merge(df_analytics,
                        df[['Id', 'data_type', 'atm_transfer_in', 'atm_transfer_out', 'bank_transfer_in', 'bank_transfer_out', 'crypto_in', 'crypto_out']],
                        on='Id',
                        how='left')


df_analytics['total_transactions'] = sum([df_analytics['atm_transfer_in'], df_analytics['atm_transfer_out'], df_analytics['bank_transfer_in'], df_analytics['bank_transfer_out'], df_analytics['crypto_in'], df_analytics['crypto_out']])


line_plot(df_analytics,
         'total_transactions',
        (10,3),
        'Years',
        'Volume',
        'Total of transactions')


df_analytics = pd.merge(df_analytics,
                        df[['Id', 'bank_transfer_in_volume', 'bank_transfer_out_volume', 'crypto_in_volume', 'crypto_out_volume']],
                        on='Id',
                        how='left')


df_analytics['total_volume'] = sum([df_analytics['bank_transfer_in_volume'], df_analytics['bank_transfer_out_volume'], df_analytics['crypto_in_volume'], df_analytics['crypto_out_volume']])


line_plot(df_analytics,
         'total_volume',
        (10,3),
        'Years',
        'Volume',
        'Total transactions volume')


df_rfm = df_analytics[['Id', 'data_type','customer_id', 'date', 'days_inactive', 'total_transactions', 'total_volume']].copy()


df_rfm

