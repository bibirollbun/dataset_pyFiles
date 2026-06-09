import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from tabulate import tabulate
from rich.console import Console
from rich.table import Table


# test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
# sample = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")


display(train.head(10))


def to_date(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df['dayofweek'] = df['date'].dt.day_name()

to_date(train)

# Set up the plotting style
plt.style.use('fivethirtyeight')
colors = ['#1F77B4', '#AEC7E8', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A']


def monthly_sales(df):
    plt.figure(figsize=(15, 8))
    monthly_sales = df.groupby('month')['num_sold'].sum().reset_index()
    monthly_sales['month'] = monthly_sales['month'].astype(str)
    plt.plot(monthly_sales['month'], monthly_sales['num_sold'], marker='o', linewidth=2)
    plt.title('Monthly Sales Trends', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    n = len(monthly_sales) // 10
    plt.xticks(range(0, len(monthly_sales), n), monthly_sales['month'][::n], rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

monthly_sales(train)


def product_trends(df):
    plt.figure(figsize=(15, 8))
    product_trends = df.pivot_table(
        values='num_sold',
        index='month',
        columns='product',
        aggfunc='sum'
    ).reset_index()
    product_trends.index = product_trends.index.astype(str)
    
    for i, product in enumerate(df['product'].unique()):
        plt.plot(product_trends.index, product_trends[product], 
                 label=product, color=colors[i % len(colors)], linewidth=2)
    plt.title('Product Performance Trends', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    n = len(product_trends) // 10
    plt.xticks(range(0, len(product_trends), n), product_trends.index[::n], rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

product_trends(train)


def sales_by_store_and_prod(df):
    plt.figure(figsize=(12, 6))
    store_product = df.pivot_table(
        values='num_sold',
        index='store',
        columns='product',
        aggfunc='sum'
    )
    ax = store_product.plot(kind='barh', stacked=False, color=colors, figsize=(12, 6))
    plt.title('Store-Product Performance', fontsize=14, pad=20)
    plt.xlabel('Sales Volume', fontsize=12)
    plt.ylabel('Store', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Products')
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    plt.tight_layout()
    plt.show()

sales_by_store_and_prod(train)


def sales_by_store(df):
    plt.figure(figsize=(15, 8))
    store_monthly = df.pivot_table(
        values='num_sold',
        index='month',
        columns='store',
        aggfunc='sum'
    ).reset_index()
    
    store_monthly.index = store_monthly.index.astype(str)
    
    for i, store in enumerate(df['store'].unique()):
        plt.plot(
            store_monthly.index,
            store_monthly[store],
            label=store,
            color=colors[i % len(colors)],
            linewidth=2
        )
    
    plt.title('Store Sales Trends Over Time', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    
    n = len(store_monthly) // 10
    plt.xticks(range(0, len(store_monthly), n), store_monthly.index[::n], rotation=45, ha='right')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title='Stores')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

sales_by_store(train)


def sales_by_country(df):
    plt.figure(figsize=(12, 8))
    country_sales = df.groupby('country')['num_sold'].sum()
    plt.pie(country_sales, labels=country_sales.index, autopct='%1.1f%%', 
            colors=colors[:len(country_sales)], textprops={'fontsize': 12})
    plt.title('Country Market Share', fontsize=14, pad=20)
    plt.legend(labels=[f'{country}: {value:,.0f}' for country, value in country_sales.items()],
              title="Sales by Country",
              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

sales_by_country(train)


def sales_by_country_over_time(df):
    plt.figure(figsize=(15, 8))
    country_monthly = df.pivot_table(
        values='num_sold',
        index='month',
        columns='country',
        aggfunc='sum'
    ).reset_index()
    country_monthly.index = country_monthly.index.astype(str)
    
    for i, country in enumerate(df['country'].unique()):
        plt.plot(country_monthly.index, country_monthly[country], 
                 label=country, color=colors[i % len(colors)], linewidth=2)
    plt.title('Country Sales Trends Over Time', fontsize=14, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    n = len(country_monthly) // 10
    plt.xticks(range(0, len(country_monthly), n), country_monthly.index[::n], rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

sales_by_country_over_time(train)


console = Console()

def print_statistics(df):
    console.rule("[bold blue] Key Statistics")
    
    console.print("\n[bold cyan]1. Overall Sales Statistics:[/bold cyan]")
    overall_stats = df['num_sold'].describe().apply(lambda x: format(x, ',.2f')).to_dict()
    overall_table = Table(show_header=True, header_style="bold magenta")
    overall_table.add_column("Metric", justify="left")
    overall_table.add_column("Value", justify="right")
    for key, value in overall_stats.items():
        overall_table.add_row(key, value)
    console.print(overall_table)

    console.print("\n[bold cyan]2. Country-wise Sales Statistics:[/bold cyan]")
    country_stats = df.groupby('country')['num_sold'].agg(['mean', 'std', 'min', 'max']).applymap(lambda x: format(x, ',.2f'))
    console.print(tabulate(country_stats, headers="keys", tablefmt="grid"))

    console.print("\n[bold cyan]3. Product Performance Statistics:[/bold cyan]")
    product_stats = df.groupby('product')['num_sold'].agg(['mean', 'std', 'min', 'max']).applymap(lambda x: format(x, ',.2f'))
    console.print(tabulate(product_stats, headers="keys", tablefmt="fancy_grid"))

    console.print("\n[bold cyan]4. Store Performance Statistics:[/bold cyan]")
    store_stats = df.groupby('store')['num_sold'].agg(['mean', 'std', 'min', 'max']).applymap(lambda x: format(x, ',.2f'))
    console.print(tabulate(store_stats, headers="keys", tablefmt="pipe"))

    pivot_data = df.pivot_table(
        values='num_sold',
        index='date',
        columns='product',
        aggfunc='sum'
    )
    console.print("\n[bold cyan]5. Product Sales Correlation Matrix:[/bold cyan]")
    correlation_matrix = pivot_data.corr().round(3)
    console.print(tabulate(correlation_matrix, headers="keys", tablefmt="rounded_outline"))

# Call the function
print_statistics(train)

