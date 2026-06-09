import pandas as pd

df_clust = pd.read_csv('/kaggle/input/k/sthnhcng/knn-kmean-cosine/Kmean_clustered.csv')
df_clust


pd.DataFrame(df_clust.Kmean_cluster.value_counts()).reset_index()



# df_clust.drop(column=)
# df_clust = df_clust.drop(columns=['CN_WH','availability', 'type_0_discount','days_since_2020','year','type_5_discount','type_6_discount',
#                        'L2_category_name_en', 'L3_category_name_en','L4_category_name_en',
#                         'type_1_discount','type_2_discount','type_3_discount','type_4_discount','year'])
# df_clust


df_clust[df_clust['unique_id']==464]


df_clust[df_clust['unique_id']==1818]


ids = pd.DataFrame(df_clust.unique_id.value_counts()).reset_index()
ids = ids[ids['count']>13].unique_id.tolist()


df_clust = df_clust[df_clust['unique_id'].isin(ids)]
df_clust


df_clust.describe().T



import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame and it has the columns 'sales' and 'cluster'

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Kmean_cluster', y='max_discount', data=df_clust)
# Add labels and title for clarity
plt.title('max_discount Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('max_discount')

# Show the plot
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
# Distribution of total orders
plt.figure(figsize=(10, 6))
sns.histplot(df_clust['total_orders'], kde=True, bins=50)
plt.title('Distribution of Total Orders')
plt.xlabel('Total Orders')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame and it has the columns 'sales' and 'cluster'

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Kmean_cluster', y='total_orders', data=df_clust)
# Add labels and title for clarity
plt.title('total_orders Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('total_orders')

# Show the plot
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
# Distribution of total orders
plt.figure(figsize=(10, 6))
sns.histplot(df_clust['sales'], kde=True, bins=50)
plt.title('Distribution of Total Orders')
plt.xlabel('Total sales')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame and it has the columns 'sales' and 'cluster'

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Kmean_cluster', y='sales', data=df_clust)
plt.yscale('log')
# Add labels and title for clarity
plt.title('Sales Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Sales')

# Show the plot
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
# Distribution of total orders
plt.figure(figsize=(10, 6))
sns.histplot(df_clust['sell_price_main'], kde=True, bins=50)
plt.title('Distribution of Total Orders')
plt.xlabel('Total sell_price_main')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame and it has the columns 'sales' and 'cluster'

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Kmean_cluster', y='sell_price_main', data=df_clust)
# Add labels and title for clarity
plt.title('sell_price_main Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('sell_price_main')

# Show the plot
plt.show()






# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.set()

# # Scatter plot for mean_orders_14d vs. CN_total_products
# plt.figure(figsize=(10, 7))
# sns.scatterplot(data=df_clust, x="mean_orders_14d", y="CN_total_products", 
#                 hue="Kmean_cluster", s=50, edgecolor="white", alpha=0.5)

# # Title and legend
# plt.title("Scatter Plot of mean_orders_14d vs CN_total_products", fontsize=16)
# plt.legend(title="Kmean_cluster")

# # Show plot
# plt.show()



# import seaborn as sns
# sns.set()

# PairGrid = sns.PairGrid(df_clust, vars =df_clust.columns.tolist(), hue = 'Kmean_cluster', 
#                         diag_sharey = False, corner = True)
# PairGrid.map_lower(sns.scatterplot, s = 50, edgecolor = 'white', alpha = 0.5)
# PairGrid.map_diag(sns.histplot)
# PairGrid.add_legend()

# PairGrid.fig.suptitle('An overview of the comparison of clusters', fontsize = 56, ha = 'center', va = 'baseline')
# plt.show()


import warnings
# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

g = sns.FacetGrid(df_clust[df_clust.unique_id.isin(ids[:50])], 
                  col="unique_id", col_wrap=10, height=3, sharey=False)

g.map_dataframe(sns.lineplot, x="date", y="sales",  hue="Kmean_cluster", linewidth=1)
g.map_dataframe(
    sns.lineplot, 
    x="date", 
    y="sales", 
    marker='o', 
    markersize=2,
    linewidth=0.2,
    markeredgecolor='red',
)
# Set axis labels
g.set_axis_labels("Date", "Sales")

# Set the x-ticks to show only June 1st for each year
for ax in g.axes.flat:
    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=6))
    ax.tick_params(axis='x', rotation=90) 

g.tight_layout()
plt.show()


df_clust['date'] = pd.to_datetime(df_clust['date'])
max_datetime = df_clust['date'].max()
max_datetime


origin = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
origin_columns = origin.columns.tolist()
origin_columns


df = df_clust[origin_columns+['Kmean_cluster']]
df.head()


# 2. split 1 ngày cuối để test
val_set = df[df.date==max_datetime]
train_set = df[df.date!=max_datetime]

val_set.to_csv('Kmean_clustered_val.csv', index = False)
train_set.to_csv('Kmean_clustered_train.csv', index = False)


# def delete_ids(df):
#     ids = pd.DataFrame(df.unique_id.value_counts()).reset_index()
#     ids = ids[ids['count']>13].unique_id.tolist()
#     return ids


# # 2. Traindata
# # zero = train_set[train_set.Kmean_cluster==0]
# # one = train_set[train_set.Kmean_cluster==1]
# # two = train_set[train_set.Kmean_cluster==2]
# # three = train_set[train_set.Kmean_cluster==3]
# # four = train_set[train_set.Kmean_cluster==4]
# # five = train_set[train_set.Kmean_cluster==5]
# # six = train_set[train_set.Kmean_cluster==6]
# # seven = train_set[train_set.Kmean_cluster==7]
# # eight = train_set[train_set.Kmean_cluster==8]
# # nine = train_set[train_set.Kmean_cluster==9]
# # four


# 1. Knn

# Get the maximum datetime value

# train 10 model 

# evaluate


# lr = .1
# es = 10
# n_est = round(5000/lr)
# seed = 2
# base_params = {
#     'n_estimators':n_est
#     ,'learning_rate':lr
#     ,'verbosity':0
#     ,'enable_categorical':True
#     ,'early_stopping_rounds':es
#     ,'random_state':seed
#     ,'objective':'reg:squarederror'
#     ,'eval_metric':'rmse'
#     ,'device':'cuda'
#     ,'reg_lambda':0
#     ,'min_child_weight':1
# }
# kf_params = {
#     'n_splits':3
#     ,'n_repeats':1
#     ,'random_state':seed
# }

