import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_sex_distribution(df_train, df_test):
    # Create 1 row and 2 columns of subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for TRAIN dataset
    axes[0].pie(df_train["Sex"].value_counts(), labels=["Male", "Female"], autopct='%1.1f%%')
    axes[0].set_title("Sex in TRAIN dataset")

    # Plot for TEST dataset
    axes[1].pie(df_test["Sex"].value_counts(), labels=["Male", "Female"], autopct='%1.1f%%')
    axes[1].set_title("Sex in TEST dataset")

    # Optional: improve layout
    plt.tight_layout()
    plt.show()


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


df_train.info()


# check NAs 

train_nas = df_train.isnull().sum()
test_nas = df_test.isnull().sum()

print(f"Traininig data Nulls {train_nas} test data Nulls {test_nas}")


plot_sex_distribution(df_train, df_test)


sns.pairplot(df_train[['Age', 'Height', 'Heart_Rate', 'Duration', 'Calories']].sample(500), corner = True)
plt.suptitle('Feature Interactions', y=1.02)
plt.tight_layout()
plt.show()


numerical_cols = df_train.select_dtypes(include='number').columns.difference(['id'])
num_cols_grid = 3
num_rows = (len(numerical_cols) + num_cols_grid-1)//num_cols_grid

fig, axes = plt.subplots(num_rows, num_cols_grid, figsize=(15, 5 * num_rows), constrained_layout=False)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(df_train[col], ax=axes[i], color = 'blue', fill = True)
    if col in df_test.columns:
        sns.kdeplot(df_test[col], ax=axes[i], color = 'red', fill = True) 
        axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.3, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=df_train[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()




df_train


df_train_male = df_train[df_train['Sex'] == 'male']
df_train_female = df_train[df_train['Sex'] == 'female']


numeric_cols = df_train_male.select_dtypes(include='number').columns.difference(['id'])

# Create a grid of subplots
n_cols = 2
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
plt.figure(figsize=(n_cols * 6, n_rows * 4))

for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.kdeplot(df_train_male[col], label='Male', fill=True, alpha=0.5)
    sns.kdeplot(df_train_female[col], label='Female', fill=True, alpha=0.5)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()


corr = df_train[numerical_cols].corr(method = 'spearman')
sns.heatmap(data = corr, cmap = 'vlag')


df_train["Height"] = df_train["Height"]/100 
df_train["BMI"] = df_train["Weight"]/(df_train["Height"]**2)
#df_train['CLI'] = df_train['Heart_Rate'] * df_train['Duration']
#df_train['EIS'] = (df_train['Heart_Rate'] * df_train['Duration']) / df_train['Body_Temp']
df_train['Calories'] = np.log1p(df_train['Calories'])


df_test["Height"] = df_test["Height"]/100 
df_test["BMI"] = df_test["Weight"]/(df_test["Height"]**2)
#df_test['CLI'] = df_test['Heart_Rate'] * df_test['Duration'] #Cardiac Load index
#df_test['EIS'] = (df_test['Heart_Rate'] * df_test['Duration']) / df_test['Body_Temp'] #Excercise intensity score



numerical_cols = df_train.select_dtypes(include='number').columns.difference(['id'])
corr = df_train[numerical_cols].corr(method = 'spearman')
sns.heatmap(data = corr, cmap = 'vlag', annot=True)
plt.title('Feature Spearman Correlation Heatmap')
plt.tight_layout()
plt.show()


numeric_cols = df_train.select_dtypes(include='number').columns.difference(['id'])

# Plot boxplots
n_cols = 2
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
plt.figure(figsize=(n_cols * 6, n_rows * 4))

for i, col in enumerate(numeric_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='Sex', y=col, data=df_train, palette='Set2')
    plt.title(f'Boxplot of {col} by Sex')

plt.tight_layout()
plt.show()


y_train = df_train['Calories']
X_train = df_train.drop(columns = ['id', 'Calories'])



df_test_1 = df_test.drop(columns = ['id'])


X_train_encoded = pd.get_dummies(X_train, columns = ['Sex'], dtype = int)
df_test_encoded = pd.get_dummies(df_test_1, columns = ['Sex'], dtype=int)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)   
X_test_scaled = scaler.transform(df_test_encoded)



# Get predictions

from catboost import CatBoostRegressor

ens_preds = []
model = CatBoostRegressor(iterations = 1000, learning_rate = 0.3, loss_function='RMSEWithUncertainty', 
                          posterior_sampling = True, 
                             verbose=True, random_seed = 42)
model.fit(X_train_scaled, y_train)
ens_preds = model.virtual_ensembles_predict(X_test_scaled, prediction_type = 'TotalUncertainty', 
                                               virtual_ensembles_count = 10)


kno_uncertainty = ens_preds[:,1]
mean_preds = ens_preds[:,0]
y_pred = np.expm1(mean_preds)



output_df = pd.DataFrame({
    'id': df_test.id,
    'Calories': y_pred
})


output_df.to_csv('predictions7_virtual_ensemble.csv', index=False)

