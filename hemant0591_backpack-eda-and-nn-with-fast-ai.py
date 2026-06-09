# Installing fastai's fastbook library

! pip install -Uqq fastbook
import fastbook


from fastbook import *
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import warnings
warnings.filterwarnings('ignore')


path = Path('/kaggle/input/playground-series-s5e2')
Path.BASE_PATH = path


path.ls()


df_train = pd.read_csv(path/'train.csv')
df_extra = pd.read_csv(path/'training_extra.csv')
df_test = pd.read_csv(path/'test.csv')


df = pd.concat([df_train,df_extra],axis=0).reset_index(drop=True)


df.head()


df.isna().sum()


df.info()


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


# Categorical and numerical columns
categorical_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
numerical_cols = ['Weight Capacity (kg)', 'Price']


# 1. UNIVARIATE ANALYSIS

# Categorical Variables - Count Plots
fig, axes = plt.subplots(nrows=len(categorical_cols)//2 + len(categorical_cols)%2, ncols=2, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(x=df[col], data=df, ax=axes[i], order=df[col].value_counts().index, palette='viridis')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Numerical Variables - Histograms & Box Plots
fig, axes = plt.subplots(nrows=2, ncols=len(numerical_cols), figsize=(12, 8))

# Histograms
for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[0, i], color='blue')
    axes[0, i].set_title(f'Histogram of {col}')

# Box Plots (Check for Outliers)
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[1, i], color='red')
    axes[1, i].set_title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()


# 2. BIVARIATE ANALYSIS

# Price Distribution Across Categorical Features - Box Plots
fig, axes = plt.subplots(nrows=len(categorical_cols)//2 + len(categorical_cols)%2, ncols=2, figsize=(16, 20))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.boxplot(x=df[col], y=df['Price'], data=df, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Price Distribution by {col}')
    axes[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Violin Plots (Show Price Density)
fig, axes = plt.subplots(nrows=len(categorical_cols)//2 + len(categorical_cols)%2, ncols=2, figsize=(16, 20))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.violinplot(x=df[col], y=df['Price'], data=df, ax=axes[i], palette='coolwarm', inner='quartile')
    axes[i].set_title(f'Price Distribution by {col}')
    axes[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Strip Plot
fig, axes = plt.subplots(nrows=len(categorical_cols)//2 + len(categorical_cols)%2, ncols=2, figsize=(16, 20))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.stripplot(x=df[col], y=df['Price'], data=df, ax=axes[i], palette='Set2', jitter=True, size=3, alpha=0.5)
    axes[i].set_title(f'Price Distribution by {col}')
    axes[i].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()



# 3. RELATIONSHIP BETWEEN NUMERICAL FEATURES

# Scatter Plot: Weight Capacity vs. Price
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Weight Capacity (kg)'], y=df['Price'], color='blue')
plt.title("Scatter Plot: Weight Capacity vs. Price")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[['Weight Capacity (kg)', 'Price']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot to visualize numeric relationships
sns.pairplot(df.drop(columns=['id']).sample(50000), hue="Material", diag_kind="kde", height=2)
plt.show()


df['Price'].min(), df['Price'].max()


skewness = df[numerical_cols].skew()

# Display skewness values
print("Skewness of numerical features:")
print(skewness)


print(df.Size.unique())
print(df.Compartments.unique())


df.Size.dtype, df.Compartments.dtype


sizes = 'Large', 'Medium', 'Small'
df['Size'] = df['Size'].astype('category')
df['Size'] = df['Size'].cat.set_categories(sizes, ordered=True)

compartments = 10,9,8,7,6,5,4,3,2,1
df['Compartments'] = df['Compartments'].astype('category')
df['Compartments'] = df['Compartments'].cat.set_categories(compartments, ordered=True)


df_test['Size'] = df_test['Size'].astype('category')
df_test['Size'] = df_test['Size'].cat.set_categories(sizes, ordered=True)

df_test['Compartments'] = df_test['Compartments'].astype('category')
df_test['Compartments'] = df_test['Compartments'].cat.set_categories(compartments, ordered=True)


df.Size.dtype, df.Compartments.dtype


df.Size.cat.codes.head(10)


train_idx, valid_idx = train_test_split(df.index, test_size=0.2, random_state=42)


train_idx, valid_idx


numerical_cols = ['Weight Capacity (kg)']
categorical_cols, numerical_cols


procs = [Categorify, FillMissing, Normalize]
splits = (list(train_idx),list(valid_idx))
to = TabularPandas(df.drop(columns=['id']), procs, categorical_cols, numerical_cols, y_names='Price', splits=splits)


len(to.train), len(to.valid)


to.show(3)


to.items.head(3)


to.classes['Size']


xs,y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y


def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


model = rf(xs, y);


error = mean_squared_error(model.predict(valid_xs), valid_y)
error, np.sqrt(error)


oob_error = mean_squared_error(model.oob_prediction_, y)
oob_error


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


fi = rf_feat_importance(model, xs)
fi


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:10]);


to_keep = fi[fi.imp>0.05].cols
len(to_keep), to_keep


xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]


xs_imp.head()


model = rf(xs_imp, y)


error = mean_squared_error(model.predict(valid_xs_imp), valid_y)
error, np.sqrt(error)


df_test.head(3)


to_test = TabularPandas(df_test.drop(columns=['id']), procs, categorical_cols, numerical_cols)


to_test.show(3)


test_xs = to_test.train.xs[to_keep]
test_xs.head()


preds = model.predict(test_xs)


sub = pd.read_csv(path/'sample_submission.csv')
sub.head()


sub['Price'] = preds
sub.head()


sub.to_csv('RF_Preds_Base.csv', index=False)


dls = to.dataloaders(1024)


dls.show_batch()


y = to.train.y
y.min(), y.max()


learn = tabular_learner(dls, y_range=(14,155), layers=[500,250],
                        n_out=1, loss_func=F.mse_loss)


learn.lr_find()


learn.fit_one_cycle(10, 1e-2)


test_dl = learn.dls.test_dl(df_test.drop(columns=['id']))


test_dl.show_batch()


preds,_ = learn.get_preds(dl=test_dl)


preds = preds.squeeze(dim=1).numpy()


sub['Price'] = preds


sub.head(5)


sub.to_csv('NN_Basic.csv', index=False)


def ensemble():
    learn = tabular_learner(dls, y_range=(15,150), layers=[500,250],
                        n_out=1, loss_func=F.mse_loss)
    with learn.no_logging(): learn.fit(10, lr=1e-2)
    return learn.get_preds(dl=test_dl)[0]


learns = [ensemble() for _ in range(5)]


preds = torch.stack(learns).mean(0)


sub['Price'] = preds


sub.head(5)


sub.to_csv('NN_Ensemble.csv', index=False)

