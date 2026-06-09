import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fgr
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
train_df.head()


train_df.shape


test_df.shape


print("Null values in dataset:")
print(train_df.isnull().sum())


print("\nDuplicate entries in dataset:")
print(train_df.duplicated().sum())


numerical_cols = train_df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = train_df.select_dtypes(exclude=['number']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title("Frequency of " + col_name)
        plt.xticks(rotation=90)

        plt.subplot(1, 2, 2)
        values = dataframe[col_name].value_counts()
        plt.pie(x=values, labels=values.index, autopct=lambda p: '{:.2f}% ({:.0f})'.format(p, p/100 * sum(values)))
        plt.title("Frequency of " + col_name)
        plt.legend(labels=['{} - {:.2f}%'.format(index, value/sum(values)*100) for index, value in zip(values.index, values)],
                   loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)
        plt.show(block=True)

for col in categorical_cols:
    cat_summary(train_df, col, True)


def my_histplot(df, col, ax):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Histogram Plot of {col}')
    
    
def my_vsplot(df, normal_col, label_col):
    plt.figure(figsize=(10, 6), dpi=80)
    plt.bar(list(dict(df[normal_col].value_counts()).keys()), dict(df[normal_col].value_counts()).values(), color='r')
    plt.bar(list(dict(df[normal_col][df[label_col] == 1].value_counts()).keys()), dict(df[normal_col][df[label_col] == 1].value_counts()).values(), color='b')

    plt.xlabel(normal_col)
    plt.ylabel('Count')
    plt.legend(['All', label_col])
    # plt.title('The number of requests from different protocols')
    
def plot_charts_grid_single_feature(df, plot_func, size=(12, 4), n_col=1):
    if len(df.columns) == 0:
        return
    n_rows = (len(df.columns) + n_col-1) // n_col
    fig, axes = plt.subplots(n_rows, n_col, figsize=(size[0]*n_col, size[1]*n_rows))
    if len(df.columns) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, label in enumerate(df.columns):
        plot_func(df, label, axes[i])
        axes[i].set_xlabel(label)

    for j in range(i+1, n_rows*n_col):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


plot_charts_grid_single_feature(train_df[numerical_cols], my_histplot)


for col in numerical_cols:
    unique_values = sorted(train_df[col].unique())
    print(f"Unique values in {col}: {unique_values}")



for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    
    # Create a count plot
    ax = sns.histplot(train_df[col], discrete=True, kde=False)
    
    # Calculate percentages
    total = len(train_df[col])
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height} ({height/total:.1%})', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=10, color='black', rotation=0)
    
    plt.title(f'Count Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--')
    plt.show()


categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
train_encoded = train_df.copy()

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_encoded[col] = le.fit_transform(train_encoded[col])
    le_dict[col] = le


plt.figure(figsize=(10, 8))
sns.heatmap(train_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


sns.pairplot(train_encoded[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer Name']], hue='Fertilizer Name')
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()


train_encoded['Fertilizer Name Original'] = le.inverse_transform(train_encoded['Fertilizer Name'])
target_col = 'Fertilizer Name Original'
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.boxplot(x=target_col, y=col, data=train_encoded, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col} by Fertilizer Type')
    axes[i].set_xlabel('Fertilizer Name')
    axes[i].set_ylabel(col)
    axes[i].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


cross_tab_soil = pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name'])
cross_tab_soil.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title("Soil Type vs Fertilizer Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

cross_tab_crop = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])
cross_tab_crop.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title("Crop Type vs Fertilizer Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


train_encoded['Temperature Binned'] = pd.cut(train_encoded['Temparature'], bins=5)
train_encoded['Humidity Binned'] = pd.cut(train_encoded['Humidity'], bins=10)
train_encoded['Moisture Binned'] = pd.cut(train_encoded['Moisture'], bins=11)
train_encoded['Nitrogen Binned'] = pd.cut(train_encoded['Nitrogen'], bins=11)
train_encoded['Phosphorous Binned'] = pd.cut(train_encoded['Phosphorous'], bins=3)
train_encoded[['Temparature', 'Temperature Binned', 
                     'Humidity', 'Humidity Binned', 
                     'Moisture', 'Moisture Binned', 
                     'Nitrogen', 'Nitrogen Binned', 
                     'Phosphorous', 'Phosphorous Binned']].head()


binned_cols = [
    ('Temperature Binned', 'Temperature (Binned)'),
    ('Humidity Binned', 'Humidity (Binned)'),
    ('Moisture Binned', 'Moisture (Binned)'),
    ('Nitrogen Binned', 'Nitrogen (Binned)'),
    ('Phosphorous Binned', 'Phosphorous (Binned)')
]
for binned_col, title in binned_cols:
    cross_tab = pd.crosstab(train_encoded[binned_col], train_encoded['Fertilizer Name'])
    
    ax = cross_tab.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
    
    plt.title(f'{title} vs Fertilizer Distribution')
    plt.ylabel('Count')
    plt.xlabel(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


binned_cols = [
    'Temperature Binned',
    'Humidity Binned',
    'Moisture Binned',
    'Nitrogen Binned',
    'Phosphorous Binned'
]
# Call the cat_summary function for each binned column
for col in binned_cols:
    cat_summary(train_encoded, col, True)


train_encoded.info()


columns_to_drop = [
    'Fertilizer Name Original', 
    'Temperature Binned', 
    'Humidity Binned', 
    'Moisture Binned', 
    'Nitrogen Binned', 
    'Phosphorous Binned'
]
train_encoded_cleaned = train_encoded.drop(columns=columns_to_drop)


train_encoded_cleaned.head()


X = train_encoded_cleaned.drop(columns=['Fertilizer Name'])
y = train_encoded_cleaned['Fertilizer Name']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


lgbm_params = {
    'num_leaves': [249], 
    'max_depth': [10],
}
skf = StratifiedKFold(n_splits=3)
lgbm = GridSearchCV(
    estimator=LGBMClassifier(random_state=42, verbosity=-1),
    param_grid=lgbm_params,
    cv=skf,
    n_jobs=3,
    verbose=-1
)
lgbm_model = lgbm.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
accuracy = accuracy_score(y_test, lgbm_pred)
precision = precision_score(y_test, lgbm_pred, average='weighted')
recall = recall_score(y_test, lgbm_pred, average='weighted')
f1 = f1_score(y_test, lgbm_pred, average='weighted')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print("\nClassification Report:")
print(classification_report(y_test, lgbm_pred))

