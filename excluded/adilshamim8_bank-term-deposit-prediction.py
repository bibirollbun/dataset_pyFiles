import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col='id')


orig = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')
orig['y'] = orig['y'].map({'no': 0, 'yes': 1})


train = pd.concat([train, orig], ignore_index=True)
train = train.drop_duplicates()


train.head()


train.info()


def custom_describe(df):
    df_ = df.select_dtypes(include=np.number)
    des = df_.describe().T
    des['skewness'] = df_.skew()
    des['kurtosis'] = df_.kurtosis()
    des['count'] = des['count'].astype('int')
    return des


features = test.columns.tolist()
print(features)


numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']
target = 'y'


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x=target, data=train, ax=axes[0])
axes[0].set_title('Distribution of Target Variable (Subscribed)', fontweight='bold', size=20)
axes[0].set_xticks(ticks=[0, 1],labels=['No', 'Yes'])
axes[0].set_xlabel("subscribed")

train[target].value_counts().plot(kind='pie', ax=axes[1], explode=(0.0, 0.1), autopct="%.2f%%", labels=['No', 'Yes'], pctdistance=0.75)
axes[1].add_artist(plt.Circle((0, 0), 0.5, fc='w'))
axes[1].set_title('Pie Chart of Target Variable', fontweight='bold', size=20)
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()


custom_describe(train[numerical_features])


def numerical_features_plot(df, feature, target):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Analysis of {feature}', fontsize=16, fontweight='bold')

    # Boxplot
    sns.boxplot(data=df, x=feature, y=target, hue=target, orient='h', ax=axes[0])
    axes[0].set_title(f'Boxplot of {feature}')
    axes[0].legend_.remove()  # Turn off legend

    # Violinplot
    sns.violinplot(data=df, x=feature, y=target, hue=target, orient='h', ax=axes[1])
    axes[1].set_title(f'Violinplot of {feature}')
    axes[1].legend_.remove()  # Turn off legend

    # Histogram with KDE
    sns.histplot(data=df, x=feature, hue=target, kde=True, ax=axes[2], alpha=0.6)
    axes[2].set_title(f'Distribution of {feature}')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


for feature in numerical_features:
    numerical_features_plot(train, feature, target)


def categorical_features_plot(df, feature):
    value_counts = df[feature].value_counts()
    
    top_n = min(10, len(value_counts))
    top_categories = value_counts.nlargest(top_n)
    
    df_plot = df[df[feature].isin(top_categories.index)]

    top_percentages = (top_categories / len(df)) * 100

    plt.figure(figsize=(25, 6))

    plt.subplot(1, 2, 1)
    sns.countplot(df_plot, x=feature, palette=sns.color_palette('viridis'))
    plt.title(f"Count Plot of{(' Top ' + str(top_n)) if len(value_counts) > 10 else ''} Categories of {feature}", size=16, fontweight='bold')

    plt.subplot(1, 2, 2)
    plt.pie(
        top_percentages,
        labels=top_percentages.index,
        autopct=lambda pct: f"{pct:.2f}%",
        pctdistance=0.75
    )
    plt.gca().add_artist(plt.Circle((0, 0), 0.5, fc='w'))  # Donut hole
    plt.title(
        f"{('Top ' + str(top_n)) if len(value_counts) > 10 else ''} {feature} Categories as % of Full Dataset",
        size=16,
        fontweight='bold'
    )
    plt.ylabel("")

    plt.tight_layout()
    plt.show()


for feature in categorical_features:
    if feature == 'day':
        continue
    categorical_features_plot(train, feature)


df = train.copy()

g = sns.FacetGrid(df[df['job'].isin(df['job'].value_counts().head(6).index)], 
                  col='job', col_wrap=3, height=4, aspect=1.2)
g.map_dataframe(sns.boxplot, x='y', y='age', palette='viridis')
g.set_titles("{col_name}")
g.fig.suptitle('Age Distribution by Job and Subscription Status', y=1.05)
plt.show()


def plot_categorical_heatmap(feature1, feature2):
    cross_tab = pd.crosstab(df[feature1], df[feature2], normalize='index') * 100
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=.5)
    plt.title(f'Percentage of Subscription by {feature1} and {feature2}')
    plt.ylabel(feature1)
    plt.xlabel(feature2)
    plt.show()


plot_categorical_heatmap('job', 'education')


plot_categorical_heatmap('marital', 'education')


plot_categorical_heatmap('poutcome', 'contact')


from sklearn.model_selection import train_test_split,StratifiedKFold

X = train.drop('y', axis=1).astype('str')
y = train['y']

test_str = test.astype('str')


from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

cat_clf = CatBoostClassifier(
    allow_writing_files=False,
    verbose=False,
    cat_features=X.columns.to_list(),
    task_type='GPU',
    n_estimators=10000,
    learning_rate=0.05,
)

N_SPLITS = 10
skfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
test_pred = np.zeros(len(test_str))
roc_scores = []

for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = clone(cat_clf)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=300, verbose=1500)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_pred)
    roc_scores.append(roc_score)

    test_pred += model.predict_proba(test_str)[:, 1]
    print(f"Fold {fold} -> ROC-AUC: {roc_score:.5f}")

print(f"Average Fold ROC-AUC: {np.mean(roc_scores):.5f} \xb1 {np.std(roc_scores):.5f}")

test_pred = test_pred / N_SPLITS


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub['y'] = test_pred
sub.to_csv("submission.csv", index=False)
sub.head()

