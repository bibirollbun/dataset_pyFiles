from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore")


def get_categorical_features(df, verbose=False):

    from collections import defaultdict
    cat_features = []
    grouped = defaultdict(list)

    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique(dropna=False)

        if dtype == "object" or dtype.name in {"category", "string"}:
            cat_features.append(col)
            grouped[nunique].append(col)

    if verbose:
        print(f"ðŸŸ¡ Categorical Features: {len(cat_features)}\n")
        for uniq in sorted(grouped):
            print(f"â€¢ {uniq} unique values: {', '.join(grouped[uniq])}")
        print()

    return cat_features


def get_numerical_features(df, low_cardinality_threshold=10, verbose=False):
    num_features = list(df.select_dtypes(include=[np.number]).columns)

    if verbose:
        print(f"ðŸ”µ Numerical Features: {len(num_features)}\n")
        for col in num_features:
            nunique = df[col].nunique(dropna=False)
            note = f"({nunique} unique values)" if nunique <= low_cardinality_threshold else ""
            print(f"â€¢ {col} {note}")
        print()

    return num_features


def build_my_info_table(df):
    if df is None or df.empty:
        return None

    # Convert boolean columns to integers
    boolean_columns = df.select_dtypes(include='bool').columns
    df[boolean_columns] = df[boolean_columns].astype(int)

    numerical_features = get_numerical_features(df)
    categorical_features = get_categorical_features(df)

    metrics = []
    for col in df.columns:
        column_data = df[col]
        dtype   = column_data.dtypes
        count   = column_data.count()
        mean    = column_data.mean()   if col in numerical_features else ''
        std     = column_data.std()    if col in numerical_features else ''
        min_val = column_data.min()    if col in numerical_features else ''
        q25     = column_data.quantile(0.25) if col in numerical_features else ''
        median  = column_data.median() if col in numerical_features else ''
        q75     = column_data.quantile(0.75) if col in numerical_features else ''
        max_val = column_data.max()    if col in numerical_features else ''
        iqr     = max_val - min_val    if col in numerical_features else ''
        nunique = column_data.nunique()
        unique_values = column_data.unique() if col in categorical_features else ''
        mode    = column_data.mode().iloc[0] if not column_data.mode().empty else ''
        mode_count = column_data.value_counts().max() if not column_data.value_counts().empty else ''
        mode_percentage = (round(mode_count * 100 / len(column_data), 1) 
                           if mode_count not in ['', None] else '')
        null_count = column_data.isnull().sum()
        null_percentage = round(column_data.isnull().mean() * 100, 1)

        metrics.append({
            "column": col,
            "dtype": dtype,
            "count": count,
            "mean": round(mean, 1)   if mean    not in ['', None] else '',
            "std": round(std, 1)     if std     not in ['', None] else '',
            "min": round(min_val, 1) if min_val not in ['', None] else '',
            "25%": round(q25, 1)     if q25     not in ['', None] else '',
            "50%": round(median, 1)  if median  not in ['', None] else '',
            "75%": round(q75, 1)     if q75     not in ['', None] else '',
            "max": round(max_val, 1) if max_val not in ['', None] else '',
            "IQR": round(iqr, 1)     if iqr     not in ['', None] else '',
            "nunique": nunique,
            "unique": unique_values,
            "mode": mode,
            "mode #": mode_count,
            "mode %": mode_percentage,
            "null #": null_count,
            "null %": null_percentage,
        })

    df_info = pd.DataFrame(metrics)

    return df_info


def my_histplot(df, col, ax):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'Histogram Plot of {col}')
def my_kdeplot(df, col, ax):
    sns.kdeplot(df[col], ax=ax, fill=True)
    ax.set_title(f'KDE Plot of {col}')
def my_distplot(df, col, ax):
    sns.histplot(df[col], ax=ax, kde=True)
    ax.set_title(f'Distribution Plot of {col}')
def my_boxplot(df, col, ax):
    sns.boxplot(y=df[col], ax=ax)
def my_violinplot(df, col, ax):
    sns.violinplot(y=df[col], ax=ax)


def my_pie_chart(df, col, ax):
    labels = df[col].value_counts()
    ax.pie(labels, labels=labels.index, autopct='%1.1f%%')
    ax.set_title(f'Pie Chart of {col}')
def my_barplot(df, col, ax):
    value_counts = df[col].value_counts().sort_values(ascending=False)
    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, 
                orient='h', order=value_counts.index)
    ax.set_title(f'Bar Plot of {col}')
    ax.set_xlabel('Count')
    ax.set_ylabel(col)


def plot_features(df, plot_funcs, width_ratios, height_ratios, 
                  n_col=1, primary_cols=0, title=None):
    def plot_feature(cols):
        cols_len = len(cols) - primary_cols
        curr_width_ratios = width_ratios[:cols_len * len(plot_funcs)]
        n_charts = len(plot_funcs) * cols_len

        # Create a figure with specified size and gridspec layout
        fig = plt.figure(figsize=(sum(curr_width_ratios), max(height_ratios)))
        gs = fig.add_gridspec(1, n_charts, 
                              width_ratios=curr_width_ratios, height_ratios=height_ratios)
        axes = [0] * n_charts
        for i in range(cols_len):
            for j in range(len(plot_funcs)):
                k = i * len(plot_funcs) + j
                axes[k] = fig.add_subplot(gs[0, k])
                # Call the specified plotting function with df, col, and axis ax
                plot_funcs[j](df, cols[i + primary_cols], axes[k])
                if title:
                    fig.suptitle(title)

        plt.tight_layout()
        plt.show()

    for i in range(primary_cols, len(df.columns), n_col):
        plot_feature(list(df.columns[:primary_cols])+list(df.columns[i:i+n_col]))


def plot_numerical_features(df, plot_funcs=[my_boxplot, my_violinplot, my_distplot], 
                            width_ratios=[2, 2, 8], height_ratios=[4], 
                            n_col=1, primary_cols=0, title=None):
    plot_features(df, plot_funcs, width_ratios * n_col, height_ratios, n_col, primary_cols, title)

def plot_categorical_features(df, plot_funcs=[my_pie_chart, my_barplot], 
                              width_ratios=[4, 8], height_ratios=[4], 
                              n_col=1, primary_cols=0, title=None):
    plot_features(df, plot_funcs, width_ratios * n_col, height_ratios, n_col, primary_cols, title)


def bar_plot(df, categorical_cols, hue=None, size=(12, 6), title=None):
    for column in categorical_cols:
        if not title:
            title = f'Distribution of {column}'
            
        plt.figure(figsize=size)
        ax = sns.countplot(data=df, x=column, hue=hue)
        total = len(df)
        
        for p in ax.patches:
            height = p.get_height()
            percentage = f'{100 * height / total:.1f}%'
            ax.text(p.get_x() + p.get_width() / 2, height / 2, percentage, ha='center', va='center', fontsize=10, color='white')
            
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.title(title)
        plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

def plot_kde_multiclass(df, num_cols, target_col, class_labels=None):

    unique_classes = sorted(df[target_col].dropna().unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_classes))

    for col in num_cols:
        plt.figure(figsize=(10, 4))

        for cls, color in zip(unique_classes, palette):
            label = class_labels.get(cls, cls) if class_labels else cls
            sns.kdeplot(df[df[target_col] == cls][col], fill=True, common_norm=False, label=label, color=color)

        plt.legend(loc='upper right')
        plt.title(f'KDE Plot of {col} by {target_col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()



def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv").drop(columns=['id'])
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv").drop(columns=['id'])


train.info()


train.head()


info_table = build_my_info_table(train)
info_table


info_table[(info_table['dtype'] == 'object')]


info_table[(info_table['dtype'] == 'int64')]


cat_cols = get_categorical_features(train, verbose=True)
print('--------------------------------------------------')
num_cols = get_numerical_features(train, verbose=True)


train['Fertilizer Name'].value_counts()/len(train) * 100


plot_numerical_features(train[num_cols])


plot_categorical_features(train[cat_cols])


plot_kde_multiclass(train, num_cols, 'Fertilizer Name')


bar_plot(train, cat_cols[:-1], 'Fertilizer Name')


oe = OrdinalEncoder()
train[cat_cols[:-1]] = oe.fit_transform(train[cat_cols[:-1]])
test[cat_cols[:-1]] = oe.transform(test[cat_cols[:-1]])


le = LabelEncoder()
train[cat_cols[-1]] = le.fit_transform(train[cat_cols[-1]])


y = train['Fertilizer Name']
X = train.drop(columns=['Fertilizer Name'])


train_X, test_X, train_y, test_y = train_test_split(X, y,test_size = 0.2, random_state =42,stratify=y)


model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(train_y)),
    n_estimators=1000,
    learning_rate=0.045,         
    max_depth=7,                
    colsample_bytree=0.6,       
    colsample_bylevel=0.8,      
    subsample=0.8,
)


model.fit(train_X, train_y)


y_pred_probs = model.predict_proba(test_X)
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]  
actual = [[label] for label in test_y]


map3_score = mapk(actual, top_3_preds)
print(f"âœ… MAP@3 Score: {map3_score:.5f}")


sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


preds = model.predict_proba(test)
top_3_labels = np.argsort(preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_labels.ravel()).reshape(top_3_labels.shape)


sub['Fertilizer Name'] = [' '.join(row) for row in top_3_labels]
sub.head()


sub.to_csv('submission.csv', index=False)




