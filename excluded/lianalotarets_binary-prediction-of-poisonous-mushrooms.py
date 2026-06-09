import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import io


df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')


df_test


df


df_test.isnull().sum() * 100 / df_test.shape[0]


df.isnull().sum() * 100 / df.shape[0]


def class_encoder(df, name_column):
    new_column = []
    for value in df[name_column]:
        if value == 'p':
            new_column.append(1) # poisonous
        else:
            new_column.append(-1) # not poisonous
            
    df[name_column] = new_column


class_encoder(df, 'class')


def no_unknown_yes_encoder(df, name_column: str):
    new_column = []
    for value in df[name_column]:
        if value == 'f':
            new_column.append(-1) # no
        elif value == 't':
            new_column.append(1) # yes
        else:
            new_column.append(0) # unknown
            
    df[name_column] = new_column


df['does-bruise-or-bleed'].value_counts(dropna=False)


no_unknown_yes_encoder(df, 'does-bruise-or-bleed')
no_unknown_yes_encoder(df_test, 'does-bruise-or-bleed')


df['has-ring'].value_counts(dropna=False)


no_unknown_yes_encoder(df, 'has-ring')
no_unknown_yes_encoder(df_test, 'has-ring')


df['season'].value_counts(dropna=False)


df_test['season'].value_counts(dropna=False)


def season_encoder(df, name_column: str):
    new_season_cos = []
    new_season_sin = []
    for value in df[name_column]:
        if value == 's':  # Autumn
            new_season_cos.append(1)
            new_season_sin.append(0)
        elif value == 'w':  # Winter
            new_season_cos.append(0)
            new_season_sin.append(-1)
        elif value == 'a':  # Spring
            new_season_cos.append(-1)
            new_season_sin.append(0)
        elif value == 'u':  # Summer
            new_season_cos.append(0)
            new_season_sin.append(1)

    df['season_cos'] = new_season_cos
    df['season_sin'] = new_season_sin


season_encoder(df, 'season')
season_encoder(df_test, 'season')


df = df.drop(columns=['season'])
df_test = df_test.drop(columns=['season'])


multicategorical_columns = ['cap-shape', 'cap-surface', 'cap-color', 
                            'gill-attachment', 'gill-spacing', 'gill-color', 
                            'stem-root', 'stem-surface', 'stem-color',
                            'veil-type', 'veil-color', 'ring-type', 
                            'spore-print-color', 'habitat']


def abc_clearing(df, name_column: str):
    abc_categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
                      'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 
                      's', 't', 'u', 'w', 'x', 'y', 'z'] 
    # there are no 'j', 'q', and 'v'

    df[name_column] = df[name_column].mask(~df[name_column].isin(abc_categories))
    df[name_column] = df[name_column].fillna(0)
    # return df


def abc_encoder(df, name_column):
    abc_categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
                      'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 
                      's', 't', 'u', 'w', 'x', 'y', 'z', 0] 
    # there are no 'j', 'q', and 'v'
    
    freq_poisonous = df[df['class'] == 1][name_column].value_counts(normalize=True)
    freq_edible = df[df['class'] == -1][name_column].value_counts(normalize=True)

    freq_poisonous_full = {category: 0 for category in abc_categories}
    freq_edible_full = {category: 0 for category in abc_categories}

    for category in abc_categories:
        if category != 0:
            if category in freq_poisonous.index:
                freq_poisonous_full[category] += freq_poisonous[category]
            if category in freq_edible.index:
                freq_edible_full[category] += freq_edible[category]

    return freq_poisonous_full, freq_edible_full


def abc_column_processing(df, df_test, name_column: str):
    abc_clearing(df, name_column)
    abc_clearing(df_test, name_column)
    
    freq_p, freq_e = abc_encoder(df, name_column)

    df['freq-' + name_column + '-p'] = df[name_column].map(freq_p)
    df['freq-' + name_column + '-e'] = df[name_column].map(freq_e)

    df_test['freq-' + name_column + '-p'] = df_test[name_column].map(freq_p)
    df_test['freq-' + name_column + '-e'] = df_test[name_column].map(freq_e)

    df = df.drop(columns=[name_column], inplace=True)
    df_test = df_test.drop(columns=[name_column], inplace=True)


for column in multicategorical_columns:
    abc_column_processing(df, df_test, column)


df.plot.box(
    column="cap-diameter",
    by="class",
)

plt.show()


cap_diameter_median = df['cap-diameter'].median()


df['cap-diameter'] = df['cap-diameter'].fillna(cap_diameter_median)
df_test['cap-diameter'] = df_test['cap-diameter'].fillna(cap_diameter_median)


df.plot.box(
    column="stem-height",
    by="class",
)

plt.show()


stem_height_median = df['stem-height'].median()


df_test['stem-height'] = df_test['stem-height'].fillna(stem_height_median)


correlation_matrix = df[df.columns[2:]].corrwith(df['class'])
print(correlation_matrix)


# Візуалізація кореляцій
sns.barplot(x=correlation_matrix.index, y=correlation_matrix.values, palette='coolwarm')

# Додавання титулу та підписів
plt.title('Correlation of each column with the class')
plt.ylabel('Correlation')
plt.xlabel('Columns')

# Показати графік
plt.xticks(rotation=90)
plt.show()


df_test


df


X = df[df.columns[2:]]
y = df['class'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=13
)


def model_classification_report(model, model_name: str, X_test, y_test):
    y_pred = model.predict(X_test)
    print(
        f"Classification report for the model {model_name}:\n",
        classification_report(y_test, y_pred),
    )


model_HGB_default = HistGradientBoostingClassifier()


model_HGB_default.fit(X_train, y_train)


model_classification_report(
    model_HGB_default, "Histogram-based Gradient Boosting Classification Tree", X_test, y_test
)


def model_size(model_path: str):
    size_in_bytes = os.path.getsize(model_path)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    print(
        f"Розмір моделі: {size_in_bytes} B ({size_in_kb:.2f} kB / {size_in_mb:.2f} MB)"
    )
    return size_in_bytes


# Створення буфера в пам'яті
buffer = io.BytesIO()

# Збереження моделі в буфер
joblib.dump(model_HGB_default, buffer)

# Оцінка розміру моделі в байтах
size_in_bytes = buffer.tell()
size_in_kb = size_in_bytes / 1024
size_in_mb = size_in_kb / 1024
print(f"Model size in memory: {size_in_bytes} B ({size_in_kb:.2f} kB / {size_in_mb:.2f} MB)")


y_pred_test = model_HGB_default.predict(df_test[df_test.columns[1:]])


classes_test = ['p' if cl == 1 else 'e' for cl in y_pred_test]


df_sample_submission['id'] = df_test['id']
df_sample_submission['class'] = classes_test


df_sample_submission


df_sample_submission.to_csv('submission.csv', index=False)

