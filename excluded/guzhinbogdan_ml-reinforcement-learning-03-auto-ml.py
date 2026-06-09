# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside the current session


import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import Trials, fmin, hp, space_eval, tpe
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ydata_profiling import ProfileReport


# Загрузка данных
train_df = pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")
print(train_df.shape)
train_df.head()


# Проведём первичный анализ признаков с помощью профилирования
profile = ProfileReport(
    train_df.drop(columns=["NObeyesdad"]),
    title="Обзор данных о риске ожирения",
    explorative=True,
)
profile


# Удалим признаки, которые могут вносить искажения или не являются информативными
cols_to_drop = ["id", "Gender", "family_history_with_overweight"]
data = train_df.drop(columns=cols_to_drop)
print(data.shape)
data.head()


# Повторное профилирование для уточнённого набора признаков
profile_filtered = ProfileReport(
    data.drop(columns=["NObeyesdad"]),
    title="Обзор после отбора признаков",
    explorative=True,
)
profile_filtered


# Разделим данные на признаки и целевую переменную
X = data.drop(columns=["NObeyesdad"])
y = data["NObeyesdad"]


# Выделим числовые и категориальные признаки
cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_features = X.select_dtypes(include=["int", "float"]).columns.tolist()
cat_features, num_features


# Создадим пайплайны предобработки для числовых и категориальных признаков
num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
cat_pipeline = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ]
)


# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Базовая модель для сравнения
baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LGBMClassifier(random_state=42, num_class=len(np.unique(y)), verbose=-1),
        ),
    ]
)

baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)


# Оценка качества базовой модели
accuracy_score(y_test, y_pred_baseline), classification_report(
    y_test, y_pred_baseline, output_dict=True
)


# Средняя точность по кросс-валидации для базовой модели
cross_val_score(baseline_model, X, y, cv=3, scoring="accuracy").mean()


# Пространство поиска гиперпараметров для Hyperopt
param_space = {
    "n_estimators": hp.choice("n_estimators", list(range(50, 600, 25))),
    "max_depth": hp.choice("max_depth", [-1] + list(range(3, 50, 3))),
    "num_leaves": hp.choice("num_leaves", list(range(2, 60, 4))),
    "subsample": hp.uniform("subsample", 0.1, 1.0),
}


def objective(params):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    **params, random_state=42, num_class=len(np.unique(y)), verbose=-1
                ),
            ),
        ]
    )
    accuracy = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
    return -accuracy


# Запуск оптимизации
trials = Trials()
best = fmin(
    fn=objective,
    space=param_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials,
    show_progressbar=True,
    catch_eval_exceptions=True,
)


# Лучшая найденная комбинация гиперпараметров
space_eval(param_space, best)


# Сбор результатов в DataFrame для визуализации
results = pd.DataFrame(
    [
        {
            **{
                param: trial["misc"]["vals"][param][0]
                for param in trial["misc"]["vals"]
            },
            "loss": trial["result"]["loss"],
        }
        for trial in trials.trials
    ]
)
results = results[["max_depth", "n_estimators", "num_leaves", "subsample", "loss"]]
results.head()


warnings.filterwarnings("ignore", category=FutureWarning)


sns.lineplot(data=results, x="max_depth", y="loss")


sns.lineplot(data=results, x="n_estimators", y="loss")


sns.lineplot(data=results, x="num_leaves", y="loss")


sns.lineplot(data=results, x="subsample", y="loss")

