# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from bokeh.io import output_notebook
import shap

shap.initjs();

output_notebook();

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv");
print(len(df));
df.head()


# from https://www.kaggle.com/competitions/playground-series-s5e3/discussion/565634
df['expected_day'] = (df['id']) % 365 + 1;
df = df.drop(columns=["id", "day"]);
print((df['expected_day'] == 365).sum())
df.head()


X = df.drop(columns=["rainfall"])
y = df["rainfall"]


from catboost import CatBoostClassifier, CatBoostRegressor;
from xgboost import XGBClassifier;
from lightgbm import LGBMClassifier;
from sklearn.linear_model import LogisticRegression;
from sklearn.ensemble import StackingClassifier;
from sklearn.pipeline import make_pipeline;
from sklearn.preprocessing import MinMaxScaler;

catboost = CatBoostClassifier(
    iterations=100,
    learning_rate=0.03,
    eval_metric="AUC",
    l2_leaf_reg=20,
    depth=7,
    loss_function="Logloss",  
    random_seed=42,
    verbose=0,
    has_time=True
);

catboost_regressor = CatBoostRegressor(
    iterations=100,
    learning_rate=0.03,
    eval_metric="AUC",
    l2_leaf_reg=20,
    depth=7,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
);

xgboost = XGBClassifier(
    n_estimators=50,
    max_depth=7, 
    learning_rate=0.02, 
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42
);

lightgbm = LGBMClassifier(
    n_estimators=50,
    metric="AUC",
    max_depth=7, 
    learning_rate=0.03, 
    random_state=42,
    verbose=-1,
    objective="binary"
);

lr = LogisticRegression(
    solver='liblinear',
    penalty='l1',
    max_iter=2000, 
    random_state=42,
    C=1.0
);

stacking_model = StackingClassifier(
    estimators=[
        ('cat', catboost),
        ('xgb', xgboost),
        ('lgbm', lightgbm)
    ],
    final_estimator=catboost,
    stack_method="predict_proba",
    cv=3
);

model = make_pipeline(
    MinMaxScaler(),
    catboost
);

lightgbm_model = make_pipeline(
    MinMaxScaler(),
    lightgbm
);


import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import gridplot;

def display_SHAP_importances(X, y, model):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);
    
    model.fit(X_train, y_train);
    
    explainer = shap.Explainer(model);
    shap_values = explainer(X_test);
    
    feature_importance = np.abs(shap_values.values).mean(axis=0);
    feature_names = X_test.columns;

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance});
    importance_df = importance_df.sort_values('importance', ascending=True);
    sorted_features = importance_df['feature'].tolist();
    source = ColumnDataSource(data={'feature': feature_names, 'importance': feature_importance});
    
    p = figure(
        y_range=sorted_features,
        x_axis_label="SHAP Feature Importance",
        title="SHAP Feature Importance Analysis",
        height=600, width=800
    );
    
    p.hbar(y="feature", right="importance", source=source, height=0.5, color="blue");
    
    hover = HoverTool();
    hover.tooltips = [("Feature", "@feature"), ("SHAP Feature Importance", "@importance")];
    p.add_tools(hover);
    
    show(p);

    sorted_features.reverse();
    
    return sorted_features;


sorted_features = display_SHAP_importances(X, y, catboost);
sorted_features


explainer = shap.Explainer(catboost);
shap_values = explainer(X);


# First we grab those samples
y_preds = catboost.predict_proba(X)[:,1];
y_pred_labels = np.where(y_preds < 0.5, 0, 1);
temp = (y_pred_labels - y) != 0
indices = temp[temp == True].index.to_numpy()

features_count = len(list(X))

index = indices[6];

print(f"SHAP waterfall diagram of index {index} in incorrect samples: ")
print(f"Ground truth label of sample index {index}: {y[index]}, predicted label: {y_pred_labels[index]}")
print(f"Sample {index}:")
print(df.loc[index])
shap.plots.waterfall(shap_values[index], max_display=features_count);


shap.dependence_plot("sunshine", shap_values.values, X)


from sklearn.metrics import classification_report, confusion_matrix;
import matplotlib.pyplot as plt;

_X = X.copy();
_y = y.copy();

# Maybe square sunshine?
#_X["sunshine"] = X["sunshine"] ** 2;
X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, random_state=42);
catboost.fit(X_train, y_train);

y_test_preds = catboost.predict_proba(X_test)[:,1];
y_test_preds_labels = np.where(y_test_preds < 0.5, 0, 1);

print(confusion_matrix(y_test, y_test_preds_labels, labels=[1,0]))

explainer = shap.Explainer(catboost);
shap_values = explainer(_X);


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
shap.dependence_plot('cloud', shap_values.values, _X, ax=axes[0, 0], show=False)
shap.dependence_plot('sunshine', shap_values.values, _X, ax=axes[0, 1], show=False)
shap.dependence_plot('humidity', shap_values.values, _X, ax=axes[1, 0], show=False)
shap.dependence_plot('windspeed', shap_values.values, _X, ax=axes[1, 1], show=False)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

def get_sorted_correlations(df):
    corr_matrix = df.corr()
    
    corr_list = []
    
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2:
                corr_value = corr_matrix.loc[col1, col2]
                corr_list.append((col1, col2, corr_value))

    corr_list = sorted(corr_list, key=lambda x: abs(x[2]), reverse=True)

    print("Feature pairs sorted by correlation (descending order):")
    for col1, col2, corr in corr_list:
        print(f"{col1} - {col2}: {corr:.4f}")

    return corr_list

#sorted_correlations = get_sorted_correlations(df);

def visualize_correlation_matrix(df, title="Feature correlation heatmap", figsize=(10, 8), cmap="RdBu_r"):
    corr_matrix = df.corr()
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        annot=True,  
        fmt=".2f",   
        cmap=cmap,   
        vmin=-1,    
        vmax=1,      
        center=0,
        square=True, 
        linewidths=.5, 
        mask=mask,  
        cbar_kws={"shrink": .8}
    )
    plt.title(title, fontsize=16)
    plt.tight_layout()
    return plt


plt_matrix = visualize_correlation_matrix(df);
plt_matrix.show();


def check_correlation(df, x_col, y_col):
    source = ColumnDataSource(df)

    p = figure(title=f"Scatter Plot: {x_col} vs {y_col}", 
               x_axis_label=x_col, 
               y_axis_label=y_col, 
               width=1000, height=500)

    p.scatter(x=x_col, y=y_col, source=source, size=8, color="blue", alpha=0.6, legend_label="Data Points")

    return p;

plots = [];

for feature in list(df):
    plots.append(check_correlation(df, feature, "rainfall"));

grids = gridplot(
    [plots[i:i+2] for i in range(0,len(plots),2)],
    width=500,
    height=250,
    toolbar_location='right'
);

show(grids);


print(df["sunshine"].max());
print(df["sunshine"].min());

threshold = 0.5;

sun_series = (df["sunshine"] <= threshold).astype(int);
rainfall_series = (df["rainfall"] == 1).astype(int);
sun_df = pd.DataFrame({"sunshine": sun_series, "rainfall": rainfall_series});

count = sun_series.sum();
indices = np.where(sun_df["sunshine"] == 1);
condition_count = (sun_df.loc[indices]["sunshine"] == sun_df.loc[indices]["rainfall"]).astype(int).sum();
print(count);
print(condition_count);
print(condition_count / count);

print(df["humidity"].max());
print(df["humidity"].min());

threshold = 72;

humidity_series = (df["humidity"] <= threshold).astype(int);
rainfall_series = (df["rainfall"] == 1).astype(int);
humidity_df = pd.DataFrame({"humidity": humidity_series, "rainfall": rainfall_series});

count = sun_series.sum();
indices = np.where(humidity_df["humidity"] == 1);
condition_count = (humidity_df.loc[indices]["humidity"] == humidity_df.loc[indices]["rainfall"]).astype(int).sum();
print(count);
print(condition_count);
print(condition_count / count);


from bokeh.palettes import Category10;
from scipy import stats;


def plot_distribution(data, column, plot_type='histogram', bins=30, 
                      width=400, height=300, title=None, color=Category10[10][0],
                      show_kde=True, show_plot=False):
    
    if isinstance(data, pd.DataFrame) and column in data.columns:
        values = data[column].dropna()
    elif isinstance(data, pd.Series):
        values = data.dropna()
        column = data.name if data.name else 'Value'
    else:
        raise ValueError("data must be pandas DataFrame or Series")
    
    if title is None:
        title = f"Distribution of {column}"
    
    p = figure(width=width, height=height, title=title,
              tools="pan,wheel_zoom,box_zoom,reset,save")
    
    if plot_type == 'histogram':
        hist, edges = np.histogram(values, bins=bins)
        source = ColumnDataSource(data={
            'top': hist,
            'left': edges[:-1],
            'right': edges[1:],
            'bottom': np.zeros(len(hist))
        })
        
        p.quad(top='top', bottom='bottom', left='left', right='right',
              source=source, fill_color=color, line_color="white", alpha=0.7,
              hover_fill_color=color, hover_alpha=1.0)
        
        hover = HoverTool(tooltips=[
            (f"Range of {column} ", "[@left{0.00} - @right{0.00}]"),
            ("Amount", "@top")
        ])
        p.add_tools(hover)
        
        if show_kde:
            kde_x = np.linspace(min(values), max(values), 1000)
            kde = stats.gaussian_kde(values)
            kde_y = kde(kde_x) * (max(hist) / max(kde(kde_x)))
            
            p.line(kde_x, kde_y, line_color="red", line_width=2, alpha=0.8, legend_label="KDE")
            p.legend.location = "top_right"
            
    elif plot_type == 'kde':
        kde_x = np.linspace(min(values), max(values), 1000)
        kde = stats.gaussian_kde(values)
        kde_y = kde(kde_x)
        
        source = ColumnDataSource(data={
            'x': kde_x,
            'y': kde_y
        })
        
        p.line('x', 'y', source=source, line_color=color, line_width=3, alpha=0.8)

        p.patch(np.append(kde_x, [kde_x[-1], kde_x[0]]),
                np.append(kde_y, [0, 0]),
                fill_color=color, fill_alpha=0.3, line_color=None)
 
        hover = HoverTool(tooltips=[
            (f"{column}", "@x{0.00}"),
            ("density", "@y{0.0000}")
        ])
        p.add_tools(hover)
    
    p.xaxis.axis_label = column
    p.yaxis.axis_label = "freq" if plot_type == 'histogram' else "density"

    return p

plots = []

for feature in list(df):
    plots.append(plot_distribution(df, feature, plot_type='histogram', title=f'{feature.capitalize()} Distribution'))

grid = gridplot(
    [plots[i:i+2] for i in range(0,len(plots),2)],
    width=400,
    height=300,
    toolbar_location='right'
)

show(grid)


X = df.drop(columns=["rainfall"]);
y = df["rainfall"];


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42);


from sklearn.metrics import confusion_matrix;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import roc_auc_score;

model.fit(X_train, y_train);
model_preds = model.predict_proba(X_test)[:,1];
print("ROC-AUC:", roc_auc_score(y_test, model_preds));


from sklearn.model_selection import cross_val_score, TimeSeriesSplit;
#from sklearn.model_selection import StratifiedKFold;
#skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42);  #suitable for imblanced dataset

tscv = TimeSeriesSplit(n_splits=5);

scores = [];

for train_index, test_index in tscv.split(X,y):
    X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index];
    y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index];

    model.fit(X_fold_train, y_fold_train);
    y_fold_preds = model.predict_proba(X_fold_test)[:,1];
    
    fold_score = roc_auc_score(y_fold_test, y_fold_preds);
    scores.append(fold_score);

scores = np.array(scores);

print("Cross-Validation Roc-auc:", scores);
print("Cross-Validation Mean Roc-auc:", scores.mean());
print("Cross-Validation Standard Deviation of Roc-auc:", scores.std());


def feature_manipulation(_X):
    X = _X.copy(deep=True);
    
    X["cloud_sunshine_interaction"] = X["cloud"] / (X["sunshine"] + 1e-5);
    X["sunshine_temp_interaction"] = (X["sunshine"] + 1e-5) * (X["temparature"]);
    X["low_sunshine"] = (X["sunshine"] < 2.0).astype(int);
    X["cloud_windspeed_interaction"] = X["cloud"] * X["windspeed"];
    X["high_cloud"] = (X["cloud"] > 62).astype(int);
    X["high_humidity"] = (X['humidity'] > 75).astype(int);
    X["cloud_temparature_interaction"] = X["cloud"] * X["temparature"];
    X["cloud_humidity_interaction"] = X["cloud"] * X["humidity"];
    X["cloud_dewpoint_interaction"] = X["cloud"] * X["dewpoint"];
    X["sunshine_humidity_interaction"] = X["humidity"] / (X["sunshine"] + 1e-5);
    X["sunshine_dewpoint_interaction"] = X["dewpoint"] / (X["sunshine"] + 1e-5);
    X["low_sunshine_high_cloud"] = X["low_sunshine"] & X["high_cloud"];

    # Three factors
    X["cloud_humidity_sunshine"] = X["cloud_humidity_interaction"] / (X["sunshine"] + 1e-5);

    X = X.drop(columns=["temparature", "maxtemp"]);

    return X;


X = df.drop(columns=["rainfall"]);
y = df["rainfall"];


X = feature_manipulation(X);


features_by_importance = display_SHAP_importances(X, y, catboost);
print(features_by_importance)
print(len(features_by_importance))


numeric_features = X;  # Since we dropped id and rainfall before, we can directly point numeric_features to df.

plt.figure(figsize=(19,12))
plt.tight_layout()
for i, feature in enumerate(numeric_features):
    plt.subplot(4,6,i+1)
    sns.boxplot(data=X, y=feature) 
    plt.title(feature, wrap=True)
    plt.ylabel('')


from sklearn.preprocessing import RobustScaler;

many_outliners = ["pressure", "dewpoint", 
                  "humidity", "cloud", "windspeed", 
                  "cloud_humidity_interaction", "cloud_windspeed_interaction", 
                  "cloud_temparature_interaction", "cloud_dewpoint_interaction"];

def preprocessing_train(_X):

    X = _X.copy();
    
    robust_scaler = RobustScaler();
    standard_scaler = StandardScaler();

    features_to_standard = [feature for feature in list(X) if feature not in many_outliners ];
    
    X[many_outliners] = robust_scaler.fit_transform(X[many_outliners]);
    X[features_to_standard] = standard_scaler.fit_transform(X[features_to_standard]);

    return X, robust_scaler, standard_scaler;

def preprocessing_test(_X, robust_scaler, standard_scaler):

    X = _X.copy();
    
    features_to_standard = [feature for feature in list(X) if feature not in many_outliners];
    
    X[many_outliners] = robust_scaler.transform(X[many_outliners]);
    X[features_to_standard] = standard_scaler.transform(X[features_to_standard]);
    
    return X;

_X, robust_scaler, standard_scaler = preprocessing_train(X);

scaled_X_df = pd.DataFrame(_X, index=X.index, columns=X.columns);


_X = scaled_X_df.drop([0]).copy();
_y = y.drop([0]).copy();


#top_features = [1,3,5,7,9,11,13,17,21,26,30,34];
top_features = [1,3,5,7,9,10,13,14,15,16,18,19,20,21,22];

mean_scores = [];

for tops in top_features:

    X = _X[features_by_importance[:tops]];
    y = _y;
    
    scores = [];
    
    tscv = TimeSeriesSplit(n_splits=5);
    
    for train_index, test_index in tscv.split(X,y):
        X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index];
        y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index];
    
        model.fit(X_fold_train, y_fold_train);
    
        y_fold_preds = model.predict_proba(X_fold_test)[:,1];
    
        fold_score = roc_auc_score(y_fold_test, y_fold_preds);
        scores.append(fold_score);
        
    scores = np.array(scores);

    print("Cross-Validation Roc-auc:", scores)
    print("Cross-Validation Mean Roc-auc:", scores.mean())
    print("Cross-Validation Standard Deviation of Roc-auc:", scores.std())

    mean_scores.append(scores.mean());

highest_pair = [0, 0.0];  # format: top_features, score
 
for index, score in enumerate(mean_scores):
    if(score > highest_pair[1]):
        highest_pair[0] = top_features[index];
        highest_pair[1] = score;
    print(f"top{top_features[index]} features mean roc-auc:{score}");

print(f"The highest score is {highest_pair[1]} and corresponding feature count is {highest_pair[0]}");


explainer = shap.Explainer(catboost);
shap_values = explainer(_X);

shap.dependence_plot('cloud_humidity_interaction', shap_values.values, _X);
shap.dependence_plot('sunshine_dewpoint_interaction', shap_values.values, _X);


def split_data(_X, _y):

    X = _X.copy();
    y = _y.copy();

    X['year'] = X.index // 365;
    years = X['year'].unique();
    
    test_year = years[-1];

    X_train = X[X['year'] != test_year];
    X_test = X[X['year'] == test_year];

    y_train = y[X['year'] != test_year];
    y_test = y[X['year'] == test_year];

    X_train = X_train.drop(columns=['year']);
    X_test = X_test.drop(columns=['year']);
    
    return X_train, X_test, y_train, y_test;


import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;
from sklearn.metrics import accuracy_score;
import tensorflow as tf;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input;
from tensorflow.keras.metrics import AUC;
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping;

import matplotlib.pyplot as plt;

np.random.seed(42)
tf.random.set_seed(42)

X = _X[features_by_importance[:highest_pair[0]]]
y = _y

n_features = len(list(X))

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = split_data(X, y);

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

side = int(np.ceil(np.sqrt(n_features)))

def reshape_data(X):
    X_reshaped = np.zeros((X.shape[0], side, side, 1))
    for i in range(X.shape[0]):
        x_padded = np.zeros(side * side)
        x_padded[:n_features] = X[i]
        X_reshaped[i, :, :, 0] = x_padded.reshape(side, side)
    return X_reshaped

X_train_reshaped = reshape_data(X_train)
X_test_reshaped = reshape_data(X_test)

model = Sequential([
     Input(shape=(side, side, 1)),
     Conv2D(16, kernel_size=3, activation='leaky_relu', padding='same'),
     MaxPooling2D(pool_size=2),
     Conv2D(8, kernel_size=3, activation='leaky_relu', padding='same'),
     Flatten(),
     Dense(64, activation='leaky_relu'),
     Dropout(0.3),
     Dense(32, activation='leaky_relu'),
     Dropout(0.2),
     Dense(1, activation='sigmoid')
]);

model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=[
        AUC(name='auc'),
        'accuracy'
    ]
)

model.summary()

callbacks = [
    ModelCheckpoint(
        'best_cnn_model.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True, 
        verbose=1
    )
]

history = model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    callbacks=callbacks,
    verbose=1
)

model.load_weights('best_cnn_model.keras')

y_pred_prob = model.predict(X_test_reshaped).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)

print(f'Test set AUC: {auc:.4f}')
print(f'Test set accuracy: {accuracy:.4f}')


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['train set', 'val set'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train set', 'val set'], loc='upper right')

plt.tight_layout()
plt.show()



features_selected = features_by_importance[:(highest_pair[0])];

test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv");
test_id = test_df["id"];

print(f"Test data sample count: {len(test_df)}");
print(f"Last id of test data: {test_df.iloc[-1]}");

X_test = test_df.drop(columns=["id", "day"]);
X_test['winddirection'].fillna(X_test['winddirection'].median());
X_test['expected_day'] = (test_id) % 365 + 1;

X_test = feature_manipulation(X_test);

X_test_scaled = preprocessing_test(X_test, robust_scaler, standard_scaler);

X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns);

X_test = X_test_scaled_df[features_selected];

print(len(X_test))


X_test_values = X_test.values;

X_test_scaled = reshape_data(X_test_values);

test_preds = model.predict(X_test_scaled).flatten()

# Ensure no nan
nan_sum = np.isnan(test_preds).sum()
if nan_sum > 0:
    test_preds = np.nan_to_num(test_preds)  
test_preds[:10]


submit_data = {'id': test_id, 'rainfall': test_preds}
submission_df = pd.DataFrame(data=submit_data)
submission_df.to_csv("/kaggle/working/submission.csv", index=False);

rows_with_nan = submission_df[submission_df.isna().any(axis=1)]

print(rows_with_nan)

submission_df.head()

