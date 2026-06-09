import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from catboost import CatBoostClassifier
import warnings


test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")


train.describe()


# count missing
print(train.isna().sum())

plt.figure(figsize=(6,4))
sns.heatmap(train.isna(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Values Map')
plt.show()


num_cols = ['Time_spent_Alone','Social_event_attendance',
            'Going_outside','Friends_circle_size','Post_frequency']
cat_cols = ['Stage_fear','Drained_after_socializing']

# numeric features
for col in num_cols:
    plt.figure()
    sns.histplot(train[col], kde=True)
    plt.title(col)
    plt.show()

# categorical features
for col in cat_cols:
    plt.figure()
    sns.countplot(x=col, data=train)
    plt.title(col)
    plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(train[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Numeric Feature Correlation')
plt.show()

# optional pairplot (subsample if large)
sns.pairplot(train[num_cols].dropna().sample(200), corner=True)
plt.show()



# Define your feature lists
num_feats = ['Time_spent_Alone','Social_event_attendance',
             'Going_outside','Friends_circle_size','Post_frequency']
cat_feats = ['Stage_fear','Drained_after_socializing']

# Separate X and y
X = train.drop(['id','Personality'], axis=1)
y_raw = train['Personality']
X_sub = test.drop(['id'], axis=1)

# Label‐encode y
le = LabelEncoder()
y = le.fit_transform(y_raw)
# => Extrovert→0, Introvert→1

# Numeric pipeline: KNN → scaling
num_estimator = RandomForestRegressor(n_estimators=15, max_depth=4, random_state=42)
num_transformer = Pipeline([
    ('imputer', IterativeImputer(estimator=num_estimator,
                                 max_iter=30,
                                 tol=0.01,
                                 initial_strategy='median',
                                 random_state=0)),
    ('scaler',  StandardScaler())
])

# Categorical pipeline: Ordinal encode → IterativeImputer → OneHotEncode
cat_transformer = Pipeline([
    # turn strings into integers (unknown→nan so imputer can fill)
    ('ord_enc', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=np.nan
    )),
    # predictive imputer: here RandomForest for “regressing” the codes
    ('imp', IterativeImputer(
        estimator=RandomForestClassifier(n_estimators=8, random_state=0),
        max_iter=30,
        tol=0.01,
        random_state=0
    )),
    # back to one‐hot so downstream models see proper dummies
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into the full ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_feats),
    ('cat', cat_transformer, cat_feats)
])


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)


# Optuna objective
def objective(trial):
    params = {
        'iterations':      trial.suggest_int('iterations', 100, 1000),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':           trial.suggest_int('depth', 2, 10),
        'l2_leaf_reg':     trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count':    trial.suggest_int('border_count', 32, 255),
        'loss_function':   'Logloss',
        'eval_metric':     'Accuracy',
        'verbose':         0,
        'random_seed':     42,
        'auto_class_weights': 'Balanced',
        'allow_writing_files': False  # avoid unnecessary disk I/O
    }

    # Pipeline with preprocessing (optional if CatBoost handles categorical natively)
    pipe = Pipeline([
        ('preproc', preprocessor),  # Optional: use if you already have a transformer
        ('catboost', CatBoostClassifier(**params))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Launch Optuna
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50, timeout=600)

# Inspect results
print("Best CV accuracy:", study.best_value)
print("Best hyperparameters:", study.best_params)

# Retrain final CatBoost model
best_params = study.best_params
final_pipe = Pipeline([
    ('preproc', preprocessor),
    ('catboost', CatBoostClassifier(
        **{**best_params},
        loss_function='Logloss',
        eval_metric='Accuracy',
        auto_class_weights='Balanced',
        allow_writing_files=False,
        random_seed=42,
        verbose=0
    ))
])
final_pipe.fit(X_train, y_train)
print("Holdout accuracy:", final_pipe.score(X_test, y_test))


y_pred = final_pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
disp = ConfusionMatrixDisplay.from_estimator(
    final_pipe, X_test, y_test,
    display_labels=le.classes_, cmap='Blues', normalize='true'
)
disp.ax_.set_title(f"Model Confusion Matrix (normed)")
plt.show()


# Build a DataFrame of X_test + true/pred labels + “is_error”
df = X_test.copy()
df['true'] = y_test
df['pred'] = y_pred
df['is_error'] = df['pred'] != df['true']

for feature in num_feats:
    # Plot, e.g., Feature for correct vs. wrong
    plt.figure(figsize=(6,3))
    sns.boxplot(
        x='is_error', y=feature,
        data=df.replace({True:'Error',False:'Correct'})
    )
    plt.title(f"Model Errors by {feature}")
    plt.xlabel("")
    plt.show()


# Boolean mask & extract error samples
err_mask = df['is_error']               # Boolean Series over X_test’s index
X_err    = X_test[err_mask]              # features of misclassified points
y_err    = pd.Series(y_test[err_mask])              # true labels of those same points

# Preprocess and embed to 2D
X_err_p = preprocessor.transform(X_err)  # pipeline’s preprocessing
tsne    = TSNE(n_components=2, random_state=42)
emb_err = tsne.fit_transform(X_err_p)

# Cluster the embedded error points
cluster_labels = DBSCAN(eps=3, min_samples=5).fit_predict(emb_err)

# Build a DataFrame to hold everything
df_err = X_err.reset_index(drop=True).copy()
df_err['true_label'] = y_err.reset_index(drop=True)
df_err['tsne1']    = emb_err[:,0]
df_err['tsne2']    = emb_err[:,1]
df_err['cluster']  = cluster_labels

# Plot t-SNE colored by error-cluster
plt.figure(figsize=(6,5))
palette = sns.color_palette("tab10", len(np.unique(cluster_labels)))
sns.scatterplot(
    x='tsne1', y='tsne2',
    hue='cluster',
    data=df_err,
    palette=palette,
    legend='full',
    s=50
)
plt.title("t-SNE of model Misclassifications (DBSCAN clusters)")
plt.show()

# Profile each cluster’s numeric stats
overall_stats = X_test[num_feats].describe().T

for cl in sorted(df_err['cluster'].unique()):
    sub = df_err[df_err['cluster'] == cl]
    print(f"\nCluster {cl} — {len(sub)} samples")
    cluster_stats = sub[num_feats].describe().T

    # Combine side by side with multi‐level columns
    combined = pd.concat(
        [cluster_stats, overall_stats],
        axis=1,
        keys=[f'Cluster {cl}', 'Overall']
    )

    print(f"\n=== Statistics for Error Cluster {cl} ===")
    display(combined)

    # Boxplot for numerical feature in that cluster vs. overall
    for feature in num_feats:
        plt.figure(figsize=(4,2.5))
        sns.boxplot(
            data=pd.concat([
                sub[[f'{feature}']].assign(dataset=f'Cluster {cl}'),
                X_test[[f'{feature}']].assign(dataset='Overall')
            ]),
            x='dataset', y=f'{feature}',
            palette=['salmon','lightgray']
        )
        plt.title(f"{feature}: Cluster {cl} vs. Overall")
        plt.show()

    # Categorical: show proportions side‐by‐side
    print(f"\n--- Categorical Distributions for Cluster {cl} vs. Overall ---")
    for feat in cat_feats:
        # cluster proportions
        p_cl = sub[feat].value_counts(normalize=True).sort_index()
        # overall proportions
        p_all = X_test[feat].value_counts(normalize=True).reindex(p_cl.index, fill_value=0)
        # combine into a single DataFrame
        cat_df = pd.concat(
            [p_cl, p_all],
            axis=1,
            keys=[f'Cluster {cl}', 'Overall']
        )
        cat_df.index.name = feat
        display(cat_df)

        # Bar‐plot comparison
        plot_df = pd.concat([
            sub[[feat]].assign(dataset=f'Cluster {cl}'),
            X_test[[feat]].assign(dataset='Overall')
        ])
        plt.figure(figsize=(4,2.5))
        sns.countplot(
            data=plot_df,
            x=feat,
            hue='dataset',
            palette=['salmon','lightgray']
        )
        plt.title(f"{feat}: Cluster {cl} vs. Overall")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


# Add tsne features
X2 = X.copy()
X2_p = preprocessor.transform(X2)
emb = tsne.fit_transform(X2_p)
X2['tsne1'] = emb[:,0]
X2['tsne2'] = emb[:,1]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=55
)


# Optuna objective
def objective(trial):
    params = {
        'iterations':      trial.suggest_int('iterations', 100, 1000),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth':           trial.suggest_int('depth', 2, 10),
        'l2_leaf_reg':     trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count':    trial.suggest_int('border_count', 32, 255),
        'loss_function':   'Logloss',
        'eval_metric':     'Accuracy',
        'verbose':         0,
        'random_seed':     42,
        'auto_class_weights': 'Balanced',
        'allow_writing_files': False  # avoid unnecessary disk I/O
    }

    # Pipeline with preprocessing (optional if CatBoost handles categorical natively)
    pipe = Pipeline([
        ('preproc', preprocessor),  # Optional: use if you already have a transformer
        ('catboost', CatBoostClassifier(**params))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X2_train, y2_train, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Launch Optuna
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50, timeout=600)

# Inspect results
print("Best CV accuracy with added features:", study.best_value)
print("Best hyperparameters with added features:", study.best_params)

# Retrain final CatBoost model
best_params = study.best_params
final_pipe = Pipeline([
    ('preproc', preprocessor),
    ('catboost', CatBoostClassifier(
        **{**best_params},
        loss_function='Logloss',
        eval_metric='Accuracy',
        auto_class_weights='Balanced',
        allow_writing_files=False,
        random_seed=42,
        verbose=0
    ))
])
final_pipe.fit(X2_train, y2_train)
print("Holdout accuracy with added features:", final_pipe.score(X2_test, y2_test))


# Add tsne features
X2_sub = X_sub.copy()
X2_sub_p = preprocessor.transform(X2_sub)
emb_sub = tsne.fit_transform(X2_sub_p)
X2_sub['tsne1'] = emb_sub[:,0]
X2_sub['tsne2'] = emb_sub[:,1]
y_sub = final_pipe.predict(X2_sub)

submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = le.inverse_transform(y_sub)
submission.to_csv('submission.csv', index=False)


sub=pd.read_csv("submission.csv")
sub

