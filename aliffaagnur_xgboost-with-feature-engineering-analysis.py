import umap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import lightgbm
import catboost
import xgboost

pd.set_option('display.max_columns', None)

import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = PerformanceWarning)
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)


# LOAD DATA

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e7/train.csv')
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e7/test.csv')

train_df.shape, test_df.shape


# COPY DATA

# COPY DATA TO BE MANIPULATED
new_train_df = train_df.copy()   
new_test_df  = test_df.copy()


# ENCODING CATEGORICAL

personality_encoding = {
    'Extrovert' : 0,
    'Introvert' : 1
}

drained_encoding = {
    'No' : 0,
    'Yes' : 1
}

stage_encoding = {
    'No' : 0, 
    'Yes' : 1
}

new_train_df['Drained_after_socializing_encoded'] = new_train_df['Drained_after_socializing'].map(drained_encoding)
new_test_df['Drained_after_socializing_encoded']  = new_test_df['Drained_after_socializing'].map(drained_encoding)

new_train_df['Stage_fear_encoded'] = new_train_df['Stage_fear'].map(stage_encoding)
new_test_df['Stage_fear_encoded'] = new_test_df['Stage_fear'].map(stage_encoding)

new_train_df['Personality_encoded'] = new_train_df['Personality'].map(personality_encoding)


display(train_df)
display(test_df)


# CHECK DATA TYPE

display(train_df.info())


# CHECK MISSING
display(train_df.isna().sum()) ; display(test_df.isna().sum())



# SET STYLE
sns.set(style="whitegrid", context="talk")

# COUNT TARGET DISTRIBUTION
target_distribution = train_df['Personality'].value_counts()

# CREATE PALETTE CUSTOM
palette = sns.color_palette("Set2", len(target_distribution))


plt.figure(figsize=(6, 5))
ax = sns.barplot(
    x=target_distribution.index,
    y=target_distribution.values,
    palette=palette,
    edgecolor='black'
)
# SHOW LABEL AND TITLE
plt.title('Personality Distribution', fontsize=15, weight='bold')
plt.xlabel('Personality', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

# ADD VALUE EACH BAR
for p in ax.patches:
    ax.annotate(
        format(p.get_height(), '.0f'),
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center',
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=12
    )

# MAKE IT AESTHETIC
sns.despine(top=True, right=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()



# NUMERIC DISTRIBUTION

# SET STYLE LAYOUT
sns.set(style="whitegrid", context="notebook")

# SETUP LAYOUT
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

# SET COLOR
train_color = sns.color_palette("pastel")[0]
test_color = sns.color_palette("muted")[2]


# CHOOSE COLS
numerical_cols = train_df.select_dtypes(include='number').columns.drop(labels='id')

# PLOT EACH FEATURE
for i, col in enumerate(numerical_cols):
    sns.histplot(train_df[col], ax=axes[i], color=train_color, label='Train', kde=True, stat='density', element='step', fill=True, alpha=0.6)
    sns.histplot(test_df[col], ax=axes[i], color=test_color, label='Test', kde=True, stat='density', element='step', fill=True, alpha=0.6)

    axes[i].set_title(col, fontsize=13, weight='bold')
    axes[i].set_xlabel('')
    axes[i].legend()
    axes[i].grid(True)

# DELETE EMPTY SUBPLOT IF COLS < 6
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout()
plt.suptitle('Numeric Distribution Train vs Test', fontsize=16, weight='bold', y=1.02)
plt.show()



# CATEGORICAL DISTRIBUTION

categorical_cols = train_df.select_dtypes(include='object').columns.drop(labels='Personality')


sns.set(style="whitegrid", context="notebook")

# SET LAYOUT
fig, axes = plt.subplots(1, len(categorical_cols), figsize=(14, 5))
if len(categorical_cols) == 1:
    axes = [axes]

for i, col in enumerate(categorical_cols):

    # MERGE TRAIN AND TEST DATA
    combined_df = pd.concat([
        train_df[[col]].assign(Source='Train'),
        test_df[[col]].assign(Source='Test')
    ], ignore_index=True)

    # COUNTPLOT
    sns.countplot(data=combined_df, x=col, hue='Source', ax=axes[i],
                  palette=['#8ecae6', '#219ebc'])
    
    # SET TITLE AND LABEL
    axes[i].set_title(col, fontsize=12, weight='bold')
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].legend(title='Dataset', loc='upper right')

plt.suptitle('Categorical Distribution Train vs Test', fontsize=16, weight='bold')
plt.tight_layout()
plt.show()


# CHECK CORRELATION

# CHOOSE ALL FITUR EXCEPT 'id'
numerical_cols = new_train_df.select_dtypes(include='number').columns.drop('id')

# SPEARMAN CORRELATION
matrix_corr = new_train_df[numerical_cols].corr(method='spearman')

# MASK (SO THAT THE DISPLAY DOES NOT DUPLICATE)
mask = np.triu(np.ones_like(matrix_corr, dtype=bool))


plt.figure(figsize=(12, 8))

sns.heatmap(matrix_corr, mask=mask,
            vmin=-1, vmax=1, cmap='coolwarm',
            annot=True, fmt=".2f",
            annot_kws={"size": 10},
            linewidths=0.5, linecolor='gray', square=True,
            cbar_kws={"shrink": .8, "label": "Spearman Correlation"}
            )

plt.title('Heatmap Correlation', fontsize=16, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()



# LETS DO SOME STATISTICS

numerical_cols = new_train_df.select_dtypes(include = 'number').drop(labels = ['id', 'Personality_encoded'], axis = 1).columns

new_train_df.groupby(by = 'Personality')[numerical_cols].agg(func = ['mean', 'median', 'std', 'min', 'max'])


# PCA


# DROP ALL NULL VALUES BCAUSE PCA CAN'T HANDLE NULL VALUES
train_copy = new_train_df.copy()
train_copy = train_copy.dropna()

x = train_copy[['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency', 'Drained_after_socializing_encoded', 'Stage_fear_encoded']]
y = train_copy['Personality']

# Z-SCORE
zscore = StandardScaler()
x_scaled = zscore.fit_transform(x)

# PCA
pca_2d = PCA(n_components = 2).fit_transform(x_scaled)  # --> PCA 2D
pca_3d = PCA(n_components = 3).fit_transform(x_scaled)  # --> PCA 3D


# VISUALIZE PCA
fig = plt.figure(figsize = (18, 10))

# 2D PCA (LEFT)
ax1 = fig.add_subplot(1,2,1)
scatter_2d = ax1.scatter(x = pca_2d[:, 0], y = pca_2d[:, 1], c = y.astype('category').cat.codes, cmap = 'Set2')

ax1.set_title('2D PCA Projection')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')


ax2 = fig.add_subplot(1,2,2, projection = '3d')
scatter_3d = ax2.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c = y.astype('category').cat.codes, cmap = 'Set2')

ax2.set_title("3D PCA Projection")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")

# ADD LEGENDS
labels = y.unique()
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=scatter_2d.cmap(scatter_2d.norm(i)), markersize=10)
           for i in range(len(labels))]
ax1.legend(handles, labels, title="Personality", fontsize = 18)
ax2.legend(handles, labels, title='Personality', fontsize = 18)
plt.tight_layout()
plt.show()



# # LET'S ANALYZE WHICH FEATURE VARIATIONS HAVE THE GREATEST IMPACT ON PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(x_scaled)

# VARIANCE EXPLAINED
explained = pca.explained_variance_ratio_ * 100

# CONVERT TO DATAFRAME
components_df = pd.DataFrame(
    np.abs(pca.components_.T),
    index=x.columns,
    columns=[f'PC{i+1} ({explained[i]:.1f}%)' for i in range(2)]
)

plt.figure(figsize=(10, 6))
sns.heatmap(components_df,
            annot=True, fmt=".2f",
            cmap='Set2',cbar=True,
            linewidths = 0.5,linecolor='gray',
            xticklabels = components_df.columns,
            yticklabels = components_df.index,
            annot_kws={"size": 11}
)

plt.title('ğŸ”�Feature Contribution to Principal Components (PCA)', fontsize=15, weight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()



%%time
# t-SNE

tsne = TSNE(n_components = 2,          
            perplexity = 50,         
            early_exaggeration = 12, 
            n_iter = 1000,           
            n_iter_without_progress = 300, 
            min_grad_norm = 1e-7,          
            metric = 'euclidean',               
            metric_params = None,          
            init = 'pca',                  
            verbose = 0,
            random_state = 2025,
            method = 'barnes_hut', 
            angle = 0.5)

tsne_result = tsne.fit_transform(x_scaled)

# t-SNE VISUALIZATION
plt.figure(figsize=(10, 7))
scatter_tsne = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                           c=y.astype('category').cat.codes, cmap='Set2')
plt.title('t-SNE Projection')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# ADD LEGEND
labels = y.unique()
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=scatter_tsne.cmap(scatter_tsne.norm(i)), markersize=10)
           for i in range(len(labels))]
plt.legend(handles, labels, title="Personality", fontsize=12)

plt.show()


%%time
# UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(x_scaled)

# UMAP VISUALIZATION
plt.figure(figsize=(10, 7))
scatter_umap = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                           c=y.astype('category').cat.codes, cmap='Set2')
plt.title('UMAP Projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# ADD LEGEND
labels = y.unique()
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=scatter_umap.cmap(scatter_umap.norm(i)), markersize=10)
           for i in range(len(labels))]
plt.legend(handles, labels, title="Personality", fontsize=12)
plt.show()



# FEATURE COMBINATION

def feature_combination(train, test, cols_to_combine, pair_size):
    
    for pair in pair_size:
        for cols in combinations(cols_to_combine, pair):

            # DEFINE NAME OF NEW FEATURE
            new_col_name = '_'.join(cols)

            # SET NEW FEATURE TRAIN DATA
            train[new_col_name] = train[cols[0]].astype(str)

            # CONCAT PAIRS FEATURE TRAIN DATA
            for col in cols[1:]:
                train[new_col_name] = train[new_col_name] + '_' + train[col].astype(str)


            test[new_col_name] = test[cols[0]].astype(str)

            # CONCAT PAIRS FEATURE TEST DATA
            for col in cols[1:]:
                test[new_col_name] = test[new_col_name] + '_' + test[col].astype(str)

    return train, test


# CHOOSE ALL INDEPENDENT COLUMNS
cols_to_combine = train_df.columns.drop(labels = ['id', 'Personality'])

# APPLY FUNCTION
new_train_df, new_test_df = feature_combination(new_train_df, new_test_df, cols_to_combine, pair_size = [2])


display(f'Train shape after FE : {new_train_df.shape}')
display(f'Test shape after FE  : {new_test_df.shape}')

new_train_df


# TARGET ENCODING 

def target_encoding(train, test, cols_to_encode, target, cv, agg= ['mean']):
    
    # FOR EVERY CAT COLS
    for col in cols_to_encode:
        
        # FIT TARGET ENCODER ONLY ON VALIDATION SPLIT
        for train_idx, valid_idx in cv.split(train, train[target]):
            X_tr = train.iloc[train_idx]
            X_val = train.iloc[valid_idx]

            # APPLY EACH AGGREGATION STATS
            for stat in agg:
                stat_result = X_tr.groupby(col)[target].agg(stat)

                # APPLY ENCODING ON VAL DATA
                name_new_cols = f'{col}_{stat}_te'
                train.loc[valid_idx, name_new_cols] = X_val[col].map(stat_result)

        # ENCODE TEST DF ON GLOBAL TRAIN DATA
        for stat in agg:
            print(f'Encoding {col} with {stat} Aggregation Successfully')
            
            global_stat = train.groupby(col)[target].agg(stat)

            name_new_cols = f'{col}_{stat}_te'
            test[name_new_cols] = test[col].map(global_stat)

    # DROP ORIGINAL COLUMN AFTER ENCODED
    train = train.drop(columns = cols_to_encode)
    test = test.drop(columns = cols_to_encode)

    return train, test

# DEFINE CAT COLS
cat_cols = new_train_df.select_dtypes(include = 'object').drop(labels = ['Stage_fear', 'Drained_after_socializing', 'Personality'], axis = 1).columns

skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state = 2025)


new_train_df, new_test_df = target_encoding(new_train_df, new_test_df, 
                                            cols_to_encode = cat_cols, 
                                            target = 'Personality_encoded', cv = skf,
                                            agg = ['mean'])

new_train_df


# STATISTIC AGGREGATION

def aggregation(data : pd.DataFrame, agg_list : list, cols_used : list, name_new_cols = ''):

    # ITERATE OVER AGGREGATION LIST
    for agg in agg_list:

        # DEFINE NAME OF NEW FEATURE
        new_cols = f'{name_new_cols}{agg}'

        # CREATE NEW FEATURE BASED ON STATISTICS 
        data.loc[:, new_cols] = data[cols_used].agg(agg, axis = 1)

    return data

# GLOBAL AGGREGATION WITH GLOBAL FEATURE (INCLUDING ENGINEERED FEATURE)
agg_list = ['mean', 'std', 'min', 'max', 'sum']
cols_used = new_test_df.drop(columns = ['id', 'Stage_fear', 'Drained_after_socializing']).columns  # ---> GET ALL FEATURES 

new_train_df = aggregation(new_train_df, agg_list, cols_used)
new_test_df  = aggregation(new_test_df, agg_list, cols_used)

# ---------------------------------------------------------------------------------


# LOCAL AGGREGATION WITH BASE FEATURES
agg_list  = ['mean', 'std', 'min', 'max', 'sum']
cols_used = ['Time_spent_Alone', 'Social_event_attendance',	'Going_outside', 'Friends_circle_size',	'Post_frequency', 'Drained_after_socializing_encoded','Stage_fear_encoded'] 


new_train_df = aggregation(new_train_df, agg_list, cols_used, name_new_cols = 'base_')
new_test_df  = aggregation(new_test_df, agg_list, cols_used, 'base_')

display(f'Train shape after Feature Aggregation : {new_train_df.shape}')
display(f'Test shape after Feature Aggregation  : {new_test_df.shape}')

new_train_df


# SPLIT DATA

x = new_train_df.drop(columns = ['id', 'Personality', 'Personality_encoded', 'Drained_after_socializing', 'Stage_fear'])
y = new_train_df['Personality_encoded']

new_test_df = new_test_df.drop(columns = ['id', 'Drained_after_socializing', 'Stage_fear'])

# DISPLAY
print(f'x shape : {x.shape}')
print(f'y shape : {y.shape}')
print(f'x datatype : {type(x)}')
print(f'y datatype : {type(y)}')


# DEFINE XGB PARAMETERS

xgb_params = {
    'objective' : 'binary:logistic',
    'eval_metric' : 'logloss',
    'booster' : 'gbtree',
    'learning_rate' : 0.1,
    'max_depth' : 4,
    #'scale_pos_weight' : len(train_df[train_df['Personality'] == 'Extrovert']) / len(train_df[train_df['Personality'] == 'Introvert']),
    'gamma' : 0,
    'colsample_bytree' : 0.4,
    'subsample' : 0.7,
    'lambda' : 2,
    'alpha' : 1,
    'tree_method' : 'hist',
    'random_state' : 2025,
    'verbosity' : 1
}


# XGBOOST KFOLD

skfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

score_train, score_val = [], []

oof_preds = np.zeros(len(x))  ; oof_preds_proba = np.zeros(len(x))
test_preds = []

# SKFOLD
for i, (train_index, val_index) in enumerate(skfold.split(x, y)):

    # SPLIT
    x_train , x_val = x.iloc[train_index], x.iloc[val_index]
    y_train, y_val  = y[train_index], y[val_index]

    # CONVERT TO XGB DATASET
    dtrain = xgboost.DMatrix(x_train, y_train)
    dval = xgboost.DMatrix(x_val, y_val)
    dtest = xgboost.DMatrix(new_test_df)

    # FIT MODEL
    xgb = xgboost.train(xgb_params, dtrain, 
                        num_boost_round = 1000, 
                        evals = [(dval, 'val')], 
                        early_stopping_rounds = 50, 
                        verbose_eval = False)
    
    # PREDICT
    y_train_predict = (xgb.predict(dtrain) >= 0.52).astype(int)  # --> SET THE THRESHOLD
    y_val_predict   = (xgb.predict(dval) >= 0.52).astype(int)

    oof_preds[val_index] = y_val_predict
    oof_preds_proba[val_index] = xgb.predict(dval)


    # CHECK METRICS
    accuracy_train = accuracy_score(y_train, y_train_predict)
    accuracy_val   = accuracy_score(y_val, y_val_predict)

    print(f'Fold {i+1} ğŸš€: 1ï¸�âƒ£ Train Accuracy = {accuracy_train}, 2ï¸�âƒ£ Val Accuracy = {accuracy_val}')

    score_train.append(accuracy_train)
    score_val.append(accuracy_val)

    # TEST PREDICTION
    y_test_prediction = xgb.predict(dtest)
    test_preds.append(y_test_prediction)

print(f'\nOverall OOF ğŸ�‰: 1ï¸�âƒ£ Train Accuracy = {np.mean(score_train)}, 2ï¸�âƒ£ Val Accuracy = {np.mean(score_val)}')


# CHECK BEST THRESHOLD

thresholds = np.arange(0.0, 1.01, 0.01)
accuracy_scores = []

for thresh in thresholds:
    y_pred = (oof_preds_proba >= thresh).astype(int)
    acc = accuracy_score(y, y_pred)
    accuracy_scores.append(acc)

best_idx = np.argmax(accuracy_scores)
best_threshold = thresholds[best_idx]
print(f'âœ… Best Threshold = {best_threshold:.2f}, Accuracy = {accuracy_scores[best_idx]:.4f}')



# VISUALIZE THRESHOLD

plt.plot(thresholds, accuracy_scores, label='Accuracy')
plt.axvline(x=best_threshold, color='gray', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Threshold vs Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# CHECK WRONG PREDICTION
wrong_idx_global = np.where(oof_preds != y)[0]

# DISPLAY WRONG PREDICTION
df_wrong_preds = pd.DataFrame({
    'y_true': y.iloc[wrong_idx_global].values,
    'y_pred': oof_preds[wrong_idx_global]
}, index=wrong_idx_global)

# 
x_wrong = x.iloc[wrong_idx_global].copy()


x_wrong['y_true'] = y.iloc[wrong_idx_global].values
x_wrong['y_pred'] = oof_preds[wrong_idx_global]

display(f'There are {len(x_wrong)} Misclassified Data\n')

x_wrong[['Time_spent_Alone', 'Social_event_attendance',	'Going_outside', 'Friends_circle_size',	'Post_frequency', 'Drained_after_socializing_encoded', 'Stage_fear_encoded', 'y_true', 'y_pred']]


# FEATURE IMPORTANCE

fig, ax = plt.subplots(figsize = (13, 20), dpi = 400)
xgboost.plot_importance(booster = xgb, importance_type = 'gain', ax = ax)


# SPLIT IMPORTANCE

fig, ax = plt.subplots(figsize = (13, 20), dpi = 400)
xgboost.plot_importance(booster = xgb, importance_type = 'weight', ax = ax)


# DISPLAY PROBABILITY DISTRIBUTION OOF PREDS

sns.histplot(oof_preds_proba, bins=50, kde=True)

plt.title("Probability Distribution on Test df (Binary Classification)")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# LETS ZOOM IN BETWEEN 0.4 - 0.6

sns.histplot(oof_preds_proba, bins=100, kde=True)

# Zoom to 0.4â€“0.6
plt.xlim(0.4, 0.6)
plt.ylim(0, 10)

plt.title("Zoom: Probability Distribution (0.4 - 0.6)")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# SUBMISSION

y_test = np.mean(test_preds, axis = 0)   # --> TEST DATA

submission = pd.read_csv(r'/kaggle/input/playground-series-s5e7/sample_submission.csv')

submission.iloc[:, 1] = (y_test > 0.5).astype(int)

inverse_encoding = {
    0 : 'Extrovert',
    1 : 'Introvert'
}

submission['Personality'] = submission['Personality'].map(inverse_encoding)

submission


submission.to_csv(r'submission.csv', index = False)


# CHECK PREDICTION PROBABILITY ON TEST DATA
sns.histplot(y_test, bins=50, kde=True)

plt.title("Probability Distribution on Test df (Binary Classification)")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# LETS ZOOM IN BETWEEN 0.4 - 0.7

sns.histplot(y_test, bins=80, kde=True)

# Zoom to 0.4â€“0.7
plt.xlim(0.4, 0.7)
plt.ylim(0, 10)

plt.title("Zoom: Probability Distribution (0.4 - 0.7)")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

