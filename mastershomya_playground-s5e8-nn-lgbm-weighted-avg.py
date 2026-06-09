import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
import plotly.io as pio
from IPython.display import IFrame
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df.drop(columns=['id'],inplace=True)
df.head()


def create_summaries(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # --- Numerical Summary ---
    num_summary = df[numerical_cols].describe().transpose()
    num_missing = df[numerical_cols].isnull().sum().rename('Missing Values')
    num_summary = pd.concat([df[numerical_cols].dtypes.rename('Data Type'), num_missing, num_summary], axis=1)
    
    # --- Categorical Summary ---
    cat_summary = df[categorical_cols].describe().transpose()
    cat_missing = df[categorical_cols].isnull().sum().rename('Missing Values')
    cat_summary = pd.concat([df[categorical_cols].dtypes.rename('Data Type'), cat_missing, cat_summary], axis=1)
    
    return num_summary, cat_summary

numerical_summary, categorical_summary = create_summaries(df)
print("ðŸ”¢ Numerical Features Summary")
display(numerical_summary)
print("ðŸ”  Categorical Features Summary")
display(categorical_summary)


group_counts = df['job'].value_counts().reset_index()
group_counts.columns = ['job', 'Count']

fig_bubble = px.scatter(
    group_counts,
    x='job',
    y='Count',
    size='Count',
    color='job',
    hover_name='job',
    size_max=70,
    title='Relative Size of Occupation'
)

fig_bubble.update_layout(
    xaxis_title='',
    yaxis_title='',
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    plot_bgcolor='white',
    showlegend=True
)

pio.write_html(fig_bubble, file="fig_bubble.html", auto_open=False)
IFrame("fig_bubble.html", width=1250, height=500)


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
bp_counts = df['marital'].value_counts()
pie_colors = ['#336699', '#a3cde3', '#AFEEEE']

# pie chart
axes[0].pie(bp_counts,
            labels=bp_counts.index,
            autopct='%.2f%%',
            startangle=90,
            colors=pie_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})

axes[0].set_title('Distribution of Marital Status', fontsize=16, fontweight='bold')


# Bar Plot
bar_palette = "Blues"
sns.countplot(ax=axes[1],
              data=df,
              x='marital',
              hue='y',
              palette=bar_palette,
              order=bp_counts.index)

axes[1].set_title('Subscription of bank term deposit by marital status', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Marital Status', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].legend(title='Subscription for Bank Term Deposit')

plt.suptitle('Marital Status', fontsize=20, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
bp_counts = df['education'].value_counts()
pie_colors = ['#336699', '#a3cde3', '#AFEEEE']

# pie chart
axes[0].pie(bp_counts,
            labels=bp_counts.index,
            autopct='%.2f%%',
            startangle=90,
            colors=pie_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})

axes[0].set_title('Distribution of Education', fontsize=16, fontweight='bold')


# Bar Plot
bar_palette = "Blues"
sns.countplot(ax=axes[1],
              data=df,
              x='education',
              hue='y',
              palette=bar_palette,
              order=bp_counts.index)

axes[1].set_title('Subscription of bank term deposit by Education', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Education', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].legend(title='Subscription for Bank Term Deposit')

plt.suptitle('Education', fontsize=20, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.countplot(ax=axes[0],
              data=df,
              x='default',
              palette='Blues') 

axes[0].set_title('Distribution of Default Status', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Default Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)


crosstab_norm = pd.crosstab(df['default'],
                           df['y'],
                           normalize='index') * 100

# Plot the 100% stacked bar chart
crosstab_norm.plot(kind='bar',
                   stacked=True,
                   ax=axes[1],
                   color=['#a3cde3', '#336699'])

axes[1].set_title('Subscription to bank term deposit by Default Status', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Default Status', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].legend(title='will subscribe?', loc='upper right')
axes[1].tick_params(axis='x', rotation=0)
plt.suptitle('Default Status', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.countplot(ax=axes[0],
              data=df,
              x='housing',
              palette='Blues') 

axes[0].set_title('Distribution of Housing Status', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Hosuing Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)


crosstab_norm = pd.crosstab(df['housing'],
                           df['y'],
                           normalize='index') * 100

# Plot the 100% stacked bar chart
crosstab_norm.plot(kind='bar',
                   stacked=True,
                   ax=axes[1],
                   color=['#a3cde3', '#336699'])

axes[1].set_title('Subscription to bank term deposit by Housing Status', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Housing Status', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].legend(title='will subscribe?', loc='upper right')
axes[1].tick_params(axis='x', rotation=0)
plt.suptitle('Housing Status', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.countplot(ax=axes[0],
              data=df,
              x='loan',
              palette='Blues') 

axes[0].set_title('Distribution of Loan Status', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Loan Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)


crosstab_norm = pd.crosstab(df['loan'],
                           df['y'],
                           normalize='index') * 100

# Plot the 100% stacked bar chart
crosstab_norm.plot(kind='bar',
                   stacked=True,
                   ax=axes[1],
                   color=['#a3cde3', '#336699'])

axes[1].set_title('Subscription to bank term deposit by Loan Status', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Loan Status', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].legend(title='will subscribe?', loc='upper right')
axes[1].tick_params(axis='x', rotation=0)
plt.suptitle('Loan Status', fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
bp_counts = df['contact'].value_counts()
pie_colors = ['#336699', '#a3cde3', '#AFEEEE']

# pie chart
axes[0].pie(bp_counts,
            labels=bp_counts.index,
            autopct='%.2f%%',
            startangle=90,
            colors=pie_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})

axes[0].set_title('Distribution of Contact', fontsize=16, fontweight='bold')


# Bar Plot
bar_palette = "Blues"
sns.countplot(ax=axes[1],
              data=df,
              x='contact',
              hue='y',
              palette=bar_palette,
              order=bp_counts.index)

axes[1].set_title('Subscription of bank term deposit by Contact', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Contact', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].legend(title='Subscription for Bank Term Deposit')

plt.suptitle('Contact', fontsize=20, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 7))
bp_counts = df['poutcome'].value_counts()
pie_colors = ['#336699', '#a3cde3', '#AFEEEE']

# pie chart
axes[0].pie(bp_counts,
            labels=bp_counts.index,
            autopct='%.2f%%',
            startangle=90,
            colors=pie_colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})

axes[0].set_title('Distribution of poutcome', fontsize=16, fontweight='bold')


# Bar Plot
bar_palette = "Blues"
sns.countplot(ax=axes[1],
              data=df,
              x='poutcome',
              hue='y',
              palette=bar_palette,
              order=bp_counts.index)

axes[1].set_title('Subscription of bank term deposit by poutcome', fontsize=16, fontweight='bold')
axes[1].set_xlabel('poutcome', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].legend(title='Subscription for Bank Term Deposit')

plt.suptitle('poutcome', fontsize=20, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


target_counts = df['y'].value_counts()

colors = ['#336699', '#a3cde3'] 
if target_counts.index[0] == 'Yes':
    colors = colors[::-1]

plt.figure(figsize=(10, 6))

plt.pie(target_counts,
        labels=target_counts.index,
        autopct='%.2f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 14, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 3})

center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)

total_patients = len(df)
plt.text(0, 0, f'Total\n{total_patients}\nPatients',
         ha='center', va='center', fontsize=20, fontweight='bold')


plt.title('Target Variable Distribution', fontsize=18, fontweight='bold')
plt.axis('equal') 
plt.show()


numerical_features = [
    'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'
]

fig, axes = plt.subplots(
    nrows=len(numerical_features),
    ncols=2,
    figsize=(12, 3 * len(numerical_features))
)

fig.suptitle('KDE and Q-Q Plots for Numerical Features', fontsize=18, y=1.0)

for i, col in enumerate(numerical_features):
    # --- KDE Plot (Left) ---
    ax_kde = axes[i, 0]
    sns.kdeplot(data=df, x=col, fill=True, ax=ax_kde, color='dodgerblue')
    ax_kde.set_title(f'KDE: {col}', fontsize=12)
    ax_kde.set_xlabel('')
    ax_kde.set_ylabel('Density')

    # --- Q-Q Plot (Right) ---
    ax_qq = axes[i, 1]
    stats.probplot(df[col].dropna(), dist="norm", plot=ax_qq)
    ax_qq.set_title(f'Q-Q Plot: {col}', fontsize=12)
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Sample Quantiles')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()


palette = {0: '#a3cde3', 1: '#336699'}
mosaic = """
    ABC
    .D.
    EFG
    """
fig, axes = plt.subplot_mosaic(mosaic, figsize=(18, 14), constrained_layout=True)
for ax_key, col in zip(axes.keys(), numerical_features):
    ax = axes[ax_key]
    sns.boxplot(ax=ax,
                data=df,
                x='y',
                y=col,
                palette=palette)
    
    ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('subscription for bank term deposit', fontsize=12)
    ax.set_ylabel(col, fontsize=12)

fig.suptitle('Numerical Feature Distributions by Target', fontsize=22, fontweight='bold')
plt.show()


# Define transformers and columns
binary_map = {'yes': 1, 'no': 0}
binary_cols = ['default', 'housing', 'loan']
one_hot_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
yeo_cols = ['balance', 'duration', 'campaign', 'pdays', 'previous']
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# --- 1. Winsorization function ---
winsor_limits = {
    'balance': (0.01, 0.01),
    'duration': (0.01, 0.01),
    'campaign': (0.005, 0.005),
    'pdays': (0.005, 0.005),
    'previous': (0.005, 0.005)
}

def apply_winsor(X_df):
    X = X_df.copy()
    for col, (low, high) in winsor_limits.items():
        X[col] = winsorize(X[col], limits=(low, high))
    return X

winsorizer = FunctionTransformer(apply_winsor)

# --- 2. Binary mapping function ---
def map_binary(X_df):
    X = X_df.copy()
    for col in binary_cols:
        X[col] = X[col].map(binary_map)
    return X

binary_mapper = FunctionTransformer(map_binary)

# Set up ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    # Binary columns (already mapped to 0/1)
    ('binary', binary_mapper, binary_cols),

    # Numerical features that need winsorization, Yeo-Johnson, scaling
    ('yeo_num', Pipeline([
        ('winsor', winsorizer),
        ('yeo', PowerTransformer(method='yeo-johnson')),
        ('scale', RobustScaler())
    ]), yeo_cols),

    # Other numerical features (no Yeo, no winsor)
    ('num_rest', RobustScaler(), list(set(numerical_cols) - set(yeo_cols))),

    # One-hot encoding categorical features
    ('onehot', OneHotEncoder(drop='first', sparse=False, dtype=int, handle_unknown='ignore'), one_hot_cols)
])


X = df.drop(columns='y')
y = df['y']


X_transformed = preprocessor.fit_transform(X)

# Save column names for test-time DataFrame creation
ohe_cols = preprocessor.named_transformers_['onehot'].get_feature_names_out(one_hot_cols)
all_columns = (
    binary_cols +
    yeo_cols +
    list(set(numerical_cols) - set(yeo_cols)) +
    list(ohe_cols)
)
X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns, index=X.index)


X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)


def build_improved_nn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

model = build_improved_nn(X_train.shape[1])


early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
test_df.head()


test_df = test_df.drop(columns='id')
X_test_transformed = preprocessor.transform(test_df)


y_test_proba = model.predict(X_test_transformed).flatten()
original_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.DataFrame({
    'id': original_test['id'],
    'y': y_test_proba
})
submission.to_csv('submission_nn.csv', index=False)


df_lgbm = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df_lgbm = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


from sklearn.preprocessing import LabelEncoder
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                          'loan', 'contact', 'month', 'poutcome']

for i in categorical_features:
    le = LabelEncoder()
    df_lgbm[i] = le.fit_transform(df_lgbm[i])
    test_df_lgbm[i] = le.transform(test_df_lgbm[i])


df_lgbm.drop(columns=['id'],inplace=True)


df_lgbm.head()


X_lgbm = df_lgbm.drop(columns=['y'])
y_lgbm = df_lgbm['y']


X_train_lgbm, X_test_lgbm, y_train_lgbm, y_test_lgbm = train_test_split(
    X_lgbm, y_lgbm, test_size=0.2, random_state=42, stratify=y
)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbosity': -1,
    'random_state': 42,
    'n_estimators': 1000,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X_train_lgbm))
test_predictions = np.zeros(len(X_test_lgbm))
best_iterations = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train_lgbm, y_train_lgbm)):
    X_train_fold, y_train_fold = X_train_lgbm.iloc[train_index], y_train_lgbm.iloc[train_index]
    X_val_fold, y_val_fold = X_train_lgbm.iloc[val_index], y_train_lgbm.iloc[val_index]
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        valid_names=['validation'],
        callbacks=callbacks
    )
    
    best_iterations.append(model.best_iteration)
    
    val_preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_preds_fold = model.predict(X_test_lgbm, num_iteration=model.best_iteration)
    
    oof_predictions[val_index] = val_preds
    test_predictions += test_preds_fold / 5

overall_auc = roc_auc_score(y_train_lgbm, oof_predictions)
print(f"\nOverall Cross-Validation AUC: {overall_auc:.5f}")

optimal_estimators = int(np.mean(best_iterations))
lgb_params['n_estimators'] = optimal_estimators

final_train_data = lgb.Dataset(X_train_lgbm, label=y_train_lgbm)

model_lgbm = lgb.train(
    lgb_params,
    final_train_data
)


y_true_labels = y_test_lgbm
model_probabilities = model_lgbm.predict(X_test_lgbm)
predicted_labels = (model_probabilities > 0.5).astype(int)


report_lgbm = classification_report(y_true_labels, predicted_labels)
print("Classification Report:\n")
print(report_lgbm)


cm_lgbm = confusion_matrix(y_true_labels, predicted_labels)
sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'], 
            yticklabels=['Actual 0', 'Actual 1'])

plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()


test_df_lgbm.drop(columns=['id'],inplace=True)


y_test_proba_lgbm = model_lgbm.predict(test_df_lgbm)
submission = pd.DataFrame({
    'id': original_test['id'],
    'y': y_test_proba_lgbm
})
submission.to_csv('submission_lgbm.csv', index=False)


weight_lgbm = 0.6
weight_nn = 0.4
final_predictions_60_40 = (weight_lgbm * y_test_proba_lgbm) + (weight_nn * y_test_proba)
final_submission = pd.DataFrame({'id': original_test['id'], 
                                 'pred': final_predictions_60_40})

final_submission.to_csv('ensembled_submission.csv', index=False)

