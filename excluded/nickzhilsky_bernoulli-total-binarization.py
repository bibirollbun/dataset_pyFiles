import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


numerical_cols = (['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency'])
binary_cols = (['Stage_fear', 'Drained_after_socializing'])


def data_analisys(df):
    overview = pd.DataFrame({
        'Missing': df.isnull().sum(),
        'Missing in %': round(df.isnull().mean() * 100, 2),
        'Data type': df.dtypes,
        'Unique': df.nunique()
    })

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    overview['Min'] = df[numeric_cols].min()
    overview['Max'] = df[numeric_cols].max()

    return overview.sort_values(by='Missing', ascending=False)


data_analisys(train)


data_analisys(test)


del train['id']
train[numerical_cols] = train[numerical_cols].fillna(train[numerical_cols].mean()).round().astype(int)
test[numerical_cols] = test[numerical_cols].fillna(test[numerical_cols].mean()).round().astype(int)


def binarize_numerical_columns(df, numerical_cols):
    df_bin = df.copy()
    for col in numerical_cols:
        max_val = int(df[col].max())
        for i in range(max_val):
            bin_col = f"{col}_{i}_{i+1}"
            df_bin[bin_col] = ((df[col] >= i) & (df[col] < i+1)).astype(int)
        df_bin.drop(columns=col, inplace=True)
    return df_bin


train = binarize_numerical_columns(train, numerical_cols)
test = binarize_numerical_columns(test, numerical_cols)


le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])

le_dict = {}


for col in binary_cols:
    le_col = LabelEncoder()
    train[col] = train[col].astype(str)
    train[col] = train[col].fillna(train[col].mode()[0])
    le_col.fit(train[col])
    train[col] = le_col.transform(train[col])
    le_dict[col] = le_col 


X = train.drop(columns='Personality')
y = train['Personality']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = BernoulliNB()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
fold = 1

for train_index, val_index in skf.split(X, y):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    
    acc = accuracy_score(y_val_fold, y_pred_fold)
    accuracies.append(acc)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    fold += 1

print(f"\nMean Accuracy across 10 folds: {np.mean(accuracies):.4f}")


target_encoder = LabelEncoder()
target_encoder.fit(['Introvert', 'Extrovert']) 

for col in binary_cols:
    if col not in test.columns:
        continue 
    le_col = LabelEncoder()
    test[col] = test[col].astype(str)
    test[col] = test[col].fillna(test[col].mode()[0])
    le_col.fit(test[col])
    test[col] = le_col.transform(test[col])
    le_dict[col] = le_col 

X_test = test.drop(columns=['id'])

y_test_pred = model.predict(X_test)

y_test_labels = target_encoder.inverse_transform(y_test_pred)

submission = test[['id']].copy()
submission['Personality'] = y_test_labels

submission.to_csv('submission.csv', index=False)


train_original = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')

plot_df = train_original.copy()
plot_df = plot_df.dropna(subset=binary_cols + ['Personality'])

binary_map = {'yes': 1, 'no': -1}
plot_df['Stage_fear'] = plot_df['Stage_fear'].str.lower().map(binary_map)
plot_df['Drained_after_socializing'] = plot_df['Drained_after_socializing'].str.lower().map(binary_map)
plot_df = plot_df.dropna(subset=['Stage_fear', 'Drained_after_socializing'])

plot_df['Stage_fear_jitter'] = plot_df['Stage_fear'] + np.random.normal(0, 0.1, size=len(plot_df))
plot_df['Drained_jitter'] = plot_df['Drained_after_socializing'] + np.random.normal(0, 0.1, size=len(plot_df))

fig, ax = plt.subplots(figsize=(8, 6))
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

cmap = LinearSegmentedColormap.from_list("grey_grad", ["#D3D3D3", "#404040"])
gradient = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[-1.5, 1.5, -1.5, 1.5], alpha=0.8)

palette = {'Introvert': '#003366', 'Extrovert': '#FFD700'}
sns.scatterplot(
    data=plot_df,
    x='Stage_fear_jitter',
    y='Drained_jitter',
    hue='Personality',
    palette=palette,
    alpha=0.7,
    s=80,
    ax=ax
)

ax.axhline(0, color='white', linestyle='-', lw=1.5)
ax.axvline(0, color='white', linestyle='-', lw=1.5)

ax.set_xlabel('Stage_fear (No = -1, Yes = +1)', color='white')
ax.set_ylabel('Drained_after_socializing (No = -1, Yes = +1)', color='white')
ax.set_title('Personality Separation by Binary Traits', color='white')

ax.grid(True, linestyle='--', alpha=0.3, color='white')

ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

leg = ax.legend(title='Personality', loc='center', bbox_to_anchor=(0.5, 0.5), frameon=True, facecolor='#2f2f2f', edgecolor='white', fontsize=12)
leg.get_title().set_color('white')
for text in leg.get_texts():
    text.set_color('white')

plt.tight_layout()
plt.show()



introvert_mask = (train_original['Stage_fear'] == 'Yes') & (train_original['Drained_after_socializing'] == 'Yes') & (train_original['Personality'] == 'Introvert')
extrovert_mask = (train_original['Stage_fear'] == 'No') & (train_original['Drained_after_socializing'] == 'No') & (train_original['Personality'] == 'Extrovert')
outliers = train_original[~(introvert_mask | extrovert_mask)].copy()


outliers.shape

