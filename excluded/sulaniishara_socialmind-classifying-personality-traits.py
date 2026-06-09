import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import skew
from IPython.display import display

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_data = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
original_data = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')

# Verify shapes
print("Original Data Shape:", original_data.shape)
print("\nTrain Data Shape:", train_data.shape)
print("\nTest Data Shape:", test_data.shape)



# Display few rows of each dataset
print("Original Data Preview:")
display(original_data.head())

print("\nTrain Data Preview:")
display(train_data.tail())

print("\nTest Data Preview:")
display(test_data.head())


# Display information about the DataFrames
print("Original Data Info:")
original_data.info()

print("\nTrain Data Info:")
train_data.info()

print("\nTest Data Info:")
test_data.info()


# Descriptive statistics for numerical columns
print("Original Data Describe:")
display(original_data.describe().T.style.background_gradient(cmap='PRGn'))

print("\nTrain Data Describe:")
display(train_data.describe().T.style.background_gradient(cmap='PRGn'))

print("\nTest Data Describe:")
display(test_data.describe().T.style.background_gradient(cmap='PRGn'))


def print_unique_and_top_values(df, categorical_columns, numerical_columns, dataset_name="Dataset", top_n=3):
    print(f"\n{'='*60}")
    print(f"ğŸ“‹ Unique & Top Frequencies Summary â€” {dataset_name}")
    print(f"{'='*60}")
    
    print("\nğŸŸª Categorical Features:")
    for col in categorical_columns:
        if col not in df.columns:
            print(f"  âš ï¸� Column '{col}' not found.\n")
            continue
        val_counts = df[col].value_counts(dropna=False)
        print(f"  â€¢ {col}")
        print(f"    â”œâ”€ Unique Values: {sorted(df[col].dropna().unique())}")
        print(f"    â””â”€ Most Frequent: '{val_counts.idxmax()}' (Count: {val_counts.max()})\n")

    print("\nğŸŸ© Numerical Features:")
    for col in numerical_columns:
        if col not in df.columns:
            print(f"  âš ï¸� Column '{col}' not found.\n")
            continue
        top_vals = df[col].value_counts(dropna=True).head(top_n)
        print(f"  â€¢ {col}")
        print(f"    â”œâ”€ Unique Count: {df[col].nunique(dropna=True)}")
        print(f"    â””â”€ Top {top_n} Values:")
        for val, count in top_vals.items():
            print(f"       â†’ {val} (Count: {count})")
        print()

cat_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']

# Test dataset has no 'Personality'
test_cat_cols = [col for col in cat_cols if col != 'Personality']


print_unique_and_top_values(original_data, cat_cols, num_cols, "Original Dataset")
print_unique_and_top_values(train_data, cat_cols, num_cols, "Train Dataset")
print_unique_and_top_values(test_data, test_cat_cols, num_cols, "Test Dataset")



def plot_missing_values_heatmap(df, dataset_name="Dataset"):
    plt.figure(figsize=(12, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap='PRGn_r', yticklabels=False)
    plt.title(f"Missing Values Heatmap â€” {dataset_name}", fontsize=14)
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.show()

plot_missing_values_heatmap(original_data, "Original Dataset")
plot_missing_values_heatmap(train_data, "Train Dataset")
plot_missing_values_heatmap(test_data, "Test Dataset")



def missing_values_summary(df, dataset_name="Dataset"):
    missing_count = df.isnull().sum()
    missing_pct = 100 * missing_count / len(df)
    data_types = df.dtypes
    summary_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_pct.round(2),
        "Dtype": data_types
    }).sort_values(by="Missing %", ascending=False)
    
    print(f"\n{'='*60}")
    print(f"ğŸ”� Missing Values Report â€” {dataset_name}")
    print(f"{'='*60}")
    print(f"Total missing values: {missing_count.sum()}\n")
    display(summary_df)
    return summary_df

def check_duplicates(df, dataset_name="Dataset"):
    dup_count = df.duplicated().sum()
    print(f"\n{'='*60}")
    print(f"ğŸ§¬ Duplicate Rows Report â€” {dataset_name}")
    print(f"{'='*60}")
    print(f"Total duplicate rows: {dup_count}")
    if dup_count > 0:
        print(f"Showing sample duplicates:")
        display(df[df.duplicated()].head())
    return dup_count

train_missing_summary = missing_values_summary(train_data, "Train Dataset")
train_duplicates = check_duplicates(train_data, "Train Dataset")

test_missing_summary = missing_values_summary(test_data, "Test Dataset")
test_duplicates = check_duplicates(test_data, "Test Dataset")

original_missing_summary = missing_values_summary(original_data, "Original Dataset")
original_duplicates = check_duplicates(original_data, "Original Dataset")



target_variable = 'Personality'

print(f"Target variable: {target_variable}")
print(f"Data type: {train_data[target_variable].dtype}")


def plot_target_distribution(data, dataset_name, target_variable='Personality', palette_name='PRGn'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    sns.countplot(y=target_variable, data=data, ax=ax, palette=palette_name)
    ax.set_title(f'Count Plot of {target_variable} in {dataset_name}', pad=20)
    ax.set_ylabel(target_variable)
    ax.set_xlabel('Count')
    ax.grid(axis='x', color='gray', linestyle=':', linewidth=0.7)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

    max_count = data[target_variable].value_counts().max()
    for p in ax.patches:
        width = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.text(width + max_count * 0.01, y,
                f'{int(width)}',
                ha='left', va='center', fontsize=10, fontweight='bold', color='black')

    ax = axes[1]
    counts = data[target_variable].value_counts().sort_index()
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette(palette_name, len(counts)),
        wedgeprops=dict(width=0.4, edgecolor='w'),
        radius=1.2
    )
    for text in texts + autotexts:
        text.set_fontsize(10)
        text.set_fontweight('bold')

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title(f'{target_variable} Distribution in {dataset_name}', pad=25)
    ax.axis('equal')

    plt.tight_layout()
    plt.show()

plot_target_distribution(train_data, 'Train Data')
plot_target_distribution(original_data, 'Original Data')



numerical_features = [
    'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
    'Friends_circle_size', 'Post_frequency'
]

cmap = plt.get_cmap('PRGn')
colors = [cmap(0.8), cmap(0.3), cmap(0)]

fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, len(numerical_features) * 4))

for i, feature in enumerate(numerical_features):
    sns.histplot(train_data[feature], color=colors[0], label='Train Data', bins=20, kde=True, ax=axes[i, 0], alpha=0.6)
    sns.histplot(test_data[feature], color=colors[1], label='Test Data', bins=20, kde=True, ax=axes[i, 0], alpha=0.6)
    sns.histplot(original_data[feature], color=colors[2], label='Original Data', bins=20, kde=True, ax=axes[i, 0], alpha=0.6)
    axes[i, 0].set_title(f'Histogram of {feature}', fontsize=13)
    axes[i, 0].legend()
    axes[i, 0].grid(color='gray', linestyle=':', linewidth=0.7)

    sns.boxplot(
        data=[train_data[feature].dropna(), test_data[feature].dropna(), original_data[feature].dropna()],
        palette=colors, orient='h', ax=axes[i, 1]
    )
    axes[i, 1].set_title(f'Horizontal Boxplot of {feature}', fontsize=13)
    axes[i, 1].set_yticklabels(['Train Data', 'Test Data', 'Original Data'])
    axes[i, 1].grid(axis='x', color='gray', linestyle=':', linewidth=0.7)

plt.tight_layout()
plt.show()



def check_skewness(data, dataset_name, highlight=True, sort=True):
    skew_vals = {feature: data[feature].skew(skipna=True) for feature in data.select_dtypes(include=np.number).columns}
    skew_df = pd.DataFrame.from_dict(skew_vals, orient='index', columns=['Skewness'])

    if sort:
        skew_df = skew_df.reindex(skew_df['Skewness'].abs().sort_values(ascending=False).index)

    print(f"\nğŸ“� Skewness Summary â€” {dataset_name}")
    print("-"*65)
    print(f"{'Feature':<25} | {'Skewness':<10} | {'Remark'}")
    print("-"*65)

    for feature, row in skew_df.iterrows():
        skew = row['Skewness']
        abs_skew = abs(skew)
        if abs_skew > 1:
            remark = "Highly skewed"
            color = '\033[91m'
        elif abs_skew > 0.5:
            remark = "Moderately skewed"
            color = '\033[93m'
        else:
            remark = "Approximately symmetric"
            color = ''
        endc = '\033[0m' if color else ''
        if highlight and color:
            print(f"{color}{feature:<25} | {skew:>+9.4f} | {remark}{endc}")
        else:
            print(f"{feature:<25} | {skew:>+9.4f} | {remark}")
    print("-"*65)
    return skew_df

skew_train = check_skewness(train_data, "Train Dataset")
skew_test = check_skewness(test_data, "Test Dataset")
skew_original = check_skewness(original_data, "Original Dataset")



def plot_correlation_heatmaps(train_data, original_data, numerical_features):
    cmap = sns.diverging_palette(145, 280, s=85, l=25, as_cmap=True)
    
    corr_train = train_data[numerical_features].corr()
    corr_original = original_data[numerical_features].corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(
        corr_train,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 11},
        ax=axes[0]
    )
    axes[0].set_title('Train Data Correlation Heatmap', fontsize=14, pad=15)
    
    sns.heatmap(
        corr_original,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        annot_kws={"size": 11},
        ax=axes[1]
    )
    axes[1].set_title('Original Data Correlation Heatmap', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.show()

plot_correlation_heatmaps(train_data, original_data, numerical_features)



def plot_categorical_distribution_across_datasets(train_data, original_data, test_data, feature):
    custom_palette = sns.color_palette("PRGn", n_colors=train_data[feature].nunique())
    dataset_names = ['Train', 'Original', 'Test']
    datasets = [train_data, original_data, test_data]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        order = data[feature].value_counts().index
        sns.countplot(y=feature, data=data, ax=axes[0, i], palette=custom_palette, order=order)
        axes[0, i].set_title(f'{name} Data: {feature} Counts')
        axes[0, i].set_xlabel('Count')
        axes[0, i].set_ylabel(feature)

        for p in axes[0, i].patches:
            axes[0, i].annotate(
                f'{int(p.get_width())}',
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center',
                color='black', fontsize=12
            )
        axes[0, i].grid(axis='x', color='gray', linestyle=':', linewidth=0.7)
        sns.despine(left=True, bottom=True, ax=axes[0, i])

    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        counts = data[feature].value_counts()
        wedges, texts, autotexts = axes[1, i].pie(
            counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=custom_palette,
            textprops={'fontsize': 12}
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[1, i].add_artist(centre_circle)
        axes[1, i].set_title(f'{name} Data: {feature} Distribution (%)')
        axes[1, i].axis('equal')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()

categorical_features = ['Stage_fear', 'Drained_after_socializing']

for feature in categorical_features:
    plot_categorical_distribution_across_datasets(train_data, original_data, test_data, feature)



def plot_discrete_numerical_counts(data, numerical_features, target_variable='Personality', palette_name='PRGn', aspect=1.5):
    palette = sns.color_palette(palette_name, 2)
    
    for feature in numerical_features:
        plt.figure(figsize=(12, 5))
        ax = sns.countplot(
            data=data,
            x=feature,
            hue=target_variable,
            palette=palette
        )
        ax.set_title(f'Countplot of {feature} by {target_variable}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        plt.grid(axis='y', linestyle=':', linewidth=0.7)
        
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

plot_discrete_numerical_counts(train_data, numerical_features)


def plot_and_describe_by_target(data, numerical_features, target_variable, palette):
    grouped = data.groupby(target_variable)
    
    for feature in numerical_features:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        sns.histplot(
            data=data, x=feature, hue=target_variable, multiple='stack',
            bins=20, palette=palette, ax=axes[0], alpha=0.7
        )
        axes[0].set_title(f'Histogram of {feature} by {target_variable}')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', linestyle=':', linewidth=0.7)
        
        sns.kdeplot(
            data=data, x=feature, hue=target_variable, palette=palette,
            fill=True, common_norm=False, alpha=0.5, ax=axes[1]
        )
        axes[1].set_title(f'KDE Plot of {feature} by {target_variable}')
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel('Density')
        axes[1].grid(axis='y', linestyle=':', linewidth=0.7)
        
        sns.boxplot(
            x=target_variable, y=feature, data=data, palette=palette, ax=axes[2]
        )
        axes[2].set_title(f'Boxplot of {feature} by {target_variable}')
        axes[2].set_xlabel(target_variable)
        axes[2].set_ylabel(feature)
        axes[2].grid(axis='y', linestyle=':', linewidth=0.7)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{'='*60}")
        print(f"ğŸ“‹ Descriptive Statistics for '{feature}' by '{target_variable}'")
        print(f"{'='*60}")
        
        desc = grouped[feature].describe()
        desc.loc[:, 'count'] = desc.loc[:, 'count'].map('{:,.0f}'.format)
        desc.loc[:, ['mean', 'std', 'min', '25%', '50%', '75%', 'max']] = desc.loc[:, ['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(3)
        
        display(desc)
        print("-" * 60)

palette = sns.color_palette('PRGn', 2)
plot_and_describe_by_target(train_data, numerical_features, target_variable, palette)



def plot_categorical_features_by_personality(df, categorical_features, target_variable='Personality', palette_name='PRGn'):
    palette = sns.color_palette(palette_name, 2)
    unique_personalities = df[target_variable].unique()
    num_personalities = len(unique_personalities)
    
    for feature in categorical_features:
        fig, axes = plt.subplots(num_personalities, 2, figsize=(12, 4 * num_personalities))
        
        if num_personalities == 1:
            axes = [axes]
        
        for i, personality in enumerate(unique_personalities):
            filtered_data = df[df[target_variable] == personality]
            
            sns.countplot(
                y=feature,
                data=filtered_data,
                ax=axes[i][0],
                palette=palette
            )
            axes[i][0].set_title(f'{feature} Counts for {personality}')
            axes[i][0].set_xlabel('Count')
            axes[i][0].set_ylabel(feature)
            axes[i][0].tick_params(axis='y', labelsize=9)
            
            for p in axes[i][0].patches:
                axes[i][0].annotate(
                    f'{int(p.get_width())}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center',
                    fontsize=10,
                    fontweight='bold',
                    color='black'
                )
            axes[i][0].grid(axis='x', linestyle=':', linewidth=0.7)
            sns.despine(left=True, bottom=True, ax=axes[i][0])
            
            counts = filtered_data[feature].value_counts()
            wedges, texts, autotexts = axes[i][1].pie(
                counts,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=palette
            )
            axes[i][1].set_title(f'{feature} Distribution for {personality}')
            axes[i][1].axis('equal')
            
            for text in texts + autotexts:
                text.set_fontsize(10)
                text.set_fontweight('bold')
        
        plt.tight_layout()
        plt.show()

plot_categorical_features_by_personality(train_data, categorical_features)



def analyze_missing_by_personality_dynamic(df, cols, target_variable='Personality', dataset_name="Dataset"):
    print(f"\n{'='*65}")
    print(f"ğŸ”� Missing Value Analysis by Personality â€” {dataset_name}")
    print(f"{'='*65}")

    summary = (
        df.groupby(target_variable)[cols]
        .apply(lambda x: x.isnull().mean() * 100)
        .T.round(2)
        .rename(columns=lambda x: f"{x} % Missing")
    )

    if set(summary.columns) == {'Introvert', 'Extrovert'}:
        summary = summary.rename(columns={
            'Introvert': 'Introvert % Missing',
            'Extrovert': 'Extrovert % Missing'
        })

    if 'Introvert % Missing' in summary.columns and 'Extrovert % Missing' in summary.columns:
        summary['Difference (Ext - Intro)'] = (
            summary['Extrovert % Missing'] - summary['Introvert % Missing']
        ).round(2)

    print(f"\nğŸ“‹ Missing Percentage Summary Table:\n")
    display(summary)

    plot_df = (
        summary
        .reset_index()
        .rename(columns={'index': 'Feature'})
        .melt(id_vars='Feature', var_name='Group', value_name='Missing %')
    )

    cmap = sns.color_palette("PRGn", as_cmap=True)
    custom_colors = [cmap(0.3), cmap(0.8), cmap(0)]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x='Feature', y='Missing %', hue='Group', palette=custom_colors)
    # ax.set_facecolor('lightgray')

    plt.title(f"{dataset_name}: Missing Value % by Personality Class", fontsize=12, pad=12)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Percentage Missing", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle=':', linewidth=0.5)
    plt.legend(title='Personality Class')
    plt.tight_layout()
    plt.show()

    return summary

missing_cols = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]

train_missing_summary = analyze_missing_by_personality_dynamic(train_data, missing_cols, dataset_name="Train Dataset")
original_missing_summary = analyze_missing_by_personality_dynamic(original_data, missing_cols, dataset_name="Original Dataset")



df = train_data.copy()
test_df = test_data.copy()
test_ids = test_df['id']

# Drop ID column
df.drop(columns='id', inplace=True)
test_df.drop(columns='id', inplace=True)

# Define feature types
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']



# Impute numerical columns
for col in numerical_cols:
    median = df[col].median()
    df[col].fillna(median, inplace=True)
    test_df[col].fillna(median, inplace=True)
    print(f"Imputed missing values in {col} with median: {median:.2f}")


# Impute categorical columns
for col in categorical_cols:
    df[col].fillna('missing', inplace=True)
    test_df[col].fillna('missing', inplace=True)
    print(f"Filled missing values in {col} with 'missing'")


# Check for remaining missing values
print("Remaining Missing Values After Preprocessing (Train):")
display(df.isnull().sum())
print("Remaining Missing Values After Preprocessing (Test):")
display(test_df.isnull().sum())



le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])

# Preview label encoding
print("Label Encoding Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))


X = df.drop(columns='Personality')
y = df['Personality']
cat_features = [X.columns.get_loc(col) for col in categorical_cols]


# Use Best Parameters Directly (from previous Optuna run)
best_params = {
    'iterations': 955, 
    'learning_rate': 0.03709315617443212, 
    'depth': 7, 
    'l2_leaf_reg': 5.809195194498532, 
    'random_strength': 0.5283229905136648, 
    'bagging_temperature': 0.35787045457512345
}

def create_model():
    return CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="Accuracy",
        cat_features=cat_features,
        random_seed=42,
        task_type="GPU",
        devices="0,1",
        verbose=0
    )


N_SPLITS = 7

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
models = []
cv_scores = []
roc_auc_scores = []
oof_preds = np.zeros(len(X))
all_val_probas = []

plt.figure(figsize=(10, 8))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*40}")
    print(f"ğŸ”� Fold {fold+1}/{N_SPLITS}")
    print(f"{'-'*40}")
    train_pool = Pool(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_features)
    valid_pool = Pool(X.iloc[valid_idx], y.iloc[valid_idx], cat_features=cat_features)

    model = create_model()
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, verbose=False)
    models.append(model)

    valid_preds = model.predict(X.iloc[valid_idx])
    valid_proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
    all_val_probas.extend(valid_proba)
    oof_preds[valid_idx] = valid_preds

    acc = accuracy_score(y.iloc[valid_idx], valid_preds)
    auc = roc_auc_score(y.iloc[valid_idx], valid_proba)
    cv_scores.append(acc)
    roc_auc_scores.append(auc)

    fpr, tpr, _ = roc_curve(y.iloc[valid_idx], valid_proba)
    plt.plot(fpr, tpr, label=f"Fold {fold+1} (AUC = {auc:.3f})")

    best_iteration = model.get_best_iteration()
    best_val_score = model.get_best_score()['validation']['Accuracy']

    print(f"Best Test Accuracy = {best_val_score}")
    print(f"bestIteration = {best_iteration}")
    print(f"Shrink model to first {best_iteration + 1} iterations.")
    print(f"âœ… Fold {fold+1} Accuracy: {acc:.4f}")
    print(f"ğŸ§  Fold {fold+1} ROC AUC: {auc:.4f}")
    print(f"{'='*40}")

# Cross-Validation Summary
print("\n" + "="*40)
print("ğŸ“‹ Cross-Validation Summary")
print("="*40)
print(f"Mean CV Accuracy : {np.mean(cv_scores):.4f} Â± {np.std(cv_scores):.4f}")
print(f"Mean ROC AUC     : {np.mean(roc_auc_scores):.4f} Â± {np.std(roc_auc_scores):.4f}\n")

# ROC Curve Across Folds
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve Across Folds")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models[-1].get_feature_importance()
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 5))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='PRGn')
plt.title("Feature Importance (Last Fold Model)")
plt.grid(color='gray', linestyle=':', linewidth=0.7)
plt.tight_layout()
plt.show()


print("\nğŸ“„ Classification Report on OOF:")
print(classification_report(y, oof_preds.round(), target_names=le.classes_))
sns.heatmap(confusion_matrix(y, oof_preds.round()), annot=True, fmt='d', cmap='PRGn',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("OOF Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


all_test_probs = np.zeros(len(test_df))
test_votes = np.zeros((len(test_df), len(models)), dtype=int)

for i, model in enumerate(models):
    test_pred = model.predict(test_df).astype(int)
    test_proba = model.predict_proba(test_df)[:, 1]
    test_votes[:, i] = test_pred
    all_test_probs += test_proba / len(models)

final_preds = np.where(test_votes.sum(axis=1) >= len(models) / 2, 1, 0)
final_labels = le.inverse_transform(final_preds)


submission = pd.DataFrame({
    'id': test_ids,
    'Personality': final_labels
})
submission.to_csv("submission.csv", index=False)
print("\nğŸ“¤ Submission file saved!")
display(submission.head())

# Prediction Distribution
print("\nğŸ”� Prediction Counts on Test Set:")
print(submission['Personality'].value_counts())



cmap = plt.get_cmap('PRGn')
colors = [cmap(0), cmap(0.3), cmap(0.8)]

plt.figure(figsize=(12, 5))
sns.histplot(all_test_probs, bins=50, kde=True, color=colors[0])
plt.title("Distribution of Predicted Probabilities (Test Set)")
plt.xlabel("Probability of Introvert")
plt.ylabel("Frequency")
plt.grid(color='gray', linestyle=':', linewidth=0.7)
plt.tight_layout()
plt.show()


