import time 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec
import seaborn as sns
import squarify
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from scipy.stats import skew
from IPython.display import display

import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
original_data = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')

# Verify shapes
print("Train Data Shape:", train_data.shape)
print("\nOriginal Data Shape:", original_data.shape)
print("\nTest Data Shape:", test_data.shape)


# Display few rows of each dataset
print("Train Data Preview:")
display(train_data.tail())

print("\nOriginal Data Preview:")
display(original_data.head())

print("\nTest Data Preview:")
display(test_data.head())


# Display information about the DataFrames
print("Train Data Info:")
train_data.info()

print("\nOriginal Data Info:")
original_data.info()

print("\nTest Data Info:")
test_data.info()


# Rename 'Gender' to 'Sex' and 'User_ID' to 'id' in original data for consistency
original_data = original_data.rename(columns={"Gender": "Sex", "User_ID": "id"})



def describe_and_style(df, name):
    desc = df.drop(columns=['id'], errors='ignore').describe().T
    print(f"\n{name} Describe:")
    display(desc.style.background_gradient(cmap='PuOr'))

describe_and_style(train_data, "Train Data")
describe_and_style(original_data, "Original Data")
describe_and_style(test_data, "Test Data")



def get_sex_distribution(data, dataset_name):
    count = data['Sex'].value_counts()
    total = count.sum()
    most_frequent = count.idxmax()
    freq = count.max()
    percentage = round(freq / total * 100, 2)

    print(f"{dataset_name}:")
    print(f"  Total Entries: {total}")
    print(f"  Unique Values: {data['Sex'].nunique()}")
    print(f"  Most Frequent: {most_frequent}")
    print(f"  Frequency: {freq}")
    print(f"  Percentage: {percentage}%\n")

get_sex_distribution(train_data, "Train Dataset")
get_sex_distribution(original_data, "Original Dataset")
get_sex_distribution(test_data, "Test Dataset")



def missing_values_report(df, dataset_name):
    missing_count = df.isnull().sum().sum()
    rows = len(df)
    
    print("=" * 40)
    print(f"{dataset_name} Missing Value Analysis")
    print("=" * 40)
    
    if missing_count == 0:
        print(f"âœ… No missing values detected in {rows:,} rows")
    else:
        print(f"âš ï¸�  {missing_count} missing values found in {rows:,} rows")

datasets = {
    "Training Data": train_data,
    "Test Data": test_data,
    "Original Data": original_data
}

for name, data in datasets.items():
    missing_values_report(data, name)
    print()  



def check_duplicates_report(df, dataset_name):
    duplicates_count = df.duplicated().sum()
    total_rows = len(df)
    
    print("=" * 40)
    print(f"ğŸ”� {dataset_name} Duplicate Analysis")
    print("=" * 40)
    
    if duplicates_count == 0:
        print(f"âœ… No duplicates found in {total_rows:,} rows")
    else:
        print(f"âš ï¸�  {duplicates_count} duplicates found ({duplicates_count/total_rows:.2%})")
        print(f"    Total rows affected: {duplicates_count:,}/{total_rows:,}")

datasets = {
    "Training Data": train_data,
    "Test Data": test_data,
    "Original Data": original_data
}

duplicate_summary = {}
for name, data in datasets.items():
    check_duplicates_report(data, name)
    duplicate_summary[name] = {
        'duplicates': data.duplicated().sum(),
        'total_rows': len(data)
    }
    print()



def visualize_calories_distribution(train_data, original_data):
    plt.figure(figsize=(16, 5))
    cmap = cm.get_cmap("PuOr")
    plt.rcParams['font.size'] = 10  

    plt.subplot(1, 3, 1)
    hist_kwargs = {'bins': 30, 'kde': True, 'alpha': 0.4}
    sns.histplot(train_data['Calories'], color=cmap(0.8), label='Train Data', **hist_kwargs)
    sns.histplot(original_data['Calories'], color=cmap(0.1), label='Original Data', **hist_kwargs)
    plt.title('Calories Distribution: Train vs Original', fontsize=12)
    plt.xlabel('Calories', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.legend()
    plt.grid(True, color='gray', linestyle=':', alpha=0.7)

    plt.subplot(1, 3, 2)
    train_kde = sns.kdeplot(train_data['Calories'], color=cmap(0.8), linewidth=2, label='Train Data')
    original_kde = sns.kdeplot(original_data['Calories'], color=cmap(0.1), linewidth=2, label='Original Data')
    
    train_line = train_kde.lines[0]
    original_line = original_kde.lines[1]
    x_train, y_train = train_line.get_data()
    x_orig, y_orig = original_line.get_data()
    
    def plot_peaks(x, y, color, label):
        peaks, _ = find_peaks(y)
        for p in peaks:
            plt.plot(x[p], y[p], 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=1)
            plt.text(x[p], y[p]*1.05, f'{x[p]:.1f}', color=color, ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        return x[peaks], y[peaks]
    
    train_peaks_x, train_peaks_y = plot_peaks(x_train, y_train, cmap(0.8), 'Train')
    orig_peaks_x, orig_peaks_y = plot_peaks(x_orig, y_orig, cmap(0.1), 'Original')
    
    plt.title('KDE Comparison with Peak Markers', fontsize=12)
    plt.xlabel('Calories', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.legend()
    plt.grid(True, color='gray', linestyle=':', alpha=0.7)

    plt.subplot(1, 3, 3)
    combined = pd.concat([
        train_data[['Calories']].assign(Source='Train'),
        original_data[['Calories']].assign(Source='Original')
    ])
    box = sns.boxplot(x='Source', y='Calories', data=combined, palette=[cmap(0.8), cmap(0.1)], width=0.5, linewidth=1.5)
    for i, artist in enumerate(box.artists):
        artist.set_edgecolor(cmap(0.8 if i==0 else 0.1))
        for j in range(6*i, 6*(i+1)):  
            box.lines[j].set_color(cmap(0.8 if i==0 else 0.1))
    plt.title('Calories Spread Comparison', fontsize=12)
    plt.xlabel('Dataset', fontsize=11)
    plt.ylabel('Calories', fontsize=11)
    plt.grid(True, color='gray', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

    print("Train Data KDE Peak Points (Calories):", np.round(train_peaks_x, 1))
    print("Original Data KDE Peak Points (Calories):", np.round(orig_peaks_x, 1))

visualize_calories_distribution(train_data, original_data)



# Define numerical features to visualize
numerical_features = ['Age' ,'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


def visualize_numerical_features(train_data, original_data, test_data, features):
    cmap = plt.get_cmap('PuOr')
    dataset_colors = [cmap(0.8), cmap(0.1), cmap(0.3)]
    dataset_names = ['Train', 'Original', 'Test']
    
    fig, axes = plt.subplots(len(features), 2, figsize=(14, len(features)*4.5))
    
    for i, feature in enumerate(features):
        for data, color, label in zip(
            [train_data, original_data, test_data],
            dataset_colors,
            dataset_names
        ):
            sns.histplot(data[feature], color=color, label=label,
                        bins=20, kde=True, alpha=0.7, ax=axes[i,0])
        
        axes[i,0].set_title(f'{feature} Distribution Comparison', fontsize=13)
        axes[i,0].legend()
        axes[i,0].grid(True, color='gray', linestyle=':', alpha=0.7)
        
        combined = pd.concat([
            train_data[[feature]].assign(Source='Train'),
            original_data[[feature]].assign(Source='Original'),
            test_data[[feature]].assign(Source='Test')
        ])
        
        sns.boxplot(x='Source', y=feature, data=combined,
                  palette=dataset_colors, width=0.6, ax=axes[i,1])
        
        axes[i,1].set_title(f'{feature} Spread Comparison', fontsize=13)
        axes[i,1].grid(True, color='gray', linestyle=':', alpha=0.5)
        
        for j, box in enumerate(axes[i,1].artists):
            box.set_edgecolor(dataset_colors[j])
            for k in range(6*j, 6*(j+1)): 
                axes[i,1].lines[k].set_color(dataset_colors[j])

    plt.tight_layout()
    plt.show()


visualize_numerical_features(train_data, original_data, test_data, numerical_features)



def check_skewness(data, dataset_name, highlight=True, sort=True):
    skewness_dict = {}
    for feature in data.select_dtypes(include=[np.number]).columns:
        skew = data[feature].skew(skipna=True)
        skewness_dict[feature] = skew

    skew_df = pd.DataFrame.from_dict(skewness_dict, orient='index', columns=['Skewness'])
    if sort:
        skew_df = skew_df.reindex(skew_df['Skewness'].abs().sort_values(ascending=False).index)
    
    print(f"\nğŸ”� Skewness for {dataset_name}:")
    print("-"*55)
    print(f"{'Feature':<18} | {'Skewness':<10} | {'Remark'}")
    print("-"*55)
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
            print(f"{color}{feature:<18} | {skew:>+9.4f} | {remark}{endc}")
        else:
            print(f"{feature:<18} | {skew:>+9.4f} | {remark}")
    print("-"*55)
    return skew_df

skew_original = check_skewness(original_data, "Original Data")
skew_train = check_skewness(train_data, "Train Data")
skew_test = check_skewness(test_data, "Test Data")



skew_original['Dataset'] = 'Original Data'
skew_train['Dataset'] = 'Train Data'
skew_test['Dataset'] = 'Test Data'

skew_all = pd.concat([skew_original, skew_train, skew_test]).reset_index()
skew_all.rename(columns={'index': 'Feature'}, inplace=True)

features = skew_all['Feature'].unique()
n_features = len(features)
palette_colors = sns.color_palette("PuOr", n_colors=n_features)
feature_color_map = dict(zip(features, palette_colors))

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, dataset_name in zip(axes, ['Original Data', 'Train Data', 'Test Data']):
    df = skew_all[skew_all['Dataset'] == dataset_name]
    colors = df['Feature'].map(feature_color_map)
    df.plot.barh(x='Feature', y='Skewness', ax=ax, color=colors, legend=False)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title(f"Skewness in {dataset_name}")
    ax.set_xlabel("Skewness")
    ax.grid(True, color='gray', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()



def plot_sex_distribution_across_datasets(train_data, original_data, test_data):
    custom_palette = sns.color_palette("PuOr", 2)
    dataset_names = ['Train', 'Original', 'Test']
    datasets = [train_data, original_data, test_data]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        order = data['Sex'].value_counts().index
        sns.countplot(y='Sex', data=data, ax=axes[0, i], palette=custom_palette, order=order)
        axes[0, i].set_title(f'{name} Data: Sex Counts')
        axes[0, i].set_xlabel('Count')
        axes[0, i].set_ylabel('Sex')
        for p in axes[0, i].patches:
            axes[0, i].annotate(f'{int(p.get_width())}', 
                                (p.get_width(), p.get_y() + p.get_height() / 2), 
                                ha='left', va='center', 
                                color='black', fontsize=12)
        axes[0, i].set_axisbelow(True)
        axes[0, i].grid(axis='x', color='gray', linestyle=':', linewidth=0.7)
        sns.despine(left=True, bottom=True, ax=axes[0, i])

    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        sex_counts = data['Sex'].value_counts()
        wedges, texts, autotexts = axes[1, i].pie(
            sex_counts, 
            labels=sex_counts.index, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=custom_palette,
            textprops={'fontsize': 12}
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[1, i].add_artist(centre_circle)
        axes[1, i].set_title(f'{name} Data: Sex Distribution (%)')
        axes[1, i].axis('equal')  

    plt.tight_layout()
    plt.show()


plot_sex_distribution_across_datasets(train_data, original_data, test_data)



numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
custom_palette = sns.color_palette("PuOr", 2)

def plot_numerical_by_sex(train_data, features, palette):
    for feature in features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(
            data=train_data, x=feature, hue='Sex', multiple='stack',
            bins=20, palette=palette, ax=axes[0], kde=True, alpha=0.7
        )
        axes[0].set_title(f'{feature} Distribution by Sex')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Frequency')
        axes[0].set_axisbelow(True)
        axes[0].grid(axis='y', color='gray', linestyle=':', linewidth=0.7)

        sns.boxplot(
            x='Sex', y=feature, data=train_data, palette=palette, ax=axes[1]
        )
        axes[1].set_title(f'Box Plot of {feature} by Sex')
        axes[1].set_xlabel('Sex')
        axes[1].set_ylabel(feature)
        axes[1].set_axisbelow(True)
        axes[1].grid(axis='y', color='gray', linestyle=':', linewidth=0.7)
        plt.tight_layout()
        plt.show()

plot_numerical_by_sex(train_data, numerical_features, custom_palette)



avg_by_sex = train_data.groupby('Sex')[numerical_features].mean().round(2)
# print(avg_by_sex)

plt.figure(figsize=(12, 5))
sns.heatmap(avg_by_sex, annot=True, cmap='PuOr', fmt='.2f', cbar_kws={'label': 'Average Value'})
plt.title('Average Numerical Feature Values by Sex (Train Data)')
plt.ylabel('Sex')
plt.xlabel('Feature')
plt.tight_layout()
plt.show()



colors = sns.color_palette("PuOr", n_colors=4)  
sns.pairplot(train_data,
             vars=['Duration', 'Heart_Rate', 'Body_Temp', 'Calories'],
             kind='scatter',
             diag_kind='kde',
             plot_kws={'color': colors[0]},
             diag_kws={'color': colors[0]})
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()



fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(train_data['Duration'], train_data['Heart_Rate'], train_data['Calories'],
                c=train_data['Body_Temp'], cmap='PuOr', alpha=0.6)
fig.colorbar(sc, ax=ax, label='Body Temp')
ax.set_xlabel('Duration')
ax.set_ylabel('Heart Rate')
ax.set_zlabel('Calories')
plt.title('3D Scatter: Duration, Heart Rate, Calories (color=Body Temp)')
plt.show()



numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
custom_palette = sns.color_palette("PuOr", 2)

def scatter_features_vs_calories_by_sex_subplot(train_data, features, palette):
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 16))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.scatterplot(
            data=train_data,
            x=feature,
            y='Calories',
            hue='Sex',
            palette=palette,
            alpha=0.5,
            edgecolor='w',
            s=40,
            ax=ax
        )
        ax.set_title(f'{feature} vs Calories by Sex')
        ax.set_xlabel(feature)
        ax.set_ylabel('Calories')
        ax.grid(axis='both', linestyle=':', alpha=0.7)
        if idx == 0:
            ax.legend(title='Sex')
        else:
            ax.get_legend().remove()

    for idx in range(len(features), n_rows * n_cols):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

scatter_features_vs_calories_by_sex_subplot(train_data, numerical_features, custom_palette)



all_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
test_features = [f for f in all_features if f in test_data.columns]

datasets = {
    "Train Data": train_data,
    "Original Data": original_data,
    "Test Data": test_data
}

fig, axes = plt.subplots(ncols=3, figsize=(20, 6))  
axs = axes.ravel() 

for i, (name, df) in enumerate(datasets.items()):
    available_features = [f for f in all_features if f in df.columns]
    df_subset = df[available_features]
    corr = df_subset.corr()
    
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="PuOr",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": .7},
        ax=axs[i]
    )
    axs[i].set_title(f'Correlation Heatmap of {name}', fontsize=14)

plt.tight_layout()
plt.show()



# Concatenate train_data and original_data
combined_data = pd.concat([train_data, original_data], ignore_index=True)

print(f"Combined dataset shape: {combined_data.shape}")


numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

def add_cross_terms(df, features):
    df = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            col_name = f"{features[i]}_x_{features[j]}"
            df[col_name] = df[features[i]] * df[features[j]]
    return df

combined_data = add_cross_terms(combined_data, numerical_features)
test_data = add_cross_terms(test_data, numerical_features)


le = LabelEncoder()
combined_data['Sex'] = le.fit_transform(combined_data['Sex'])
test_data['Sex'] = le.transform(test_data['Sex'])

combined_data['Sex'] = combined_data['Sex'].astype('category')
test_data['Sex'] = test_data['Sex'].astype('category')


combined_data_types = pd.DataFrame({
    'Column Name': combined_data.columns,
    'Combined Data Type': combined_data.dtypes.astype(str)  
})

test_data_types = pd.DataFrame({
    'Column Name': test_data.columns,
    'Test Data Type': test_data.dtypes.astype(str)
})

data_types_comparison = pd.merge(
    combined_data_types,
    test_data_types,
    on='Column Name',
    how='outer'  
)

print("Data Types Comparison of Combined and Test Datasets:\n")
display(data_types_comparison)


colors = sns.color_palette('PuOr', 2)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(combined_data['Calories'], bins=50, kde=True, ax=axes[0], color=colors[0])
axes[0].axvline(combined_data['Calories'].mean(), color='red', linestyle='-', linewidth=2, label='Mean')
axes[0].axvline(combined_data['Calories'].median(), color='blue', linestyle='--', linewidth=2, label='Median')
axes[0].set_title('Original Calories Distribution')
axes[0].set_xlabel('Calories')
axes[0].grid(axis='y', linestyle=':', alpha=0.7)
axes[0].legend()

log_calories = np.log1p(combined_data['Calories'])
sns.histplot(log_calories, bins=50, kde=True, ax=axes[1], color=colors[1])
axes[1].axvline(log_calories.mean(), color='red', linestyle='-', linewidth=2, label='Mean')
axes[1].axvline(np.median(log_calories), color='blue', linestyle='--', linewidth=2, label='Median')
axes[1].set_title('Log-Transformed Calories Distribution')
axes[1].set_xlabel('Log(Calories + 1)')
axes[1].grid(axis='y', linestyle=':', alpha=0.7)
axes[1].legend()

plt.tight_layout()
plt.show()



# Select features and target
features = [col for col in combined_data.columns if col not in ['id', 'Calories']]
target = 'Calories'



# Set Training and Test Dataset
X = combined_data[features]
y = np.log1p(combined_data[target]) # log1p transform
X_test = test_data[features]

cat_features = ['Sex']



best_params = {
    'learning_rate': 0.10888450868862715,
    'depth': 4,
    'l2_leaf_reg': 7.159552441895846,
    'border_count': 45,
    'bagging_temperature': 0.40906797780457893,
    'random_strength': 0.3606368619503747,
    'leaf_estimation_iterations': 18,
    'grow_policy': 'Depthwise',
    'iterations': 2000,
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'eval_metric': 'RMSE',
    'verbose': 100
}



FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

results = {
    'oof': np.zeros(len(X)),
    'pred': np.zeros(len(X_test)),
    'rmsle': [],
    'train_times': []
}

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nğŸ“‚ Fold {fold + 1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(**best_params, cat_features=cat_features)

    start = time.time()
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    elapsed = time.time() - start

    oof_preds = model.predict(X_valid)
    test_preds = model.predict(X_test)

    results['oof'][valid_idx] = oof_preds
    results['pred'] += test_preds / FOLDS
    results['rmsle'].append(np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_preds))))
    results['train_times'].append(elapsed)

    print(f"âœ… Fold RMSLE: {results['rmsle'][-1]:.4f} | Time: {elapsed:.1f}s")



mean_rmsle = np.mean(results['rmsle'])
std_rmsle = np.std(results['rmsle'])
print(f"\nğŸ“Œ Mean RMSLE: {mean_rmsle:.4f} Â± {std_rmsle:.4f}")
print(f"Avg Train Time: {np.mean(results['train_times']):.2f}s")



feature_importance = model.get_feature_importance()
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
display(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="PuOr")
plt.title("CatBoost Feature Importance")
plt.grid(axis='both', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



# Final predictions
y_preds = np.expm1(results['pred'])  # Reverse log1p
y_preds = np.clip(y_preds, 1, 314)   # Clip to valid range

submission['Calories'] = y_preds
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved!")



actual = np.expm1(y)
predicted = np.expm1(results['oof'])

plt.figure(figsize=(10, 6))
puor_palette = sns.color_palette("PuOr", 3)
sns.scatterplot(x=actual, y=predicted, alpha=0.3, color=puor_palette[2])
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='Ideal')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories (OOF)")
plt.grid(axis='both', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.kdeplot(actual, label="Actual", fill=True, color=puor_palette[0], alpha=0.7)
sns.kdeplot(predicted, label="Predicted", fill=True, color=puor_palette[2], alpha=0.7)
plt.title("Distribution of Actual vs Predicted Calories")
plt.xlabel("Calories")
plt.legend()
plt.grid(axis='both', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()



print(submission.head())

