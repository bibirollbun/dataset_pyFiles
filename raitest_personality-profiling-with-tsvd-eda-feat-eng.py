import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score
from scipy import sparse

train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train_df.info()
print("-"*35,"Shape(train_df): ",train_df.shape,"\n")

test_df.info()
print("-"*35,"Shape(test_df): ",test_df.shape,"\n")


def missing_value_percentages(df):
    percents = (df.isnull().mean() * 100).round(1)
    return percents.astype(str) + '%'

print("ğŸ“Š Missing Values (%) in train_df:")
print(missing_value_percentages(train_df))

print("\nğŸ“Š Missing Values (%) in test_df:")
print(missing_value_percentages(test_df))


# Loop through each non-null Personality value
for personality_value in train_df['Personality'].dropna().unique():
    print(f"\nğŸ”� Personality: {personality_value}")
    grouped = train_df[train_df['Personality'] == personality_value]
    
    for column in train_df.columns:
        unique_vals = grouped[column].dropna().unique()
        print(f"- {column}: {unique_vals}  (Count: {len(unique_vals)})")


print("ğŸ“‹ Features in train_df:")
print(list(train_df.columns))


def plot_feature_histograms(df, selected_features, hue='Personality', figsize=(11, 5), verbose=False):
    fig, axes = plt.subplots(1, len(selected_features), figsize=figsize)

    for idx, feature in enumerate(selected_features):
        if verbose:
            print(f"\nğŸ”� Checking feature: {feature}")

        try:
            # Try converting unique values to integers
            unique_vals = df[feature].dropna().unique()
            int_bins = sorted([int(val) for val in unique_vals])
            if verbose:
                print(f"âœ… {feature}: convertible to integers â†’ {int_bins}")
        except ValueError as e:
            if verbose:
                print(f"â�Œ {feature}: not fully convertible to integers â†’ {e}")
            int_bins = sorted(df[feature].dropna().unique())

        sns.histplot(data=df,
                     x=feature,
                     hue=hue,
                     multiple='stack',
                     palette='muted',
                     edgecolor='gray',
                     bins=int_bins,
                     discrete=True,
                     ax=axes[idx])

        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Count')
        axes[idx].set_xticks(int_bins)
        axes[idx].set_xticklabels([str(val) for val in int_bins], ha='center')

    plt.tight_layout()
    plt.show()



# plot histograms for train_df selected features

plot_feature_histograms(train_df, ['Stage_fear', 'Drained_after_socializing', 'Going_outside'])


# plot histograms for train_df selected features

plot_feature_histograms(train_df, ['Social_event_attendance', 'Friends_circle_size'])


# plot histograms for train_df selected features

plot_feature_histograms(train_df, ['Time_spent_Alone', 'Post_frequency'])


# Let's create a feature that gives bins to individuals with Introverted / Extroverted or mixed behavior 

def compute_bins(df, column):
    def classify(val):
        if val < 3:
            return 0  # Introvert
        elif val == 3:
            return 1  # Mixed
        else:
            return 2  # Extrovert

    return df[column].apply(classify)


# Applying bins to most polarized features

# bins to 'Going_outside' 
train_df['outside_bins'] = compute_bins(train_df, 'Going_outside')
test_df['outside_bins']  = compute_bins(test_df, 'Going_outside')

train_df.drop(columns='Going_outside', inplace=True)
test_df.drop(columns='Going_outside', inplace=True)


# bins to 'Social_event_attendance' 
train_df['event_bins'] = compute_bins(train_df, 'Social_event_attendance')
test_df['event_bins']  = compute_bins(test_df, 'Social_event_attendance')

train_df.drop(columns='Social_event_attendance', inplace=True)
test_df.drop(columns='Social_event_attendance', inplace=True)


# bins to 'Post_frequency' 
train_df['post_bins'] = compute_bins(train_df, 'Post_frequency')
test_df['post_bins'] = compute_bins(test_df, 'Post_frequency')

train_df.drop(columns='Post_frequency', inplace=True)
test_df.drop(columns='Post_frequency', inplace=True)


print(train_df.head(9))


# Preparing the data for Truncated SVD
# All features must have numerical values
X_train = train_df.drop(columns=['id', 'Personality'], axis=1)
X_train['Stage_fear'].replace({'No': 0, 'Yes': 1}, inplace=True)
X_train['Drained_after_socializing'].replace({'No': 0, 'Yes': 1}, inplace=True)

X_test = test_df.drop(columns=['id'], axis=1)
X_test['Stage_fear'].replace({'No': 0, 'Yes': 1}, inplace=True)
X_test['Drained_after_socializing'].replace({'No': 0, 'Yes': 1}, inplace=True)

# By scaling the data we convert current 0 values into different number
# and later we replace NaNs with zeros before making a sparse matrix.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled[np.isnan(X_scaled)] = 0
X_test_scaled = scaler.transform(X_test)
X_test_scaled[np.isnan(X_test_scaled)] = 0

# Reduce the data to 2 dimensions
n_comp = 2
svd = TruncatedSVD(n_components=n_comp)
X_svd = svd.fit_transform(sparse.csr_matrix(X_scaled))

# Create a DataFrame for the tSVD results
svd_df = pd.DataFrame(data=X_svd, columns=['tSVD1', 'tSVD2'])
svd_df.to_csv('train_SVD_components.csv', index=False)

# Plotting the tSVD results
y = train_df['Personality'].replace({'Extrovert': 0, 'Introvert': 1}).values
target_names = np.unique(y)
colors = ['blue', 'red']

plt.figure(1, figsize=(9, 9))
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(svd_df.values[y == i, 0], svd_df.values[y == i, 1], color=color, alpha=.8, s=10,
                label=target_name, marker='.')
plt.title("tSVD of train data")
plt.xlabel("tSVD component 1 explains %.1f %% of variance" % (svd.explained_variance_ratio_[0] * 100.0))
plt.ylabel("tSVD component 2 explains %.1f %% of variance" % (svd.explained_variance_ratio_[1] * 100.0))
plt.legend(['Extrovert', 'Introvert'], loc='best', shadow=False, scatterpoints=3, markerscale=3.0, prop={'size':14})
plt.xlim(-3.2, 5.3)
plt.ylim(-2.5, 2.8)
plt.tight_layout()
plt.savefig('tSVD_train_data.png', dpi=300)
plt.show()
plt.close()


X_test_svd = svd.transform(sparse.csr_matrix(X_test_scaled))
svd_test_df = pd.DataFrame(data=X_test_svd, columns=['tSVD1', 'tSVD2'])
svd_test_df.to_csv('test_SVD_components.csv', index=False)

plt.figure(2, figsize=(10, 10))
plt.scatter(svd_test_df.values[:,0], svd_test_df.values[:,1], c='k', alpha=.8, s=10, marker='.')
plt.title("tSVD of test data")
plt.xlabel("tSVD component 1")
plt.ylabel("tSVD component 2")
plt.xlim(-3.2, 5.3)
plt.ylim(-2.5, 2.8)
plt.tight_layout()
plt.savefig('tSVD_test_data.png', dpi=300)
plt.show()
plt.close()



X_train['Personality'] = 'Extrovert'
X_train.loc[svd_df['tSVD1'] > 1.0, 'Personality'] = 'Introvert'
print('\n Accuracy with added feature engineering: %.6f' % (accuracy_score(train_df.Personality, X_train.Personality)) )



submission['Personality'] = 'Extrovert'
submission.loc[svd_test_df['tSVD1'] > 1.0, 'Personality'] = 'Introvert'
submission.to_csv('submission.csv', index=False)


