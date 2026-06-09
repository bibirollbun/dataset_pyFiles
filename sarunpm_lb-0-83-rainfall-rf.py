import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def train_data_preprocessor(df, target_column):
    # Handle missing values (if any)
    df = df.fillna(0)

    # Encode categorical features (if any)
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop(['id',target_column], axis=1)
    y = df[target_column]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def test_data_preprocessor(df):
    # Handle missing values (if any)
    df = df.fillna(0)
    df = df.drop(['id'], axis=1)

    # Encode categorical features (if any)
    df = pd.get_dummies(df, drop_first=True)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    return X


def chistplot(df, columns, nrows=1, ncols=1):
    # Only take the first nrows * ncols columns if there are more than that
    num_cols = nrows * ncols
    columns = columns[:num_cols]
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for i, cname in enumerate(columns):
        if nrows == 1 and ncols == 1:
            ax = axes
        elif nrows == 1 or ncols == 1:
            ax = axes[i]
        else:
            ax = axes[i // ncols, i % ncols]
        
        sns.histplot(data=df, x=cname, kde=True, palette='viridis', ax=ax)
        
        ax.set_xlabel(cname)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {cname}')
        
    # Hide any unused subplots
    for j in range(num_cols, nrows * ncols):
        if nrows == 1 or ncols == 1:
            fig.delaxes(axes[j])
        else:
            fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


def cross_validate_xgb(X, y, n_splits=6):
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    aucs = []
    test_preds = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
       
        # Train the model
        model.fit(X_train, y_train)

        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        test_preds.append(model.predict_proba(test)[:, 1])

        # Calculate AUC-ROC
        auc = roc_auc_score(y_test, y_pred_prob)
        aucs.append(auc)
        print(f"Fold AUC-ROC: {auc}")
        
        # Plot ROC curve
        #fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        #plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.4f})")
        #plt.plot([0, 1], [0, 1], linestyle='--', color='r')
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Characteristic (ROC) Curve')
        #plt.legend(loc="lower right")
        #plt.show()
    #print(test_preds)
    #print(aucs)

    mean_auc = np.mean(aucs)
    print(f"{'-'*50}\nMean AUC-ROC: {mean_auc}\n{'-'*50}")

    # Create submission DataFrame
    submission_oof = pd.DataFrame({
        'id': submission_raw.id,  
        'rainfall': np.mean(test_preds, axis=0)
    })

    # Save submission DataFrame to CSV
    submission_oof.to_csv(f'submission_oof.csv', index=False)

    return model


ROOTS = '/kaggle/input/playground-series-s5e3'

train_dataset = ROOTS + '/' + 'train.csv'
test_dataset= ROOTS + '/' + 'test.csv'
submission_dataset = ROOTS + '/' + 'sample_submission.csv'

train_raw = pd.read_csv(train_dataset)
test_raw = pd.read_csv(test_dataset)
submission_raw = pd.read_csv(submission_dataset)

train_df = train_raw

test_df = test_raw


print('The dimension of the train dataset is:', train_df.shape)
print('The dimension of the test dataset is:', test_df.shape)


train_df.head()


train_df.info()


train_df.isna().sum()


chistplot(train_df, train_df.columns[1:],4,3)


# Preprocess the training data
X, y = train_data_preprocessor(train_df, 'rainfall')


# Preprocess the test data
test = test_data_preprocessor(test_df)


# Perform cross-validation and obtain the trained model
model = cross_validate_xgb(X, y)


# Predict probabilities for the test data
y_pred_prob_test = model.predict_proba(test)[:, 1]


# Create submission DataFrame
submission = pd.DataFrame({
    'id': submission_raw.id,  
    'rainfall': y_pred_prob_test
})

# Save submission DataFrame to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
print(submission.shape)


submission.head()


submission_oof = pd.read_csv('/kaggle/working/submission_oof.csv')
# Create submission files DataFrame for analysis
join_df = pd.DataFrame({
    'id': submission_raw.id,  
    'oof': submission_oof['rainfall'],
    'sub': submission['rainfall'],
    
})


join_df


chistplot(join_df, join_df.columns[1:],1,2)

