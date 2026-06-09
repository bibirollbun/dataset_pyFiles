import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def main():
    # 1. LOAD DATA
    train_feats = pd.read_csv('/kaggle/input/soil-nutrient-gap-prediction-for-sustainable-maize/Soil Nutrient/Train.csv')
    gap_train   = pd.read_csv('/kaggle/input/soil-nutrient-gap-prediction-for-sustainable-maize/Soil Nutrient/Gap_Train.csv').rename(columns={'Required':'Gap'})
    test_feats  = pd.read_csv('/kaggle/input/soil-nutrient-gap-prediction-for-sustainable-maize/Soil Nutrient/Test.csv')
    sample_sub  = pd.read_csv('/kaggle/input/soil-nutrient-gap-prediction-for-sustainable-maize/Soil Nutrient/SampleSubmission.csv')

    # 2. MERGE FEATURES + TARGETS
    df = pd.merge(train_feats,
                  gap_train[['PID','Nutrient','Gap']],
                  on='PID', how='inner')


    df = pd.concat([df, pd.get_dummies(df['Nutrient'], prefix='nutr')], axis=1)

    submission = sample_sub.copy()
    submission[['PID','Nutrient']] = submission['ID'].str.rsplit('_', n=1, expand=True)
    test = pd.merge(submission[['ID','PID','Nutrient']],
                    test_feats,
                    on='PID', how='left')
    test = pd.concat([test, pd.get_dummies(test['Nutrient'], prefix='nutr')], axis=1)

    nutrients = ['N','P','K','Ca','Mg','S','Fe','Mn','Zn','Cu','B']
    drop_cols = ['site','PID','Nutrient','ID','Gap'] + nutrients
    feature_cols = [c for c in df.columns
                    if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]

    X_raw      = df[feature_cols]
    y          = df['Gap'].values
    X_test_raw = test[feature_cols]

    # 4. IMPUTE MISSING VALUES
    imputer = SimpleImputer(strategy='median')
    X      = pd.DataFrame(imputer.fit_transform(X_raw),      columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test_raw),     columns=feature_cols)

    # 5. BASELINE MODEL: Linear Regression + 5‑Fold CV
    kf    = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()

    neg_mse    = cross_val_score(model, X, y,
                                 scoring='neg_mean_squared_error',
                                 cv=kf)
    rmse_scores = np.sqrt(-neg_mse)
    print(f"Baseline CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    # 6. FIT ON FULL TRAIN & PREDICT ON TEST
    model.fit(X, y)
    preds = model.predict(X_test)

    # 7. SAVE SUBMISSION
    submission['Gap'] = preds
    submission[['ID','Gap']].to_csv('BaselineSubmission.csv', index=False)
    print("Saved ▶ BaselineSubmission.csv")

if __name__ == "__main__":
    main()




