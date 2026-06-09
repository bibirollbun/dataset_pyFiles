import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


print("LightGBM Approach")
print("=" * 60)

train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

target = 'BeatsPerMinute'

print(f"Train dataset shape: {train.shape}")
print(f"Test dataset shape: {test.shape}")

print(f"Missing values in train: {train.isnull().sum().sum()}")
print(f"Missing values in test: {test.isnull().sum().sum()}")


X = train.drop(['id', target], axis=1)
y = train[target]
X_test = test.drop('id', axis=1)

print(f"Using {X.shape[1]} features")
print("Features:", list(X.columns))


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
models = []
fold_metrics = []


print(f"Starting {n_splits}-Fold Cross Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}/{n_splits}")
    print("-" * 30)
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"Training: {X_train.shape[0]:,} samples")
    print(f"Validation: {X_val.shape[0]:,} samples")
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    y_pred_val = lgb_model.predict(X_val)
    oof_predictions[val_idx] = y_pred_val
    
    test_preds = lgb_model.predict(X_test)
    test_predictions += test_preds / n_splits
    
    models.append(lgb_model)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    
    fold_metrics.append({
        'fold': fold + 1,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_iteration': lgb_model.best_iteration_
    })
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Best iteration: {lgb_model.best_iteration_}")


oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
oof_mae = mean_absolute_error(y, oof_predictions)
oof_r2 = r2_score(y, oof_predictions)

print("\n" + "="*50)
print("FINAL OUT-OF-FOLD RESULTS")
print("="*50)
print(f"OOF RMSE: {oof_rmse:.4f}")
print(f"OOF MAE: {oof_mae:.4f}")
print(f"OOF R²: {oof_r2:.4f}")
print("="*50)

print("\nFold-wise Results:")
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df.round(4))
print(f"Mean RMSE across folds: {metrics_df['rmse'].mean():.4f} (±{metrics_df['rmse'].std():.4f})")


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.mean([model.feature_importances_ for model in models], axis=0)
}).sort_values('importance', ascending=False)

print(f"\nTop Feature Importance:")
for i, (idx, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.1f}")


fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[0,0])
axes[0,0].set_title('Feature Importance')

axes[0,1].scatter(y, oof_predictions, alpha=0.5)
axes[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[0,1].set_xlabel('Actual Values')
axes[0,1].set_ylabel('Predicted Values')
axes[0,1].set_title('Actual vs Predicted')

residuals = y - oof_predictions
axes[1,0].scatter(oof_predictions, residuals, alpha=0.5)
axes[1,0].axhline(y=0, color='r', linestyle='--')
axes[1,0].set_xlabel('Predicted Values')
axes[1,0].set_ylabel('Residuals')
axes[1,0].set_title('Residual Plot')

axes[1,1].bar(range(1, n_splits+1), metrics_df['rmse'])
axes[1,1].set_xlabel('Fold')
axes[1,1].set_ylabel('RMSE')
axes[1,1].set_title('RMSE by Fold')
axes[1,1].set_xticks(range(1, n_splits+1))

plt.tight_layout()
plt.show()


def post_process_predictions(predictions, method='snap_curve'):
    
    preds = predictions.copy()
    min_pred = np.min(preds)
    max_pred = np.max(preds)
    mean_pred = np.mean(preds)
    
    print(f"Before processing - Min: {min_pred:.3f}, Max: {max_pred:.3f}, Mean: {mean_pred:.3f}")
    
    if method == 'coordinate_values':
        for i in range(len(preds)):
            if preds[i] < (min_pred + 7):
                preds[i] = preds[i] - 0.6
            if preds[i] > (max_pred - 9):
                preds[i] = preds[i] + 0.5
                
    elif method == 'snap_curve':
        guide = mean_pred
        for i in range(len(preds)):
            per_guide = (preds[i] + guide) / 2
            if preds[i] <= guide:
                preds[i] = (preds[i] * 1.10) - (per_guide * 0.10)
            else:
                preds[i] = (preds[i] * 1.00) - (per_guide * 0.00)
                
    elif method == 'power_coordinate':
        for i in range(len(preds)):
            if preds[i] < (min_pred + 7):
                preds[i] = preds[i] ** 0.993
            if preds[i] > (max_pred - 9):
                preds[i] = preds[i] ** 1.007
    
    min_pred_after = np.min(preds)
    max_pred_after = np.max(preds)
    mean_pred_after = np.mean(preds)
    
    print(f"After processing  - Min: {min_pred_after:.3f}, Max: {max_pred_after:.3f}, Mean: {mean_pred_after:.3f}")
    
    return preds


print(f"\nApplying Post-Processing Methods...")
print("=" * 50)

print("Method 1: Coordinate Values")
processed_1 = post_process_predictions(test_predictions, 'coordinate_values')
print("\nMethod 2: Snap to Curve")
processed_2 = post_process_predictions(test_predictions, 'snap_curve') 
print("\nMethod 3: Power Coordinate")
processed_3 = post_process_predictions(test_predictions, 'power_coordinate')

final_predictions = (processed_1 * 0.4 + processed_2 * 0.4 + processed_3 * 0.2)

print(f"\nFinal blended predictions:")
print(f"Min: {final_predictions.min():.3f}")
print(f"Max: {final_predictions.max():.3f}")
print(f"Mean: {final_predictions.mean():.3f}")


print(f"\nCreating submissions...")

submission_original = pd.DataFrame({
    'id': test['id'],
    target: test_predictions
})
submission_original.to_csv('submission_original.csv', index=False)

submission_processed = pd.DataFrame({
    'id': test['id'],
    target: final_predictions
})
submission_processed.to_csv('submission_postprocessed.csv', index=False)

print(f"Original submission saved as 'submission_original.csv'")
print(f"Post-processed submission saved as 'submission_postprocessed.csv'")

print(f"\nSample predictions comparison:")
comparison = pd.DataFrame({
    'id': test['id'][:10],
    'original': test_predictions[:10],
    'processed': final_predictions[:10],
    'difference': final_predictions[:10] - test_predictions[:10]
})
print(comparison.round(6))


print(f"\nTraining final model on all data...")
avg_best_iteration = int(np.mean([m.best_iteration_ for m in models]))
print(f"Using average best iteration: {avg_best_iteration}")

final_model = lgb.LGBMRegressor(
    n_estimators=avg_best_iteration,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

final_model.fit(X, y)
final_predictions = final_model.predict(X_test)

final_submission = pd.DataFrame({
    'id': test['id'],
    target: final_predictions
})

final_submission.to_csv('final_model_submission.csv', index=False)
print(f"Alternative submission saved as 'final_model_submission.csv'")


print(f"\nKey Insights:")
print(f"LightGBM achieved {oof_rmse:.4f} RMSE")
print(f"Top feature: {feature_importance.iloc[0]['feature']}")
print(f"Stable performance across folds (std: {metrics_df['rmse'].std():.4f})")

