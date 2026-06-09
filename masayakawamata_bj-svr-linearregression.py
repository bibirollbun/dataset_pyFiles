import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv")
test = pd.read_csv("/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv")
print("Train Shape:", train.shape)
print("Test Shape :", test.shape)
train.head(3)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from cuml.svm import SVR, LinearSVR
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time


TARGET = 'ev'
X = train.drop([TARGET, "id"], axis=1).copy()
y = train[TARGET].copy()
X_test = test.drop(columns='id').copy()


models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=10000, alpha=0.001),
    "SVR_RBF": SVR(kernel='rbf', C=0.1, epsilon=0.01, gamma='scale'),
    "SVR_linear": SVR(kernel='linear', C=0.1, epsilon=0.01),
    "SVR_poly": SVR(kernel='poly', C=0.1, epsilon=0.01),
    "SVR_sigmoid": SVR(kernel='sigmoid'),
    "LinearSVR": LinearSVR(max_iter=10000, C=0.1, epsilon=0.01)
}

# Set up 7-fold cross-validation
FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

results = {}

# Evaluate each model using cross-validation with a StandardScaler in the pipeline
for name, model in models.items():
    print(f"Evaluating {name} ...")
    # Create a pipeline that scales the data and applies the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', model)
    ])
    start_time = time.time()
    # Use negative MSE as scoring (will be converted to positive later)
    scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
    elapsed_time = time.time() - start_time
    mse_scores = -scores  # convert negative MSE to positive values
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    
    results[name] = {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "time": elapsed_time
    }
    
    print(f"{name}: Mean MSE = {mean_mse:.8f}, Std = {std_mse:.8f}, Time taken = {elapsed_time:.2f} seconds")
    print("-" * 50)

# Rank models based on the mean MSE (lower is better)
ranking = sorted(results.items(), key=lambda x: x[1]["mean_mse"])
print("\n=== Ranking of Models based on CV Mean MSE ===")
for rank, (model_name, res) in enumerate(ranking, start=1):
    print(f"{rank}. {model_name}: Mean MSE = {res['mean_mse']:.8f}, Time = {res['time']:.2f} sec")


from sklearn.model_selection import cross_val_predict

# Function to plot predictions vs actual values
def plot_predictions_vs_actual(models, X, y, kf):
    # Dictionary to store predictions
    predictions = {}
    
    # Get predictions for each model
    for name, model in models.items():
        print(f"Processing {name} for visualization...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', model)
        ])
        
        # Get predictions using cross-validation
        y_pred = cross_val_predict(pipeline, X, y, cv=kf)
        predictions[name] = y_pred
        
        # Calculate MSE
        mse = mean_squared_error(y, y_pred)
        print(f"{name} MSE: {mse:.8f}")
    
    # Comparison plot of Linear Regression vs SVR
    plt.figure(figsize=(15, 10))
    
    # 1. Scatter plot of actual vs predicted values (Linear Regression vs SVR_RBF)
    plt.subplot(2, 2, 1)
    plt.scatter(y, predictions['LinearRegression'], alpha=0.5, label='Linear Regression')
    plt.scatter(y, predictions['SVR_RBF'], alpha=0.5, label='SVR (RBF)')
    
    # Perfect prediction line
    min_val = min(min(y), min(predictions['LinearRegression']), min(predictions['SVR_RBF']))
    max_val = max(max(y), max(predictions['LinearRegression']), max(predictions['SVR_RBF']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.title('Linear Regression vs SVR Predictions')
    plt.xlabel('Actual EV')
    plt.ylabel('Predicted EV')
    plt.legend()
    plt.grid(True)
    
    # 2. Residual plot (Linear Regression)
    plt.subplot(2, 2, 2)
    residuals_lr = y - predictions['LinearRegression']
    plt.scatter(predictions['LinearRegression'], residuals_lr, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    mean_residual = np.mean(residuals_lr)
    std_residual = np.std(residuals_lr)
    
    plt.title(f'Linear Regression Residuals\nMean: {mean_residual:.8f}, Std: {std_residual:.8f}')
    plt.xlabel('Predicted EV')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    
    # 3. Residual plot (SVR_RBF)
    plt.subplot(2, 2, 3)
    residuals_svr = y - predictions['SVR_RBF']
    plt.scatter(predictions['SVR_RBF'], residuals_svr, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    mean_residual = np.mean(residuals_svr)
    std_residual = np.std(residuals_svr)
    
    plt.title(f'SVR (RBF) Residuals\nMean: {mean_residual:.8f}, Std: {std_residual:.8f}')
    plt.xlabel('Predicted EV')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    
    # 4. Distribution of predictions
    plt.subplot(2, 2, 4)
    sns.kdeplot(predictions['LinearRegression'], label='Linear Regression')
    sns.kdeplot(predictions['SVR_RBF'], label='SVR (RBF)')
    sns.kdeplot(y, label='Actual EV')
    plt.title('Distribution of Predictions vs Actual Values')
    plt.xlabel('Expected Value (EV)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison_plots.png')
    plt.show()
    
    # Analyze feature importance
    plt.figure(figsize=(12, 6))
    
    # Train linear regression model on the full dataset
    lr_model = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', LinearRegression())
    ])
    lr_model.fit(X, y)
    
    # Get coefficients
    coefficients = lr_model.named_steps['estimator'].coef_
    feature_names = X.columns.tolist()
    
    # Sort by absolute coefficient value
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    # Plot coefficients
    plt.subplot(1, 2, 1)
    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=colors)
    plt.title('Feature Importance (Linear Regression Coefficients)')
    plt.grid(True)
    
    # Visualize the effect of each card value
    plt.subplot(1, 2, 2)
    card_values = feature_names
    card_effects = coefficients
    
    # Plot card value effects
    plt.bar(card_values, card_effects, color=['green' if c > 0 else 'red' for c in card_effects])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Effect of Removing Each Card Value on EV')
    plt.xlabel('Card Value')
    plt.ylabel('Effect on EV')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('feature_importance_plots.png')
    plt.show()
    
    return predictions

# Execute the above code
predictions = plot_predictions_vs_actual(models, X, y, kf)

