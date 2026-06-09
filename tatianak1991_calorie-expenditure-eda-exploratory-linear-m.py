import numpy as np 
import pandas as pd 
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_log_error
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.display.float_format = '{:.2f}'.format
sns.set()


train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col = 'id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')


train_sample = train_data.sample(n=1000)


g = sns.pairplot(train_sample, hue=('Sex'), diag_kind="hist")
g.map_lower(sns.kdeplot, levels=4, color=".2")


for cal in ['Duration', 'Heart_Rate', 'Body_Temp']:
    g = sns.lmplot(
        data=train_sample, x=cal, y="Calories",
        col="Sex", hue='Sex', height=5,
        facet_kws=dict(sharex=False, sharey=True), fit_reg=True, lowess=True,
        scatter_kws=dict(alpha = 0.3), 
    )
    g.fig.suptitle(f'{cal} vs Calories', y=1.08)


train_sample['log_calories'] = np.log(train_sample['Calories'])


for cal in ['Duration', 'Heart_Rate', 'Body_Temp']:
    train_sample[f'log_{cal}'] = np.log(train_sample[cal])
    g = sns.lmplot(
        data=train_sample, x=f'log_{cal}', y="log_calories",
        col="Sex", hue='Sex', height=5,
        facet_kws=dict(sharex=False, sharey=True), fit_reg=True,
        scatter_kws=dict(alpha = 0.3), 
    )
    g.fig.suptitle(f'Log-Transformed Relationship: {cal} vs Calories', y=1.08)



ig, axes = plt.subplots(4, 2, figsize=(10, 15), sharey=False)
axes = axes.flatten()

for i, col in enumerate(train_data.columns):
    if col == 'Sex':
        sns.histplot(data=train_data, x=col, ax=axes[i], shrink=.8)
    else: 
        sns.kdeplot(data=train_data, x=col, ax=axes[i], hue="Sex", fill=True, common_norm=False, alpha=.6, linewidth=0)

plt.tight_layout()
plt.show()


train_data['Sex'] = train_data["Sex"].map({'male': 1, 'female': 0})


plt.figure(figsize=(7, 7))
corr = train_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt ='.2f',  linewidth=.5, cmap='vlag', square=True)
plt.show()


def check_vif(df):
  df = df.select_dtypes(include=['number'])
  vif = pd.DataFrame()
  vif["feature"] = df.columns
  vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
  return vif


# COUNT VIF SCORES
vif_count = check_vif(train_data.drop(['Calories'], axis=1))
vifs = vif_count.sort_values('VIF', ascending=False)
vifs.reset_index(drop=True, inplace=True)



sns.set()
plt.figure(figsize=(8, 4))
ax = plt.gca()
sns.barplot(data=vifs, y='feature', x='VIF', color='blue', ax=ax)

# Add VIF values as text labels on the bars
for i, row in vifs.iterrows():
    ax.text(x=row['VIF']+1, y=i, s=f"{row['VIF']:.1f}", 
            ha='left', va='center', fontsize=10, color='black')

plt.title('Variance Inflation Factor for all Features')
plt.tight_layout()
plt.show()


def count_mi_scores(X, y, entropy_estimate):
    """
    Calculates mutual information scores between features and a continuous target,
    and expresses how much information each feature provides about the target 
    as a percentage of the target's estimated entropy.
    """
    X = X.copy()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    
    # Estimate mutual information between each feature and the target
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    
    # Calculate the percentage of information each feature contributes
    info_percent = 100 * mi_scores / entropy_estimate
    
    mi_df = pd.DataFrame({
        'mi_scores': mi_scores,
        'info_percent': info_percent
    }, index=X.columns).sort_values(by='info_percent', ascending=True)
    
    return mi_df


def plot_mi_scores(df_scores):
    """
    Plots mutual information scores as a horizontal bar chart,
    annotated with the percentage of target information each feature explains.
    """
    df_sorted = df_scores.sort_values(by='mi_scores', ascending=True)
    scores = df_sorted['mi_scores']
    info_percent = df_sorted['info_percent']
    ticks = df_sorted.index
    width = np.arange(len(scores))

    # Plot
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.barh(width, scores, color='skyblue')
    ax.set_yticks(width)
    ax.set_yticklabels(ticks)
    ax.set_title("Proportion of Target Information Captured via Mutual Information", fontsize=14)
    ax.set_xlabel("MI Score")
    
    # Annotate with info_percent
    for i, (score, percent) in enumerate(zip(scores, info_percent)):
        ax.text(x=score + 0.01, y=i, s=f"{percent:.1f}%", 
                ha='left', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()


## KDE-Based Estimation of Differential Entropy from Sampled Continuous Data

# sample_for_entropy_count = train_data['Calories'].sample(n=75000, random_state=42)
# kde = gaussian_kde(sample_for_entropy_count, bw_method='scott')  
# log_probs = np.log(kde.evaluate(sample_for_entropy_count))
# entropy = -np.mean(log_probs)
# entropy


entropy = 5.388935558311368


mi_df = count_mi_scores(train_data.drop(['Calories'], axis=1), train_data['Calories'], entropy_estimate=entropy)


plot_mi_scores(mi_df)


# create log transformed df
log_df = np.log1p(train_data.drop(['Sex'], axis=1))

# Xy split 
X = log_df.drop(['Calories'], axis=1)
y_log = log_df['Calories']
y = train_data['Calories']

# X scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(data= scaler.fit_transform(X), columns=X.columns)
X_scaled['Sex'] = train_data["Sex"]

# PCA (to reduce multicollinearity)
pca = PCA(n_components=6)
X_scaled_pca = pd.DataFrame(data= pca.fit_transform(X_scaled), columns=[f'PC_{i}' for i in range(1, 7)])

X_full = pd.concat([X_scaled, X_scaled_pca], axis=1)


# Find all PC
pca = PCA()
pca.fit(X_scaled)

# Plot cumulative explained variance
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()


# RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# rmse_ scorer
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# RMSLE func
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, a_min=0, a_max=None)))

# rmsle scorer
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# Models
models = {'lr_model': LinearRegression(), 'ridge_model': Ridge(alpha=0.35), 'lasso_model': Lasso(alpha=0.007)}

# KFold setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)


for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_log, cv=kf, scoring=rmse_scorer)    
    print(name)
    # print("RMSE scores for each log1p-transformed data fold:", scores)
    print("Average RMSE:", np.mean(scores)) 


## Train models and save predictions

predictions = {}

# Fit models and generate predictions
for name, model in models.items():
    
    # Train
    model.fit(X_scaled.iloc[:500000, :], y_log.iloc[:500000])
    # Predict 
    y_pred_log = model.predict(X_scaled.iloc[500001:, :]) 

    # Reverse data from y-log and save predictions
    y_pred = np.expm1(y_pred_log)     
    predictions[f'{name}_pred'] = y_pred 
    
    # Calculate RMSLE 
    rmsle_val = rmse(y_log.iloc[500001:],  y_pred_log)
    print(f"Model: {model}")
    print(f"RMSE for log_transf y: {rmsle_val}")

# Create a DataFrame to hold predictions for each model
df_pred = pd.DataFrame(predictions)


for name, model in models.items():

    coefficients = model.coef_
    coef_df = pd.DataFrame({'Feature': X_scaled.columns,'Coefficient': coefficients})
    print(f'\n{name}\n')
    print(coef_df)


## Let's explore residuals from Linear Regression model
df_pred['actual_y'] = train_data['Calories'].iloc[500001:].reset_index(drop=True)
df_pred['regr_residuals'] = df_pred['lr_model_pred'] - df_pred['actual_y']
df_pred['relative_error_%'] = (abs(df_pred['regr_residuals']) / df_pred['actual_y']) * 100


df_pred[['lr_model_pred', 'actual_y', 'regr_residuals', 'relative_error_%']].describe()


df_pred[['lr_model_pred', 'actual_y']].plot(kind='scatter', x='lr_model_pred', y='actual_y',  alpha=0.3, color='grey')
plt.title('Comparison of Actual and Predicted Target Values')


df_pred.regr_residuals.plot(kind='hist', x='lr_model_pred', y='actual_y',  alpha=0.8, color='grey', bins=50)
plt.ylim(0, 100)
plt.title('Residuals')


# Filter subsets
under_preds = df_pred[(df_pred['relative_error_%'] > 25) & (df_pred['lr_model_pred'] < df_pred['actual_y'])]
over_preds  = df_pred[(df_pred['relative_error_%'] > 25) & (df_pred['lr_model_pred'] > df_pred['actual_y'])]

# Plot both on the same axes
plt.figure(figsize=(8,6))
plt.scatter(under_preds['lr_model_pred'], under_preds['actual_y'], alpha=0.3, color='blue', label='Underpredictions')
plt.scatter(over_preds['lr_model_pred'], over_preds['actual_y'], alpha=0.3, color='red', label='Overpredictions')

# Labels and legend
plt.xlabel('Predicted (lr_model_pred)')
plt.ylabel('Actual (actual_y)')
plt.title(f'Predictions with >25% Relative Error\n({((len(under_preds)+len(over_preds))/len(df_pred))*100:.2f}% of all predictions)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


male_X= X_scaled[X_scaled.Sex == 1]
female_X = X_scaled[X_scaled.Sex == 0]

male_y = y_log.loc[male_X.index]
female_y = y_log.loc[female_X.index]

male_model = LinearRegression()
female_model = LinearRegression()


# For Male

# Train
male_model.fit(male_X.iloc[:300000, :], male_y.iloc[:300000])
# Predict 
y_pred_log_male = male_model.predict(male_X.iloc[300001:, :]) 
# Reverse data from y-log and save predictions
y_pred_male = np.expm1(y_pred_log_male)     
   
# Calculate RMSLE 
rmsle_val = rmse(male_y.iloc[300001:],  y_pred_log_male)
print(f"Male Regression Model")
print(f"RMSE for log_transf y: {rmsle_val}")


# For Female


female_model.fit(female_X.iloc[:300000, :], female_y.iloc[:300000])
y_pred_log_female = female_model.predict(female_X.iloc[300001:, :]) 
y_pred_female = np.expm1(y_pred_log_female)     
   
# Calculate RMSLE 
rmsle_val = rmse(female_y.iloc[300001:],  y_pred_log_female)
print(f"Female Regression Model")
print(f"RMSE for log_transf y: {rmsle_val}")


y_actual_male = np.expm1(male_y.iloc[300001:])
y_actual_female = np.expm1(female_y.iloc[300001:])


df_male_pred = pd.DataFrame()
df_male_pred['lr_model_pred'] = y_pred_male
df_male_pred['actual_y'] = y_actual_male.reset_index(drop=True)
df_male_pred['regr_residuals'] = df_male_pred['lr_model_pred'] - df_male_pred['actual_y']
df_male_pred['relative_error_%'] = (abs(df_male_pred['regr_residuals']) / df_male_pred['actual_y']) * 100

df_female_pred = pd.DataFrame()
df_female_pred['lr_model_pred'] = y_pred_female
df_female_pred['actual_y'] = y_actual_female.reset_index(drop=True)
df_female_pred['regr_residuals'] = df_female_pred['lr_model_pred'] - df_female_pred['actual_y']
df_female_pred['relative_error_%'] = (abs(df_female_pred['regr_residuals']) / df_female_pred['actual_y']) * 100


male_under_preds=df_male_pred[(df_male_pred['relative_error_%'] > 25) & (df_male_pred['lr_model_pred'] < df_male_pred['actual_y'])]
male_over_preds =df_male_pred[(df_male_pred['relative_error_%'] > 25) & (df_male_pred['lr_model_pred'] > df_male_pred['actual_y'])]
female_under_preds=df_female_pred[(df_female_pred['relative_error_%'] > 25) & (df_female_pred['lr_model_pred'] < df_female_pred['actual_y'])]
female_over_preds =df_female_pred[(df_female_pred['relative_error_%'] > 25) & (df_female_pred['lr_model_pred'] > df_female_pred['actual_y'])]


fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, sharex=True)

# --- Plot 1: Actual vs Predicted for Males ---
axes[0, 0].scatter(y_pred_male, y_actual_male, alpha=0.4, color='orange')
axes[0, 0].set_title('Actual vs Predicted (Male)\nRMSLE = 0.088')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].grid(True)

# --- Plot 2: Actual vs Predicted for Females ---
axes[0, 1].scatter(y_pred_female, y_actual_female, alpha=0.4, color='blue')
axes[0, 1].set_title('Actual vs Predicted (Female)\nRMSLE = 0.060')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].grid(True)

# --- Plot 3: Male Relative Error > 25% ---
axes[1, 0].scatter(male_under_preds['lr_model_pred'], male_under_preds['actual_y'],
                   alpha=0.4, color='#f5be3d', label='Underpredictions')
axes[1, 0].scatter(male_over_preds['lr_model_pred'], male_over_preds['actual_y'],
                   alpha=0.3, color='#a17a20', label='Overpredictions')
axes[1, 0].set_title(f'Male: Predictions with >25% Relative Error\n({((len(male_under_preds)+len(male_over_preds))/len(df_male_pred))*100:.2f}% of all predictions)')
axes[1, 0].set_xlabel('Predicted (lr_model_pred)')
axes[1, 0].set_ylabel('Actual (actual_y)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# --- Plot 4: Female Relative Error > 25% ---
axes[1, 1].scatter(female_under_preds['lr_model_pred'], female_under_preds['actual_y'],
                   alpha=0.4, color='#3e66f0', label='Underpredictions')
axes[1, 1].scatter(female_over_preds['lr_model_pred'], female_over_preds['actual_y'],
                   alpha=0.3, color='#192b69', label='Overpredictions')
axes[1, 1].set_title(f'Female: Predictions with >25% Relative Error\n({((len(female_under_preds)+len(female_over_preds))/len(df_female_pred))*100:.2f}% of all predictions)')
axes[1, 1].set_xlabel('Predicted (lr_model_pred)')
axes[1, 1].set_ylabel('Actual (actual_y)')
axes[1, 1].legend()
axes[1, 1].grid(True)

for ax in axes.ravel():
    ax.tick_params(labelbottom=True, labelleft=True)

# --- Layout ---
plt.tight_layout()
plt.show()

