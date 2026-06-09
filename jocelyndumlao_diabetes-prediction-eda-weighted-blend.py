import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# Load the datasets
df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



df_train.head().style.background_gradient(cmap='gist_rainbow')


df_test.head().style.background_gradient(cmap='gist_rainbow')


df_train.describe().style.background_gradient(cmap='tab20c')


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Initialize the figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Diabetes Prediction: Train vs Test & Key Drivers', fontsize=24, weight='bold', y=0.96)

# --- Plot 1: Target Distribution (Train Only) ---
# Goal: Check class imbalance
sns.countplot(data=df_train, x='diagnosed_diabetes', palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Target Distribution (Train Set)', fontsize=16)
axes[0, 0].set_xlabel('Diagnosed Diabetes (0=No, 1=Yes)', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
for container in axes[0, 0].containers:
    axes[0, 0].bar_label(container, fmt='%.0f', fontsize=12)

# --- Plot 2: Numerical Feature Distribution (BMI) - Train vs Test ---
# Goal: Ensure BMI distribution is consistent between datasets
sns.kdeplot(df_train['bmi'], fill=True, label='Train', color='#3498db', ax=axes[0, 1])
sns.kdeplot(df_test['bmi'], fill=True, label='Test', color='#e74c3c', ax=axes[0, 1])
axes[0, 1].set_title('Distribution Comparison: BMI', fontsize=16)
axes[0, 1].set_xlabel('BMI', fontsize=12)
axes[0, 1].legend()

# --- Plot 3: Categorical Feature (Gender) - Train vs Test ---
# Goal: Check demographic consistency
# Prepare data for plotting
train_gender = df_train['gender'].value_counts(normalize=True).reset_index()
train_gender['Set'] = 'Train'
test_gender = df_test['gender'].value_counts(normalize=True).reset_index()
test_gender['Set'] = 'Test'
gender_comp = pd.concat([train_gender, test_gender])

sns.barplot(data=gender_comp, x='gender', y='proportion', hue='Set', palette=['#3498db', '#e74c3c'], ax=axes[1, 0])
axes[1, 0].set_title('Gender Proportion: Train vs Test', fontsize=16)
axes[1, 0].set_ylabel('Proportion', fontsize=12)

# --- Plot 4: Correlation Heatmap (Top Correlations to Target) ---
# Goal: See what physically drives the diagnosis
# Select numerical columns only
numeric_df = df_train.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()[['diagnosed_diabetes']].sort_values(by='diagnosed_diabetes', ascending=False)
# Remove the target itself from the top of the list for cleaner view
corr = corr.drop('diagnosed_diabetes')

sns.heatmap(corr.head(10), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 1], linewidths=1)
axes[1, 1].set_title('Top 10 Features Correlated with Diabetes', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



def analyze_and_blend(weight_dict, output_path="submission.csv"):
    print("--- 1. Loading Submissions ---")
    paths = list(weight_dict.keys())
    weights = list(weight_dict.values())
    
    # Load files
    df1 = pd.read_csv(paths[0]).set_index("id")
    df2 = pd.read_csv(paths[1]).set_index("id")

    # --- ADDED: CORRELATION CHECK ---
    # This tells you how similar your two models are.
    # If correlation is > 0.95, they are very similar.
    correlation = df1['diagnosed_diabetes'].corr(df2['diagnosed_diabetes'])
    print(f"ğŸ“Š Correlation between Model A and Model B: {correlation:.4f}")
    
    if correlation > 0.99:
        print("   âš ï¸� Note: Models are extremely similar. Blending may have minimal effect.")
    else:
        print("   âœ… Models differ slightly. Blending should smooth predictions.")

    # --- BLENDING LOGIC ---
    print("\n--- 2. Applying Weighted Blend ---")
    total_weight = sum(weights)
    
    # Vectorized calculation (Fast & Safe)
    blended_pred = (df1['diagnosed_diabetes'] * weights[0] + 
                    df2['diagnosed_diabetes'] * weights[1]) / total_weight
    
    df_base = df1.copy()
    df_base['diagnosed_diabetes'] = blended_pred
    
    # --- VISUALIZE THE FINAL PREDICTIONS ---
    # This shows you if your model is confident (peaks at 0 and 1) or uncertain
    plt.figure(figsize=(10, 4))
    sns.histplot(df_base['diagnosed_diabetes'], bins=100, kde=True, color='purple')
    plt.title("Distribution of Final Predicted Probabilities")
    plt.xlabel("Probability of Diabetes (0 to 1)")
    plt.show()

    # Save
    df_base = df_base.reset_index()
    df_base.to_csv(output_path, index=False)
    print(f"âœ… Blended submission saved to: {output_path}")
    
    return df_base.head()

# --- Execution ---
if __name__ == "__main__":
    weights = {
        "/kaggle/input/diabetes-prediction-vault/submission.csv": 2.2,
        "/kaggle/input/diabetes-prediction-vault/submission (1).csv": 0.01,
    }
    
    head = analyze_and_blend(weights)
    print("\n--- Final Submission Head ---")
    print(head)




