import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df = pd.read_csv(f'../../kaggle/input/playground-series-s5e3/train.csv', index_col=False)

df_test = pd.read_csv(f'../../kaggle/input/playground-series-s5e3/test.csv', index_col=False)

print(df.shape)
print(df_test.shape)


df[df.isna().any(axis=1)]


df_test[df_test.isna().any(axis=1)]


df[df.duplicated()]


df_test[df_test.duplicated()]


sns.boxplot(df)


fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 10))

axes = axes.flatten() 

for i in range(df.shape[1]): 
    sns.histplot(df.iloc[:, i], ax=axes[i])  
    axes[i].set_title(f"Histogram {i+1}")

for j in range(df.shape[1], len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = (Q3-Q1) * 1.5

lower_limit = Q1 - IQR
upper_limit = Q3 + IQR

df_cleaned = df[(df['humidity'] > lower_limit['humidity']) & (df['humidity'] < upper_limit['humidity'])]
df_cleaned = df_cleaned[(df_cleaned['temparature'] > lower_limit['temparature']) & (df_cleaned['temparature'] < upper_limit['temparature'])]
df_cleaned = df_cleaned[(df_cleaned['mintemp'] > lower_limit['mintemp']) & (df_cleaned['mintemp'] < upper_limit['mintemp'])]
df_cleaned = df_cleaned[(df_cleaned['pressure'] > lower_limit['pressure']) & (df_cleaned['pressure'] < upper_limit['pressure'])]
df_cleaned = df_cleaned[(df_cleaned['dewpoint'] > lower_limit['dewpoint']) & (df_cleaned['dewpoint'] < upper_limit['dewpoint'])]


df_cleaned.shape


plt.figure(figsize=(8, 6))

sns.heatmap(df_cleaned.corr(), annot=True, cmap="coolwarm", fmt=".2f")



df_cleaned.drop(columns=['id'], inplace=True)
df_outid=df_test.drop(columns=['id'])

print(df_cleaned.shape)
print(df_outid.shape)


corr_matrix = df_cleaned.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]

print("Highly correlated features to be removed:", high_corr_features)

df_cleaned_lowcorr = df_cleaned.drop(columns=high_corr_features)
X_test = df_outid.drop(columns=high_corr_features)

print(f"Shape before removing highly correlated features: {df_cleaned.shape}")
print(f"Shape after removing highly correlated features: {df_cleaned_lowcorr.shape}")


X_test.shape


X = df_cleaned_lowcorr.drop(columns=['rainfall'])
print(X.shape)
y = df_cleaned_lowcorr['rainfall']


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
x_test_scaled = scaler.transform(X_test)

# 3. PCA’yı tüm bileşenlerle uygulama
pca = PCA()
pca.fit(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)  # Kümülatif toplam

# 5. Görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varyans')
plt.axhline(y=0.90, color='g', linestyle='--', label='90% Varyans')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Kümülatif Açıklanan Varyans Oranı')
plt.title('PCA - Açıklanan Varyans Oranı')
plt.legend()
plt.grid()
plt.show()


xtestpd = pd.DataFrame(x_test_scaled)

xtestpd[xtestpd.isna().any(axis=1)]



x_test_scaled = np.where(np.isnan(x_test_scaled), np.nanmean(x_test_scaled, axis=0), x_test_scaled)


n_components_96 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"%96 varyansı açıklayan bileşen sayısı: {n_components_96}")

pca_optimal = PCA(n_components=n_components_96)
X_pca_cor = pca_optimal.fit_transform(X_scaled)
print(f"Yeni veri şekli: {X_pca_cor.shape}")

X_test_pca = pca_optimal.transform(x_test_scaled)
print(f"Yeni test verisi şekli: {X_test_pca.shape}")


from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(X_pca_cor, y, test_size=0.2, random_state=42, stratify=y)


random_state = 42


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


models = {
    'SGD': SGDClassifier(loss='log_loss', random_state=random_state), 
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(random_state=random_state),
    'DecisionTree': DecisionTreeClassifier(random_state=random_state),
    'MLP': MLPClassifier(random_state=random_state, max_iter=300),
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(probability=True),
    'ExtraTrees': ExtraTreesClassifier(random_state=random_state)
}


from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1 Score': report['1']['f1-score'],
        'ROC AUC': roc_auc
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC AUC', ascending=False).reset_index(drop=True)
results_df.to_excel('model_evaluation_results.xlsx', index=False)
print("Results saved to model_evaluation_results.xlsx")


best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"Best model based on ROC AUC: {best_model_name}")
print(f"Performance metrics:")
print(f"  - Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"  - Precision: {results_df.iloc[0]['Precision']:.4f}")
print(f"  - Recall: {results_df.iloc[0]['Recall']:.4f}")
print(f"  - F1 Score: {results_df.iloc[0]['F1 Score']:.4f}")
print(f"  - ROC AUC: {results_df.iloc[0]['ROC AUC']:.4f}")

best_model.fit(X_train, y_train)



test_predictions = best_model.predict(X_test_pca)


submission = pd.DataFrame({'id': df_test['id'], 'rainfall': test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

