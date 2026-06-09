import numpy as np
import pandas as pd


# load file
labels_df = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')

text_true = [] 
text_fake = []

# for any article in file
for i, row in labels_df.iterrows():
    article_id = row['id']  # ×œ×“×•×’×�×”: article_0000
    real_id = row['real_text_id']   # 1 ×�×• 2
    str_article_id = 'article_' + str(i).zfill(4)  # ==> 'article_0005'

    #the pathes pf  articales
    file1_path = f'/kaggle/input/fake-or-real-the-impostor-hunt/data/train/{str_article_id}/file_1.txt'
    file2_path = f'/kaggle/input/fake-or-real-the-impostor-hunt/data/train/{str_article_id}/file_2.txt'

    
    #read texts
    with open(file1_path, 'r', encoding='utf-8') as f1:
        text1 = f1.read()

    with open(file2_path, 'r', encoding='utf-8') as f2:
        text2 = f2.read()

    #save file by true or fake
    if real_id == 1:
        text_true.append(text1)
        text_fake.append(text2)
    else:
        text_true.append(text2)
        text_fake.append(text1)

    # 94 files exists
    if i == 94:
        break


print(f"ğŸŸ¢ Real texts count: {len(text_true)}")
print(f"ğŸ”´ Fake texts count: {len(text_fake)}")

import re

def clean_and_split(text):
    #Return lowercase words from text without punctuation.
    return re.findall(r'\b\w+\b', text.lower())

zero_pair_text_in_file = []
min_len = min(len(text_true), len(text_fake))


for i in range(len(text_true)):
    true_words = clean_and_split(text_true[i])
    fake_words = clean_and_split(text_fake[i])
    both_words = [word for word in true_words if word in fake_words]
    if (len(both_words) == 0 ):
        zero_pair_text_in_file.append(i+1)
        print(f'ğŸ”� Pair {i+1}: {len(both_words)}' )


if zero_pair_text_in_file:
    print(f"\nğŸ˜¶ No common words found in {len(zero_pair_text_in_file)} pairs: {zero_pair_text_in_file}")
else:
    print("\nğŸ�‰ All pairs have at least one common word!")


df = pd.DataFrame({'True': text_true,'Fake': text_fake}) 


df.head()


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text):
    """
    Generates and displays a word cloud from the given text.

    Args:
        text (str): The input text to visualize.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("â˜�ï¸� Word Cloud", fontsize=16)
    plt.show()

for num_df_file in [0,1]:
    if num_df_file == 0:
        print(f"ğŸš€ {'-'*20} detect for True file {'-'*20}")
    else:
        print(f"ğŸš¨ {'-'*20} detect for Fake file {'-'*20}")
    for i in range(10):
        print(f'â�±ï¸� file {i}')
        sample_text = df.iloc[i, num_df_file]  # ×©×•×¨×” i, ×¢×�×•×“×” 0
        print(f"ğŸŒŸ Word Cloud for row {i+1}")
        generate_wordcloud(sample_text)




from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
def create_vector_from_list(col_values):
    # create vector from texts
    vectorizer = TfidfVectorizer(stop_words='english')  # auto stop word
    X = vectorizer.fit_transform(col_values)
    X = X.toarray()
    dump(vectorizer, 'vectorizer.joblib')
    return X ,vectorizer,vectorizer.get_feature_names_out()
all_texts = list(text_true) + list(text_fake)  # ×›×œ ×”×˜×§×¡×˜×™×� ×‘×™×—×“
vectors_all, vectorizer_all, feature_names_all = create_vector_from_list(all_texts)
vectors_t = vectors_all[:len(text_true), :]
vectors_f = vectors_all[len(text_true):, :]


print(len(vectorizer_all.get_feature_names_out()))



len(vectors_t)


df_all_txt = pd.DataFrame(vectors_all)
print(f"ğŸ§� Total missing values (NaN) in dataframe: {df_all_txt.isnull().sum().sum()} â�Œ")



print(df_all_txt.head(2))
# ×©×™×˜×” 1: ×”×“×¤×¡×ª ×�×¡×¤×¨ ×”×¢×�×•×“×•×ª
print(len(df_all_txt.columns))

# ×©×™×˜×” 2: ×”×“×¤×¡×ª ×”×¦×•×¨×” (×�×¡×¤×¨ ×©×•×¨×•×ª, ×�×¡×¤×¨ ×¢×�×•×“×•×ª)
print(df_all_txt.shape)  # ×”×ª×•×¦×�×”: (rows, columns)



df_all_txt.describe()


df_vec_t_f = df_all_txt
print(f"ğŸ”� Total missing values (NaN) in dataframe: {df_vec_t_f.isnull().sum().sum()} â�Œ")


df_vectors_t_mean = pd.DataFrame(np.mean(vectors_t, axis=1), columns=['mean_vector_t'])
df_vectors_t_std = pd.DataFrame(np.std(vectors_t, axis=1), columns=['std_vector_t'])
df_vectors_f_mean = pd.DataFrame(np.mean(vectors_f, axis=1), columns=['mean_vector_f'])
df_vectors_f_std = pd.DataFrame(np.std(vectors_t, axis=1), columns=['std_vector_f'])

df_true = pd.DataFrame({
    'mean_vector': df_vectors_t_mean['mean_vector_t'],
    'std_vector': df_vectors_t_std['std_vector_t'],
    'label': 'true'
})

df_fake = pd.DataFrame({
    'mean_vector': df_vectors_f_mean['mean_vector_f'],
    'std_vector': df_vectors_f_std['std_vector_f'],
    'label': 'fake'
})

df_vec_t_f_mean = pd.concat([df_true, df_fake], axis=0).reset_index(drop=True)


print(f"ğŸ§� Total missing values (NaN) in dataframe mean: {df_vec_t_f.isnull().sum().sum()} â�Œ")
print(df_vec_t_f_mean.iloc[[20,25,34, 40, 102,112,122, 129]])



from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
kmeans.fit(df_all_txt)
df_vec_t_f['cluster'] = kmeans.labels_

print(df_vec_t_f.shape)


print("Number of iterations run:", kmeans.n_iter_)
print("Number of clusters:", kmeans.n_clusters)
print("Random state:", kmeans.random_state)


import matplotlib.pyplot as plt

def plot_clusters(df, kmeans, normalize=True):
    """
    Plot the clustered data points in 2D with cluster centers.
    
    Args:
        df (pd.DataFrame): Original DataFrame of features.
        kmeans: Fitted KMeans model.
        normalize (bool): Whether the data was normalized (to plot in scaled space).
    """
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(df)
    else:
        X = df.values

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')

    plt.title('KMeans Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# ×©×™×�×•×© ×œ×“×•×’×�×”:
plot_clusters(new_df, k_mean_model_1)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_evaluate_random_forest(X, y, test_size=0.2, random_state=42):
    """
    ×�×�×�×Ÿ ×�×•×“×œ RandomForestClassifier ×•×�×—×–×™×¨ ×�×ª ×“×™×•×§ ×”×�×•×“×œ ×¢×œ ×§×‘×•×¦×ª ×”×‘×“×™×§×”

    ×¤×¨×�×˜×¨×™×�:
    - X: numpy array ×�×• DataFrame ×©×œ ×�×�×¤×™×™× ×™×�
    - y: numpy array ×�×• ×¡×“×¨×” ×©×œ ×ª×•×•×™×•×ª (×�×¡×¤×¨×™×•×ª)
    - test_size: ×—×œ×§ ×”× ×ª×•× ×™×� ×©×™×©×�×© ×œ×‘×“×™×§×” (×‘×¨×™×¨×ª ×�×—×“×œ 0.2)
    - random_state: ×�×¡×¤×¨ ×œ×–×¨×¢ ×�×§×¨×�×™ (×‘×¨×™×¨×ª ×�×—×“×œ 42)

    ×�×—×–×™×¨:
    - accuracy: ×“×™×•×§ ×”×�×•×“×œ ×¢×œ ×§×‘×•×¦×ª ×”×‘×“×™×§×” (float)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

from xgboost import XGBClassifier

def train_xgboost(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

from lightgbm import LGBMClassifier

def train_lightgbm(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = LGBMClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

from sklearn.ensemble import ExtraTreesClassifier

def train_extra_trees(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = ExtraTreesClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def train_neural_net(X, y, test_size=0.2, random_state=42, epochs=20, batch_size=32):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    return accuracy_score(y_test, y_pred)





from sklearn.preprocessing import LabelEncoder
X = df_vec_t_f_mean.drop(columns = 'label')
y = df_vec_t_f_mean['label']
# ×™×¦×™×¨×ª ×�×§×•×“×“ ×•×”×�×¨×”
le = LabelEncoder()
y_encoded = le.fit_transform(y)

accuracy = train_lightgbm(X, y_encoded)
print(f"ğŸ�¯ Cross-validated Accuracy: {accuracy:.4f}")



import numpy as np
import pandas as pd
from scipy.stats import entropy


def sigmoid_entropy_like(arr):
    s = np.sum(arr)
    # ×—×™×©×•×‘ ×¡×™×’×�×•×�×™×“: 1 / (1 + e^(-s))
    return 1 / (1 + np.exp(-s))


# ×—×™×©×•×‘ ×�× ×˜×¨×•×¤×™×” ×œ×›×œ ×©×•×¨×” ×‘×�×˜×¨×™×¦×” (×•×•×§×˜×•×¨×™×� ×˜×§×¡×˜ ×�×�×™×ª×™×™×�)
entropies_t = np.array([sigmoid_entropy_like(row) for row in vectors_t])

df_vectors_t_stats = pd.DataFrame({
    'mean_vector': np.mean(vectors_t, axis=1),
    'std_vector': np.std(vectors_t, axis=1),
    'var_vector': np.var(vectors_t, axis=1),
    'median_vector': np.median(vectors_t, axis=1),
    'entropy_vector': entropies_t,
    'label': 'true'
})

# ×—×™×©×•×‘ ×�× ×˜×¨×•×¤×™×” ×œ×›×œ ×©×•×¨×” ×‘×�×˜×¨×™×¦×” (×•×•×§×˜×•×¨×™×� ×˜×§×¡×˜ ×�×–×•×™×¤×™×�)
entropies_f = np.array([sigmoid_entropy_like(row) for row in vectors_f])

df_vectors_f_stats = pd.DataFrame({
    'mean_vector': np.mean(vectors_f, axis=1),
    'std_vector': np.std(vectors_f, axis=1),
    'var_vector': np.var(vectors_f, axis=1),
    'median_vector': np.median(vectors_f, axis=1),
    'entropy_vector': entropies_f,
    'label': 'fake'
})

# ×�×™×—×•×“ ×˜×‘×œ×�×•×ª True ×•-Fake
df_stats = pd.concat([df_vectors_t_stats, df_vectors_f_stats], axis=0).reset_index(drop=True)

print(df_stats.head())



from itertools import combinations

metrics = [
    'mean_vector',    # ×�×�×•×¦×¢
    'std_vector',     # ×¡×˜×™×™×ª ×ª×§×Ÿ
    'median_vector',  # ×—×¦×™×•×Ÿ
    'var_vector',     # ×©×•× ×•×ª
    'entropy_vector'
]
label_col = 'label'

all_combinations = []

for r in range(1, len(metrics)+1):  # 1 ×¢×“ ×›×œ ×”×�×“×“×™×�
    combs = list(combinations(metrics, r))
    for comb in combs:
        cols = list(comb) + [label_col]
        df_subset = df_stats[cols]
        all_combinations.append((comb, df_subset))

for i in all_combinations:
    print(i[0])


df_stats


# ×”×’×“×¨×ª ×”×¤×•× ×§×¦×™×•×ª ×œ×›×œ ×�×•×“×œ
model_funcs = {
    'RandomForest': train_and_evaluate_random_forest,
    'XGBoost': train_xgboost,
    'LightGBM': train_lightgbm,
    'ExtraTrees': train_extra_trees,
    'NeuralNetwork':train_neural_net,
}

# ×�×�×’×¨ ×”×ª×•×¦×�×•×ª ×”×¡×•×¤×™
overall_results = {}

# ×‘×—×™×¨×ª ×¤×™×¦'×¨×™×�
selected_features = ['mean_vector'	,'std_vector'	,'var_vector',	'median_vector'	,'entropy_vector']  # ×ª×•×›×œ ×œ×”×•×¡×™×£: 'vector_variance', ×•×›×•'

# ×”×›× ×ª X ×•-y
X = df_stats[selected_features]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_stats['label'])  # ×¢×›×©×™×• y ×™×”×™×” [0, 1]

# ×”×¨×¦×ª ×›×œ ×�×•×“×œ ×¤×¢×� ×�×—×ª ×‘×œ×‘×“
for model_name, func in model_funcs.items():
    print(f"\nğŸš€ Evaluating model: {model_name}")
    
    # ×”×¨×¦×ª ×”×�×•×“×œ
    accuracy = func(X, y)  # ×”×¤×•× ×§×¦×™×” ×©×œ×š ×�×—×–×™×¨×” ×�×ª ×”×“×™×•×§ (float)

    # ×©×�×™×¨×ª ×ª×•×¦×�×”
    overall_results[model_name] = {
        'accuracy': accuracy,
        'features': selected_features
    }

    print(f"âœ… Model: {model_name} | Accuracy: {accuracy:.4f}")

# ×¡×™×›×•×� ×›×•×œ×œ
print("\nğŸ�¯ Summary of Results:")
for model_name, result in overall_results.items():
    print(f"{model_name}: Accuracy = {result['accuracy']:.4f} using features {result['features']}")



def get_df_for_combination(all_combinations, target_comb):
    for comb_columns, df_subset in all_combinations:
        if comb_columns == target_comb:
            return df_subset
    return None

target_features = ('mean_vector'	,'std_vector'	,	'median_vector','var_vector','entropy_vector'	) # ×ª×•×›×œ ×œ×”×•×¡×™×£: 'vector_variance', ×•×›×•'
df_target = get_df_for_combination(all_combinations, target_features)

df_target


!pip install optuna



import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

def optimize_random_forest_with_optuna(X, y, n_trials=30):
    """
    ×�×•×¤×˜×™×�×™×–×¦×™×™×ª RandomForest ×‘×¢×–×¨×ª Optuna â€“ ×¢×œ ×›×œ ×”×“×�×˜×” (×‘×œ×™ ×•×œ×™×“×¦×™×” ×—×™×¦×•× ×™×ª).

    ×¤×¨×�×˜×¨×™×�:
    - X: ×�×�×¤×™×™× ×™×�
    - y: ×ª×•×•×™×•×ª
    - n_trials: ×�×¡×¤×¨ × ×™×¡×™×•× ×•×ª (×—×™×¤×•×©×™ ×¤×¨×�×˜×¨×™×�)

    ×�×—×–×™×¨:
    - best_model: ×”×�×•×“×œ ×¢×� ×”×¤×¨×�×˜×¨×™×� ×”×˜×•×‘×™×� ×‘×™×•×ª×¨
    - best_params: ×”×¤×¨×�×˜×¨×™×�
    - best_score: ×”×“×™×•×§ ×”×�×�×•×¦×¢
    """

    def objective(trial):
        # ×”×¦×¢×•×ª ×¤×¨×�×˜×¨×™×�
        n_estimators = trial.suggest_int("n_estimators", 100, 300)
        max_depth = trial.suggest_int("max_depth", 5, 11)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )

        # ×“×™×•×§ ×¢×œ ×›×œ ×”×“×�×˜×” (×�×¤×©×¨ ×œ×”×—×œ×™×£ ×œ-cross_val_score)
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        return acc  # ×“×™×•×§ ×’×‘×•×” ×™×•×ª×¨ = ×˜×•×‘ ×™×•×ª×¨

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    # ×�×™×�×•×Ÿ ×”×�×•×“×œ ×”×¡×•×¤×™ ×¢×� ×”×¤×¨×�×˜×¨×™×� ×”×˜×•×‘×™×� ×‘×™×•×ª×¨
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X, y)

    return best_model, best_params, best_score



# ×™×¦×™×¨×ª X ×•-y (× × ×™×— ×©×›×‘×¨ ×©×˜×—×ª ×•×§×˜×•×¨×™×�)


# ×”×¨×¦×ª Optuna
best_model, best_params, best_acc = optimize_random_forest_with_optuna(X, y, n_trials=40)

print("ğŸ�† Best RandomForest Model:")
print(f"Params: {best_params}")
print(f"Accuracy: {best_acc:.4f}")



from pprint import pprint
pprint(best_model.get_params())
print('-'*70)
import numpy as np
print(f"Most used feature index: {np.argmax(best_model.feature_importances_)}")
print('-'*70)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
import joblib
joblib.dump(best_model, "best_random_forest_model.pkl")



import os

folder_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'
lab = {}

for root, dirs, files in os.walk(folder_path):
    # ×©×•×œ×£ ×�×ª ×©×� ×”×ª×§×™×” ×�×ª×•×š ×”× ×ª×™×‘ (×”×ª×§×™×” ×”× ×•×›×—×™×ª)
    folder_name = os.path.basename(root)
    
    for filename in files:
        file_path = os.path.join(root, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ×�×� ×”×�×¤×ª×— ×§×™×™×�, ×�×•×¡×™×£ ×œ×¨×©×™×�×”, ×�×—×¨×ª ×™×•×¦×¨ ×¨×©×™×�×” ×—×“×©×”
        if folder_name not in lab:
            lab[folder_name] = []
        lab[folder_name].append(content)



import numpy as np
import pandas as pd

# ×©×� ×”×�×�×�×¨ ×œ×‘×“×™×§×”
article_id = 'article_0000'

# ×ª×›×•× ×•×ª ×©× ×‘×—×¨×• ×¢×‘×•×¨ ×”×�×•×“×œ
target_features = selected_features

# ×”×“×¤×¡×ª ×�×¡×¤×¨ ×”×˜×§×¡×˜×™×� ×‘×�×�×�×¨
num_texts = len(lab[article_id])
print(f"Number of texts in {article_id}: {num_texts}")

# ×©×œ×™×¤×ª ×©× ×™ ×”×˜×§×¡×˜×™×� ×”×¨×�×©×•× ×™×�
text1 = lab[article_id][0]
text2 = lab[article_id][1]

# ×™×¦×™×¨×ª ×•×§×˜×•×¨×™×� ×�×”×˜×§×¡×˜×™×�
vector1, _, _ = create_vector_from_list([text1])
vector2, _, _ = create_vector_from_list([text2])

# ×”×ª×�×�×ª ×�×‘× ×” ×—×“-×�×�×“×™
vec1 = vector1 if len(vector1.shape) == 1 else vector1[0]
vec2 = vector2 if len(vector2.shape) == 1 else vector2[0]

# ×—×™×©×•×‘ ×ª×›×•× ×•×ª ×œ×˜×§×¡×˜ ×”×¨×�×©×•×Ÿ
features1 = {
    'mean_vector': np.mean(vec1),
    'std_vector': np.std(vec1),
    'median_vector': np.median(vec1),
    'var_vector': np.var(vec1),
    'entropy_vector': sigmoid_entropy_like(vec1)
}

# ×—×™×©×•×‘ ×ª×›×•× ×•×ª ×œ×˜×§×¡×˜ ×”×©× ×™
features2 = {
    'mean_vector': np.mean(vec2),
    'std_vector': np.std(vec2),
    'median_vector': np.median(vec2),
    'var_vector': np.var(vec2),
    'entropy_vector': sigmoid_entropy_like(vec2)
}

# ×™×¦×™×¨×ª ×˜×‘×œ×�×•×ª × ×ª×•× ×™×�
df1 = pd.DataFrame([features1])
df2 = pd.DataFrame([features2])

# ×—×™×–×•×™ ×¢×‘×•×¨ ×˜×§×¡×˜ ×¨×�×©×•×Ÿ
prob1 = best_model.predict_proba(df1[target_features])[0][1]
pred1 = int(prob1 > 0.5)

# ×—×™×–×•×™ ×¢×‘×•×¨ ×˜×§×¡×˜ ×©× ×™
prob2 = best_model.predict_proba(df2[target_features])[0][1]
pred2 = int(prob2 > 0.5)

# ×”×“×¤×¡×ª ×ª×•×¦×�×•×ª
print(f"\nText 1: Prediction probability: {prob1:.4f} | Binary prediction: {pred1}")
print(f"Text 2: Prediction probability: {prob2:.4f} | Binary prediction: {pred2}")



print(len(vectorizer_all.get_feature_names_out()))



from joblib import load
import numpy as np
from sklearn.base import ClusterMixin

def predict_and_distance(text: str, vectorizer: str, model: ClusterMixin) -> tuple[int, float]:
    """
    Predict cluster and compute Euclidean distance from cluster center for a given text.

    Args:
        text (str): Input text to classify.
        vectorizer_path (str): Path to saved vectorizer (.joblib).
        model (ClusterMixin): Pre-trained clustering model (e.g., KMeans).

    Returns:
        tuple[int, float]: (predicted_cluster, distance_to_cluster_center)
    """

    vec = vectorizer.transform([text])

    cluster = model.predict(vec)[0]
    center = model.cluster_centers_[cluster]
    distance = np.linalg.norm(vec.toarray() - center)

    return cluster, distance

# ×˜×¢×™× ×ª ×”×�×•×“×œ ×¤×¢×� ×�×—×ª ×�×—×•×¥ ×œ×¤×•× ×§×¦×™×” (×œ×�×©×œ ×‘×ª×—×™×œ×ª ×”×¡×§×¨×™×¤×˜)
# from joblib import load
# k_mean_model_1 = load('kmeans_model.joblib')

# ×©×™×�×•×©
cluster_id, dist = predict_and_distance(text1, vectorizer_all, k_mean_model_1)

print(f"Cluster: {cluster_id}")
print(f"Distance from center: {dist:.4f}")



results = []

# ×¨×©×™×�×ª ×”×¤×™×¦'×¨×™×� ×©×”×�×•×“×œ ×”×•×›×©×¨ ×¢×œ×™×”×�
features = target_features
#target_features = ('mean_vector'	,'std_vector'	,	'median_vector','var_vector','entropy_vector'	) # ×ª×•×›×œ ×œ×”×•×¡×™×£: 'vector_variance', ×•×›×•'
for article_id, texts in lab.items():
    for idx, text in enumerate(texts):
        try:
            # ×™×¦×™×¨×ª ×”×•×§×˜×•×¨ ×�×”×˜×§×¡×˜
            vector, _, _ = create_vector_from_list([text])

            # ×•×§×˜×•×¨ ×—×“-×�×�×“×™ (×�×� ×™×© ×�×™×�×“ × ×•×¡×£, × ×‘×—×¨ ×�×ª ×”×©×•×¨×” ×”×¨×�×©×•× ×”)
            vec = vector if len(vector.shape) == 1 else vector[0]



            # ×™×¦×™×¨×ª DataFrame ×¢×� ×›×œ ×”×�×“×“×™×�
            row_df = pd.DataFrame([{
                'article_id': article_id,
                'file_index': idx + 1,
                'mean_vector': np.mean(vec),
                'std_vector': np.std(vec),
                'median_vector': np.median(vec),
                'var_vector': np.var(vec),
                'entropy_vector': sigmoid_entropy_like(vec)
                
            }])

            # ×‘×•×“×§×™×� ×©×”×¢×�×•×“×•×ª ×©×”×�×•×“×œ ×�×¦×¤×” ×œ×”×Ÿ ×�×›×Ÿ ×§×™×™×�×•×ª ×•×‘×�×•×ª×• ×¡×“×¨
            X_to_predict = row_df[features]

            # ×—×™×–×•×™ ×”×¡×ª×‘×¨×•×ª×™ ×•×§×‘×œ×ª ×ª×•×¦×�×” ×‘×™× ×�×¨×™×ª
            prob = best_model.predict_proba(X_to_predict)[0][1]
            binary_pred = int(prob > 0.5)

        except Exception as e:
            print(f"Error processing article {article_id}, file_index {idx + 1}: {e}")

            # ×‘×�×§×¨×” ×©×œ ×©×’×™×�×” - ×¡×™×•×•×’ 0 ×•×�×™×œ×•×™ ×¢×¨×›×™×� ×—×¡×¨×™×�
            binary_pred = 0
            prob = 0.0

            row_df = pd.DataFrame([{
                'article_id': article_id,
                'file_index': idx + 1,
                'mean_vector':0,
                'std_vector': 0,
                'median_vector':0,
                'var_vector': 0,
                'entropy_vector':0
                
            }])
            X_to_predict = row_df[features]

        # ×”×•×¡×¤×ª ×¢×�×•×“×•×ª ×”×ª×•×¦×�×” ×œ-DataFrame
        row_df['prediction'] = binary_pred
        row_df['probability'] = prob

        # ×©×�×™×¨×ª ×”×ª×•×¦×�×” ×œ×¨×©×™×�×”
        results.append(row_df)

# ×—×™×‘×•×¨ ×›×œ ×”×ª×•×¦×�×•×ª ×œ-DataFrame ×�×—×“
final_results_df = pd.concat(results, ignore_index=True)



final_results_df ['index_column'] = range(len(final_results_df ))
df_predictions =final_results_df 
df_predictions['probability'].unique


df_predictions.to_csv('prediction.csv')


import pandas as pd

# × × ×™×— ×©-df_predictions ×§×™×™×� ×•×™×© ×‘×• ×¢×�×•×“×ª 'probability'

# ×§×•×“×� ×�×•×¡×™×¤×™×� ×¢×�×•×“×ª index_column ×�×� ×œ×� ×§×™×™×�×ª
df_predictions['index_column'] = range(len(df_predictions))

# ×™×•×¦×¨×™×� ×¢×�×•×“×” ×©×�×–×”×” ×�×ª ×”×–×•×’ - ×›×œ×•×�×¨ ×›×œ ×©× ×™ ×�×™× ×“×§×¡×™×� ×©×™×™×›×™×� ×œ×�×•×ª×• ×–×•×’
df_predictions['pair_id'] = df_predictions['index_column'] // 2

# ×¢×›×©×™×• ×¢×‘×•×¨ ×›×œ ×–×•×’ ×‘×•×—×¨×™×� ×�×ª ×”×©×•×¨×” ×¢×� ×”-probability ×”×’×‘×•×” ×‘×™×•×ª×¨
idx_max_prob = df_predictions.groupby('pair_id')['probability'].idxmax()

# ×™×•×¦×¨×™×� ×“×�×˜×”×¤×¨×™×™×� ×—×“×© ×¢×� ×”×©×•×¨×•×ª ×”×�×œ×•
df_best_predictions = df_predictions.loc[idx_max_prob].reset_index(drop=True)

# ×�×� ×¨×•×¦×™×�, ×�×¤×©×¨ ×’×� ×œ×�×—×•×§ ×�×ª ×”×¢×�×•×“×” 'pair_id' ×•'index_column' ×›×“×™ ×œ× ×§×•×ª
df_best_predictions = df_best_predictions.drop(columns=['pair_id', 'index_column'])

df_best_predictions



df_best_predictions[['article_id','prediction']]


# × × ×™×— ×©×™×© ×œ×š ×‘×¢×�×•×“×” 'article_id' ×�×ª ×�×–×”×” ×”×�×�×�×¨
# × ×—×œ×¥ ×�×ª ×”×ª×—×–×™×ª ×”×�×§×¡×™×�×œ×™×ª ×œ×›×œ article_id
df_unique = final_results_df.groupby('article_id')['prediction'].max().reset_index()

# ×¦×•×¨ ×¢×�×•×“×ª id ×¨×¦×”
df_unique['id'] = range(0, len(df_unique) )

# ×‘×—×¨ ×¢×�×•×“×•×ª ×œ×©×�×™×¨×”
df_to_save = df_unique[['id', 'prediction']]

print(df_to_save)
df_to_save.to_csv('predictions.csv', index=False)



results = []


for article_id, texts in lab.items():
    for idx, text in enumerate(texts):
        try:
            vector, _, _ = create_vector_from_list([text])
            vec = vector if len(vector.shape) == 1 else vector[0]

            X_to_predict = vec

            # ×—×™×–×•×™ ×§×œ×�×¡×˜×¨ ×¢×� k_mean_model_1
            cluster_label = k_mean_model_1.predict(X_to_predict)[0]

            # ×—×™×©×•×‘ ×�×¨×—×§ ×œ×�×¨×›×– ×”×§×œ×�×¡×˜×¨
            distances = k_mean_model_1.transform(X_to_predict)
            distance_to_center = distances[0][cluster_label]

            # ×�×¤×©×¨ ×œ×”×•×¡×™×£ ×¤×” ×’×� ×—×™×–×•×™ ×”×¡×ª×‘×¨×•×ª×™ ×�×� ×™×©, ×�×‘×œ ×‘-KMeans ×�×™×Ÿ.
            # ×œ×›×Ÿ × ×•×¡×™×£ ×¨×§ ×�×ª ×”×ª×•×•×™×ª ×•×”×�×¨×—×§.

        except Exception as e:
            print(f"Error processing article {article_id}, file_index {idx + 1}: {e}")

            cluster_label = -1  # ×�×• ×¢×¨×š ×©×�×ª×�×™×� ×œ"×©×’×™×�×”"
            distance_to_center = np.nan

            X_to_predict = vec

        row_df['cluster'] = cluster_label
        row_df['distance_to_center'] = distance_to_center

        results.append(row_df)

final_results_df = pd.concat(results, ignore_index=True)


