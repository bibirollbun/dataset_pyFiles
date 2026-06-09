# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Configure the style of the graphs for better visualization
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')


# Upload the competition data files
# Kaggle stores the data in the '../input/<competition-name>/' directory

try:
    df_train = pd.read_csv('../input/map-charting-student-math-misunderstandings/train.csv')
    df_test = pd.read_csv('../input/map-charting-student-math-misunderstandings/test.csv')
    sample_submission = pd.read_csv('../input/map-charting-student-math-misunderstandings/sample_submission.csv')
except FileNotFoundError:
    print("Make sure the path to the files is correct!")

# View the first 5 lines of our training set
print("Sample Training Data:")
display(df_train.head())

print(f"\nNumber of training samples: {len(df_train)}")
print(f"Number of test samples: {len(df_test)}")


# 1. Create the 'target' column that we will predict
df_train['target'] = df_train['Category'] + ':' + df_train['Misconception']

# 2. Count how many unique targets there are
num_unique_targets = df_train['target'].nunique()
print(f"There are {num_unique_targets} unique targets to predict.")

#3. See what the most common targets are
print("\nThe 15 most common targets:")
display(df_train['target'].value_counts().head(15))

# 4. Visualize the distribution of the 30 most common targets
plt.figure(figsize=(12,10))
sns.countplot(y='target', data=df_train, order=df_train['target'].value_counts().iloc[:30].index, palette='viridis')
plt.title('Distribution of the 30 Most Common Targets')
plt.xlabel('Count')
plt.ylabel('Category:Misconception') 
plt.show()


# Create a 'full_text' column that combines all textual information
# We use [SEP] as a special separator that many NLP models understand well

df_train['full_text'] = 'QUESTION: ' + df_train['QuestionText'] + \
                       ' [SEP] ANSWER: ' + df_train['MC_Answer'] + \
                       ' [SEP] EXPLANATION: ' + df_train['StudentExplanation']


# Do the same for the test set
df_test['full_text'] = 'QUESTION: ' + df_test['QuestionText'] + \
                       ' [SEP] ANSWER: ' + df_test['MC_Answer'] + \
                       ' [SEP] EXPLANATION: ' + df_test['StudentExplanation']


# Let's see a complete example
print("Example of a combined 'full_text':")
print(df_train['full_text'].iloc[0])


# Calculate the number of words in each 'full_text'
df_train['word_count'] = df_train['full_text'].apply(lambda x: len(str(x).split()))

# View word count distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_train['word_count'], bins=50, kde=True)
plt.title('Distribution of the Number of Words in the Full Text')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# See some descriptive statistics
print("\nStatistics on the number of words:")
display(df_train['word_count'].describe())


# Import the necessary libraries for modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import lightgbm as lgb


# --- Data Preparation ---

# 1. Fill NaN values in Misconception (if any) to avoid errors
# The combination has already been done, but it is good practice to ensure there are no stray NaNs
df_train['Misconception'] = df_train['Misconception'].fillna('NA')


# 2. Create the 'target' again to ensure it is up to date
df_train['target'] = df_train['Category'] + ':' + df_train['Misconception']


# 3. Create the LabelEncoder to transform the targets into numbers
# It will create a mapping: 0 -> 'target_A', 1 -> 'target_B', etc.
label_encoder = LabelEncoder()
df_train['target_encoded'] = label_encoder.fit_transform(df_train['target'])


# View the result
print('Mapping some targets to numbers:')
print(df_train[['target', 'target_encoded']].head())


# --- TF-IDF Vectorization ---

# Initialize the TF-IDF Vectorizer
# max_features=5000: We will consider only the 5000 most important words
# ngram_range=(1, 3): We will consider single words, word pairs, and word triplets.

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))


# Apply TF-IDF to training and test texts
X_train = tfidf_vectorizer.fit_transform(df_train['full_text'])
X_test = tfidf_vectorizer.transform(df_test['full_text'])


# Set the target variable 'y'
y_train = df_train['target_encoded']


print(f"Dimensions of our training matrix (samples, features): {X_train.shape}")
print(f"Dimensions of our test matrix (samples, features): {X_test.shape}")


# --- Model Training ---

# 1. Create a basic LightGBM model.
# The 'objective':'multiclass' and 'num_class' parameters would be an alternative,
# but OneVsRest gives us more control over the probabilities of each class.
lgbm = lgb.LGBMClassifier(objective='binary', random_state=42)


# 2. Wrap the model with OneVsRestClassifier
# n_jobs=-1 means using all available processors to speed up training
ovr_classifier = OneVsRestClassifier(lgbm, n_jobs=-1)


# 3. Train the classifier
print('Starting model training... (This may take a few minutes)')
ovr_classifier.fit(X_train, y_train)
print('Training complete!')


# --- Forecast and Submission ---

#1. Predict the probabilities for each class in the test set
print('Generating predictions on the test set...')
test_probabilities = ovr_classifier.predict_proba(X_test)


# 2. Get the 3 classes with the highest probability for each test sample
# argsort() sorts the indices, and [:, -3:] gets the last 3 (largest)
top_3_preds_indices = np.argsort(test_probabilities, axis=1)[:, -3:]


#3. Reverse the order so that the highest probability comes first
top_3_preds_indices = np.flip(top_3_preds_indices, axis=1)


#4. Convert the numeric indices back to the original class names
predictions = label_encoder.inverse_transform(top_3_preds_indices.flatten())
predictions = predictions.reshape(top_3_preds_indices.shape)


#5. Format predictions into a single string, separated by spaces
submission_preds = [' '.join(pred) for pred in predictions]


# 6. Create the submission DataFrame
submission_df = pd.DataFrame({'row_id': df_test['row_id'], 'Category:Misconception': submission_preds})


# 7. Save the submission file
submission_df.to_csv('submission.csv', index=False)


print("\n'Submission.csv' file generated successfully!")
display(submission_df.head())


# ======================================================================================
# ENVIRONMENT AND INSTALLATION CONFIGURATION
# ======================================================================================
# !pip install transformers datasets accelerate -q # This command is commented out for offline submission

# --- Necessary imports ---
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import gc # Garbage Collector to clean memory

# --- Global Configuration ---
class CFG:
    # Path corrected to point to the subfolder within the dataset
    model_name = "/kaggle/input/deberta-v3-base/deberta-v3-base"
    max_length = 256  # <<<--- FIX: THIS LINE WAS MISSING
    batch_size = 16
    learning_rate = 2e-5
    epochs = 3
    
# --- Set the device (GPU or CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ======================================================================================
#  DATA PREPARATION AND SPLITTING
# ======================================================================================

# --- Import necessary libraries ---
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

# --- Load the data ---
df_train_full = pd.read_csv('../input/map-charting-student-math-misunderstandings/train.csv')
df_test = pd.read_csv('../input/map-charting-student-math-misunderstandings/test.csv')

# --- Create the 'target' and 'full_text' columns ---
df_train_full['Misconception'] = df_train_full['Misconception'].fillna('NA')
df_train_full['target'] = df_train_full['Category'] + ':' + df_train_full['Misconception']
df_train_full['full_text'] = 'QUESTION: ' + df_train_full['QuestionText'] + ' [SEP] ANSWER: ' + df_train_full['MC_Answer'] + ' [SEP] EXPLANATION: ' + df_train_full['StudentExplanation']
df_test['full_text'] = 'QUESTION: ' + df_test['QuestionText'] + ' [SEP] ANSWER: ' + df_test['MC_Answer'] + ' [SEP] EXPLANATION: ' + df_test['StudentExplanation']


# --- FIX: Handle ALL extremely rare classes before splitting ---
N_SPLITS = 5 # Define our number of splits here

# Find how many times each class appears
target_counts = df_train_full['target'].value_counts()
# Identify all classes that have fewer members than N_SPLITS
rare_targets = target_counts[target_counts < N_SPLITS].index

# Separate these rare classes into a holdout set
rare_df = df_train_full[df_train_full['target'].isin(rare_targets)]
# Create a dataframe with the rest of the data for splitting
main_df = df_train_full[~df_train_full['target'].isin(rare_targets)]

# --- Split the main data (which now has no rare classes) ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Get the indices for the first fold from the main data
train_indices, valid_indices = next(skf.split(main_df, main_df['target']))

# Create the initial training and validation DataFrames
train_df = main_df.iloc[train_indices]
valid_df = main_df.iloc[valid_indices]

# --- Add the rare classes back to the TRAINING set ---
# This ensures the model sees every class at least once
train_df = pd.concat([train_df, rare_df]).reset_index(drop=True)

print(f"Original training data size: {len(df_train_full)} samples")
print(f"New training set size: {len(train_df)} samples (includes rare classes)")
print(f"New validation set size: {len(valid_df)} samples")
print("-" * 50)


# --- Multi-Label Binarization (using ONLY train_df) ---
mlb = MultiLabelBinarizer()
y_train_binarized = mlb.fit_transform([[label] for label in train_df['target']])
CLASSES = mlb.classes_

print(f"Number of unique classes found (in training): {len(CLASSES)}")
print("Example of a binarized target (from the training set):")
print(f"Original Target: {train_df['target'].iloc[0]}")
print(f"Binarized Target (partial): {y_train_binarized[0][:10]}...")


# ======================================================================================
# TOKENIZATION AND DATASET CREATION
# ======================================================================================

# --- Load the Tokenizer ---
# This cell may take a moment on its first run to download the tokenizer files
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

# --- Custom Dataset Class ---
# This class serves as a "recipe" for preparing each sample for the model
class MathMisunderstandingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

# --- Create Datasets and DataLoaders for TRAINING and VALIDATION ---

# Create the training dataset using train_df
train_dataset = MathMisunderstandingDataset(
    texts=train_df['full_text'].values,
    labels=y_train_binarized, # y_train_binarized was already created from train_df
    tokenizer=tokenizer,
    max_len=CFG.max_length
)

# Create the validation dataset using valid_df
# Note: we use mlb.transform() here, as the mlb was already fitted on the training data
valid_dataset = MathMisunderstandingDataset(
    texts=valid_df['full_text'].values,
    labels=mlb.transform([[label] for label in valid_df['target']]),
    tokenizer=tokenizer,
    max_len=CFG.max_length
)

# Create the DataLoaders for both datasets
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
# It's good practice to use a larger batch_size and shuffle=False for validation to speed up the process
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size * 2, shuffle=False)

print("Training and validation Datasets and DataLoaders created successfully.")
print(f"Batches in train_loader: {len(train_loader)}")
print(f"Batches in valid_loader: {len(valid_loader)}")


# --- Evaluation Metric Function: MAP@3 ---

def map_at_3(true_labels, pred_labels):
    """
    Calculates the Mean Average Precision @ 3.
    true_labels: A list of lists with the true labels. Ex: [['label_A'], ['label_B']]
    pred_labels: A list of lists with the top 3 predictions. Ex: [['pred_1', 'pred_2', 'pred_3'], ...]
    """
    score = 0.0
    for i in range(len(true_labels)):
        # Get the single true label for the sample
        true_label = true_labels[i][0]
        
        # Check if the true label is in the top 3 predictions
        if true_label in pred_labels[i]:
            # Find the rank (index) of the correct prediction (0, 1, or 2)
            rank = pred_labels[i].index(true_label)
            # The score is 1 / (rank + 1)
            score += 1.0 / (rank + 1.0)
            
    # Return the average score across all samples
    return score / len(true_labels)


# ======================================================================================
# MODEL, OPTIMIZER, AND SCHEDULER DEFINITION
# ======================================================================================

# Load the pre-trained model from Hugging Face
model = AutoModelForSequenceClassification.from_pretrained(
    CFG.model_name,
    num_labels=len(CLASSES),                  # Informs the model how many outputs it needs
    problem_type="multi_label_classification" # Configures the model for multi-label classification
)
# Send the model to the GPU for acceleration
model.to(device)

# Define the AdamW optimizer, the standard for training Transformers
optimizer = AdamW(model.parameters(), lr=CFG.learning_rate)

# Calculate the total number of training steps
num_training_steps = CFG.epochs * len(train_loader)

# Define the learning rate scheduler
# It will gradually decrease the learning rate during training, which helps with stabilization
lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer,
    num_warmup_steps=0, 
    num_training_steps=num_training_steps
)

print("Model, Optimizer, and Scheduler defined successfully.")


# ======================================================================================
# STEP 5: MODEL TRAINING AND VALIDATION LOOP
# ======================================================================================
import gc
import matplotlib.pyplot as plt # Import here for future use in Step 6
import seaborn as sns           # Import here for future use in Step 6
from sklearn.metrics import confusion_matrix # Import here for future use in Step 6

# Ensure the loss function is defined
loss_fn = torch.nn.BCEWithLogitsLoss()

# --- Dictionary to store training history ---
history = {
    'train_loss': [],
    'valid_score': []
}
# --- Variables to store the results from the best epoch for later analysis ---
best_preds_for_analysis = []
best_true_for_analysis = []


# --- Main Training Loop ---
best_score = 0.0
model_path = "best_model.pth"

for epoch in range(CFG.epochs):
    print(f"--- Epoch {epoch+1}/{CFG.epochs} ---")
    
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    train_progress_bar = tqdm(train_loader, desc="Training")
    for batch in train_progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_progress_bar.set_postfix(loss=loss.item())
    
    avg_train_loss = total_train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    model.eval()
    all_valid_preds = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.sigmoid(outputs.logits)
            all_valid_preds.append(probabilities.cpu().numpy())
            
    valid_probabilities = np.vstack(all_valid_preds)
    top_3_indices = np.argsort(valid_probabilities, axis=1)[:, -3:]
    top_3_indices = np.flip(top_3_indices, axis=1)
    pred_labels = mlb.classes_[top_3_indices]
    true_labels = [[label] for label in valid_df['target']]
    score = map_at_3(true_labels, pred_labels.tolist())
    history['valid_score'].append(score)
    print(f"Validation MAP@3 Score: {score:.4f}")
    
    if score > best_score:
        print(f"New best score! Saving model to {model_path}")
        best_score = score
        # Store the predictions and true labels from this best epoch for analysis
        best_preds_for_analysis = pred_labels
        best_true_for_analysis = valid_df['target']
        torch.save(model.state_dict(), model_path)
    
    torch.cuda.empty_cache()
    gc.collect()

# --- End of Training Loop ---
print("\n" + "="*50 + "\nTRAINING COMPLETE!\n" + "="*50 + "\n")


# ======================================================================================
# STEP 6: POST-TRAINING ANALYSIS AND VISUALIZATIONS
# ======================================================================================

# --- 1. Visualize Learning Curves ---
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.set_title("Model Learning Curves", fontsize=16)
ax1.plot(history['train_loss'], 'r.-', label='Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.grid(True)

# Create a second y-axis for the validation score
ax2 = ax1.twinx()
ax2.plot(history['valid_score'], 'b.-', label='Validation MAP@3')
ax2.set_ylabel('MAP@3 Score', color='b')
ax2.tick_params(axis='y', labelcolor='b')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.85))
plt.show()

# --- 2. Analyze Errors with a Confusion Matrix ---
print("\n" + "="*50 + "\nERROR ANALYSIS\n" + "="*50)
# Get the top-1 prediction for each validation sample
best_preds_top1 = [p[0] for p in best_preds_for_analysis]

cm = confusion_matrix(best_true_for_analysis, best_preds_top1, labels=mlb.classes_)
cm_df = pd.DataFrame(cm, index=mlb.classes_, columns=mlb.classes_)

# Find the most common errors
errors = []
for true_label in mlb.classes_:
    for pred_label in mlb.classes_:
        if true_label != pred_label:
            count = cm_df.loc[true_label, pred_label]
            if count > 0:
                errors.append({'True': true_label, 'Predicted': pred_label, 'Count': count})

error_df = pd.DataFrame(errors).sort_values(by='Count', ascending=False)
print("Top 15 most common classification errors (based on the best epoch):")
if error_df.empty:
    print("No classification errors found in the validation set! (This is rare)")
else:
    display(error_df.head(15))


# ======================================================================================
# STEP 7: FINAL INFERENCE AND SUBMISSION GENERATION
# ======================================================================================
print(f"\nLoading the best model (score: {best_score:.4f}) for final inference.")
model.load_state_dict(torch.load(model_path)) # Load the weights of the best model
model.eval()

# Create dataset and dataloader for the test set
test_labels_dummy = np.zeros((len(df_test), len(CLASSES)))
test_dataset = MathMisunderstandingDataset(df_test['full_text'].values, test_labels_dummy, tokenizer, CFG.max_length)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size * 2, shuffle=False)

# Make predictions on the test set
all_test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on Test Set"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.sigmoid(outputs.logits)
        all_test_preds.append(probabilities.cpu().numpy())

# Format the predictions for the submission file
test_probabilities = np.vstack(all_test_preds)
top_3_indices = np.argsort(test_probabilities, axis=1)[:, -3:]
top_3_indices = np.flip(top_3_indices, axis=1)
predictions = mlb.classes_[top_3_indices]
submission_preds = [' '.join(pred) for pred in predictions]

# Create and save the final file
submission_df = pd.DataFrame({'row_id': df_test['row_id'], 'Category:Misconception': submission_preds})
submission_df.to_csv('submission.csv', index=False)
print("\n'submission.csv' file generated successfully!")
display(submission_df.head())

