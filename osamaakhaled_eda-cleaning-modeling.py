import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from tqdm import tqdm
#Scikit learn
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
#PyTorch
import torch
from torch import nn, optim
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_df.head()


train_df.info()


train_df.duplicated().sum(), test_df.duplicated().sum()


train_df['y'].value_counts()


def plot_count(df, col, title_name='Train'):
    # Set background color
    
    f, ax = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.2)

    s1 = col.value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = ['#186F65', '#B5CB99', '#071952']
    inner_colors = ['#001524', '#445D48', '#D6CC99']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90, frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size': 13, 
        'weight': 'bold', 
        'color': 'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = s1.index.tolist()
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette='mako', orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='black', fontweight='bold', fontsize=12)

    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel('target', fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_count(train_df, train_df['y'], 'Target Variable Distribution')


textual = train_df.select_dtypes(include=[object])
textual.head()


train_df.contact.value_counts()


train_df.education.value_counts()


train_df.default.value_counts()


train_df.job.value_counts()


train_df.poutcome.value_counts()


test_df.poutcome.value_counts()


train_df[train_df['contact'] == 'unknown']['poutcome'].value_counts()


train_df[train_df['contact'] == 'unknown']['poutcome'].value_counts()


numerical = train_df.select_dtypes(include=[int, float]).drop(['id', 'y'], axis = 1)
numerical.head()


numerical.describe()


test_df.select_dtypes(include=['int','float']).drop(['id'],axis=1).describe()


num = numerical.columns

df = pd.concat([train_df[num].copy().assign(Source = 'Train'), test_df[num].copy().assign(Source = 'Test')], ignore_index = True)

fig, axes = plt.subplots(len(num), 3 ,figsize = (16, len(num) * 4), gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]})

for i,col in enumerate(num):
    ax = axes[i,0]
    sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', palette=['#456cf0', '#ed7647'], linewidth = 2.1, warn_singular=False, ax = ax) # Use of seaborn with artistic interface
    ax.set_title(f"\n{col}",fontsize = 9)
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75)
    ax.set(xlabel = '', ylabel = '')

    ax = axes[i,1]
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#456cf0', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Train", fontsize = 9)

    ax = axes[i,2]
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#ed7647', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Test", fontsize = 9)

plt.show();


cat_list = list(train_df.select_dtypes(include=['object']))

for cat in cat_list:
    train_df[cat] = train_df[cat].astype('category')
    categories = train_df[cat].cat.categories
    
    test_df[cat] = pd.Categorical(test_df[cat], categories=categories)

    train_df[cat] = train_df[cat].cat.codes
    test_df[cat] = test_df[cat].cat.codes


train_df.head()


test_df.head()


cat = textual.columns

df = pd.concat(
    [train_df[cat].assign(Source='Train'),
     test_df[cat].assign(Source='Test')],
    ignore_index=True
)

fig, axes = plt.subplots(
    len(cat), 2,
    figsize=(16, len(cat) * 4),
    gridspec_kw={'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.50, 0.50]}
)

for i, col in enumerate(cat):
    # Train
    ax = axes[i, 0]
    sns.countplot(data=df[df.Source == 'Train'], y=col, color='#456cf0', ax=ax)
    ax.set(xlabel='Count', ylabel=col)
    ax.set_title(f"Train - {col}", fontsize=9)
    ax.bar_label(ax.containers[0])  # Add counts on bars

    # Test
    ax = axes[i, 1]
    sns.countplot(data=df[df.Source == 'Test'], y=col, color='#ff6f61', ax=ax)
    ax.set(xlabel='Count', ylabel=col)
    ax.set_title(f"Test - {col}", fontsize=9)
    ax.bar_label(ax.containers[0])  # Add counts on bars

plt.tight_layout()
plt.show();


#Separate the id and ground truth columns in two separate variables from the training data 
#Separate the id column from the test data
IDs_train = train_df['id']
IDs_test = test_df['id']

target = train_df['y']
train_df.drop(['id', 'y'], axis=1, inplace=True)
test_df.drop(['id'], axis=1, inplace=True)


#Detect and remove outliers
def remove_outliers(train_df):
    model = IsolationForest(contamination=0.01)
    model.fit(train_df)
    outlier_predictions = model.predict(train_df)
        
    return train_df[outlier_predictions == -1].index

indices = remove_outliers(train_df)

train_df.drop(list(indices),inplace = True)
target.drop(list(indices), inplace = True)
print(f'after dropping the outliers\n the training set and the target shapes are now {train_df.shape} and {target.shape}')


plot_count(train_df, target, 'Target Variable Distribution')


train_df.describe()


test_df.describe()


train_copy = train_df.copy()
test_copy = test_df.copy()


train_copy[['balance','pdays']] = train_copy[['balance','pdays']] - train_copy[['balance','pdays']].min() 
test_copy[['balance','pdays']] = test_copy[['balance','pdays']] - test_copy[['balance','pdays']].min() 


train_copy.describe()


test_copy.describe()


train_copy[numerical.columns] = np.log(train_copy[numerical.columns] + 1e-10)
test_copy[numerical.columns] = np.log(test_copy[numerical.columns] + 1e-10)


df = pd.concat([train_copy[num].copy().assign(Source = 'Train'), test_copy[num].copy().assign(Source = 'Test')], ignore_index = True)

fig, axes = plt.subplots(len(num), 3 ,figsize = (16, len(num) * 4), gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]})

for i,col in enumerate(num):
    ax = axes[i,0]
    sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', palette=['#456cf0', '#ed7647'], linewidth = 2.1, warn_singular=False, ax = ax) # Use of seaborn with artistic interface
    ax.set_title(f"\n{col}",fontsize = 9)
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75)
    ax.set(xlabel = '', ylabel = '')

    ax = axes[i,1]
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#456cf0', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Train", fontsize = 9)

    ax = axes[i,2]
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#ed7647', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Test", fontsize = 9)

plt.show();


#Make pipeline for data pre-processing
pipeline = make_pipeline(

    QuantileTransformer(output_distribution = 'normal',random_state = 42),
    StandardScaler()
)


#Transform the training distribtution
transformer = make_column_transformer(
    (
        pipeline,
        make_column_selector(dtype_include=np.number) # We want to apply numerical_pipeline only on numerical columns
    ),
    remainder = 'passthrough',
    verbose_feature_names_out=False
)

transformer


#Fit the pipeline made to the training set
train = transformer.fit_transform(train_copy)
test = transformer.fit_transform(test_copy)


#Convert the training data to a dataframe
train = pd.DataFrame(data=train,columns=transformer.get_feature_names_out())
test = pd.DataFrame(data=test,columns=transformer.get_feature_names_out())


df = pd.concat([train[num].copy().assign(Source = 'Train'), test[num].copy().assign(Source = 'Test')], ignore_index = True)

fig, axes = plt.subplots(len(num), 3 ,figsize = (16, len(num) * 4), gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]})

for i,col in enumerate(num):
    ax = axes[i,0]
    sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', palette=['#456cf0', '#ed7647'], linewidth = 2.1, warn_singular=False, ax = ax) # Use of seaborn with artistic interface
    ax.set_title(f"\n{col}",fontsize = 9)
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75)
    ax.set(xlabel = '', ylabel = '')

    ax = axes[i,1]
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#456cf0', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Train", fontsize = 9)

    ax = axes[i,2]
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#ed7647', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Test", fontsize = 9)

plt.show();


train.shape, target.shape, test.shape


rf_classifier = RandomForestClassifier()
rf_classifier.fit(train.copy(), target.copy())

feature_importance = rf_classifier.feature_importances_

importance_df = pd.DataFrame({'feature': train.columns,
                              'importance': feature_importance})

importance_df = importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(20, 10))
plt.bar(range(train.shape[1]), importance_df['importance'])  
plt.xticks(range(train.shape[1]), train.columns)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


train_modified = train[['age', 'job', 'marital','education','default','balance','housing','contact']]
test_modified = test[['age', 'job', 'marital','education','default','balance','housing', 'contact']]


training, validating, training_target, validating_target = train_test_split(train_modified, target, test_size=0.05, random_state=42)


training.shape, validating.shape, training_target.shape, validating_target.shape


training = torch.tensor(training.to_numpy(), dtype=torch.float16)
validating = torch.tensor(validating.to_numpy(), dtype=torch.float16)
test = torch.tensor(test_modified.to_numpy())

training_target = torch.tensor(training_target.to_numpy(), dtype=torch.float16)
validating_target = torch.tensor(validating_target.to_numpy(), dtype=torch.float16)

training_target = training_target.view(-1,1)
validating_target = validating_target.view(-1,1)

train_set = torch.utils.data.TensorDataset(training, training_target)
val_set = torch.utils.data.TensorDataset(validating, validating_target)

trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32)
valoader = torch.utils.data.DataLoader(dataset=val_set, batch_size=32)


dataiter = iter(trainloader)
data, labels = next(dataiter)


model = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(8, 4)),
          ('relu1', nn.ReLU()),
          ('Dropout1',nn.Dropout(0.1)),
    
          ('fc2', nn.Linear(4, 2)),
          ('relu2', nn.ReLU()),
          ('Dropout2',nn.Dropout(0.1)),
    
          ('fc4', nn.Linear(2, 1)),

]))


def train(model, trainloader, valoader, epochs, lrate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    pos_weight = torch.tensor([0.80 / 0.20], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.SGD(model.parameters(), lr=lrate)
    scaler     = torch.cuda.amp.GradScaler()

    train_loss = []
    val_loss   = []
    train_perf = []
    val_perf   = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        running_val_loss   = 0.0
        running_train_score  = 0.0
        running_val_score    = 0.0

        train_loop = tqdm(trainloader, desc=f"Train Epoch {epoch+1}", leave=False)
        for data, labels in train_loop:
            data = data.to(device).float()
            labels = labels.to(device).float().view(-1,1)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.50).float()
            
            running_train_score += f1_score(labels.cpu().numpy(), preds.cpu().numpy())

        train_loss.append(running_train_loss)
        train_perf.append(running_train_score / len(trainloader.dataset))

        model.eval()
        with torch.no_grad():
            val_loop = tqdm(valoader, desc=f"Validation Epoch {epoch+1}", leave=False)
            for data, labels in val_loop:
                data   = data.to(device).float()
                labels = labels.to(device).float().view(-1,1)
                with torch.cuda.amp.autocast():
                    logits = model(data)
                    loss = criterion(logits, labels)
                
                running_val_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.50).float()
                running_val_score += f1_score(labels.cpu().numpy(), preds.cpu().numpy())

        val_loss.append(running_val_loss)
        val_perf.append(running_val_score / len(valoader.dataset))

        print(f"[Epoch {epoch+1}] Train loss: {running_train_loss/len(trainloader.dataset):.4f}  "
              f"Val loss: {running_val_loss/len(valoader.dataset):.4f}")
        print(f"Train F1-score: {train_perf[-1]:.4f}  Val F1-score: {val_perf[-1]:.4f}")

    return model, train_loss, val_loss, train_perf, val_perf


model, train_loss, val_loss, train_perf, val_perf = train(model, trainloader, valoader, 4, 1e-3)


fig, axes = plt.subplots(1, 2, figsize=(12,4))

axes[0].plot(train_loss, label="Train Loss")
axes[0].plot(val_loss, label="Val Loss")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(train_perf, label="Train Perf")
axes[1].plot(val_perf, label="Val Perf")
axes[1].set_title("Performance")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show();


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test = test.float()
model.eval()
y_pred = model(test.to(device))
y_pred = torch.sigmoid(y_pred)


y_pred = y_pred.cpu().detach().view(*IDs_test.shape).numpy()


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv', index_col='id')
submission['y'] = y_pred


submission


submission.to_csv('/kaggle/working/submission.csv')




