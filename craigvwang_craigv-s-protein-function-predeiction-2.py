# Load the libraries.
import numpy as np 
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import locale
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,random_split,DataLoader
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


print(f"Numpy.version:{np.__version__} has been loaded sucessfully. ")
print(f"Pandas.version:{pd.__version__} has been loaded sucessfully. ")
print(f"Pytorch.version:{torch.__version__} has been loaded sucessfully. ")
print(locale.getpreferredencoding())


# Load the files.

# The t5embeds file path:
t5_test_embeds_file_path = "/kaggle/input/t5embeds/test_embeds.npy"
t5_test_ids_file_path = "/kaggle/input/t5embeds/test_ids.npy"
t5_train_embeds_file_path = "/kaggle/input/t5embeds/train_embeds.npy"
t5_train_ids_file_path = "/kaggle/input/t5embeds/train_ids.npy"

# The train terms file path:
train_terms_file_path = "/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv"

# The IA weight file path:
IA_file_path = "/kaggle/input/cafa-5-protein-function-prediction/IA.txt"

print("Files load successfully.")


# Define a function to read the *.tsv file.
def read_tsv_to_df(file_path):
    df = pd.read_csv(file_path,sep="\t")
    return df

print("Functions loading successfully.")


# Define a function to read the *.txt file.
def read_txt_to_df(file_path):
    with open(file_path,"r",encoding="UTF-8") as f:
        df_ia_weight = pd.DataFrame(columns=["terms","ia_weight"])
        i = 0
        for line in f:
            terms,ia_weight = line.strip().split()
            df_ia_weight.loc[i,"terms"] = terms
            df_ia_weight.loc[i,"ia_weight"] = float(ia_weight)
            i += 1
        return df_ia_weight

print("Functions loading successfully.")


# Put the data into DataFrame.

# Load the files.
t5_train_embeds = np.load(t5_train_embeds_file_path)
t5_test_embeds = np.load(t5_test_embeds_file_path)
t5_train_ids = np.load(t5_train_ids_file_path)
t5_test_ids = np.load(t5_test_ids_file_path)
df_train_terms = read_tsv_to_df(train_terms_file_path)
df_ia_weight = read_txt_to_df(IA_file_path)

# Check the shape of the files.
print(t5_train_embeds.shape)
print(t5_train_ids.shape)
print(t5_test_embeds.shape)
print(t5_test_ids.shape)
print(df_train_terms.shape)
print(df_ia_weight.shape)

# t5_train_embeds and t5_test_embeds share the same numeric value of the dimision for the embeddings.
# embeds_dim = t5_train_embeds.shape[1]

# Put t5_train_embeds, t5_test_embeds, t5_train_ids, t5_test_ids into DataFrame.
# df_train_embeds = pd.DataFrame(t5_train_embeds,columns=["feature_"+str(i) for i in range(1,embeds_dim+1)])
# df_test_embeds = pd.DataFrame(t5_test_embeds,columns=["feature_"+str(i) for i in range(1,embeds_dim+1)])
df_train_ids = pd.DataFrame(t5_train_ids,columns=["EntryID"])
# df_test_ids = pd.DataFrame(t5_test_ids,columns=["EntryID"])

# Check the dataframes.
# print(df_train_embeds.head())
# print(df_test_embeds.head())
print(df_train_ids.head())
# print(df_test_ids.head())
print(df_train_terms.head())
print(df_ia_weight.head())


# Prepare the label y of trainset.
# Each index of the labels must be as the same as the features‘.

# Check the total number of the GO terms
print("The total number of the GO terms:")
print(df_train_terms.term.nunique())

# Check the count number of the terms values. 
value_count = df_train_terms.term.value_counts()
print(value_count)

# Confirm the number to be 1500, and use it to select the GO terms of the first 1500 in the value_count series.
select_value_count = value_count[0:1500]
print("The count number of the selected GO terms:")
print(select_value_count)

# Put the first 1600 high frequency terms into a list named "select_terms".
select_terms = select_value_count.index.tolist()
# print(select_terms)
print(len(select_terms))

# Drop the terms that haven't be chosen in the 1500, and create the selected terms dataframe.
df_select_terms = df_train_terms.loc[df_train_terms.term.isin(select_terms)]
print(len(df_select_terms))


# Get all the GO terms in the select_terms each protein has. 
series_term_lst = df_select_terms.groupby("EntryID").term.unique()
print(series_term_lst[0:5])
print(type(series_term_lst)) 

# Transform the series into a dataframe with the index reset.
df_label_obj = pd.DataFrame(series_term_lst).reset_index()
print(df_label_obj)

# Merge the dataframes to sort by the order of df_train_ids ,corresponding to df_train_embeds.
df_train_ordered = df_train_ids.merge(df_label_obj,on="EntryID",how="left")
print(df_train_ordered)
print(df_train_ordered["term"])

# Use the MultiLAbelBinarizer to do multi binary classifacation.
myMLB = MultiLabelBinarizer()
df_train_encode_array = myMLB.fit_transform(df_train_ordered["term"])
df_train_encode = pd.DataFrame(df_train_encode_array,columns=myMLB.classes_)
label_order = myMLB.classes_

print(df_train_encode_array)
print(df_train_encode)
print(df_train_encode.loc[0,"GO:0044249"])
print(label_order)

# Get the list of the IA weight in labels' order.
lst_label_weight = []
for label in label_order:
    label_weight = df_ia_weight.loc[df_ia_weight["terms"] == label].ia_weight.values[0]
    lst_label_weight.append(label_weight)
print(len(lst_label_weight))


# Prepare the dataset.

# load the numpy ndarray.
X_np = t5_train_embeds
y_np = df_train_encode_array
test_np = t5_test_embeds

# Transform them into tensor.
X_tensor = torch.from_numpy(X_np).float()
y_tensor = torch.from_numpy(y_np).float()
test_tensor = torch.from_numpy(test_np).float()
weights_tensor = torch.tensor(lst_label_weight,dtype=torch.float32)

# Create the dataset
dataset_total = TensorDataset(X_tensor,y_tensor)

# Split the dataset into train set and validation set.
train_size = int(0.8 * len(dataset_total))
valid_size = len(dataset_total) - train_size
train_set,valid_set = random_split(dataset_total,[train_size,valid_size])

# Prepare the whole trainset and testset.
train_set_whole = dataset_total
test_set = test_tensor

print(weights_tensor)
print("Loading successfully.")


# Set up the models.
network = nn.Sequential(
    nn.Linear(1024,512),nn.ReLU(),
    nn.Linear(512,512),nn.ReLU(),
    nn.Linear(512,512),nn.ReLU(),
    nn.Linear(512,512),nn.ReLU(),
    nn.Linear(512,1500)
)

network_dropout = nn.Sequential(
    nn.Linear(1024,1024),nn.ReLU(),
    nn.Linear(1024,1024),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(1024,1024),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(1024,1024),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(1024,1500)
)

network_deep_drop = nn.Sequential(
    nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(512,512),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(512,1500)
)

network_deep_drop_bt_norm_0 = nn.Sequential(
    nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(512,1500)
)

network_deep_drop_bt_norm_1 = nn.Sequential(
    nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(512,1500)
)

network_deep_drop_bt_norm_2 = nn.Sequential(
    nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(512,1500)
)

print("Setting successfully.")


# Define a loss function to using the IA weights.
class ia_weighted_BCEWithLogitsLoss(nn.Module):
    def __init__(self,weights):
        super().__init__()
        self.weights = weights

    def forward(self,logits,targets):
        ele_loss = nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction="none")
        weighted_loss = ele_loss * self.weights + ele_loss
        return weighted_loss.mean()

# Define a function to get the accuracy of the batch.
def get_accuracy(y_hat,y,threshold=0.5):
    y_probs = torch.sigmoid(y_hat)
    y_pred_label = (y_probs >= threshold).float()
    num_acc = (y_pred_label == y).sum().item()
    return num_acc / y.numel()

# Define a function to get the loss and accuracy.
def get_loss_and_acc(l,X,y,y_hat,running_loss,total_correct,total_labels):
    running_loss += l.item() * X.size(0)
    batch_accuracy = get_accuracy(y_hat,y,threshold=0.5)
    total_correct += batch_accuracy * y.numel()
    total_labels += y.numel()
    return running_loss,total_correct,total_labels
    
# Print the results of loss and accuracy.
def get_loss_accuracy(running_loss,total_correct,total_labels,data_set,epoch):
    epoch_loss = running_loss / len(data_set)
    epoch_acc = 100 * total_correct / total_labels
    return epoch_loss,epoch_acc

print("Functions loading successfully.")


# Define the function to train and score.
def train_and_score(net,train_set,valid_set,batch_size,weight_decay,num_epochs,learning_rate,):
    # Create dataloader
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(valid_set,batch_size=batch_size,shuffle=False)

    # Set updater
    updater = optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)

    # Set Loss function
    loss = ia_weighted_BCEWithLogitsLoss(weights_tensor)

    # Set the loss and accuracy list.
    train_losses,train_accs,valid_losses,valid_accs = [],[],[],[]
            
    # Train and valid.
    for epoch in range(num_epochs):
        
        # Set the initial value.
        running_loss = 0
        total_correct = 0
        total_labels = 0
        
        # Start to train.
        net.train()
        for X,y in train_loader:
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            updater.step()
            running_loss,total_correct,total_labels = get_loss_and_acc(l,X,y,y_hat,running_loss,total_correct,total_labels)
        train_loss,train_acc = get_loss_accuracy(running_loss,total_correct,total_labels,train_set,epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Reset the initial value.
        running_loss = 0
        total_correct = 0
        total_labels = 0
        
        # start to valid.
        net.eval()
        with torch.no_grad():
            for X,y in valid_loader:
                y_hat = net(X)
                l = loss(y_hat,y)
                running_loss,total_correct,total_labels = get_loss_and_acc(l,X,y,y_hat,running_loss,total_correct,total_labels)
        valid_loss,valid_acc = get_loss_accuracy(running_loss,total_correct,total_labels,valid_set,epoch)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        print(f"Epoch: {epoch + 1} / {num_epochs}")
        print(f"Train set: Loss: {train_loss:.4f} | Accuracy: {train_acc:.3f}%")
        print(f"Valid set: Loss: {valid_loss:.4f} | Accuracy: {valid_acc:.3f}%")
        print("-" * 50)

    # Set the plot.
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    
    x_data = list(range(1,num_epochs + 1))
    
    ax1.plot(x_data,train_losses,"b-",label="Training loss")
    ax1.plot(x_data,valid_losses,"r--",label="Validating loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(x_data,train_accs,"g-",label="Training accuracy")
    ax2.plot(x_data,valid_accs,"m--",label="Validating accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
        
    plt.tight_layout()
    
    plt.draw()
    plt.pause(0.1)

print("Function loading successfully.")


# Set num_epochs.
num_epochs = 30
# Set learning_rate.
learning_rate = 0.001
# set weight decay range in (0.0001-0.01)
weight_decay = 0.00001
# Set batch size
batch_size = 5120

print("Loading successfully.")


# Test the model.
# Comment the models that have not been selected.
# train_and_score(
#     network,train_set,valid_set,batch_size=batch_size,weight_decay=weight_decay,\
#     num_epochs=20,learning_rate=learning_rate,
# )

# train_and_score(
#     network_dropout,train_set,valid_set,batch_size=batch_size,weight_decay=weight_decay,\
#     num_epochs=20,learning_rate=learning_rate,
# )

# train_and_score(
#     network_deep_drop,train_set,valid_set,batch_size=batch_size,weight_decay=weight_decay,\
#     num_epochs=20,learning_rate=learning_rate,
# )


# train_and_score(
#     network_deep_drop_bt_norm_0,train_set,valid_set,batch_size=batch_size,weight_decay=weight_decay,\
#     num_epochs=30,learning_rate=learning_rate,
# )

# train_and_score(
#     network_deep_drop_bt_norm_1,train_set,valid_set,batch_size=batch_size,weight_decay=weight_decay,\
#     num_epochs=30,learning_rate=learning_rate,
# )

# train_and_score(
#     network_deep_drop_bt_norm_2,train_set,valid_set,batch_size=batch_size,weight_decay=weight_decay,\
#     num_epochs=30,learning_rate=learning_rate,
# )


# Use it to train the whole trainset.
def train_and_pred(net,train_set,test_set,batch_size,weight_decay,num_epochs,learning_rate,):
    # Create dataloader
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    
    # Set updater
    updater = optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)

    # Set Loss function
    loss = ia_weighted_BCEWithLogitsLoss(weights_tensor)

    # Set the loss and accuracy list.
    train_losses,train_accs = [],[]
            
    # Train and valid.
    for epoch in range(num_epochs):
        
        # Set the initial value.
        running_loss = 0
        total_correct = 0
        total_labels = 0
        
        # Start to train.
        net.train()
        for X,y in train_loader:
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            updater.step()
            running_loss,total_correct,total_labels = get_loss_and_acc(l,X,y,y_hat,running_loss,total_correct,total_labels)
        train_loss,train_acc = get_loss_accuracy(running_loss,total_correct,total_labels,train_set,epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"Epoch: {epoch + 1} / {num_epochs}")
        print(f"Train set: Loss: {train_loss:.4f} | Accuracy: {train_acc:.3f}%")
        print("-" * 50)

    # Set the plot.
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
    
    x_data = list(range(1,num_epochs + 1))
    
    ax1.plot(x_data,train_losses,"b-",label="Training loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(x_data,train_accs,"g-",label="Training accuracy")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
        
    plt.tight_layout()
    
    plt.draw()
    plt.pause(0.1)
    
    # start to valid.
    net.eval()
    with torch.no_grad():
        test_preds = net(test_set)
        predictions = torch.sigmoid(test_preds)
    return predictions

print("Function loading successfully.") 


# The best model is network_deep_drop_bt_norm_0, choose it.
final_model = network_deep_drop_bt_norm_1


# Train and get the final prediction.
final_predictions = train_and_pred(final_model,train_set_whole,test_set,batch_size,weight_decay,num_epochs=50,learning_rate=0.001,)


# Put the predictions into a dataframe by pandas.
print(final_predictions.shape)
df_final_pred = pd.DataFrame(final_predictions,index=t5_test_ids,columns=label_order)
print(df_final_pred.head())

# Transform the dataframe into the shape that we want.
df_final_reset = df_final_pred.reset_index().rename(columns={"index":"Protein Id"})
df_submit = df_final_reset.melt(
    id_vars="Protein Id",
    value_vars=label_order,
    var_name="GO Terms",
    value_name="Prediction"
)

print(df_final_reset.head())
print(df_submit.head())
print(df_submit.shape)


# save the submission file to output.
df_submit.to_csv("submission.tsv",header=False,index=False,sep="\t")

