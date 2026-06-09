import numpy as np
import pandas as pd



train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')


train_labels.head()


train_labels.shape


train_sequences.head()


print(train_sequences.shape)


## left join ##
train_labels['ID'] = train_labels['ID'].str.rsplit('_', n=1).str[0]
train_df  = train_labels.merge(how = 'left' , left_on = 'ID' , right_on = 'target_id' , right = train_sequences )



print('No of rows and cols ' , train_df.shape)
print()
print('Missing values' , train_df.isna().sum())
print()


##datetime conversion ###
train_df['temporal_cutoff'] = pd.to_datetime(train_df['temporal_cutoff']).astype('int64') // 10**9
test_sequences['temporal_cutoff'] = pd.to_datetime(test_sequences['temporal_cutoff']).astype('int64') // 10**9
## using len of sequence 
train_df['seq_length'] = train_df['sequence'].str.len()
test_sequences['seq_length'] =  test_sequences['sequence'].str.len()


submission['ID']  = submission['ID'].str.rsplit('_' ,n =1  ).str[0]



test  = test_sequences.merge(how = 'left' , left_on = 'target_id' , right_on = 'ID' , right = submission)


# Define columns
categorical_cols = ['resname', 'target_id', 'description', 'all_sequences']
numerical_cols = ['temporal_cutoff', 'resid', 'seq_length']
target_cols = ['x_1', 'y_1', 'z_1']


# Encode categoricals
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_cols:
    
    le = LabelEncoder()
    train_df[col] = train_df[col].astype(str).fillna('missing')
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le



# Get vocab sizes
vocab_sizes = {col : train_df[col].nunique() + 1  for col in categorical_cols}


## dropna values 
train_df.dropna(inplace = True)


##Train test splits
from sklearn.model_selection import train_test_split

x_cols  = train_df.columns.difference(target_cols)
trainx , testx , trainy , testy = train_test_split(train_df[x_cols] , train_df[target_cols ] , test_size = 0.25 , random_state = 0)
print(trainx.shape)
print(trainy.shape)
print(testx.shape)
print(testy.shape)


## Model Building 
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Normalization , Dropout
from tensorflow.keras.models import Model


embedding_dim = 16 
categorical_inputs = []
embeddings = []

for col in categorical_cols:
    vocab_size = vocab_sizes[col]
    inp = Input(shape =(1,) , name = f'{col}_input')
    categorical_inputs.append(inp)
    emb = Embedding(input_dim = vocab_size , output_dim = embedding_dim)(inp)
    flatten = Flatten()(emb)
    embeddings.append(flatten)

concatenated_embeddings = Concatenate()(embeddings)

## Numericals ###
numerical_inputs = Input(shape = (len(numerical_cols) ,) , name = 'numerical_input')
norm_layer = Normalization()
norm_layer.adapt(trainx[numerical_cols].values)
normalized_num = norm_layer(numerical_inputs)

##concatenate ##
combined = Concatenate()([concatenated_embeddings , normalized_num])


x = Dense(1024, activation='relu')(combined)
x = Dropout(0.40)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.40)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.40)(x)
x = Dense(64, activation='relu')(x)
output = Dense(3)(x)

# Building Model ##

model = Model(inputs = categorical_inputs + [numerical_inputs] , outputs = output)

model.compile(loss = 'mse' , metrics = ['mae'] , optimizer = 'adam')


print(model.summary())


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

model_checkpoint =  ModelCheckpoint('best_model.keras' , save_best_only = True , mode = 'min' , monitor='val_loss' , verbose = 1)
early_stopping = EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss' , verbose = 1)


## preparing dataset for model trainiong ##
x_train_cat =  [trainx[col].values.reshape(-1 , 1) for col in categorical_cols]
x_train_num = trainx[numerical_cols].values
x_train = x_train_cat + [x_train_num]
y_train = trainy.values

x_val_cat = [testx[col].values.reshape(-1 , 1) for col in categorical_cols] 
x_val_num = testx[numerical_cols].values
x_val = x_val_cat + [x_val_num]
y_val = testy.values


history = model.fit(x_train , y_train , validation_data = (x_val , y_val) , epochs = 1000 , 
                    batch_size = 256, verbose = 1, callbacks = [model_checkpoint , early_stopping])





# Preprocess test data (ensure categoricals are label-encoded)

for col in categorical_cols:
    # Handle NaNs and unseen categories
    test[col] = test[col].astype(str).fillna('missing')

    le = label_encoders[col]
    known_classes = set(le.classes_)

    # Replace unseen labels with 'missing'
    test[col] = test[col].apply(lambda x: x if x in known_classes else 'missing')

    # Ensure 'missing' exists in label encoder
    if 'missing' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'missing')

    test[col] = le.transform(test[col])


# Prepare test inputs
X_test_cat = [test[col].values.reshape(-1, 1) for col in categorical_cols ]
X_test_num = test[numerical_cols].values
X_test = X_test_cat + [X_test_num]

# Generate predictions
predictions = model.predict(X_test)
submission[['x_1', 'y_1', 'z_1']] = predictions
# submission.to_csv('submission.csv', index=False)


submission[['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1']].head()


# Create x_2-x_5, y_2-y_5, z_2-z_5 columns with same values as x_1/y_1/z_1
for i in range(2, 6):
    submission[f'x_{i}'] = submission['x_1']
    submission[f'y_{i}'] = submission['y_1']
    submission[f'z_{i}'] = submission['z_1']


submission['ID'] = submission['ID'].astype(str) + '_' + submission['resid'].astype(str)



submission.to_csv('submission.csv' , index = False)




