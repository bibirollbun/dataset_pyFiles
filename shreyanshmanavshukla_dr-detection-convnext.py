!nvidia-smi


import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, mixed_precision


import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB_API_KEY")

wandb.login(key=secret_value_0)


# import tensorflow as tf
print(tf.__version__)
# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from allocating all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and will be used.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error: {e}")
else:
    print("GPU is not available, running on CPU.")

tf.test.is_gpu_available()

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("Mixed precision enabled:", policy)


IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3


train_csv_path = "/kaggle/input/diabetic-retinopathy-detection/trainLabels.csv.zip"
train_images_dir = "/kaggle/input/diabetic-retinopathy-train-unzipped/train/"
test_images_dir = "/kaggle/input/diabetic-retinopathy-test-unzipped/test/"
submission_csv_path = "/kaggle/input/diabetic-retinopathy-detection/sampleSubmission.csv.zip"


img_number = 10016

df = pd.read_csv(train_csv_path)
df_train = df[:img_number].copy()
df_train.head()


# Convert multiclass level to binary: 0 = No DR, 1 = DR Present
df_train["Label"] = df_train["level"].apply(lambda x: False if x == 0 else True)

df_train["filepath"] = df_train["image"].apply(lambda x: os.path.join(train_images_dir, f"{x}.jpeg"))

train_df, val_df = train_test_split(df_train, test_size=0.4, stratify=df_train["Label"], random_state=42)
val_df, test_df = train_test_split(val_df, test_size=0.5, stratify=val_df["Label"], random_state=42)



print(train_df.shape, val_df.shape, test_df.shape)
# train_df.filepath[0]
train_df


import cv2
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))

# Show first 25 images from train_df
for idx, (i, row) in enumerate(train_df.head(25).iterrows()):
    plt.subplot(5, 5, idx + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    # Load and preprocess image
    img_path = row['filepath']
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Warning: Image not found at {img_path}")
        continue
    
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Set label from binary column
    label = row.get('binary_label', row.get('Label', None))
    label_text = 'Yes' if label else 'No'
    
    plt.imshow(img)
    plt.xlabel(label_text, fontsize=12, color='green' if label else 'red')

plt.tight_layout()
plt.suptitle("Sample Images with Binary DR Labels", fontsize=20, y=1.02)
plt.show()






# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
#     average_precision_score, confusion_matrix, cohen_kappa_score,
#     precision_recall_curve, roc_curve
# )
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import os

# def Results(model, model_name,test_set, save_csv=False):
#     # Evaluate the model
#     results = model.evaluate(test_set)
#     print("Evaluation results: ", results)

#     y_true = []
#     y_pred_probs = []

#     print("Loading....", flush=True)

#     for images, labels in test_set:
#         y_true.extend(labels.numpy())
#         y_pred_probs.extend(model.predict(images, verbose=0).flatten())

#     y_true = tf.convert_to_tensor(y_true)
#     y_pred_probs = tf.convert_to_tensor(y_pred_probs)
#     y_pred = (y_pred_probs > 0.5).numpy().astype("int32")

#     acc = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, zero_division=1)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     roc_auc = roc_auc_score(y_true, y_pred_probs)
#     prc_auc = average_precision_score(y_true, y_pred_probs)
#     conf_matrix = confusion_matrix(y_true, y_pred)
#     kappa = cohen_kappa_score(y_true, y_pred)
#     tn, fp, fn, tp = conf_matrix.ravel()
#     npv = tn / (tn + fn)

#     print(f"Accuracy: {acc}")
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"F1 Score: {f1}")
#     print(f"ROC AUC: {roc_auc}")
#     print(f"PRC AUC: {prc_auc}")
#     print(f"Confusion Matrix: \n{conf_matrix}")
#     print(f"Kappa Coefficient: {kappa}")
#     print(f"NPV: {npv}")

#     if save_csv:
#         # Compute ROC and Precision-Recall curves
#         fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
#         prec, rec, _ = precision_recall_curve(y_true, y_pred_probs)

#         # Prepare DataFrame
#         max_len = max(len(fpr), len(prec))
#         fpr = np.pad(fpr, (0, max_len - len(fpr)), 'constant', constant_values=np.nan)
#         tpr = np.pad(tpr, (0, max_len - len(tpr)), 'constant', constant_values=np.nan)
#         prec = np.pad(prec, (0, max_len - len(prec)), 'constant', constant_values=np.nan)
#         rec = np.pad(rec, (0, max_len - len(rec)), 'constant', constant_values=np.nan)

#         df = pd.DataFrame({
#             'FPR': fpr,
#             'TPR': tpr,
#             'Precision': prec,
#             'Recall': rec
#         })


#         file_name = model_name
#         df.to_csv(file_name, index=False)
#         print(f"Saved metrics to: {file_name}")



from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, cohen_kappa_score,
    precision_recall_curve, roc_curve
)
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt

def Results(model, test_set, model_name="Model", save_csv=False, wandb_log=False):
    # Evaluate the model
    results = model.evaluate(test_set)
    print("Evaluation results: ", results)

    y_true = []
    y_pred_probs = []

    print("Loading....", flush=True)

    for images, labels in test_set:
        y_true.extend(labels.numpy())
        y_pred_probs.extend(model.predict(images, verbose=0).flatten())

    y_true = tf.convert_to_tensor(y_true)
    y_pred_probs = tf.convert_to_tensor(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).numpy().astype("int32")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    prc_auc = average_precision_score(y_true, y_pred_probs)
    conf_matrix = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    npv = tn / (tn + fn)

    # Print metrics
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")
    print(f"PRC AUC: {prc_auc}")
    print(f"Confusion Matrix: \n{conf_matrix}")
    print(f"Kappa Coefficient: {kappa}")
    print(f"NPV: {npv}")

    # Log metrics to W&B
    if wandb_log:
        wandb.log({
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "PRC AUC": prc_auc,
            "Kappa": kappa,
            "NPV": npv
        })

    if save_csv or wandb_log:
        # Compute ROC and Precision-Recall curves
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        prec, rec, _ = precision_recall_curve(y_true, y_pred_probs)

        # Pad arrays to same length
        max_len = max(len(fpr), len(prec))
        fpr = np.pad(fpr, (0, max_len - len(fpr)), 'constant', constant_values=np.nan)
        tpr = np.pad(tpr, (0, max_len - len(tpr)), 'constant', constant_values=np.nan)
        prec = np.pad(prec, (0, max_len - len(prec)), 'constant', constant_values=np.nan)
        rec = np.pad(rec, (0, max_len - len(rec)), 'constant', constant_values=np.nan)

        df = pd.DataFrame({
            'FPR': fpr,
            'TPR': tpr,
            'Precision': prec,
            'Recall': rec
        })

        file_name = model_name+'.metrices.csv'
        df.to_csv(file_name, index=False)
        print(f"Saved metrics to: {file_name}")

        if wandb_log:
            wandb.save(file_name)

            # Plot ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            wandb.log({"ROC Curve": wandb.Image(plt)})
            plt.close()

            # Plot PR Curve
            plt.figure()
            plt.plot(rec, prec, label=f'PRC AUC = {prc_auc:.2f}')
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            wandb.log({"PR Curve": wandb.Image(plt)})
            plt.close()



import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def get_flops(model, input_shape):
    # Convert Keras model to ConcreteFunction
    concrete_func = tf.function(model).get_concrete_function(tf.TensorSpec(input_shape, tf.float32))
    
    # Convert ConcreteFunction to a frozen graph
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    
    # Create a session to run the frozen graph
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")
        graph = tf.compat.v1.get_default_graph()
        
        # Calculate FLOPs using TensorFlow profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        
    return flops.total_float_ops


# # Define input shape
# input_shape = (1, 224, 224,3 )
# # Calculate FLOPs
# flops = get_flops(model, input_shape)
# macs = flops // 2  # MACs is half the FLOPs for Conv2D

# print(f"MACs: {macs}")
# print(f"FLOPs: {flops}")


import numpy as np

def model_parametersInfo(model):
    """
    Prints total, trainable, and non-trainable parameters of a Keras model.
    """
    total_params = model.count_params()
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_variables])

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")



import tensorflow as tf
import psutil
import os
import time

def inference_Report(model, test_set):
    process = psutil.Process(os.getpid())

    if tf.config.list_physical_devices('GPU'):
        gpu_enabled = True
        gpu_device = 'GPU:0'
    else:
        gpu_enabled = False

    print("ğŸ”� Capturing memory stats and inference time for 100 images...\n")
    # --- Before Inference ---
    ram_before = process.memory_info().rss
    gpu_before = tf.config.experimental.get_memory_info(gpu_device)['current'] if gpu_enabled else 0

    total_images = 0
    start_time = time.time()

    for images, _ in test_set:
        for i in range(images.shape[0]):
            if total_images >= 100:
                break
            image = tf.expand_dims(images[i], axis=0)
            _ = model(image, training=False)
            total_images += 1
        if total_images >= 100:
            break

    end_time = time.time()

    # --- After Inference ---
    ram_after = process.memory_info().rss
    gpu_after = tf.config.experimental.get_memory_info(gpu_device)['current'] if gpu_enabled else 0

    # --- Results ---
    print("ğŸ“Š Inference Resource Usage Summary (100 images):")
    print(f"CPU RAM Before: {ram_before / (1024 ** 2):.2f} MB")
    print(f"CPU RAM After : {ram_after / (1024 ** 2):.2f} MB")
    print(f"CPU RAM Used  : {(ram_after - ram_before) / (1024 ** 2):.2f} MB\n")

    if gpu_enabled:
        print(f"GPU Mem Before: {gpu_before / (1024 ** 2):.2f} MB")
        print(f"GPU Mem After : {gpu_after / (1024 ** 2):.2f} MB")
        print(f"GPU Mem Used  : {(gpu_after - gpu_before) / (1024 ** 2):.2f} MB\n")
    else:
        print("GPU Not Detected.\n")
    time_per_img = (end_time - start_time)/100
    print(f"ğŸ•’ Inference Time per image: {time_per_img:.4f} seconds")




def retreive_dataset(set_name):
    images,labels=[],[]
    def imgResize(img,width,height):
        imgResize = cv2.resize(img,(width,height))
        return imgResize
    
    for (img, imclass) in zip(set_name['filepath'], set_name['Label']):
        img = cv2.imread(img)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #BGR to RGB
        img = cv2.resize(img, (224, 224))    
        # img = img.astype(np.float32) / 255.0   #preprocess
        images.append(img)
        if(imclass==True):
            labels.append(1)
        else:
            labels.append(0)
    print(f"Done")
    return np.array(images),np.array(labels)


X_train,y_train=retreive_dataset(train_df)
X_val,y_val=retreive_dataset(val_df)
X_test,y_test=retreive_dataset(test_df)




index = 0  # Change this to view another image
image = X_train[index]
label = y_train[index]

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title(f"Label: {'DR Present' if label == 1 else 'No DR'}")
plt.axis('off')
plt.show()


train_set_raw=tf.data.Dataset.from_tensor_slices((X_train,y_train))
valid_set_raw=tf.data.Dataset.from_tensor_slices((X_val,y_val))
test_set_raw=tf.data.Dataset.from_tensor_slices((X_test,y_test))


tf.keras.backend.clear_session()  # extra code â€“ resets layer name counter

batch_size = 32
preprocess = tf.keras.applications.convnext.preprocess_input
train_set = train_set_raw.map(lambda X, y: (preprocess(tf.cast(X, tf.float32)), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(lambda X, y: (preprocess(tf.cast(X, tf.float32)), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(tf.cast(X, tf.float32)), y)).batch(batch_size)


plt.figure(figsize=(12, 12))
for X_batch, y_batch in train_set.take(1):
    for index in range(9):
        plt.subplot(3, 3, index + 1)
        img = (X_batch[index].numpy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img /np.max(img)) # rescale to 0â€“1 for imshow()
        if(y_batch[index]==1):
            classt= 'DR Present'
        else:
            classt= "No DR"
        plt.title(f"Class: {classt}")
        plt.axis("off")

plt.show()


tf.random.set_seed(42)  # Ensures reproducibility

tf.keras.backend.clear_session()
# Ensures reproducibility


base_model = tf.keras.applications.ConvNeXtBase(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(1, activation="sigmoid")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="binary_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])





wandb.init(
    project="Diabetic Retinopathy Detection",   # change this
    name="ConvNeXtBase",        # optional
    config={                       # optional: log hyperparameters
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    }
)


model_name = "DR_ConvNeXtBase"
model_name


from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
# from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint



# checkpoint_callback = ModelCheckpoint(
#     filepath='best_model.keras',  # Path to save the model
#     monitor='val_accuracy',        # yMetric to monitor
#     save_best_only=True,       # Save only the best model
#     save_weights_only=False,   # Save the entire model, not just weights
#     mode='max',                # Mode to minimize the monitored metric
#     verbose=1                  # Verbosity mode
# )
csv_logger = CSVLogger(model_name+'.csv',append = True)

# callbacks = [checkpoint_callback,csv_logger]
callbacks = [csv_logger,wandb.keras.WandbMetricsLogger(log_freq=5)]





history = model.fit(train_set, validation_data=valid_set, epochs=10, callbacks= callbacks)


# model.save_weights(model_name+'.weights.h5')

model.load_weights('/kaggle/working/DR_ConvNeXtBase.weights.h5')


Results(model,test_set,model_name=model_name,save_csv=True,wandb_log=False)


wandb.save(f"/kaggle/working/{model_name}.metrices.csv", policy="now")
wandb.save(f"/kaggle/working/{model_name}.weights.h5", policy="now")



inference_Report(model,test_set)
model_parametersInfo(model)


# Define input shape
input_shape = (1, 224, 224,3 )
# Calculate FLOPs
flops = get_flops(model, input_shape)
macs = flops // 2  # MACs is half the FLOPs for Conv2D

print(f"MACs: {macs}")
print(f"FLOPs: {flops}")


wandb.finish()


n =len(base_model.layers)
L = int(0.3*n)  # 30% of n
f"{L} trainable of {n} layers"


for layer in base_model.layers[-L:]:
    layer.trainable = True


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="binary_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])


wandb.init(
    project="Diabetic Retinopathy Detection", 
    name="CovNeXtBase_ft",   # Change this to "VGG_Model_Run", etc. for next model
    resume="allow"
)



model_name = model_name+'_ft'
model_name


from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

csv_logger = CSVLogger(model_name+'.csv',append = True)

callbacks = [csv_logger,wandb.keras.WandbMetricsLogger(log_freq=5)]


history = model.fit(train_set, validation_data=valid_set, epochs=10, callbacks= callbacks)



model.save_weights(model_name+'.weights.h5')

# model.load_weights('/kaggle/working/DR_Xception.weights.h5')



Results(model,test_set,model_name=model_name,save_csv=True,wandb_log=True)


wandb.save(f"/kaggle/working/{model_name}.metrices.csv", policy="now")
wandb.save(f"/kaggle/working/{model_name}.weights.h5", policy="now")
print(f"saved to wandb {model_name}")


inference_Report(model,test_set)
model_parametersInfo(model)


# Define input shape
input_shape = (1, 224, 224,3 )
# Calculate FLOPs
flops = get_flops(model, input_shape)
macs = flops // 2  # MACs is half the FLOPs for Conv2D

print(f"MACs: {macs}")
print(f"FLOPs: {flops}")


wandb.finish()




