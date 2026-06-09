#pip install -U LibRecommender -q


!git clone https://github.com/jihongyan/LibRecommender.git
%cd LibRecommender
!pip install .


!pip install --upgrade tensorflow==2.12.0


import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from libreco.data import DataInfo, DatasetFeat, split_by_ratio_chrono, random_split
from libreco.algorithms import ItemCF, UserCF, LightGCN, YouTubeRanking, YouTubeRetrieval, DeepFM, GraphSage, GraphSageDGL, PinSage, PinSageDGL, TwoTower
import libreco


print(tf.__version__)


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train = pd.concat([train, original], axis=0, ignore_index=True)


train.reset_index(inplace=True)
train.drop('id', axis=1, inplace=True)
train.rename(columns={'index': 'user', 'Fertilizer Name':'item'}, inplace=True)
train.head(-10)


item_encoder = LabelEncoder()
train['item'] = item_encoder.fit_transform(train['item'])


train['label'] = 1
train.head()


#train['user_feathers'] = train['Temparature'].astype(str) + train['Humidity'].astype(str) + train['Moisture'].astype(str) + train['Soil Type'].astype(str) + train['Crop Type'].astype(str) + train['Nitrogen'].astype(str) + train['Potassium'].astype(str) + train['Phosphorous'].astype(str)

#train.head(-10)


train.insert(1, 'item', train.pop('item'))
train.insert(2, 'label', train.pop('label'))
train['Soil Type']= train['Soil Type'].astype(str).apply(lambda x: x.strip())
train['Crop Type']= train['Crop Type'].astype(str).apply(lambda x: x.strip())
train.head(-10)


train_datas, eval_datas = random_split(train, multi_ratios=[0.8, 0.2], filter_unknown=False, pad_unknown=True)


eval_datas.info()


sparse_col = ['Soil Type', 'Crop Type']

dense_col = [
    'Temparature',
    'Humidity',
    'Moisture',
    'Nitrogen',
    'Potassium',
    'Phosphorous'
]

user_col = sparse_col + dense_col

train_data, data_info = DatasetFeat.build_trainset(
    train_datas,
    user_col=user_col,
    item_col=None,
    sparse_col=sparse_col,
    dense_col=dense_col
)

eval_data = DatasetFeat.build_evalset(eval_datas)

print(data_info.sparse_col)


metrics = [
   "loss",
   "balanced_accuracy",
   "roc_auc",
   "pr_auc",
   "precision",
   "recall",
   "map",
   "ndcg"
]

#user_cf = UserCF(
#    data_info=data_info,
#    k_sim=10,
#    sim_type="cosine",
#    mode="invert",
#    num_threads=2,
#    min_common=1
#)

#user_cf.fit(
#    train_data,
#    neg_sampling=True,
#    verbose=2,
#    eval_data=eval_data,
#    metrics=metrics
#)


def path_exists_with_regex(directory, pattern):
    if not os.path.exists(directory):
        return False
    for entry in os.scandir(directory):
        if re.match(pattern, entry.name):
            return True
    return False


# Save results
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


# Step 4: Initialize and train LightGCN
#reset_state("LightGCN")
# if path_exists_with_regex("/kaggle/working/LibRecommender/output", 'lightGCN_model_*'):
#     # load data_info
#     data_info = DataInfo.load("/kaggle/working/LibRecommender/output", model_name="lightGCN_model", manual=True)
#     print(data_info)
#     # load model, should specify the model name, e.g., DeepFM
#     model = DeepFM.load(
#         path="/kaggle/working/LibRecommender/output", model_name="lightGCN_model", data_info=data_info
#     )
# else:
    # model = LightGCN(
    #    task="ranking",
    #    data_info=data_info,
    #    loss_type="bpr",
    #    embed_size=16,
    #    n_epochs=5,
    #    lr=1e-3,
    #    batch_size=2048,
    #    num_neg=1,
    #    device='cuda'
    # )


# #model.fit(train_data, neg_sampling=True, verbose=2, eval_data=eval_data, metrics=metrics, k=3)
#model.fit(train_data, neg_sampling=True, verbose=2)
#model.save('/kaggle/working/LibRecommender/output', 'lightGCN_model', manual=True, inference_only=True)


# reset_state("DeepFM")

# if path_exists_with_regex("/kaggle/working/LibRecommender/output", 'deepfm_model_*'):
#     #model.rebuild_model(path="/kaggle/working/LibRecommender/output", model_name="deepfm_model", full_assign=True)
#     #load data_info
#     data_info = DataInfo.load("/kaggle/working/LibRecommender/output", model_name="deepfm_model")
#     print(data_info)
#     # load model, should specify the model name, e.g., DeepFM
#     model = DeepFM.load(
#         path="/kaggle/working/LibRecommender/output", model_name="deepfm_model", data_info=data_info, manual=True
#     )
# else:
#     model = DeepFM(
#         task="ranking",
#         data_info=data_info,
#         loss_type="cross_entropy",
#         embed_size=16,
#         n_epochs=5,
#         lr=1e-3,
#         use_bn=True,
#         batch_size=2048,
#         hidden_units=(128, 64, 32),
#         num_neg=1
#     )


# #model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True, eval_data=eval_data, metrics=metrics, k=3)
# model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
# data_info.save('/kaggle/working/LibRecommender/output', 'deepfm_model')
# model.save('/kaggle/working/LibRecommender/output', 'deepfm_model', manual=True, inference_only=True)


reset_state("YouTubeRanking")
if path_exists_with_regex("/kaggle/working/LibRecommender/output", 'youTubeRanking_model_*'):
    # load data_info
    data_info = DataInfo.load("/kaggle/working/LibRecommender/output", model_name="youTubeRanking_model")
    print(data_info)
    # load model, should specify the model name, e.g., DeepFM
    model = YouTubeRanking.load(
        path="/kaggle/working/LibRecommender/output", model_name="youTubeRanking_model", data_info=data_info, manual=True
    )
else:
    model = YouTubeRanking(
        task="ranking",
        data_info=data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=5,
        lr=1e-3,
        use_bn=True,
        batch_size=2048,
        hidden_units=(128, 64, 32),
        num_neg=1
    )


#model.fit(train_data, neg_sampling=True, verbose=2, eval_data=eval_data, shuffle=True, metrics=metrics, k=3)
model.fit(train_data, neg_sampling=True, verbose=2)
data_info.save('/kaggle/working/LibRecommender/output', 'youTubeRanking_model')
model.save('/kaggle/working/LibRecommender/output', 'youTubeRanking_model', manual=True, inference_only=True)


# reset_state("GraphSage")
# if path_exists_with_regex("/kaggle/working/LibRecommender/output", 'graphSage_model_*'):
#     # load data_info
#     data_info = DataInfo.load("/kaggle/working/LibRecommender/output", model_name="graphSage_model")
#     print(data_info)
#     # load model, should specify the model name, e.g., DeepFM
#     model = GraphSage.load(
#         path="/kaggle/working/LibRecommender/output", model_name="graphSage_model", data_info=data_info, manual=True
#     )
# else:
#     model = GraphSage(
#         "ranking",
#         data_info,
#         loss_type="max_margin",
#         paradigm="u2i",
#         embed_size=16,
#         n_epochs=5,
#         lr=3e-4,
#         lr_decay=False,
#         reg=None,
#         batch_size=2048,
#         num_neg=1,
#         dropout_rate=0.0,
#         num_layers=1,
#         num_neighbors=10,
#         num_walks=10,
#         sample_walk_len=5,
#         margin=1.0,
#         sampler="random",
#         start_node="random",
#         focus_start=False,
#         seed=42,
#     )


# #model.fit(train_data, neg_sampling=True, verbose=2, eval_data=eval_data, shuffle=True, metrics=metrics, k=3)
# model.fit(train_data, neg_sampling=True, verbose=2)
# data_info.save('/kaggle/working/LibRecommender/output', 'graphSage_model')
# model.save('/kaggle/working/LibRecommender/output', 'graphSage_model', manual=True, inference_only=True)


# MAP@3 metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


eval_datas.reset_index(drop=True, inplace=True)
eval_datas.head()


eval_y = eval_datas["item"]
eval_datas.pop('item')
eval_datas.pop('label')
eval_y = [[label] for label in eval_y]
eval_pred = []
for i in range(eval_datas.shape[0]):
    user_features = eval_datas.iloc[i]
    id = user_features.pop('user')
    recommend = model.recommend_user(user=id, n_rec=7, cold_start="popular", user_feats=dict(user_features))
    eval_pred.append(recommend[id])
score = mapk(eval_y, eval_pred)
print(f"MAP@3 Score: {score:.5f}")


#model.recommend_user(user=750000, n_rec=7, cold_start="popular", user_feats={"Temparature": 31, "Humidity":70, "Moisture": 52, "Soil Type":'Sandy', "Crop Type":'Wheat', "Nitrogen": 34, "Potassium":11, "Phosphorous": 24})


for i in range(test.shape[0]):
    user_features = test.iloc[i]
    id = user_features.pop('id')
    recommend = model.recommend_user(user=id, n_rec=3, cold_start="popular", user_feats=dict(user_features))
    labels = item_encoder.inverse_transform(recommend[id])
    labels = ' '.join(labels)
    submission.loc[submission['id'] == id, 'Fertilizer Name'] = labels
submission.head()
submission.to_csv('/kaggle/working/submission.csv', index=False)

