!pip install open-clip-torch


import open_clip
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from einops import repeat
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

from torchvision import transforms as T


def load_and_process_video(path):
    """Custom pre-processing based on Echo2022 dataset."""

    transform = T.Compose([
        T.Resize(size=(358, 224)),
        T.CenterCrop(224),
        T.Normalize(mean=0.5, std=0.25)
    ])
    
    video = np.load(path) / 255
       
    # Model expects input to have 3 channels C.
    video = repeat(video, "Frames Height Width -> Frames C Height Width", C=3)
    
    video = torch.from_numpy(video)
    video = transform(video)
    
    return video.to(torch.bfloat16)

@torch.no_grad()
def generate_video_embeddings(video, model) -> torch.Tensor:
    model.eval()
    video = video.cuda()
    
    # Be sure to normalize the CLIP embedding after calculating it to make
    # cosine similarity between embeddings easier to calculate.
    embedding = F.normalize(model.encode_image(video), dim=-1)
    return embedding.unsqueeze(0)

@torch.no_grad()
def generate_ef_prompt_embedding(model, tokenizer) -> tuple[torch.Tensor, list[int]]:
    """Ejection Fraction (EF) prompt embedding for EchoClip."""
    model.eval()
    
    # prompt taken from: https://github.com/echonet/echo_CLIP/blob/main/utils.py
    ejection_fraction_prompts = [
            "THE LEFT VENTRICULAR EJECTION FRACTION IS ESTIMATED TO BE <#>% ",
            "LV EJECTION FRACTION IS <#>%. ",
        ]
    
    prompts = []
    prompt_values = []
    
    for prompt in ejection_fraction_prompts:
        for i in range(101):
            prompts.append(prompt.replace("<#>", str(i)))
            prompt_values.append(i)
    
    tokens = tokenizer(prompts).cuda()
    return F.normalize(model.encode_text(tokens), dim=-1), prompt_values

def compute_regression_metric(
    video_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_values: torch.Tensor,
):
    """See: https://github.com/echonet/echo_CLIP/blob/main/utils.py#L82"""
    per_frame_similarities = (
        video_embeddings @ prompt_embeddings.T
    )  # (N x Frames x Candidates)

    # Sort the candidates by their similarity to the video
    ranked_candidate_phrase_indices = torch.argsort(
        per_frame_similarities, dim=-1, descending=True
    )

    # Convert matrix of indices to their corresponding continuous values.
    prompt_values = torch.tensor(
        prompt_values, device=video_embeddings.device
    )  # (N x Frames x Candidates)
    all_frames_ranked_values = prompt_values[ranked_candidate_phrase_indices]

    # Taking the mean along dim=1 collapses the frames dimension
    avg_frame_ranked_values = all_frames_ranked_values.float().mean(
        dim=1
    )  # (N x Candidates)

    # The median of only the top 20% of predicted values is taken
    # as the final predicted value
    twenty_percent = int(avg_frame_ranked_values.shape[1] * 0.2)
    final_prediction = avg_frame_ranked_values[:, :twenty_percent].median(dim=-1)[0]

    return final_prediction


@torch.no_grad()
def predict_ejection_fraction(video, model, tokenizer=None, prompt_embeddings=None, prompt_values=None) -> float:

    video_embeddings = generate_video_embeddings(video, model)
    
    if tokenizer:
        prompt_embeddings, prompt_values = generate_ef_prompt_embedding(model, tokenizer)
    else:
        assert prompt_embeddings is not None
        assert prompt_values is not None

    return compute_regression_metric(video_embeddings, prompt_embeddings, prompt_values).item()


# See https://github.com/echonet/echo_CLIP/blob/main/zero_shot_example.py

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    'hf-hub:mkaichristensen/echo-clip',
    precision="bf16",
    device="cuda"
)
tokenizer = open_clip.get_tokenizer('hf-hub:mkaichristensen/echo-clip')

model = model.to("cuda")
model.eval();


train_df = pd.read_csv("/kaggle/input/echo2022/train_data.csv")
train_df["prediction"] = -1

ch4_path = "/kaggle/input/echo2022/train_data/train_data/4CH/{}_4CH_sequence.npy"


preds_4ch = []
prompt_embeddings, prompt_values = generate_ef_prompt_embedding(model, tokenizer)


for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    input_path = ch4_path.format(row.Patient_number)

    input_video = load_and_process_video(input_path)
    
    prediction = predict_ejection_fraction(input_video, model,
                                           prompt_embeddings=prompt_embeddings,
                                           prompt_values=prompt_values)

    train_df.loc[idx, "prediction"] = prediction


from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


mae = mean_absolute_error(train_df.LV_ef, train_df.prediction)
r2 = r2_score(train_df.LV_ef, train_df.prediction)
pearson = pearsonr(train_df.LV_ef, train_df.prediction)

plt.plot([10, 90], [10, 90], color="gray", ls="--", zorder=0)
plt.scatter(train_df.LV_ef, train_df.prediction, alpha=0.8, s=70, linewidths=1, edgecolors="white", zorder=1)

plt.annotate(f"MAE={mae:.1f}", (65, 30), color="red")
plt.annotate(f"$R^2$={r2:.3f}", (65, 25), color="red")
plt.annotate("Pearson:", (65, 20), color="red")
plt.annotate(f"    -> corr coef={pearson.statistic:.3f}", (65, 15), color="red")
plt.annotate(f"    -> p-value={pearson.pvalue:.3f}", (65, 10), color="red")

plt.xlabel("Ground Truth")
plt.ylabel("Prediction")
plt.title("EchoCLIP zero-shot EF prediction")
plt.grid()




