!pip install numpy pandas Pillow torch git+https://github.com/openai/CLIP.git pydicom tqdm scikit-learn imageio opencv-python scipy



from scipy.spatial import distance
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
import pydicom
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
import os
import imageio
import cv2


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
print("Path Join Completed")



df_meta= pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/meta.csv')
df_meta.head()


df_dicom = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv')
df_dicom.head()


df_dicom.SeriesDescription.unique()


cropped_images = df_dicom[df_dicom.SeriesDescription=='cropped images'].image_path
cropped_images.head(5)


full_mammo = df_dicom[df_dicom.SeriesDescription=='full mammogram images'].image_path
full_mammo.head()



roi_img = df_dicom[df_dicom.SeriesDescription=='ROI mask images'].image_path
roi_img.head()


# Check if row 8027 exists in roi_img
if 8027 in roi_img.index:
    print("Row 8027 exists in roi_img.")
else:
    print("Row 8027 does not exist in roi_img.")



imdir = '../input/cbis-ddsm-breast-cancer-image-dataset/jpeg'


cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', imdir, regex=True)
full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', imdir, regex=True)
roi_img = roi_img.replace('CBIS-DDSM/jpeg', imdir, regex=True)

# view new paths
print('Cropped Images paths:\n')
print(cropped_images.iloc[0])
print('Full mammo Images paths:\n')
print(full_mammo.iloc[0])
print('ROI Mask Images paths:\n')
print(roi_img.iloc[0])


full_mammo_dict = dict()
cropped_images_dict = dict()
roi_img_dict = dict()

for dicom in full_mammo:
    key = dicom.split("/")[4]
    full_mammo_dict[key] = dicom
for dicom in cropped_images:
    key = dicom.split("/")[4]
    cropped_images_dict[key] = dicom
for dicom in roi_img:
    key = dicom.split("/")[4]
    roi_img_dict[key] = dicom

# view keys
next(iter((full_mammo_dict.items())))


mass_train = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv')
mass_test = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv')

mass_train.head()


def fix_image_path(data):
    """correct dicom paths to correct image paths"""
    for index, img in enumerate(data.values):
        img_name = img[11].split("/")[2]
        data.iloc[index,11] = full_mammo_dict[img_name]
        img_name = img[12].split("/")[2]
        data.iloc[index,12] = cropped_images_dict[img_name]
        img_name = img[13].split("/")[2]
        data.iloc[index,13] = roi_img_dict[img_name]
        
# apply to datasets
fix_image_path(mass_train)
fix_image_path(mass_test)



mass_train = mass_train.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

mass_train.head(5)


mass_train['mass_shape'] = mass_train['mass_shape'].bfill()
mass_train['mass_margins'] = mass_train['mass_margins'].bfill()

#check null values
mass_train.isnull().sum()


print(f'Shape of mass_train: {mass_train.shape}')
print(f'Shape of mass_test: {mass_test.shape}')


mass_test = mass_test.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})


mass_test['mass_margins'] = mass_test['mass_margins'].bfill()

#check null values
mass_test.isnull().sum()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg



print(mass_train['cropped_image_file_path'].head())


print(mass_train['ROI_mask_file_path'].head())


def display_images(column, number):
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    
    # Loop through rows and display images
    for index, row in mass_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        image = mpimg.imread(image_path)
        ax = axes[index]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print('Full Mammograms:\n')
display_images('image_file_path', 5)
print('Cropped Mammograms:\n')
display_images('cropped_image_file_path', 5)
print('ROI mask:\n')
display_images('ROI_mask_file_path', 5)


df_1122= pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
print(df_1122.head(5))


df_2211 = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
print(df_2211.head(5))




def process_dcm(file_path):
    ds = pydicom.dcmread(file_path)
    im = ds.pixel_array
    im = im.astype(float)
    # simple normalization, convert to RGB
    im = im / im.max()
    im2 = np.zeros(list(im.shape) + [3])
    for i in range(3):
        im2[:, :, i] = im
    im = (255 * im2).astype(np.uint8)

    return im


def create_clip_feature_mat(file_list, clip_model, preprocess_fxn):
    X = np.zeros((len(file_list), 512)) # 512 is feature dimension
    for i, f in tqdm(enumerate(file_list), total=len(file_list)):
        if '.dcm' in f:
            im = Image.fromarray(process_dcm(f))
        else:
            im = Image.open(f)
            if im.mode != 'RGB':             
                im = im.convert('RGB')    
        im = preprocess_fxn(im).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(im)
        X[i] = image_features[0].cpu()

    return X


from scipy.spatial import distance
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix
)
from statsmodels.stats.proportion import proportion_confint
import pandas as pd
import numpy as np
import os

def fit_words(train_df, test_df, device, word_list, save_dir, save_tag):
    # â€”â€”â€” 1) åŠ è½½ CLIP â€”â€”â€”
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)

    # â€”â€”â€” 2) æ��å�– CLIP image ç‰¹å¾� â€”â€”â€”
    X_train = create_clip_feature_mat(train_df.file_path.values, clip_model, preprocess_fxn)

    # â€”â€”â€” 3) CLIP feature ä¸Šçš„ LogisticRegression â€”â€”â€”
    classifier = LogisticRegression(
        random_state=0, C=1, max_iter=1000, verbose=1, fit_intercept=False
    )
    classifier.fit(X_train, train_df.label.values)

    # â€”â€”â€” 4) æ��å�– word embeddings & æ‹Ÿå�ˆ wordâ†’coef â€”â€”â€”
    tokened_words  = clip.tokenize(word_list).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)  # [n_words, 512]

    weights_model = LinearRegression(fit_intercept=False)
    weights_model.fit(
        word_features.cpu().T,            # [512, n_words]
        classifier.coef_[0]               # [512, ]
    )

    # â€”â€”â€” 5) ä¿�å­˜å¹¶æ‰“å�° word weights 
    word_df = pd.DataFrame({
        'word':    word_list,
        'weight':  weights_model.coef_
    }).set_index('word').sort_values('weight')

    # è®¡ç®—æ¯�ä¸ªè¯�çš„ç»�å¯¹æ�ƒé‡�å� æ¯”
    abs_sum = word_df['weight'].abs().sum()
    word_df['prop'] = word_df['weight'].abs() / abs_sum


    word_df.to_csv(os.path.join(save_dir, f'word_weights-{save_tag}.csv'))
    print(f"\n=== [{save_tag}] Descriptor Weights & Proportions ===")
    print(word_df.round(3))

    # â€”â€”â€” 6) æµ‹è¯•é›†ä¸Š CLIPâ†’LR çš„æ€§èƒ½ â€”â€”â€”
    X_test   = create_clip_feature_mat(test_df.file_path.values, clip_model, preprocess_fxn)
    y_true   = test_df.label.values
    y_score  = classifier.predict_proba(X_test)[:, 1]
    y_pred   = classifier.predict(X_test)

    # 6a) åŸºæœ¬æŒ‡æ ‡
    acc   = accuracy_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_score)
    f1    = f1_score(y_true, y_pred)
    sens  = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec  = tn/(tn+fp)

    # 6b) ç½®ä¿¡åŒºé—´
    n = len(y_true)
    ci_acc_low, ci_acc_hi   = proportion_confint(acc*n, n, method='wilson')
    ci_sens_low, ci_sens_hi = proportion_confint(tp, tp+fn, method='wilson')
    ci_spec_low, ci_spec_hi = proportion_confint(tn, tn+fp, method='wilson')

    print(f"\n=== [{save_tag}] CLIPâ†’LR  Performance ===")
    print(f"Accuracy:    {acc:.3f} (95% CI [{ci_acc_low:.3f},{ci_acc_hi:.3f}])")
    print(f"AUC:         {auc:.3f}")
    print(f"F1:          {f1:.3f}")
    print(f"Sensitivity: {sens:.3f} (95% CI [{ci_sens_low:.3f},{ci_sens_hi:.3f}])")
    print(f"Specificity: {spec:.3f} (95% CI [{ci_spec_low:.3f},{ci_spec_hi:.3f}])")

    # â€”â€”â€” 7) Descriptorâ†’LR Performance â€”â€”â€”
    W = word_features.cpu().numpy()                # [n_words,512]
    X_desc_train = X_train.dot(W.T)                # [n_samples, n_words]
    X_desc_test  = X_test.dot(W.T)

    desc_clf = LogisticRegression(random_state=0, C=1, max_iter=1000, fit_intercept=False)
    desc_clf.fit(X_desc_train, train_df.label.values)

    y_score_d = desc_clf.predict_proba(X_desc_test)[:,1]
    y_pred_d  = desc_clf.predict(X_desc_test)

    acc_d   = accuracy_score(y_true, y_pred_d)
    auc_d   = roc_auc_score(y_true, y_score_d)
    f1_d    = f1_score(y_true, y_pred_d)
    sens_d  = recall_score(y_true, y_pred_d)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_d).ravel()
    spec_d  = tn/(tn+fp)

    print(f"\n=== [{save_tag}] Descriptorâ†’LR Performance ===")
    print(f"Accuracy:    {acc_d:.3f}")
    print(f"AUC:         {auc_d:.3f}")
    print(f"F1:          {f1_d:.3f}")
    print(f"Sensitivity: {sens_d:.3f}")
    print(f"Specificity: {spec_d:.3f}")

    # â€”â€”â€” 8) ä½™å¼¦ç›¸ä¼¼åº¦ 
    pred_coef = weights_model.predict(W.T)
    cos_sim   = 1 - distance.cosine(pred_coef, classifier.coef_[0])
    print(f"\nCosine similarity (wordâ†’coef vs CLIPâ†’coef): {cos_sim:.3f}\n")


def get_prototypes(df, words, device, save_dir, n_save=20):
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    X = create_clip_feature_mat(df.file_path.values, clip_model, preprocess_fxn)

    tokened_words = clip.tokenize(words).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    file_dot = np.zeros((len(df), len(words)))
    for i in range(len(df)):
        for j in range(len(words)):
            file_dot[i, j] = np.dot(X[i], word_features[j].cpu())

    file_dot_pred = np.zeros((len(df), len(words)))
    for j in range(len(words)):
        fit_j = [k for k in range(len(words)) if k != j]
        dot_regression = LinearRegression()
        dot_regression.fit(file_dot[:, fit_j], file_dot[:, j])
        file_dot_pred[:, j] = dot_regression.predict(file_dot[:, fit_j])

    dot_df_diff = pd.DataFrame(file_dot - file_dot_pred, columns=words)
    dot_df_diff['label'] = df['label'].values
    dot_df_diff.set_index(df.file_path, inplace=True)

    for w in words:
        print(w)
        for sort_dir in ['top']:
            this_df = dot_df_diff.sort_values(w, ascending=(sort_dir == 'bottom'))
            save_files = this_df.index.values[:n_save]
            these_labels = this_df.label.values[:n_save]
            this_out_dir = save_dir + w + '_' + sort_dir + '/'
            if not os.path.exists(this_out_dir):
                os.mkdir(this_out_dir)

            for i, f in enumerate(save_files):
                if '.dcm' in f:
                    im = process_dcm(f)
                else:
                    im = imageio.imread(f)
                    if im.ndim == 2:                        
                        im = np.stack([im]*3, axis=-1)      
                    elif im.shape[-1] == 4:                   
                        im = im[..., :3]
                # make square and downsample for efficiency 
                min_dim = min(im.shape[:2])
                for dim in [0, 1]:
                    if im.shape[dim] > min_dim:
                        n_start = int((im.shape[dim] - min_dim) / 2)
                        n_stop = n_start + min_dim
                        if dim == 0:
                            im = im[n_start:n_stop, :, :]
                        else:
                            im = im[:, n_start:n_stop, :]
                if min_dim > 500:
                    im = cv2.resize(im, (500, 500))
                f_name = f'rank{i}_label{these_labels[i]}.png'
                imageio.imwrite(os.path.join(this_out_dir, f_name), im)


if __name__ == '__main__':
    dataset_name = 'cbis'
    device = 'cuda:0'

    if dataset_name == 'cbis':
        train_df = mass_train[['cropped_image_file_path', 'pathology']].copy()
        train_df.rename(columns={'cropped_image_file_path': 'file_path'}, inplace=True)
        train_df['label'] = (train_df['pathology'] == 'MALIGNANT').astype(int)
        train_df.drop(columns=['pathology'], inplace=True)

        test_df = mass_test[['cropped_image_file_path', 'pathology']].copy()
        test_df.rename(columns={'cropped_image_file_path': 'file_path'}, inplace=True)
        test_df['label'] = (test_df['pathology'] == 'MALIGNANT').astype(int)
        test_df.drop(columns=['pathology'], inplace=True)

    elif dataset_name == 'melanoma':
        train_df = pd.read_csv('./data/siim_melanoma_train.csv')
        test_df  = pd.read_csv('./data/siim_melanoma_test.csv')

    words = [
        'dark', 'light', 'round', 'pointed', 'large', 'small',
        'smooth', 'coarse', 'transparent', 'opaque',
        'symmetric', 'asymmetric', 'high contrast', 'low contrast'
    ]

    base_out_dir = './results/'
    os.makedirs(base_out_dir, exist_ok=True)

    save_tag = dataset_name
    save_dir = os.path.join(base_out_dir, save_tag)
    os.makedirs(save_dir, exist_ok=True)

    fit_words(train_df, test_df, device, words,
              save_dir=save_dir, save_tag=save_tag)

    prot_save_dir = os.path.join(save_dir, save_tag + '_prototypes')
    os.makedirs(prot_save_dir, exist_ok=True)
    get_prototypes(train_df, words, device, prot_save_dir, n_save=5)


# CLIP+MLP å¾®è°ƒ
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from PIL import Image

# â€”â€” 1) å‡†å¤‡æ•°æ�®é›† â€”â€” 
class PatchDataset(Dataset):
    def __init__(self, df, preprocess):
        self.paths = df.file_path.values
        self.labels = df.label.values.astype('float32')
        self.preprocess = preprocess
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        x = self.preprocess(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# â€”â€” 2) åŠ è½½ CLIP backbone & preproces
clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
clip_model = clip_model.float()  

# Dataset & DataLoader
train_ds = PatchDataset(train_df, preprocess_fxn)
test_ds  = PatchDataset(test_df,  preprocess_fxn)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

# â€”â€” 3) æ�­å»ºæ¨¡å�‹ â€”â€” 
class FineTuneMLP(nn.Module):
    def __init__(self, backbone, hidden_dim=1024, dropout_p=0.3):
        super().__init__()
        self.vision = backbone.visual.float()   # ç¡®ä¿� vision ä¹Ÿæ˜¯å�•ç²¾åº¦
        for p in self.vision.parameters():
            p.requires_grad = False

        d = self.vision.output_dim
        self.head = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim//2, 1)
        ).float() 

    def forward(self, x):
        f = self.vision(x)         
        return self.head(f).squeeze(1)  

model_ft = FineTuneMLP(clip_model).to(device)

# â€”â€” 4) æ�Ÿå¤±ã€�ä¼˜åŒ–å™¨ã€�è°ƒåº¦ â€”â€” 
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW([
    {'params': model_ft.vision.parameters(), 'lr': 5e-6},
    {'params': model_ft.head.parameters(),   'lr': 2e-4}
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# â€”â€” 5) è®­ç»ƒ â€”â€” 
num_epochs = 20
for epoch in range(1, num_epochs+1):
    model_ft.train()
    running_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model_ft(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

# â€”â€” 6) åœ¨è®­ç»ƒ & æµ‹è¯•é›†ä¸Šè¯„ä¼° â€”â€” 
def eval_split(loader):
    model_ft.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logit = model_ft(xb).cpu().numpy()
            prob  = 1 / (1 + np.exp(-logit))
            ys.append(yb.numpy()); ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob

y_tr, p_tr, prob_tr = eval_split(train_loader)
y_te, p_te, prob_te = eval_split(test_loader)

acc_tr = accuracy_score(y_tr, p_tr)
auc_tr = roc_auc_score(y_tr, prob_tr)
acc_te = accuracy_score(y_te, p_te)
auc_te = roc_auc_score(y_te, prob_te)

print("\n=== [CBIS] CLIPâ†’MLP Improved Performance ===")
print(f"[Train set] Accuracy: {acc_tr:.3f}, AUC: {auc_tr:.3f}")
print(f"[Test  set] Accuracy: {acc_te:.3f}, AUC: {auc_te:.3f}")



# === Shortcut Analysis Inline ===

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance
from PIL import Image

# å�Ÿæ�¥çš„14ä¸ªæ��è¿°è¯�
orig_words = [
    'dark', 'light', 'round', 'pointed',
    'large', 'small', 'smooth', 'coarse',
    'transparent', 'opaque', 'symmetric', 'asymmetric',
    'high contrast', 'low contrast'
]
# \è¦�æ£€æµ‹çš„ shortcut è¯�
shortcut_words = ['left', 'right', 'CC', 'MLO']
all_words = orig_words + shortcut_words

# 1) è®­ç»ƒ CLIPâ†’LR
clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
X_train = create_clip_feature_mat(train_df.file_path.values, clip_model, preprocess_fxn)
clf = LogisticRegression(random_state=0, C=1, max_iter=1000, fit_intercept=False)
clf.fit(X_train, train_df.label.values)

# 2) è®¡ç®—æ‰€æœ‰è¯�çš„ embedding å¹¶æ‹Ÿå�ˆçº¿æ€§å›�å½’ä»¥å¾—åˆ°æ�ƒé‡�
tokened = clip.tokenize(all_words).to(device)
with torch.no_grad():
    word_feats = clip_model.encode_text(tokened).cpu().numpy()   # shape [len(all_words),512]

lm = LinearRegression(fit_intercept=False)
lm.fit(word_feats.T, clf.coef_[0])
weights = dict(zip(all_words, lm.coef_))

# æ‰“å�°æ‰€æœ‰è¯�çš„æ�ƒé‡�å�Šå…¶å� æ¯”
abs_w = np.array([abs(weights[w]) for w in all_words])
prop = abs_w / abs_w.sum()
# æ�„é€  DataFrame æ�’åº�æ˜¾ç¤º
import pandas as pd
df_w = pd.DataFrame({
    'word': all_words,
    'weight': [weights[w] for w in all_words],
    'prop'  : prop
})
df_w = df_w.sort_values('prop', ascending=False).reset_index(drop=True)
print("\n=== All Word Weights and Proportions ===")
print(df_w.to_string(index=False, float_format="%.4f"))

# 3) æ‰“å�° shortcut è¯�çš„æ�ƒé‡�
print("ğŸš¦ Shortcut Word Weights ğŸš¦")
for w in shortcut_words:
    print(f"{w:>6s} : {weights[w]: .4f}")
print()

# 4) è®¡ç®— residuals = dot - dot_predï¼Œç”¨æ�¥æ‰¾ prototypical examples
#    file_dot[i,j] = X_train[i]Â·word_feats[j]
file_dot = X_train.dot(word_feats.T)  # shape [N_train, len(all_words)]
file_dot_pred = np.zeros_like(file_dot)
for j in range(len(all_words)):
    others = [k for k in range(len(all_words)) if k != j]
    reg = LinearRegression(fit_intercept=False).fit(file_dot[:, others], file_dot[:, j])
    file_dot_pred[:, j] = reg.predict(file_dot[:, others])
residuals = file_dot - file_dot_pred   # shape [N_train, len(all_words)]

# 5) å¯¹æ¯�ä¸ª shortcut è¯�ï¼Œå±•ç¤º top-3 çš„è®­ç»ƒå›¾åƒ�
TOP_K = 3
for w in shortcut_words:
    j = all_words.index(w)
    top_idxs = np.argsort(residuals[:, j])[::-1][:TOP_K]
    print(f"ğŸ”� Prototypes for â€œ{w}â€� (label shown under each):")
    fig, axes = plt.subplots(1, TOP_K, figsize=(TOP_K*3, 3))
    for ax, idx in zip(axes, top_idxs):
        path = train_df.file_path.iloc[idx]
        img = Image.open(path).convert('RGB')
        ax.imshow(img)
        ax.set_title(train_df.label.iloc[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show()



#  Vocabulary Expansion + Full Pipeline 

# 1. å®šä¹‰æ‰©å……å��çš„è¯�è¡¨
extended_words = [
    # å�Ÿå§‹é€šç”¨è§†è§‰è¯�
    'dark', 'light', 'round', 'pointed', 'large', 'small',
    'smooth', 'coarse', 'transparent', 'opaque',
    'symmetric', 'asymmetric', 'high contrast', 'low contrast',
    # æ–°å¢�åŒ»å­¦ç›¸å…³è¯�
    'spiculated', 'lobulated', 'microlobulated', 'heterogeneous texture',
    'calcified', 'architectural distortion', 'microcalcifications'
]

# 2. é‡�æ–°è¿�è¡Œ fit_wordsï¼Œå¾—åˆ°æ–° word_weights
fit_words(
    train_df, test_df, device,
    word_list=extended_words,
    save_dir=save_dir,
    save_tag='cbis_extvocab'
)


# 3. é‡�æ–°åŠ è½½æ–°æ�ƒé‡�ï¼Œæ‰“å�°å®Œæ•´åˆ—è¡¨
import pandas as pd

wd = pd.read_csv(
    os.path.join(save_dir, 'word_weights-cbis_extvocab.csv'),
    index_col=0
)

print("=== Descriptor Weights (extended vocab) ===")
# wd é‡Œç¬¬ä¸€åˆ—å�³æ˜¯æ�ƒé‡�ï¼Œç›´æ�¥ to_string æ‰“å�°å…¨éƒ¨è¯�æ±‡
print(wd.to_string(float_format="%.4f"))

# 4. åœ¨æ‰©å……è¯�è¡¨ä¸Šè®­ç»ƒ Descriptorâ†’LR & Descriptorâ†’MLP

# 4.1 æ�„é€  descriptor ç‰¹å¾� (é‡�æ–°è®¡ç®—æˆ–å¤�ç”¨)
clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
X_train = create_clip_feature_mat(train_df.file_path.values, clip_model, preprocess_fxn)
X_test  = create_clip_feature_mat(test_df.file_path.values,  clip_model, preprocess_fxn)

# æ–‡æœ¬ç‰¹å¾�
tokened = clip.tokenize(extended_words).to(device)
with torch.no_grad():
    word_feats = clip_model.encode_text(tokened).cpu().numpy()  # [W,512]

# ç‚¹ä¹˜å¾—åˆ° descriptor ç‰¹å¾�
Xw_train = X_train.dot(word_feats.T)  # [N_train, W]
Xw_test  = X_test.dot(word_feats.T)   # [N_test,  W]

# 4.2 Descriptorâ†’LR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

clf2 = LogisticRegression(
    max_iter=1000,
    random_state=0,
    fit_intercept=False
)
clf2.fit(Xw_train, train_df.label.values)

y2_tr = clf2.predict(Xw_train)
p2_tr = clf2.predict_proba(Xw_train)[:,1]
y2_te = clf2.predict(Xw_test)
p2_te = clf2.predict_proba(Xw_test)[:,1]

print("\n=== Descriptorâ†’LR with extended vocab ===")
print(f"Train Acc {accuracy_score(train_df.label, y2_tr):.3f}, "
      f"AUC {roc_auc_score(train_df.label, p2_tr):.3f}")
print(f" Test Acc {accuracy_score(test_df.label, y2_te):.3f}, "
      f"AUC {roc_auc_score(test_df.label, p2_te):.3f}")

# 4.3 Descriptorâ†’MLP
from sklearn.neural_network import MLPClassifier

mlp2 = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
mlp2.fit(Xw_train, train_df.label.values)

y3_tr = mlp2.predict(Xw_train)
p3_tr = mlp2.predict_proba(Xw_train)[:,1]
y3_te = mlp2.predict(Xw_test)
p3_te = mlp2.predict_proba(Xw_test)[:,1]

print("\n=== Descriptorâ†’MLP with extended vocab ===")
print(f"Train Acc {accuracy_score(train_df.label, y3_tr):.3f}, "
      f"AUC {roc_auc_score(train_df.label, p3_tr):.3f}")
print(f" Test Acc {accuracy_score(test_df.label, y3_te):.3f}, "
      f"AUC {roc_auc_score(test_df.label, p3_te):.3f}")



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import os

# def make_word_weights_plot(word_df, base_save_path):
#     x = word_df.index.values
#     y = word_df.weights.values
#     x = np.flipud(x)
#     y = np.flipud(y)

#     cmap = plt.get_cmap('RdBu_r')
#     eps = 1e-8
#     normalized_data = (y - np.min(y)) / (np.max(y) - np.min(y) + eps)
#     colors = cmap(normalized_data)

#     sns.set_theme()
#     sns.set_style("ticks")

#     fig, ax = plt.subplots(figsize=(10, 6))
#     bars = ax.bar(range(len(x)), y, color=colors, edgecolor="black")

#     ax.set_xticks(range(len(x)))
#     ax.set_xticklabels(x, rotation=60, ha="right", fontweight="bold")

#     yticks = ax.get_yticks()
#     ax.set_yticks(yticks)
#     ax.set_yticklabels(np.round(yticks, 2), fontweight="bold")

#     ax.set_ylabel("Malignancy Weight", fontweight="bold", size=14)

#     sns.despine(top=True, right=True)
#     plt.tight_layout()

#     for ext in ['png', 'pdf']:
#         plt.savefig(f'{base_save_path}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.05)
#     plt.close()

# if __name__ == '__main__':
#     save_tag = 'cbis'
#     save_dir = f'./results/{save_tag}/'
#     word_df = pd.read_csv(save_dir + f'word_weights-{save_tag}.csv', index_col=0)
#     make_word_weights_plot(word_df, os.path.join(save_dir, f'{save_tag}_weights_plot'))



# import os

# save_tag = 'cbis'
# save_dir = f'./results/{save_tag}/'
# print("PNG exists:", os.path.exists(os.path.join(save_dir, f'{save_tag}_weights_plot.png')))
# print("PDF exists:", os.path.exists(os.path.join(save_dir, f'{save_tag}_weights_plot.pdf')))


