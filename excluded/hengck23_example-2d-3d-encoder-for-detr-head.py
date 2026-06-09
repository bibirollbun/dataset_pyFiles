#model.py
import timm
print('timm.__version__',timm.__version__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def encode_for_pvtv2(e, x, B, depth_scaling=[2,2,2,2,]):
	#poor man's attention = avg + max pool
    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0) \
                + F.gelu(F.max_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0))
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode=[]  #x = 1,256,512,512
    x = e.patch_embed(x) # x seq: 1024, 128, 128, 64

    x = e.stages_0(x)   #4, 64, 128, 128
    x, x1 = pool_in_depth(x, depth_scaling[0])
    #encode.append(x1)
    x = e.stages_1(x)   #4, 128, 64, 64
    x, x1 = pool_in_depth(x, depth_scaling[1])
    #encode.append(x1)
    x = e.stages_2(x)   #4, 320, 32, 32
    x, x1 = pool_in_depth(x, depth_scaling[2])
    #encode.append(x1)
    x = e.stages_3(x)   #4, 512, 16, 16
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)

    return encode


class Net(nn.Module):
    def __init__(self, pretrained=False, cfg=None):
        super(Net, self).__init__()
        self.output_type = ['infer', 'loss', ]
        self.register_buffer('D', torch.tensor(0))

        self.arch = 'pvt_v2_b1'
        encoder_dim = {
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
        }.get(self.arch, [1024])

        self.encoder = timm.create_model(
            model_name=self.arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.mask = nn.Conv3d(encoder_dim[-1],1, kernel_size=1)

    def forward(self, batch):
        device = self.D.device

        image = batch['image'].to(device)
        B, D, H, W = image.shape
        image = image.reshape(B*D, 1,H, W)

        x = (image.half() - 128) / 128
        x = x.expand(-1, 3, -1, -1)

        encode = encode_for_pvtv2(self.encoder, x, B)
        last = encode[-1] #this is the feature map !!!!
        logit = self.mask(last) .squeeze(1)

        
        #print(f'last', last.shape)
        #[print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]
        #print('logit', logit.shape)

        output = {} 

        #loss for pretraining 2d-3d encoder
        if 'loss' in self.output_type:
            truth = batch['truth'].to(device)
            output['mask_loss'] = F.binary_cross_entropy_with_logits(logit,truth)

        if 'infer' in self.output_type:
            output['motor'] = torch.sigmoid(logit)
        return output


###---------------------------------------------
# run some dummy data
def run_check_net():

    B = 1
    slice_shape = (192,384,384) 
    mask_shape  = (12,12,12)

    batch = {
        'image': torch.from_numpy(np.random.uniform(0,1, (B, *slice_shape))).byte(),
        'truth': torch.from_numpy(np.random.choice(2, (B, *mask_shape))).half(),
    }
    net = Net(pretrained=False, cfg=None).cuda()

    with torch.no_grad():
        with torch.amp.autocast('cuda',enabled=True):
            output = net(batch)
    # ---
    print('batch')
    for k, v in batch.items():
        if k == 'D':
            print(f'{k:>32} : {v} ')
        else:
            print(f'{k:>32} : {v.shape} ')

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print(f'{k:>32} : {v.shape} ')
    print('loss')
    for k, v in output.items():
        if 'loss' in k:
            print(f'{k:>32} : {v.item()} ')

run_check_net()


import matplotlib
import matplotlib.pyplot as plt 
import cv2


checkpoint ='/kaggle/input/example-2d-3d-encoder-weight/00003808.pth'


net = Net(pretrained=False)
net.cuda()
net.output_type = ['infer']
f = torch.load(checkpoint, map_location=lambda storage, loc: storage, weights_only=False)
state_dict = f.get('state_dict', f) 
print(net.load_state_dict(state_dict, strict=False))

valid_data=[
    {
        'tomo_id':'tomo_00e047',
        'z0z1'  :[73,265],
        'label' :1,  
    },
    {
        'tomo_id':'tomo_17143f',
        'z0z1'  :[0,256],
        'label' :0,  
    },
    {
        'tomo_id':'tomo_0fe63f',
        'z0z1'  :[101,293],
        'label' :1,  
    },
]

#helper function
KAGGLE_DATA_DIR ='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
def read_image_stack(tomo_id, start_no, end_no, resize=-1, mode=cv2.IMREAD_GRAYSCALE):
    image = []
    for z in range(start_no,end_no):
        jpg_file = f'{KAGGLE_DATA_DIR}/train/{tomo_id}/slice_{z:04d}.jpg'
        m=cv2.imread(jpg_file, mode)
        if resize>0:
            m=cv2.resize(m, (resize, resize),cv2.INTER_LINEAR)
        image.append(m)
    image = np.stack(image)
    return image

def make_overlay(image, prob, axis=0): 
    m = image.mean(axis)
    p = prob.max(axis)
    #t = truth.max(axis)
    
    overlay = np.stack([m, m, m], 2)
    op = p[..., None] * [[[1, 0, 0]]]
    overlay = 255 - (255 - overlay) * (1 - op)
    #ot = t[...,None]*[[[0,1,0]]]
    #overlay = 255-(255-overlay)*(1-ot)
    overlay = overlay.astype(np.uint8)
    return overlay


for r in valid_data:
    tomo_id = r['tomo_id']
    label = r['label']
    z0,z1 = r['z0z1']
    image = read_image_stack(tomo_id, z0, z1, resize=384, mode=cv2.IMREAD_GRAYSCALE)
    batch = {
        'image': torch.from_numpy(image).unsqueeze(0).byte(),
    }		
    with torch.amp.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            output = net(batch)

    prob = F.interpolate(
	    output['motor'].unsqueeze(1),
	    scale_factor=(16, 32, 32), mode='nearest',  
	)[0,0].float().data.cpu().numpy()

    #---draw the results ---
    print('tomo_id', tomo_id)
    print('label', label)
    print('image', image.shape)
    print('prob', prob.shape)

    overlay0 = make_overlay(image, prob, axis=0)
    overlay1 = make_overlay(image, prob, axis=1)
    plt.imshow(image.mean(0), cmap='gray')
    plt.show()
    plt.imshow(overlay0)
    plt.show()
    plt.imshow(overlay1)
    plt.show()

