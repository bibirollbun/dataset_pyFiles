import torch
from torch import nn
import timm


def reconstruct_tensor(output_tensor, cls_token_first=True):
    bs = output_tensor.shape[0]
    # 1. Remove CLS token (first token)
    if cls_token_first:
        patches = output_tensor[:, 1:, :]  # Shape [1, 625, 196]
    else:
        patches = output_tensor
        
    # 2. Reshape to [1, 25, 25, 14, 14] (25x25 grid of 14x14 patches)
    patches = patches.view(bs, CFG.image_size[0]//14, CFG.image_size[1]//14, 14, 14)

    # 3. Re-arrange patches into spatial order
    #    - Combine grid rows first, then patch rows
    reconstructed = patches.permute(0, 1, 3, 2, 4).contiguous()
    reconstructed = reconstructed.view(bs, 1, CFG.image_size[0], CFG.image_size[1])  # Final shape

    return reconstructed

def attention_rope_forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            npt = 1 #self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], timm.layers.apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], timm.layers.apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)
        
        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def block_rope_forward(self, x: torch.Tensor, rope) -> torch.Tensor:
    #x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope)))
    x = x + self.drop_path1(self.ls1(attention_rope_forward(self.attn, self.norm1(x), rope)))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x

class Model(nn.Module):
    def __init__(self, pretrained=True, drop=0.):
        super(Model
              , self).__init__()
        
        patch_size = 14
        
        self.encoder = timm.create_model(CFG.model_name, pretrained=pretrained, in_chans=1, img_size=CFG.image_size, num_classes=0, global_pool='')
        self.decoder = timm.create_model(CFG.model_name, pretrained=pretrained, in_chans=1, img_size=CFG.image_size, num_classes=0, global_pool='')
        
        self.feats = self.encoder.num_features
        
        self.head = nn.Linear(self.feats, patch_size*patch_size)
        #self.head = nn.Linear(self.feats, 16)
        self.sigmoid = nn.Sigmoid()
        
        #st = torch.load('./data/pretrained/eva02_small_patch14_224.mim_in22k_pytorch_model.bin', map_location='cpu')
        #del st['pos_embed'], st['patch_embed.proj.weight']
        #self.encoder.load_state_dict(st, strict=False)
        
        self.rope = timm.layers.RotaryEmbeddingCat(
                self.feats // 6,
                in_pixels=False,
                feat_shape=None,# if dynamic_img_size else self.patch_embed.grid_size,
                ref_feat_shape=None,#ref_feat_shape,
            )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.feats))
        self.reg_token = None#nn.Parameter(torch.zeros(1, 1, self.feats))
        
        #for block in self.encoder.blocks:
            #block.attn = Attention(384, num_heads=6, qkv_bias=False, qk_norm=False, proj_bias=True, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm)
        
    def _pos_embed(self, x):
        dynamic_img_size = True
        self.pos_embed = None
        
        if dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                prev_grid_size = self.patch_embed.grid_size
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=(H, W),
                    old_size=prev_grid_size,
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        #print(x.shape, self.cls_token.shape)
            
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.reg_token is not None:
            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            x = torch.cat(to_cat + [x], dim=1)
        
        #print(rot_pos_embed.shape)
        
        return x, rot_pos_embed
        
    def forward_features(self, model, x: torch.Tensor) -> torch.Tensor:
        x = model.patch_embed(x)
        #x = self.encoder._pos_embed(x)
        x = x.reshape(x.shape[0], CFG.image_size[0]//14, CFG.image_size[1]//14, x.shape[2])
        x, rope = self._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        
        for blk in model.blocks:
            #x = blk(x)
            x = block_rope_forward(blk, x, rope)
        
        x = model.norm(x)
        return x
        
    def forward(self, inp):
        inp = torch.nan_to_num(inp, 0, 0, 0)
        
        #features = self.encoder(inp)
        #print(features.shape)
        features = self.forward_features(self.encoder, inp)
        features = self.head(features)
        
        interm_masks = reconstruct_tensor(features)
        
        features = self.forward_features(self.decoder, interm_masks)
        features = self.head(features)
        
        masks = reconstruct_tensor(features)
        
        masks = nn.functional.interpolate(masks, (70, 70), mode='bilinear')
        
        masks = self.sigmoid(masks)
        
        masks = (masks*3000.)+1500.
        
        return None, masks


class CFG:
    model_name = 'vit_small_patch14_dinov2.lvd142m'
    image_size = (350, 350)


model = Model()
model


with torch.no_grad():
    inp = torch.zeros((2, 1, 350, 350))
    _, output = model(inp)
output.shape










