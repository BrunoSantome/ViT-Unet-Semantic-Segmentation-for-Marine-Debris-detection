
import torch
import torch.nn as nn
import timm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.assets import S2_RGB_INDICES, MODEL_DEFAULT
import math
import torch.nn.functional as functional

def build_vit_encoder(in_chans=11, img_size=256, pretrained=True, rgb_indices=S2_RGB_INDICES, model=MODEL_DEFAULT):
    model = timm.create_model(model, pretrained=pretrained, img_size=img_size)

    old_proj = model.patch_embed.proj #Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    embed_dim = old_proj.out_channels     # 768
    patch_size = old_proj.kernel_size     # (16, 16)
    old_weight = old_proj.weight.data     # [768, 3, 16, 16]
    old_bias = old_proj.bias.data if old_proj.bias is not None else None
    #print(old_bias)
    new_proj = nn.Conv2d(
        in_channels=in_chans,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size,
        bias=old_bias is not None,
    )

    with torch.no_grad():
        # Since the positions are inversly proportional 1 becomes 3, 2 stays 2 and 3 becomes 1
        for rgb_channel, s2_index in enumerate(rgb_indices):
            new_proj.weight[:, s2_index, :, :] = old_weight[:, rgb_channel, :, :]

        #only if it has bias, 
        if old_bias is not None:
            new_proj.bias.copy_(old_bias)

    model.patch_embed.proj = new_proj
    # model.patch_embed.img_size = (img_size, img_size)     # (256, 256)
    # model.patch_embed.grid_size = (16, 16)  # (16, 16)
    # model.patch_embed.num_patches = 16 * 16  # 256
    # (model, img_size)
    return model


 
# After coding this function, it appears timm can do this interpolation automatically with the argument img_size...
def adapt_positional_embeddings(model, img_size):
    pos_embed = model.pos_embed
    print(pos_embed.shape) #torch.Size([1, 197, 768]), change 197. 
    embed_dim = pos_embed.shape[-1] #768
    n_tokens= pos_embed.shape[1]
    
    patch_vit_old_size= int(math.sqrt(n_tokens-1)) # 14 ==> 14x14, 196+CLS
    patch_vit_new_size = model.patch_embed.proj.kernel_size[0] # retrieve the 16. Could put 16 straight
 
    cls = pos_embed[:,:1,:] # torch.Size([1, 1, 768])
    #print(cls.shape)
    patch=pos_embed[:,1:,:] # torch.Size([1, 196, 768])
    #print(patch.shape)
    
    patch = patch.reshape(1, patch_vit_old_size, patch_vit_old_size, embed_dim) # From [1, 196, 768] to [1, 14,14, 768]: [batch, row, column, embedding]
    patch = patch.permute(0, 3, 1, 2) # From [batch, row, column, embedding] to [batch, channels, height, width] (format that interpolate expects) [N, C, H, W]
    
    patch = functional.interpolate(patch, size=(patch_vit_new_size, patch_vit_new_size), mode='bicubic', align_corners=False) # try with others modes (bilinear)
    
    print(patch.shape) # torch.Size([1, 768, 16, 16])
    patch = patch.permute(0,2,3,1)
    patch = patch.reshape(1, patch_vit_new_size*patch_vit_new_size, embed_dim)
    print(patch.shape) # torch.Size([1, 256, 768]), add CLS token
    
    pos_embed_new = torch.cat([cls, patch], dim=1) 
    print(pos_embed_new.shape) #torch.Size([1, 257, 768]) good
    
    model.pos_embed = nn.Parameter(pos_embed_new)
    

    

if __name__ == "__main__":
    model = build_vit_encoder(in_chans=11)

    print("Patch embedding after surgery:")
    print(model.patch_embed.proj) #Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    print("Weight shape:", model.patch_embed.proj.weight.shape)
   
    # dummy = torch.randn(2, 11, 224, 224)
    # out = model.forward_features(dummy)
    # print("ViT feature output shape:", out.shape) #  torch.Size([2, 197, 768]): 2 batch size, 224/16=14, 14x14=196 + CLS token for transformers= 197
    # todo: we need to feed 256x256 ==> Original ViT accepts 224 images so 14x14 positional embeddings we need 16x16
    
    dummy = torch.randn(2, 11, 256, 256)
    out = model.forward_features(dummy)
    print("ViT feature output shape:", out.shape)   