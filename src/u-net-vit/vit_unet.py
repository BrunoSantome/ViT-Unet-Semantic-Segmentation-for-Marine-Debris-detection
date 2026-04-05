
import torch
import torch.nn as nn
import timm
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.assets import S2_RGB_INDICES, MODEL_DEFAULT
import math
import torch.nn.functional as F

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
    
    patch = F.interpolate(patch, size=(patch_vit_new_size, patch_vit_new_size), mode='bicubic', align_corners=False) # try with others modes (bilinear)
    
    print(patch.shape) # torch.Size([1, 768, 16, 16])
    patch = patch.permute(0,2,3,1)
    patch = patch.reshape(1, patch_vit_new_size*patch_vit_new_size, embed_dim)
    print(patch.shape) # torch.Size([1, 256, 768]), add CLS token
    
    pos_embed_new = torch.cat([cls, patch], dim=1) 
    print(pos_embed_new.shape) #torch.Size([1, 257, 768]) good
    
    model.pos_embed = nn.Parameter(pos_embed_new)
    

################ Encoder ########################

class VitEncoder(nn.Module):
    def __init__(self,  in_chans=11, img_size=256, pretrained=True, layers=(2, 5, 8, 11)):
        super().__init__()
        self.vit = build_vit_encoder(in_chans=in_chans, img_size=img_size, pretrained=pretrained)
        self.skip_connections_layers = layers

    def forward(self, x):
        #get_intermediate_layers runs a forward pass internally
        extract_layers= self.vit.get_intermediate_layers(x, n=self.skip_connections_layers, reshape=True)
        return list(extract_layers)
        
 
################## Decoder Blocks ################# 
 
class SingleDeConv2DBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x): 
        return self.block(x)      


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2)) #The padding is to keep the spatial size

    def forward(self, x):
        return self.block(x)  


class Conv2DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_channel, out_channel, kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeConv2DBlock(in_channel, out_channel),
            SingleConv2DBlock(out_channel, out_channel, kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


################## Decoder ####################

class Decoder(nn.Module):
    def __init__(self, input_dim=11, output_dim=11,embed_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        
        self.decoder0 = \
            nn.Sequential(
                Conv2DBlock(input_dim,32,3),
                Conv2DBlock(32,64,3)
            )
        
        self.decoder3 = \
            nn.Sequential(
                Deconv2DBlock(embed_dim,512),
                Deconv2DBlock(512, 256),
                Deconv2DBlock(256, 128)
            )
            
        self.decoder6 = \
            nn.Sequential(
                Deconv2DBlock(embed_dim, 512),
                Deconv2DBlock(512, 256)
            )
        
        self.decoder9 = Deconv2DBlock(embed_dim, 512)
        
        self.decoder12_upsampler = SingleDeConv2DBlock(embed_dim, 512)
        
        self.decoder9_upsampler = \
            nn.Sequential(
                Conv2DBlock(1024, 512),
                Conv2DBlock(512, 512),
                Conv2DBlock(512, 512),
                SingleDeConv2DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2DBlock(512, 256),
                Conv2DBlock(256, 256),
                SingleDeConv2DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2DBlock(256, 128),
                Conv2DBlock(128, 128),
                SingleDeConv2DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2DBlock(128, 64),
                Conv2DBlock(64, 64),
                SingleConv2DBlock(64, output_dim, 1)
            )
    
    def forward(self, x, encoder_layers):
        z0, z3, z6, z9, z12 = x, *encoder_layers
        
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output
        
        

class VitUnet(nn.Module):
    def __init__(self, 
                 in_channel=11,
                 num_classes=11,
                 img_size=256,
                 embed_dim=768,
                 pretrained=True,
                 layers=(2, 5, 8, 11)):
        super().__init__()
        self.encoder = VitEncoder(in_chans=in_channel, img_size=img_size, pretrained=pretrained, layers=layers)
        self.decoder = Decoder(input_dim=in_channel, output_dim=num_classes, embed_dim=embed_dim)
        
    def forward(self, x):
        x_out = self.encoder(x)
        logits = self.decoder(x, x_out)
        return logits
        

if __name__ == "__main__":
    model = build_vit_encoder(in_chans=11)

    print("Patch embedding after surgery:")
    print(model.patch_embed.proj) #Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    print("Weight shape:", model.patch_embed.proj.weight.shape)
    #print(model) to view the whole model architecture
    # dummy = torch.randn(2, 11, 224, 224)
    # out = model.forward_features(dummy)
    # print("ViT feature output shape:", out.shape) #  torch.Size([2, 197, 768]): 2 batch size, 224/16=14, 14x14=196 + CLS token for transformers= 197
    # todo: we need to feed 256x256 ==> Original ViT accepts 224 images so 14x14 positional embeddings we need 16x16
    
    dummy = torch.randn(2, 11, 256, 256)
    out = model.forward_features(dummy)
    
    print("ViT feature output shape:", out.shape)   
    extract_layers = model.get_intermediate_layers(dummy, n=[2, 5, 8, 11], reshape=True) #timm manages the permutation and reshape for me, no need to do it manually.
    
    print(len(extract_layers))
    for i, f in enumerate(extract_layers):
          print(f"  Layer {[2,5,8,11][i]+1}: shape={f.shape}")

    """
    Layer 3: shape=torch.Size([2, 768, 16, 16])
    Layer 6: shape=torch.Size([2, 768, 16, 16])
    Layer 9: shape=torch.Size([2, 768, 16, 16])
    Layer 12: shape=torch.Size([2, 768, 16, 16])
    
    CLS token not included. 
    """
    
    model = VitUnet(in_channel=11, num_classes=11, img_size=256)
    dummy = torch.randn(2, 11, 256, 256)
    out = model(dummy)
    print(out.shape)  # [2, 11, 256, 256] READY FOR TRAINING

          
   