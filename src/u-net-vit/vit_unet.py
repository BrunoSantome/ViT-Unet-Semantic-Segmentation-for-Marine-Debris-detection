
import torch
import torch.nn as nn
import timm
from utils.assets import S2_RGB_INDICES, MODEL_DEFAULT


def build_vit_encoder(in_chans=11, pretrained=True, rgb_indices=S2_RGB_INDICES, model=MODEL_DEFAULT):
    model = timm.create_model(model, pretrained=pretrained)

    old_proj = model.patch_embed.proj #Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    embed_dim = old_proj.out_channels     # 768
    patch_size = old_proj.kernel_size     # (16, 16)
    old_weight = old_proj.weight.data     # [768, 3, 16, 16]
    old_bias = old_proj.bias.data if old_proj.bias is not None else None
    print(old_bias)
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

    return model


if __name__ == "__main__":
    model = build_vit_encoder(in_chans=11)

    print("Patch embedding after surgery:")
    print(model.patch_embed.proj) #Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    print("Weight shape:", model.patch_embed.proj.weight.shape)

    dummy = torch.randn(2, 11, 224, 224)
    out = model.forward_features(dummy)
    print("ViT feature output shape:", out.shape) #  torch.Size([2, 197, 768]): 2 batch size, 224/16=14, 14x14=196 + CLS token for transformers= 197
    # todo: we need to feed 256x256
