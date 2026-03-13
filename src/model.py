# Possible Models
# SegVit instead of U-Net
# U-net with Attention
# Swin transformer as encoder ==> # Swin-T encoder with 11-band input adaptation + U-Net decoder model


# U-Net with a pretrained transformer encoder that replaces the paper basic CNN encoder. I still need to figure out which transformer it has to fit the U-Net skip connections. Check resize dataset for ViT (224x224)
# Handle 11-band multispectral input, replacing the encoder patch embedding layer with one accepting 11 channels. 
# The pre trained ImageNet accepts only RGB, i will attribute those to the 3 corresponding band RGB
# The other 8 will have random weights
# With two learning rates, the new channels can learn faster than the pre-trained RGB ones. 
