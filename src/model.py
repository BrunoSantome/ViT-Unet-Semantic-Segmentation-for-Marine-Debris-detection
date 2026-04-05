import tensorflow as tf
import numpy as np

from keras import Input
# from keras import Conv2D
# from keras import MaxPooling2D
# from keras import Dropout 
# from keras import Conv2DTranspose
# from keras import concatenate
import matplotlib.pyplot as plt
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import timm 




print(timm.list_models())

# Possible Models
# SegVit instead of U-Net
# U-net with Attention
# Swin transformer as encoder ==> # Swin-T encoder with 11-band input adaptation + U-Net decoder model


# U-Net with a pretrained transformer encoder that replaces the paper basic CNN encoder. I still need to figure out which transformer it has to fit the U-Net skip connections. Check resize dataset for ViT (224x224)
# Handle 11-band multispectral input, replacing the encoder patch embedding layer with one accepting 11 channels. 
# The pre trained ImageNet accepts only RGB, i will attribute those to the 3 corresponding band RGB
# The other 8 will have random weights
# With two learning rates, the new channels can learn faster than the pre-trained RGB ones. 
