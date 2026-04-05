# -*- coding: utf-8 -*-
'''
Original Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Source: https://github.com/marine-debris/marine-debris.github.io
Licence: MIT

Description: dataloader.py includes the appropriate data loader for
             pixel-level semantic segmentation.

Modifications:
- Replaced GDAL (osgeo.gdal) with rasterio for reading .tif files.
- Replaced relative path resolution (os.path.dirname) with explicit data_path parameter.
'''

import os
import torch
import random
import rasterio
import numpy as np
#from osgeo import gdal
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Pixel-Level class distribution (total sum equals 1.0)
class_distr = torch.Tensor([0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
 0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052])

bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

bands_std = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
 0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419]).astype('float32')

###############################################################
# Pixel-level Semantic Segmentation Data Loader               #
###############################################################

class GenDEBRIS(Dataset): # Extend PyTorch's Dataset class
    def __init__(self, mode = 'train', transform=None, standardization=None, path = 'data', agg_to_water= True):

        if mode=='train':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'train_X.txt'),dtype='str')

        elif mode=='test':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'test_X.txt'),dtype='str')

        elif mode=='val':
            self.ROIs = np.genfromtxt(os.path.join(path, 'splits', 'val_X.txt'),dtype='str')

        else:
            raise

        self.X = []           # Loaded Images
        self.y = []           # Loaded Output masks

        for roi in tqdm(self.ROIs, desc = 'Load '+mode+' set to memory'):

            # Construct file and folder name from roi
            roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])               # Get Folder Name
            roi_name = '_'.join(['S2'] + roi.split('_'))                      # Get File Name
            roi_file = os.path.join(path, 'patches', roi_folder,roi_name + '.tif')       # Get File path
            roi_file_cl = os.path.join(path, 'patches', roi_folder,roi_name + '_cl.tif') # Get Class Mask

            # Load Classification Mask (rasterio instead of GDAL)
            with rasterio.open(roi_file_cl) as src:
                temp = np.copy(src.read(1).astype(np.int64))

            # Aggregation
            if agg_to_water:
                temp[temp==15]=7          # Mixed Water to Marine Water Class
                temp[temp==14]=7          # Wakes to Marine Water Class
                temp[temp==13]=7          # Cloud Shadows to Marine Water Class
                temp[temp==12]=7          # Waves to Marine Water Class

            # Categories from 1 to 0
            temp = np.copy(temp - 1)

            self.y.append(temp)

            # Load Patch (rasterio instead of GDAL)
            with rasterio.open(roi_file) as src:
                temp = np.copy(src.read())
            self.X.append(temp)

        self.impute_nan = np.tile(bands_mean, (temp.shape[1],temp.shape[2],1))
        self.mode = mode
        self.transform = transform
        self.standardization = standardization
        self.length = len(self.y)
        self.path = path
        self.agg_to_water = agg_to_water

    def __len__(self):

        return self.length

    def getnames(self):
        return self.ROIs

    def __getitem__(self, index):

        img = self.X[index]
        target = self.y[index]

        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1]).astype('float32')       #(C, H, W) → (H, W, C)

        nan_mask = np.isnan(img) # Satellite images can have NaN values from sensor gaps, cloud masking, or missing data in certain pixels
        img[nan_mask] = self.impute_nan[nan_mask] # replaces any NaN pixels in the image with the precomputed band means

        if self.transform is not None:
            target = target[:,:,np.newaxis]
            stack = np.concatenate([img, target], axis=-1).astype('float32') # In order to rotate-transform both mask and image

            stack = self.transform(stack) 

            img = stack[:-1,:,:] # pulls out 11 bands
            target = stack[-1,:,:].long()   #  pulls out  mask                               # Recast target values back to int64 or torch long dtype

        if self.standardization is not None:
            img = self.standardization(img)

        return img, target

###############################################################
# Transformations                                             #
###############################################################
class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)

###############################################################
# Weighting Function for Semantic Segmentation                #
###############################################################
def gen_weights(class_distribution, c = 1.02):
    # Computes inverse-log class weights for the loss function to handle class imbalance
    # the model is penalized more for misclassifying rare classes like Marine Debris and Foam, and less for 
    # misclassifying dominant classes like water
    return 1/torch.log(c + class_distribution)
