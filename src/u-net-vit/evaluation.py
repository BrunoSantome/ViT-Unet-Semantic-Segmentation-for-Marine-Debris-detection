# -*- coding: utf-8 -*-
'''
Original Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Source: https://github.com/marine-debris/marine-debris.github.io
Licence: MIT

Description: evaluation.py includes the code in order to produce
             the evaluation for each class as well as the prediction
             masks for the pixel-level semantic segmentation.

Modifications:
- Replaced relative path resolution (sys.path / os.path.dirname chains) with direct imports.
- Updated default paths for new project structure.
- Use rasterio to load .tif images
'''

import os
import random
import logging
import rasterio  #Python-friendly wrapper around GDAL for raster data
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from vit_unet import VitUnet
from dataloader import GenDEBRIS, bands_mean, bands_std
from utils.metrics import Evaluation, confusion_matrix, save_confusion_matrix
from utils.assets import labels

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

logging.basicConfig(filename=os.path.join('logs','evaluating_vit_unet.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)

def main(options):
    # Transformations

    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = transforms.Normalize(bands_mean, bands_std)

    # Construct Data loader

    dataset_test = GenDEBRIS('test', transform=transform_test, standardization = standardization, path = options['data_path'], agg_to_water = options['agg_to_water'])

    test_loader = DataLoader(   dataset_test,
                                batch_size = options['batch'],
                                shuffle = False)

    global labels
    # Aggregate Distribution Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water
    if options['agg_to_water']:
        labels = labels[:-4] # Drop Mixed Water, Wakes, Cloud Shadows, Waves

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = VitUnet(    
        in_channel=options['input_channels'],
        num_classes=options['output_channels'],
        img_size=options['img_size'],
        pretrained=options['pretrained'],
  )
    model.to(device)

    # Load model from specific epoch to continue the training or start the evaluation
    model_file = options['model_path']
    logging.info('Loading model files from folder: %s' % model_file)

    checkpoint = torch.load(model_file, map_location = device)
    model.load_state_dict(checkpoint)

    del checkpoint  # dereference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_predicted = []

    with torch.no_grad():
        for (image, target) in tqdm(test_loader, desc="testing"):

            image = image.to(device)
            target = target.to(device)

            logits = model(image)

            # Accuracy metrics only on annotated pixels
            logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
            logits = logits.reshape((-1,options['output_channels']))
            target = target.reshape(-1)
            mask = target != -1
            logits = logits[mask]
            target = target[mask]

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            target = target.cpu().numpy()

            y_predicted += probs.argmax(1).tolist()
            y_true += target.tolist()

        ####################################################################
        # Save Scores to the .log file                                     #
        ####################################################################
        acc = Evaluation(y_predicted, y_true)
        logging.info("\n")
        logging.info("STATISTICS: \n")
        logging.info("Evaluation: " + str(acc))
        print("Evaluation: " + str(acc))
        conf_mat = confusion_matrix(y_true, y_predicted, labels)
        save_confusion_matrix(y_true, y_predicted, labels)
        logging.info("Confusion Matrix:  \n" + str(conf_mat.to_string()))
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))
       

        if options['predict_masks']:

            path = os.path.join(options['data_path'], 'patches')
            ROIs = np.genfromtxt(os.path.join(options['data_path'], 'splits', 'test_X.txt'),dtype='str')

            impute_nan = np.tile(bands_mean, (256,256,1))

            for roi in tqdm(ROIs):

                roi_folder = '_'.join(['S2'] + roi.split('_')[:-1])             # Get Folder Name
                roi_name = '_'.join(['S2'] + roi.split('_'))                    # Get File Name
                roi_file = os.path.join(path, roi_folder,roi_name + '.tif')     # Get File path

                os.makedirs(options['gen_masks_path'], exist_ok=True)

                output_image = os.path.join(options['gen_masks_path'], os.path.basename(roi_file).split('.tif')[0] + '_vitunet.tif')

                # Read metadata of the initial image
                with rasterio.open(roi_file, mode ='r') as src:
                    tags = src.tags().copy()
                    meta = src.meta
                    image = src.read()
                    image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
                    dtype = src.read(1).dtype

                # Update meta to reflect the number of layers
                meta.update(count = 1)

                # Write it
                with rasterio.open(output_image, 'w', **meta) as dst:

                    # Preprocessing before prediction
                    nan_mask = np.isnan(image)
                    image[nan_mask] = impute_nan[nan_mask]

                    image = transform_test(image)

                    image = standardization(image)

                    # Image to Cuda if exist
                    image = image.to(device)

                    # Predictions
                    logits = model(image.unsqueeze(0))

                    probs = torch.nn.functional.softmax(logits.detach(), dim=1).cpu().numpy()

                    probs = probs.argmax(1).squeeze()+1

                    # Write the mask with georeference
                    dst.write_band(1, probs.astype(dtype).copy()) # In order to be in the same dtype
                    dst.update_tags(**tags)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--agg_to_water', default=True, type=bool,  help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')

    parser.add_argument('--batch', default=5, type=int, help='batch size')

    # Unet parameters
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    #parser.add_argument('--hidden_channels', default=16, type=int, help='Number of hidden features')

    parser.add_argument('--img_size', default=256, type=int,help='The size (resolution) of the input images')
    parser.add_argument('--pretrained', default=False, type=bool,help='Whether to use the pre-trained weights for the RGB bands or not')
    parser.add_argument('--data_path', default='data', type=str, help='Path to MARIDA dataset root')

   
    # Unet model path
    parser.add_argument('--model_path', default=os.path.join('checkpoints', 'run-decay-lr2e-4-batch16-epochs-60-REPRODUCIBLE', 'best_model.pth'), help='Path to Vit-Unet pytorch model')

    # Produce Predicted Masks
    parser.add_argument('--predict_masks', default= False, type=bool, help='Generate test set prediction masks?')
    parser.add_argument('--gen_masks_path', default=os.path.join('data', 'predicted_unet'), help='Path to where to produce store predictions')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict

    os.makedirs('logs', exist_ok=True)
    main(options)
