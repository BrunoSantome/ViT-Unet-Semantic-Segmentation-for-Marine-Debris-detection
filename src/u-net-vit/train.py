# -*- coding: utf-8 -*-
'''
Original Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Source: https://github.com/marine-debris/marine-debris.github.io
Licence: MIT

Description: train.py includes the training process for the
             pixel-level semantic segmentation.

Modifications:
- Replaced tensorboard (SummaryWriter) with wandb for experiment tracking.
- Replaced relative path resolution (os.path.dirname) with explicit paths.
- Removed --tensorboard arg, added --wandb_project arg. for logging

todo: Change training loop and pipeline for our models.
Cross-Entropy with class weighting or Focal Loss (like paper), 
Optimizer AdamW, do augmentation, Maybe early stopping, Hyperparameters with Optuna?
keep class weighting logic

'''

import os
import sys
import ast
import json
import random
import logging
import argparse

# Add parent directory (src/) to path so utils can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import wandb

from vit_unet import VitUnet
from dataloader import GenDEBRIS, bands_mean, bands_std, RandomRotationTransform , class_distr, gen_weights
from utils.metrics import Evaluation
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from utils.assets import labels_agg
# from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

logging.basicConfig(filename=os.path.join('logs','log_vit_unet.log'), filemode='a',level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info('*'*10)

# class SpectralJitter:
#     def __init__(self, brightness=0.1, contrast=0.1, p=0.5):
#         self.b, self.c, self.p = brightness, contrast, p
#     def __call__(self, x):
#         if torch.rand(1).item() < self.p:
#             C = x.shape[0]
#             bright = 1.0 + (torch.rand(C, 1, 1) * 2 - 1) * self.b
#             contr  = 1.0 + (torch.rand(C, 1, 1) * 2 - 1) * self.c
#             mean = x.mean(dim=(1, 2), keepdim=True)
#             x = (x - mean) * contr + mean * bright
#         return x

def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    # DataLoader Workers Reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

###############################################################
# Training                                                    #
###############################################################

def main(options):
    # Reproducibility
    # Limit the number of sources of nondeterministic behavior
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    # Wandb (replaces tensorboard)
    wandb.init(
        project=options['wandb_project'],
        name=options['run_name'],
        config=options,
    )

    # Transformations

    transform_train = transforms.Compose([transforms.ToTensor(),
                                    RandomRotationTransform([-90, 0, 90, 180]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    # transforms.RandomResizedCrop(size=256, scale=(0.7, 1.0), antialias=True) 
                                    # SpectralJitter(brightness=0.1, contrast=0.1, p=0.5),
                                    ])

    transform_test = transforms.Compose([transforms.ToTensor()])

    standardization = transforms.Normalize(bands_mean, bands_std)

    # Construct Data loader
    prefetch = options['prefetch_factor'] if options['num_workers'] > 0 else None
    persist  = options['persistent_workers'] if options['num_workers'] > 0 else False

    if options['mode']=='train':

        dataset_train = GenDEBRIS('train', transform=transform_train, standardization = standardization, path = options['data_path'], agg_to_water = options['agg_to_water'])
        dataset_test = GenDEBRIS('val', transform=transform_test, standardization = standardization, path = options['data_path'], agg_to_water = options['agg_to_water'])

        train_loader = DataLoader(  dataset_train,
                                    batch_size = options['batch'], #number of images processed at once
                                    shuffle = True, 
                                    num_workers = options['num_workers'], # handles multiple sub-process to load the dataset
                                    pin_memory = options['pin_memory'], # allocates data in page-locked (pinned) CPU memory, which makes CPU→GPU transfers faster useful if trained with cuda
                                    prefetch_factor = prefetch, # How many batches each worker loads in advance
                                    persistent_workers= persist,  # !
                                    worker_init_fn=seed_worker, #to recreate deterministic results
                                    generator=g) #  shuffling random number generator. Makes the shuffle order reproducible across runs

        test_loader = DataLoader(   dataset_test,
                                    batch_size = options['batch'],
                                    shuffle = False,
                                    num_workers = options['num_workers'],
                                    pin_memory = options['pin_memory'],
                                    prefetch_factor = prefetch,
                                    persistent_workers= persist,
                                    worker_init_fn=seed_worker,
                                    generator=g)

    elif options['mode']=='test':

        dataset_test = GenDEBRIS('test', transform=transform_test, standardization = standardization, path = options['data_path'], agg_to_water = options['agg_to_water'])

        test_loader = DataLoader(   dataset_test,
                                    batch_size = options['batch'],
                                    shuffle = False,
                                    num_workers = options['num_workers'],
                                    pin_memory = options['pin_memory'],
                                    prefetch_factor = prefetch,
                                    persistent_workers= persist,
                                    worker_init_fn=seed_worker,
                                    generator=g)
    else:
        raise

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
    if options['resume_from_epoch'] > 1:

        resume_model_dir = os.path.join(options['checkpoint_path'], str(options['resume_from_epoch']))
        model_file = os.path.join(resume_model_dir, 'model.pth')
        logging.info('Loading model files from folder: %s' % model_file)

        checkpoint = torch.load(model_file, map_location = device)
        model.load_state_dict(checkpoint) #it restores the exact learned weights from a previous run

        del checkpoint  # dereference ,free from memory !
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global class_distr
    # Aggregate Distribution Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water
    if options['agg_to_water']:
        agg_distr = sum(class_distr[-4:]) # Density of Mixed Water, Wakes, Cloud Shadows, Waves
        class_distr[6] += agg_distr       # To Water
        class_distr = class_distr[:-4]    # Drop Mixed Water, Wakes, Cloud Shadows, Waves

    # Weighted Cross Entropy Loss & adam optimizer
    weight = gen_weights(class_distr, c = options['weight_param']) #handles class imbalance
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction= 'mean', weight=weight.to(device))

    #optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['decay']) # weight_decay ==> L2 regularisation
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], weight_decay=options['decay']) # weight_decay ==> L2 regularisation
    # Learning Rate scheduler
    if options['reduce_lr_on_plateau']==1:
        #Adaptative
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) #  if the test loss doesn't improve for 10 consecutive evaluations, it reduces the LR
    else:
        #Fixed scheduler Changed from MultiStepLR
        # warmup_epochs = options['warmup_steps']
        # warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        # cosine = CosineAnnealingLR(optimizer, T_max=options['epochs'] - warmup_epochs, eta_min=1e-6)
        # scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, options['lr_steps'], gamma=0.1) # fixed changes of the learning rate

    # Start training
    start = options['resume_from_epoch'] + 1
    epochs = options['epochs']
    eval_every = options['eval_every']


    if options['mode']=='train':

        ###############################################################
        # Start Training                                              #
        ###############################################################
        model.train()
        best_f1=0
        for epoch in range(start, epochs+1):
            training_loss = []
            training_batches = 0

            for (image, target) in tqdm(train_loader, desc=f"training epoch {epoch}/{epochs}"):

                image = image.to(device)
                target = target.to(device)

                optimizer.zero_grad() # resets the gradients

                logits = model(image) #raw scores before softmax after the forward of the model

                loss = criterion(logits, target) #computes Categorical cross entropy

                loss.backward()

                training_batches += target.shape[0]

                training_loss.append((loss.data*target.shape[0]).tolist()) #un-averages the loss 

                optimizer.step()

                # Log running loss to wandb
                wandb.log({'training_loss_step': loss.item()})

            epoch_train_loss = sum(training_loss) / training_batches # gives a correct weighted average across the whole epoch
            logging.info("Training loss was: " + str(epoch_train_loss))

            ###############################################################
            # Start Evaluation                                            #
            ###############################################################

            if epoch % eval_every == 0 or epoch==1:
                model.eval()

                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for (image, target) in tqdm(test_loader, desc="testing"):

                        image = image.to(device)
                        target = target.to(device)

                        logits = model(image)

                        loss = criterion(logits, target)

                        # Accuracy metrics only on annotated pixels, not unlabeled ones (-1) meaningless for evaluation
                        # reshapes and filters predictions to only evaluate on labeled pixels
                        logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2)) # from (batch, classes, H, W) to (batch, H, W, classes)
                        logits = logits.reshape((-1,options['output_channels']))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]

                        
                        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy() #computes the softmax. need to move back to CPU because of evaluation metrics
                        target = target.cpu().numpy()

                        test_batches += target.shape[0]
                        test_loss.append((loss.data*target.shape[0]).tolist())
                        y_predicted += probs.argmax(1).tolist() #picks the highest probability, the predicted class
                        y_true += target.tolist() # accumulates all predictions and true for evaluation when epoch finish


                    y_predicted = np.asarray(y_predicted) #converstion required for sklearn metrics
                    y_true = np.asarray(y_true)

                    ####################################################################
                    # Save Scores to the .log file and log to wandb                    #
                    ####################################################################

                    acc = Evaluation(y_predicted, y_true)
                    epoch_test_loss = sum(test_loss) / test_batches
                    logging.info("\n")
                    logging.info("Test loss was: " + str(epoch_test_loss))
                    logging.info("STATISTICS AFTER EPOCH " +str(epoch) + ": \n")
                    logging.info("Evaluation: " + str(acc))
                    logging.info("Saving models")
                    # model_dir = os.path.join(options['checkpoint_path'], str(epoch))
                    checkpoint_path = options['checkpoint_path']
                    run_dir = os.path.join(checkpoint_path, options['run_name'])
                     #only save specific checkpoints, not 1 per epoch
                    os.makedirs(run_dir, exist_ok=True)
                    
                   
                    if acc["macroF1"] > best_f1:
                        best_f1 = acc["macroF1"] # we focus on macro F1 to save a checkpoint over another
                        torch.save(model.state_dict(), os.path.join(f"{checkpoint_path}/{options['run_name']}", 'best_model.pth'))

                    torch.save(model.state_dict(), os.path.join(f"{checkpoint_path}/{options['run_name']}", 'last_model.pth'))
                    # artifact = wandb.Artifact(
                    #     name=f"model-{wandb.run.name}",
                    #     type='model',
                    #     metadata={'epoch': epoch, 'macroF1': acc["macroF1"]}
                    # )
                    #artifact.add_file(os.path.join(run_dir, 'best_model.pth'))
                    #wandb.log_artifact(artifact, aliases=['best'])
                    
                    per_class_f1 = f1_score(y_true, y_predicted, average=None, zero_division=0)
                    per_class_iou = jaccard_score(y_true, y_predicted, average=None, zero_division=0)
                    per_class_prec = precision_score(y_true, y_predicted, average=None, zero_division=0)
                    per_class_rec = recall_score(y_true, y_predicted, average=None, zero_division=0)
                    
                    per_class_log = {}
                    for i, label in enumerate(labels_agg):
                        per_class_log[f'F1_per_class/{label}'] = per_class_f1[i]
                        per_class_log[f'IoU_per_class/{label}'] = per_class_iou[i]
                        per_class_log[f'Precision_per_class/{label}'] = per_class_prec[i]
                        per_class_log[f'Recall_per_class/{label}'] = per_class_rec[i]
                    # Log epoch-level metrics to wandb
                    wandb.log(per_class_log)
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': epoch_train_loss,
                        'test_loss': epoch_test_loss,
                        'Precision/test_macroPrec': acc["macroPrec"],
                        'Precision/test_microPrec': acc["microPrec"],
                        'Precision/test_weightPrec': acc["weightPrec"],
                        'Recall/test_macroRec': acc["macroRec"],
                        'Recall/test_microRec': acc["microRec"],
                        'Recall/test_weightRec': acc["weightRec"],
                        'F1/test_macroF1': acc["macroF1"],
                        'F1/test_microF1': acc["microF1"],
                        'F1/test_weightF1': acc["weightF1"],
                        'IoU/test_MacroIoU': acc["IoU"],
                    })


                if options['reduce_lr_on_plateau'] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()

                model.train()

    # CODE ONLY FOR EVALUATION - TESTING MODE !
    elif options['mode']=='test':

        model.eval()

        test_loss = []
        test_batches = 0
        y_true = []
        y_predicted = []

        with torch.no_grad():
            for (image, target) in tqdm(test_loader, desc="testing"):

                image = image.to(device)
                target = target.to(device)

                logits = model(image)

                loss = criterion(logits, target)

                # Accuracy metrics only on annotated pixels
                logits = torch.movedim(logits, (0,1,2,3), (0,3,1,2))
                logits = logits.reshape((-1,options['output_channels']))
                target = target.reshape(-1)
                mask = target != -1
                logits = logits[mask]
                target = target[mask]

                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                target = target.cpu().numpy()

                test_batches += target.shape[0]
                test_loss.append((loss.data*target.shape[0]).tolist())
                y_predicted += probs.argmax(1).tolist()
                y_true += target.tolist()

            y_predicted = np.asarray(y_predicted)
            y_true = np.asarray(y_true)

            ####################################################################
            # Save Scores to the .log file                                     #
            ####################################################################
            acc = Evaluation(y_predicted, y_true)
            logging.info("\n")
            logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
            logging.info("STATISTICS: \n")
            logging.info("Evaluation: " + str(acc))

    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Options 
    parser.add_argument('--agg_to_water', default=True, type=bool,  help='Aggregate Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water')

    parser.add_argument('--mode', default='train', help='select between train or test ')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to run')
    parser.add_argument('--batch', default=5, type=int, help='Batch size')
    parser.add_argument('--resume_from_epoch', default=0, type=int, help='load model from previous epoch')

    parser.add_argument('--img_size', default=256, type=int,help='The size (resolution) of the input images')
    parser.add_argument('--pretrained', default=True, type=bool,help='Whether to use the pre-trained weights for the RGB bands or not')
    
    parser.add_argument('--input_channels', default=11, type=int, help='Number of input bands')
    parser.add_argument('--output_channels', default=11, type=int, help='Number of output classes')
    parser.add_argument('--weight_param', default=1.03, type=float, help='Weighting parameter for Loss Function')

    
    # Optimization (tune with experiments) 
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--decay', default=0, type=float, help='learning rate decay')
    parser.add_argument('--reduce_lr_on_plateau', default=0, type=int, help='reduce learning rate when no increase (0 or 1)')
    parser.add_argument('--lr_steps', default='[40]', type=str, help='Specify the steps that the lr will be reduced')
    parser.add_argument('--warmup_steps', default=3, type=int, help='Specify the steps that the learning rate will warmup')
    # Evaluation/Checkpointing
    parser.add_argument('--checkpoint_path', default='checkpoints', help='folder to save vit Unet checkpoints into')
    parser.add_argument('--eval_every', default=1, type=int, help='How frequently to run evaluation (epochs)')

    # misc3
    parser.add_argument('--num_workers', default=1, type=int, help='How many cpus for loading data (0 is the main process)')
    parser.add_argument('--pin_memory', default=False, type=bool, help='Use pinned memory or not')
    parser.add_argument('--prefetch_factor', default=1, type=int, help='Number of sample loaded in advance by each worker')
    parser.add_argument('--persistent_workers', default=True, type=bool, help='This allows to maintain the workers Dataset instances alive.')
    parser.add_argument('--wandb_project', default='vit-Unet marida-segmentation', type=str, help='Wandb project name')
    parser.add_argument('--run_name', default=None, type=str, help='wandb run name (else random)')
    parser.add_argument('--data_path', default='data', type=str, help='Path to MARIDA dataset root')

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict

    # lr_steps list or single float
    lr_steps = ast.literal_eval(options['lr_steps'])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise

    options['lr_steps'] = lr_steps

    os.makedirs('logs', exist_ok=True)
    os.makedirs(options['checkpoint_path'], exist_ok=True)

    logging.info('parsed input parameters:')
    logging.info(json.dumps(options, indent = 2))
    main(options)
