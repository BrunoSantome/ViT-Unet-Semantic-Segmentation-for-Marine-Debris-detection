# -*- coding: utf-8 -*-
'''
Original Author: Ioannis Kakogeorgiou
Email: gkakogeorgiou@gmail.com
Source: https://github.com/marine-debris/marine-debris.github.io
Licence: MIT

Description: metrics.py includes the proposed metrics for
             pixel-level semantic segmentation.

Modifications: Removed multi-label classification functions (Evaluation_ML,
print_confusion_matrix_ML) as they are not used in this project.
'''
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score
import sklearn.metrics as metr
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Evaluation for Pixel-level semantic segmentation
def Evaluation(y_predicted, y_true):

    micro_prec = precision_score(y_true, y_predicted, average='micro')
    macro_prec = precision_score(y_true, y_predicted, average='macro')
    weight_prec = precision_score(y_true, y_predicted, average='weighted')

    micro_rec = recall_score(y_true, y_predicted, average='micro')
    macro_rec = recall_score(y_true, y_predicted, average='macro')
    weight_rec = recall_score(y_true, y_predicted, average='weighted')

    macro_f1 = f1_score(y_true, y_predicted, average="macro")
    micro_f1 = f1_score(y_true, y_predicted, average="micro")
    weight_f1 = f1_score(y_true, y_predicted, average="weighted")

    subset_acc = accuracy_score(y_true, y_predicted)

    iou_acc = jaccard_score(y_true, y_predicted, average='macro')

    info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "weightPrec" : weight_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "weightRec" : weight_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "weightF1" : weight_f1,
            "subsetAcc" : subset_acc,
            "IoU": iou_acc
            }

    return info

def print_confusion_matrix_only(y_gt, y_pred, labels):
    cm = metr.confusion_matrix(y_gt, y_pred)
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.index.name = "True \\ Pred"
    print(df.to_string())
    return df

def confusion_matrix(y_gt, y_pred, labels):

    # compute metrics
    cm      = metr.confusion_matrix  (y_gt, y_pred)
    f1_macro= metr.f1_score          (y_gt, y_pred, average='macro')
    mPA      = metr.recall_score      (y_gt, y_pred, average='macro')
    OA      = metr.accuracy_score    (y_gt, y_pred)
    UA      = metr.precision_score   (y_gt, y_pred, average=None)
    PA      = metr.recall_score      (y_gt, y_pred, average=None)
    f1      = metr.f1_score          (y_gt, y_pred, average=None)
    IoC     = metr.jaccard_score     (y_gt, y_pred, average=None)
    mIoC     = metr.jaccard_score    (y_gt, y_pred, average='macro')

    # confusion matrix
    sz1, sz2 = cm.shape
    cm_with_stats             = np.zeros((sz1+4,sz2+2))
    cm_with_stats[0:-4, 0:-2] = cm
    cm_with_stats[-3  , 0:-2] = np.round(IoC,2)
    cm_with_stats[-2  , 0:-2] = np.round(UA,2)
    cm_with_stats[-1  , 0:-2] = np.round(f1,2)
    cm_with_stats[0:-4,   -1] = np.round(PA,2)

    cm_with_stats[-4  , 0:-2] = np.sum(cm, axis=0)
    cm_with_stats[0:-4,   -2] = np.sum(cm, axis=1)

    # convert to list
    cm_list = cm_with_stats.tolist()

    # first row
    first_row = []
    first_row.extend (labels)
    first_row.append ('Sum')
    first_row.append ('Recall')

    # first col
    first_col = []
    first_col.extend(labels)
    first_col.append ('Sum')
    first_col.append ('IoU')
    first_col.append ('Precision')
    first_col.append ('F1-score')

    # fill rest of the text
    idx = 0
    for sublist in cm_list:
        if   idx == sz1:
            sublist[-2]  = 'mPA:'
            sublist[-1]  = round(mPA,2)
            cm_list[idx] = sublist
        elif   idx == sz1+1:
            sublist[-2]  = 'mIoU:'
            sublist[-1]  = round(mIoC,2)
            cm_list[idx] = sublist

        elif idx == sz1+2:
            sublist[-2]  = 'OA:'
            sublist[-1]  = round(OA,2)
            cm_list[idx] = sublist

        elif idx == sz1+3:
            cm_list[idx] = sublist
            sublist[-2]  = 'F1-macro:'
            sublist[-1]  = round(f1_macro,2)
        idx +=1

    # Convert to data frame
    df = pd.DataFrame(np.array(cm_list))
    df.columns = first_row
    df.index = first_col

    return df
