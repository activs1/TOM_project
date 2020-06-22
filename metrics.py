import numpy as np

'''
This file contains metrics used to evaluate our model.
'''

def np_dice_coef(y_true, y_pred, smooth = 1e-05):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    
    return (2. * intersection + smooth) / (union + smooth)


def np_VD_score(y_true, y_pred):  
    y_true = np.sum(y_true.flatten() == True)
    y_pred = np.sum(y_pred.flatten() == True)
    
    return (y_pred - y_true)/y_true

def np_IoU(y_true, y_pred, smooth = 1.): 
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    
    return (intersection + smooth) / (union + smooth)