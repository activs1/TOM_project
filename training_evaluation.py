# IMPORTS
from UNET_model import *
from load_data import save_train_test_cases_arrays
from plots import *
from metrics import *

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import Model
from segmentation_models.metrics import FScore, IOUScore
from segmentation_models.losses import DiceLoss

import numpy as np


'''
This file is used to train  and evaluate model
'''

IMG_WIDTH, IMG_HEIGHT = 256,256
BATCH_SIZE = 4
EPOCHS = 15
IMG_CHANNELS = 1

dice_loss = DiceLoss(beta=1) 
dice_metric = FScore(beta=1)
iou_metric = IOUScore()



'''
Training kidney
'''


def train_network(X_train, Y_train, X_test, Y_test, type_):
    '''
    Function trains model in regard to type_ (kidney model or tumour model)

    Takes:
        X_train -> arrays with training images
        Y_train -> array with training masks
        X_test -> array with test images
        Y_test -> array with test masks
        type_ -> string, 'kidney' or 'tumour'

    Saves best model during training (with ModelCheckpoint callback), also saves model at the end of training
    Plots changes of loss, dice coefficient and IoU score during training (plot_training function)
    '''


    model = get_small_UNET((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), classes=1)
    callbacks = [
        EarlyStopping(patience=2, monitor='val_loss'),
        ModelCheckpoint(f"Unet_{type_}_checkpoint.h5", monitor='val_loss', verbose=True, save_freq=10, save_best_only=True)]

    TRAIN_SAMPLES = Y_train.shape[0]
    TEST_SAMPLES = Y_test.shape[0]

    model.compile(optimizer=Adam(learning_rate=1e-4), loss = dice_loss, metrics=[dice_metric,iou_metric])                                                                                                                                         
    results = model_kidney.fit(X_train, Y_train, validation_data =(X_test, Y_test), batch_size=BATCH_SIZE, epochs=EPOCHS,
                        steps_per_epoch = TRAIN_SAMPLES//BATCH_SIZE, validation_steps = TEST_SAMPLES//BATCH_SIZE, callbacks=callbacks,verbose = 1)

    model.save(f"Unet_{type_}.h5", save_format='h5')

    plot_training(results)

'''
Evaluate
'''

def evaluate_network(X_train, Y_train, X_test, Y_test, type_):
    '''
    This function evaluates the model according to the type_ (kidney or tumour), it loads model from disk.
    It makes prediction on train and test images and calculates VD score.

    Takes:
        X_train -> arrays with training images
        Y_train -> array with training masks
        X_test -> array with test images
        Y_test -> array with test masks
        type_ -> string, 'kidney' or 'tumour'

    Plots images with masks (original and predicted) for visual evaulation of segmentation
    coords is a list of subjectively chosen indexes (previously) which show the best, medium and worst segmentation results
    '''
    model_fname = f"Unet_{type_}_checkpoint.h5"
    model = load_model(model_fname, compile = True,
                       custom_objects={"dice_loss": dice_loss, "f1-score":dice_metric, "iou_score":iou_metric})

    preds_train = model.predict(X_train, verbose=1)
    preds_test = model.predict(X_test, verbose=1)
    preds_train_mask = (preds_train > 0.5).astype(np.uint8)
    preds_test_mask = (preds_test > 0.5).astype(np.uint8)


    vd_train = np_VD_score(np.squeeze(Y_train), np.squeeze(preds_train_mask))
    vd_test = np_VD_score(np.squeeze(Y_test), np.squeeze(preds_test_mask))

    print('VD train: ', np.round(vd_train, 3))
    print('VD test: ', np.round(vd_test, 3))

    if type_ == 'kidney':
        coords = [44, 74, 290, 108, 119, 1123, 101, 105, 1124]
    elif type_ == 'tumour':
        coords = [361, 448, 7, 89, 3, 1126]



    for i in coords:
        org = np.squeeze(Y_test[i])
        seg = np.squeeze(preds_test_mask[i])
        dice = np_dice_coef(org, seg)
        iou = np_IoU(org, seg)
        vd = np_VD_score(org, seg)
        print(f' Slice Index: {i}, Dice: {dice}, IoU: {iou}, VD: {vd}')
        show_subplot_results(np.squeeze(X_test[i]), np.squeeze(Y_test[i]), np.squeeze(preds_test_mask[i]))



if __name__ == '__main__':

    # Split data into train and test sets for kidney model
    X_train_k, Y_train_k, X_test_k, Y_test_k = save_train_test_cases_arrays(test_size=0.3, num_cases=60, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH= IMG_WIDTH, type_ = 'kidney')
    train_network(X_train_k, Y_train_k, X_test_k, Y_test_k, 'kidney')

    # Split data into train and test sets for tumour model
    X_train_t, Y_train_t, X_test_t, Y_test_t = save_train_test_cases_arrays(test_size=0.3, num_cases=60, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH= IMG_WIDTH, type_ = 'tumour')
    train_network(X_train_t, Y_train_t, X_test_t, Y_test_t, 'tumour')


    evaluate_network(X_train_k, Y_train_k, X_test_k, Y_test_k, 'kidney')
    evaluate_network(X_train_t, Y_train_t, X_test_t, Y_test_t, 'tumour')

