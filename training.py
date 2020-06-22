from UNET_model import *
from load_data import save_train_test_cases_arrays
from plots import plot_training

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import Model
from segmentation_models.metrics import FScore, IOUScore
from segmentation_models.losses import DiceLoss



'''
This file is used to train model
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


model_kidney = get_small_UNET((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), classes=1)
callbacks_kidney = [
        EarlyStopping(patience=2, monitor='val_loss'),
        ModelCheckpoint("Unet_kidney_check.h5", monitor='val_loss', verbose=True, save_freq=10, save_best_only=True)]


X_train, Y_train, X_test, Y_test = save_train_test_cases_arrays(test_size=0.3, num_cases=60, IMG_HEIGHT, IMG_WIDTH, type_ = 'kidney')
TRAIN_SAMPLES_KIDNEY = Y_train.shape[0]
TEST_SAMPLES_KIDNEY = Y_test.shape[0]

model_kidney.compile(optimizer=Adam(learning_rate=1e-4), loss = dice_loss, metrics=[dice_metric,iou_metric])                                                                                                                                         
results_kidney = model_kidney.fit(X_train, Y_train, validation_data =(X_test, Y_test), batch_size=BATCH_SIZE, epochs=EPOCHS,
                    steps_per_epoch = TRAIN_SAMPLES_KIDNEY/BATCH_SIZE, validation_steps = TEST_SAMPLES_KIDNEY/BATCH_SIZE, callbacks=callbacks_kidney,verbose = 1)
    
model.save("Unet_kidney.h5", save_format='h5')

plot_training(results_kidney)


'''
Training tumor
'''


model_tumor = get_small_UNET((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), classes=1)
callbacks_tumor = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        ModelCheckpoint("Unet_tumor_check.h5", monitor='val_loss', verbose=True, save_freq=10, save_best_only=True)]

X_train, Y_train, X_test, Y_test = save_train_test_cases_arrays(test_size=0.3, num_cases=60, IMG_HEIGHT, IMG_WIDTH, type_ = 'tumor')
TRAIN_SAMPLES_TUMOR = Y_train.shape[0]
TEST_SAMPLES_TUMOR = Y_test.shape[0]

model_tumor.compile(optimizer=Adam(learning_rate=1e-4), loss = dice_loss, metrics=[dice_metric,iou_metric])
results_tumor = model_tumor.fit(X_train, Y_train, validation_data =(X_test, Y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    steps_per_epoch = TRAIN_SAMPLES_TUMOR/BATCH_SIZE, validation_steps = TEST_SAMPLES_TUMOR/BATCH_SIZE, callbacks=callbacks, verbose = 1)
    
model.save("Unet_tumor.h5", save_format='h5')

plot_training(results_tumor)