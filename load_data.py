from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Concatenate, Input, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import he_normal, he_uniform
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy, Precision, Recall 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

from starter_code.utils import load_case

'''
This file contains implemention of function used to save data to .png files in train and test folders
Also, there's an implementation of generator, which yields images and masks in batches
'''

CASES_TRAIN = 20
CASES_TEST = 5

# TODO: Implement code fragment to remove all_black images
# Right now we are loading all slices in an image, which causes drastic imbalance

def save_train_test_cases(train_cases=20, test_cases=5, path_train, path_test, verbose = False, verbose_step=100):
    for case in range(0, train_cases):
        vol, seg = load_case(case)
        vol = vol.get_data()
        seg = seg.get_data()
        SLICES = vol.shape[0]

        for slice in range(0, SLICES):
            plt.imsave(f'{path_train}/image/img/img_case{case}_slice{slice}.png', rgb2gray(vol[slice, :, :]))
            plt.imsave(f'{path_train}/mask/img/seg_case{case}_slice{slice}.png', rgb2gray(seg[slice, :, :]))
            
            if verbose and slice % verbose_step == 0:
                print(f'Saved train case {case} slice {slice}')


    for case in range(train_cases, train_cases + test_cases):
        vol, seg = load_case(case)
        vol = vol.get_data()
        seg = seg.get_data()
        SLICES = vol.shape[0]

        for slice in range(0, SLICES):
            plt.imsave(f'{path_test}/image/img/img_case{case}_slice{slice}_test.png', rgb2gray(vol[slice, :, :]))
            plt.imsave(f'{path_test}/mask/img/seg_case{case}_slice{slice}_test.png', rgb2gray(seg[slice, :, :]))
            
            if verbose and slice % verbose_step == 0:
                print(f'Saved test case {case} slice {slice}')


# Image generator
def my_image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
      # LABELS 30 110 215 -> 30 background, 110 -> kidney, 215 -> tumour
      #background = 30 / 255
      kidney = 110 #/ 255
      tumour = 215 #/ 255

      # ONE HOT ENCODING (is it OK?)
      mask = rgb2gray(mask.reshape(BATCH_SIZE, 512, 512, 1)) #/ 255
      new_mask = np.zeros((BATCH_SIZE, 512, 512, 2))
      for batch in range(0, BATCH_SIZE):
        for x in range (0, 512):
          for y in range (0, 512):
            #if mask[batch, x, y] == background: ??
              #new_mask[batch, x, y, 0] = 1 ??
            if mask[batch, x, y] == kidney:
              new_mask[batch, x, y, 0] = 1
            elif mask[batch, x, y] == tumour:
              new_mask[batch, x, y, 1] = 1

    
      yield ((rgb2gray(img).reshape(BATCH_SIZE, 512, 512, 1)) / 255, new_mask)  


path_train = 'path' # to be changed     
path_test = 'path'

save_train_test_cases(train_cases=20, test_cases=5, path_train, path_test, verbose = True, verbose_step=100)

