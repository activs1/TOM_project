import numpy as np
import matplotlib.pyplot as plt

from starter_code.utils import load_case

'''
This file contains implemention of function used to save data to .png files in train and test folders
// Not saving to .png in this version, saving it straight to numpy arrays
'''

def save_train_test_cases_arrays(test_size=0.3, num_cases=60, type_='kidney', IMG_HEIGHT=256, IMG_WIDTH=256):
  '''
  Function deals with splitting dataset into train and test sets in view of cases from kits19 dataset
  Takes:
      test_size -> value of test set size (train_size = 1-test_size)
      num_cases -> how many cases from kits19 dataset to take (counting from 0)
      type_ -> string 'kidney' or 'tumour', what ind of mask to make
      
  Returns:
      X_train, Y_train, X_test, Y_test -> training and test sets of images and ground truths

  '''  
    
    
    
  from sklearn.model_selection import train_test_split
  from skimage.transform import resize
  CASES = np.linspace(0, num_cases-1, num_cases, dtype=np.uint8)
  print(CASES)
 
  CASES_TRAIN, CASES_TEST, _, _ = train_test_split(CASES, np.zeros((num_cases, 1)), test_size=test_size, random_state=123)
  print(f'Train {CASES_TRAIN}')
  print(f'Test {CASES_TEST}')
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []
  slice_train = 0
  slice_test = 0
 
  for case in range(0,len(CASES)):
    try:
      vol, seg = load_case(case)
      vol = vol.get_fdata()
      seg = seg.get_fdata()
      SLICES = vol.shape[0]
      print("SLICES: ", SLICES)
      if case in CASES_TRAIN:
        for slice in range(0, SLICES):
          if len(np.unique(seg[slice, :, :])) != 1:
            print(f'Saving slice {slice} of case {case} TRAIN')
            
            slice_train += 1
            
            img = vol[slice, :, :]
            img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='symmetric', preserve_range=True), axis=-1)
            
            mask = seg[slice, :, :]
            
            try:
                mask = make_mask(mask, type_)
            except exc:
                print(exc)
            mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='symmetric', preserve_range=True),axis=-1)
            
            X_train.append(img)
            Y_train.append(mask)
            
      elif case in CASES_TEST:
        for slice in range(0, SLICES):
          if len(np.unique(seg[slice, :, :])) != 1:
            print(f'Saving slice {slice} of case {case} TEST')
            
            slice_test +=1
            
            img = vol[slice, :, :]
            img = np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='symmetric', preserve_range=True), axis=-1)
            
            mask = seg[slice, :, :]
            try:
                mask = make_mask(mask, type_)
            except exc:
                print(exc)
            mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='symmetric', preserve_range=True),axis=-1)
            
            X_test.append(img)
            Y_test.append(mask)
      else:
        print("End")
    except err:
      print(err)
      
      
  X_train = np.reshape(np.array(X_train), (slice_train, IMG_HEIGHT, IMG_WIDTH, 1))
  Y_train = np.reshape(np.array(Y_train), (slice_train, IMG_HEIGHT, IMG_WIDTH, 1))
  X_test = np.reshape(np.array(X_test), (slice_test, IMG_HEIGHT, IMG_WIDTH, 1))
  Y_test = np.reshape(np.array(Y_test), (slice_test, IMG_HEIGHT, IMG_WIDTH, 1))
  return X_train, Y_train, X_test, Y_test


def make_mask(img, type_):
    '''
    Function makes mask from original mask from kits19 (containing 0, 1, 2, where
    0 is background, 1 is kidney, 2 is tumour)
    
    Takes:
        img -> original mask with 3 classes
        type_ -> which mask to make (string 'kidney' or 'tumour')
    
    Returns:
        mask -> mask of type Bool, showing kidney or tumour according to type_ argument
    '''
    
    TŁO = 0.
    
    if type_ == 'kidney':
        KIDNEY = 1.
        mask = np.where(img == KIDNEY, True,  False)
    
    elif type == 'tumour':
        TUMOUR = 2.
        mask = np.where(img == TUMOUR, True, False)
   
    else:
        raise Exception('MaskTypeNotFoundError') 
        
    return mask




