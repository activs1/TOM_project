from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from load_data import my_image_mask_generator
from UNET_model import get_UNET

'''
This file is used to train model
'''

WIDTH, HEIGHT = 512, 512
BATCH_SIZE = 4
SAVE_PATH = ''
EPOCHS = 1
SEED = 100
TARGET_SIZE = (WIDTH, HEIGHT)

model = get_UNET((WIDTH, HEIGHT, 1))

image_data_generator = ImageDataGenerator().flow_from_directory('/content/drive/My Drive/kits19/train/image/', class_mode = None, batch_size = BATCH_SIZE, target_size = TARGET_SIZE, seed = SEED)
mask_data_generator = ImageDataGenerator().flow_from_directory('/content/drive/My Drive/kits19/train/mask/', class_mode = None, batch_size = BATCH_SIZE, target_size = TARGET_SIZE, seed = SEED)

train_generator = my_image_mask_generator(image_data_generator, mask_data_generator, BATCH_SIZE)

image_test_data_generator = ImageDataGenerator().flow_from_directory('/content/drive/My Drive/kits19/test/image/', class_mode = None, batch_size = BATCH_SIZE, target_size = TARGET_SIZE, seed = SEED)
mask_test_data_generator = ImageDataGenerator().flow_from_directory('/content/drive/My Drive/kits19/test/mask/', class_mode = None, batch_size = BATCH_SIZE, target_size = TARGET_SIZE, seed = SEED)

test_generator = my_image_mask_generator(image_test_data_generator, mask_test_data_generator)

model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)   ##fit_generator(train_generator, steps_per_epoch=4514//4, validation_data=test_generator, validation_steps=771//4, epochs=1)


model.save("/content/drive/My Drive/kits19/model_unet_.h5", save_format='h5')
