import matplotlib.pyplot as plt
import numpy as np

'''
This file contains all functions that plots parameters progress and comparison of predicted mask with original one
'''

def plot_training(results):
    '''
    Function plots a (1,3) subplot that shows training progress of Dice Loss, Dice Coefficent and Intersection over
    Union.
    Takes:
        results ->  output of model.fit containing all history allowing to plot its' loss and metrics
    '''

    # print(results.history.keys()) #wszystkie klucze w historii
    plt.figure(figsize = (13,3), dpi = 200)
    plt.subplot(1,3,1)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Dice Loss')
    plt.ylabel('Dice loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1,3,2)
    plt.plot(results.history['iou_score'])
    plt.plot(results.history['val_iou_score'])
    plt.title('Intersection over Union')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1,3,3)
    plt.plot(results.history['f1-score'])
    plt.plot(results.history['val_f1-score'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def get_mask_on_img(img, mask):
    '''
    img - original image (grayscale, float)
    mask - predicted mask (uint with values for different elements or True and False in case of binary)
    Returns: rgb image with mask on image (red - tumour, green - kidney)
    '''
    from skimage.color import gray2rgb
    rgb_img = gray2rgb(img)

    ##

    mask_slice_tum = np.where(mask == 1, 1, 0)
    mask_slice_kid = np.where(mask == 2, 1, 0)

    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb_mask[:, :, 0] = mask_slice_tum
    rgb_mask[:, :, 1] = mask_slice_kid

    return rgb_img + rgb_mask


def show_subplot_results(img, mask, mask_pred, path=None):
    '''
    img - original image (grayscale, float)
    mask - original mask (uint or bool)
    mask_pred - predicted mask (uint or bool)
    path - path to save the subplot to

    Draws subplot with images and saves if path != None

    '''
    mask_orig_on_img = get_mask_on_img(img, mask)
    mask_pred_on_img = get_mask_on_img(img, mask_pred)

    plt.figure(num=1, dpi=300)
    plt.subplot(131)
    plt.imshow(mask_orig_on_img, cmap='gray')
    plt.title("Original image\nwith original mask", fontsize=10)
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask_pred, cmap='gray')
    plt.title("Predicted mask", fontsize=10)
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(mask_pred_on_img, cmap='gray')
    plt.title("Predicted mask\non image", fontsize=10)
    plt.axis('off')

    if path is not None:
        plt.savefig(path)