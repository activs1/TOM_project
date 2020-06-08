from keras import backend as K

#to be changed and updated
def DICE(y_true, y_pred, smooth=1e-7):
    X = K.flatten(y_true)
    Y = K.flatten(y_pred)
    intersection = K.sum(X * Y, axis = 0)
    return (2. * intersection) / (K.sum(X, axis = 0) + K.sum(Y, axis = 0) + smooth)

    
def DICE_loss(y_true, y_pred):
  return -DICE(y_true, y_pred)