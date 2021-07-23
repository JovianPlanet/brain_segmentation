from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

'''
CONSTRUCCION RED UNET2D
'''
def unet2D(image_row, image_col):

    inputs = Input((image_row, image_col, 1)) 
    
    # IMAGEN DE ENTRADA
    p0 = inputs

    # PRIMERA CONVOLUCION  
    c1 = Conv2D(16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p0)
    c2 = Conv2D(16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c1)
    p1 = MaxPool2D((2,2))(c2)
    
    # SEGUNDA CONVOLUCION
    c3 = Conv2D(32, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p1)
    c4 = Conv2D(32, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c3)
    p2 = MaxPool2D((2,2))(c4)
    
    # TERCERA CONVOLUCION
    c5 = Conv2D(64, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p2)
    c6 = Conv2D(64, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c5)
    p3 = MaxPool2D((2,2))(c6)
    
    # CUARTA CONVOLUCION
    c7 = Conv2D(128, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p3)
    c8 = Conv2D(128, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c7)
    p4 = MaxPool2D((2,2))(c8)
    
    # QUINTA CONVOLUCION SIN POOLING
    c9 = Conv2D(256, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(p4)
    c10 = Conv2D(256, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c9)
    
    # PRIMERA DECONVOLUCION
    us1 = Conv2DTranspose(128 ,(2,2),strides=(2, 2), padding='same')(c10)
    concat1 = Concatenate()([us1,c8])
    c11 = Conv2D(128, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat1)
    c12 = Conv2D(128, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c11)
    
    # SEGUNDA DECONVOLUCION
    us2 = Conv2DTranspose(64 ,(2,2),strides=(2, 2), padding='same')(c12)
    concat2 = Concatenate()([us2,c6])
    c13 = Conv2D(64, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat2)
    c14 = Conv2D(64, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c13)
    
    # TERCERA DECONVOLUCION
    us3 = Conv2DTranspose(32 ,(2,2),strides=(2, 2), padding='same')(c14)
    concat3 = Concatenate()([us3,c4])
    c15 = Conv2D(32, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat3)
    c16 = Conv2D(32, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c15)
    
    # CUARTA DECONVOLUCION
    us4 = Conv2DTranspose(16 ,(2,2),strides=(2, 2), padding='same')(c16)
    concat4 = Concatenate()([us4,c2])
    c17 = Conv2D(16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(concat4)
    c18 = Conv2D(16, kernel_size=(3, 3), padding="same", strides=1, activation="relu")(c17)
    
    # SALIDAS
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(c18)

    model = Model(inputs, outputs)

    return model

'''
METRICA
'''
def dice_coeff(y_true, y_pred, axis=(1, 2), epsilon=0.00001):
  
    dice_numerator = K.sum(2 * y_true * y_pred, axis=axis) + epsilon
    dice_denominator = K.sum(y_true,axis=axis) + K.sum(y_pred, axis=axis) + epsilon
    dice_coefficient = K.mean(dice_numerator/dice_denominator, axis=0)
    return dice_coefficient
    '''
    if dice_coefficient>0.1:
        return dice_coefficient
    else:
        return 1.0'''

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

'''
Plotear imagen original, mascara y prediccion
'''
def plot_slice(prueba_img, test_mask_img, predictions_img):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1 ,wspace=1)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.title.set_text('imagen')
    ax1.axis("off")
    ax1.imshow(prueba_img, cmap="gray")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.title.set_text('mask')
    ax2.axis("off")
    ax2.imshow(test_mask_img, cmap="gray")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.title.set_text('prediccion')
    ax3.axis("off")
    ax3.imshow(predictions_img>0.9, cmap="gray")

    plt.show()
