import os
import nibabel as nib
import numpy as np
import argparse
import h5py
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint
from unet2D.unet2D import unet2D, dice_coeff, dice_coef_loss
from segmentation_and_analysis.read_h5py import get_array, get_h5_keys

parser = argparse.ArgumentParser()
parser.add_argument("tejido", 
	choices=['GM', 'CSF', 'WM'], 
	help="El argumento debe ser uno de los tres tejidos a segmentar")
args = parser.parse_args()

TISSUE = args.tejido

weights_path = 'Pesos/2D/' + TISSUE + '.h5'
Path('Pesos/2D/').mkdir(parents=True, exist_ok=True)

tissues = { 'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
            'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

datos_ = 'dataset/dataset_completo.h5'

feats = get_h5_keys(datos_, 'caracteristicas')
masks = get_h5_keys(datos_, 'etiquetas_'+args.tejido)

#print(feats)
#print(masks)

'''
Debe quedar as'i para el caso general, con el mismo seed para unet3D
'''

np.random.seed(76)
idx = np.random.choice(len(feats), len(feats), replace=False)

'''
Para prueba organizo yo mismo los vectores iguales a como quedo el de unet3D 
sin el seed

idx = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 29, 
	   0, 14, 25, 
	   3, 7, 22]'''

print('{}'.format(idx))

train_images = []
train_masks = []
val_images = []
val_masks = []
test_images = []
test_masks = []

for index_, i in enumerate(idx[:]):
	#print('index_ = {}, img = {}'.format(index_, feats[index_]))
	#print('i = {}'.format(i))
	
	img_data = get_array(datos_, 'caracteristicas', feats[i])
	msk_data = get_array(datos_, 'etiquetas_'+args.tejido, masks[i])
	
	if index_ < len(feats)-int(len(feats)*0.2):
	    for j in range (img_data.shape[2]):
	        slice_ = img_data[:, :, j]
	        train_images.append(slice_)
  
	        slice_ = msk_data[:, :, j]==tissues[TISSUE] #np.where(msk_data[:, :, i]==3,1,0) #Convertir a binario
	        train_masks.append(slice_)

	elif index_ >= len(feats)-int(len(feats)*0.2) and index_ < len(feats)-int(len(feats)*0.2/2):
		for j in range (img_data.shape[2]):
			slice_ = img_data[:, :, j]
			val_images.append(slice_)

			slice_ = msk_data[:, :, j]==tissues[TISSUE] #np.where(msk_data[:, :, i]==3,1,0) #Convertir a binario
			val_masks.append(slice_)

	else:
		for j in range (img_data.shape[2]):
			slice_ = img_data[:, :, j]
			test_images.append(slice_)

			slice_ = msk_data[:, :, j]==tissues[TISSUE] #np.where(msk_data[:, :, i]==3,1,0) #Convertir a binario
			test_masks.append(slice_)

train_images = np.array(train_images).astype(np.int16)
train_masks = np.array(train_masks).astype(np.int16)
val_images = np.array(val_images).astype(np.int16)
val_masks = np.array(val_masks).astype(np.int16)
test_images = np.array(test_images).astype(np.int16)
test_masks = np.array(test_masks).astype(np.int16)

print('train_images = {}'.format(train_images.shape))
print('train_masks = {}'.format(train_masks.shape))
print('val_images = {}'.format(val_images.shape))
print('val_masks = {}'.format(val_masks.shape))
print('test_images = {}'.format(test_images.shape))
print('test_masks = {}'.format(test_masks.shape))

'''
DEFINICION DEL MODELO
'''
model_unet = unet2D(240, 240)

'''
Original
'''
model_unet.compile(optimizer='adam', loss = "binary_crossentropy", 
                    metrics = ['accuracy', dice_coeff]) 

'''
Igual que unet3D
model_unet.compile(optimizer='adam', loss = "categorical_crossentropy", 
                    metrics = ['accuracy', dice_coeff])
'''

#model_unet.summary()

'''
Original
callbacks = ModelCheckpoint(weights_path, 
                            verbose=1, save_best_only=True)
'''

callbacks = ModelCheckpoint(weights_path, 
                            monitor='val_dice_coeff', #accuracy', # val_acc
                            verbose=1, 
                            mode='max',
                            save_best_only=True)

#earlystopper = EarlyStopping(patience=10, verbose=1)

'''
ENTRENAMIENTO DE LA RED
'''
train_images = np.expand_dims(train_images, axis=3)
val_images=np.expand_dims(val_images, axis=3)

history = model_unet.fit(train_images,
					    train_masks,
					    epochs=100,
					    validation_data=(val_images, val_masks),
					    verbose=1,
					    callbacks=[callbacks])