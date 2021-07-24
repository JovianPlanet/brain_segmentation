import os
import sys
import h5py
import argparse
import numpy as np
import nibabel as nib

from dipy.io.image import load_nifti
from natsort import natsorted
from pathlib import Path

from tensorflow.keras.losses import categorical_crossentropy

from unet2D.unet2D import unet2D, dice_coeff, dice_coef_loss, plot_slice
from segmentation_and_analysis.read_h5py import get_array, get_h5_keys

'''
Codigo que utiliza los pesos entrenados por la UNET2D para segmentar un nuevo volumen, generando un
nuevo volumen reconstruido a partir de cada slice
'''

'''
CORTICAL_GRAY_MATTER = 1
BASAL_GANGLIA = 2
WHITE_MATTER = 3 
WHITE_MATTER_LESIONS = 4
CEREBROSPINAL_FLUID = 5
VENTRICLES = 6
CEREBELLUM = 7
BRAINSTEM = 8
'''

parser = argparse.ArgumentParser()
parser.add_argument("tejido", choices=['GM', 'CSF', 'WM'], 
                    help="Tejido a reconstruir")
args = parser.parse_args()

TISSUE = args.tejido
tissues = { 'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
            'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

result_folder = os.path.join('predictions', '2D') #'predictions/2D'
datos_ = os.path.join('dataset', 'dataset_completo.h5') #'dataset/dataset_completo.h5'
pesos_folder = os.path.join('Pesos', '2D')

feats = get_h5_keys(datos_, 'caracteristicas')
masks = get_h5_keys(datos_, 'etiquetas_'+args.tejido)

'''
Debe quedar asi para el caso general, con el mismo seed para unet3D
El seed se escoge de manera que queden:
- Una imagen de MRBrainS13 para validacion y una para test
- Una imagen de MRBrainS18 para validacion y una para test
- Una imagen de IBSR para validacion y una para test
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

model_unet = unet2D(240, 240)

model_unet.compile(optimizer='adam', loss = "binary_crossentropy", 
                    metrics = ['accuracy', dice_coeff])

'''
model_unet.compile(optimizer='adam', 
                   loss="categorical_crossentropy", 
                   metrics=['accuracy', dice_coeff])'''

model_unet.load_weights(os.path.join(pesos_folder, args.tejido + '.h5'))

for index_, i in enumerate(idx[len(feats)-int(len(feats)*0.2/2):]):
	test_images = []
	test_masks = []
	print('imagen = {}'.format(feats[i]))
	img_data = get_array(datos_, 'caracteristicas', feats[i])
	msk_data = get_array(datos_, 'etiquetas_'+args.tejido, masks[i])
	affine = get_array(datos_, 'affine', feats[i])

	for j in range (img_data.shape[2]):
		slice_ = img_data[:, :, j]
		test_images.append(slice_)

		slice_ = msk_data[:, :, j]==tissues[TISSUE] #np.where(msk_data[:, :, i]==3,1,0) #Convertir a binario
		test_masks.append(slice_)

	test_images = np.array(test_images).astype(np.int16)
	test_masks = np.array(test_masks).astype(np.int16)

	print('\ntest images = {}'.format(test_images.shape))
	print('test masks = {}\n'.format(test_masks.shape))

	'''
	VALIDACION DEL MODELO
	Genera la mascara con las predicciones y reconstruye el volumen 3D 
	a partir de ellas
	'''
	segmented = np.zeros((240, 240, 48))

	for k in range (test_masks.shape[0]):
		prueba = test_images[k,...]
		prueba = prueba[np.newaxis]
		#print(prueba.shape)
		predictions = model_unet.predict(prueba)
		predictions = predictions.squeeze()
		prueba = prueba.squeeze()

		#print('prueba = {}'.format(prueba.shape))
		#print('predictions = {}'.format(predictions.shape))

		segmented[:,:,k] = predictions > 0.5

		#plot_slice(prueba, test_masks[i,:,:], predictions)

	print('segmented = {}\n'.format(segmented.shape))

	Path(os.path.join(result_folder, feats[i])).mkdir(parents=True, exist_ok=True)
	nii_segmented = nib.Nifti1Image(segmented, affine) # np.eye(4)
	nib.save(nii_segmented, 
		     os.path.join(result_folder, feats[i], 'unet2D_predictions_'+TISSUE+'.nii'))

