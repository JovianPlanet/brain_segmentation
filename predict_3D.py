import os
import numpy as np
import argparse
import nibabel as nib

from dipy.io.image import load_nifti

from tensorflow.keras.losses import categorical_crossentropy

from natsort import natsorted
from pathlib import Path

from unet3D.tf_unet3D import unet_3D, dice_coeff, plot_slice

'''
Codigo que utiliza los pesos entrenados por la UNET3D para segmentar un nuevo volumen, generando un
nuevo volumen reconstruido a partir de cada subvolumen
'''

np.set_printoptions(precision=2, suppress=True)

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

result_folder = os.path.join('predictions', '3D') #'predictions/3D'
test_feat_path = os.path.join('dataset', '3D', 'test_feat') #'dataset/3D/test_feat'
test_mask_path = os.path.join('dataset', '3D', 'test_mask') #'dataset/3D/test_mask'
pesos_folder = os.path.join('Pesos', '3D')

model_unet=unet_3D(80, 80, 16)
model_unet.compile(optimizer='adam', loss = "categorical_crossentropy", 
                    metrics = ['accuracy', dice_coeff]) 

model_unet.load_weights(os.path.join(pesos_folder, args.tejido + '.h5'))

for subject in natsorted(os.listdir(test_feat_path)):
    print ('Predicting labels for subject {}...'.format(subject))

    test_images=[]

    test_files = next(os.walk(os.path.join(test_feat_path, subject)))[2]

    for test_subvol in natsorted(test_files):
        img_data, img_affine = load_nifti(os.path.join(test_feat_path, 
                                        subject, test_subvol)) 
        test_images.append(img_data)
    
    test_mask=[]

    mask_files = next(os.walk(os.path.join(test_mask_path, args.tejido, 
                                            subject)))[2]

    for test_submask in natsorted(mask_files):
        img_mask, msk_affine = load_nifti(os.path.join(test_mask_path, 
                                args.tejido, subject, test_submask)) 
        img_mask= img_mask==tissues[TISSUE]
        test_mask.append(img_mask)

    test_images = np.array(test_images)
    test_mask = np.array(test_mask)

    test_images = test_images.astype(np.float32) #/ 255
    test_mask = test_mask.astype(np.float32)

    '''
    VALIDACION DEL MODELO
    Genera la mascara con las predicciones y reconstruye el volumen 3D a partir de ellas
    '''
    segmented = np.zeros((240, 240, 48))
    j = 0

    for i in range (test_mask.shape[0]):
        prueba = test_images[i,...]
        prueba = prueba[np.newaxis]

        predictions = model_unet.predict(prueba)
        predictions = predictions.squeeze()
        prueba = prueba.squeeze()

        if i%3 == 0:
        	j = 0
        else:
        	j += 1

        segmented[j*80:(j*80)+80, 
                ((i%9)//3)*80:(((i%9)//3)*80)+80, 
                (i//9)*16:(i//9)*16+16] = predictions>0.5

        #plot_slice(prueba, test_mask, predictions)

    Path(os.path.join(result_folder, subject)).mkdir(parents=True, exist_ok=True)
    nii_segmented = nib.Nifti1Image(segmented, img_affine) # np.eye(4)
    # nib.save(nii_segmented, 
    #         os.path.join(result_folder, args.tejido, 
    #         subject, 'unet3D_predictions_'+TISSUE+'.nii'))

    nib.save(nii_segmented, 
            os.path.join(result_folder, 
            subject, 'unet3D_predictions_'+TISSUE+'.nii'))


