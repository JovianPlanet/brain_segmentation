import os
import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
from unet3D.tf_unet3D import *

'''
Codigo para entrenar la red UNET3D
'''

np.set_printoptions(precision=2, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("tejido", 
                    choices=['GM', 'CSF', 'WM'], 
                    help="Tejido a modelar")

args = parser.parse_args()

TISSUE = args.tejido

tissues = { 'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
            'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

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

feat_dir = os.path.join('dataset', '3D', 'feat') #'dataset/3D/feat/'
mask_dir = os.path.join('dataset', '3D', 'mask', args.tejido)
val_feat_dir = os.path.join('dataset', '3D', 'val_feat')
val_mask_dir = os.path.join('dataset', '3D', 'val_mask', args.tejido)
pesos_folder = os.path.join('Pesos', '3D')

weights_dir = os.path.join(pesos_folder, TISSUE+'.h5') #'Pesos/3D/' + TISSUE + '.h5'
Path(pesos_folder).mkdir(parents=True, exist_ok=True)

train_ids = next(os.walk(feat_dir))[1] # [2]: files; [1]: directories
print(train_ids)

images=[]
for feat_folder in train_ids:
    for subvol in sorted(os.listdir(os.path.join(feat_dir, feat_folder))):
        
        img = nib.load(os.path.join(feat_dir,feat_folder,subvol))
        img_data = img.get_fdata()
        images.append(img_data)
        img.uncache()

mask=[]
for submasks_folder in train_ids:
    for submask in sorted(os.listdir(os.path.join(mask_dir, submasks_folder))):
        msk = nib.load(os.path.join(mask_dir,submasks_folder,submask))
        img_mask = msk.get_fdata()  
        img_mask= img_mask==tissues[TISSUE]
        mask.append(img_mask)
        img.uncache()

train_ids = next(os.walk(val_feat_dir))[1]

val_images=[]
for val_feat_folder in train_ids:
    for val_subvol in sorted(os.listdir(os.path.join(val_feat_dir, val_feat_folder))):
        img = nib.load(os.path.join(val_feat_dir, val_feat_folder, val_subvol))
        img_data = img.get_fdata()  
        val_images.append(img_data)
    
val_mask=[]
for val_mask_folder in train_ids:
    for val_submask in sorted(os.listdir(os.path.join(val_mask_dir, val_mask_folder))):
        msk = nib.load(os.path.join(val_mask_dir, val_mask_folder, val_submask))
        img_mask = msk.get_fdata()  
        img_mask= img_mask==tissues[TISSUE]
        val_mask.append(img_mask)

images=np.array(images)
print(images.shape)
mask=np.array(mask)
print(mask.shape)
val_images=np.array(val_images)
print(val_images.shape)
val_mask=np.array(val_mask)
print('suma valmask = {}'.format(np.sum(val_mask)))

images = images.astype(np.int16) #/ 255
mask = mask.astype(np.int16) 

val_images = val_images.astype(np.int16) #/ 255
val_mask = val_mask.astype(np.int16)

model_unet=unet_3D(80, 80, 16) 
model_unet.compile(optimizer='adam', loss = 'binary_crossentropy', #"categorical_crossentropy", 
                    metrics = ['accuracy', dice_coeff]) 

'''
Callback para guardar el mejor modelo segun el valor del coeficiente DICE en los datos de validacion
'''
callbacks = ModelCheckpoint(weights_dir, 
                            monitor='val_dice_coeff', #accuracy', # val_acc
                            verbose=1, 
                            mode='max',
                            save_best_only=True)

history = model_unet.fit(images,
                         mask,
                         epochs=100,
                         validation_data=(val_images, val_mask),
                         verbose=2,
                         callbacks=[callbacks]) #Guardar la mejor epoca para validación

# model_unet.summary()

'''
#VALIDACION DEL MODELO
'''
for i in range (val_mask.shape[0]):
    prueba = val_images[i,...]
    prueba = prueba[np.newaxis]

    prueba_mask = val_mask[i,...]

    predictions = model_unet.predict(prueba)
    predictions = predictions.squeeze()
    prueba = prueba.squeeze()
    #print('predictions shape: ', predictions.shape)

    plot_slice(prueba, prueba_mask, predictions)

print(val_mask.shape)