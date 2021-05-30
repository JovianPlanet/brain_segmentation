import os

import numpy as np
from pathlib import Path
import argparse

from unet3D.get_subvolume import make_folders, get_all_training_sub_volumes, get_all_test_subvolumes
from segmentation_and_analysis.read_h5py import get_array, get_h5_keys 

parser = argparse.ArgumentParser()
parser.add_argument("h5_filename", help="Ruta del contenedor h5")
args = parser.parse_args()

tissues = {'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
            'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

feat_out_dir = 'dataset/3D/feat/'
mask_out_dir = 'dataset/3D/mask/'
val_feat_out_dir = 'dataset/3D/val_feat/'
val_mask_out_dir = 'dataset/3D/val_mask/'
test_feat_out_dir = 'dataset/3D/test_feat/'
test_mask_out_dir = 'dataset/3D/test_mask/'

vols = get_h5_keys(args.h5_filename, 'caracteristicas')
masks_GM = get_h5_keys(args.h5_filename, 'etiquetas_GM')
masks_WM = get_h5_keys(args.h5_filename, 'etiquetas_WM')
masks_CSF = get_h5_keys(args.h5_filename, 'etiquetas_CSF')
headers = get_h5_keys(args.h5_filename, 'affine')

np.random.seed(76)
idx = np.random.choice(len(vols), len(vols), replace=False)
print(idx)

for index_, i in enumerate(idx[:]):

	print(index_)

	img = get_array(args.h5_filename, 'caracteristicas', vols[i])
	label_GM = get_array(args.h5_filename, 'etiquetas_GM', masks_GM[i])
	label_WM = get_array(args.h5_filename, 'etiquetas_WM', masks_WM[i])
	label_CSF = get_array(args.h5_filename, 'etiquetas_CSF', masks_CSF[i])
	affine = get_array(args.h5_filename, 'affine', headers[i])

	if index_ < len(vols)-int(len(vols)*0.2):

		'''
		Crear subvolumenes de entrenamiento
		'''
		make_folders(feat_out_dir, mask_out_dir, vols[i])

		FEATURES_SAVE_PATH = os.path.join(feat_out_dir, vols[i])
		MASK_GM_SAVE_PATH = os.path.join(mask_out_dir, 'GM', vols[i])
		MASK_WM_SAVE_PATH = os.path.join(mask_out_dir, 'WM', vols[i])
		MASK_CSF_SAVE_PATH = os.path.join(mask_out_dir, 'CSF', vols[i])

		get_all_training_sub_volumes(img, label_GM, label_WM, label_CSF, affine, 
				   FEATURES_SAVE_PATH, MASK_GM_SAVE_PATH, 
				   MASK_WM_SAVE_PATH, MASK_CSF_SAVE_PATH, #classes=tissues[args.tejido],
				   orig_x = 240, orig_y = 240, orig_z = 48, 
				   output_x = 80, output_y = 80, output_z = 16,
				   stride_x = 40, stride_y = 40, stride_z = 16,
				   background_threshold=0.2)

	elif index_ >= len(vols)-int(len(vols)*0.2) and index_ < len(vols)-int(len(vols)*0.2/2):

		'''
		Crear subvolumenes de validacion
		'''
		make_folders(val_feat_out_dir, val_mask_out_dir, vols[i])

		FEATURES_SAVE_PATH = os.path.join(val_feat_out_dir, vols[i])
		MASK_GM_SAVE_PATH = os.path.join(val_mask_out_dir, 'GM', vols[i])
		MASK_WM_SAVE_PATH = os.path.join(val_mask_out_dir, 'WM', vols[i])
		MASK_CSF_SAVE_PATH = os.path.join(val_mask_out_dir, 'CSF', vols[i])

		get_all_test_subvolumes(img, label_GM, label_WM, label_CSF, affine, 
                        FEATURES_SAVE_PATH, MASK_GM_SAVE_PATH, 
				   		MASK_WM_SAVE_PATH, MASK_CSF_SAVE_PATH, 
                        orig_x = 240, orig_y = 240, orig_z = 48, 
                        output_x = 80, output_y = 80, output_z = 16,
                        stride_x = 80, stride_y = 80, stride_z = 16)

	else:

		'''
		Crear subvolumenes de prueba
		'''
		make_folders(test_feat_out_dir, test_mask_out_dir, vols[i])
		
		FEATURES_SAVE_PATH = os.path.join(test_feat_out_dir, vols[i])
		MASK_GM_SAVE_PATH = os.path.join(test_mask_out_dir, 'GM', vols[i])
		MASK_WM_SAVE_PATH = os.path.join(test_mask_out_dir, 'WM', vols[i])
		MASK_CSF_SAVE_PATH = os.path.join(test_mask_out_dir, 'CSF', vols[i])

		get_all_test_subvolumes(img, label_GM, label_WM, label_CSF, affine, 
                        FEATURES_SAVE_PATH, MASK_GM_SAVE_PATH, 
				   		MASK_WM_SAVE_PATH, MASK_CSF_SAVE_PATH, 
                        orig_x = 240, orig_y = 240, orig_z = 48, 
                        output_x = 80, output_y = 80, output_z = 16,
                        stride_x = 80, stride_y = 80, stride_z = 16)

