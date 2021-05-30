'''
Script para generar el dataset de volumenes
'''

from utils import *
from segmentation_libraries import get_manual_mask
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dataset", 
	choices=['13', '18', 'IBSR'], 
	help="Nombre del dataset a crear")
args = parser.parse_args()


MRBrains13_dir = '/media/henryareiza/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData/'
MRBrains18_dir = '/media/henryareiza/Disco_Compartido/david/datasets/MRBrainS18DataNii/training/'

MRBrains13_FSL_strip = '/media/henryareiza/Disco_Compartido/david/datasets/MRBrains13_fsl_strip/'
MRBrains18_FSL_strip = '/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/MRBrains18_oriented_fsl_strip'

FSL_skull_stripped_ds = '/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/features_fsl_strip'

IBSR_stripped_dir = '/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/IBSR_strip_registered'
IBSR_seg_labels = '/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/IBSR_seg_registered'
labels_dir = '/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/labels'

MRBrainS13_MASKS = '/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/MRBrainS13_MASKS/'
MRBrainS18_MASKS = '/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/MRBrainS18_MASKS/'

Path(FSL_skull_stripped_ds).mkdir(parents=True, exist_ok=True)
Path(MRBrains13_FSL_strip).mkdir(parents=True, exist_ok=True)
Path(MRBrains18_FSL_strip).mkdir(parents=True, exist_ok=True)


#girarVolumen(out_dir=MRBrains18_FSL_strip)
#girarVolumen(out_dir=MRBrains18_FSL_strip, mask=True)
'''
subdirectories = next(os.walk(MRBrains13_dir))[1]

for dir_ in subdirectories:
	skull_strip_fsl_BET(os.path.join(MRBrains13_dir, dir_, 'T1.nii'), 
						os.path.join(MRBrains13_FSL_strip, 'strip_MRBrainS13_' + dir_+'_T1.nii'))

	change_dtype(os.path.join(MRBrains13_FSL_strip, 'strip_MRBrainS13_' + dir_+'_T1.nii.gz'), 
				os.path.join(FSL_skull_stripped_ds, 'strip_MRBrainS13_' + dir_+'_T1.nii'))
'''
if args.dataset == '18':

	filenames = next(os.walk(MRBrains18_FSL_strip))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?

	for vol in filenames:
		'''
		if 'reg_T1' in vol:

			skull_strip_fsl_BET(os.path.join(MRBrains18_FSL_strip, vol), 
								os.path.join(MRBrains18_FSL_strip, 'strip_' + vol))

			change_dtype(os.path.join(MRBrains18_FSL_strip, 'strip_' + vol), 
						os.path.join(FSL_skull_stripped_ds, 'strip_'+vol))
		'''
		if 'segm' in vol:
			get_manual_mask({'GM': 1, 'WM': 3, 'CSF': 5}, 
							os.path.join(MRBrains18_FSL_strip, vol), 
							MRBrainS18_MASKS, 
							'MRBrainS18_mask_'+vol.split('_')[1]+'_')

	masks = next(os.walk(MRBrainS18_MASKS))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?
	for mask in masks:
		change_dtype(os.path.join(MRBrainS18_MASKS, mask), 
					os.path.join(MRBrainS18_MASKS, mask))

elif args.dataset == '13':
	filenames = next(os.walk(MRBrains13_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?

	for mri in filenames:
    
	    #img_dir = sorted(filenames)[mri]
	    print(mri)
	    
	    #img = nib.load(os.path.join(MRBrains13_dir, mri, 'LabelsForTraining.nii'))
	    #img_data = img.get_fdata()

	    #msk = nib.load(os.path.join(MRBrains13_dir, mri, 'LabelsForTraining.nii')) 
	    #img_mask = msk.get_fdata()

	    get_manual_mask({'GM': 1, 'WM': 3, 'CSF': 5}, 
						os.path.join(MRBrains13_dir, mri, 'LabelsForTraining.nii'), 
						MRBrainS13_MASKS, 
						'MRBrainS13_mask_'+mri+'_')

	masks = next(os.walk(MRBrainS13_MASKS))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?
	for mask in masks:
		change_dtype(os.path.join(MRBrainS13_MASKS, mask), 
					os.path.join(MRBrainS13_MASKS, mask))
	
'''
mri_dir = next(os.walk(IBSR_stripped_dir))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?
for mri in mri_dir[:]:
	#change_dtype(os.path.join(IBSR_stripped_dir, mri), os.path.join(FSL_skull_stripped_ds, 'strip_'+mri))
	normalize_labels(IBSR_seg_labels, mri, labels_dir)


girarMRSrain18()
extraerCraneo(metodo, dataset)
registrarIBSR()
generarH5PY()
'''
