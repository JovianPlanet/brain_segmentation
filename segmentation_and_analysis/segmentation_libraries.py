import os

import nibabel as nib
import numpy as np

from pathlib import Path

from nipype.interfaces.freesurfer import ReconAll, MRIConvert
from nipype.interfaces.fsl import BET, FAST

from dipy.io.image import load_nifti_data, load_nifti
from dipy.segment.tissue import TissueClassifierHMRF

np.set_printoptions(precision=2, suppress=True)


'''
Performs freesurfer's skull stripping
'''
def freesurfer_skullstrip(): #(subjects, in_files, sub_id):
	reconall = ReconAll()
	reconall.inputs.subject_id = '1' #sub_id
	reconall.inputs.directive = 'autorecon1'
	reconall.inputs.flags = ['-nomotioncor', '-nonuintensitycor', '-notalairach', '-nonormalization']
	reconall.inputs.subjects_dir = '/media/henryareiza/Disco_Compartido/david/datasets/skullStrippedMRBrainS13DataNii/' #subjects
	reconall.inputs.T1_files = '/media/henryareiza/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData/1/T1.nii' #in_files 
	reconall.inputs.openmp = 4 					# Number of processors to use in parallel

	reconall.run()

'''
Converts an .mgz image file to .nii
'''
def freesurfer_mriconvert(in_files, out_files):
	mriconvert = MRIConvert()
	mriconvert.inputs.in_file = in_files #'/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/skullStrippedMRBrainS13DataNii/1/mri/brainmask.mgz'
	mriconvert.inputs.out_file = out_files #'/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/skullStrippedMRBrainS13DataNii/1/mri/fs_brainmask.nii'
	mriconvert.inputs.out_type = 'nii'

	mriconvert.run()

'''
Performs FSL's skull striping
'''
def fsl_bet(input_file, out_file):
	skullstrip = BET()
	skullstrip.inputs.in_file = input_file 		#os.path.join(head_path, head)
	skullstrip.inputs.out_file = out_file 		#os.path.join(brain_path, fsl_brain)
	skullstrip.inputs.frac = 0.2				# [0-1] Valores mas pequenos estiman un area mayor de cerebro
	skullstrip.inputs.robust = True
	skullstrip.run()

'''
Employs the FSL's FAST algorithm to segment brain tissues [CSF, GM, WM]
''' 
def fsl_fast(input_file, out_file, TCM):
	fslfast = FAST()
	fslfast.inputs.in_files = input_file
	#fslfast.inputs.out_basename = out_file
	fslfast.inputs.img_type = 1 				# 1=T1, 2=T2, 3=PD
	fslfast.inputs.number_classes = 3 			# WM, GM, CSF
	fslfast.inputs.segments = True
	fslfast.inputs.no_pve = True
	fslfast.inputs.output_type = 'NIFTI'
	#fslfast.outputs.tissue_class_map = TCM

	fslfast.run()

'''
Performs segmentation of brain tissues by calling the DIPY library
'''
def dipy_segmentation(brain_path, out_path):
	nclass = 4
	beta = 0.1
	img, static_affine = load_nifti(os.path.join(brain_path, 'fsl_brainmask.nii.gz'))
	hmrf = TissueClassifierHMRF()
	initial_segmentation, final_segmentation, PVE = hmrf.classify(img, nclass, beta, max_iter=20)

	nii_CSF = nib.Nifti1Image(PVE[:,:,:,0], static_affine)
	nib.save(nii_CSF, os.path.join(out_path, 'fsl_dipy_CSF.nii'))
	nii_GM = nib.Nifti1Image(PVE[:,:,:,1], static_affine)
	nib.save(nii_GM, os.path.join(out_path, 'fsl_dipy_GM.nii'))
	nii_WM = nib.Nifti1Image(PVE[:,:,:,2], static_affine)
	nib.save(nii_WM, os.path.join(out_path, 'fsl_dipy_WM.nii'))

'''
En Desarrollo

def apply_affine_to_freesurfer_brain():
	img_orig, static_affine = load_nifti(os.path.join(head_path, head))
	img_brain, fs_static_affine = load_nifti(os.path.join(brain_path, 'brainmask.nii'))
	#new_orientation = nib.orientations.ornt_transform(fs_static_affine, static_affine)
	print(static_affine)
	print(fs_static_affine)
	#print(new_orientation)
	affined_nii = nib.Nifti1Image(img_brain, np.eye(4))
	nib.save(affined_nii, os.path.join(brain_path, 'fs_affined_brainmask.nii'))
'''

def get_manual_mask(tissues, img, out_file, filename):
	Path(out_file).mkdir(parents=True, exist_ok=True)
	img_mask, static_affine = load_nifti(img)	 
	for tissue, value in tissues.items():
		print(tissue)
		print(value)
		tissue_mask= np.where(img_mask==value, value, 0)
		nii_mask = nib.Nifti1Image(tissue_mask*1, static_affine)
		nib.save(nii_mask, os.path.join(out_file, filename+tissue+'.nii'))

'''
heads_path = '/media/henryareiza/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData/'
head = 'T1.nii'
#head = 'reg_T1.nii.gz'
brain_path = '/media/henryareiza/Disco_Compartido/david/datasets/skullStrippedMRBrainS13DataNii/'
freesurfer_brain = 'mri/brainmask.mgz'
fsl_brain = 'fsl_brainmask.nii.gz'
fsl_fast_root_name = 'fsl_fast_'
#mask = 'segm.nii.gz'
mask = 'LabelsForTraining.nii'

cortical_tissues = {'CORTICAL_GRAY_MATTER': 1, 'BASAL_GANGLIA': 2, 
					'WHITE_MATTER': 3, 'WHITE_MATTER_LESIONS': 4,
					'CEREBROSPINAL_FLUID': 5, 'VENTRICLES': 6, 
					'CEREBELLUM': 7, 'BRAINSTEM': 8}'''

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
'''
subjects = next(os.walk(heads_path))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?

for subject in subjects:

	subject_dir = os.path.join(brain_path, subject)
	subject_head = os.path.join(heads_path, subject, head)
	subject_fsl_brain_path = os.path.join(brain_path, subject, fsl_brain)

	Path(subject_dir).mkdir(parents=True, exist_ok=True)

	#freesurfer_skullstrip() #(brain_path, os.path.join(heads_path, subject, head), subject)
	#freesurfer_mriconvert() #(os.path.join(brain_path, subject, freesurfer_brain), os.path.join(brain_path, subject, 'fs_brainmask.nii'))

	fsl_bet(subject_head, subject_fsl_brain_path)
	print('Brain of subject {} extracted using FSL BET'.format(subject))
	
	fsl_fast(subject_fsl_brain_path, fsl_fast_root_name)
	print('Head of subject {} segmented using FSL FAST'.format(subject))
	
	dipy_segmentation(subject_dir, subject_dir)
	print('Brain of subject {} segmented using Dipy'.format(subject))
	
	#get_manual_mask({'GM': 1, 'WM': 3, 'CSF': 5}, os.path.join(heads_path, subject, mask), subject_dir)
	get_manual_mask({'GM': 1, 'WM': 3, 'CSF': 5}, 
		os.path.join(heads_path, subject, mask), 
		'/media/henryareiza/Disco_Compartido/david/datasets/full_dataset/MRBrainS13_MASKS/', 
		'mrb13_mask_'+subject+'_')

	print('Manual masks for subject {} obtained\n'.format(subject))'''
	