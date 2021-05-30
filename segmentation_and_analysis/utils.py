import os
from pathlib import Path
import nibabel as nib
import numpy as np
from nipype.interfaces.fsl import BET

def girarVolumen(ref_path='/media/henryareiza/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData/2/T1.nii', 
					in_dir='/media/henryareiza/Disco_Compartido/david/datasets/MRBrainS18DataNii/training', 
					out_dir='',
					mask=False):

	ref_vol = nib.load(ref_path)

	mri_dir = next(os.walk(in_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?

	Path(out_dir).mkdir(parents=True, exist_ok=True)

	if mask:
		for directory in mri_dir[:]:
			target = nib.load(os.path.join(in_dir, directory, 'segm.nii.gz'))
			target_data = np.int16(target.get_fdata())

			new_vol = nib.Nifti1Image(target_data, ref_vol.affine)
			nib.save(new_vol, os.path.join(out_dir, 'MRBrainS18_'+directory+'_segm.nii.gz'))
			print('done')
		return


	for directory in mri_dir[:]:
		target = nib.load(os.path.join(in_dir, directory, 'reg_T1.nii.gz'))
		target_data = np.int16(target.get_fdata())

		new_vol = nib.Nifti1Image(target_data, ref_vol.affine)
		nib.save(new_vol, os.path.join(out_dir, 'MRBrainS18_'+directory+'_reg_T1.nii.gz'))
		print('done')

def change_dtype(vol, filename):
	target = nib.load(vol)
	target_data = np.int16(target.get_fdata())

	new_vol = nib.Nifti1Image(target_data, target.affine)
	nib.save(new_vol, filename)

def skull_strip_fsl_BET(input_file, out_file):
	skullstrip = BET()
	skullstrip.inputs.in_file = input_file 		#os.path.join(head_path, head)
	skullstrip.inputs.out_file = out_file 		#os.path.join(brain_path, fsl_brain)
	skullstrip.inputs.frac = 0.2
	skullstrip.inputs.robust = True
	skullstrip.run()

def normalize_labels(in_dir, in_file, out_dir):

	filepath = os.path.join(in_dir, in_file)
	csf_dir = out_dir+'_CSF'
	gm_dir = out_dir+'_GM'
	wm_dir = out_dir+'_WM'

	Path(csf_dir).mkdir(parents=True, exist_ok=True)
	Path(gm_dir).mkdir(parents=True, exist_ok=True)
	Path(wm_dir).mkdir(parents=True, exist_ok=True)

	target = nib.load(filepath)
	target_data = target.get_fdata()

	csf_data = np.int16(np.where(target_data == 1, 5, 0)) #5 if target_data == 1 else 0
	gm_data = np.int16(np.where(target_data == 2, 1, 0)) #1 if target_data == 2 else 0
	wm_data = np.int16(np.where(target_data == 3, 3, 0)) #target_data == 3

	csf_vol = nib.Nifti1Image(csf_data, target.affine)
	nib.save(csf_vol, os.path.join(csf_dir, in_file[:-4]+'_MASK_CSF.nii'))

	gm_vol = nib.Nifti1Image(gm_data, target.affine)
	nib.save(gm_vol, os.path.join(gm_dir, in_file[:-4]+'_MASK_GM.nii'))

	wm_vol = nib.Nifti1Image(wm_data, target.affine)
	nib.save(wm_vol, os.path.join(wm_dir, in_file[:-4]+'_MASK_WM.nii'))