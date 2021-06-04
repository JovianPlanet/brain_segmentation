import os
from pathlib import Path
from segmentation_and_analysis.segmentation_libraries import fsl_fast, dipy_segmentation, ants_segmentation
from segmentation_and_analysis.utils import change_dtype

brains_dir = 'dataset/pretrain_datasets/features_fsl_strip'
dipy_out_dir = 'dataset/reference_segmentations/dipy'
fsl_out_dir = 'dataset/reference_segmentations/fsl'

Path(dipy_out_dir).mkdir(parents=True, exist_ok=True)
Path(fsl_out_dir).mkdir(parents=True, exist_ok=True)

subjects = next(os.walk(brains_dir))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?

for subject in subjects:

	fast = fsl_fast(os.path.join(brains_dir, subject), 
	'/media/henryareiza/Disco_Compartido/david/codes/dataset/reference_segmentations/fsl/'+'HOLAfast_'+subject)

	ext = 7 if '.gz' in subject else 4

	Path(os.path.join(brains_dir, subject[:-ext]+'_seg_0.nii')).rename(os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_CSF.nii'))
	Path(os.path.join(brains_dir, subject[:-ext]+'_seg_1.nii')).rename(os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_GM.nii'))
	Path(os.path.join(brains_dir, subject[:-ext]+'_seg_2.nii')).rename(os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_WM.nii'))
	Path(os.path.join(brains_dir, subject[:-ext]+'_seg.nii')).unlink()

	change_dtype(os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_CSF.nii'), 
				 os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_CSF.nii'))
	change_dtype(os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_GM.nii'), 
				 os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_GM.nii'))
	change_dtype(os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_WM.nii'), 
				 os.path.join(fsl_out_dir, subject[:-ext]+'_fsl_WM.nii')) 			 
	print('Brain of subject {} segmented using FSL FAST'.format(subject))	

	dipy_segmentation(os.path.join(brains_dir, subject), 
					  os.path.join(dipy_out_dir, subject[:-ext]))  

	print('Brain of subject {} segmented using Dipy'.format(subject))

print('Done!')

