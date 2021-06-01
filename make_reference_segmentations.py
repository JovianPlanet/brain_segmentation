import os
from pathlib import Path
from segmentation_and_analysis.segmentation_libraries import fsl_fast, dipy_segmentation

brains_dir = 'dataset/pretrain_datasets/features_fsl_strip'

dipy_out_dir = 'dataset/reference_segmentations/dipy'

fsl_out_dir = 'dataset/reference_segmentations/fsl'


Path(dipy_out_dir).mkdir(parents=True, exist_ok=True)
Path(fsl_out_dir).mkdir(parents=True, exist_ok=True)

subjects = next(os.walk(brains_dir))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?

for subject in subjects:

	fsl_fast(os.path.join(brains_dir, subject), 
			 os.path.join('/media/henryareiza/Disco_Compartido/david/codes', fsl_out_dir, 'fast_'+subject),
			 os.path.join('/media/henryareiza/Disco_Compartido/david/codes', fsl_out_dir, 'fast_'+subject))
	print('Head of subject {} segmented using FSL FAST'.format(subject))	

	#dipy_segmentation(subject_dir, subject_dir)
	#print('Brain of subject {} segmented using Dipy'.format(subject))

