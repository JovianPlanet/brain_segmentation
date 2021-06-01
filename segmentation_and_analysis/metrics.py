import os
import sys
import nibabel as nib
import numpy as np

from sklearn.metrics import jaccard_score, accuracy_score, f1_score
from skimage import metrics

#import tensorflow as tf
#from tensorflow.keras import backend as K

np.set_printoptions(precision=3, suppress=True)

working_dir = '/media/henryareiza/Disco_Compartido/david/datasets/skullStrippedMRBrainS18DataNii/'
unet_dir = '/media/henryareiza/Disco_Compartido/david/datasets/skullStripped18Subvolumes/results'

subjects = next(os.walk(working_dir))[1]

def scikit_jaccard_score(seg_img, mask):
	return jaccard_score(mask, seg_img)

def dice_coeff(y_true, y_pred, axis=(1, 2, 3), 
                     epsilon=0.00001):
    #y_pred = tf.where(K.greater_equal(y_pred,0.5),1.,0.)
    # y_pred = y_pred.astype(np.float32) 
    dice_numerator = np.sum(2 * y_true * y_pred) + epsilon #K.sum(2 * y_true * y_pred, axis= axis) + epsilon
    dice_denominator = np.sum(y_true) + np.sum(y_pred) + epsilon
    dice_coefficient = np.mean(dice_numerator / dice_denominator)
    return dice_coefficient


if len(sys.argv)>1:
    TISSUE = sys.argv[1]
else:
    TISSUE = 'GM'

tissues = {'GM': '1', 'WM': '2', 'CSF': '0'}

for subject in subjects:
	files = next(os.walk(os.path.join(working_dir, subject)))[2]

	mask = nib.load(os.path.join(working_dir, subject, 'manual_mask_'+TISSUE+'.nii'))
	mask_data = np.where(mask.get_fdata()>0.5, 1, 0)
	
	dipy_seg = nib.load(os.path.join(working_dir, subject, 'fsl_dipy_'+TISSUE+'.nii'))
	dipy_seg_data = np.where(dipy_seg.get_fdata()>0.5, 1, 0)

	fsl_seg = nib.load(os.path.join(working_dir, subject, 'fsl_brainmask_seg_'+tissues[TISSUE]+'.nii'))
	fsl_seg_data = np.where(fsl_seg.get_fdata()>0.5, 1, 0)
	
	unet_seg = nib.load(os.path.join(unet_dir, subject, 'unet_seg_'+TISSUE+'.nii'))
	unet_seg_data = np.where(unet_seg.get_fdata()>0.5, 1, 0)
	
	print('Subject: {}'.format(subject))
	print('Tissue: {}\n'.format(TISSUE))

	print('FSL accuracy score: {:.3f}'.format(accuracy_score(mask_data.flatten(), fsl_seg_data.flatten())))
	print('DIPY accuracy score: {:.3f}'.format(accuracy_score(mask_data.flatten(), dipy_seg_data.flatten())))
	print('UNET3D accuracy score: {:.3f}\n'.format(accuracy_score(mask_data.flatten(), unet_seg_data.flatten())))

	print('FSL F1 score: {:.3f}'.format(f1_score(mask_data.flatten(), fsl_seg_data.flatten())))
	print('DIPY F1 score: {:.3f}'.format(f1_score(mask_data.flatten(), dipy_seg_data.flatten())))
	print('UNET3D F1 score: {:.3f}\n'.format(f1_score(mask_data.flatten(), unet_seg_data.flatten())))

	print('FSL Jaccard score: {:.3f}'.format(jaccard_score(mask_data.flatten(), fsl_seg_data.flatten())))
	print('DIPY Jaccard score: {:.3f}'.format(jaccard_score(mask_data.flatten(), dipy_seg_data.flatten())))
	print('UNET3D Jaccard score: {:.3f}\n'.format(jaccard_score(mask_data.flatten(), unet_seg_data.flatten())))
	'''
	print('FSL Hausdorff score: {}'.format(metrics.hausdorff_distance(mask_data, fsl_seg_data)))
	print('DIPY Hausdorff score: {}'.format(metrics.hausdorff_distance(mask_data, dipy_seg_data)))
	print('UNET3D Hausdorff score: {}\n'.format(metrics.hausdorff_distance(mask_data, unet_seg_data)))
	'''
	print('FSL structural similarity score: {:.3f}'.format(metrics.structural_similarity(mask_data, fsl_seg_data, data_range=np.max(mask_data) - np.min(mask_data))))
	print('DIPY structural similarity score: {:.3f}'.format(metrics.structural_similarity(mask_data, dipy_seg_data, data_range=np.max(mask_data) - np.min(mask_data))))
	print('UNET3D structural similarity score: {:.3f}\n'.format(metrics.structural_similarity(mask_data, unet_seg_data, data_range=np.max(mask_data) - np.min(mask_data))))

	print('FSL DICE score: {:.3f}'.format(dice_coeff(mask_data, fsl_seg_data)))
	print('DIPY DICE score: {:.3f}'.format(dice_coeff(mask_data, dipy_seg_data)))
	print('UNET3D DICE score: {:.3f}\n'.format(dice_coeff(mask_data, unet_seg_data)))
 
	