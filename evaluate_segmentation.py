import os
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import jaccard_score, accuracy_score, f1_score, roc_auc_score
from skimage import metrics
from segmentation_and_analysis.read_h5py import get_array, get_h5_keys
#from segmentation_and_analysis.metrics import dice_coeff

np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("tejido", choices=['GM', 'CSF', 'WM'], 
                    help="Tejido a evaluar")
args = parser.parse_args()

TISSUE = args.tejido
tissues = { 'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
            'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

data_ = os.path.join('dataset', 'dataset_completo.h5') #'dataset/dataset_completo.h5'
unet3d_dir = os.path.join('predictions', '3D') #'predictions/3D'
unet2d_dir = os.path.join('predictions', '2D') #'predictions/2D'
dipy_dir = os.path.join('dataset', 'reference_segmentations', 'dipy') #'dataset/reference_segmentations/dipy'
fsl_dir = os.path.join('dataset', 'reference_segmentations', 'fsl') #'dataset/reference_segmentations/fsl'
analysis_folder = os.path.join('predictions', 'plots_and_tables') #'predictions/plots_and_tables'

group = 'etiquetas_'+TISSUE
masks = get_h5_keys(data_, group)
feat = get_h5_keys(data_, 'caracteristicas')

np.random.seed(76)
idx = np.random.choice(len(masks), len(masks), replace=False)[-3:]
#print(feat)

roc__ = []
f1__ = []
jac__ = []
ssim__ = []
DICE__ = []
for index_, i in enumerate(idx):

	#print('masks[i] {}'.format(masks[i]))
	print('feat[i] {}'.format(feat[i]))

	subject = feat[i].split('.')[0]

	mask = get_array(data_, group, masks[i])
	print(np.unique(mask))

	ext = 7 if '.gz' in feat[i] else 4

	dipy_seg = nib.load(os.path.join(dipy_dir, feat[i][:-ext] + '_dipy_'+TISSUE+'.nii')).get_fdata()

	fsl_seg = nib.load(os.path.join(fsl_dir, feat[i][:-ext] + '_fsl_'+TISSUE+'.nii')).get_fdata()

	unet2d_seg = nib.load(os.path.join(unet2d_dir, feat[i], 'unet2D_predictions_'+TISSUE+'.nii')).get_fdata()

	unet3d_seg = nib.load(os.path.join(unet3d_dir, feat[i], 'unet3D_predictions_'+TISSUE+'.nii')).get_fdata()
	#print(np.unique(unet3d_seg))

	'''
	ROC AUC
	'''
	roc_ = []
	roc = roc_auc_score(mask.flatten(), fsl_seg.flatten())
	roc_.append(roc)
	print('FSL roc score: {:.3f}'.format(roc))

	roc = roc_auc_score(mask.flatten(), dipy_seg.flatten())
	roc_.append(roc)
	print('DIPY roc score: {:.3f}'.format(roc))

	roc = roc_auc_score(mask.flatten(), unet2d_seg.flatten())
	roc_.append(roc)
	print('UNET2D roc score: {:.3f}'.format(roc))

	roc = roc_auc_score(mask.flatten(), unet3d_seg.flatten())
	roc_.append(roc)
	roc__.append(roc_)
	print('UNET3D roc score: {:.3f}\n'.format(roc))

	'''
	Jaccard Score
	'''
	jac_ = []
	jac = jaccard_score(mask.flatten(), fsl_seg.flatten())
	jac_.append(jac)
	print('FSL Jaccard score: {:.3f}'.format(jac))

	jac = jaccard_score(mask.flatten(), dipy_seg.flatten())
	jac_.append(jac)
	print('DIPY Jaccard score: {:.3f}'.format(jac))

	jac = jaccard_score(mask.flatten(), unet2d_seg.flatten())
	jac_.append(jac)
	print('UNET2D Jaccard score: {:.3f}'.format(jac))

	jac = jaccard_score(mask.flatten(), unet3d_seg.flatten())
	jac_.append(jac)	
	jac__.append(jac_)
	print('UNET3D Jaccard score: {:.3f}\n'.format(jac))

	'''
	Structural Similarity
	'''
	ssim_ = []
	ssim = metrics.structural_similarity(mask, fsl_seg, data_range=np.max(mask) - np.min(mask))
	ssim_.append(ssim)
	print('FSL structural similarity score: {:.3f}'.format(ssim))

	ssim = metrics.structural_similarity(mask, dipy_seg, data_range=np.max(mask) - np.min(mask))
	ssim_.append(ssim)
	print('DIPY structural similarity score: {:.3f}'.format(ssim))

	ssim = metrics.structural_similarity(mask, unet2d_seg, data_range=np.max(mask) - np.min(mask))
	ssim_.append(ssim)
	print('UNET2D structural similarity score: {:.3f}'.format(ssim))

	ssim = metrics.structural_similarity(mask, unet3d_seg, data_range=np.max(mask) - np.min(mask))
	ssim_.append(ssim)	
	ssim__.append(ssim_)
	print('UNET3D structural similarity score: {:.3f}\n'.format(ssim))

	'''
	Dice Coefficient
	'''
	DICE_ = []
	DICE = f1_score(mask.flatten(), fsl_seg.flatten())
	DICE_.append(DICE)
	print('FSL DICE score: {:.3f}'.format(DICE))

	DICE = f1_score(mask.flatten(), dipy_seg.flatten())
	DICE_.append(DICE)
	print('DIPY DICE score: {:.3f}'.format(DICE))

	DICE = f1_score(mask.flatten(), unet2d_seg.flatten())
	DICE_.append(DICE)
	print('UNET2D DICE score: {:.3f}'.format(DICE))

	DICE = f1_score(mask.flatten(), unet3d_seg.flatten())
	DICE_.append(DICE)	
	DICE__.append(DICE_)
	print('UNET3D DICE score: {:.3f}\n'.format(DICE))

	'''
	Create Pandas' Dataframes, save table and figures, plot figure
	'''
	df_roc = pd.DataFrame(roc__, columns=['FSL', 'Dipy', 'U-Net 2D', 'U-Net 3D']).assign(Métrica='ROC AUC')
	df_jac = pd.DataFrame(jac__, columns=['FSL', 'Dipy', 'U-Net 2D', 'U-Net 3D']).assign(Métrica='Jaccard Score')
	df_ssim = pd.DataFrame(ssim__, columns=['FSL', 'Dipy', 'U-Net 2D', 'U-Net 3D']).assign(Métrica='SSIM')
	df_DICE = pd.DataFrame(DICE__, columns=['FSL', 'Dipy', 'U-Net 2D', 'U-Net 3D']).assign(Métrica='Dice')
		
	cdf = pd.concat([df_roc, df_jac, df_ssim, df_DICE])

	mdf = pd.melt(cdf, id_vars=['Métrica'], var_name=['Técnica'], value_name='Puntaje')

	Path(analysis_folder).mkdir(parents=True, exist_ok=True)

	ax = sns.boxplot(x="Técnica", y="Puntaje", hue="Métrica", data=mdf)
	ax.axvline(x=1.5, color='0.4', linestyle=':')
	ax.axvline(x=0.5, color='0.4', linestyle=':')
	ax.axvline(x=2.5, color='0.4', linestyle=':')

	ax.set_title(TISSUE)

	fig = ax.get_figure()
	savename = os.path.join(analysis_folder, 'boxplot_'+subject+'_'+TISSUE)
	fig.savefig(savename, dpi=300, format='eps')

	cdf = pd.concat([df_roc, df_jac, df_ssim, df_DICE], join='inner').sort_index(level=['0', '1', '2', '3', '4', '5', '6'])
	cdf = cdf.reindex(columns=['Métrica', 'FSL', 'Dipy', 'U-Net 2D', 'U-Net 3D'])
	
	savename = os.path.join(analysis_folder, 'table_'+subject+'_'+TISSUE+'.xlsx')
	cdf.to_excel(savename)

	plt.show()
	plt.clf()
	plt.close()






#subjects = next(os.walk(working_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?
