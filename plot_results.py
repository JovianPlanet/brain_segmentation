# -*- coding: utf-8 -*-
import os
import sys
import nibabel as nib
import numpy as np
import imageio
import argparse

import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

import gif_your_nifti.core as gif2nif

from segmentation_and_analysis.read_h5py import get_array, get_h5_keys

'''
En este codigo se crean la imagenes para mostrar los resultados de las segmentaciones
'''

np.set_printoptions(precision=3, suppress=True)

def visualize_data_gif(original_, data_, data2, sujeto, tejido):
    images = []
    for i in range(data_.shape[2]):
    	
        x = original_[:, :, min(i, data_.shape[2] - 1)]
        y = data_[:, :, min(i, data_.shape[2] - 1)]
        
        z = data2[:, :, min(i, data_.shape[2] - 1)] #min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)#.astype(np.uint8)) # Se pasa a uint8 para evitar un warning
        
    name = os.path.join(analysis_folder, sujeto+'_'+tejido+'.gif')
    imageio.mimsave(name, images, 'GIF', duration=0.1)
    #imageio.mimsave(analysis_folder+'/'+sujeto+'_'+tejido+'.gif', images, 'GIF', duration=0.1)
    return #Image(filename="/tmp/gif.gif", format='png')

parser = argparse.ArgumentParser()
parser.add_argument("tejido", choices=['GM', 'CSF', 'WM'], 
                    help="Tejido a evaluar")
args = parser.parse_args()

TISSUE = args.tejido
tissues = { 'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
            'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

feats13 = os.path.join('dataset', 'input_datasets', 'MRBrainS13DataNii', 'TrainingData') #'dataset/input_datasets/MRBrainS13DataNii/TrainingData'
feats18 = os.path.join('dataset', 'input_datasets', 'MRBrainS18DataNii', 'training') #'dataset/input_datasets/MRBrainS18DataNii/training'
featsIBSR = os.path.join('dataset', 'input_datasets', 'IBSR_nifti_stripped') #'dataset/input_datasets/IBSR_nifti_stripped'

data_dir = os.path.join('dataset', 'dataset_completo.h5') #'dataset/dataset_completo.h5'

fsl_dir = os.path.join('dataset', 'reference_segmentations', 'fsl') #'dataset/reference_segmentations/fsl'
dipy_dir = os.path.join('dataset', 'reference_segmentations', 'dipy') #'dataset/reference_segmentations/dipy'
unet3D_dir = os.path.join('predictions', '3D') #'predictions/3D'
unet2D_dir = os.path.join('predictions', '2D') #'predictions/2D'

analysis_folder = os.path.join('predictions', 'plots_and_tables') #'predictions/plots_and_tables'

tissues = {'GM': '1', 'WM': '2', 'CSF': '0'}
tech = {'FSL':0, 'DIPY':1, 'UNET3D':2, 'UNET2D':3}

group = 'etiquetas_'+TISSUE
masks = get_h5_keys(data_dir, group)
feat = get_h5_keys(data_dir, 'caracteristicas')

np.random.seed(76)
idx = np.random.choice(len(masks), len(masks), replace=False)[-3:]

print(idx)

for index_, i in enumerate(idx):
    img_data = get_array(data_dir, 'caracteristicas', feat[i])
    msk_data = get_array(data_dir, 'etiquetas_'+args.tejido, masks[i])

    dipy = nib.load(os.path.join(dipy_dir, feat[i].split('.')[0]+'_dipy_'+args.tejido+'.nii'))
    dipy_data = dipy.get_fdata()

    fsl = nib.load(os.path.join(fsl_dir, feat[i].split('.')[0]+'_fsl_'+args.tejido+'.nii'))
    fsl_data = fsl.get_fdata()

    unet2D = nib.load(os.path.join(unet2D_dir, feat[i], 'unet2D_predictions_'+args.tejido+'.nii'))
    unet2D_data = unet2D.get_fdata()

    unet3D = nib.load(os.path.join(unet3D_dir, feat[i], 'unet3D_predictions_'+args.tejido+'.nii'))
    unet3D_data = unet3D.get_fdata()

    print(np.unique(unet3D_data))

    plt.figure(figsize=(10,8)).suptitle(feat[i].split('.')[0], fontsize=20)
    plt.subplot(231).set_title("Brain"), plt.imshow(img_data[:,:,24], cmap = 'gray'),plt.axis("off")
    plt.subplot(232).set_title("Ground Truth"), plt.imshow(msk_data[:,:,24], cmap = 'gray'),plt.axis("off")

    plt.subplot(233).set_title("Dipy"), plt.imshow(msk_data[:,:,24], cmap = 'gray'),plt.axis("off")
    dipy_data[:,:,24][dipy_data[:,:,24]< 0.1] = np.nan 
    plt.imshow(dipy_data[:,:,24], cmap="Spectral",interpolation='none', alpha=0.6),plt.axis("off") #Graficar U-Net2D para esa imagen

    plt.subplot(234).set_title("FSL"), plt.imshow(msk_data[:,:,24], cmap = 'gray'),plt.axis("off")
    fsl_data[:,:,24][fsl_data[:,:,24]< 0.1] = np.nan 
    plt.imshow(fsl_data[:,:,24], cmap="Spectral",interpolation='none', alpha=0.6),plt.axis("off") #Graficar U-Net2D para esa imagen

    plt.subplot(235).set_title("2D"), plt.imshow(msk_data[:,:,24], cmap = 'gray'),plt.axis("off")
    unet2D_data[:,:,24][unet2D_data[:,:,24]< 0.1] = np.nan 
    plt.imshow(unet2D_data[:,:,24], cmap="Spectral",interpolation='none', alpha=0.6),plt.axis("off") #Graficar U-Net2D para esa imagen

    plt.subplot(236).set_title("3D"), plt.imshow(msk_data[:,:,24], cmap = 'gray'),plt.axis("off")
    unet3D_data[:,:,24][unet3D_data[:,:,24]< 0.1] = np.nan 
    plt.imshow(unet3D_data[:,:,24], cmap="Spectral",interpolation='none', alpha=0.6),plt.axis("off") #Graficar U-Net2D para esa imagen

    #fig = ax.get_figure()
    #plt.savefig(analysis_folder+'/projection_seg_MRBrainS'+DATASET+'-'+SUBJECT+'.tiff', dpi=300, format='tiff')

    name = os.path.join(analysis_folder, feat[i].split('.')[0]+'_'+args.tejido+'.pdf')
    plt.savefig(name, #analysis_folder+'/'+feat[i].split('.')[0]+'_'+args.tejido+'.tiff', 
                dpi=300, 
                format='pdf', #'tiff', 
                transparent=True, 
                bbox_inches='tight', 
                pad_inches=0.01)

    plt.show()

    # GIF
    visualize_data_gif(msk_data, 
                       unet2D_data[:,:,:], 
                       unet3D_data[:,:,:],
                       feat[i].split('.')[0],
                       args.tejido)

