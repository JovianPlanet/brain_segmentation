import nibabel as nib
import numpy as np
import argparse
from sklearn.metrics import jaccard_score, accuracy_score, f1_score
from skimage import metrics

np.set_printoptions(precision=3, suppress=True)

working_dir = '/media/henryareiza/Disco_Compartido/david/datasets/skullStrippedMRBrainS18DataNii/'
unet3d_predictions_dir = 'predictions/3D'
unet2d_predictions_dir = 'predictions/2D'

subjects = next(os.walk(working_dir))[1]
