import sys
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from get_subvolumes import get_training_sub_volumes, get_test_subvolumes


if len(sys.argv)>1:
    DATA_PATH_IMG = sys.argv[1]
    DATA_PATH_MASK = sys.argv[2]
    NAME_MASK = sys.argv[3]
    NAME_IMG = sys.argv[4]
    SAVE_PATH = sys.argv[5]
    TISSUE = sys.argv[6]

else:
    DATA_PATH_IMG = "MRBrainS13DataNii/TrainingData" #skullStripped
    DATA_PATH_MASK = "MRBrainS13DataNii/TrainingData" #"MRBrainS13DataNii/TrainingData"
    NAME_MASK = "LabelsForTraining.nii"
    NAME_IMG = "T1.nii"
    SAVE_PATH = ""
    TISSUE = 'GM'


PATH_SUBVOLUME = "train_subvolumes_" + TISSUE
PATH_TESTSUBVOLUME = "test_subvolumes_" + TISSUE

PATH_SUBMASK = "train_submasks_" + TISSUE
PATH_TESTSUBMASK = "test_submasks_" + TISSUE

DATA_ROOT = "/media/henryareiza/Disco_Compartido/david/datasets/"
#DATA_ROOT = "/media/david/datos1/Coding/maestria/trabajo_de_grado/databases/"
DATA_PATH_IMG = os.path.join(DATA_ROOT, DATA_PATH_IMG)
DATA_PATH_MASK = os.path.join(DATA_ROOT, DATA_PATH_MASK)

print(DATA_PATH_IMG)

np.seterr(over='raise')

tissues = {'GM' : 1,
            'BG' : 2,
            'WM' : 3,
            'WML' : 4,
            'CSF' : 5,
            'VEN' : 6,
            'CER' : 7,
            'BSTEM' : 8}

train_ids = next(os.walk(DATA_PATH_IMG))[1] # [2]: files; [1]: directories
print(train_ids)

for mri in range(len(train_ids)):
    
    img_dir = sorted(train_ids)[mri]
    print(img_dir)
    
    img = nib.load(os.path.join(DATA_PATH_IMG, img_dir, NAME_IMG)) #TRAIN_PATH+"/"+img_dir+NAME_IMG)
    img_data = img.get_fdata()

    msk = nib.load(os.path.join(DATA_PATH_MASK, img_dir, NAME_MASK)) #TRAIN_PATH+"/"+img_dir+NAME_MASK)
    img_mask = msk.get_fdata()
    
    '''
    Se crean subvolumenes para las 4 cabezas de entrenamiento
    '''
    SAVE_PATH_SUBVOLUME = os.path.join(DATA_ROOT, SAVE_PATH, PATH_SUBVOLUME, img_dir)
    SAVE_PATH_SUBMASK = os.path.join(DATA_ROOT, SAVE_PATH, PATH_SUBMASK, img_dir)
    SAVE_PATH_TESTSUBVOLUME = os.path.join(DATA_ROOT, SAVE_PATH, PATH_TESTSUBVOLUME, img_dir)
    SAVE_PATH_TESTSUBMASK = os.path.join(DATA_ROOT, SAVE_PATH, PATH_TESTSUBMASK, img_dir)
    
    if mri != 4:
        Path(SAVE_PATH_SUBVOLUME).mkdir(parents=True, exist_ok=True)
        Path(SAVE_PATH_SUBMASK).mkdir(parents=True, exist_ok=True)
        get_training_sub_volumes(img_data, img.affine, img_mask, msk.affine, 
                                SAVE_PATH_SUBVOLUME, SAVE_PATH_SUBMASK, 
                                classes=tissues[TISSUE], 
                                orig_x = 240, orig_y = 240, orig_z = 48, 
                                output_x = 80, output_y = 80, output_z = 16,
                                stride_x = 40, stride_y = 40, stride_z = 8,
                                background_threshold=0.0)

    '''
    Se crean subvolumenes de prueba para las 5 cabezas del dataset
    '''
    Path(SAVE_PATH_TESTSUBVOLUME).mkdir(parents=True, exist_ok=True)
    Path(SAVE_PATH_TESTSUBMASK).mkdir(parents=True, exist_ok=True)
    get_test_subvolumes(img_data, img.affine, img_mask, msk.affine, 
                            SAVE_PATH_TESTSUBVOLUME, SAVE_PATH_TESTSUBMASK,
                            orig_x = 240, orig_y = 240, orig_z = 48, 
                            output_x = 80, output_y = 80, output_z = 16,
                            stride_x = 80, stride_y = 80, stride_z = 16)