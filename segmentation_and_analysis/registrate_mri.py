import os
from ants import image_read, image_write, registration, apply_transforms
#from nilearn.image import resample_img, reorder_img

import argparse
import time
from pathlib import Path


'''
Funcion para registrar la imagen en el nuevo espacio
'''

def ants_registration():

	'''
	Cargar los argumentos
	'''

	parser = argparse.ArgumentParser()

	parser.add_argument("type", choices=['normal', 'segmentada'], help="Tipo de imagen")
	parser.add_argument("--ref_mri", help="Ruta del volumen de referencia")
	parser.add_argument("--reg_mri", choices=['strip', 'seg', 'segTRI', 'segTRI_fill'], help="Volumen a registrar")
	parser.add_argument("--reg_dir", help="Directorio de los volumenes a registrar")
	parser.add_argument("--out_dir", help="Directorio de salida de los volumenes registrados")
	args = parser.parse_args()


	if args.ref_mri:
		ref_path = args.ref_mri
	else:
		ref_path = '/media/henryareiza/Disco_Compartido/david/datasets/MRBrainS13DataNii/TrainingData/2/T1.nii'

	if args.reg_mri:
		reg_mri = '_'+args.reg_mri
	else:
		reg_mri = ''

	if args.reg_dir:
		reg_path = args.reg_dir
	else:
		reg_path = '/media/henryareiza/Disco_Compartido/david/datasets/NITRC-multi-file-downloads/IBSR_nifti_stripped'

	if args.out_dir:
		out_path = args.out_dir
	else:
		out_path = '/media/henryareiza/Disco_Compartido/david/datasets/IBSR_registered'


	'''
	Se itera sobre las subcarpetas del directorio
	'''

	mri_dir = next(os.walk(reg_path))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?
	print(mri_dir)

	ref = image_read(ref_path)#, pixeltype='unsigned char')

	Path(out_path).mkdir(parents=True, exist_ok=True)

	for directory in mri_dir[:]:

		if args.type == 'segmentada':
			img_IBSR = image_read(os.path.join(reg_path, directory, directory+reg_mri+'_ana.nii.gz'))#, pixeltype='unsigned char')
			rs2_reg = registration(fixed=ref, moving=img_IBSR, type_of_transform = 'DenseRigid' )
			rs2 = apply_transforms(fixed=ref, moving=img_IBSR, transformlist=rs2_reg['fwdtransforms'], interpolator='multiLabel')
		else:
			img_IBSR = image_read(os.path.join(reg_path, directory, directory+'_ana'+reg_mri+'.nii.gz'))#, pixeltype='unsigned char')
			rs2_reg = registration(fixed=ref, moving=img_IBSR, type_of_transform = 'DenseRigid' )
			rs2 = apply_transforms(fixed=ref, moving=img_IBSR, transformlist=rs2_reg['fwdtransforms'])
		print(directory+reg_mri+'_ana.nii.gz')
		image_write(rs2, os.path.join(out_path, 'REG_' + directory + '.nii'), ri=False)
		
start_time = time.time()
ants_registration()
print('Tiempo de ejecucion = {}s'.format(time.time() - start_time))
