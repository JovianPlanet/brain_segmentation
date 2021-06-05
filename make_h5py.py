import argparse
import os
from pathlib import Path
import h5py
import nibabel as nib
from natsort import natsorted


'''
Cargar los argumentos
'''

parser = argparse.ArgumentParser()

parser.add_argument("groupname", 
	choices=['caracteristicas', 'etiquetas_CSF', 
	'etiquetas_GM', 'etiquetas_WM', 'affine'], 
	help="Nombre del grupo a crear")
parser.add_argument("--in_dir", help="Directorio donde se encuentran los volumenes a integrar")
args = parser.parse_args()

out_path = 'dataset/dataset_completo.h5'

if args.in_dir:
	in_path = args.in_dir
else:
	in_path = 'dataset/pretrain_datasets/features_fsl_strip'


mri_dir = next(os.walk(in_path))[2] # [2]: lists files; [1]: lists subdirectories; [0]: ?

#Path(out_path).mkdir(parents=True, exist_ok=True)

with h5py.File(out_path, "a") as f:
	grp = f.require_group(args.groupname)

	for mri in mri_dir: #natsorted(mri_dir):
		print(mri)

		if 'CSF' in args.groupname:
			if 'CSF' in mri:
				img = nib.load(os.path.join(in_path, mri))
			else:
				continue
		elif 'GM' in args.groupname:
			if 'GM' in mri:
				img = nib.load(os.path.join(in_path, mri))
			else:
				continue
		elif 'WM' in args.groupname:
			if 'WM' in mri:
				img = nib.load(os.path.join(in_path, mri))
			else:
				continue
		else:
			img = nib.load(os.path.join(in_path, mri))

		'''
		if args.groupname == 'header':
			if 'affine' not in f[args.groupname]:
				dset = grp.create_dataset('affine', data=img.affine)

			if 'voxel_size' not in f[args.groupname]:
				dset = grp.create_dataset('voxel_size', data=img.header.get_zooms())
			#break
			continue
		'''

		img_data = img.get_fdata()
		print(img_data.dtype)

		'''
		Por cada imagen se crea un dataset, el nombre del dataset es el valor de mri y los datos son los valores de los pixeles
		'''

		if args.groupname == 'affine':
			dset = grp.create_dataset(mri, data=img.affine)
		else:
			dset = grp.create_dataset(mri, data=img_data) 


'''
>>> import h5py
>>> f=h5py.File('h5_test.hdf5', 'r')
>>> r=f['etiquetas']
>>> r
<HDF5 group "/etiquetas" (30 members)>
>>> f
<HDF5 file "h5_test.hdf5" (mode r)>
>>> f.keys()
<KeysViewHDF5 ['caracteristicas', 'etiquetas', 'header']>
>>> r=f['etiquetas/1-LabelsForTraining.nii']
>>> r
<HDF5 dataset "1-LabelsForTraining.nii": shape (240, 240, 48), type "<f8">
>>> r[:]
array([[[0., 0., 0., ..., 0., 0., 0.],

>>> r=f['etiquetas']
>>> r.keys()
>>> r.values()
ValuesViewHDF5(<HDF5 group "/etiquetas" (30 members)>)
>>> lis=r.keys()
>>> lis
>>> lis=list(r.keys())
>>> lis
>>> list(f['caracteristicas'].keys())
'''
