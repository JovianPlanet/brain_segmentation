# Brain Tissue Segmentation Tool

This is an algorithm that extracts gray matter, white matter and cerebro-spinal fluid from MRI images. It is trained
on images of the MRBrainS13, MRBrainS18 and IBSR datasets using UNET2D and UNET3D algorithms.

### Preparations:

First create a folder in the working directory called 'dataset' and another folder named 'input_datasets' inside it, all the databases should be located inside this folder and should have this structure:

	dataset
		input_datasets
			IBSR_nifti_stripped
			MRBrainS13DataNii
			MRBrainS18DataNii

If your working with the MRBrainS18 dataset you must modify it to have this structure:

	dataset/input_datasets/MRBrainS18DataNii
		training
			070
				reg_T1.nii.gz
				segm.nii.gz
			1
				reg_T1.nii.gz
				segm.nii.gz

			...etc

### Pipeline:

1. Run normalize_datasets.py with either one of the following arguments: ['13', '18', 'IBSR'] according to the dataset
 you want to normalize.

 	e.g.: python3.8 normalize_datasets.py IBSR

2. Run make_h5.py, the possible arguments are: ['caracteristicas', 'etiquetas_CSF', 'etiquetas_GM', 'etiquetas_WM', 'affine']. Run once for each argument. It outputs the 'dataset_completo.h5' file which contains all the images in the dataset with the respective manual segmentations and affine matrix. --in_dir es un argumento opcional para especificar el directorio donde se encuentran las segmentaciones maniales. --in_dir is an optional argument for specifying the input directory where manual segmentation volumes are located, it defaults to 'dataset/pretrain_datasets/features_fsl_strip'

	e.g.: python3.8 make_h5py.py caracteristicas
	python3.8 make_h5py.py etiquetas_CSF --in_dir dataset/pretrain_datasets/MRBrainS13_MASKS
	python3.8 make_h5py.py etiquetas_GM --in_dir dataset/pretrain_datasets/MRBrainS13_MASKS
	python3.8 make_h5py.py etiquetas_WM --in_dir dataset/pretrain_datasets/MRBrainS13_MASKS
	python3.8 make_h5py.py etiquetas_WM --in_dir dataset/pretrain_datasets/MRBrainS18_MASKS
	python3.8 make_h5py.py etiquetas_GM --in_dir dataset/pretrain_datasets/MRBrainS18_MASKS
	python3.8 make_h5py.py etiquetas_CSF --in_dir dataset/pretrain_datasets/MRBrainS18_MASKS
	python3.8 make_h5py.py etiquetas_CSF --in_dir dataset/pretrain_datasets/labels_CSF
	python3.8 make_h5py.py etiquetas_GM --in_dir dataset/pretrain_datasets/labels_GM
	python3.8 make_h5py.py etiquetas_WM --in_dir dataset/pretrain_datasets/labels_WM
	python3.8 make_h5py.py affine

3. Run 'make_subvolumes_h5.py' to create 3D patches of the original MRI, this is for training the UNET3D later. Argument is the path to the input h5 file 

	e.g.: python3.8 make_subvolumes_h5.py dataset/dataset_completo.h5

4. Run either 'train_model_unet2D_h5.py' or 'train_model_unet3D_h5.py' or both to train the model to segment any TISSUE selected as argument in ['GM', 'CSF', 'WM']. It trains the network and outputs best trained model's weights to the 'Pesos/2D' or 'Pesos/3D' folder.

	e.g.: python3.8 train_model_unet3D_h5.py WM
	python3.8 train_model_unet3D_h5.py GM
	python3.8 train_model_unet2D_h5.py WM
	python3.8 train_model_unet2D_h5.py GM

5. Make predictions for each model with images in a test dataset running either 'predict_2D.py' or 'predict_3D.py'

	e.g.: python3.8 predict_2D.py GM

### Optional

Optionally you can evaluate performance of each model by running 'make_reference_segmentations.py' which runs the FSL FAST and Dipy algorithms for automatic head tissue segmentation and then running 'evaluate_segmentations.py' to evaluate the trained models against these.