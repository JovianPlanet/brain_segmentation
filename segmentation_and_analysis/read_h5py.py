import argparse
import h5py
from natsort import natsorted

def get_array(file, grupo, index):

	with h5py.File(file, "r") as f:
		#vol_list = natsorted(list(f[args.grupo].keys()))
		#vol=f[grupo+'/'+index]
		vol = f.get(grupo+'/'+index)[:]
	return vol

def get_h5_keys(in_file, grupo):
	with h5py.File(in_file, "r") as f:
		vol_list = list(f[grupo].keys()) #natsorted(list(f[args.grupo].keys()))
	return vol_list
