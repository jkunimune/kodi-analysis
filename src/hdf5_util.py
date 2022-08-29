# convert csv to hdf5

import h5py
import numpy as np
import os
import pandas


def save_as_hdf5(filename, **kwargs):
	if not filename.endswith('.h5'):
		filename += '.h5'
	with h5py.File(filename, 'w') as f:
		for col_name, col_values in kwargs.items():
			col_values = np.array(col_values)
			dataset = f.create_dataset(col_name, col_values.shape)
			dataset[...] = col_values

def load_hdf5(filename: str, keys: list[str]):
	if not filename.endswith('.h5'):
		filename += '.h5'
	objects = []
	with h5py.File(filename, 'r') as f:
		for key in keys:
			objects.append(np.array(f[key]))
	return objects

if __name__ == '__main__':
	# FOLDER = r'C:\Users\Justin Kunimune\Documents\GitHub\MRSt\output'
	# FILE = 'ensemble_4_10_5.0_2_1200_2020-12-24'
	FOLDER = r'C:\Users\Justin Kunimune\Dropbox\Labori\MRSt'
	FILE = 'nora_scan'

	data = pandas.read_csv(os.path.join(FOLDER, FILE+'.csv'))

	# data.to_hdf(os.path.join(FOLDER, FILE+'.h5'), 'data') # mandates an 'optional' dependency that requires a non-pip dependency that doesn't seem to work on Windows

	save_as_hdf5(os.path.join(FOLDER, FILE+'.h5'), **{col: data[col].values for col in data.columns})

	print(f"saved to {os.path.join(FOLDER, FILE+'.h5')}")
