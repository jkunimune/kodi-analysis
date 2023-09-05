# convert csv to hdf5

import h5py
import numpy as np
import os
import pandas


def save_as_hdf5(filepath, **kwargs):
	if not filepath.endswith('.h5'):
		filepath += '.h5'
	os.makedirs(os.path.dirname(filepath), exist_ok=True)
	with h5py.File(filepath, 'w') as f:
		for col_name, col_values in kwargs.items():
			if type(col_values) is str or np.size(col_values) == 1:
				f.attrs[col_name] = col_values
			else:
				try:
					f[col_name] = col_values
				except TypeError:
					raise TypeError(f"cannot save ’{col_name}’ object of type {type(col_values)} to HDF5: {col_values}")


def load_hdf5(filepath: str, keys: list[str]):
	if not filepath.endswith('.h5'):
		filepath += '.h5'
	objects = []
	with h5py.File(filepath, 'r') as f:
		for key in keys:
			try:
				objects.append(f.attrs[key])
			except KeyError:
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
