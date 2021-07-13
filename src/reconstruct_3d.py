# reconstruct_3d.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import segnal
from coordinate import tim_coordinates

images = pd.read_csv('../results/summary.csv', sep=r'\s*,\s*', engine='python')
images = images[images['energy_cut'] == 'hi'] # for now, we only worry about hot spot images

reconstructions = []

for shot in images['shot'].unique(): # go thru each shot
	print(shot)
	relevant_images = images[images['shot'] == shot]
	number_of_tims = len(relevant_images['tim'].unique())
	if number_of_tims < 3: # make sure we have enuff tims
		print(f"skipping {shot} because it only has {number_of_tims} lines of site")
		continue

	coordinate_2d_matrix = []
	covariance_vector = []
	coordinate_1d_matrix = []
	offset_vector = []

	for i, line_of_site in relevant_images.iterrows(): # on each los we have
		# with open(f'../results/image-{line_of_site.shot}-{line_of_site.tim}-hi.pkl', 'rb') as f:
		# 	x, y, image = pickle.load(f)
		# [[a, b], [c, d]] = segnal.fit_ellipse(x, y, image)

		basis = tim_coordinates(int(line_of_site['tim'])) # get the absolute direccion of the axes of this image

		cov, _ = segnal.covariance_from_harmonics(
			line_of_site.P0_magnitude,
			0, 0,
			line_of_site.P2_magnitude,
			line_of_site.P2_angle) # get the covariance matrix from the reconstructed image

		for j in range(2):
			for k in range(2):
				coordinate_2d_matrix.append(np.ravel(np.outer(basis[:,j], basis[:,k])))
				covariance_vector.append(cov[j,k])

		dr = [line_of_site.offset_magnitude*np.cos(line_of_site.offset_angle),
		      line_of_site.offset_magnitude*np.sin(line_of_site.offset_angle)]

		for j in range(2):
			coordinate_1d_matrix.append(basis[:,j])
			offset_vector.append(dr[j])

	ellipsoid_covariances = np.linalg.lstsq(coordinate_2d_matrix, covariance_vector, rcond=None)[0]
	ellipsoid_covariances = ellipsoid_covariances.reshape((3, 3))
	
	eigval, eigvec = np.linalg.eig(ellipsoid_covariances)
	order = np.argsort(eigval)

	for i in order:
		print(f"extends {np.sqrt(eigval[i]):.2f} μm in the direccion ⟨{eigvec[0,i]: .4f}, {eigvec[1,i]: .4f}, {eigvec[2,i]: .4f} ⟩")

	absolute_offset = np.linalg.lstsq(coordinate_1d_matrix, offset_vector, rcond=None)[0]

	print(f"offset by ⟨{absolute_offset[0]: .2f}, {absolute_offset[1]: .2f}, {absolute_offset[2]: .2f}⟩ μm")

	# reconstructions.append([])
