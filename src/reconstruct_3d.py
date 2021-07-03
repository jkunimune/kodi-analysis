# reconstruct_3d.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import segnal

TIM_LOCATIONS = [
	[np.nan, np.nan],
	[ 37.38, 162.00],
	[np.nan, np.nan],
	[ 63.44, 342.00],
	[100.81, 270.00],
	[np.nan, np.nan]]

images = pd.read_csv('../results/fake-summary.csv', sep=r'\s*,\s*')
images = images[images['energy_cut'] == 'hi'] # for now, we only worry about hot spot images

reconstructions = []

for shot in images['shot'].unique(): # go thru each shot
	print(shot)
	relevant_images = images[images['shot'] == shot]
	number_of_tims = len(relevant_images['tim'].unique())
	if number_of_tims < 3: # make sure we have enuff tims
		print(f"skipping {shot} because it only has {number_of_tims} lines of site")
		continue

	coordinate_matrix = []
	covariance_vector = []
	for i, line_of_site in relevant_images.iterrows(): # on each los we have
		# with open(f'../results/image-{line_of_site.shot}-{line_of_site.tim}-hi.pkl', 'rb') as f:
		# 	x, y, image = pickle.load(f)
		# [[a, b], [c, d]] = segnal.fit_ellipse(x, y, image)

		cov = segnal.covariance_from_harmonics(
			line_of_site.P0_magnitude,
			line_of_site.P2_magnitude,
			line_of_site.P2_angle) # get the covariance matrix from the reconstructed image

		θ_TIM, ɸ_TIM = np.radians(TIM_LOCATIONS[int(line_of_site['tim'])-1])
		basis = np.array([
			[-np.sin(ɸ_TIM), np.cos(ɸ_TIM), 0],
			[-np.cos(θ_TIM)*np.cos(ɸ_TIM), -np.cos(θ_TIM)*np.sin(ɸ_TIM), np.sin(θ_TIM)], # XXX THETA IS THE POLAR ANGLE NOT PHI
			[ np.sin(θ_TIM)*np.cos(ɸ_TIM), np.sin(θ_TIM)*np.sin(ɸ_TIM), np.cos(θ_TIM)],
		]).T # get the absolute direccion of the axes of this image

		for j in range(2):
			for k in range(2):
				coordinate_matrix.append(np.ravel(np.outer(basis[j,:], basis[k,:])))
				covariance_vector.append(cov[j,k])

	ellipsoid_covariances = np.linalg.lstsq(coordinate_matrix, covariance_vector, rcond=None)[0]
	ellipsoid_covariances = ellipsoid_covariances.reshape((3, 3))
	
	eigval, eigvec = np.linalg.eig(ellipsoid_covariances)
	order = np.argsort(eigval)

	for i in order:
		print(f"extends {np.sqrt(eigval[i]):.2f} μm in the direccion ⟨{eigvec[0,i]: .4f}, {eigvec[1,i]: .4f}, {eigvec[2,i]: .4f} ⟩")

	coordinate_matrix = []
	observation_vector = []
	for i, line_of_site in relevant_images.iterrows(): # then do the same thing with the offset vectors
		dx = line_of_site.offset_magnitude*np.cos(np.radians(line_of_site.offset_angle))
		dy = line_of_site.offset_magnitude*np.sin(np.radians(line_of_site.offset_angle))

		θ_TIM, ɸ_TIM = np.radians(TIM_LOCATIONS[int(line_of_site.tim)-1])
		basis = np.array([
			[0, 0, 0],
			[np.sin(θ_TIM-np.pi/2)*np.cos(ɸ_TIM), np.sin(θ_TIM-np.pi/2)*np.sin(ɸ_TIM), np.cos(θ_TIM-np.pi/2)],
			[np.sin(θ_TIM)*np.cos(ɸ_TIM), np.sin(θ_TIM)*np.sin(ɸ_TIM), np.cos(θ_TIM)],
		]).T
		basis[:,0] = np.cross(basis[:,1], basis[:,2]) # get the absolute direccion of the axes of this image

		for j in range(2):
			coordinate_matrix.append(basis[j,:])
			observation_vector.append([dx, dy][j])

	absolute_offset = np.linalg.lstsq(coordinate_matrix, observation_vector, rcond=None)[0]

	print(f"offset by ⟨{absolute_offset[0]: .2f}, {absolute_offset[1]: .2f}, {absolute_offset[2]: .2f}⟩ μm")

	# reconstructions.append([])
