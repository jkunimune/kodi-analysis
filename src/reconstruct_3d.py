# reconstruct_3d.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
		# image, x, y = load_hdf5(f'../results/{line_of_site.shot}-{line_of_site.tim}-hi-reconstruction')
		# [[a, b], [c, d]] = segnal.fit_ellipse(x, y, image)

		basis = tim_coordinates(int(line_of_site['tim'])) # get the absolute direccion of the axes of this image

		cov, _ = segnal.covariance_from_harmonics(
			line_of_site.P0_magnitude,
			0, 0,
			line_of_site.P2_magnitude,
			np.radians(line_of_site.P2_angle)) # get the covariance matrix from the reconstructed image

		for j in range(2):
			for k in range(2):
				coordinate_2d_matrix.append(np.ravel(np.outer(basis[:,j], basis[:,k])))
				covariance_vector.append(cov[j,k])

		dr = [line_of_site.offset_magnitude*np.cos(np.radians(line_of_site.offset_angle)),
		      line_of_site.offset_magnitude*np.sin(np.radians(line_of_site.offset_angle))]

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

	fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
	ax = fig.add_subplot(111, projection='3d')

	# Set of all planical angles:
	u = np.linspace(0, 2*np.pi, 100)
	v = np.linspace(0, np.pi, 100)

	# Cartesian coordinates that correspond to the planical angles:
	# (this is the equacion of an ellipsoid):
	x = np.sqrt(eigval[0]) * np.outer(np.cos(u), np.sin(v))
	y = np.sqrt(eigval[1]) * np.outer(np.sin(u), np.sin(v))
	z = np.sqrt(eigval[2]) * np.outer(np.ones_like(u), np.cos(v))

	x, y, z = np.transpose(np.matmul(eigvec, np.transpose([x, y, z], axes=(1,0,2))), axes=(1,0,2))

	ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='C0', zorder=-1)

	# Adjustment of the axen, so that they all have the same span:
	max_radius = 1.5*np.sqrt(max(eigval))

	for tim in [2, 4, 5]:
		tim_direction = tim_coordinates(tim)[:,2]
		plt.plot([0, max_radius*tim_direction[0]], [0, max_radius*tim_direction[1]], [0, max_radius*tim_direction[2]], f'C{tim}--', label=f"To TIM{tim}")

	for axis in 'xyz':
	    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

	plt.show()

	# reconstructions.append([])
