# reconstruct_ellipsoid.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import segnal
from coordinate import tim_coordinates

matplotlib.rc('font', family='serif', size=18)
matplotlib.rc('legend', framealpha=1)
matplotlib.rc('axes', axisbelow=True)

images = pd.read_csv('../images/summary.csv', sep=r'\s*,\s*', engine='python')
images = images[images['energy_cut'] == 'hi'] # for now, we only worry about hot spot images

shot_numbers = []
offset_magnitudes = []
separations = []
prolatenesses = []
oblatenesses = []
errorbars = []

for shot in images['shot'].unique(): # go thru each shot
	print(shot)
	relevant_images = images[images['shot'] == shot]
	number_of_tims = len(relevant_images['tim'].unique())
	if number_of_tims < 3: # make sure we have enuff tims
		print(f"skipping {shot} because it only has {number_of_tims} lines of site")
		continue

	errors = []

	coordinate_2d_matrix = []
	covariance_vector = []
	coordinate_1d_matrix = []
	separation_vector = []

	for i, image in relevant_images.iterrows(): # on each los we have
		# image, x, y = load_hdf5(f'../results/{image.shot}-{image.tim}-hi-reconstruction')
		# [[a, b], [c, d]] = segnal.fit_ellipse(x, y, image)

		basis = tim_coordinates(int(image['tim'])) # get the absolute direccion of the axes of this image

		cov, _ = segnal.covariance_from_harmonics(
			image.P0_magnitude,
			0, 0,
			image.P2_magnitude,
			np.radians(image.P2_angle)) # get the covariance matrix from the reconstructed image

		for j in range(2):
			for k in range(2):
				coordinate_2d_matrix.append(np.ravel(np.outer(basis[:,j], basis[:,k])))
				covariance_vector.append(cov[j,k])

		dr = [image.offset_magnitude*np.cos(np.radians(image.offset_angle)),
		      image.offset_magnitude*np.sin(np.radians(image.offset_angle))]

		for j in range(2):
			coordinate_1d_matrix.append(basis[:,j])
			separation_vector.append(dr[j])

		errors.append(np.maximum(image.dP0_magnitude, 10**2/image.P0_magnitude))

	ellipsoid_covariances = np.linalg.lstsq(coordinate_2d_matrix, covariance_vector, rcond=None)[0]
	ellipsoid_covariances = ellipsoid_covariances.reshape((3, 3))
	
	eigval, eigvec = np.linalg.eig(ellipsoid_covariances)
	order = np.argsort(eigval)

	for i in order:
		print(f"extends {np.sqrt(eigval[i]):.2f} μm in the direccion ⟨{eigvec[0,i]: .4f}, {eigvec[1,i]: .4f}, {eigvec[2,i]: .4f} ⟩")

	absolute_separation = np.linalg.lstsq(coordinate_1d_matrix, separation_vector, rcond=None)[0]

	print(f"separated by ⟨{absolute_separation[0]: .2f}, {absolute_separation[1]: .2f}, {absolute_separation[2]: .2f}⟩ μm")

	if shot.startswith('9552'):
		offset = [[39.9, 80.3, 28.9], [40.3, 101.4, 212.3], [.3, 84, 23], [39.6, 79.7, 29.9], [76.6, 115.5, 151.3]][int(shot)-95520]
	elif shot.startswith('9738'):
		continue#offset = [[80, 40, 0, 80][int(shot)-97385], 63.44, 342.00]
	else:
		continue
	offset_magnitudes.append(offset[0])
	r, θ, ɸ = offset[0], *np.radians(offset[1:])
	x = r*np.cos(ɸ)*np.sin(θ)
	y = r*np.sin(ɸ)*np.sin(θ)
	z = r*np.cos(θ)
	offset = [x, y, z]

	shot_numbers.append(shot)
	errorbars.append(np.hypot(errors[0], errors[-1]))

	fig = plt.figure(figsize=(5, 5))  # Square figure
	ax = fig.add_subplot(111, projection='3d')
	ax.set_box_aspect([1,1,1])

	# # Set of all planical angles:
	# u = np.linspace(0, 2*np.pi, 100)
	# v = np.linspace(0, np.pi, 100)

	# # Cartesian coordinates that correspond to the planical angles:
	# # (this is the equacion of an ellipsoid):
	# x = np.sqrt(eigval[0]) * np.outer(np.cos(u), np.sin(v))
	# y = np.sqrt(eigval[1]) * np.outer(np.sin(u), np.sin(v))
	# z = np.sqrt(eigval[2]) * np.outer(np.ones_like(u), np.cos(v))
	# x, y, z = np.transpose(np.matmul(eigvec, np.transpose([x, y, z], axes=(1,0,2))), axes=(1,0,2))
	# ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='C0', zorder=-1)

	for i in range(3):
		ax.plot(*[[-np.sqrt(eigval[i])*eigvec[j,i], np.sqrt(eigval[i])*eigvec[j,i]] for j in range(3)], color='k')
	ax.plot(*[[0, absolute_separation[i]] for i in range(3)], color='C0')

	ax.plot(*[[0, offset[i]] for i in range(3)], 'C1')

	# Adjustment of the axen, so that they all have the same span:
	max_radius = 1.5*np.sqrt(max(eigval))
	ax.set_xlim(-1.0*max_radius, 1.0*max_radius)
	ax.set_ylim(-1.0*max_radius, 1.0*max_radius)
	ax.set_zlim(-1.0*max_radius, 1.0*max_radius)

	for tim in [2, 4, 5]:
		tim_direction = tim_coordinates(tim)[:,2]
		plt.plot([0, max_radius*tim_direction[0]], [0, max_radius*tim_direction[1]], [0, max_radius*tim_direction[2]], f'C{tim}--', label=f"To TIM{tim}")

	plt.title(shot)

	separation_in_offset_disha = np.dot(absolute_separation, offset)/np.sqrt(np.dot(offset, offset))
	separations.append([
		-separation_in_offset_disha,
		np.sqrt(np.dot(absolute_separation, absolute_separation) - separation_in_offset_disha**2)
	])

	prolateness = (np.sqrt(eigval[order[2]]) - np.mean(np.sqrt(eigval)))*eigvec[:,order[2]]
	prolateness_in_offset_disha = np.dot(prolateness, offset)/np.sqrt(np.dot(offset, offset))
	prolatenesses.append([
		abs(prolateness_in_offset_disha),
		np.sqrt(np.dot(prolateness, prolateness) - prolateness_in_offset_disha**2)
	])

	oblateness = (np.mean(np.sqrt(eigval)) - np.sqrt(eigval[order[0]]))*eigvec[:,order[0]]
	oblateness_in_offset_disha = np.dot(oblateness, offset)/np.sqrt(np.dot(offset, offset))
	oblatenesses.append([
		abs(oblateness_in_offset_disha),
		np.sqrt(np.dot(oblateness, oblateness) - oblateness_in_offset_disha**2)
	])
	print(f"the semimajor axis is {np.sqrt(eigval.max()):.3f} μm")
	print(f"the mean radius is {np.mean(np.sqrt(eigval)):.3f} μm")
	print(f"the semiminor axis is {np.sqrt(eigval.min()):.3f} μm")
	print(f"the prolateness is {np.hypot(*prolatenesses[-1]):.3f} μm")
	print(f"the oblateness is {np.hypot(*oblatenesses[-1]):.3f} μm")

plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'serif', 'font.size': 16})

separations = np.array(separations)
prolatenesses = np.array(prolatenesses)
oblatenesses = np.array(oblatenesses)

plt.figure(figsize=(6,5))
plt.grid()
for indices, marker, color, label in [([2], 'o', '#f2ab23', "No offset"), ([0,1,3], 'D', '#118acd', "40 μm offset"), ([4], 'v', '#79098c', "80 μm offset")]:
	plt.scatter(separations[indices,1], separations[indices,0], c=[color]*len(indices), marker=marker, zorder=100, label=label, s=60)
plt.legend(loc="upper right")
for i in range(len(separations)):
	plt.plot([0, separations[i,1]], [0, separations[i,0]], 'k-', linewidth=1)
plt.axis('equal')
plt.yticks([-10, 0, 10, 20])
plt.xlabel("Separation perpendicular to offset (μm)")
plt.ylabel("Separation along offset (μm)")
plt.tight_layout()
# plt.xlim(0, 50)

plt.figure()
plt.grid()
plt.scatter(x=offset_magnitudes, y=separations[:,0], s=70, color='C2', marker='o', zorder=100, label=label)
plt.errorbar(x=offset_magnitudes, y=separations[:,0], yerr=errorbars, color='C2', linestyle='', zorder=100, label=label)
for i in range(len(shot_numbers)):
	if i < 3:
		plt.text(offset_magnitudes[i], separations[i,0], f" {shot_numbers[i]}", horizontalalignment="left")
	else:
		plt.text(offset_magnitudes[i], separations[i,0], f"{shot_numbers[i]} ", horizontalalignment="right")
plt.axis('equal')
# plt.yticks([-10, 0, 10, 20])
plt.xlabel("Imposed offset (μm)")
plt.ylabel("Separation along offset (μm)")
plt.tight_layout()
plt.savefig("scatter-p1.png", dpi=150)
plt.savefig("scatter-p1.eps")
# plt.xlim(0, 50)

# plt.figure(figsize=(6,5))
# plt.grid()
# for indices, marker, color, label in [([2], 'o', '#f2ab23', "No offset"), ([0,1,3], 'D', '#118acd', "40 μm offset"), ([4], 'v', '#79098c', "80 μm offset")]:
# 	plt.scatter(prolatenesses[indices,1], prolatenesses[indices,0], c=[color]*len(indices), marker=marker, zorder=100, label=label, s=60)
# plt.legend(loc="upper right")
# for i in range(len(prolatenesses)):
# 	plt.plot([0, prolatenesses[i,1]], [0, prolatenesses[i,0]], 'k-', linewidth=1)
# plt.axis('equal')
# plt.xlabel("Prolateness perpendicular to offset (μm)")
# plt.ylabel("Prolateness along offset (μm)")
# plt.tight_layout()
# # plt.xlim(0, 80)

plt.figure(figsize=(6,5))
plt.grid()
for indices, marker, color, label in [([2], 'o', '#f2ab23', "No offset"), ([0,1,3], 'D', '#118acd', "40 μm offset"), ([4], 'v', '#79098c', "80 μm offset")]:
	plt.scatter(oblatenesses[indices,1], oblatenesses[indices,0], c=[color]*len(indices), marker=marker, zorder=100, label=label, s=60)
plt.legend(loc="upper right")
for i in range(len(oblatenesses)):
	plt.plot([0, oblatenesses[i,1]], [0, oblatenesses[i,0]], 'k-', linewidth=1)
plt.axis('equal')
plt.xlabel("Oblateness perpendicular to offset (μm)")
plt.ylabel("Oblateness along offset (μm)")
plt.tight_layout()
# plt.xlim(0, 80)

plt.show()
