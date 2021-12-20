# coordinates.py - I assume I will have more coordinate system garbage code to put here soon.

import numpy as np


TIM_LOCATIONS = {
	2: [ 37.38, 162.00],
	4: [ 63.44, 342.00],
	5: [100.81, 270.00],
	'x': [90, 0],
	'y': [90, 270],
	'z': [0, 0],
}
# TIM_LOCATIONS = [
# 	None,
# 	[90, 0],
# 	None,
# 	[90, 90],
# 	[1, 0],
# 	None]


def tim_coordinates(tim):
	return orthogonal_basis(*TIM_LOCATIONS[tim])


def tim_direction(tim):
	return tim_coordinates(tim)[:,-1]


def orthogonal_basis(polar_angle, azimuthal_angle):
	""" return a 3×3 array whose colums are the i, j, and k unit vectors in the TIM coordinate
		system specified by the given sferickal angles in degrees.
		in the TIM coordinate system, k points away from TCC, and j points as close to [0, 0, 1] as possible.
		in the TC coordinate system, i points toward 90-00, j points toward 90-90, and k points toward 00-00.
	"""
	polar_angle, azimuthal_angle = np.radians([polar_angle, azimuthal_angle])

	i = [-np.sin(azimuthal_angle),
		  np.cos(azimuthal_angle),
		  0] # x points to the TIM's rite

	j = [-np.cos(azimuthal_angle)*np.cos(polar_angle),
		 -np.sin(azimuthal_angle)*np.cos(polar_angle),
		                          np.sin(polar_angle)]

	k = [ np.cos(azimuthal_angle)*np.sin(polar_angle),
		  np.sin(azimuthal_angle)*np.sin(polar_angle),
		                          np.cos(polar_angle)]

	assert np.dot(i,j) < 1e-15 and np.dot(j,k) < 1e-15 and np.dot(k,i) < 1e-15
	return np.array([i, j, k]).T


def project(r, polar_angle, azimuthal_angle, basis):
	""" project given spherical coordinates (with angles in degrees) into the
		detector plane x and y, as well as z out of the page.
	"""
	polar_angle, azimuthal_angle = np.radians([polar_angle, azimuthal_angle])
	v = [
		r*np.cos(azimuthal_angle)*np.sin(polar_angle),
		r*np.sin(azimuthal_angle)*np.sin(polar_angle),
		r*np.cos(polar_angle)]
	return np.matmul(basis.T, v)


if __name__ == '__main__':
	print(tim_direction(2))
