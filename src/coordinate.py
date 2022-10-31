# coordinate.py - I assume I will have more coordinate system garbage code to put here soon.
from math import cos, sin

import numpy as np
from numpy.typing import NDArray

TIM_LOCATIONS = {
	1: [ 63.44, 126.00],
	2: [ 37.38, 162.00],
	3: [142.62, 342.00],
	4: [ 63.44, 342.00],
	5: [100.81, 270.00],
	6: [ 63.44, 342.00],
	'x': [90, 0],
	'y': [90, 270],
	'z': [0, 0],
	'xy': [90, 315],
}
# TIM_LOCATIONS = [
# 	None,
# 	[90, 0],
# 	None,
# 	[90, 90],
# 	[1, 0],
# 	None]
for tim in list(TIM_LOCATIONS.keys()):
	TIM_LOCATIONS[str(tim)] = TIM_LOCATIONS[tim]


def tim_coordinates(tim: int | str) -> NDArray[float]:
	return orthogonal_basis(*TIM_LOCATIONS[tim])


def tim_direction(tim: int | str) -> NDArray[float]:
	return tim_coordinates(tim)[:, -1]


def orthogonal_basis(polar_angle: float, azimuthal_angle: float) -> NDArray[float]:
	""" return a 3×3 array whose collums are the i, j, and k unit vectors in the TIM coordinate
		system specified by the given spherical angles in degrees.
		in the TIM coordinate system, k points away from TCC, and j points as close to [0, 0, 1] as possible.
		in the TC coordinate system, i points toward 90-00, j points toward 90-90, and k points toward 00-00.
	"""
	polar_angle, azimuthal_angle = np.radians([polar_angle, azimuthal_angle])

	# x points to the imager’s rite
	i = [-sin(azimuthal_angle),
	      cos(azimuthal_angle),
	      0]

	# y points upward
	j = [-cos(azimuthal_angle)*cos(polar_angle),
	     -sin(azimuthal_angle)*cos(polar_angle),
	                           sin(polar_angle)]

	# z points from TCC to the detector
	k = [ cos(azimuthal_angle)*sin(polar_angle),
	      sin(azimuthal_angle)*sin(polar_angle),
	                           cos(polar_angle)]

	assert abs(np.dot(i, j)) < 1e-15 and abs(np.dot(j, k)) < 1e-15 and abs(np.dot(k, i)) < 1e-15
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


def rotation_matrix(angle: float) -> NDArray[float]:
	""" return a 2×2 array whose collums are orthogonal vectors rotated by the given angle relative
	    to the x and y axes. """
	return np.array([[cos(angle), -sin(angle)],
	                 [sin(angle),  cos(angle)]])


if __name__ == '__main__':
	tim2 = tim_direction(2)
	tim4 = tim_direction(4)
	tim5 = tim_direction(5)
	print(np.degrees(np.arccos(np.sum(tim2*tim4))))
	print(np.degrees(np.arccos(np.sum(tim4*tim5))))
	print(np.degrees(np.arccos(np.sum(tim5*tim2))))
