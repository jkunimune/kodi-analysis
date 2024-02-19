# coordinate.py - varius coordinate system math type stuff
from __future__ import annotations

from math import cos, sin, ceil, hypot
from typing import Sequence, Union

import numpy as np
from numpy import ScalarType
from numpy.typing import NDArray

NAMED_LOS = {
	"TIM1": [ 63.44, 126.00],
	"TIM2": [ 37.38, 162.00],
	"TIM3": [142.62, 342.00],
	"TIM4": [ 63.44, 342.00],
	"TIM5": [100.81, 270.00],
	"TIM6": [ 63.44, 342.00],
	"SRTE": [100.70, 134.27],
	"TPS2": [ 37.38,  90.00],
	"CPS1": [100.81, 126.50],
	"CPS2": [ 37.38,  18.00],
	"MRS":  [119.3412, 308.0208],
	'x':    [90, 0],
	'y':    [90, 270],
	'z':    [0, 0],
	'timx': [90, 0],
	'timy': [90, 270],
	'timz': [0, 0],
	'xy':   [90, 315],
	'yz':   [45, 0],
	'xz':   [45, 270],
	'timxy': [90, 0],
	'timyz': [90, 270],
	'timxz': [0, 0],
}
for name in list(NAMED_LOS.keys()):
	NAMED_LOS[name.lower()] = NAMED_LOS[name]
	NAMED_LOS[name.upper()] = NAMED_LOS[name]


def los_coordinates(name: str) -> NDArray[float]:
	""" return a 3×3 array whose collums are the i, j, and k unit vectors in this LOS’s coordinate system """
	return orthogonal_basis(*NAMED_LOS[name])


def los_direction(name: str) -> NDArray[float]:
	return los_coordinates(name)[:, -1]


def tps_direction(tps: Union[int, str] = "2") -> NDArray[float]:
	return orthogonal_basis(*NAMED_LOS[f"tps{tps}"])[:, -1]


def orthogonal_basis(polar_angle: float, azimuthal_angle: float) -> NDArray[float]:
	""" return a 3×3 array whose collums are the i, j, and k unit vectors in the coordinate system of the LOS specified
		by the given spherical angles in degrees. k points away from TCC, and j points as close to [0, 0, 1] as possible.
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


class Interval:
	def __init__(self, minimum: float, maximum: float):
		""" a pair of numbers that define an interval """
		self.minimum = minimum
		self.maximum = maximum

	def __str__(self) -> str:
		return f"Interval[{self.minimum}, {self.maximum}]"

	def __lt__(self, other: Interval) -> bool:
		if self == other:
			return False
		elif self.minimum >= other.minimum and self.maximum >= other.maximum:
			return False
		elif self.minimum <= other.minimum and self.maximum <= other.maximum:
			return True
		else:
			raise ValueError(f"the comparison of {self} and {other} is ambiguus.")

	def __eq__(self, other: Interval) -> bool:
		return self.minimum == other.minimum and self.maximum == other.maximum

	@property
	def center(self) -> float:
		return (self.minimum + self.maximum)/2

	@property
	def width(self) -> float:
		return self.maximum - self.minimum


class LinSpace:
	def __init__(self, minimum: float, maximum: float, num_bins: int):
		""" a class that efficiently describes a finite set of equally spaced numbers """
		self.minimum = minimum
		self.maximum = maximum
		self.num_bins = num_bins

	def __eq__(self, other: LinSpace) -> bool:
		return self.minimum == other.minimum and \
		       self.maximum == other.maximum and \
		       self.num_bins == other.num_bins

	def __add__(self, shift: float) -> LinSpace:
		return LinSpace(self.minimum + shift, self.maximum + shift, self.num_bins)

	def __mul__(self, factor: float) -> LinSpace:
		if factor < 0:
			return LinSpace(factor*self.maximum, factor*self.minimum, self.num_bins)
		else:
			return LinSpace(factor*self.minimum, factor*self.maximum, self.num_bins)

	def __neg__(self) -> LinSpace:
		return self*(-1)

	@property
	def num_edges(self) -> int:
		return self.num_bins + 1

	@property
	def odd(self) -> bool:
		return self.num_bins%2 == 1

	@property
	def bin_width(self) -> float:
		return self.range/self.num_bins

	@property
	def center(self) -> float:
		return (self.minimum + self.maximum)/2

	@property
	def half_range(self) -> float:
		return self.range/2

	@property
	def range(self) -> float:
		return self.maximum - self.minimum

	def get_edges(self) -> NDArray[float]:
		return np.linspace(self.minimum, self.maximum, self.num_edges)

	def get_bins(self) -> NDArray[float]:
		return np.linspace(self.minimum + self.bin_width/2, self.maximum - self.bin_width/2, self.num_bins)

	def get_index(self, values: Sequence[float]) -> Sequence[float]:
		return (np.array(values) - self.minimum)/self.bin_width

	def get_edge(self, index: int) -> float:
		return self.minimum + index*self.bin_width


class Grid:
	def __init__(self, x: LinSpace, y: LinSpace = None):
		""" a 2D rectangular array of evenly spaced points """
		self.x = x
		self.y = y if y is not None else x

	def __eq__(self, other: Grid) -> bool:
		return self.x == other.x and self.y == other.y

	@classmethod
	def from_edge_array(cls, x_edges: NDArray[float], y_edges: NDArray[float]) -> Grid:
		return Grid(LinSpace(x_edges[0], x_edges[-1], x_edges.size - 1),
		            LinSpace(y_edges[0], y_edges[-1], y_edges.size - 1))

	@classmethod
	def from_bin_array(cls, x_bins: NDArray[float], y_bins: NDArray[float]) -> Grid:
		return Grid(LinSpace(1.5*x_bins[0] - 0.5*x_bins[1], 1.5*x_bins[-1] - 0.5*x_bins[-2], x_bins.size),
		            LinSpace(1.5*y_bins[0] - 0.5*y_bins[1], 1.5*y_bins[-1] - 0.5*y_bins[-2], y_bins.size))

	@classmethod
	def from_resolution(cls, min_radius: float, pixel_width: float, odd: bool) -> Grid:
		num_bins = ceil(min_radius/pixel_width + 1)*2 + (1 if odd else 0)
		return Grid(LinSpace(-pixel_width*num_bins/2, pixel_width*num_bins/2, num_bins))

	@classmethod
	def from_size(cls, radius: float, max_bin_width: float, odd: bool) -> Grid:
		num_bins = ceil(radius/max_bin_width)*2 + (1 if odd else 0)
		return Grid(LinSpace(-radius, radius, num_bins))

	@classmethod
	def from_num_bins(cls, radius: float, num_bins: int) -> Grid:
		return Grid(LinSpace(-radius, radius, num_bins))

	@classmethod
	def from_pixels(cls, num_bins: int, pixel_width: float) -> Grid:
		return Grid(LinSpace(-pixel_width*num_bins/2, pixel_width*num_bins/2, num_bins))

	def shifted(self, dx: float, dy: float) -> Grid:
		return Grid(self.x + dx, self.y + dy)

	def scaled(self, factor: float) -> Grid:
		return Grid(self.x*factor, self.y*factor)

	def flipped_horizontally(self) -> Grid:
		return Grid(-self.x, self.y)

	def flipped_vertically(self) -> Grid:
		return Grid(self.x, -self.y)

	def rotated_180(self) -> Grid:
		return Grid(-self.x, -self.y)

	@property
	def total_area(self) -> float:
		return self.x.range*self.y.range

	@property
	def pixel_area(self) -> float:
		return self.x.bin_width*self.y.bin_width

	@property
	def num_pixels(self) -> int:
		return self.x.num_bins*self.y.num_bins

	@property
	def shape(self) -> tuple[int, int]:
		return self.x.num_bins, self.y.num_bins

	@property
	def extent(self) -> tuple[float, float, float, float]:
		return self.x.minimum, self.x.maximum, self.y.minimum, self.y.maximum

	@property
	def pixel_width(self) -> float:
		if abs(self.x.bin_width - self.y.bin_width)/self.x.bin_width < 1e-5:
			return (self.x.bin_width + self.y.bin_width)/2
		else:
			raise ValueError(f"this isn't a square coordinate system ({self.x.bin_width} != {self.y.bin_width})")

	@property
	def diagonal(self) -> float:
		return hypot(self.x.range, self.y.range)

	def get_pixels(self, sparse=False) -> tuple[NDArray[float], NDArray[float]]:
		return np.meshgrid(self.x.get_bins(), self.y.get_bins(), sparse=sparse, indexing="ij")


class Image:
	def __init__(self, domain: Grid, values: NDArray[ScalarType]):
		""" package a spacial coordinate system with a 2D (or higher) array of values to fill it.
		    the last two dims will be treated as the x and y dims, in that order.
		"""
		if values.ndim < 2:
			raise ValueError("an Image's values must be at least 2D.")
		if domain.shape != values.shape[-2:]:
			raise ValueError(f"this domain {domain.shape} doesn't match these values {values.shape}!")
		self.domain = domain
		self.values = values

	@property
	def x(self) -> LinSpace:
		return self.domain.x

	@property
	def y(self) -> LinSpace:
		return self.domain.y

	@property
	def shape(self) -> tuple[int, ...]:
		return self.values.shape

	@property
	def num_pixels(self) -> int:
		return self.domain.num_pixels

	def __getitem__(self, index: int | tuple[int, ...]) -> Image:
		if self.values.ndim <= 2:
			raise IndexError("this image has no channels and therefore cannot be indexed")
		return Image(self.domain, self.values[index])

	def __add__(self, other: Image) -> Image:
		if self.domain != other.domain:
			raise ValueError("these two images cannot be added because they have different domains")
		return Image(self.domain, self.values + other.values)

	def __sub__(self, other: Image) -> Image:
		if self.domain != other.domain:
			raise ValueError("these two images cannot be subtracted because they have different domains")
		return Image(self.domain, self.values - other.values)

	def __mul__(self, other: Image) -> Image:
		if self.domain != other.domain:
			raise ValueError("these two images cannot be multiplied because they have different domains")
		return Image(self.domain, self.values * other.values)

	def __truediv__(self, other: Image) -> Image:
		if self.domain != other.domain:
			raise ValueError("these two images cannot be divided because they have different domains")
		return Image(self.domain, self.values / other.values)

	def flipped_vertically(self) -> Image:
		return Image(self.domain.flipped_vertically(), self.values[..., :, ::-1])

	def flipped_horizontally(self) -> Image:
		return Image(self.domain.flipped_horizontally(), self.values[..., ::-1, :])

	def rotated_180(self) -> Image:
		return Image(self.domain.rotated_180(), self.values[..., ::-1, ::-1])


if __name__ == '__main__':
	tim2 = los_direction("tim2")
	tim4 = los_direction("tim4")
	tim5 = los_direction("tim5")
	print(np.degrees(np.arccos(np.sum(tim2*tim4))))
	print(np.degrees(np.arccos(np.sum(tim4*tim5))))
	print(np.degrees(np.arccos(np.sum(tim5*tim2))))
