# a centralized location for information about aperture array arrangements
from math import ceil, sqrt, hypot, pi, inf
from typing import Generator

import numpy as np
from numpy.typing import NDArray

from util import Point

SRTE_APERTURE_RADIUS = 66e-4
SRTE_APERTURE_SPACING = 819.912e-4


ANGULAR_PERIOD = {"single": 0, "square": pi/2, "hex": pi/3, "srte": pi}
Ξ_PERIOD = {"single": inf, "square": 1., "hex": 1/2., "srte": 1/2.}
Υ_PERIOD = {"single": inf, "square": 1., "hex": sqrt(3)/2., "srte": 1.}


def positions(shape: str, spacing: float, transform: NDArray[float],
              r_img: float, r_max: float, x0: float = 0., y0: float = 0.
              ) -> Generator[Point, None, None]:
	""" yield the positions of the individual penumbral images in the array relative
		to the center, in the detector plane
	"""
	# estimate how many images to yield
	if shape == "single":
		yield x0, y0
	else:
		true_spacing = spacing*np.linalg.norm(transform, ord=2)
		n = ceil(r_max/true_spacing)
		for i in range(-n, n + 1):
			for j in range(-n, n + 1):
				if shape == "square":
					dξ, dυ = i, j
				elif shape == "hex":
					dξ, dυ = (2*i + j%2)/2, j*sqrt(3)/2
				elif shape == "srte":
					dξ, dυ = (2*i + j%2)/2, j
				else:
					raise ValueError(f"unrecognized aperture arrangement: {shape!r} (must be "
					                 f"'single', 'square', 'hex' or 'srte')")
				dx, dy = spacing*transform@[dξ, dυ]
				if hypot(dx, dy) + r_img <= r_max:
					yield x0 + dx, y0 + dy
