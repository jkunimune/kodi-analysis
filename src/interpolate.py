from typing import Union

import numpy as np
from numpy.typing import NDArray


class RegularInterpolator:
	def __init__(self, x_min: float, x_max: float, y: NDArray[float]):
		""" create a new object to efficiently linearly interpolate onto a 1D regular grid """
		self.x_min = x_min
		self.x_max = x_max
		self.y = y

	def __call__(self, x: Union[NDArray[float], float]):
		""" if x is out of bounds, this will extrapolate """
		i = (x - self.x_min)/(self.x_max - self.x_min)*(self.y.size - 1)
		i_r = np.maximum(1, np.minimum(self.y.size - 1, np.ceil(i).astype(int)))
		i_l = i_r - 1
		return (i - i_l)/(i_r - i_l)*(self.y[i_r] - self.y[i_l]) + self.y[i_l]
