from typing import Sequence, Optional

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp


class SparseMatrixBuilder:
	def __init__(self, row_shape: Sequence[int]):
		""" start putting together a large sparse matrix where the rows are uncorrelated and the
		    columns are actually flattened from an n-dimensional structure, along with a vector of
		    the same hight.  I'm not sure if I explaind that well.  just look at how it's used in
		    electric_field.py; I think it's not *too* obleke.
		    I wrote this class because I didn't know the dok_array class existed because the
		    scipy.sparse documentation is confusingly ritten and made me think coo_array and
		    bsr_array were the only nondeprecated ones.
		"""
		self.row_count = 0
		self.row_shape = row_shape
		self.section: Optional[Section] = None
		self.sections: list[Section] = []

	def start_new_section(self, num_rows):
		self.finalize_current_section()
		self.section = Section(num_rows,
		                       np.empty((0, 1 + len(self.row_shape)), dtype=int),
		                       np.empty((0,), dtype=float))

	def finalize_current_section(self):
		if self.section is not None:
			self.section.indices[:, 0] += self.row_count
			self.row_count += self.section.num_rows
			self.sections.append(self.section)
			self.section = None

	def __setitem__(self, indices: Sequence[NDArray[int]], values: NDArray[float] | float):
		indices = [full_index.ravel() for full_index in np.broadcast_arrays(*indices)]
		for i, index in enumerate(indices):
			index[index < 0] += ([self.section.num_rows] + list(self.row_shape))[i]
		if type(values) is float:
			values = np.full(indices[0].size, values) # type: ignore
		else:
			if values.size != indices[0].size: # type: ignore
				raise ValueError("the number of indices ")
			values = values.ravel()
		self.section.indices = np.concatenate([self.section.indices, np.stack(indices, axis=-1)])
		self.section.values = np.concatenate([self.section.values, values])

	def to_coo(self):
		self.finalize_current_section()
		all_values = np.concatenate([section.values for section in self.sections])
		all_indices = np.concatenate([section.indices for section in self.sections])
		row_indices = all_indices[:, 1]
		for i in range(2, all_indices.shape[1]):
			row_indices = row_indices*self.row_shape[i - 1] + all_indices[:, i]
		return sp.csr_array(sp.coo_matrix((all_values, (all_indices[:, 0], row_indices))))


class Section:
	def __init__(self, num_rows: int, indices: NDArray[int], values: NDArray[float]):
		self.num_rows = num_rows
		self.indices = indices
		self.values = values
