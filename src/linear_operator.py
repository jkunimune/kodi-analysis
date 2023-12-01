""" some classes for treating linear operations in the abstract, as is useful for implementing
    inversion algorithms like Richardson–Lucy.
"""
from __future__ import annotations

from functools import cached_property
from math import inf

import numpy as np
from numpy import transpose, array, ones, ndim, shape, size, ravel, reshape, zeros, concatenate, cumsum, split, where
from numpy.typing import NDArray
from scipy import signal


class LinearOperator:
	""" a generalized superclass for things that can linearly modify a set of numbers. effectively every implementation
	    of this is just a 2D matrix, but most of them take advantage of symmetries and constraints such that they save
	    much memory and computation time.
	"""

	def __matmul__(self, inputs: NDArray[float]):
		""" apply this operator to some inputs """
		raise NotImplementedError

	def __mul__(self, factor: float) -> LinearOperator:
		""" combine this operator with multiplication by a scalar """
		raise NotImplementedError

	def __truediv__(self, factor: float) -> LinearOperator:
		""" combine this operator with division by a scalar """
		return self * (1/factor)

	def transpose(self) -> LinearOperator:
		""" find an operator that transforms from the output space to the input space (note, this is *not* an inverse) """
		raise NotImplementedError

	def normalized(self) -> LinearOperator:
		""" return a version of this where each collum is scaled such that it sums to 1 (so inputs that sum to 1 will always produce outputs that sum to 1) """
		raise NotImplementedError

	def sum(self, axis: int) -> NDArray[float]:
		""" compute the sum of each row (the result of operating on an array of 1s), or of each collum (the detection efficiency of each input) """
		if axis == 1:
			return self @ ones(self.input_size)
		elif axis == 0:
			return self.transpose() @ ones(self.output_size)
		else:
			raise ValueError(f"axis must be either 0 or 1, not {axis}.")

	def all_is_nonnegative(self) -> bool:
		""" whether this will definitely always produce nonnegative outputs given nonnegative inputs """
		raise NotImplementedError

	@cached_property
	def output_size(self) -> int:
		raise NotImplementedError

	@cached_property
	def input_size(self) -> int:
		raise NotImplementedError


class Matrix(LinearOperator):
	def __init__(self, coefficients: NDArray[float]):
		""" it's just a Matrix. it operates on a 1D array and yields a 1D array. """
		if ndim(coefficients) != 2:
			raise ValueError("only 2D arrays may be made into Matrices.")
		self.coefficients = array(coefficients)

	def __matmul__(self, inputs: NDArray[float]) -> NDArray[float]:
		if shape(inputs)[0] != shape(self.coefficients)[1]:
			raise ValueError(f"this input has {shape(inputs)[0]} rows, which is incompatible with the Matrix's {shape(self.coefficients)[1]} columns.")
		return self.coefficients @ inputs

	def __mul__(self, factor: float) -> Matrix:
		return Matrix(self.coefficients*factor)

	def transpose(self) -> Matrix:
		return Matrix(transpose(self.coefficients))

	def normalized(self) -> Matrix:
		return Matrix(self.coefficients/np.sum(self.coefficients, axis=0, keepdims=True))

	def sum(self, axis: int) -> NDArray[float]:
		return np.sum(self.coefficients, axis)

	def all_is_nonnegative(self) -> bool:
		return np.all(self.coefficients >= 0)

	@cached_property
	def output_size(self) -> int:
		return shape(self.coefficients)[0]

	@cached_property
	def input_size(self) -> int:
		return shape(self.coefficients)[1]


class ConvolutionKernel(LinearOperator):
	def __init__(self, kernel: NDArray[float], expanding=True, *,
	             input_scaling: NDArray[float], output_scaling: NDArray[float]):
		""" a linear transformation that can be expressed as a convolution.  specifically, applying this
		    object using the @ operator to an array x is equivalent to evaluating
		        ravel(output_scaling*convolve2d(kernel, input_scaling*reshape(x))),
		    where input_scaling, kernel, and output_scaling, and result are all 2D arrays of different shapes.
		    :param kernel: the titular convolutional kernel
		    :param expanding: True if the convolution should increase the size of the input (i.e. mode="full") and False if it should
		                      decrease the size of the input (i.e. mode="valid")
		    :param input_scaling: an array of factors that get applied elementwise to the values of the input before the convolution
		    :param output_scaling: an array of factors that get applied elementwise to the values of the output after convolution
		"""
		if ndim(input_scaling) != 2 or ndim(kernel) != 2 or ndim(output_scaling) != 2:
			raise ValueError("everything must be 2D for the ConvolutionKernel")
		if expanding:
			desired_input_shape = (shape(output_scaling)[0] - kernel.shape[0] + 1,
			                       output_scaling.shape[1] - kernel.shape[1] + 1)
		else:
			desired_input_shape = (output_scaling.shape[0] + kernel.shape[0] - 1,
			                       output_scaling.shape[1] + kernel.shape[1] - 1)
		if input_scaling.shape != desired_input_shape:
			raise ValueError(f"for an output shape of {output_scaling.shape} and a kernel shape of "
			                 f"{kernel.shape}, the input should be of shape {desired_input_shape}, "
			                 f"but instead I see {input_scaling.shape}.")
		self.input_scaling = input_scaling
		self.kernel = kernel
		self.output_scaling = output_scaling
		self.expanding = expanding

	def __matmul__(self, input_signal: NDArray[float]) -> NDArray[float]:
		if size(input_signal) != self.input_size:
			raise ValueError(f"this convolution is for arrays of size {shape(self.input_scaling)[0]}*{shape(self.input_scaling)[1]}, not {size(input_signal)}")
		return ravel(
			self.output_scaling*signal.fftconvolve(
				self.kernel,
				self.input_scaling*reshape(input_signal, self.input_scaling.shape),
				mode="full" if self.expanding else "valid"))

	def __mul__(self, factor: float) -> ConvolutionKernel:
		return ConvolutionKernel(self.kernel, self.expanding,
		                         input_scaling=self.input_scaling*factor,
		                         output_scaling=self.output_scaling)

	def transpose(self) -> ConvolutionKernel:
		return ConvolutionKernel(kernel=self.kernel[::-1, ::-1], expanding=not self.expanding,
		                         input_scaling=self.output_scaling,
		                         output_scaling=self.input_scaling)

	def normalized(self) -> ConvolutionKernel:
		column_sums = reshape(self.sum(axis=0), self.input_scaling.shape)
		return ConvolutionKernel(kernel=self.kernel, expanding=self.expanding,
		                         input_scaling=self.input_scaling/where(column_sums != 0, column_sums, inf),
		                         output_scaling=self.output_scaling)

	def all_is_nonnegative(self) -> bool:
		return np.all(self.input_scaling >= 0) and \
		       np.all(self.kernel >= 0) and \
		       np.all(self.output_scaling >= 0)

	@cached_property
	def output_size(self) -> int:
		return size(self.output_scaling)

	@cached_property
	def input_size(self) -> int:
		return size(self.input_scaling)


class CompoundLinearOperator(LinearOperator):
	def __init__(self, operators: list[list[LinearOperator]]):
		""" a linear transformation between sets of multiple spaces of different shapes.  operates on a tuple
		    of nD arrays and yields a tuple of nD arrays.  for example, if the first space is a square array
		    and the twoth space is a scalar, then this might represent convolution with a kernel plus a bias.
		    :param operators: the array of LinearOperators that transform data from each input space into
		                      a component of the data in each output space, like a metaarray.
		"""
		self.output_sizes = []
		for i in range(len(operators)):
			self.output_sizes.append(operators[i][0].output_size)
		self.input_sizes = []
		for j in range(len(operators[0])):
			self.input_sizes.append(operators[0][j].input_size)
		self.operators = operators
		self.split_indices = cumsum(self.input_sizes)[:-1]

	def __matmul__(self, input_signal: NDArray[float]) -> NDArray[float]:
		if size(input_signal) != self.input_size:
			raise ValueError(f"this convolution is for arrays of size {'+'.join(str(n) for n in self.input_sizes)}, not {size(input_signal)}")
		# break the input apart into its components
		inputs = split(input_signal, self.split_indices)
		# instantiate the outputs as a list
		outputs = [zeros(output_size) for output_size in self.output_sizes]
		# apply each operator in turn
		for i in range(len(self.output_sizes)):
			for j in range(len(self.input_sizes)):
				outputs[i] += self.operators[i][j]@inputs[j]
		# finally, concatenate them all into one vector
		return concatenate(outputs)

	def __mul__(self, factor: float) -> CompoundLinearOperator:
		return CompoundLinearOperator(
			[[operator*factor for operator in row] for row in self.operators])

	def transpose(self) -> CompoundLinearOperator:
		transposed_operators = []
		for i in range(len(self.input_sizes)):
			transposed_operators.append([])
			for j in range(len(self.output_sizes)):
				transposed_operators[i].append(self.operators[j][i].transpose())
		return CompoundLinearOperator(transposed_operators)

	def normalized(self) -> CompoundLinearOperator:
		return CompoundLinearOperator(
			[[operator.normalized()/len(self.output_sizes) for operator in row] for row in self.operators])

	def all_is_nonnegative(self) -> bool:
		return np.all([[operator.all_is_nonnegative() for operator in row] for row in self.operators])

	@cached_property
	def output_size(self) -> int:
		return sum(self.output_sizes)

	@cached_property
	def input_size(self) -> int:
		return sum(self.input_sizes)
