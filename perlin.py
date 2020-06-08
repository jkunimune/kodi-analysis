import numpy as np
import matplotlib.pyplot as plt


def perlin_generator(x_min, x_max, y_min, y_max, wavelength, amplitude):
	""" return random values with the same shapes as x and y """
	node_x = np.arange(x_min, x_max + wavelength, wavelength)
	node_y = np.arange(y_min, y_max + wavelength, wavelength)
	grad_θ = np.random.uniform(0, 2*np.pi, (len(node_x), len(node_y)))
	grad_x, grad_y = np.cos(grad_θ), np.sin(grad_θ)
	def weight(z):
		return np.sin(z*(np.pi/2))**2

	def aplai(x, y):
		i_part = np.interp(x, node_x, np.arange(len(node_x)))
		j_part = np.interp(y, node_y, np.arange(len(node_y)))
		i, j = np.floor(i_part).astype(int), np.floor(j_part).astype(int)
		a, b = weight(i_part - i), weight(j_part - j)
		unnormed = (grad_x[i,j]*(x - node_x[i]) + grad_y[i,j]*(y - node_y[j])) * (1 - a)*(1 - b) +\
			       (grad_x[i,j+1]*(x - node_x[i]) + grad_y[i,j+1]*(y - node_y[j+1])) * (1 - a)*b +\
			       (grad_x[i+1,j]*(x - node_x[i+1]) + grad_y[i+1,j]*(y - node_y[j])) * a*(1 - b) +\
			       (grad_x[i+1,j+1]*(x - node_x[i+1]) + grad_y[i+1,j+1]*(y - node_y[j+1])) * a*b
		return unnormed*amplitude/wavelength

	return aplai


def perlin(x, y, wavelength, amplitude):
	""" return random values with the same shapes as x and y """
	pg = perlin_generator(x.min(), x.max()+1e-15, y.min(), y.max()+1e-15, wavelength, amplitude)
	return pg(x, y)


if __name__ == '__main__':
	X, Y = np.meshgrid(np.linspace(-5, 5, 216), np.linspace(-5, 5, 216))
	Z = perlin(X, Y, np.pi/2)
	plt.contourf(X, Y, Z)
	plt.colorbar()
	plt.show()
