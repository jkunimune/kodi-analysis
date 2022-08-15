import numpy as np

def perlin_generator(x_min, x_max, y_min, y_max, wavelength, amplitude):
	""" return random values with the same shapes as x and y """
	node_x = np.arange(x_min - wavelength/2, x_max + wavelength, wavelength)
	node_y = np.arange(y_min - wavelength/2, y_max + wavelength, wavelength)
	grad_θ = np.random.uniform(0, 2*np.pi, (len(node_x), len(node_y)))
	grad_x, grad_y = np.cos(grad_θ), np.sin(grad_θ)
	def weight(z):
		return np.sin(z*(np.pi/2))**2

	def aplai(x, y):
		if x.ndim != 2:
			x, y = np.meshgrid(x, y, indexing="ij")
		i_part = np.interp(x, node_x, np.arange(len(node_x)))
		i_part = np.clip(i_part, 0, grad_x.shape[0]-1.001)
		j_part = np.interp(y, node_y, np.arange(len(node_y)))
		j_part = np.clip(j_part, 0, grad_y.shape[1]-1.001)
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


def wave_generator(x_min, x_max, y_min, y_max, wavelength, amplitude, dimensions=1):
	θ_0 = np.random.uniform(0, 2*np.pi)
	ɸ_0 = np.random.uniform(0, 2*np.pi)
	z_0 = np.random.normal((x_min + x_max)/2*np.cos(θ_0) + (y_min + y_max)/2*np.sin(θ_0), wavelength/(2*np.pi))

	def aplai(x, y):
		z = x*np.cos(θ_0) + y*np.sin(θ_0)
		return amplitude*np.sin(2*np.pi*z/wavelength + ɸ_0)/(1 + (np.pi*(z - z_0)/wavelength)**2)

	if dimensions == 1:
		return aplai
	elif dimensions == 2:
		return lambda x,y: aplai(x,y)*np.cos(θ_0), lambda x,y: aplai(x,y)*np.sin(θ_0)


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	r = np.sqrt(np.random.random(1000000))
	t = 2*np.pi*np.random.random(1000000)

	x, y = r*np.cos(t), r*np.sin(t)

	dx, dy = np.zeros(x.shape), np.zeros(y.shape)
	for n in range(0, 3):
		dx += perlin(x, y, 2**(-n), 0.1*2**(-2*n))
		dy += perlin(x, y, 2**(-n), 0.1*2**(-2*n))

	plt.hist2d(x + dx, y + dy, bins=72, range=[[-1.1, 1.1], [-1.1, 1.1]])
	plt.axis('square')
	plt.show()

