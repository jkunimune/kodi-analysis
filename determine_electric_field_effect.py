import numpy as np
import matplotlib.pyplot as plt

from electric_field_model import get_analytic_brightness

if __name__ == '__main__':
	q = np.logspace(-3.5, 0.5)
	w = np.empty(q.shape)

	for i in range(q.size):
		r, s = get_analytic_brightness(1, q[i])
		dsdr = -np.gradient(s, r)
		w[i] = 1/dsdr.max()

	plt.plot(q, w)
	plt.plot(q, 14*q, '--')
	plt.xscale('log')
	plt.yscale('log')
	plt.show()
