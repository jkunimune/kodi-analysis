import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

n = 144
z = np.linspace(0, 12e-3, n)
r = np.linspace(0, 12e-3, n)
Z, R = np.meshgrid(z, r)

def convolve(A, kernel):
	v = np.zeros(A.shape)
	for k, di in enumerate([-1, 0, 1] if kernel.shape[2] == 3 else [-2, -1, 0, 1, 2]):
		for l, dj in enumerate([-1, 0, 1] if kernel.shape[3] == 3 else [-2, -1, 0, 1, 2]):
			v += kernel[:,:, k, l]*np.roll(np.roll(A, -di, axis=0), -dj, axis=1)
	return v

bound_laplacian = np.zeros((n, n, 3, 3))
even_laplacian = np.zeros((n, n, 3, 3))
odd_laplacian = np.zeros((n, n, 3, 3))
for i in range(n):
	for j in range(n):
		if i == 0 and j == 0:
			bound_laplacian[0,0,:,:] = [[0, 0, 0], [0, -1, 1/3], [0, 2/3, 0]]
		elif j == n-1:
			pass
		elif i == n-1:
			pass
		elif i == 0:
			bound_laplacian[0,j,:,:] = [[0, 0, 0], [1/6, -1, 1/6], [0, 2/3, 0]]
		elif j == 0:
			if r[i] < 2e-3 or r[i] >= 5e-3:
				bound_laplacian[i,0,:,:] = [[0, (r[i-1] + r[i])/(r[i-1] + 6*r[i] + r[i+1]), 0], [0, -1, 4*r[i]/(r[i-1] + 6*r[i] + r[i+1])], [0, (r[i] + r[i+1])/(r[i-1] + 6*r[i] + r[i+1]), 0]]
		elif (i+j)%2 == 0:
			even_laplacian[i,j,:,:] = [[0, (r[i-1] + r[i])/(r[i-1] + 6*r[i] + r[i+1]), 0], [2*r[i]/(r[i-1] + 6*r[i] + r[i+1]), -1, 2*r[i]/(r[i-1] + 6*r[i] + r[i+1])], [0, (r[i] + r[i+1])/(r[i-1] + 6*r[i] + r[i+1]), 0]]
		else:
			odd_laplacian[i,j,:,:] = [[0, (r[i-1] + r[i])/(r[i-1] + 6*r[i] + r[i+1]), 0], [2*r[i]/(r[i-1] + 6*r[i] + r[i+1]), -1, 2*r[i]/(r[i-1] + 6*r[i] + r[i+1])], [0, (r[i] + r[i+1])/(r[i-1] + 6*r[i] + r[i+1]), 0]]

V = np.zeros(Z.shape)
V[(Z == 0) & (R >= 2e-3) & (R < 5e-3)] = 1

if __name__ == '__main__':
	plt.figure()
	plt.contour(Z, R, V, vmin=0, vmax=1)
	plt.xlabel("z (m)")
	plt.ylabel("r (m)")
	plt.pause(.01)
for i in range(10000):
	V += convolve(V, even_laplacian)
	V += convolve(V, odd_laplacian)
	V += convolve(V, bound_laplacian)
	V[0,0] = (2*V[1,0] + V[0,1])/3
	if __name__ == '__main__' and i%100 == 0:
		plt.clf()
		plt.contour(Z, R, V, levels=12, vmin=0, vmax=1)
		plt.axis([0, 10e-3, 0, 10e-3])
		plt.pause(.01)

V_path = np.sum((V[:,1:] + V[:,:-1])*(z[1] - z[0]), axis=1) # path-integrated voltage
Er = np.gradient(V_path, r) # path-integrated electric field
Er[0] = 0
RADIAL_COORDINATE = r # radius in m
RADIAL_DISPLACEMENT = 1/2*Er*(4.21*14)*1e-6 # displacement at 1V in m; just divide by energy in MeV!

if __name__ == '__main__':
	plt.figure()
	plt.plot(r, V_path)
	plt.plot(r, Er)

	plt.show()
