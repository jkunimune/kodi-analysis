""" some signal utility functions, including the all-important Gelfgat reconstruction """
import numpy as np
import scipy.signal as pysignal
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cmap import GREYS


def linregress(x, y, weights=None):
	""" fit a line to y(x) using least squares. """
	if weights is None:
		weights = np.ones(x.shape)
	weights /= np.sum(weights)
	μx = np.sum(weights*x)
	μy = np.sum(weights*y)
	m = np.sum(weights*(x - μx)*(y - μy))/np.sum(weights*(x - μx)**2)
	b = μy - m*μx
	return m, b


def shape_parameters(x, y, f, contour=0):
	""" get some scalar parameters that describe the shape of this distribution. """
	if contour == 0:
		μ0 = np.sum(f) # image sum
		μx = np.sum(x*f)/μ0 # image centroid
		μy = np.sum(y*f)/μ0
		μxx = np.sum(x**2*f)/μ0 - μx**2 # image rotational inertia
		μxy = np.sum(x*y*f)/μ0 - μx*μy
		μyy = np.sum(y**2*f)/μ0 - μy**2
		eigval, eigvec = np.linalg.eig([[μxx, μxy], [μxy, μyy]])
		i1, i2 = np.argmax(eigval), np.argmin(eigval)
		p0 = np.sqrt(μxx + μyy)
		p1, θ1 = np.hypot(μx, μy), np.arctan2(μy, μx)
		p2, θ2 = np.sqrt(eigval[i1]) - np.sqrt(eigval[i2]), np.arctan2(eigvec[1,i1], eigvec[0,i1])
	else:
		raise NotImplementedError

	return p0, (p1, θ1), (p2, θ2)


def sl(a, b, c):
	""" slice convenience function """
	if (b < -1 and a > 0) or (a < -1 and b > 0): raise ValueError()
	if b == -1 and c < 0: b = None
	return slice(a, b, c)


def convolve2d(a, b, where=None):
	""" full 2d convolution, allowing for masks. more efficient when where is mostly False,
		less efficient otherwise. I don't know which way b is supposed to face, so make it
		symmetric.
	"""
	if where is None:
		where = np.full((a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1), True)
	n, m = a.shape
	c = np.zeros(where.shape)
	for i, j in zip(*np.nonzero(where)):
		nt = max( 0, i - b.shape[0] + 1) # omitted rows on top
		nb = max( 0, n - i - 1) # omitted rows on bottom
		ml = max( 0, j - b.shape[1] + 1) # omitted columns on right
		mr = max( 0, m - j - 1) # omitted rows on left
		c[i,j] = np.sum(a[nt:n-nb, ml:m-mr]*b[sl(i-nt, i-n+nb, -1), sl(j-ml, j-m+mr, -1)])
	return c


def gelfgat_deconvolve2d(F, Q, where=None, illegal=None, verbose=False, show_plots=False):
	""" perform the algorithm outlined in
			Gelfgat V.I. et al.'s "Programs for signal recovery from noisy
			data…" in *Computer Physics Communications* 74 (1993)
		to deconvolve the simple discrete 2d kernel q from the full histogrammed measurement
		F. a background value will be automatically inferred. a mask with the same shape as
		F can be applied; if so, only bins where where[i,j]>0 will be considered valid for
		the deconvolution. pixels in the deconvolved image marked with illegal will be
		coerced to zero.
	"""
	assert len(F.shape) == 2 and F.shape[0] == F.shape[1], "I don't kno how too doo that."
	n = m = F.shape[0] - Q.shape[0] + 1
	# if where is None:
	# 	where = np.full(F.shape, True)
	# else:
	# 	where = where.astype(bool)
	where = np.full(F.shape, True) # TODO I can figure this out. I just need better normalization

	g = np.ones((n, m)) # define the normalized signal matrix
	# if illegal is not None:
	# 	g[illegal] = 0
	g0 = 1 # and the normalized signal background pixel
	G = F.sum()/Q.sum()

	N = F.sum()
	f = F/N # make sure the counts are normalized
	q = Q/Q.sum() # make sure the kernel is normalized

	χ2_red_95 = stats.chi2.ppf(.05, np.sum(where))/np.sum(where)
	iterations, χ2_red = 0, np.inf
	while iterations < 80 and χ2_red > χ2_red_95:
		gsum = g.sum() + g0
		g, g0 = g/gsum, g0/gsum # correct for roundoff
		s = convolve2d(g, q, where=where) + g0/f.size
		dlds = f/s - 1

		δg, δg0 = np.zeros(g.shape), 0 # step direction
		for i, j in zip(*np.nonzero(where)): # we need a for loop for this part because of memory constraints
			nt = max( 0, i - q.shape[0] + 1)
			nb = max( 0, n - i - 1)
			ml = max( 0, j - q.shape[1] + 1)
			mr = max( 0, m - j - 1)
			δg[nt:n-nb, ml:m-mr] += g[nt:n-nb, ml:m-mr] * q[sl(i-nt, i-n+nb, -1), sl(j-ml, j-m+mr, -1)] * dlds[i,j]
		δg0 = g0*np.mean(f/s - 1)

		δs = convolve2d(δg, q, where=where) + δg0/f.size # step projected into measurement space
		dLdh = N*(np.sum(δg**2/g, where=g!=0) + δg0**2/g0)
		d2Ldh2 = -N*np.sum(f*δs**2/s**2)
		assert dLdh > 0 and d2Ldh2 < 0, f"{dLdh} > 0; {d2Ldh2} < 0"

		h = -dLdh/d2Ldh2/2 # compute step length
		assert np.all(g >= 0) and g0 >= 0, g
		if np.amin(g + h*δg) < 0: # if one of the pixels would become negative from this step,
			print(f"a pixel would go negative.")
			print(f"{h} -> {np.amin(-g/δg, where=δg < 0, initial=h)}")
			h = np.amin(-g/δg, where=δg < 0, initial=h) # stop short
		if g0 + h*δg0 < 0: # that applies to the background pixel, as well
			print("background would go negative")
			print(f"{h} -> {-g0/δg0}")
			h = -g0/δg0
		assert h > 0, h

		g += h*δg # step
		g0 += h*δg0
		g = np.maximum(0, g) # sometimes roundoff makes this dip negative. that mustn't be allowed.
		
		χ2_red = N*np.sum((s - f)**2/s, where=where)/np.sum(where) # TODO can I use a better chi squared
		iterations += 1
		if verbose: print("[{}],".format(χ2_red))
		if show_plots:
			fig, axes = plt.subplots(3, 2)
			gs1 = gridspec.GridSpec(4, 4)
			gs1.update(wspace= 0, hspace=0) # set the spacing between axes.
			fig.subplots_adjust(hspace= 0, wspace=0)
			axes[0,0].set_title("Previous step")
			plot = axes[0,0].pcolormesh(G*h/2*δg, cmap='plasma')
			axes[0,0].axis('square')
			fig.colorbar(plot, ax=axes[0,0])
			axes[0,1].set_title("Fit source image")
			plot = axes[0,1].pcolormesh(G*g, vmin= 0, vmax=G*g.max(), cmap='plasma')
			axes[0,1].axis('square')
			fig.colorbar(plot, ax=axes[0,1])
			axes[1,0].set_title("Penumbral image")
			plot = axes[1,0].pcolormesh(F.T, vmin= 0, vmax=F.max(where=where, initial=0), cmap='viridis')
			axes[1,0].axis('square')
			fig.colorbar(plot, ax=axes[1,0])
			axes[1,1].set_title("Fit penumbral image")
			plot = axes[1,1].pcolormesh(N*s.T, vmin= 0, vmax=F.max(where=where, initial=0), cmap='viridis')
			axes[1,1].axis('square')
			fig.colorbar(plot, ax=axes[1,1])
			axes[2,0].set_title("Point spread function")
			plot = axes[2,0].pcolormesh(Q, vmin= 0, vmax=Q.max(), cmap='viridis')
			axes[2,0].axis('square')
			fig.colorbar(plot, ax=axes[2,0])
			axes[2,1].set_title("Chi squared")
			plot = axes[2,1].pcolormesh(np.where(where, N*(s - f)**2/s, 0).T, vmin= 0, vmax=10, cmap='inferno')
			axes[2,1].axis('square')
			fig.colorbar(plot, ax=axes[2,1])
			plt.tight_layout()
			plt.show()
	return G*g, χ2_red


if __name__ == '__main__':
	source = np.array([
		[ 0,  0,  0,  0,  0],
		[ 0,  0,  0, 20, 20],
		[ 0,  0, 40,  0,  0],
		[10, 20,  0,  0,  0],
		[ 0, 10,  0,  0,  0],
	])
	kernel = np.array([
		[ 0,  1,  0],
		[ 1,  1,  1],
		[ 0,  1,  0],
	])
	signal = convolve2d(source, kernel) + 10
	signal = np.random.poisson(signal)

	reconstruction, χ2 = gelfgat_deconvolve2d(signal, kernel, verbose=True)

	plt.figure()
	plt.pcolormesh(source, vmin=0, vmax=source.max())
	plt.colorbar()
	plt.title('source')

	plt.figure()
	plt.pcolormesh(kernel)
	plt.colorbar()
	plt.title('krenel')

	plt.figure()
	plt.pcolormesh(signal)
	plt.colorbar()
	plt.title('signal')

	plt.figure()
	plt.pcolormesh(reconstruction, vmin=0, vmax=source.max())
	plt.colorbar()
	plt.title('reconstruccion')

	plt.show()
