""" some signal utility functions, including the all-important Gelfgat reconstruction """
import numpy as np
import scipy.signal as pysignal
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
	if b == -1 and c < 0: b = None
	return slice(a, b, c)


def convolve2d(a, b, where=None):
	""" full 2d convolution, allowing for masks. more efficient when where is mostly False,
		less efficient otherwise. I don't know which way b is supposed to face, so make it
		symmetric.
	"""
	if where is None:
		where = np.full((a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
	c = np.zeros(where.shape)
	for i, j in zip(*np.nonzero(where)):
		mt = max( 0, i - b.shape[0] + 1) # omitted rows on top
		mb = max( 0, a.shape[0] - i - 1) # omitted rows on bottom
		mr = max( 0, j - b.shape[1] + 1) # omitted columns on right
		ml = max( 0, a.shape[1] - j - 1) # omitted rows on left
		c[i,j] = np.sum(a[mt:a.shape[0]-mb, mr:a.shape[1]-ml]*b[sl(i-mt, i-a.shape[0]+mb, -1), sl(j-mr, j-a.shape[1]+ml, -1)])
	return c


def gelfgat_deconvolve2d(F, q, D, tolerance, where=None, illegal=None, verbose=False, show_plots=False):
	""" perform the algorithm outlined in
			Gelfgat V.I. et al.'s "Programs for signal recovery from noisy
			data…" in *Computer Physics Communications* 74 (1993)
		to deconvolve the simple discrete 2d kernel q from the full measurement F and
		expected uncertainty D. a mask with the same shape as F can be applied; if so,
		only bins where where[i,j]>0 will be considered valid for the deconvolution. as
		this is an iterative algorithm, a relative tolerance must be provided to
		dictate the stop condition. pixels in the deconvolved image marked with illegal
		will be coerced to zero
	"""
	assert len(F.shape) == 2 and F.shape[0] == F.shape[1], "I don't kno how too doo that."
	n = m = F.shape[0] - q.shape[0] + 1
	if where is None:
		where = np.full(F.shape, True)
	else:
		where = where.astype(bool)
	n_data_bins = np.sum(where)

	g = np.ones((n, m))
	if illegal is not None:
		g[illegal] = 0

	# χ2_95 = stats.chi2.ppf(.95, n_data_bins)
	# while iterations < 50 and χ2 > χ2_95:
	χ2, χ2_prev, iterations = np.inf, np.inf, 0
	while iterations < 1 or ((χ2_prev - χ2)/n_data_bins > tolerance and iterations < 50):
		g /= g.sum() # correct for roundoff
		s = convolve2d(g, q, where=where)
		G = np.sum(F*s/D, where=where)/np.sum(s**2/D, where=where)
		S = G*s
		dLdF = (F - S)/D
		δg = np.zeros(g.shape) # step direction
		for i, j in zip(*np.nonzero(where)): # we need a for loop for this part because of memory constraints
			nt = max( 0, i - q.shape[0] + 1)
			nb = max( 0, n - i - 1)
			ml = max( 0, j - q.shape[1] + 1)
			mr = max( 0, m - j - 1)
			δg[nt:n-nb, ml:m-mr] += g[nt:n-nb, ml:m-mr]*dLdF[i,j]*q[sl(i-nt, i-n+nb, -1), sl(j-ml, j-m+mr, -1)]
		δs = convolve2d(δg, q, where=where) # step projected into measurement space
		Fs, Fδ = np.sum(F*s/D, where=where), np.sum(F*δs/D, where=where)
		Ss, Sδ = np.sum(s**2/D, where=where), np.sum(s*δs/D, where=where)
		Dδ = np.sum(δs**2/D, where=where)
		h = (Fδ - G*Sδ)/(G*Dδ - Fδ*Sδ/Ss)
		g += h/2*δg
		χ2_prev, χ2 = χ2, np.sum((F - S)**2/D, where=where)
		iterations += 1
		if verbose: print("[{}],".format(χ2/n_data_bins))
		if show_plots:
			fig, axes = plt.subplots(3, 2)
			fig.subplots_adjust(hspace= 0, wspace=0)
			gs1 = gridspec.GridSpec(4, 4)
			gs1.update(wspace= 0, hspace=0) # set the spacing between axes.
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
			plot = axes[1,1].pcolormesh(S.T, vmin= 0, vmax=F.max(where=where, initial=0), cmap='viridis')
			axes[1,1].axis('square')
			fig.colorbar(plot, ax=axes[1,1])
			axes[2,0].set_title("Expected variance")
			plot = axes[2,0].pcolormesh(D.T, vmin= 0, vmax=F.max(where=where, initial=0), cmap='viridis')
			axes[2,0].axis('square')
			fig.colorbar(plot, ax=axes[2,0])
			axes[2,1].set_title("Chi squared")
			plot = axes[2,1].pcolormesh(((F - S)**2/D).T, vmin= 0, vmax=10, cmap='inferno')
			axes[2,1].axis('square')
			fig.colorbar(plot, ax=axes[2,1])
			plt.tight_layout()
			plt.show()
	return G*g, χ2


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
	signal = pysignal.convolve2d(source, kernel, mode='full')
	signal = np.random.poisson(signal)

	reconstruction = gelfgat_deconvolve2d(signal, kernel, signal + 1, 1e-3)

	plt.figure()
	plt.pcolormesh(source)
	plt.title('source')
	plt.figure()
	plt.pcolormesh(kernel)
	plt.title('krenel')
	plt.figure()
	plt.pcolormesh(signal)
	plt.title('signal')
	plt.figure()
	plt.pcolormesh(reconstruction)
	plt.title('reconstruccion')
	plt.show()
