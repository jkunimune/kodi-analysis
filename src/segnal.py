""" some signal utility functions, including the all-important Gelfgat reconstruction """
import numpy as np
import scipy.signal as pysignal
from scipy.special import comb, erf, factorial
import scipy.stats as stats
import scipy.interpolate as interpolate
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cmap import GREYS


SMOOTHING = 100 # entropy weight


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


def covariance_from_harmonics(p0, p1, θ1, p2, θ2):
	""" convert a circular harmonic representation of a conture to a covariance matrix """
	Σ = np.matmul(np.matmul(
			np.array([[np.cos(θ2), -np.sin(θ2)], [np.sin(θ2), np.cos(θ2)]]),
			np.array([[(p0 + p2)**2, 0], [0, (p0 - p2)**2]])),
			np.array([[np.cos(θ2), np.sin(θ2)], [-np.sin(θ2), np.cos(θ2)]]))
	μ = np.array([p1*np.cos(θ1), p1*np.sin(θ1)])
	return ((Σ + Σ.T)/2, μ)


def harmonics_from_covariance(Σ, μ):
	""" convert a covariance matrix to a circular harmonic representation of its conture """
	try:
		eigval, eigvec = np.linalg.eig(Σ)
	except np.linalg.LinAlgError:
		return np.nan, (np.nan, np.nan), (np.nan, np.nan)
	i1, i2 = np.argmax(eigval), np.argmin(eigval)
	a, b = np.sqrt(eigval[i1]), np.sqrt(eigval[i2])
	p0 = (a + b)/2

	p1, θ1 = np.hypot(μ[0], μ[1]), np.arctan2(μ[1], μ[0])

	p2, θ2 = (a - b)/2, np.arctan2(eigvec[1,i1], eigvec[0,i1])
	return p0, (p1, θ1), (p2, θ2)


def fit_ellipse(x, y, f, contour):
	""" fit an ellipse to the given image, and represent that ellipse as a symmetric matrix """
	assert len(x.shape) == len(y.shape) and len(x.shape) == 1
	X, Y = np.meshgrid(x, y, indexing='ij') # f should be indexd in the ij convencion

	if contour is None:
		μ0 = np.sum(f) # image sum
		if μ0 == 0: return np.full((2, 2), np.nan)
		μx = np.sum(X*f)/μ0 # image centroid
		μy = np.sum(Y*f)/μ0
		μxx = np.sum(X**2*f)/μ0 - μx**2 # image rotational inertia
		μxy = np.sum(X*Y*f)/μ0 - μx*μy
		μyy = np.sum(Y**2*f)/μ0 - μy**2
		return np.array([[μxx, μxy], [μxy, μyy]]), np.array([μx, μy])

	else:
		contour_paths = measure.find_contours(f, contour*f.max())
		if len(contour_paths) == 0:
			return np.full((2,2), np.nan), np.full(2, np.nan)
		contour_path = max(contour_paths, key=len)
		x_contour = np.interp(contour_path[:,0], np.arange(x.size), x)
		y_contour = np.interp(contour_path[:,1], np.arange(y.size), y)
		x0 = np.average(X, weights=f)
		y0 = np.average(Y, weights=f)
		r = np.hypot(x_contour - x0, y_contour - y0)
		θ = np.arctan2(y_contour - y0, x_contour - x0)
		θL, θR = np.concatenate([θ[1:], θ[:1]]), np.concatenate([θ[-1:], θ[:-1]])
		dθ = abs(np.arcsin(np.sin(θL)*np.cos(θR) - np.cos(θL)*np.sin(θR)))/2

		p0 = np.sum(r*dθ)/np.pi/2

		p1x = np.sum(r*np.cos(θ)*dθ)/np.pi + x0
		p1y = np.sum(r*np.sin(θ)*dθ)/np.pi + y0
		p1 = np.hypot(p1x, p1y)
		θ1 = np.arctan2(p1y, p1x)

		p2x = np.sum(r*np.cos(2*θ)*dθ)/np.pi
		p2y = np.sum(r*np.sin(2*θ)*dθ)/np.pi
		p2 = np.hypot(p2x, p2y)
		θ2 = np.arctan2(p2y, p2x)/2

		return covariance_from_harmonics(p0, p1, θ1, p2, θ2)


def shape_parameters(x, y, f, contour=None):
	""" get some scalar parameters that describe the shape of this distribution. """
	return harmonics_from_covariance(*fit_ellipse(x, y, f, contour))


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


def gelfgat_deconvolve2d(F, q, g_inicial=None, where=None, illegal=None, verbose=False, show_plots=False):
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
	n = m = F.shape[0] - q.shape[0] + 1
	if where is None:
		where = np.full(F.shape, True)
	else:
		where = where.astype(bool)

	N = F.sum(where=where)
	f = F/N # make sure the counts are normalized

	if g_inicial is not None:
		assert g_inicial.shape == (n, m), g_inicial.shape
		assert np.all(g_inicial >= 0), g_inicial.min()
		g = g_inicial/np.mean(g_inicial)
	else:
		g = np.ones((n, m)) # define the normalized signal matrix
	η = np.empty(g.shape) # find the total effect of each pixel
	for i in range(n):
		for j in range(m):
			η[i,j] = np.sum(q*where[i:i+q.shape[0], j:j+q.shape[1]])
	if illegal is not None:
		g[illegal] = 0
	g0 = n*m - (np.sum(illegal) if illegal is not None else 0) # and the normalized signal background pixel
	η0 = np.sum(where)

	s = convolve2d(g/η, q, where=where) + g0/η0 # get the starting thing

	np.seterr('ignore')
	L0 = N*np.sum(f*np.log(f), where=where & (f > 0))
	scores, best_G, best_S = [], None, None
	while len(scores) < 200:
		gsum = g.sum() + g0
		g, g0, s = g/gsum, g0/gsum, s/gsum

		dlds = f/s - 1
		δg, δg0 = np.zeros(g.shape), 0 # step direction
		for i, j in zip(*np.nonzero(where)): # we need a for loop for this part because of memory constraints
			nt = max( 0, i - q.shape[0] + 1)
			nb = max( 0, n - i - 1)
			ml = max( 0, j - q.shape[1] + 1)
			mr = max( 0, m - j - 1)
			δg[nt:n-nb, ml:m-mr] += g[nt:n-nb, ml:m-mr] * q[sl(i-nt, i-n+nb, -1), sl(j-ml, j-m+mr, -1)]/η[nt:n-nb, ml:m-mr] * dlds[i,j]
			δg0 += g0 / η0 * dlds[i,j]

		δs = convolve2d(δg/η, q, where=where) + δg0/η0 # step projected into measurement space
		dLdh = N*(np.sum(δg**2/g, where=g!=0) + (δg0**2/g0 if g0!=0 else 0))
		d2Ldh2 = -N*np.sum(f*δs**2/s**2, where=where)
		assert dLdh > 0 and d2Ldh2 < 0, f"{dLdh} > 0; {d2Ldh2} < 0"

		h = -dLdh/d2Ldh2 # compute step length
		assert np.all(g >= 0) and g0 >= 0, g
		if np.amin(g + h*δg) < 0: # if one of the pixels would become negative from this step,
			h = np.amin(-g/δg, where=δg < 0, initial=h) # stop when it hits zero
		if g0 + h*δg0 < 0: # don't let the background pixel even reach zero
			h = -g0/δg0*5/6
		assert h > 0, h

		g += h*δg # step # take the step
		g[abs(g) < 1e-17] = 0 # and immediately correct for roundoff
		g0 += h*δg0
		s += h*δs
		γ = g/η/np.sum(g/η)
		α = np.sqrt(N*1e7)/F.size*SMOOTHING#N/F.size*SMOOTHING # this entropy weiting parameter was determined quasiempirically (it's essentially the strength of my prior)

		L = N*np.sum(f*np.log(s), where=where) # compute likelihood
		S = np.sum(γ*np.log(γ), where=γ!=0) # compute entropy
		assert np.all(np.isfinite(S))
		scores.append(L - α*S) # the score is a combination of both
		if verbose: print(f"[{L - L0}, {S}, {scores[-1] - L0}],")
		if show_plots and len(scores)%10 == 1:
			fig, axes = plt.subplots(3, 2, figsize=(6, 9))
			fig.subplots_adjust(top=0.95, bottom=0.05, hspace=.05, wspace=.05)
			axes[0,0].set_title("Previous step")
			plot = axes[0,0].pcolormesh(N*h/2*δg/η, cmap='plasma')
			axes[0,0].axis('square')
			fig.colorbar(plot, ax=axes[0,0])
			axes[0,1].set_title("Fit source image")
			axes[0,1].set_title(f"Iteration {len(scores)}")
			plot = axes[0,1].pcolormesh(N*g/η, vmin=0, vmax=N*(g/η).max(), cmap=GREYS)
			axes[0,1].axis('square')
			axes[0,1].set_xticks([])
			axes[0,1].set_yticks([])
			fig.colorbar(plot, ax=axes[0,1])
			axes[1,0].set_title("Penumbral image")
			plot = axes[1,0].pcolormesh(np.where(where, F, np.nan).T, vmin=0, vmax=F.max(where=where, initial=0), cmap='viridis')
			axes[1,0].axis('square')
			fig.colorbar(plot, ax=axes[1,0])
			axes[1,1].set_title("Fit penumbral image")
			plot = axes[1,1].pcolormesh(N*s.T, vmin=0, vmax=F.max(where=where, initial=0), cmap='viridis')
			axes[1,1].axis('square')
			axes[1,1].set_xticks([])
			axes[1,1].set_yticks([])
			fig.colorbar(plot, ax=axes[1,1])
			axes[2,0].set_title("Point spread function")
			plot = axes[2,0].pcolormesh(q, vmin=0, vmax=q.max(), cmap='viridis')
			axes[2,0].axis('square')
			fig.colorbar(plot, ax=axes[2,0])
			axes[2,1].set_title("Chi squared")
			plot = axes[2,1].pcolormesh(np.where(where&(s>0), N*s - F*np.log(N*s) + np.log(factorial(F)), 0).T, vmin= 0, vmax=6, cmap='inferno')
			axes[2,1].axis('square')
			fig.colorbar(plot, ax=axes[2,1])
			gs1 = gridspec.GridSpec(4, 4)
			gs1.update(wspace= 0, hspace=0) # set the spacing between axes.
			plt.show()

		if np.argmax(scores) == len(scores) - 1: # keep track of the best we've found
			best_G = N*g/η
			best_S = N*s
		elif np.argmax(scores) < len(scores) - 12: # if the value function decreases six times in a row, quit
			break

	np.seterr('warn')
	return best_G, np.sum((best_S - N*f)**2/best_S, where=where)/np.sum(where)


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
