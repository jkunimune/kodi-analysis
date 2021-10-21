import logging
import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd
import pandas as pd
import pickle
import scipy.optimize as optimize
import scipy.signal as pysignal
from scipy.special import comb, erf, factorial
from scipy.spatial import Delaunay
import gc
import re
import warnings

import diameter
from hdf5_util import save_as_hdf5
import segnal as mysignal
from fake_srim import get_E_out, get_E_in
from electric_field_model import get_analytic_brightness

warnings.filterwarnings("ignore")


SHOW_RAW_DATA = False
SHOW_CROPD_DATA = False
SHOW_POINT_SPREAD_FUNCCION = False
SHOW_INICIAL_RESIDUALS = False
SHOW_1D_FIT_PROCESS = False
SHOW_2D_FIT_PROCESS = False

MAX_NUM_PIXELS = 1000
EXPECTED_MAGNIFICATION_ACCURACY = 4e-3
EXPECTED_SIGNAL_TO_NOISE = 5
NON_STATISTICAL_NOISE = .0
# SMOOTHING = 1e-3
CONTOUR = .50 #XXX.25 TODO: what is the significance of the 17% contour?


def where_is_the_ocean(x, y, z, title, timeout=None):
	""" solicit the user's help in locating something """
	fig = plt.figure()
	plt.pcolormesh(x, y, z, vmax=np.quantile(z, .999))
	plt.axis('square')
	plt.colorbar()
	plt.title(title)

	center_guess = (None, None)
	def onclick(event):
		center_guess = (event.xdata, event.ydata)
	fig.canvas.mpl_connect('button_press_event', onclick)

	start = time.time()
	while center_guess[0] is None and (timeout is None or time.time() - start < timeout):
		plt.pause(.01)
	plt.close('all')
	if center_guess[0] is not None:
		return center_guess
	else:
		raise TimeoutError

def hessian(f, x, args, epsilon=None):
	def f_red(xp):
		return f(xp, *args)
	H = nd.Hessian(f_red, step=epsilon)
	return H(x)


def simple_penumbra(r, δ, Q, r0, minimum, maximum, e_min=0, e_max=1):
	""" compute the shape of a simple analytic single-apeture penumbral image """
	rB, nB = get_analytic_brightness(r0, Q, e_min=e_min, e_max=e_max) # start by accounting for aperture charging but not source size
	n_pixel = 72
	# if 3*δ >= r.max(): # if 3*source size is bigger than the image radius
	# 	raise ValueError("δ cannot be this big")
	if 3*δ >= r.max()/n_pixel: # if 3*source size is smaller than the image radius but bigger than the pixel size
		r_kernel = np.linspace(-3*δ, 3*δ, int(3*δ/r.max()*n_pixel)*2+1) # make a little kernel
		n_kernel = np.exp(-r_kernel**2/δ**2)
		r_point = np.arange(-3*δ, r.max() + 3*δ, r_kernel[1] - r_kernel[0]) # rebin the existing image to match the kernel spacing
		n_point = np.interp(r_point, rB, nB, right=0)
		assert len(n_point) >= len(n_kernel)
		penumbra = np.convolve(n_point, n_kernel, mode='same') # and convolve
	elif δ >= 0: # if 3*source size is smaller than one pixel and nonnegative
		r_point = np.linspace(0, r.max(), n_pixel) # use a dirac kernel instead of a gaussian
		penumbra = np.interp(r_point, rB, nB, right=0)
	else:
		raise ValueError("δ cannot be negative")
	w = np.interp(r, r_point, penumbra/np.max(penumbra), right=0) # map to the requested r values
	return minimum + (maximum-minimum)*w


def simple_fit(*args, a=1, b=0, c=1, plot=False):
	""" compute how close these data are to this penumbral image """
	if len(args[0]) == 4 and len(args) == 12: # first, parse the parameters
		(x0, y0, δ, background), Q, r0, s0, r_img, X, Y, exp, where, e_min, e_max, config = args
	elif len(args[0]) == 5 and len(args) == 11:
		(x0, y0, δ, background, Q), r0, s0, r_img, X, Y, exp, where, e_min, e_max, config = args
	elif len(args[0]) == 5 and len(args) == 10:
		(x0, y0, δ, background, r0), s0, r_img, X, Y, exp, where, e_min, e_max, config = args
		Q = 0
	else:
		raise ValueError("unsupported set of arguments")
	if Q < 0 or δ <= 0: return float('inf') # and reject impossible ones

	dr = 2*(X[1,0] - X[0,0])
	x_eff = a*(X - x0) + b*(Y - y0)
	y_eff = b*(X - x0) + c*(Y - y0) #TODO: this funccion take a lot of time... can I speed it at all?
	teo = np.zeros(X.shape) # and build up the theoretical image
	for xA, yA in get_relative_aperture_positions(s0, r_img, X.max(), mode=config): # go to each circle
		r_rel = np.hypot(x_eff - xA, y_eff - yA)
		in_penumbra = (r_rel <= r_img + dr)
		antialiasing = np.minimum(1, (r_img + dr - r_rel[in_penumbra])/dr) # apply a beveld edge to make sure it is continuous
		for dx in [-dr/6, dr/6]:
			for dy in [-dr/6, dr/6]:
				r_rel = np.hypot(x_eff - xA - dx, y_eff - yA - dy)
				try:
					teo[in_penumbra] += simple_penumbra(r_rel[in_penumbra], δ, Q, r0, 0, 1, e_min, e_max) # and add in its penumbrum
				except ValueError:
					return np.inf
		teo[in_penumbra] *= antialiasing

	if np.any(np.isnan(teo)):
		return np.inf

	# α = np.sum(teo*exp)*SMOOTHING

	if background is not None: # if the min is specified
		scale = abs(np.sum(exp, where=where & (teo > 0)) - background*np.sum(where & (teo > 0)))/ \
				np.sum(teo, where=where & (teo > 0)) # compute the best gess at the signal scale
	else: # if the max and min are unspecified
		scale, background = mysignal.linregress(teo, exp, 1/(1 + teo), where=where)
		background = max(0, background)
		scale = abs(scale)
	teo = background + scale*teo

	penalty = \
		+ (np.sqrt(a*c - b**2) - 1)**2/(4*EXPECTED_MAGNIFICATION_ACCURACY**2)
		# - α*(2*np.log(δ)) \

	where &= (teo != 0) # from now on, ignore problematic pixels

	if np.any((teo == 0) & (exp != 0)): # if they are truly problematic, quit now
		return np.inf
	elif NON_STATISTICAL_NOISE > 1/6*np.max(exp, where=where, initial=0)**(-1/2):
		α = 1/NON_STATISTICAL_NOISE**2 # use a poisson error model with a gamma-distributed rate
		error = (α + exp)*np.log(α/teo + 1) - α*np.log(α/teo) #- np.log(comb(α + exp - 1, exp))
	else: # but you can just use plain poisson if it's all the same to you
		error = -exp*np.log(teo) + teo #+ np.log(factorial(exp))

	if plot:
		plt.figure()
		plt.pcolormesh(np.where(where, error, 2), vmin=0, vmax=6, cmap='inferno')
		# plt.pcolormesh(np.where(where, (teo - exp)/np.sqrt(teo), 0), cmap='RdBu', vmin=-5, vmax=5)#, norm=CenteredNorm())
		plt.colorbar()
		plt.axis('square')
		plt.title(f"r0 = ({x0:.2f}, {y0:.2f}), δ = {δ:.3f}, Q = {Q:.3f}")
		plt.text(0, 0, f"ɛ = {np.sum(error, where=where) + penalty:.1f} Np")
		plt.show()

	# if aggressive:
	# 	return penalty
	# else:
	return np.sum(error, where=where) + penalty


def minimize_repeated_nelder_mead(fun, x0, args, simplex_size, **kwargs):
	""" it's like scipy.optimize.minimize(method='Nelder-Mead'),
		but it tries harder. """
	if hasattr(x0[0], '__iter__'):
		assert hasattr(x0[1], '__iter__'), "I haven't implemented scans for any combo other than the first two parameters"
		for i in range(2, len(x0)):
			assert not hasattr(x0[i], '__iter__'), "I haven't implemented scans for any combo other than the first two parameters"
		best_x0 = None
		best_fun = np.inf
		for x00_test in x0[0]: # do an inicial scan over the first two variables
			for x01_test in x0[1]:
				x0_test = np.concatenate([[x00_test, x01_test], x0[2:]])
				fun_test = fun(x0_test, *args, plot=False)
				if fun_test < best_fun:
					best_fun = fun_test
					best_x0 = x0_test
		assert best_x0 is not None, x0
		x0 = best_x0
	else:
		for i in range(len(x0)):
			assert not hasattr(x0[i], '__iter__'), "I haven't implemented scans for any combo other than the first two parameters"

	past_bests = []
	while len(past_bests) < 2 or past_bests[-2] - past_bests[-1] >= 1:
		inicial_simplex = [x0]
		for i in range(len(x0)):
			inicial_simplex.append(np.copy(x0))
			inicial_simplex[i+1][i] += simplex_size[i]

		opt = optimize.minimize( # do the 1D fit
			fun=fun,
			x0=[None]*len(x0),
			args=args,
			method='Nelder-Mead',
			options=dict(
				initial_simplex=inicial_simplex,
			),
			**kwargs,
		)
		if not opt.success:
			logging.warning(f"  could not find good fit because {opt.message}")
			opt.x = x0
			return opt

		x0 = opt.x
		past_bests.append(opt.fun)

	return opt


def convex_hull(x, y, N):
	""" return an array of the same shape as z with pixels inside the convex hull
		of the data markd True and those outside markd false.
	"""
	triangulacion = Delaunay(np.transpose([x[N > 0], y[N > 0]]))
	hull_object = triangulacion.find_simplex(np.transpose([x.ravel(), y.ravel()]))
	inside = ~((N == 0) & (hull_object == -1).reshape(N.shape))
	inside[:, 1:] &= inside[:, :-1] # erode hull by one pixel in all directions to ignore edge effects
	inside[:, :-1] &= inside[:, 1:]
	inside[1:, :] &= inside[:-1, :]
	inside[:-1, :] &= inside[1:, :]
	return inside


def get_relative_aperture_positions(spacing, r_img, r_max, mode='hex'):
	""" yield the positions of the individual penumbral images in the array relative
		to the center, in the detector plane
	"""
	if spacing == 0:
		yield (0, 0)
	elif mode == 'hex':
		for i in range(-6, 6):
			dy = i*np.sqrt(3)/2*spacing
			for j in range(-6, 6):
				dx = (2*j + i%2)*spacing/2
				if np.hypot(dx, dy) + r_img <= r_max:
					yield (dx, dy)
	elif mode == 'square':
		for dx in [-spacing/2, spacing/2]:
			for dy in [-spacing/2, spacing/2]:
				yield (dx, dy)
	else:
		raise Exception(f"what in davy jones's locker is a {mode}")


def reconstruct(input_filename, output_filename, rA, sA, L, M, rotation,
	etch_time, object_size, resolution, expansion_factor,
	aperture_configuration, aperture_charge_fitting,
	only_1d=False, verbose=False, ask_for_help=False, show_plots=False):
	""" reconstruct a penumbral KOD image.
		length parameters should be given in cm, and rotation in radians.
		aperture_configuration can be 'hex' or 'square'.
		aperture_charge_fitting can be 'all', 'same', or 'none'
	"""
	assert rotation < 2*np.pi
	
	binsS = 2*object_size/resolution
	r0 = (M + 1)*rA # calculate the penumbra parameters
	s0 = (M + 1)*sA
	δ0 = M*object_size*(1 + resolution/object_size)
	r_img = expansion_factor*r0 + δ0
	if s0 != 0 and r_img > s0/2:
		r_img = s0/2 # make sure the image at which we look is small enough to avoid other penumbrae
		δ0 = r_img - expansion_factor*r0

	if aperture_charge_fitting == 'none':
		Q, dQ = 0, 0
	else:
		Q, dQ = None, None

	if input_filename.endswith('.txt'): # if it is a typical CPSA-derived text file
		mode = 'hist'
		track_list = pd.read_csv(input_filename, sep=r'\s+', header=20, skiprows=[24], encoding='Latin-1', dtype='float32') # load all track coordinates

		x_temp, y_temp = track_list['x(cm)'].copy(), track_list['y(cm)'].copy()
		track_list['x(cm)'] =  np.cos(rotation+np.pi)*x_temp - np.sin(rotation+np.pi)*y_temp # apply any requested rotation, plus 180 flip to deal with inherent flip due to aperture
		track_list['y(cm)'] =  np.sin(rotation+np.pi)*x_temp + np.cos(rotation+np.pi)*y_temp
		hicontrast = (track_list['cn(%)'] < 35) & (track_list['e(%)'] < 15)

		track_list['x(cm)'] -= np.mean(track_list['x(cm)'][hicontrast]) # do your best to center
		track_list['y(cm)'] -= np.mean(track_list['y(cm)'][hicontrast])

		binsI = min(MAX_NUM_PIXELS, int(binsS/δ0*r_img)) # get the image resolution needed to resolve the object
		r_full = max(np.max(track_list['x(cm)']), np.max(track_list['y(cm)']))
		binsC = int(min(2*MAX_NUM_PIXELS, binsI*r_full/r_img))
		xC_bins, yC_bins = np.linspace(-r_full, r_full, binsC+1), np.linspace(-r_full, r_full, binsC+1) # this is the CR39 coordinate system, centered at 0,0
		dxC, dyC = xC_bins[1] - xC_bins[0], yC_bins[1] - yC_bins[0] # get the bin widths
		xC, yC = (xC_bins[:-1] + xC_bins[1:])/2, (yC_bins[:-1] + yC_bins[1:])/2 # change these to bin centers
		XC, YC = np.meshgrid(xC, yC, indexing='ij') # change these to matrices
		# plt.hist2d(track_list['x(cm)'], track_list['y(cm)'], bins=(xC_bins, yC_bins), vmin=0, vmax=6)
		# plt.axis('square')
		# plt.show()
		# return None

	else: # if it is a pickle file, load the histogram directly like a raster image
		mode = 'raster'
		with open(input_filename, 'rb') as f:
			xI_bins, yI_bins, NI = pickle.load(f)
		dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
		xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2
		XI, YI = np.meshgrid(xI, yI, indexing='ij')

		xC_bins, yC_bins, NC = xI_bins, yI_bins, NI
		XC, YC = XI, YI

	image_layers, X_layers, Y_layers = [], [], []

	if mode == 'raster' or np.std(track_list['d(µm)']) == 0: # choose which cuts to use depending on whether this is synthetic or real
		cuts = [('synth', [0, 100])]
	else:
		# cuts = [('all', [0, 100])] # [MeV] (pre-filtering)
		# cuts = [('hi', [9, 100]), ('lo', [0, 6])] # [MeV] (pre-filtering)
		cuts = [('7', [11, 100]), ('6', [10, 11]), ('5', [9, 10]), ('4', [8, 9]), ('3', [6, 8]), ('2', [4, 6]), ('1', [2, 4]), ('0', [0, 2])]
		# cuts = [('lo', [0, 6]), ('hi', [10, 100])] # [MeV] (pre-filtering)
		# cuts = [('all', [0, 100]), ('lo', [0, 6]), ('hi', [10, 100])] # [MeV] (pre-filtering)

	outputs = []
	for cut_name, e_in_bounds in cuts: # iterate over the cuts
		e_out_bounds = get_E_out(1, 2, e_in_bounds, ['Ta'], 16) # convert scattering energies to CR-39 energies TODO: parse filtering specification
		e_in_bounds = get_E_in(1, 2, e_out_bounds, ['Ta'], 16) # convert back to exclude particles that are ranged out
		d_bounds = diameter.D(e_out_bounds, τ=etch_time, a=2, z=1)[::-1] # convert to diameters
		if np.isnan(d_bounds[1]):
			d_bounds[1] = np.inf # and if the bin goes down to zero energy, make sure all large diameters are counted

		if mode == 'hist': # if we still need to tally the histogram

			logging.info(f"d in [{d_bounds[0]:5.2f}, {d_bounds[1]:5.2f}] μm")
			track_x = track_list['x(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
			track_y = track_list['y(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()

			if len(track_x) <= 0:
				logging.info("  No tracks found in this cut.")
				continue

			if aperture_charge_fitting == 'all':
				Q = None

			NC, xC_bins, yC_bins = np.histogram2d( # make a histogram
				track_x, track_y, bins=(xC_bins, yC_bins))
			assert NC.shape == XC.shape

			if ask_for_help:
				try: # ask the user for help finding the center
					x0, y0 = where_is_the_ocean(xC_bins, yC_bins, NC, "Please click on the center of a penumbrum.", timeout=8.64)
				except:
					x0, y0 = (0, 0)
			else:
				x0, y0 = (0, 0)
		else:
			x0, y0 = (0, 0)

		if np.sum(NC > 0) < 4:
			logging.info("  Not enuff tracks found in this cut.")
			continue

		save_as_hdf5(f'{output_filename}-{cut_name}-raw', x=xC_bins, y=yC_bins, z=NC)

		if SHOW_RAW_DATA:
			plt.pcolormesh(xC_bins, yC_bins, NC)
			plt.tight_layout()
			plt.show()

		hullC = convex_hull(XC, YC, NC) # compute the convex hull for future uce

		spacial_scan = np.linspace(-r0, r0, 5)
		gess = [ spacial_scan, spacial_scan, .20*min(δ0, r0), .5*np.mean(NC)]
		step = [        .2*r0,        .2*r0, .16*min(δ0, r0), .3*np.mean(NC)]
		bounds = [(None,None), (None,None), (0,None), (0,None)]
		args = (r0, s0, r_img, XC, YC, NC, hullC, *e_in_bounds, aperture_configuration)
		if Q is None: # decide whether to fit the electrick feeld
			logging.info("  fitting electrick feeld")
			gess.append(.15)
			step.append(.10)
			bounds.append((0, None))
		else: # or not
			logging.info(f"  setting electrick feeld to {Q}")
			args = (Q, *args)

		opt = minimize_repeated_nelder_mead(
			simple_fit,
			x0=gess,
			args=args,
			simplex_size=step
		)

		for i in range(len(step)): # adjust the step value for use in Hessian-estimacion
			for bound in bounds[i]:
				if bound != None:
					step[i] = min(step[i], abs(opt.x[i] - bound)/2)
		hess = hessian( # and then get the hessian
			simple_fit, x=opt.x, args=args,
			epsilon=np.concatenate([[1e-3, 1e-3], .1*np.array(step[2:])]),
		)

		try:
			if Q is None: # and use it to get error bars
				x0, y0, δ, _, Q = opt.x
				dx0, dy0, dδ, _, dQ = np.sqrt(np.diagonal(np.linalg.inv(hess)))
			else:
				x0, y0, δ, _ = opt.x
				dx0, dy0, dδ, _ = np.sqrt(np.diagonal(np.linalg.inv(hess)))
				dQ = 0
		except np.linalg.LinAlgError:
			dx0, dy0, dδ, dQ = 0, 0, 0, 0

		logging.debug(f"  {simple_fit(opt.x, *args, plot=show_plots)}")
		logging.debug(f"  {opt}")

		# x0, y0, δ, Q, dx0, dy0, dδ, dQ = 0, 0, .01, .1, 0, 0, 0, 0

		δ_eff = δ + 4*Q/e_in_bounds[0]

		if mode == 'hist':
			xI_bins, yI_bins = np.linspace(x0 - r_img, x0 + r_img, binsI+1), np.linspace(y0 - r_img, y0 + r_img, binsI+1) # this is the CR39 coordinate system, but encompassing a single superpenumbrum
			dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
			xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2
			XI, YI = np.meshgrid(xI, yI, indexing='ij')
			NI = np.zeros(XI.shape) # and N combines all penumbra on that square
			for dx, dy in get_relative_aperture_positions(s0, r_img, xC_bins.max(), mode=aperture_configuration):
				NI += np.histogram2d(track_x, track_y, bins=(xI_bins + dx, yI_bins + dy))[0] # do that histogram

			if s0 == 0: # also do a radial histogram because that might be useful
				e_min, e_max = e_in_bounds
				track_r = np.hypot(track_x - x0, track_y - y0)
				nI, rI_bins = np.histogram(track_r, bins=np.linspace(0, r_img, 51)) # take the histogram
				rI = (rI_bins[:-1] + rI_bins[1:])/2
				zI = nI/(np.pi*(rI_bins[1:]**2 - rI_bins[:-1]**2)) # normalize it by bin area
				z_test = simple_penumbra(rI, δ, Q, r0, 0, 1, e_min=e_min, e_max=e_max)
				signal, background = mysignal.linregress(z_test, zI, rI/(1 + nI))
				r = np.linspace(0, r_img, 216)
				z_actual = simple_penumbra(r, δ, Q, r0, background, background+signal, e_min=e_min, e_max=e_max)
				z_naive = simple_penumbra(r, δ_eff, 0, r0, background, background+signal, e_min=e_min, e_max=e_max)
				save_as_hdf5(f'{output_filename}-{cut_name}-radial', x1=rI_bins, y1=zI, x2=r, y2=z_actual, x3=r, y3=z_naive)

			del(track_x)
			del(track_y)
			gc.collect()

		save_as_hdf5(f'{output_filename}-{cut_name}-projection', x=xI_bins, y=yI_bins, z=NI)

		if SHOW_CROPD_DATA:
			plt.pcolormesh(xI_bins, yI_bins, NI)
			plt.tight_layout()
			plt.show()

		binsK = xI.size - 2*int(δ0/dxI) # now make the kernel (from here on, it's the same in both modes)
		if binsK%2 == 0: # make sure the kernel is odd
			binsK += 1
		xK_bins, yK_bins = np.linspace(-dxI*binsK/2, dxI*binsK/2, binsK+1), np.linspace(-dyI*binsK/2, dyI*binsK/2, binsK+1)
		dxK, dyK = xK_bins[1] - xK_bins[0], yK_bins[1] - yK_bins[0]
		XK, YK = np.meshgrid((xK_bins[:-1] + xK_bins[1:])/2, (yK_bins[:-1] + yK_bins[1:])/2, indexing='ij') # this is the kernel coordinate system, measured from the center of the umbra

		xS_bins, yS_bins = xI_bins[binsK//2:-(binsK//2)]/M, yI_bins[binsK//2:-(binsK//2)]/M # this is the source system.
		dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
		xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
		XS, YS = np.meshgrid(xS, yS, indexing='ij')

		penumbral_kernel = np.zeros(XK.shape) # build the point spread function
		for dx in [-dxK/3, 0, dxK/3]: # sampling over a few pixels
			for dy in [-dyK/3, 0, dyK/3]:
				penumbral_kernel += simple_penumbra(np.hypot(XK+dx, YK+dy), 0, Q, r0, 0, 1, *e_in_bounds)
		penumbral_kernel = penumbral_kernel/np.sum(penumbral_kernel)

		source_bins = np.hypot(XS - x0/M, YS - y0/M) <= (xS_bins[-1] - xS_bins[0])/2
		reach = pysignal.convolve2d(source_bins, penumbral_kernel, mode='full')
		penumbra_low = .005*np.sum(source_bins)*penumbral_kernel.max()# np.quantile(penumbral_kernel/penumbral_kernel.max(), .05)
		penumbra_hih = .99*np.sum(source_bins)*penumbral_kernel.max()# np.quantile(penumbral_kernel/penumbral_kernel.max(), .70)
		data_bins = (np.hypot(XI - x0, YI - y0) <= r_img) & np.isfinite(NI) & \
				(reach > penumbra_low) & (reach < penumbra_hih) # exclude bins that are NaN and bins that are touched by all or none of the source pixels
		# data_bins = np.full(XI.shape, True)

		try:
			data_bins &= convex_hull(XI, YI, NI) # crop it at the convex hull where counts go to zero
		except MemoryError:
			logging.warning("  could not allocate enough memory to crop data by convex hull; some non-data regions may be getting considered in the analysis.")

		if SHOW_POINT_SPREAD_FUNCCION:
			plt.figure()
			plt.pcolormesh(xK_bins, yK_bins, penumbral_kernel)
			plt.axis('square')
			plt.title("Point spread function")
			plt.figure()
			plt.pcolormesh(xI_bins, yI_bins, np.where(data_bins, reach, np.nan))
			plt.axis('square')
			plt.title("Maximum convolution")
			plt.show()

		logging.info(
			f"  n = {np.sum(NI[data_bins]):.4g}, (x0, y0) = ({x0:.3f}, {y0:.3f}) ± {np.hypot(dx0, dy0):.3f} cm, "+\
			f"δ = {δ/M/1e-4:.2f} ± {dδ/M/1e-4:.2f} μm, Q = {Q:.3f} ± {dQ:.3f} cm*MeV, M = {M:.2f}")
		if δ_eff/δ > 1.1:
			logging.info(f"  Charging artificially increased source size by {(δ_eff - δ)/M/1e-4:.3f} μm (a {δ_eff/δ:.3f}× change!)")

		B, χ2_red = mysignal.gelfgat_deconvolve2d(
			NI,
			penumbral_kernel,
			# g_inicial=np.exp(-((XS - x0/M)**2 + (YS - y0/M)**2)/(δ/M)**2),
			where=data_bins,
			illegal=np.logical_not(source_bins),
			verbose=verbose,
			show_plots=False) # deconvolve!

		# B, χ2_red = np.ones(XS.shape), 0

		logging.info(f"  χ^2/n = {χ2_red}")
		if χ2_red >= 1.5: # throw it away if it looks unreasonable
			logging.info("  Could not find adequate fit")
			continue
		B = np.maximum(0, B) # we know this must be nonnegative

		def func(x, A, mouth):
			return A*(1 + erf((100e-4 - x)/mouth))/2
		real = source_bins
		cx, cy = np.average(XS, weights=B), np.average(YS, weights=B)
		(A, mouth), _ = optimize.curve_fit(func, np.hypot(XS - cx, YS - cy)[real], B[real], p0=(2*np.average(B), 10e-4)) # fit to a circle thing
		logging.debug(f"  XXX {mouth/1e-4:.1f}")

		save_as_hdf5(f'{output_filename}-{cut_name}-reconstruction', x=xS_bins, y=yS_bins, z=B)

		p0, (p1, θ1), (p2, θ2) = mysignal.shape_parameters(
			xS, yS, B, contour=CONTOUR) # compute the three number summary
		logging.info(f"  P0 = {p0/1e-4:.2f} μm")
		logging.info(f"  P2 = {p2/1e-4:.2f} μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2):.1f}°")

		image_layers.append(B/B.max())
		X_layers.append(XS)
		Y_layers.append(YS)

		outputs.append(dict(
			energy_cut=cut_name,
			energy_min=e_in_bounds[0],
			energy_max=e_in_bounds[1],
			x0=x0, dx0=dx0,
			y0=y0, dy0=dy0,
			δ=δ/M/1e-4, dδ=dδ/M/1e-4,
			Q=Q, dQ=dQ,
			M=M, dM=0,
			P0_magnitude=p0/1e-4, dP0_magnitude=p0/1e-4*dδ/δ,
			P2_magnitude=p2/1e-4, P2_angle=np.degrees(θ2),
		))
	return outputs
