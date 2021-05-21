import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import CenteredNorm, ListedColormap, LinearSegmentedColormap, LogNorm
import numdifftools as nd
import pandas as pd
import pickle
import os
import scipy.optimize as optimize
import scipy.signal as pysignal
from scipy.special import comb, erf, factorial
from scipy.spatial import Delaunay
import gc
import re
import time
import warnings

import diameter
from hdf5_util import save_as_hdf5
import segnal as mysignal
from fake_srim import get_E_out, get_E_in
from electric_field_model import get_analytic_brightness
from cmap import REDS, GREENS, BLUES, VIOLETS, GREYS, COFFEE

warnings.filterwarnings("ignore")
np.seterr('ignore')
plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'serif', 'font.size': 16})


e_in_bounds = 2

SKIP_RECONSTRUCTION = False
SHOW_PLOTS = True
SHOW_RAW_PLOTS = False
SHOW_DEBUG_PLOTS = False
SHOW_OFFSET = False
VERBOSE = False
ASK_FOR_HELP = False
OBJECT_SIZE = 250e-4 # cm
APERTURE_CHARGE_FITTING = 'all'#'same'

NON_STATISTICAL_NOISE = .10
SPREAD = 1.10
EXPECTED_MAGNIFICATION_ACCURACY = 4e-3
RESOLUTION = 60
APERTURE_CONFIGURACION = 'hex'

CONTOUR = .5

FOLDER = 'scans/'
SHOT = 'Shot number'
TIM = 'TIM'
APERTURE_RADIUS = 'Aperture Radius'
APERTURE_SPACING = 'Aperture Separation'
APERTURE_DISTANCE = 'L1'
MAGNIFICATION = 'Magnification'
ROTATION = 'Rotation'
ETCH_TIME = 'Etch time'
R_OFFSET = 'Offset (um)'
Θ_OFFSET = 'Offset theta (deg)'
Φ_OFFSET = 'Offset phi (deg)'
R_FLOW = 'Flow (km/s)'
Θ_FLOW = 'Flow theta (deg)'
Φ_FLOW = 'Flow phi (deg)'

TIM_LOCATIONS = [
	[np.nan,np.nan],
	[ 37.38, 162.00],
	[np.nan,np.nan],
	[ 63.44, 342.00],
	[100.81, 270.00],
	[np.nan,np.nan]]


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
	plt.close()
	if center_guess[0] is not None:
		return center_guess
	else:
		raise TimeoutError


def plot_raw_data(track_list, x_bins, y_bins, title):
	""" plot basic histograms of the tracks that have been loaded. """
	plt.figure()
	plt.hist2d(track_list['x(cm)'], track_list['y(cm)'], bins=(x_bins, y_bins))
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	plt.title(title)
	plt.axis('square')
	plt.title(f"TIM {data[TIM]} on shot {data[SHOT]}")
	plt.tight_layout()

	plt.figure()
	plt.hist2d(track_list['d(µm)'], track_list['cn(%)'], bins=(np.linspace(0, 20, 41), np.linspace(0, 40, 41)), cmap=COFFEE, norm=LogNorm())#, vmin=0, vmax=13000)
	# plt.plot([2, 2], [0, 40], 'k--')
	# plt.plot([3, 3], [0, 40], 'k--')
	plt.xlabel("Diameter (μm)") # plot N(d,c)
	plt.ylabel("Contrast (%)")
	plt.tight_layout()
	plt.show()


def plot_cooked_data(xC_bins, yC_bins, NC, xI_bins, yI_bins, NI, rI_bins, nI,
					 x0, y0, r0, r_img, δ, Q, e_min, e_max):
	""" plot the data along with the initial fit to it, and the
		reconstructed superaperture.
	"""
	plt.figure()
	plt.pcolormesh(xC_bins, yC_bins, NC.T, vmax=np.quantile(NC, (NC.size-6)/NC.size))
	T = np.linspace(0, 2*np.pi)
	# for dx, dy in get_relative_aperture_positions(s0, r_img, xC_bins.max(), mode=APERTURE_CONFIGURACION):
	# 	plt.plot(x0 + dx + r0*np.cos(T),    y0 + dy + r0*np.sin(T),    '--w')
	# 	plt.plot(x0 + dx + r_img*np.cos(T), y0 + dy + r_img*np.sin(T), '--w')
	plt.axis('square')
	plt.title(f"{e_min:.1f} MeV – {min(12.5, e_max):.1f} MeV")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Counts")
	plt.tight_layout()

	plt.figure()
	plt.pcolormesh(xI_bins, yI_bins, NI.T, vmax=np.quantile(NI, (NI.size-6)/NI.size))
	plt.axis('square')
	plt.title(f"TIM {data[TIM]} on shot {data[SHOT]} ({e_min:.1f} – {min(12.5, e_max):.1f} MeV)")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Counts")
	plt.tight_layout()
	for filetype in ['png', 'eps']:
		plt.savefig(f'results/{data[SHOT]} TIM{data[TIM]} [{e_min:.1f},{e_max:.1f}] projection.{filetype}')
	save_as_hdf5(f'results/{data[SHOT]} TIM{data[TIM]} [{e_min:.1f},{e_max:.1f}] projection', x=xI_bins, y=yI_bins, z=NI)

	if mode == 'hist' and s0 == 0:
		rI, drI = (rI_bins[1:] + rI_bins[:-1])/2, rI_bins[:-1] - rI_bins[1:]
		nI = nI/(np.pi*(rI_bins[1:]**2 - rI_bins[:-1]**2))
		plt.figure()
		plt.bar(x=rI, height=nI, width=drI,  label="Data", color=(0.773, 0.498, 0.357))
		n_test = simple_penumbra(rI, δ, Q, r0, r_img, 0, 1, e_min=e_min, e_max=e_max)
		signal, background = mysignal.linregress(n_test, nI, rI)
		r = np.linspace(0, r_img, 216)
		n_actual = simple_penumbra(r, δ, Q, r0, r_img, background, background+signal, e_min=e_min, e_max=e_max)
		plt.plot(r, n_actual, '-', color=(0.208, 0.455, 0.663), linewidth=2, label="Fit with charging")
		n_uncharged = simple_penumbra(r, δ+5*Q/e_max, 0, r0, r_img, background, background+signal, e_min=e_min, e_max=e_max)
		plt.plot(r, n_uncharged, '--', color=(0.278, 0.439, 0.239), linewidth=2, label="Fit without charging")
		plt.xlim(0, r_img)
		plt.xlabel("Radius (cm)")
		plt.ylabel("Track density (1/cm^2)")
		plt.legend()
		plt.title(f"TIM {data[TIM]} on shot {data[SHOT]} ({e_min:.1f} – {min(12.5, e_max):.1f} MeV)")
		plt.tight_layout()
		for filetype in ['png', 'eps']:
			plt.savefig(f'results/{data[SHOT]} TIM{data[TIM]} [{e_min:.1f},{e_max:.1f}] radial-lineout.{filetype}')
		save_as_hdf5(f'results/{data[SHOT]} TIM{data[TIM]} [{e_min:.1f},{e_max:.1f}] radial-lineout', x1=rI, y1=nI, x2=r, y2=n_actual, x3=r, y3=n_uncharged)

	plt.show()


def project(r, θ, ɸ, basis):
	""" project given spherical coordinates (with angles in degrees) into the
		detector plane x and y, as well as z out of the page.
	"""
	θ, ɸ = np.radians(θ), np.radians(ɸ)
	v = [r*np.sin(θ)*np.cos(ɸ), r*np.sin(θ)*np.sin(ɸ), r*np.cos(θ)]
	return np.matmul(basis.T, v)


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


def hessian(f, x, args, epsilon=None):
	def f_red(xp):
		return f(xp, *args)
	H = nd.Hessian(f_red, step=epsilon)
	return H(x)


def simple_penumbra(r, δ, Q, r0, r_max, minimum, maximum, e_min=0, e_max=1):
	""" compute the shape of a simple analytic single-apeture penumbral image """
	rB, nB = get_analytic_brightness(r0, Q, e_min=e_min, e_max=e_max) # start by accounting for aperture charging but not source size
	if 4*δ >= r_max: # if the source size is over 1/4 of the image radius
		raise ValueError("δ is too big compared to r_max: 4*{}/{} >= 1".format(δ, r_max)) # give up
	elif 4*δ >= r_max/n_bins: # if 4*source size is smaller than the image radius but bigger than the pixel size
		r_kernel = np.linspace(-4*δ, 4*δ, int(4*δ/r_max*n_bins)*2+1) # make a little kernel
		n_kernel = np.exp(-r_kernel**2/δ**2)
		r_point = np.arange(-4*δ, r_max + 4*δ, r_kernel[1] - r_kernel[0]) # rebin the existing image to match the kernel spacing
		n_point = np.interp(r_point, rB, nB, right=0)
		assert len(n_point) >= len(n_kernel)
		penumbra = np.convolve(n_point, n_kernel, mode='same') # and convolve
	elif δ >= 0: # if 4*source size is smaller than one pixel and nonnegative
		r_point = np.linspace(0, r_max, n_bins) # use a dirac kernel instead of a gaussian
		penumbra = np.interp(r_point, rB, nB, right=0)
	else:
		raise ValueError("δ cannot be negative")
	w = np.interp(r, r_point, penumbra/np.max(penumbra), right=0) # map to the requested r values
	return minimum + (maximum-minimum)*w


def simple_fit(*args, a=1, b=0, c=1):
	""" compute how close these data are to this penumbral image """
	if len(args[0]) == 3 and len(args) == 13: # first, parse the parameters
		(x0, y0, δ), Q, r0, s0, r_img, minimum, maximum, X, Y, exp, populated_region, e_min, e_max = args
	elif len(args[0]) == 4 and len(args) == 11:
		(x0, y0, δ, r0), s0, r_img, minimum, maximum, X, Y, exp, populated_region, e_min, e_max = args
		Q = 0
	elif len(args[0]) == 4 and len(args) == 12:
		(x0, y0, δ, Q), r0, s0, r_img, minimum, maximum, X, Y, exp, populated_region, e_min, e_max = args
	elif len(args[0]) == 7 and len(args) == 12:
		(x0, y0, δ, Q, a, b, c), r0, s0, r_img, minimum, maximum, X, Y, exp, populated_region, e_min, e_max = args
	else:
		raise ValueError("unsupported set of arguments")
	if Q < 0 or abs(x0) > abs(X).max() or abs(y0) > abs(Y).max(): return float('inf') # and reject impossible ones

	x_eff = a*(X - x0) + b*(Y - y0)
	y_eff = b*(X - x0) + c*(Y - y0)
	include = np.full(X.shape, False) # decide at which pixels to even look
	teo = np.zeros(X.shape) # and build up the theoretical image
	for dx, dy in get_relative_aperture_positions(s0, r_img, X.max(), mode=APERTURE_CONFIGURACION):
		r_rel = np.hypot(x_eff - dx, y_eff - dy)
		include[r_rel <= r_img] = True
		try:
			teo[r_rel <= r_img] += simple_penumbra(r_rel[r_rel <= r_img], δ, Q, r0, r_img, 0, 1, e_min, e_max) # as an array of penumbrums
		except ValueError:
			return np.inf
	if np.any(np.isnan(teo)):
		return np.inf

	include = include & populated_region # exclude outer regions where there are clearly no data
	
	if np.sum(include) == 0:
		return np.inf
	if minimum is None: # if the max and min are unspecified
		scale, minimum = mysignal.linregress(teo, exp, include/(1 + teo))
		maximum = minimum + scale
		minimum = max(0, minimum)
	if minimum > maximum:
		minimum, maximum = maximum, 2*minimum
	teo = minimum + teo*(maximum - minimum)

	penalty = \
		- 2*np.sum(include) \
		+ (np.sqrt(a*c - b**2) - 1)**2/(4*EXPECTED_MAGNIFICATION_ACCURACY**2) \
		- Q/.1

	include = include & (teo != 0) # from now on, ignore problematic pixels

	if np.any((teo == 0) & (exp != 0)): # if they are truly problematic, quit now
		return np.inf
	elif NON_STATISTICAL_NOISE > 1/6*np.max(exp, where=include, initial=0)**(-1/2):
		α = 1/NON_STATISTICAL_NOISE**2 # use a poisson error model with a gamma-distributed rate
		error = (α + exp)*np.log(α/teo + 1) - α*np.log(α/teo) - np.log(comb(α + exp - 1, exp))
	else: # but you can just use plain poisson if it's all the same to you
		error = teo - exp*np.log(teo) + np.log(factorial(exp))

	if SHOW_DEBUG_PLOTS:
		plt.pcolormesh(np.where(include, error, 2), vmin=0, vmax=10)
		# plt.pcolormesh(np.where(include, exp - teo, 0), cmap='RdBu', norm=CenteredNorm())
		plt.axis('square')
		plt.title(f"r0 = ({x0:.2f}, {y0:.2f}), δ = {δ:.3f}, Q = {Q:.3f}")
		plt.text(0, 0, f"ɛ = {np.sum(error, where=include) + penalty:.1f} Np")
		plt.show()

	return np.sum(error, where=include) + penalty


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
		raise f"what in davy jones's locker is a {mode}"


if __name__ == '__main__':
	shot_list = pd.read_csv('shot_list.csv')

	for i, data in shot_list.iterrows():
		if APERTURE_CHARGE_FITTING == 'none':
			Q, dQ = 0, 0
		else:
			Q, dQ = None, None

		L = data[APERTURE_DISTANCE] # cm
		M = data[MAGNIFICATION] # cm
		rA = data[APERTURE_RADIUS]/1.e4 # cm
		sA = data[APERTURE_SPACING]/1.e4 # cm
		rotation = np.radians(data[ROTATION]) # rad
		etch_time = float(data[ETCH_TIME].strip(' h'))

		r0 = (M + 1)*rA # calculate the penumbra parameters
		s0 = (M + 1)*sA
		r_img = SPREAD*r0 + 1.05*M*OBJECT_SIZE
		if s0 != 0 and r_img > s0/2:
			r_img = s0/2 # make sure the image at which we look is small enough to avoid other penumbrae
		n_bins = min(1000, int(RESOLUTION/(1.05*M*OBJECT_SIZE)*r_img)) # get the image resolution needed to resolve the object

		θ_TIM, ɸ_TIM = np.radians(TIM_LOCATIONS[int(data[TIM])-1])
		basis = np.array([
			[0, 0, 0],
			[np.sin(θ_TIM-np.pi/2)*np.cos(ɸ_TIM), np.sin(θ_TIM-np.pi/2)*np.sin(ɸ_TIM), np.cos(θ_TIM-np.pi/2)],
			[np.sin(θ_TIM)*np.cos(ɸ_TIM), np.sin(θ_TIM)*np.sin(ɸ_TIM), np.cos(θ_TIM)],
		]).T
		basis[:,0] = np.cross(basis[:,1], basis[:,2])

		x_off, y_off, z_off = project(float(data[R_OFFSET]), float(data[Θ_OFFSET]), float(data[Φ_OFFSET]), basis)*1e-4 # cm
		x_flo, y_flo, z_flo = project(float(data[R_FLOW]), float(data[Θ_FLOW]), float(data[Φ_FLOW]), basis)*1e-4 # cm/ns

		filename = None
		for fname in os.listdir(FOLDER):
			if (fname.endswith('.txt') or fname.endswith('.pkl')) \
					and	str(data[SHOT]) in fname and ('tim'+str(data[TIM]) in fname.lower() or 'tim' not in fname.lower()) \
					and data[ETCH_TIME].replace(' ','') in fname:
				filename = fname
				print("\nBeginning reconstruction for TIM {} on shot {}".format(data[TIM], data[SHOT]))
				break
		if filename is None:
			print("Could not find text file for TIM {} on shot {}".format(data[TIM], data[SHOT]))
			continue
		if filename.endswith('.txt'): # if it is a typical CPSA-derived text file
			mode = 'hist'
			track_list = pd.read_csv(FOLDER+filename, sep=r'\s+', header=20, skiprows=[24], encoding='Latin-1', dtype='float32') # load all track coordinates

			x_temp, y_temp = track_list['x(cm)'].copy(), track_list['y(cm)'].copy()
			track_list['x(cm)'] =  np.cos(rotation+np.pi)*x_temp - np.sin(rotation+np.pi)*y_temp # apply any requested rotation, plus 180 flip to deal with inherent flip due to aperture
			track_list['y(cm)'] =  np.sin(rotation+np.pi)*x_temp + np.cos(rotation+np.pi)*y_temp
			if re.fullmatch(r'[0-9]+', str(data[SHOT])): # adjustments for real data:
				track_list['ca(%)'] -= np.min(track_list['cn(%)']) # shift the contrasts down if they're weird
				track_list['cn(%)'] -= np.min(track_list['cn(%)'])
				track_list['d(µm)'] -= np.min(track_list['d(µm)']) # shift the diameters over if they're weird
			hicontrast = (track_list['cn(%)'] < 35) & (track_list['e(%)'] < 15)

			track_list['x(cm)'] -= np.mean(track_list['x(cm)'][hicontrast]) # do your best to center
			track_list['y(cm)'] -= np.mean(track_list['y(cm)'][hicontrast])

			view_radius = max(np.max(track_list['x(cm)']), np.max(track_list['y(cm)']))
			xC_bins, yC_bins = np.linspace(-view_radius, view_radius, n_bins+1), np.linspace(-view_radius, view_radius, n_bins+1) # this is the CR39 coordinate system, centered at 0,0
			dxC, dyC = xC_bins[1] - xC_bins[0], yC_bins[1] - yC_bins[0] # get the bin widths
			xC, yC = (xC_bins[:-1] + xC_bins[1:])/2, (yC_bins[:-1] + yC_bins[1:])/2 # change these to bin centers
			XC, YC = np.meshgrid(xC, yC, indexing='ij') # change these to matrices

			if SHOW_RAW_PLOTS:
				plot_raw_data(track_list[hicontrast], xC_bins, yC_bins, "")#f"Penumbral image, TIM{data[TIM]}, shot {data[SHOT]}")

		else: # if it is a pickle file, load the histogram directly like a raster image
			mode = 'raster'
			with open(FOLDER+filename, 'rb') as f:
				xI_bins, yI_bins, NI = pickle.load(f)
			dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
			xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2
			XI, YI = np.meshgrid(xI, yI, indexing='ij')

			xC_bins, yC_bins, NC = xI_bins, yI_bins, NI
			XC, YC = XI, YI

		image_layers, X_layers, Y_layers = [], [], []

		if mode == 'raster' or np.std(track_list['d(µm)']) == 0: # choose which cuts to use depending on whether this is synthetic or real
			cuts = [('plasma', [0, 100])]
		else:
			# cuts = [(GREYS, [0, 100])]
			# cuts = [(GREYS, [0, 100]), (REDS, [0, 7]), (BLUES, [10, 100])] # [MeV] (pre-filtering)
			cuts = [(REDS, [0, 7]), (BLUES, [9, 100])] # [MeV] (pre-filtering)

		for color, (cmap, e_in_bounds) in enumerate(cuts): # iterate over the cuts
			e_out_bounds = get_E_out(1, 2, e_in_bounds, ['Ta'], 16) # convert scattering energies to CR-39 energies TODO: parse filtering specification
			e_in_bounds = get_E_in(1, 2, e_out_bounds, ['Ta'], 16) # convert back to exclude particles that are ranged out
			d_bounds = diameter.D(e_out_bounds, τ=etch_time)[::-1] # convert to diameters
			if np.isnan(d_bounds[1]):
				d_bounds[1] = np.inf # and if the bin goes down to zero energy, make sure all large diameters are counted

			if mode == 'hist': # if we still need to tally the histogram

				print(f"d ∈ {d_bounds} μm")
				track_x = track_list['x(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
				track_y = track_list['y(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()

				if len(track_x) <= 0:
					print("No tracks found in this cut.")
					continue

				if APERTURE_CHARGE_FITTING == 'all':
					Q = None

				NC, xC_bins, yC_bins = np.histogram2d( # make a histogram
					track_x, track_y, bins=(xC_bins, yC_bins))
				assert NC.shape == XC.shape

				if ASK_FOR_HELP:
					try: # ask the user for help finding the center
						x0, y0 = where_is_the_ocean(xC_bins, yC_bins, NC, "Please click on the center of a penumbrum.", timeout=8.64)
					except:
						x0, y0 = (0, 0)
				else:
					x0, y0 = (0, 0)
			else:
				x0, y0 = (0, 0)

			hullC = convex_hull(XC, YC, NC) # compute the convex hull for future uce

			if Q is None:
				print('fitting electric field')
				args = (r0, s0, r_img, None, None, XC, YC, NC, hullC, *e_in_bounds)
				opt = optimize.minimize(simple_fit, x0=[None]*4, args=args,
					method='Nelder-Mead', options=dict(initial_simplex=[
						[x0+r_img*.4, y0,         OBJECT_SIZE*M/4, 1.0e-1],
						[x0-r_img*.2, y0+r_img*.3, OBJECT_SIZE*M/4, 1.0e-1],
						[x0-r_img*.2, y0-r_img*.3, OBJECT_SIZE*M/4, 1.0e-1],
						[x0,         y0,         OBJECT_SIZE*M/3, 1.0e-1],
						[x0,         y0,         OBJECT_SIZE*M/4, 1.9e-1]]))
				x0, y0, δ, Q = opt.x
				hess = hessian(simple_fit, opt.x, args=args)
				print(hess)
				dx0, dy0, dδ, dQ = np.sqrt(np.diagonal(np.linalg.inv(hess)))
			else:
				args = (Q, r0, s0, r_img, None, None, XC, YC, NC, hullC, *e_in_bounds)
				opt = optimize.minimize(simple_fit, x0=[None]*3, args=args,
					method='Nelder-Mead', options=dict(initial_simplex=[
						[x0+r_img*.4, y0,         OBJECT_SIZE*M/4],
						[x0-r_img*.2, y0+r_img*.3, OBJECT_SIZE*M/4],
						[x0-r_img*.2, y0-r_img*.3, OBJECT_SIZE*M/4],
						[x0,         y0,         OBJECT_SIZE*M/3]]))
				x0, y0, δ = opt.x
				hess = hessian(simple_fit, opt.x, args=args)
				dx0, dy0, dδ = np.sqrt(np.diagonal(np.linalg.inv(hess)))
			if VERBOSE: print(opt)
			print(f"n = {np.sum(NC):.4g}, (x0, y0) = ({x0:.3f}, {y0:.3f}), δ = {δ/M/1e-4:.3f} ± {dδ:.3f} μm, Q = {Q:.3f} ± {dQ:.3f} cm*MeV, M = {M:.2f}")

			if mode == 'hist':
				xI_bins, yI_bins = np.linspace(x0 - r_img, x0 + r_img, n_bins+1), np.linspace(y0 - r_img, y0 + r_img, n_bins+1) # this is the CR39 coordinate system, but encompassing a single superpenumbrum
				dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
				xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2
				XI, YI = np.meshgrid(xI, yI, indexing='ij')
				NI = np.zeros(XI.shape) # and N combines all penumbra on that square
				for dx, dy in get_relative_aperture_positions(s0, r_img, xC_bins.max(), mode=APERTURE_CONFIGURACION):
					NI += np.histogram2d(track_x, track_y, bins=(xI_bins + dx, yI_bins + dy))[0] # do that histogram

				track_r = np.hypot(track_x - x0, track_y - y0)
				nI, rI_bins = np.histogram(track_r, bins=np.linspace(0, r_img, 36)) # also do a radial histogram because that might be useful

				del(track_x)
				del(track_y)
				gc.collect()
			else:
				nI, rI_bins = None, None

			kernel_size = xI_bins.size - 2*int(1.05*M*OBJECT_SIZE/dxI) # now make the kernel (from here on, it's the same in both modes)
			if kernel_size%2 == 0: # make sure the kernel is odd
				kernel_size -= 1
			xK_bins, yK_bins = np.linspace(-dxI*kernel_size/2, dxI*kernel_size/2, kernel_size+1), np.linspace(-dyI*kernel_size/2, dyI*kernel_size/2, kernel_size+1)
			dxK, dyK = xK_bins[1] - xK_bins[0], yK_bins[1] - yK_bins[0]
			XK, YK = np.meshgrid((xK_bins[:-1] + xK_bins[1:])/2, (yK_bins[:-1] + yK_bins[1:])/2, indexing='ij') # this is the kernel coordinate system, measured from the center of the umbra

			xS_bins, yS_bins = xI_bins[kernel_size//2:-(kernel_size//2)]/M, yI_bins[kernel_size//2:-(kernel_size//2)]/M # this is the source system.
			dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
			xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
			XS, YS = np.meshgrid(xS, yS, indexing='ij')

			if SHOW_PLOTS:
				assert e_in_bounds[0] != 0
				plot_cooked_data(xC_bins, yC_bins, NC, xI_bins, yI_bins, NI, rI_bins, nI, x0, y0, r0, r_img, δ, Q, *e_in_bounds)
			if SKIP_RECONSTRUCTION:
				continue

			penumbral_kernel = np.zeros(XK.shape) # build the point spread function
			for dx in [-dxK/3, 0, dxK/3]: # sampling over a few pixels
				for dy in [-dyK/3, 0, dyK/3]:
					penumbral_kernel += simple_penumbra(np.hypot(XK+dx, YK+dy), 0, Q, r0, r_img, 0, 1, *e_in_bounds)
			penumbral_kernel = penumbral_kernel/np.sum(penumbral_kernel)

			source_bins = np.hypot(XS - XS.mean(), YS - YS.mean()) <= (xS_bins[-1] - xS_bins[0])/2
			reach = pysignal.convolve2d(source_bins, penumbral_kernel, mode='full')
			penumbra_low = .005*np.sum(source_bins)*penumbral_kernel.max()# np.quantile(penumbral_kernel/penumbral_kernel.max(), .05)
			penumbra_hih = .99*np.sum(source_bins)*penumbral_kernel.max()# np.quantile(penumbral_kernel/penumbral_kernel.max(), .70)
			data_bins = (np.hypot(XI, YI) <= r_img) & np.isfinite(NI) & (reach > penumbra_low) & (reach < penumbra_hih) # exclude bins that are NaN and bins that are touched by all or none of the source pixels
			try:
				data_bins &= convex_hull(XI, YI, NI) # crop it at the convex hull where counts go to zero
			except MemoryError:
				print("WARN: could not allocate enough memory to crop data by convex hull; some non-data regions may be getting considered in the analysis.")

			if SHOW_DEBUG_PLOTS:
				plt.figure()
				plt.pcolormesh(xK_bins, yK_bins, penumbral_kernel)
				plt.axis('square')
				plt.title("Point spread function")
				plt.figure()
				plt.pcolormesh(xI_bins, yI_bins, np.where(data_bins, reach, np.nan))
				plt.axis('square')
				plt.title("Maximum convolution")
				plt.show()

			B, χ2_red = mysignal.gelfgat_deconvolve2d(
				NI,
				penumbral_kernel,
				where=data_bins,
				illegal=np.logical_not(source_bins),
				verbose=VERBOSE,
				show_plots=SHOW_DEBUG_PLOTS) # deconvolve!

			if χ2_red >= 1.5: # throw it away if it looks unreasonable
				print("Could not find adequate fit.")
				continue
			print(f"χ^2/n = {χ2_red}")
			# if χ2_red >= 2.0:
			# 	print("Warn: χ^2/n is suspiciously hi.")
			B[np.hypot(XS, YS) >= OBJECT_SIZE] = 0 # trim the edges
			B = np.maximum(0, B) # we know this must be nonnegative

			p0, (p1, θ1), (p2, θ2) = mysignal.shape_parameters(xS, yS, B, contour=CONTOUR) # compute the three number summary
			print(f"P0 = {p0/1e-4:.2f} μm")
			print(f"P2 = {p2/1e-4:.2f} μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2):.1f}°")

			def func(x, A, mouth):
				return A*(1 + erf((100e-4 - x)/mouth))/2
			real = source_bins
			# cx, cy = np.average(XS, weights=B), np.average(YS, weights=B)
			# (A, mouth), _ = optimize.curve_fit(func, np.hypot(XS - cx, YS - cy)[real], B[real], p0=(2*np.average(B), 10e-4)) # fit to a circle thing
			# print(f"XXX[{data[SHOT][-5:]}, {mouth/1e-4:.1f}],")

			x0 = XS[np.unravel_index(np.argmax(B), XS.shape)]
			y0 = YS[np.unravel_index(np.argmax(B), YS.shape)]

			cut_name = 'lo' if cmap is REDS else 'md' if cmap is GREENS else 'hi' if cmap is BLUES else 'all'

			plt.figure() # plot the reconstructed source image
			plt.pcolormesh((xS_bins - x0)/1e-4, (yS_bins - y0)/1e-4, B.T, cmap=cmap, vmin=0)
			T = np.linspace(0, 2*np.pi)
			# plt.plot(ronnot*np.cos(T)/1e-4, ronnot*np.sin(T)/1e-4, 'w--')
			# plt.colorbar()
			plt.axis('square')
			# plt.title("TIM {} on shot {}, {:.1f} MeV – {:.1f} MeV".format(data[TIM], data[SHOT], *e_in_bounds))
			plt.title(f"{e_in_bounds[0]:.1f} MeV – {min(12.5, e_in_bounds[1]):.1f} MeV")
			plt.xlabel("x (μm)")
			plt.ylabel("y (μm)")
			plt.axis([-150, 150, -150, 150])
			plt.tight_layout()
			for filetype in ['png', 'eps']:
				plt.savefig(f"results/{data[SHOT]} TIM{data[TIM]} {cut_name} {etch_time}h reconstruction.{filetype}")
			save_as_hdf5(f"results/{data[SHOT]} TIM{data[TIM]} {cut_name} {etch_time}h reconstruction", x=xS_bins, y=yS_bins, z=B.T)
			plt.tight_layout()

			if SHOW_PLOTS:
				plt.show()
			else:
				plt.close()

			image_layers.append(B/B.max())
			X_layers.append(XS)
			Y_layers.append(YS)

		try:
			xX_bins, yX_bins = np.linspace(-100, 100, 101), np.linspace(-100, 100, 101)
			xX, yX = (xX_bins[:-1] + xX_bins[1:])/2, (yX_bins[:-1] + yX_bins[1:])/2 # change these to bin centers
			XX, YX = np.meshgrid(xX, yX, indexing='ij') # change these to matrices
			xray = np.loadtxt('scans/KoDI_xray_data1 - {:d}-TIM{:d}-{:d}.mat.csv'.format(int(data[SHOT]), int(data[TIM]), [2,4,5].index(int(data[TIM]))+1), delimiter=',').T
		except (ValueError, OSError):
			xray = None
		if xray is not None:
			print("Xray image")
			p0, (p1, θ1), (p2, θ2) = mysignal.shape_parameters(xX, yX, xray.T, contour=CONTOUR) # compute the three number summary
			print(f"P0 = {p0:.2f} μm")
			print(f"P2 = {p2:.2f} μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2):.1f}°")

			plt.figure()
			plt.pcolormesh(xX_bins, yX_bins, xray, cmap=VIOLETS, vmin=0)
			# plt.colorbar()
			plt.axis('square')
			plt.title("X-ray image of TIM {} on shot {}".format(data[TIM], data[SHOT]), fontsize=12)
			plt.xlabel("x (μm)")
			plt.ylabel("y (μm)")
			plt.axis([-150, 150, -150, 150])
			plt.tight_layout()
			for filetype in ['png', 'eps']:
				plt.savefig(f"results/{data[SHOT]} TIM{data[TIM]} xray sourceimage.{filetype}")
			save_as_hdf5(f"results/{data[SHOT]} TIM{data[TIM]} xray sourceimage", x=xX_bins, y=yX_bins, z=xray)
			plt.close()

		if len(image_layers) >= 2:
			x0 = X_layers[0][np.unravel_index(np.argmax(image_layers[0]), image_layers[0].shape)]
			y0 = Y_layers[0][np.unravel_index(np.argmax(image_layers[0]), image_layers[0].shape)]

			if len(image_layers) == 2:
				red = 0
				blu = 1
			else:
				red = 1
				blu = -1

			dx = np.average(X_layers[blu], weights=image_layers[blu]) - np.average(X_layers[red], weights=image_layers[red])
			dy = np.average(Y_layers[blu], weights=image_layers[blu]) - np.average(Y_layers[red], weights=image_layers[red])
			print(f"Δ = {np.hypot(dx, dy)/1e-4:.1f} μm, θ = {np.degrees(np.arctan2(dx, dy)):.1f}")

			plt.figure()
			plt.contourf((X_layers[red] - x0)/1e-4, (Y_layers[red] - y0)/1e-4, image_layers[red], levels=[0, 0.15, 1], colors=['#00000000', '#FF5555BB', '#000000FF'])
			plt.contourf((X_layers[blu] - x0)/1e-4, (Y_layers[blu] - y0)/1e-4, image_layers[blu], levels=[0, 0.15, 1], colors=['#00000000', '#5555FFBB', '#000000FF'])
			# if xray is not None:
			# 	plt.contour(XX, YX, xray, levels=[.25], colors=['#550055BB'])
			if SHOW_OFFSET:
				plt.plot([0, x_off/1e-4], [0, y_off/1e-4], '-k')
				plt.scatter([x_off/1e-4], [y_off/1e-4], color='k')
				plt.arrow(0, 0, x_flo/1e-4, y_flo/1e-4, color='k', head_width=5, head_length=5, length_includes_head=True)
				plt.text(0.05, 0.95, "offset out of page = {:.3f}\nflow out of page = {:.3f}".format(
					z_off/np.sqrt(x_off**2 + y_off**2 + z_off**2), z_flo/np.sqrt(x_flo**2 + y_flo**2 + z_flo**2)),
					verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)
			plt.axis('square')
			plt.axis([-150, 150, -150, 150])
			plt.xlabel("x (μm)")
			plt.ylabel("y (μm)")
			plt.title("TIM {} on shot {}".format(data[TIM], data[SHOT]))
			plt.tight_layout()
			for filetype in ['png', 'eps']:
				plt.savefig(f"results/{data[SHOT]} TIM{data[TIM]} nested sourceimage.{filetype}")
			plt.close()
