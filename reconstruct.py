import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import os
import scipy.optimize as optimize
import scipy.signal as signal
import scipy.stats as stats
import scipy.special as special
from scipy.spatial import Delaunay
import gc
import re
import time

import diameter
from electric_field_model import get_analytic_brightness
from cmap import REDS, GREENS, BLUES, VIOLETS, GREYS, COFFEE

np.seterr('ignore')

SHOW_PLOTS = False
SHOW_DEBUG_PLOTS = False
VERBOSE = True
OBJECT_SIZE = 400e-4 # cm
THRESHOLD = 3e-5
ASK_FOR_HELP = False

VIEW_RADIUS = 3.0 # cm
NON_STATISTICAL_NOISE = 0.0
EXPECTED_MAGNIFICATION_ACCURACY = 4e-3
n_bins = 400

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


def sl(a, b, c):
	if b == -1 and c < 0: b = None
	return slice(a, b, c)

def paste_slices(tup):
	pos, w, max_w = tup
	wall_min = max(pos, 0)
	wall_max = min(pos+w, max_w)
	block_min = -min(pos, 0)
	block_max = max_w-max(pos+w, max_w)
	block_max = block_max if block_max != 0 else None
	return slice(wall_min, wall_max), slice(block_min, block_max)

def paste(wall, block, loc):
	""" insert block into wall; loc is the position of block[0,0] in wall """
	loc_zip = zip(loc, block.shape, wall.shape)
	wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
	wall[wall_slices] = block[block_slices]

def convolve2d(a, b, where):
	""" more efficient when where is mostly False, less efficient otherwise.
		I don't know which way b is supposed to face, so make it symmetric. """
	c = np.zeros(where.shape)
	for i, j in zip(*np.nonzero(where)):
		mt = max(0, i - b.shape[0] + 1) # omitted rows on top
		mb = max(0, a.shape[0] - i - 1) # omitted rows on bottom
		mr = max(0, j - b.shape[1] + 1) # omitted columns on right
		ml = max(0, a.shape[1] - j - 1) # omitted rows on left
		c[i,j] += np.sum(a[mt:a.shape[0]-mb, mr:a.shape[1]-ml]*b[sl(i-mt, i-a.shape[0]+mb, -1), sl(j-mr, j-a.shape[1]+ml, -1)])
	return c

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
	if len(args[0]) == 3 and len(args) == 12: # first, parse the parameters
		(x0, y0, δ), Q, r0, s0, r_img, minimum, maximum, X, Y, exp, e_min, e_max = args
	elif len(args[0]) == 4 and len(args) == 10:
		(x0, y0, δ, r0), s0, r_img, minimum, maximum, X, Y, exp, e_min, e_max = args
		Q = 0
	elif len(args[0]) == 4 and len(args) == 11:
		(x0, y0, δ, Q), r0, s0, r_img, minimum, maximum, X, Y, exp, e_min, e_max = args
	elif len(args[0]) == 7 and len(args) == 11:
		(x0, y0, δ, Q, a, b, c), r0, s0, r_img, minimum, maximum, X, Y, exp, e_min, e_max = args
	else:
		raise ValueError("unsupported set of arguments")
	if Q < 0 or abs(x0) > VIEW_RADIUS or abs(y0) > VIEW_RADIUS: return float('inf') # and reject impossible ones

	x_eff = a*(X - x0) + b*(Y - y0)
	y_eff = b*(X - x0) + c*(Y - y0)
	teo = np.zeros(X.shape) # build up the theoretical image
	include = np.full(X.shape, False) # and decide at which pixels to even look
	for i in range(-6, 6):
		dy = i*np.sqrt(3)/2*s0
		for j in range(-6, 6):
			dx = (2*j + i%2)*s0/2
			if dx-s0 < VIEW_RADIUS and dy-s0 < VIEW_RADIUS and dx+s0 > -VIEW_RADIUS and dy+s0 > -VIEW_RADIUS:
				r_rel = np.hypot(x_eff - dx, y_eff - dy)
				try:
					teo[r_rel <= r_img] += simple_penumbra(r_rel[r_rel <= r_img], δ, Q, r0, r_img, 0, 1, e_min, e_max) # as an array of penumbrums
				except ValueError:
					return np.inf
				include[r_rel <= r_img] = True
				if np.any(np.isnan(teo)):
					return np.inf
	if minimum is None or maximum is None:
		minimum = np.average(exp, weights=teo < 1/6) # then compute the best scaling for it
		maximum = np.average(exp, weights=teo > 5/6)
	if minimum > maximum:
		minimum, maximum = maximum, minimum
	teo = minimum + (teo - teo.min())/(teo.max() - teo.min())*(maximum - minimum)
	error = np.sum((exp - teo)**2/(teo + (NON_STATISTICAL_NOISE*teo)**2),
			where=include) # use a gaussian error model
	penalty = -np.sum(include) \
		+ (a**2 + 2*b**2 + c**2)/(4*EXPECTED_MAGNIFICATION_ACCURACY**2)
	return error + penalty


if __name__ == '__main__':
	xC_bins, yC_bins = np.linspace(-VIEW_RADIUS, VIEW_RADIUS, n_bins+1), np.linspace(-VIEW_RADIUS, VIEW_RADIUS, n_bins+1) # this is the CR39 coordinate system, centered at 0,0
	dxC, dyC = xC_bins[1] - xC_bins[0], yC_bins[1] - yC_bins[0] # get the bin widths
	xC, yC = (xC_bins[:-1] + xC_bins[1:])/2, (yC_bins[:-1] + yC_bins[1:])/2 # change these to bin centers
	XC, YC = np.meshgrid(xC, yC, indexing='ij') # change these to matrices

	shot_list = pd.read_csv('shot_list.csv')

	for i, scan in shot_list.iterrows():
		filename = None
		for fname in os.listdir(FOLDER):
			if fname.endswith('.txt') and str(scan[SHOT]) in fname and 'tim'+str(scan[TIM]) in fname.lower() and scan[ETCH_TIME].replace(' ','') in fname:
				filename = fname
				print("Beginning reconstruction for TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
				break
		if filename is None:
			print("Could not find text file for TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
			continue

		Q = None
		L = scan[APERTURE_DISTANCE] # cm
		M = scan[MAGNIFICATION] # cm
		rA = scan[APERTURE_RADIUS]/1.e4 # cm
		sA = scan[APERTURE_SPACING]/1.e4 # cm
		rotation = np.radians(scan[ROTATION]) # rad
		if sA == 0: sA = 6*VIEW_RADIUS/(M + 1)
		etime = float(scan[ETCH_TIME].strip(' h'))
		track_list = pd.read_csv(FOLDER+filename, sep=r'\s+', header=20, skiprows=[24], encoding='Latin-1', dtype='float32')

		θ_TIM, ɸ_TIM = np.radians(TIM_LOCATIONS[int(scan[TIM])-1])
		w_TIM = [np.sin(θ_TIM)*np.cos(ɸ_TIM), np.sin(θ_TIM)*np.sin(ɸ_TIM), np.cos(θ_TIM)]
		v_TIM = [np.sin(θ_TIM-np.pi/2)*np.cos(ɸ_TIM), np.sin(θ_TIM-np.pi/2)*np.sin(ɸ_TIM), np.cos(θ_TIM-np.pi/2)]
		u_TIM = np.cross(v_TIM, w_TIM)

		r_off, θ_off, ɸ_off = float(scan[R_OFFSET])*1e-4, np.radians(float(scan[Θ_OFFSET])), np.radians(float(scan[Φ_OFFSET])) # cm
		offset = [r_off*np.sin(θ_off)*np.cos(ɸ_off), r_off*np.sin(θ_off)*np.sin(ɸ_off), r_off*np.cos(θ_off)]
		x_off, y_off, z_off = np.dot(u_TIM, offset), np.dot(v_TIM, offset), np.dot(w_TIM, offset)

		r_flo, θ_flo, ɸ_flo = float(scan[R_FLOW])*1e-4, np.radians(float(scan[Θ_FLOW])), np.radians(float(scan[Φ_FLOW])) # cm/ns
		flow = [r_flo*np.sin(θ_flo)*np.cos(ɸ_flo), r_flo*np.sin(θ_flo)*np.sin(ɸ_flo), r_flo*np.cos(θ_flo)]
		x_flo, y_flo, z_flo = np.dot(u_TIM, flow), np.dot(v_TIM, flow), np.dot(w_TIM, flow)

		track_list['y(cm)'] *= -1 # cpsa files invert y
		x_temp, y_temp = track_list['x(cm)'].copy(), track_list['y(cm)'].copy() # rotate the flipped penumbral image 45 degrees clockwise
		track_list['x(cm)'] =  np.cos(rotation+np.pi/2)*x_temp - np.sin(rotation+np.pi/2)*y_temp
		track_list['y(cm)'] =  np.sin(rotation+np.pi/2)*x_temp + np.cos(rotation+np.pi/2)*y_temp
		if re.fullmatch(r'[0-9]+', str(scan[SHOT])): # adjustments for real data:
			track_list['ca(%)'] -= np.min(track_list['cn(%)']) # shift the contrasts down if they're weird
			track_list['cn(%)'] -= np.min(track_list['cn(%)'])
			track_list['d(µm)'] -= np.min(track_list['d(µm)']) # shift the diameters over if they're weird
		hicontrast = (track_list['cn(%)'] < 35) & (track_list['e(%)'] < 15)
		track_list['x(cm)'] -= np.mean(track_list['x(cm)'][hicontrast]) # do your best to center
		track_list['y(cm)'] -= np.mean(track_list['y(cm)'][hicontrast])

		# plt.hist2d(track_list['d(µm)'], track_list['cn(%)'], bins=(np.linspace(0, 10, 51), np.linspace(0, 40, 41)), cmap=COFFEE, vmin=0, vmax=13000)
		# plt.plot([2, 2], [0, 40], 'k--')
		# plt.plot([3, 3], [0, 40], 'k--')
		# plt.xlabel("Diameter (μm)", fontsize=16) # plot N(d,c)
		# plt.ylabel("Contrast (%)", fontsize=16)
		# plt.title(" ", fontsize=16)
		# plt.gca().xaxis.set_tick_params(labelsize=16)
		# plt.gca().yaxis.set_tick_params(labelsize=16)
		# plt.tight_layout()
		# plt.show()

		r0 = (M + 1)*rA
		s0 = (M + 1)*sA
		r_img = r0 + M*OBJECT_SIZE
		
		image_layers, x_layers, y_layers = [], [], []

		if np.std(track_list['d(µm)']) == 0:
			cuts = [('plasma', [0, 5])]
		else:
			cuts = [(GREYS, [0, 13]), (REDS, [0, 5]), (GREENS, [5, 9]), (BLUES, [9, 13])] # [MeV] (post-filtering)

		for color, (cmap, e_out_bounds) in enumerate(cuts):
			d_bounds = diameter.D(np.array(e_out_bounds), τ=etime)[::-1]
			e_in_bounds = np.clip(np.array(e_out_bounds) + 2, 0, 12)
			print(d_bounds)

			# Q = None

			track_x = track_list['x(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
			track_y = track_list['y(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
			if len(track_x) <= 0:
				print("No tracks found in this cut.")
				continue

			center_guess = [None, None] # ask the user for help finding the center
			if ASK_FOR_HELP:
				fig = plt.figure()
				N, xC_bins, yC_bins = np.histogram2d( # make a histogram
					track_x, track_y, bins=(xC_bins, yC_bins))
				plt.pcolormesh(xC_bins, yC_bins, N.T, vmax=np.quantile(N, .999))
				plt.axis('square')
				plt.colorbar()
				plt.title("Please click on the center of a penumbrum.")
				def onclick(event):
					center_guess[0] = event.xdata
					center_guess[1] = event.ydata
				fig.canvas.mpl_connect('button_press_event', onclick)
				start = time.time()
				while center_guess[0] is None and time.time() - start < 8.64:
					plt.pause(.01)
				plt.close()
			x0, y0 = center_guess if center_guess[0] is not None else (0, 0)

			N, xC_bins, yC_bins = np.histogram2d( # make a histogram
				track_x, track_y, bins=(xC_bins, yC_bins))
			assert N.shape == XC.shape

			if Q is None:
				opt = optimize.minimize(simple_fit, x0=[None]*4, args=(r0, s0, r_img, None, None, XC, YC, N, *e_in_bounds),
					method='Nelder-Mead', options=dict(
						initial_simplex=[
							[x0+r_img/2, y0, .06, 1e-1],
							[x0-r_img/2, y0+r_img/2, .06, 1e-1],
							[x0-r_img/2, y0-r_img/2, .06, 1e-1],
							[x0, y0, .1, 1e-1],
							[x0, y0, .06, 1.9e-1]]),
					# tol=1e-9,
					)
				x0, y0, δ, Q = opt.x
			else:
				opt = optimize.minimize(simple_fit, x0=[None]*3, args=(Q, r0, s0, r_img, None, None, XC, YC, N, *e_in_bounds),
					method='Nelder-Mead', options=dict(
						initial_simplex=[
							[x0+r_img/2, y0, .06],
							[x0-r_img/2, y0+r_img/2, .06],
							[x0-r_img/2, y0-r_img/2, .06],
							[x0, y0, .1]]),
					# tol=1e-9,
					)
				x0, y0, δ = opt.x
			M = r0/rA - 1
			if VERBOSE: print(opt)
			print("n = {0:.4g}, (x0, y0) = ({1:.3f}, {2:.3f}), δ = {3:.3f} μm, Q = {4:.3f} cm/MeV, M = {5:.2f}".format(np.sum(N), x0, y0, δ/M/1e-4, Q, M))

			xI_bins, yI_bins = np.linspace(x0 - r_img, x0 + r_img, n_bins+1), np.linspace(y0 - r_img, y0 + r_img, n_bins+1) # this is the CR39 coordinate system, but encompassing a single superpenumbrum
			dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
			xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2
			XI, YI = np.meshgrid(xI, yI, indexing='ij')
			N = np.zeros(XI.shape) # and N combines all penumbra on that square
			for i in range(-6, 6):
				dy = i*np.sqrt(3)/2*s0
				for j in range(-6, 6):
					dx = (2*j + i%2)*s0/2
					if np.hypot(dx, dy) + r_img <= VIEW_RADIUS:
						N += np.histogram2d(track_x, track_y, bins=(xI_bins + dx, yI_bins + dy))[0] # do that histogram

			kernel_size = int(2.05*r0/dxI) + 4 if int(2.05*r0/dxI)%2 == 1 else int(2.05*(r0+.2)/dxI) + 5
			n_pixs = n_bins - kernel_size + 1 # the source image will be smaller than the penumbral image
			xK_bins, yK_bins = np.linspace(-dxI*kernel_size/2, dxI*kernel_size/2, kernel_size+1), np.linspace(-dyI*kernel_size/2, dyI*kernel_size/2, kernel_size+1)
			dxK, dyK = xK_bins[1] - xK_bins[0], yK_bins[1] - yK_bins[0]
			XK, YK = np.meshgrid((xK_bins[:-1] + xK_bins[1:])/2, (yK_bins[:-1] + yK_bins[1:])/2, indexing='ij') # this is the kernel coordinate system, measured from the center of the umbra

			xS_bins, yS_bins = xI_bins[kernel_size//2:-(kernel_size//2)]/M, -yI_bins[kernel_size//2:-(kernel_size//2)]/M # this is the source system. note that it is reversed so that indices measured from the center match between I and S
			dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
			xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
			XS, YS = np.meshgrid(xS, yS, indexing='ij')

			if SHOW_PLOTS:
				plt.figure()
				plt.hist2d(track_x, track_y, bins=(xC_bins, yC_bins))
				T = np.linspace(0, 2*np.pi)
				plt.plot(x0 + r0*np.cos(T), y0 + r0*np.sin(T), '--w')
				# plt.plot(x0 - s0 + r0*np.cos(T), y0 + r0*np.sin(T), '--w')
				# plt.plot(x0 + s0 + r0*np.cos(T), y0 + r0*np.sin(T), '--w')
				plt.plot(x0 + r_img*np.cos(T), y0 + r_img*np.sin(T), '--w')
				plt.plot(x0 + VIEW_RADIUS*np.cos(T), y0 + VIEW_RADIUS*np.sin(T), '--w')
				plt.axis('square')
				plt.colorbar()
				plt.show()
				plt.figure()
				plt.pcolormesh(xI_bins, yI_bins, N.T, vmax=np.quantile(N, .999))
				plt.axis('square')
				plt.colorbar()
				plt.show()

			del(track_x)
			del(track_y)
			gc.collect()

			penumbral_kernel = np.zeros(XK.shape) # build the penumbral kernel
			for dx in [-dxK/3, 0, dxK/3]: # sampling over a few pixels
				for dy in [-dyK/3, 0, dyK/3]:
					penumbral_kernel += simple_penumbra(np.hypot(XK+dxK, YK+dyK), 0, Q, r0, r_img, 0, 1, *e_in_bounds)
			penumbral_kernel = penumbral_kernel/np.sum(penumbral_kernel)

			background = np.average(N,
				weights=(np.hypot(XI - x0, YI - y0) > (r_img + r0)/2)) # compute these with the better centering
			umbra = np.average(N,
				weights=(np.hypot(XI - x0, YI - y0) < r0/2))
			D = simple_penumbra(np.hypot(XI - x0, YI - y0), δ, Q, r0, r_img, background, umbra, *e_in_bounds) # make D equal to the rough fit to N

			penumbra_low = np.quantile(penumbral_kernel/penumbral_kernel.max(), .05)
			penumbra_hih = np.quantile(penumbral_kernel/penumbral_kernel.max(), .70)
			reach = signal.convolve2d(np.ones(XS.shape), penumbral_kernel, mode='full')
			data_bins = np.isfinite(N) & (reach/reach.max() > penumbra_low) & (reach/reach.max() < penumbra_hih) # exclude bins that are NaN and bins that are touched by all or none of the source pixels
			data_bins &= ~((N == 0) & (Delaunay(np.transpose([XI[N > 0], YI[N > 0]])).find_simplex(np.transpose([XI.ravel(), YI.ravel()])) == -1).reshape(N.shape)) # crop it at the convex hull where counts go to zero
			n_data_bins = np.sum(data_bins)
			N[np.logical_not(data_bins)] = np.nan

			B = np.full((n_pixs, n_pixs), 1/n_pixs**2) # note that B is currently normalized
			B[np.hypot(XS - (xS[0] + xS[-1])/2, YS - (yS[0] + yS[-1])/2) >= (xS[-1] - xS[0])/2 + (xS[1] - xS[0])] = 0 # remove the corners
			F = N - background

			χ2_95 = stats.chi2.ppf(.95, n_data_bins)
			χ2 = []
			iterations, remaining_decrease = 0, -np.inf
			print(np.sqrt(1/n_data_bins))
			while iterations < 50 and (remaining_decrease < 0 or remaining_decrease > np.sqrt(.01/n_data_bins)):
			# while iterations < 1 or ((χ2_prev - χ2)/n_data_bins > THRESHOLD and iterations < 50):
				B /= B.sum() # correct for roundoff
				s = convolve2d(B, penumbral_kernel, where=data_bins)
				G = np.sum(F*s/D, where=data_bins)/np.sum(s**2/D, where=data_bins)
				N_teo = G*s + background
				dLdN = (N - N_teo)/D
				δB = np.zeros(B.shape) # step direction
				for i, j in zip(*np.nonzero(data_bins)): # we need a for loop for this part because of memory constraints
					mt = max(0, i - penumbral_kernel.shape[0] + 1)
					mb = max(0, n_pixs - i - 1)
					ml = max(0, j - penumbral_kernel.shape[1] + 1)
					mr = max(0, n_pixs - j - 1)
					δB[mt:n_pixs-mb, ml:n_pixs-mr] += B[mt:n_pixs-mb, ml:n_pixs-mr]*dLdN[i,j]*penumbral_kernel[sl(i-mt, i-n_pixs+mb, -1), sl(j-ml, j-n_pixs+mr, -1)]
				δs = convolve2d(δB, penumbral_kernel, where=data_bins) # step projected into measurement space
				Fs, Fδ = np.sum(F*s/D, where=data_bins), np.sum(F*δs/D, where=data_bins)
				Ss, Sδ = np.sum(s**2/D, where=data_bins), np.sum(s*δs/D, where=data_bins)
				Dδ = np.sum(δs**2/D, where=data_bins)
				h = (Fδ - G*Sδ)/(G*Dδ - Fδ*Sδ/Ss)
				B += h/2*δB
				χ2.append(np.sum((N - N_teo)**2/D, where=data_bins)) # compute reduced chi-squared
				if len(χ2) >= 3: # estimate how much more it will decrease before bottoming out
					remaining_decrease = -(χ2[-2] - χ2[-1])**2/(2*χ2[-2] - χ2[-3] - χ2[-1])/n_data_bins
				iterations += 1
				if VERBOSE: print(f"[{χ2[-1]/n_data_bins}, {remaining_decrease}, {χ2[-1]/n_data_bins - remaining_decrease}],")
				if SHOW_DEBUG_PLOTS:
					fig, axes = plt.subplots(3, 2)
					fig.subplots_adjust(hspace=0, wspace=0)
					gs1 = gridspec.GridSpec(4, 4)
					gs1.update(wspace=0, hspace=0) # set the spacing between axes.
					axes[0,0].set_title("Previous step")
					plot = axes[0,0].pcolormesh(xS_bins, yS_bins, G*h/2*δB, cmap=cmap)
					axes[0,0].axis('square')
					fig.colorbar(plot, ax=axes[0,0])
					axes[0,1].set_title("Fit source image")
					plot = axes[0,1].pcolormesh(xS_bins, yS_bins, G*B, vmin=0, vmax=G*B.max(), cmap=cmap)
					axes[0,1].axis('square')
					fig.colorbar(plot, ax=axes[0,1])
					axes[1,0].set_title("Penumbral image")
					plot = axes[1,0].pcolormesh(xI_bins, yI_bins, N.T, vmin=0, vmax=N.max(where=data_bins, initial=0))
					axes[1,0].axis('square')
					fig.colorbar(plot, ax=axes[1,0])
					axes[1,1].set_title("Fit penumbral image")
					plot = axes[1,1].pcolormesh(xI_bins, yI_bins, N_teo.T, vmin=0, vmax=N.max(where=data_bins, initial=0))
					axes[1,1].axis('square')
					fig.colorbar(plot, ax=axes[1,1])
					axes[2,0].set_title("Expected variance")
					plot = axes[2,0].pcolormesh(xI_bins, yI_bins, D.T, vmin=0, vmax=N.max(where=data_bins, initial=0))
					axes[2,0].axis('square')
					fig.colorbar(plot, ax=axes[2,0])
					axes[2,1].set_title("Chi squared")
					plot = axes[2,1].pcolormesh(xI_bins, yI_bins, ((N - N_teo)**2/D).T, vmin=0, vmax=10)
					axes[2,1].axis('square')
					fig.colorbar(plot, ax=axes[2,1])
					plt.tight_layout()
					plt.show()

			if χ2[-1]/n_data_bins >= 2.0:
				print("Could not find adequate fit.")
				B = np.zeros(B.shape)
			else:
				B = G*np.maximum(0, B) # you can unnormalize now
				npargmaxB = np.unravel_index(B.argmax(), B.shape)
				print(f"σ = {np.sqrt(np.average(np.square(np.hypot(XS - XS[npargmaxB], YS - YS[npargmaxB])), weights=B)/2)/1e-4:.3f} μm")

			plt.figure()
			plt.pcolormesh((xS_bins - x0/M)/1e-4, (yS_bins + y0/M)/1e-4, B.T, cmap=cmap, vmin=0)
			plt.colorbar()
			plt.axis('square')
			plt.title("B(x, y) of TIM {} on shot {} with d ∈ [{:.1f}μm,{:.1f}μm)".format(scan[TIM], scan[SHOT], *d_bounds))
			plt.xlabel("x (μm)")
			plt.ylabel("y (μm)")
			plt.axis([-100, 100, -100, 100])
			plt.tight_layout()
			plt.savefig("results/{} TIM{} {:.1f}-{:.1f} {}h.png".format(scan[SHOT], scan[TIM], *d_bounds, etime))
			plt.show()

			# plt.show()

			image_layers.append(B/B.max())
			x_layers.append(XS)
			y_layers.append(YS)

		try:
			xray = np.loadtxt('scans/KoDI_xray_data1 - {:d}-TIM{:d}-{:d}.mat.csv'.format(int(scan[SHOT]), int(scan[TIM]), [2,4,5].index(int(scan[TIM]))+1), delimiter=',')
		except (ValueError, OSError):
			xray = None
		if xray is not None:
			plt.figure()
			# plt.pcolormesh(np.linspace(-300, 300, 3), np.linspace(-300, 300, 3), np.zeros((2, 2)), cmap=VIOLETS, vmin=0, vmax=1)
			plt.pcolormesh(np.linspace(-100, 100, 101), np.linspace(-100, 100, 101), xray, cmap=VIOLETS, vmin=0)
			plt.colorbar()
			plt.axis('square')
			plt.title("X-ray image of TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
			plt.xlabel("x (μm)")
			plt.ylabel("y (μm)")
			plt.axis([-100, 100, -100, 100])
			plt.tight_layout()
			plt.savefig("results/{} TIM{} xray sourceimage.png".format(scan[SHOT], scan[TIM]))
			plt.close()

		x0 = x_layers[0][np.unravel_index(np.argmax(image_layers[0]), image_layers[0].shape)]
		y0 = y_layers[0][np.unravel_index(np.argmax(image_layers[0]), image_layers[0].shape)]

		if len(image_layers) > 1:
			plt.figure()
			plt.contourf((x_layers[1] - x0)/1e-4, (y_layers[1] - y0)/1e-4, image_layers[1], levels=[0, 0.25, 1], colors=['#00000000', '#FF5555BB', '#000000FF'])
			plt.contourf((x_layers[2] - x0)/1e-4, (y_layers[2] - y0)/1e-4, image_layers[2], levels=[0, 0.25, 1], colors=['#00000000', '#55FF55BB', '#000000FF'])
			plt.contourf((x_layers[3] - x0)/1e-4, (y_layers[3] - y0)/1e-4, image_layers[3], levels=[0, 0.25, 1], colors=['#00000000', '#5555FFBB', '#000000FF'])
			# if xray is not None:
			# 	plt.contour(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), xray, levels=[.25], colors=['#550055BB'])
			plt.plot([0, x_off/1e-4], [0, y_off/1e-4], '-k')
			plt.scatter([x_off/1e-4], [y_off/1e-4], color='k')
			plt.arrow(0, 0, x_flo/1e-4, y_flo/1e-4, color='k', head_width=5, head_length=5, length_includes_head=True)
			plt.text(0.05, 0.95, "offset out of page = {:.3f}\nflow out of page = {:.3f}".format(z_off/r_off, z_flo/r_flo),
				verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)
			plt.axis('square')
			plt.axis([-100, 100, -100, 100])
			plt.xlabel("x (μm)")
			plt.ylabel("y (μm)")
			plt.title("TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
			plt.tight_layout()
			plt.savefig("results/{} TIM{} nestplot.png".format(scan[SHOT], scan[TIM]))
			plt.close()

		μ0 = np.sum(image_layers[0]) # image sum
		μx = np.sum(XS*image_layers[0])/μ0 # image centroid
		μy = np.sum(YS*image_layers[0])/μ0
		μxx = np.sum(XS**2*image_layers[0])/μ0 - μx**2 # image rotational inertia
		μxy = np.sum(XS*YS*image_layers[0])/μ0 - μx*μy
		μyy = np.sum(YS**2*image_layers[0])/μ0 - μy**2
		eigval, eigvec = np.linalg.eig([[μxx, μxy], [μxy, μyy]])
		i1, i2 = np.argmax(eigval), np.argmin(eigval)
		p0 = np.sqrt(μxx + μyy)
		p1, θ1 = np.hypot(μx, μy), np.arctan2(μy, μx)
		p2, θ2 = np.sqrt(eigval[i1]) - np.sqrt(eigval[i2]), np.arctan2(eigvec[1,i1], eigvec[0,i1])
		print(f"P0 = {p0/1e-4:.2f}μm")
		print(f"P1 = {p1/1e-4:.2f}μm = {p1/p0*100:.1f}%, θ = {np.degrees(θ1)}°")
		print(f"P2 = {p2/1e-4:.2f}μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2)}°")
