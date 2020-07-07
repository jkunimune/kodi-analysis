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
import gc
import re

from cmap import REDS, GREENS, BLUES, VIOLETS, GREYS

np.seterr('ignore')

SHOW_PLOTS = False
METHOD = 'gelfgat'

FOLDER = 'scans/'
SHOT = 'Shot number'
TIM = 'TIM'
APERTURE = 'Aperture Radius'
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

CR_39_RADIUS = 2.2 # cm
n_MC = 1000000
n_bins = 250

M = 14
L = 4.21 # cm
fill_centers = False

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

def simple_penumbra(x, y, δ, Q, r0, minimum, maximum):
	rN = np.concatenate([np.linspace(0, r0 - Q*1e1, 36)[:-1], r0 - Q*np.geomspace(1e1, 1e-4, 36)])
	rB = rN + Q*np.log((1 + 1/(1 - rN/r0)**2)/(1 + 1/(1 + rN/r0)**2))
	nB = 1/(np.gradient(rB, rN)*rB/rN) # I have the closed form for this derivative, but it takes too long to write
	nB[0] = nB[1] # deal with this singularity

	if 4*δ/CR_39_RADIUS*n_bins >= 1:
		r_kernel = np.linspace(-4*δ, 4*δ, int(4*δ/CR_39_RADIUS*n_bins)*2+1)
		n_kernel = np.exp(-r_kernel**2/δ**2)
		r_point = np.arange(-4*δ, CR_39_RADIUS, r_kernel[1] - r_kernel[0])
		n_point = np.interp(r_point, rB, nB, right=0)
		penumbra = np.convolve(n_point, n_kernel, mode='same')
	else:
		r_point = np.linspace(0, CR_39_RADIUS, n_bins)
		penumbra = np.interp(r_point, rB, nB, right=0)
	return minimum + (maximum-minimum)*np.interp(np.hypot(x, y), r_point, penumbra/np.max(penumbra), right=0)

def simple_fit(*args):
	if len(args[0]) == 3:
		(x0, y0, δ), Q, r0, minimum, maximum, X, Y, exp = args
	else:
		(x0, y0, δ, Q), r0, minimum, maximum, X, Y, exp = args
	if Q < 0 or 10*Q >= r0: return float('inf')
	if minimum is None or maximum is None:
		minimum = np.average(exp, weights=np.hypot(X - x0, Y - y0) > .95*CR_39_RADIUS)
		maximum = np.average(exp, weights=np.hypot(X - x0, Y - y0) < .25*CR_39_RADIUS)
	teo = simple_penumbra(X - x0, Y - y0, δ, Q, r0, minimum, maximum)
	error = np.sum(teo - exp*np.log(teo))
	return error


xI_bins_0, yI_bins_0 = np.linspace(-CR_39_RADIUS, CR_39_RADIUS, n_bins+1), np.linspace(-CR_39_RADIUS, CR_39_RADIUS, n_bins+1)
dxI, dyI = xI_bins_0[1] - xI_bins_0[0], yI_bins_0[1] - yI_bins_0[0]
xI_0, yI_0 = (xI_bins_0[:-1] + xI_bins_0[1:])/2, (yI_bins_0[:-1] + yI_bins_0[1:])/2 # change these to bin centers
XI_0, YI_0 = np.meshgrid(xI_0, yI_0, indexing='ij')

shot_list = pd.read_csv('shot_list.csv')

for i, scan in shot_list.iterrows():
	filename = None
	for fname in os.listdir(FOLDER):
		if fname.endswith('.txt') and str(scan[SHOT]) in fname and 'TIM'+str(scan[TIM]) in fname:
			filename = fname
			print("INFO: Beginning reconstruction for TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
			break
	if filename is None:
		print("WARN: Could not find text file for TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
		continue

	rA = scan[APERTURE]/1.e4 # cm
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
	if str(scan[SHOT]) == '95519' or str(scan[SHOT]) == '95520': # these shots were tilted for some reason
		x_temp, y_temp = track_list['x(cm)'].copy(), track_list['y(cm)'].copy() # rotate the flipped penumbral image 45 degrees clockwise
		track_list['x(cm)'] =  np.sqrt(2)/2*x_temp + np.sqrt(2)/2*y_temp
		track_list['y(cm)'] = -np.sqrt(2)/2*x_temp + np.sqrt(2)/2*y_temp
	if re.fullmatch(r'[0-9]+', str(scan[SHOT])): # adjustments for real data:
		track_list['ca(%)'] -= np.min(track_list['cn(%)']) # shift the contrasts down if they're weird
		track_list['cn(%)'] -= np.min(track_list['cn(%)'])
		track_list['d(µm)'] -= np.min(track_list['d(µm)']) # shift the diameters over if they're weird
	hicontrast = track_list['cn(%)'] < 35
	track_list['x(cm)'] -= np.mean(track_list['x(cm)'][hicontrast]) # do your best to center
	track_list['y(cm)'] -= np.mean(track_list['y(cm)'][hicontrast])

	# plt.hist2d(track_list['d(µm)'], track_list['cn(%)'], bins=(np.linspace(0, 10, 101), np.linspace(0, 50, 51)))
	# plt.show()

	r0 = (M + 1)*rA
	kernel_size = int(2*(r0+.2)/dxI) + 4 if int(2*(r0+.2)/dxI)%2 == 1 else int(2*(r0+.2)/dxI) + 5
	n_pixs = n_bins - kernel_size + 1 # the source image will be smaller than the penumbral image
	
	xK_bins, yK_bins = np.linspace(-dxI*kernel_size/2, dxI*kernel_size/2, kernel_size+1), np.linspace(-dyI*kernel_size/2, dyI*kernel_size/2, kernel_size+1)
	dxK, dyK = xK_bins[1] - xK_bins[0], yK_bins[1] - yK_bins[0]
	XK, YK = np.meshgrid((xK_bins[:-1] + xK_bins[1:])/2, (yK_bins[:-1] + yK_bins[1:])/2, indexing='ij')

	image_layers, x_layers, y_layers = [], [], []

	if np.std(track_list['d(µm)']) == 0:
		cuts = [('plasma', [0, 40])]
	else:
		cuts = [(REDS, [3.0, 15]), (GREENS, [2.0, 3.0]), (BLUES, [0, 2.0]), (GREYS, [0, 15])] # [(GREYS, [3.0, 15]), (GREYS, [2.0, 3.0]), (GREYS, [0, 2.0]), (GREYS, [0, 15])]

	for color, (cmap, d_bounds) in enumerate(cuts):
		print(d_bounds)

		track_x = track_list['x(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
		track_y = track_list['y(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
		if len(track_x) <= 0:
			print("No tracks found in this cut.")
			continue

		N, xI_bins_0, yI_bins_0 = np.histogram2d( # make a histogram
			track_x, track_y, bins=(xI_bins_0, yI_bins_0))
		assert N.shape == XI_0.shape

		opt = optimize.minimize(simple_fit, x0=[None]*4, args=(r0, None, None, XI_0, YI_0, N),
			method='Nelder-Mead', options=dict(
				initial_simplex=[[.5, 0, .06, .01], [-.5, .5, .06, .01], [-.5, -.5, .06, .01], [0, 0, .1, .01], [0, 0, .06, .019]]))
		x0, y0, δ, Q = opt.x
		print(opt)

		background = np.sum( # recompute these, but with better centering
			(np.hypot(track_x - x0, track_y - y0) < CR_39_RADIUS) &\
			(np.hypot(track_x - x0, track_y - y0) > CR_39_RADIUS*0.95))/\
			(np.pi*CR_39_RADIUS**2*(1 - 0.95**2))*dxI*dyI
		umbra = np.sum(
			(np.hypot(track_x - x0, track_y - y0) < CR_39_RADIUS*0.25))/\
			(np.pi*CR_39_RADIUS**2*(0.25**2))*dxI*dyI

		xI_bins, yI_bins = xI_bins_0 + x0, yI_bins_0 + y0
		xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2
		XI, YI = np.meshgrid(xI, yI, indexing='ij')

		xS_bins, yS_bins = xI_bins[kernel_size//2:-(kernel_size//2)]/M, -yI_bins[kernel_size//2:-(kernel_size//2)]/M
		dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
		xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
		XS, YS = np.meshgrid(xS, yS, indexing='ij')

		N, xI_bins, yI_bins = np.histogram2d( # make a histogram
			track_x, track_y, bins=(xI_bins, yI_bins))

		del(track_x)
		del(track_y)
		gc.collect()

		if SHOW_PLOTS:
			plt.figure()
			plt.pcolormesh(xI_bins, yI_bins, N.T, vmax=np.quantile(N, .99))
			T = np.linspace(0, 2*np.pi)
			plt.plot(x0 + r0*np.cos(T), y0 + r0*np.sin(T), '--w')
			plt.axis('square')
			plt.colorbar()
			plt.show()

		penumbral_kernel = np.zeros(XK.shape)
		for dx in [-dxK/3, 0, dxK/3]:
			for dy in [-dyK/3, 0, dyK/3]:
				penumbral_kernel += simple_penumbra(XK+dxK, YK+dyK, 0, Q, r0, 0, 1)
		penumbral_kernel = penumbral_kernel/np.sum(penumbral_kernel)
		# plt.pcolormesh(xK_bins, yK_bins, penumbral_kernel)
		# plt.show()

		if METHOD == 'quasinewton':
			N[np.hypot(XI, YI) > CR_39_RADIUS] = background
			n_data = np.sum(N - background)
			exp = N.ravel()
			lagrange = 0.1*n_pixs**2/n_data
			lim = np.sum(exp)/n_pixs**2/6 # put a threshold on the entropy to avoid infinite derivatives (not sure why LBFGSB can't use the curvature condition to stay away from those)
			def posterior(*tru):
				tru = np.reshape(tru, (n_pixs, n_pixs))
				teo = signal.convolve2d(tru, penumbral_kernel, mode='full') + background
				teo = teo.ravel()
				tru = tru.ravel()
				entropy = np.where(tru >= lim,
					lagrange*tru*np.log(tru/n_data),
					lagrange*lim*np.log(lim/n_data) + (tru - lim)*lagrange*(1 + np.log(lim/n_data))) # log prior, split up by input element
				error = teo - exp*np.log(teo) # log likelihood, split up by output element
				error[(teo == 0) & (exp == 0)] = 0
				error[(teo == 0) & (exp != 0)] = np.inf
				assert not np.isnan(np.sum(entropy) + np.sum(error))
				return np.sum(entropy) + np.sum(error)
			def grad_posterior(*tru):
				tru = np.reshape(tru, (n_pixs, n_pixs))
				teo = signal.convolve2d(tru, penumbral_kernel, mode='full') + background
				teo = teo.ravel()
				tru = tru.ravel()
				grad = np.empty(n_pixs**2)
				for i in range(n_pixs):
					for j in range(n_pixs): # we need a for loop for this part because of memory constraints
						k = n_pixs*i + j
						dteo_dtru = np.zeros((n_bins, n_bins)) # derivative of teo image by this particular source pixel
						paste(dteo_dtru, penumbral_kernel, (i, j))
						dteo_dtru = dteo_dtru.ravel()
						dentropy = max(
							lagrange*(1 + np.log(tru[k]/n_data)),
							lagrange*(1 + np.log(lim/n_data))) # derivative of log prior, for just this input element
						derror = (1 - exp/teo)*dteo_dtru # derivative log likelihood, split up by output element
						derror[(teo == 0) & (exp == 0)] = 1 - dteo_dtru[(teo == 0) & (exp == 0)]
						derror[(teo == 0) & (exp != 0)] = np.inf
						grad[k] = dentropy + np.sum(derror)
						assert not np.isnan(grad[k])
				return grad
			opt = optimize.minimize(posterior, jac=grad_posterior,
				x0=np.full(n_pixs**2, np.sum(exp)/n_pixs**2),
				bounds=np.stack([np.full(n_pixs**2, 0), np.full(n_pixs**2, np.inf)], axis=1),
				method='L-BFGS-B', options=dict(
					ftol=1e-9,
					# iprint=1,
				)
			)
			print(opt)
			B = opt.x.reshape((n_pixs, n_pixs))
			B = B.T # go from i~xI,j~yI to i~yS,j~xS (xI~xS, yI~-yS) (but also yS is already negated)

		elif METHOD == 'gelfgat':
			D = simple_penumbra(XI - x0, YI - y0, δ, Q, r0, background, umbra) # make D equal to the fit to N

			reach = signal.convolve2d(np.ones(XS.shape), penumbral_kernel, mode='full')
			data_bins = (reach > .001) & (reach < .999*reach.max()) # exclude bins that are touched by all or none of the source pixels
			n_data_bins = np.sum(data_bins)
			N[np.logical_not(data_bins)] = np.nan

			B = np.full((n_pixs, n_pixs), 1/n_pixs**2) # note that B is currently normalized
			F = N - background

			# χ2_95 = stats.chi2.ppf(.95, n_data_bins)
			χ2, χ2_prev, iterations = np.inf, np.inf, 0
			# while iterations < 50 and χ2 > χ2_95:
			while iterations < 1 or ((χ2_prev - χ2)/n_data_bins > 1e-4 and iterations < 50):
				B /= B.sum() # correct for roundoff
				s = signal.convolve2d(B, penumbral_kernel, mode='full')
				G = np.sum(F*s/D, where=data_bins)/np.sum(s**2/D, where=data_bins)
				N_teo = G*s + background
				δB = np.empty(B.shape) # step direction
				for i in range(n_pixs):
					for j in range(n_pixs): # we need a for loop for this part because of memory constraints
						dsdB = np.zeros((n_bins, n_bins)) # derivative of teo image by this particular source pixel
						paste(dsdB, penumbral_kernel, (i, j))
						δB[i,j] = B[i,j]*np.sum(dsdB*(N - N_teo)/D, where=data_bins) # derivative of L with respect to Bij
				δs = signal.convolve2d(δB, penumbral_kernel, mode='full') # step projected into measurement space
				Fs, Fδ = np.sum(F*s/D, where=data_bins), np.sum(F*δs/D, where=data_bins)
				Ss, Sδ = np.sum(s**2/D, where=data_bins), np.sum(s*δs/D, where=data_bins)
				Dδ = np.sum(δs**2/D, where=data_bins)
				h = (Fδ - G*Sδ)/(G*Dδ - Fδ*Sδ/Ss)
				B += h/2*δB
				χ2_prev, χ2 = χ2, np.sum((N - N_teo)**2/D, where=data_bins)
				iterations += 1
				print("[{}],".format(χ2/n_data_bins))
				# fig, axes = plt.subplots(3, 2)
				# gs1 = gridspec.GridSpec(4, 4)
				# gs1.update(wspace=0, hspace=0) # set the spacing between axes. 
				# axes[0,0].axis('off')
				# axes[0,1].set_title("Fit source image")
				# plot = axes[0,1].pcolormesh(xS_bins, yS_bins, G*np.flip(B.T, 0).T, vmin=0, vmax=G*B.max(), cmap='plasma')
				# axes[0,1].axis('square')
				# fig.colorbar(plot, ax=axes[0,1])
				# axes[1,0].set_title("Penumbral image")
				# plot = axes[1,0].pcolormesh(xI_bins, yI_bins, N.T, vmin=0, vmax=N.max(where=data_bins, initial=0))
				# axes[1,0].axis('square')
				# fig.colorbar(plot, ax=axes[1,0])
				# axes[1,1].set_title("Fit penumbral image")
				# plot = axes[1,1].pcolormesh(xI_bins, yI_bins, N_teo.T, vmin=0, vmax=N.max(where=data_bins, initial=0))
				# axes[1,1].axis('square')
				# fig.colorbar(plot, ax=axes[1,1])
				# axes[2,0].set_title("Expected variance")
				# plot = axes[2,0].pcolormesh(xI_bins, yI_bins, D.T, vmin=0, vmax=N.max(where=data_bins, initial=0))
				# axes[2,0].axis('square')
				# fig.colorbar(plot, ax=axes[2,0])
				# axes[2,1].set_title("Chi squared")
				# plot = axes[2,1].pcolormesh(xI_bins, yI_bins, ((N - N_teo)**2/D).T, vmin=0, vmax=10)
				# axes[2,1].axis('square')
				# fig.colorbar(plot, ax=axes[2,1])
				# plt.tight_layout()
				# plt.show()

			B = G*B # you can unnormalize now
			B = B.T # go from i~xI,j~yI to i~yS,j~xS (xI~xS, yI~-yS) (but also yS is already negated)

		plt.figure()
		plt.pcolormesh((xS_bins - x0/M)/1e-4, (yS_bins + y0/M)/1e-4, B, cmap=cmap, vmin=0)
		plt.colorbar()
		plt.axis('square')
		plt.title("B(x, y) of TIM {} on shot {} with d ∈ [{}μm,{}μm)".format(scan[TIM], scan[SHOT], *d_bounds))
		plt.xlabel("x (μm)")
		plt.ylabel("y (μm)")
		plt.axis([-300, 300, -300, 300])
		plt.tight_layout()
		plt.savefig("results/{}_TIM{}_{}-{}_sourceimage.png".format(scan[SHOT], scan[TIM], *d_bounds))
		plt.close()

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
		plt.pcolormesh((xS_bins - x0/M)/1e-4, (yS_bins + y0/M)/1e-4, np.zeros(XS.shape).T, cmap=VIOLETS, vmin=0, vmax=1)
		plt.pcolormesh(np.linspace(-100, 100, 101), np.linspace(-100, 100, 101), xray, cmap=VIOLETS, vmin=0)
		plt.colorbar()
		plt.axis('square')
		plt.title("X-ray image of TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
		plt.xlabel("x (μm)")
		plt.ylabel("y (μm)")
		plt.axis([-300, 300, -300, 300])
		plt.tight_layout()
		plt.savefig("results/{}_TIM{}_xray_sourceimage.png".format(scan[SHOT], scan[TIM]))
		plt.close()

	x0 = np.average(x_layers[-1], weights=image_layers[-1])
	y0 = np.average(y_layers[-1], weights=image_layers[-1])

	if len(image_layers) > 1:
		plt.figure()
		plt.contourf((x_layers[0] - x0)/1e-4, (y_layers[0] - y0)/1e-4, image_layers[0], levels=[0, 0.25, 1], colors=['#00000000', '#FF5555BB', '#000000FF'])
		plt.contourf((x_layers[0] - x0)/1e-4, (y_layers[1] - y0)/1e-4, image_layers[1], levels=[0, 0.25, 1], colors=['#00000000', '#55FF55BB', '#000000FF'])
		plt.contourf((x_layers[0] - x0)/1e-4, (y_layers[2] - y0)/1e-4, image_layers[2], levels=[0, 0.25, 1], colors=['#00000000', '#5555FFBB', '#000000FF'])
		if xray is not None:
			# plt.contourf(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), xray, levels=[0, .25, 1], colors=['#00000000', '#550055BB', '#000000FF'])
			plt.contour(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), xray, levels=[.25], colors=['#550055BB'])
		plt.plot([0, x_off/1e-4], [0, y_off/1e-4], '-k')
		plt.scatter([x_off/1e-4], [y_off/1e-4], color='k')
		plt.arrow(0, 0, 2*x_flo/1e-4, 2*y_flo/1e-4, color='k', head_width=15, head_length=15, length_includes_head=True)
		plt.text(0.05, 0.95, "offset out of page = {:.3f}\nflow out of page = {:.3f}".format(z_off/r_off, z_flo/r_flo),
			verticalalignment='top', transform=plt.gca().transAxes, fontsize=12)
		plt.axis('square')
		plt.axis([-300, 300, -300, 300])
		plt.xlabel("x (μm)")
		plt.ylabel("y (μm)")
		plt.title("TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
		plt.tight_layout()
		plt.savefig("results/{}_TIM{}_nestplot.png".format(scan[SHOT], scan[TIM]))
		plt.close()
