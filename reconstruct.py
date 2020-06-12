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

CR_39_RADIUS = 2.1 # cm
n_MC = 1000000
n_bins = 250

M0 = 14
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

def simple_penumbra(x0, y0, δ, M, minimum, maximum):
	r = np.linspace(0, CR_39_RADIUS, n_bins)
	Pp = np.exp(-(r - (M+1)*rA)**2/(2*δ**2))
	l = np.clip(r/((M+1)*rA) - 1, 0.1, 1.9)
	dN = Pp/((1 - .22*l)*(np.pi/3*np.sqrt(1 - l**2)/np.arccos((l+1)/2))**1.4) # apply Frederick's circular correction factor
	N = 1 - np.cumsum(dN)/np.sum(dN)
	return minimum + (maximum-minimum)*np.interp(np.hypot(XI - x0, YI - y0), r, N, right=0)

def simple_fit(*args):
	if len(args[0]) == 3:
		(x0, y0, δ), M, minimum, maximum, exp = args
	else:
		(x0, y0, δ, M), minimum, maximum, exp = args
	if M >= (1 - 4/n_bins)*CR_39_RADIUS/rA - 1: return float('inf')
	teo = simple_penumbra(x0, y0, δ, M, minimum, maximum)
	error = np.sum(teo - exp*np.log(teo))
	penalty = 10*(M/M0 - np.log(M))
	return error + penalty


xI_bins, yI_bins = np.linspace(-CR_39_RADIUS, CR_39_RADIUS, n_bins+1), np.linspace(-CR_39_RADIUS, CR_39_RADIUS, n_bins+1)
dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2 # change these to bin centers
XI, YI = np.meshgrid(xI, yI)

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
	if re.fullmatch(r'[0-9]+', str(scan[SHOT])):
		track_list['ca(%)'] -= np.min(track_list['cn(%)']) # shift the contrasts down if they're weird
		track_list['cn(%)'] -= np.min(track_list['cn(%)'])
		track_list['d(µm)'] -= np.min(track_list['d(µm)']) # shift the diameters over if they're weird
	hicontrast = track_list['cn(%)'] < 35
	track_list['x(cm)'] -= np.mean(track_list['x(cm)']) # do you best to center
	track_list['y(cm)'] -= np.mean(track_list['y(cm)'])

	# plt.hist2d(track_list['d(µm)'], track_list['cn(%)'], bins=(np.linspace(0, 10, 101), np.linspace(0, 50, 51)))
	# plt.show()

	track_x, track_y = track_list['x(cm)'][hicontrast], track_list['y(cm)'][hicontrast]
	maximum = np.sum(np.hypot(track_x, track_y) < CR_39_RADIUS*.25)/\
			(np.pi*CR_39_RADIUS**2*.25**2)*dxI*dyI
	minimum = np.sum((np.hypot(track_x, track_y) < CR_39_RADIUS) & (np.hypot(track_x, track_y) > CR_39_RADIUS*0.95))/\
			(np.pi*CR_39_RADIUS**2*(1 - 0.95**2))*dxI*dyI
	exp = np.histogram2d(track_x, track_y, bins=(xI_bins, yI_bins))[0]
	opt = optimize.minimize(simple_fit, x0=[None]*4, args=(minimum, maximum, exp),
		method='Nelder-Mead', options=dict(
			initial_simplex=[[.5, 0, .06, M0], [-.5, .5, .06, M0], [-.5, -.5, .06, M0], [0, 0, .1, M0], [0, 0, .06, M0+1]]))
	x0, y0, _, M = opt.x
	r0 = (M + 1)*rA

	print(opt)
	plt.figure()
	plt.pcolormesh(xI_bins, yI_bins, exp)
	T = np.linspace(0, 2*np.pi, 361)
	plt.plot(rA*(M+1)*np.cos(T) + x0, rA*(M+1)*np.sin(T) + y0, 'w--')
	plt.colorbar()
	plt.axis('square')
	plt.tight_layout()
	plt.show()

	kernel_size = int(2*rA*(M+1)/dxI) + 4 if int(2*rA*(M+1)/dxI)%2 == 1 else int(2*rA*(M+1)/dxI) + 5
	xK_bins, yK_bins = np.linspace(-dxI*kernel_size/2, dxI*kernel_size/2, kernel_size+1), np.linspace(-dyI*kernel_size/2, dyI*kernel_size/2, kernel_size+1)
	XK, YK = np.meshgrid((xK_bins[:-1] + xK_bins[1:])/2, (yK_bins[:-1] + yK_bins[1:])/2)
	xS_bins, yS_bins = xI_bins[kernel_size//2:-(kernel_size//2)]/M, yI_bins[kernel_size//2:-(kernel_size//2)]/M
	dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
	xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
	XS, YS = np.meshgrid(xS, yS)

	n_pixs = n_bins - kernel_size + 1 # the source image will be smaller than the penumbral image
	assert xS.shape[0] == n_pixs
	img = np.empty((n_pixs, n_pixs, 3))

	for color, (cmap, d_bounds) in enumerate([(REDS, [3.0, 15]), (GREENS, [2.0, 3.0]), (BLUES, [0, 2.0]), (GREYS, [0, 15])]):
	# for color, (cmap, d_bounds) in enumerate([(GREYS, [3.0, 15]), (GREYS, [2.0, 3.0]), (GREYS, [0, 2.0]), (GREYS, [0, 15])]):
	# for color, (cmap, d_bounds) in enumerate([('plasma', [0, 40])]):
		print(d_bounds)

		track_x = track_list['x(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
		track_y = track_list['y(cm)'][hicontrast & (track_list['d(µm)'] >= d_bounds[0]) & (track_list['d(µm)'] < d_bounds[1])].to_numpy()
		background = np.sum(
			(np.hypot(track_x, track_y) < CR_39_RADIUS) &\
			(np.hypot(track_x, track_y) > CR_39_RADIUS*0.9))/\
			(np.pi*CR_39_RADIUS**2*(1 - 0.9**2))*dxI*dyI
		umbra = np.sum(
			(np.hypot(track_x, track_y) < CR_39_RADIUS*0.5))/\
			(np.pi*CR_39_RADIUS**2*(0.5**2))*dxI*dyI

		statistics = np.sum(
				(np.hypot(track_x - x0, track_y - y0) > r0) & (np.hypot(track_x - x0, track_y - y0) < CR_39_RADIUS)
		) - background/(dxI*dyI)*np.pi*(CR_39_RADIUS**2 - r0**2)
		if statistics == 0:
			print("No tracks found in this cut")
			continue
		print("Penumbra statistics: {:.1e}".format(statistics))

		N, xI_bins, yI_bins = np.histogram2d( # make a histogram
			track_x, track_y, bins=(xI_bins, yI_bins))

		del(track_x)
		del(track_y)
		gc.collect()

		xS_MC, yS_MC = np.random.uniform(-dxS/2, dxS/2, n_MC), np.random.uniform(-dyS/2, dyS/2, n_MC) # compute the transfer matrix using Monte-Carlo
		rA_MC, θA_MC = L*np.arccos(np.random.uniform(np.cos(rA/L), np.cos(.95*rA/L), n_MC)), np.random.uniform(0, 2*np.pi, n_MC)
		xA_MC, yA_MC = rA_MC*np.cos(θA_MC), rA_MC*np.sin(θA_MC)
		xI_MC = xA_MC + (xA_MC + xS_MC)*M
		yI_MC = yA_MC + (yA_MC - yS_MC)*M
		j, i = np.digitize(xI_MC, xK_bins) - 1, np.digitize(yI_MC, yK_bins) - 1 # subtract 1 to correct the bizzare behavior of digitize
		penumbral_kernel = np.histogram2d(i, j, bins=(np.arange(kernel_size+1), np.arange(kernel_size+1)))[0]
		penumbral_kernel[np.hypot(XK, YK) < rA*(M+1)-dxI] = n_MC*dxI*dyI/(np.pi*(rA*(M+1))**2)/(1 - .95**2)
		penumbral_kernel = np.minimum(penumbral_kernel, penumbral_kernel[kernel_size//2, kernel_size//2])
		penumbral_kernel = penumbral_kernel/np.sum(penumbral_kernel)

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
			B = np.flip(B.T, 0) # go from i~xI,j~yI to i~yS,j~xS (xI~xS, yI~-yS)

		elif METHOD == 'gelfgat':
			opt = optimize.minimize(simple_fit, x0=[None]*3, args=(M, background, umbra, N),
				method='Nelder-Mead', options=dict(
					initial_simplex=[[.5, 0, .06], [-.5, .5, .06], [-.5, -.5, .06], [0, 0, .1]]))
			D = simple_penumbra(*opt.x, M, background, umbra) # make D equal to a fit to N

			reach = signal.convolve2d(np.ones(XS.shape), penumbral_kernel, mode='full')
			data_bins = (reach > 0) & (reach < reach.max()) # exclude bins that are touched by all or none of the source pixels
			n_data_bins = np.sum(data_bins)

			B = np.full((n_pixs, n_pixs), 1/n_pixs**2) # note that B is currently normalized
			F = N - background

			# χ2_95 = stats.chi2.ppf(.95, n_data_bins)
			χ2, χ2_prev, iterations = np.inf, np.inf, 0
			# while iterations < 50 and χ2 > χ2_95:
			while iterations < 1 or ((χ2_prev - χ2)/χ2 > 1e-4 and iterations < 50):
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
				# plot = axes[0,1].pcolormesh(xS_bins, yS_bins, G*np.flip(B.T, 0), vmin=0, vmax=G*B.max(), cmap='plasma')
				# axes[0,1].axis('square')
				# fig.colorbar(plot, ax=axes[0,1])
				# axes[1,0].set_title("Penumbral image")
				# plot = axes[1,0].pcolormesh(xI_bins, yI_bins, N, vmin=0, vmax=N.max(where=data_bins, initial=0))
				# axes[1,0].axis('square')
				# fig.colorbar(plot, ax=axes[1,0])
				# axes[1,1].set_title("Fit penumbral image")
				# plot = axes[1,1].pcolormesh(xI_bins, yI_bins, N_teo, vmin=0, vmax=N.max(where=data_bins, initial=0))
				# axes[1,1].axis('square')
				# fig.colorbar(plot, ax=axes[1,1])
				# axes[2,0].set_title("Expected variance")
				# plot = axes[2,0].pcolormesh(xI_bins, yI_bins, D, vmin=0, vmax=N.max(where=data_bins, initial=0))
				# axes[2,0].axis('square')
				# fig.colorbar(plot, ax=axes[2,0])
				# axes[2,1].set_title("Chi squared")
				# plot = axes[2,1].pcolormesh(xI_bins, yI_bins, (N - N_teo)**2/D, vmin=0, vmax=10)
				# axes[2,1].axis('square')
				# fig.colorbar(plot, ax=axes[2,1])
				# plt.tight_layout()
				# plt.show()

			B = G*B # you can unnormalize now
			B = np.flip(B.T, 0) # go from i~xI,j~yI to i~yS,j~xS (i~-yI, j~xi)

		if color < 3:
			img[:,:,color] = B/B.max()

		# viridis = cm.get_cmap('viridis', 100)
		# plasma = cm.get_cmap('plasma', 100)
		# newcolors = np.concatenate([plasma(np.linspace(0, .5, 50)), viridis(np.linspace(.5, 1, 50))])
		# cmap = ListedColormap(newcolors)

		plt.figure()
		plt.pcolormesh(xS_bins/1e-4, yS_bins/1e-4, B, cmap=cmap, vmin=0)
		plt.colorbar()
		plt.axis('square')
		plt.title("B(x, y) of TIM {} on shot {} with d ∈ [{}μm,{}μm)".format(scan[TIM], scan[SHOT], *d_bounds))
		plt.xlabel("x (μm)")
		plt.ylabel("y (μm)")
		plt.axis([-300, 300, -300, 300])
		plt.tight_layout()
		plt.savefig("results/{}_TIM{}_{}-{}_sourceimage.png".format(scan[SHOT], scan[TIM], *d_bounds))
		plt.close()

		plt.show()

	try:
		xray = np.loadtxt('scans/KoDI_xray_data1 - {:d}-TIM{:d}-{:d}.mat.csv'.format(int(scan[SHOT]), int(scan[TIM]), [2,4,5].index(int(scan[TIM]))+1), delimiter=',')
	except (ValueError, OSError):
		xray = None
	if xray is not None:
		plt.figure()
		plt.pcolormesh(xS_bins/1e-4, yS_bins/1e-4, np.zeros(XS.shape), cmap=VIOLETS, vmin=0, vmax=1)
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

	xS -= np.average(XS, weights=img[:,:,2])
	yS -= np.average(YS, weights=img[:,:,2])

	# xC, yC = X.ravel()[np.argmax(img.sum(axis=2).ravel())], Y.ravel()[np.argmax(img.sum(axis=2).ravel())]
	plt.figure()
	plt.contourf(xS/1e-4, yS/1e-4, img[:,:,0], levels=[0, 0.25, 1], colors=['#00000000', '#FF5555BB', '#000000FF'])
	plt.contourf(xS/1e-4, yS/1e-4, img[:,:,1], levels=[0, 0.25, 1], colors=['#00000000', '#55FF55BB', '#000000FF'])
	plt.contourf(xS/1e-4, yS/1e-4, img[:,:,2], levels=[0, 0.25, 1], colors=['#00000000', '#5555FFBB', '#000000FF'])
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
