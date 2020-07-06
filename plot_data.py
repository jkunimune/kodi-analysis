import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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


FOLDER = 'scans/'
SHOT = 'Shot number'
TIM = 'TIM'
APERTURE = 'Aperture Radius'
MAGNIFICATION = 'Magnification'
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

CR_39_RADIUS = 2.5 # cm
n_bins = 250

L = 4.21 # cm

def simple_penumbra(x, y, δ, Q, r0, minimum, maximum):
	rS = np.concatenate([np.linspace(0, r0*.9, 36)[:-1], r0*(1 - np.geomspace(.1, 1e-6, 36))])
	# rB = rS + Q*np.log((1 + rS/r0)**2/(1 - rS/r0)**2)
	rB = rS + Q*np.log((1 + 1/(1 - rS/r0)**2)/(1 + 1/(1 + rS/r0)**2))
	# rB = rS + Q*np.log((1 + 1/(1 - rS/(2*r0))**2)/(1 + 1/(1 + rS/(2*r0))**2))
	nB = 1/(np.gradient(rB, rS)*rB/rS)
	nB[rS > r0] = 0
	nB[0] = nB[1] # deal with this singularity
	if 4*δ/CR_39_RADIUS >= 1:
		return np.full(x.shape, np.nan)
	elif 4*δ/CR_39_RADIUS*n_bins >= 1:
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
	if Q < 0: return float('inf')
	teo = simple_penumbra(X - x0, Y - y0, δ, Q, r0, minimum, maximum)
	error = np.sum(teo - exp*np.log(teo))
	return error

shot_list = pd.read_csv('shot_list.csv')

xI_bins, yI_bins = np.linspace(-CR_39_RADIUS, CR_39_RADIUS, n_bins+1), np.linspace(-CR_39_RADIUS, CR_39_RADIUS, n_bins+1)
dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
xI, yI = (xI_bins[:-1] + xI_bins[1:])/2, (yI_bins[:-1] + yI_bins[1:])/2 # change these to bin centers
XI, YI = np.meshgrid(xI, yI)

for i, scan in shot_list.iterrows():
	filename = None
	for fname in os.listdir(FOLDER):
		if fname.endswith('.txt') and str(scan[SHOT]) in fname and 'TIM'+str(scan[TIM]) in fname:
			filename = fname
			break
	if filename is None:
		print("WARN: Could not find text file for TIM {} on shot {}".format(scan[TIM], scan[SHOT]))
		continue

	rA = scan[APERTURE]/1.e4 # cm
	M = scan[MAGNIFICATION] # cm
	track_list = pd.read_csv(FOLDER+filename, sep=r'\s+', header=20, skiprows=[24], encoding='Latin-1', dtype='float32')
	r0 = (M + 1)*rA

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
	hicontrast = (track_list['cn(%)'] < 35)
	track_list['x(cm)'] -= np.mean(track_list['x(cm)']) # do your best to center
	track_list['y(cm)'] -= np.mean(track_list['y(cm)'])

	# plt.hist2d(track_list['d(µm)'], track_list['cn(%)'], bins=(np.linspace(0, 10, 101), np.linspace(0, 50, 51)), cmap='magma_r')
	# plt.show()

	track_x, track_y = track_list['x(cm)'][hicontrast], track_list['y(cm)'][hicontrast]
	maximum = np.sum(np.hypot(track_x, track_y) < CR_39_RADIUS*.25)/\
			(np.pi*CR_39_RADIUS**2*.25**2)*dxI*dyI
	minimum = np.sum((np.hypot(track_x, track_y) < CR_39_RADIUS) & (np.hypot(track_x, track_y) > CR_39_RADIUS*0.95))/\
			(np.pi*CR_39_RADIUS**2*(1 - 0.95**2))*dxI*dyI
	exp = np.histogram2d(track_x, track_y, bins=(xI_bins, yI_bins))[0]
	opt = optimize.minimize(simple_fit, x0=[None]*4, args=(r0, minimum, maximum, XI, YI, exp),
		method='Nelder-Mead', options=dict(
			initial_simplex=[[.5, 0, .06, 1e-2], [-.5, .5, .06, 1e-2], [-.5, -.5, .06, 1e-2], [0, 0, .1, 1e-2], [0, 0, .06, 1.9e-2]]))
	x0, y0, δ, Q = opt.x

	print(opt)

	rS = np.linspace(0, r0*(1-1e-6), 216)
	displacement = Q*np.log((1 + 1/(1 - rS/r0)**2)/(1 + 1/(1 + rS/r0)**2))
	# displacement = Q*np.log((1 + 1/(1 - rS/r0/2)**2)/(1 + 1/(1 + rS/r0/2)**2))

	displacement = np.sum(rS*displacement**2)/np.sum(rS*displacement) # mean displacement [cm]
	integrated_field = 2*3e6*displacement/(L*M)
	print("{:.1f} kV".format(integrated_field/1e3))

	plt.figure()
	plt.hist(np.hypot(track_x - x0, track_y - y0), weights=1/np.hypot(track_x - x0, track_y - y0), bins=np.linspace(0, CR_39_RADIUS, 36), density=True)
	r = np.linspace(0, CR_39_RADIUS, 216)
	n = simple_penumbra(r, 0, δ, Q, r0, minimum, maximum)
	n /= np.sum(n*np.gradient(r))
	plt.plot(r, n)
	plt.xlabel("Radius (cm)")
	
	plt.figure()
	plt.pcolormesh(xI_bins, yI_bins, exp, vmin=0, vmax=np.quantile(exp[np.hypot(XI-x0, YI-y0) < rA*(M+1)], .999))
	# T = np.linspace(0, 2*np.pi, 361)
	# plt.plot(rA*(M+1)*np.cos(T) + x0, rA*(M+1)*np.sin(T) + y0, 'w--')
	plt.colorbar()
	plt.title("Penumbral image of TIM {} of shot {}".format(scan[TIM], scan[SHOT]))
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	plt.axis('square')
	plt.tight_layout()
	plt.show()
