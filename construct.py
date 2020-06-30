import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas as pd

from simulations import load_shot, make_image
from perlin import perlin_generator


NOISE_SCALE = 1 # [cm]
DISPLACEMENT_NOISE = 0.25 # [cm*MeV]
EFFICIENCY_NOISE = 0#.25

SYNTH_RESOLUTION = 1600

M = 14
L = 4.21 # cm
rs = 60e-4 # cm
rA = 1000e-4 # cm

RADIAL_COORDINATE = np.linspace(0, .999*rA*1e-2)
RADIAL_DISPLACEMENT = 0.1e-2*np.log((1 + 1/(1 - RADIAL_COORDINATE/(rA*1e-2))**2)/(1 + 1/(1 + RADIAL_COORDINATE/(rA*1e-2))**2))
# from laplace import RADIAL_COORDINATE, RADIAL_DISPLACEMENT
# RADIAL_DISPLACEMENT *= 10e3 # 10 kV
plt.plot(RADIAL_COORDINATE, RADIAL_DISPLACEMENT)
plt.show()

FOLDER = 'scans/'

short_header = """\
TABULATION OF TRACKS in SCAN FILE O95520_PCIS_TIM2_3p1415_2hr_s1.cpsa
---------------------------------------------------------------------

SYSTEM: MIT-1   Scan program version: 2019-07-17; 14:00
     Objective = 40x; Pixel = 0.33120 x 0.33120 (µm); pixel aspect ratio = 1.0000;
SCAN: Filename = O95520_PCIS_TIM2_3p1415_2hr_s1.cpsa
     XY Limits:  (-3.4;  3.4)  ( 3.4; -3.4) [elliptical shape];  
     Thresholds = 85 (border); 75 (contrast); 40 (M=2);   
     Scanner = ED


Parameters:
     cn = 100. - (optical contrast in %); this is the peak contrast
     ca = 100. - (optical contrast in %); this is the average contrast
     e = eccentricity (in %), as defined in RSI 74, 975 (2003), Eq. (20)
-----------------------------------------
Limits imposed on tracks listed below:
  c-normal: between 0 and 50 %
  e: between 0 and 15 %
  d: no limits
  x: between -3.40 and  3.40 cm
  y between -3.40 and  3.40 cm
-----------------------------------------
  x(cm)     y(cm)    d(µm) cn(%) ca(%) e(%)
-----------------------------------------

"""


# for shot, N, SNR in [(95520, 1000000, 8), (95521, 1000000, 8), (95522, 300000, 4), (95523, 300000, 4), (95524, 300000, 4)]:
for shot, N, SNR in [('square', 1000000, 8)]:
	if type(shot) == int:
		t, (R, ρ, P, V, Te, Ti) = load_shot(shot)
		img_hi, x_bins, y_bins = make_image(t, R, ρ, Ti, [10, 15])
		img_md, x_bins, y_bins = make_image(t, R, ρ, Ti, [ 7, 10])
		img_lo, x_bins, y_bins = make_image(t, R, ρ, Ti, [2.2, 7])
		x_bins, y_bins = x_bins*1e-4, y_bins*1e-4 # convert to cm
		x, y = (x_bins[1:] + x_bins[:-1])/2, (y_bins[1:] + y_bins[:-1])/2
		X, Y = np.meshgrid(x, y)
	else:
		x = np.linspace(-200e-4, 200e-4, SYNTH_RESOLUTION)
		y = np.linspace(-200e-4, 200e-4, SYNTH_RESOLUTION)
		X, Y = np.meshgrid(x, y)
		img_lo = np.zeros(X.shape)
		img_md = np.zeros(X.shape)
		if shot == 'square':
			img_hi = np.where((np.absolute(X) < 100e-4) & (np.absolute(Y) < 100e-4), 1, 0)
		elif shot == 'eclipse':
			img_hi = np.where((np.hypot(X, Y) < 160e-4) & (np.hypot(X-30e-4, Y-15e-4) > 80e-4), 1, 0)
		elif shot == 'gaussian':
			img_hi = np.exp(-(X**2 + Y**2)/(2*100e-4**2))
		elif shot == 'hypergaussian':
			img_hi = np.exp(-(X**3 + Y**3)/(2*100e-4**3))
		elif shot == 'ellipse':
			img_hi = np.exp(-(X**2*2 + Y**2/2)/(2*100e-4**2))

	δx_noise, δy_noise = perlin_generator(-rA, rA, -rA, rA, NOISE_SCALE/M, DISPLACEMENT_NOISE), perlin_generator(-rA, rA, -rA, rA, NOISE_SCALE/M, DISPLACEMENT_NOISE)
	δε_noise = perlin_generator(-4, 4, -4, 4, NOISE_SCALE, EFFICIENCY_NOISE)

	xq, yq = np.meshgrid(np.linspace(-rA, rA, 12), np.linspace(-rA, rA, 12))
	xq, yq = xq[np.hypot(xq, yq) < rA], yq[np.hypot(xq, yq) < rA]
	plt.quiver(xq, yq, δx_noise(xq, yq)/6, δy_noise(xq, yq)/6)
	plt.axis('square')
	# plt.axis([-rA, rA, -rA, rA])
	plt.show()

	x_list = []
	y_list = []
	d_list = []
	for i in range(1):
		for img, diameter, energy in [(img_hi, 1, (10, 12)), (img_md, 2.5, (7, 10)), (img_lo, 5, (2, 7))]: # do each diameter bin separately
			if img.sum() == 0: # but skip synthetically empty bins
				continue

			random_index = np.random.choice(np.arange(len(X.ravel())), int(N*(np.sum(img)/np.sum(img_hi + img_md + img_lo))), p=img.ravel()/img.sum()) # sample from the distribution
			xS, yS = np.stack([X.ravel(), Y.ravel()], axis=1)[random_index].T

			r = L*np.arccos(1 - np.random.random(len(xS))*(1 - np.cos(rA/L))) # sample over the aperture
			θ = 2*np.pi*np.random.random(len(xS))
			xA = r*np.cos(θ)
			yA = r*np.sin(θ)

			E = np.random.uniform(*energy, len(xS))

			δr = np.interp(r, RADIAL_COORDINATE/1e-2, RADIAL_DISPLACEMENT/1e-2)/E
			δx = δx_noise(xA, yA)/E + δr*np.cos(θ)
			δy = δy_noise(xA, yA)/E + δr*np.sin(θ)

			xD = -(xA + (xA - xS)*M + δx) # make the image (this minus is from flipping from the TIM's perspective to looking at the CR_39)
			yD =   yA + (yA - yS)*M + δy

			rD = np.hypot(xD, yD)

			N_background = int(N*(np.sum(img)/np.sum(img_hi + img_md + img_lo))/SNR*7.8**2/(np.pi*rA**2*(M + 1)**2)) #  compute the desired background

			# x_list += list(xD[(np.hypot(xS, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])
			# y_list += list(yD[(np.hypot(xS, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])
			x_list += list(xD) + list(np.random.uniform(-3.4, 3.4, N_background))
			y_list += list(yD) + list(np.random.uniform(-3.4, 3.4, N_background))
			d_list += list(np.full(len(xS) + N_background, diameter)) # and add it in with the signal

	with open(FOLDER+'simulated shot {} TIM{}.txt'.format(shot, 2), 'w') as f:
		f.write(short_header)
		for i in range(len(x_list)):
			if np.random.random() < .8 + δε_noise(x_list[i], y_list[i]):
				f.write("{:.5f}  {:.5f}  {:.3f}  {:.0f}  {:.0f}  {:.0f}\n".format(x_list[i], -y_list[i], d_list[i], 1, 1, 1)) # note that cpsa y coordinates are inverted

	plt.figure()
	plt.hist2d(xS/1e-4, yS/1e-4, bins=(np.linspace(-300, 300, 51), np.linspace(-300, 300, 51)), cmap='plasma')
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.colorbar()
	plt.axis('square')
	plt.tight_layout()
	plt.savefig("simulated_shot_{}.png".format(shot))
	plt.close()
	# plt.show()
