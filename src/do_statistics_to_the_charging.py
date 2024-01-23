import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cm = 1e-2
μm = 1e-6
ns = 1e-9
kV = 1e3
Da = 1.66054e-27
e = 1.6e-19
MeV = 1e6*e
ɛ0 = 8.854e-12

M = 14
L = 4.21*cm
rA = 1000*μm


def r2_to_std(r2):
	return r2/np.sqrt(2*np.log(2))


# set it to work from the base directory regardless of whence we call the file
if os.path.basename(os.getcwd()) == "src":
	os.chdir(os.path.dirname(os.getcwd()))

data = pd.read_csv("images/summary.csv", dtype={'shot': str}) # start by reading the existing data or creating a new file
good = data.shot.str.contains("9552") & (data.energy_cut != 'xray')
energy_min_lookup_table = pd.Series(
	index=['lo', 'hi', 'xray', 'synth'],
	data=[2.2, 9, np.nan, np.nan])
energy_max_lookup_table = pd.Series(
	index=['lo', 'hi', 'xray', 'synth'],
	data=[6, 12.5, np.nan, np.nan])
energy_min = energy_min_lookup_table[data.energy_cut]
energy_max = energy_max_lookup_table[data.energy_cut]
energy_min.index = energy_max.index = data.index
data['energy (MeV)'] = (energy_min + energy_max)/2
MLeσdz4πɛ0 = data.Q*cm*MeV # in SI
data['TIM'] = [int(los[-1]) for los in data["LOS"]]
data['energy (J)'] = data['energy (MeV)']*MeV
data['time (s)'] = L/np.sqrt(2*data['energy (J)']/(2.014*Da))
data['time (ns)'] = data['time (s)']/ns
data['Er*dz (V)'] = MLeσdz4πɛ0/(M*L*e)*2
data['Er*dz (kV)'] = data['Er*dz (V)']/kV
data['σ*dz (C/m)'] = data['Er*dz (V)']*(2*np.pi*ɛ0)
data['σ (C/m^2)'] = data['σ*dz (C/m)']/(.02*cm)
data['Q (C/cm^2)'] = data['σ (C/m^2)']/cm**-2
data['M'] = M + 1
data['D_aperture (cm)'] = L/cm
data['D_detector (cm)'] = L*(M + 1)/cm
data['r_aperture (cm)'] = rA/cm
data['V (C/cm^2/MeV)'] = L/(rA*(M+1))*data['Q (C/cm^2)']*M/data['energy (MeV)']
data['Meff/M'] = 1 + (data.Q/energy_min*cm)/(rA*(M+1))
data['reff/r'] = 1 + (2.0*data.Q/energy_min*cm)/(r2_to_std(data.P0_magnitude)*μm*M)
data = data[good]

# print(pd.DataFrame(data={'my way': data['Meff/M']-1, "hans's way": data['V (C/cm^2/MeV)']*(cm**-2/MeV)/(4*np.pi*ɛ0/(e*.02*cm))}))

# select_columns = ['TIM', 'energy (MeV)', 'D_aperture (cm)', 'D_detector (cm)', 'M', 'r_aperture (cm)', 'Er*dz (kV)', 'V (C/cm^2/MeV)', 'Q (C/cm^2)']
select_columns = ['shot', 'TIM', 'energy (MeV)', 'Er*dz (kV)', 'Meff/M', 'reff/r']
output = pd.DataFrame(data={key: data[key] for key in select_columns})
print(output)

output.to_csv('images/charging_info.csv', index=False)

plt.scatter(x=data['energy (MeV)'], y=data['Q (C/cm^2)'], c=data['TIM'])
plt.show()
