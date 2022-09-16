import numpy as np
import os
import pandas as pd

for filename in os.listdir('.'):
	if filename.startswith('Hydrogen in ') and filename.endswith('.csv'):
		print(filename)
		target = filename[12:-4]
		table = pd.read_csv(filename, delimiter=r'\s\s+', engine='python', names=['E', 'electric', 'nuclear', 'range', 'straggling', 'stroggling'])
		values = np.empty((len(table), 2))
		for i, row in table.iterrows():
			if row.E.endswith('keV'):
				values[i,0] = float(row.E[:-4])
			elif row.E.endswith('MeV'):
				values[i,0] = float(row.E[:-4])*1e3
			else:
				raise Exception(f"whut is {row.E}")
			values[i,1] = row.electric + row.nuclear
		assert values[:,0].max() >= 15000
		with open(f"stopping_power_protons_{target}.csv", 'w') as f:
			for E, power in values:
				f.write(f"{E:7.1f},{power:10.5f}\n")
