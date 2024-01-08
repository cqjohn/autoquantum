import numpy as np

path_to_file = "./data/DC/data.npz"
path_to_file = "./data/RF/0.05/data.npz"

data = np.load(path_to_file)

#%% RF measurements 

x, y, z, z2 = data['x'], data['y'], data['z'], data['z2']


#%% DC measurements 

x, y, z = data['x'], data['y'], data['z']
