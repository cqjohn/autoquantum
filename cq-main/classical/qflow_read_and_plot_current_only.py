import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from classical_algorithm import locate_first_point, locate_nearby_max_point

# Define the folder path and the name of the NumPy file to load
folder_path = './'
npy_file = "20171118-123151236063.npy"
filepath = os.path.join(folder_path, npy_file)

# Load the NumPy file into a variable named 'dat'
dat = np.load(filepath, allow_pickle=True).item() 

# Extract voltage vectors and create a mesh grid
V_P1 = -dat['V_P1_vec']
V_P2 = -dat['V_P2_vec']
X,Y=np.meshgrid(V_P1,V_P2)

# Define the size of the current vector grid
N_v = 100

# Extract the current vector and reshape it into a 2D array
current_vec = np.array([x['current'] for x in dat['output']]).reshape(N_v,N_v)   #what exactly is this variable
print(current_vec)

# Print values at specific corners of the current vector array
print(current_vec[0][0])
print(current_vec[0][N_v-1])
print(current_vec[N_v-1][N_v-1])

# print(current_vec)

# Find the first point with a value greater than a specified threshold
min_coords, min_dist_from_origin, value_at_min = locate_first_point(current_vec, 0.00001, N_v, N_v)

# Find the coordinates and value of the nearby maximum point
min_coords_adjusted, value_at_min_adjusted = locate_nearby_max_point(current_vec, 20, N_v, N_v, min_coords[0], min_coords[1])

# Print information about the first point and the nearby maximum point
print(min_coords, min_dist_from_origin, value_at_min)
print(min_coords_adjusted, value_at_min_adjusted)

matplotlib.rcParams.update({'font.size': 12})

# Create a Matplotlib figure and axis for plotting
fig, ax = plt.subplots(figsize=(11,10))
fig.tight_layout(w_pad=7.0, h_pad=6.0)

# Set y-axis ticks
plt.yticks(np.arange(0.0, 0.5, 0.1))

# Create a pseudocolor plot of the current vector on the specified mesh grid
cd = ax.pcolor(X,Y,current_vec,vmax=1e-4,cmap=cm.summer)

# Set plot title and y-axis ticks
ax.set_title('Current data')
ax.set_yticks(np.arange(0.0,0.5,0.1))

# Add colorbar to the plot
fig.colorbar(cd, ax = ax, fraction=0.045)

# Plot red and blue points at the coordinates of the minimum point and the nearby maximum point, respectively
ax.scatter(V_P1[min_coords[0]], V_P2[min_coords[1]], color='red')
ax.scatter(V_P1[min_coords_adjusted[0]], V_P2[min_coords_adjusted[1]], color='blue')

plt.show()
