######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import ehtplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pdb
import argparse
import os
import glob
#from scipy.stats import wasserstein_distance_nd
from utilities import *

colors, titles, labels, mfcs, mss = common()
plt.rcParams["xtick.direction"]="out"
plt.rcParams["ytick.direction"]="out"

obs = eh.obsdata.load_uvfits('/mnt/disks/shared/eht/sgra_dynamics_april11/DAR/submissions/SGRA_LO_onsky.uvfits')
time_array = np.linspace(10.9, 14.03, 100)

# Create a sample image for the background
kine='/mnt/disks/shared/eht/sgra_dynamics_april11/DAR/submissions/SGRA_LO_onsky_kine.hdf5'
resolve = '/mnt/disks/shared/eht/sgra_dynamics_april11/DAR/submissions/SGRA_LO+HI_onsky_resolve_mean.hdf5'
modeling = '/mnt/disks/shared/eht/sgra_dynamics_april11/DAR/submissions/SGRA_LO_onsky_modeling_mean.hdf5'

mvk = eh.movie.load_hdf5(kine)
mvr = eh.movie.load_hdf5(resolve)
mvm = eh.movie.load_hdf5(modeling)

datak = [mvk.get_image(t).regrid_image(160*eh.RADPERUAS,160).imarr() for t in time_array]
datar = [mvr.get_image(t).regrid_image(160*eh.RADPERUAS,160).imarr() for t in time_array]
datam = [mvm.get_image(t).regrid_image(160*eh.RADPERUAS,160).imarr() for t in time_array]

rotation_speed_k = -0.6/20.4  # Degrees per second
rotation_speed_r = -0.63/20.4  # Degrees per second
rotation_speed_m = -0.66/20.4  # Degrees per second

# Compute cumulative angles based on time differences
angles_k = np.cumsum(np.diff(time_array*3600, prepend=time_array[0]*3600) * rotation_speed_k)
angles_r = np.cumsum(np.diff(time_array*3600, prepend=time_array[0]*3600) * rotation_speed_r)
angles_m = np.cumsum(np.diff(time_array*3600, prepend=time_array[0]*3600) * rotation_speed_m)

vmin=np.min(datak)
vmax=np.max(datak)

def update(frame):
    # Clear the current figure
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    
    # Display the image
    ax[0].imshow(datak[frame], cmap='afmhot_us', extent=(-1, 1, -1, 1), vmin=vmin, vmax=vmax)
    ax[1].imshow(datar[frame], cmap='afmhot_us', extent=(-1, 1, -1, 1), vmin=vmin, vmax=vmax)
    ax[2].imshow(datam[frame], cmap='afmhot_us', extent=(-1, 1, -1, 1), vmin=vmin, vmax=vmax)
    
    # Calculate the angle of the line for the current frame
    angle_rad = np.radians(angles_k[frame])
    x =0.75*np.cos(angle_rad)
    y = 0.75*np.sin(angle_rad)
    ax[0].plot([0, x], [0, y], color='green', linewidth=2)
    
    angle_rad = np.radians(angles_r[frame])
    x =0.75*np.cos(angle_rad)
    y = 0.75*np.sin(angle_rad)    
    ax[1].plot([0, x], [0, y], color='green', linewidth=2)
    
    
    angle_rad = np.radians(angles_m[frame])
    x =0.75*np.cos(angle_rad)
    y = 0.75*np.sin(angle_rad)    
    ax[2].plot([0, x], [0, y], color='green', linewidth=2)
    
    # Set limits and turn off axes for better display
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title('kine')
    ax[0].axis('off')
    
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_title('resolve')
    ax[1].axis('off')
    
    ax[2].set_xlim(-1, 1)
    ax[2].set_ylim(-1, 1)
    ax[2].set_title('modeling')
    ax[2].axis('off')
    plt.suptitle(f'{time_array[frame]:.2f} UT')

# Create a figure and axis
def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
num_subplots=3
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(linear_interpolation(num_subplots, 2, 8, 7, 16),linear_interpolation(num_subplots, 2, 4, 7, 3)))
fig.subplots_adjust(hspace=0.01, wspace=0.05, top=0.8, bottom=0.01, left=0.01, right=0.8)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(time_array), interval=500)
plt.tight_layout()
output_file = "/mnt/disks/shared/eht/sgra_dynamics_april11/evaluation/scripts/sgra-dynamics-evaluation/rotating.gif"
ani.save(output_file, writer="ffmpeg", fps=10)
# Display the animation
#plt.show()