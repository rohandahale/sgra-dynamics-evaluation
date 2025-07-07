######################################################################
# Author: Nick Conroy, Date: 04 April 2024
# A cleaned version of the script used in "Rotation in Event Horizon Telescope Movies", by Conroy et al. 2023
# Optional variables to adjust when running on new simulations: folder names (file_path, output_dir), xi_crit
# (which is the threshold between autocorrelation signal/noise), and the cylinder we analyze (e.g. qsn vs qs)
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
# import ehtim as eh
# import glob

import h5py
import sys
import scipy.ndimage as ndimage
import numpy.fft as fft

from utilities import *
colors, titles, labels, mfcs, mss = common()

############################## Chapter 1: Defining Functions and Paramters ##############################
######## Chapter 1.1: Defining Functions
### gnw colorbar implementation
def colorbar(mappable):
    """ the way matplotlib colorbar should have been implemented """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

### deals with exactly 0 edge cases
def myinterp(dat, x, y):
    i = int(round(x-0.5))
    j = int(round(y-0.5))
    ### bilinear:
    di = x-i
    dj = y-j
    z00 = dat[i,j]
    z01 = dat[i,j+1]
    z10 = dat[i+1,j]
    z11 = dat[i+1,j+1]
    idat = z00*(1.-di)*(1.-dj) + z01*(1.-di)*dj + z10*di*(1.-dj) + z11*di*dj
    return idat

def mean_subtract(m):
    for i in range(ntheta):
        m[:,i] = (m[:,i] - m[:,i].mean())
    for i in range(nftot):
        m[i,:] = (m[i,:] - m[i,:].mean())

############################## Chapter 2: Reading Files and calculating cylinder plots ##############################
######## Chapter 2.1: Defining File Paths and Image Parameters

file_path = sys.argv[1]
ring_file_path = sys.argv[2]
output_dir = sys.argv[3] + '_cylinder_output/'

if output_dir.endswith('_cylinder_output/'):
    pass
else:
    output_dir = output_dir + '_cylinder_output/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

hfp = h5py.File(file_path,'r')

scale = 1.0
N, M = hfp['I'].shape[1:]                       ## x,y-size of array

RADPERUAS = 4.848136811094136e-12 
GM_c3 = 20.46049 / 3600 # hours
M_PER_RAD = 41139576306.70914

dx = dy = float(hfp['header'].attrs['psize']) / RADPERUAS     ## x,y-size of pixel
times = np.array(hfp['times'])
nftot = len(times)
dt = np.diff(times).mean() / GM_c3
dt = round(dt, 1)

FOV_uas = dx * N
FOV_rad = FOV_uas * RADPERUAS
FOV_M = FOV_rad * M_PER_RAD

Tmax = times[-1] / GM_c3

######## Chapter 2.2: Reading Files

### Allocate space for 2-D Stokes I image for each file
Iall = np.transpose(hfp['I'], [1,2,0])

######## Chapter 2.3: Creating Unnormalized Cylinder Plots
print("Calculating Cylinder Plots...")

# print('Iall.shape', Iall.shape )
### Smooth (no need to smooth for reconstructed videos) 
sig = 20/(dx *(2*np.sqrt(2*np.log(2)))) ## We have 20 uas FWHM resolution. dx = uas/pixels. so 20/dx is FWHM in pixel units.
sIall = ndimage.gaussian_filter(Iall, sigma=(sig,sig,0))

### Make circle
ntheta = 180
theta = np.linspace(0.,2.*np.pi,ntheta,endpoint=False)
dtheta = 360/ntheta
pa = theta + 0.5*np.pi  ## position angle


= np.mean(Iall, axis = 2)
mean_sim = np.mean(sIall, axis = 2)

### Old analytical method for finding ring:
### OG model:
# y0= M/2 ## start with y centered.
# x0 = N/2
# y0= M/2 + 2*spin*np.sin(inclination * np.pi/180) *(N/FOV_M) ## Note: the final factor is a unit conversion. We want in pixel units
# r_pix = 1*np.sqrt(27)*(N/FOV_M) ## radius in pixel units
# icirc = y0 + r_pix*np.sin(pa)
# jcirc = x0 + r_pix*np.cos(pa)
# x = (jcirc - N/2.)*dx
# y = (icirc - M/2.)*dy
# print('plotting...')

### New VIDA fit method for finding ring:
### Get ring location in uas 
df = pd.read_csv(ring_file_path)

# r = np.mean(df['model_1_r0'])/RADPERUAS ## for eht cloud
# x0 = np.mean(df['model_1_x0'])/RADPERUAS
# y0 = np.mean(df['model_1_y0'])/RADPERUAS
r = np.mean(df['r0'])/RADPERUAS
x0 = np.mean(df['x0'])/RADPERUAS
y0 = np.mean(df['y0'])/RADPERUAS






### convert ring to pixels
r_pix = r * (N/FOV_uas)
x0 = x0 * (N/FOV_uas)
y0 = y0 * (N/FOV_uas)

"""
ishift = -y0                        ## should be right for VIDA automated output
jshift = -x0
icirc = ishift + r_pix*np.sin(pa) 
jcirc = jshift+ r_pix*np.cos(pa)
x = (icirc )*dx
y = (jcirc )*dy
"""
ishift = -y0  + M/2                      ## should be right for VIDA automated output
jshift = -x0  + N/2                      ## should be right for VIDA automated output
icirc = ishift + r_pix*np.sin(pa)
jcirc = jshift+ r_pix*np.cos(pa)
x = (icirc - N/2 )*dx
y = (jcirc - M/2 )*dy

# ishift = -x0 
# jshift = -y0 
# icirc = ishift + r_pix*np.sin(pa)
# jcirc = jshift + r_pix*np.cos(pa)
# x = (jcirc )*dx
# y = (icirc )*dy

if not os.path.exists(output_dir + 'Figure0_Mean_unsmoothed_ring.png'):
    ### Plot Mean Image
    fig = plt.figure(figsize = (5,5))
    ax = plt.subplot(111)
    ax.imshow(mean_im, cmap='afmhot', aspect = 'auto', interpolation='bilinear', origin='lower', extent=[-FOV_uas/2, FOV_uas/2, -FOV_uas/2, FOV_uas/2])
    # plt.title('Mean Image')
    plt.axis('scaled')
    plt.xlim(-FOV_uas/2, FOV_uas/2)
    plt.ylim(-FOV_uas/2, FOV_uas/2)
    plt.xlabel(r'$x\, [\mu{\rm as}]$')
    plt.ylabel(r'$y\, [\mu{\rm as}]$')
    plt.plot(x,y )
    #plt.colorbar()
    # print(plot_dir + model_path + 'Figure1_Mean_unsmoothed_ring.png')
    plt.savefig(output_dir + 'Figure0_Mean_unsmoothed_ring.png', bbox_inches = 'tight')
    # plt.show()
    plt.close(fig)
    # plt.clf()

if not os.path.exists(output_dir + 'Figure1_Mean_smoothed_ring.png'):
    ### Plot Mean smoothed Image
    fig = plt.figure(figsize = (5,5))
    ax = plt.subplot(111)
    ax.imshow(mean_sim, cmap='afmhot', aspect = 'auto', interpolation='bilinear', origin='lower', extent=[-FOV_uas/2, FOV_uas/2, -FOV_uas/2, FOV_uas/2])
    # plt.title('Mean Image')
    plt.axis('scaled')
    plt.xlim(-FOV_uas/2, FOV_uas/2)
    plt.ylim(-FOV_uas/2, FOV_uas/2)
    plt.xlabel(r'$x\, [\mu{\rm as}]$')
    plt.ylabel(r'$y\, [\mu{\rm as}]$')
    plt.plot(x,y )
    #plt.colorbar()
    plt.savefig(output_dir + 'Figure1_Mean_smoothed_ring.png', bbox_inches = 'tight')
    # plt.show()
    plt.close(fig)
    # plt.clf()

# for i in range(nftot):
#     ### Plot Mean Image
#     fig = plt.figure(figsize = (5,5))
#     ax = plt.subplot(111)
#     ax.imshow(Iall[:,:,i], cmap='afmhot', aspect = 'auto', interpolation='bilinear', origin='lower', extent=[-FOV_uas/2, FOV_uas/2, -FOV_uas/2, FOV_uas/2])
#     # plt.title('Mean Image')
#     plt.axis('scaled')
#     plt.xlim(-FOV_uas/2, FOV_uas/2)
#     plt.ylim(-FOV_uas/2, FOV_uas/2)
#     plt.xlabel(r'$x\, [\mu{\rm as}]$')
#     plt.ylabel(r'$y\, [\mu{\rm as}]$')
#     plt.plot(x,y )
#     #plt.colorbar()
#     # print(output_dir + 'Figure1_Mean_unsmoothed_ring.png')
#     plt.show()
#     # plt.close(fig)
#     # plt.clf()

### Create Unsmoothed Unnormalized Cylinder Plot
q = np.zeros((nftot,ntheta))
for k in range(nftot):
    for l in range(ntheta):
        q[k,l] = myinterp(Iall[:,:,k], icirc[l], jcirc[l])

### Save resulting Cylinder Plot
# np.save(output_dir + 'raw_cylinder.npy', q)

### Plot resulting Unsmoothed Unnormalized cylinder plot

if not os.path.exists(output_dir + 'Figure2_Cylinder_raw.png'):
    fig = plt.figure(figsize = (5,5))
    ax = plt.subplot(111)
    im = ax.imshow(q.T, cmap='afmhot', extent=[0, nftot*dt, 0, 360], aspect='auto', origin='lower' )
    ax.set_ylim(0, 360)
    ax.set_xlim(0, nftot*dt)
    ax.set_title('Raw Cylinder Plot')
    ax.set_xlabel(r'$t [G M/c^3]$')
    ax.set_ylabel(r'$\mathrm{PA} [{\rm deg}]$')
    colorbar(im)
    plt.savefig(output_dir + 'Figure2_Cylinder_raw.png', bbox_inches = 'tight')
    # plt.show()
    plt.close(fig)


### Create Smoothed Unnormalized Cylinder Plot
### get points on circle on smoothed images
qs = np.zeros((nftot,ntheta))
for k in range(nftot):
    for l in range(ntheta):
        qs[k,l] = myinterp(sIall[:,:,k], icirc[l], jcirc[l])

### Save resulting Smoothed Unnormalized cylinder plot
# np.save(output_dir + 'smoothed_cylinder.npy', qs)

### Plot resulting Smoothed Unnormalized cylinder plot
fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
im = ax.imshow(qs.T, cmap='afmhot', extent=[0, nftot*dt, 0, 360], aspect='auto', origin='lower')
ax.set_ylim(0, 360)
ax.set_xlim(0, nftot*dt)
ax.set_title('Smoothed Cylinder Plot')
ax.set_xlabel(r'$t [G M/c^3]$')
ax.set_ylabel(r'$\mathrm{PA} [{\rm deg}]$')
colorbar(im)
# plt.savefig(output_dir +'Cylinder_Smoothed.png', bbox_inches = 'tight')
# plt.show()
plt.close(fig)


######## Chapter 2.4: Creating Normalized Cylinder Plots
### Create Unsmoothed Normalized Cylinder Plot
qn =  np.copy(q)
# qn = np.log10(qn) ## note that log can produce nans if values in q are too small or negative
mean_subtract(qn)

### Save resulting Cylinder Plot
# np.save(output_dir + 'normalized_cylinder.npy', qn)

if not os.path.exists(output_dir + 'Figure4_Cylinder_Normalized.png'):
    ### Plot resulting Unsmoothed Normalized cylinder plot
    fig = plt.figure(figsize = (5,5))
    ax = plt.subplot(111)
    ax.imshow(qn.T, cmap='afmhot', extent=[0, nftot*dt, 0, 360], aspect='auto', origin='lower')
    ax.set_ylim(0, 360)
    ax.set_xlim(0, nftot*dt)
    ax.set_title('Normalized Cylinder Plot')
    ax.set_xlabel(r'$t [G M/c^3]$')
    ax.set_ylabel(r'$\mathrm{PA} [{\rm deg}]$')
    plt.savefig(output_dir + 'Figure4_Cylinder_Normalized.png', bbox_inches = 'tight')
    # plt.show()
    plt.close(fig)


### Create Smoothed Normalized Cylinder Plot
qsn =  np.copy(qs)
# qsn = np.log10(qsn) ## note that log can produce nans if values in qs are too small or negative
mean_subtract(qsn)

### Save resulting Cylinder plot
# np.save(output_dir + 'smoothed_normalized_cylinder.npy', qsn)

# print('qsn.shape', qsn.shape )
### plot resulting Smoothed Normalized Cylinder Plot
if not os.path.exists(output_dir + 'Figure5_Cylinder_Smoothed_Normalized.png'):
    fig = plt.figure(figsize = (5,5))
    ax = plt.subplot(111)
    im = ax.imshow(qsn.T, cmap='afmhot', extent=[0, nftot*dt, 0, 360], aspect='auto', origin='lower')
    ax.set_ylim(0, 360)
    ax.set_xlim(0, nftot*dt)
    ax.set_title('Smoothed Normalized Cylinder Plot')
    ax.set_xlabel(r'$t [G M/c^3]$')
    ax.set_ylabel(r'$\mathrm{PA} [{\rm deg}]$')
    plt.savefig(output_dir + 'Figure5_Cylinder_Smoothed_Normalized.png', bbox_inches = 'tight')
    # plt.show()
    plt.close(fig)

if not os.path.exists(output_dir + 'Figure6_Autocorrelation.png'):
    ############################## Chapter 3: Producing Autocorrelation Function ##############################
    ######## Chapter 3.1: Finding the Autocorrelation
    print("Calculating Autocorrelations...")
    ### find the Autocorrelation Function
    # qk = fft.fft2(qn)              ##  USE UNSMOOTHED, since output is already smoothed to EHT resolution? 
    qk = fft.fft2(qsn)           ## take fourier transform of smooothed normalized cylinder

    Pk = np.absolute(qk)**2        ## take square of the magnitude of the FT
    acf = np.real( fft.ifft2(Pk) ) ## reverse transform, take real component which drops ~0 imaginary component
    acf = acf/acf[0,0]             ## #normalize

    ### Shift so that the peak correlation is in the middle
    shifti = int(acf.shape[0]/2.)
    shiftj = int(acf.shape[1]/2.)
    racf = np.roll(acf, (shifti, shiftj), axis=(0,1))



    ######## Chapter 3.2: Normalizing the Autocorrelation
    racf /= np.max(racf) ## Normalize the Autocorrelation, so that the peak is 1
    racf_cut = np.copy(racf) ## define racf_cut, which we'll use for the Omega_p calculation



    ######## Chapter 3.3: Calculate the 2nd Moments
    ts = np.linspace(-len(racf)/2, len(racf)/2, len(racf), endpoint = False)
    phis = np.linspace(-len(racf[0])/2, len(racf[0])/2, len(racf[0]), endpoint = False)
    delta_t = ts[1] - ts[0]
    delta_phi = phis[1] - phis[0]

    moment_t = 0 ## initialize 1st and 2nd moments, then loop over range to calcualte them
    moment_phi = 0
    moment_t_phi = 0
    moment = 0

    ### Calculate xi_crit
    racf_std = np.std(racf)
    #xi_crit = 1*racf_std    ## ## start with 1 standard deviation for reconstructions. May need to be fine tuned once we have a larger sample of reconstructions 
    # xi_crit = 0.6*racf_std
    
    if 'truth' in file_path:
        xi_crit = 3*racf_std
    if 'modeling_mean' in file_path:
        xi_crit = 4.4*racf_std
    if 'resolve_mean' in file_path:
        xi_crit = 0.5*racf_std
    if 'doghit' in file_path:
        xi_crit = 1.3*racf_std
    if 'ehtim' in file_path:
        xi_crit = 0.8*racf_std
    if 'kine' in file_path:
        xi_crit = 0.3*racf_std
    if 'ngmem' in file_path:
        xi_crit = 0.2*racf_std

    ### Make sure no noise external to the central peak is included in the calculation. filter external noise using 'labels'
    from scipy.ndimage import label
    labels, num_features = label((racf > xi_crit).astype(int)) ## label every feature > cut
    Q = labels == labels[racf.shape[0] // 2, racf.shape[1] // 2] ## create mask for central region
    non_zero_columns = np.count_nonzero( np.sum(Q, axis=0) )

    ## add ceiling, if cut is too high or if not enough data:
    if (xi_crit > 1) or (non_zero_columns < 5):
        non_zero_columns = 5

        racf = racf.T ## these hdf5 files are transpose from what they should be. Some people... :P 

        ## Find the central index
        central_row_index = racf.shape[0] // 2
        central_column_index = (racf.shape[1]) // 2 

        ## Calculate the edge_value
        xi_crit = np.max(racf[:, int(central_column_index + non_zero_columns // 2) ])

        ## Create the boolean mask (Q) based on your requirements
        threshold_mask = (racf >= xi_crit).astype(int) & \
            (np.abs(np.arange(racf.shape[1]) - central_column_index) <= non_zero_columns // 2)

        ### use scipy ndimage 'label' function to distinguish central region > cut from external region > cut
        central_row_index = racf.shape[0] // 2
        central_column_index = racf.shape[1] // 2
        labels, num_features = label(threshold_mask) ## label every feature > cut
        central_peak_label = labels[central_row_index, central_column_index] ## Find label of the central peak feature

        Q = labels == central_peak_label ## create mask for central region

        racf = racf.T ## now we transpose back to the format that's expected
        Q = Q.T

    for i in range(len(ts)):
        for j in range(len(phis)):

            ### I would like to add code like the above, but instead of using 'racf[i,j] < xi_crit', I would like to use the 'Q' mask that I created above.
            if Q[i,j] == False:
                racf_cut[i,j] = 0.0        ## Produce a cropped RACF that's zeroed beneath the min correlation brightness
                continue
            moment += racf_cut[i,j]
            moment_t += racf_cut[i,j] * ts[i] * ts[i]
            moment_phi += racf_cut[i,j] * phis[j] * phis[j]
            moment_t_phi += racf_cut[i,j] * ts[i] * phis[j]

    non_zero_pixels = np.count_nonzero(racf_cut) ## save area of central peak (in pixel units)
    if non_zero_pixels < 20:
        print("Caution: calculating moments over fewer than 20 pixels! Consider setting a lower xi_crit threshold")

    ### Correct 2nd moment for units and normalize with 1st moment
    moment *= delta_t*delta_phi
    moment_t *= delta_t*delta_phi/moment
    moment_phi *= delta_t*delta_phi/moment
    moment_t_phi *= delta_t*delta_phi/moment

    ### Find the pattern speed from the moments
    pattern_speed = moment_t_phi/moment_t ## Omega_p = M_t_phi / M_t_t
    pattern_speed = pattern_speed*dtheta/dt ## adjust for units



    ######## Chapter 3.4: Plotting Slopes
    extent = [-(0.5*nftot + 0.5)*dt,(0.5*nftot - 0.5)*dt,-(0.5*ntheta + 0.5)*dtheta,(0.5*ntheta - 0.5)*dtheta]
    if (nftot % 2) != 0: ## for an odd numer of frames, we shift by half a pixel right so that central peak is at \Delta t = 0. 
            extent = extent + np.array([+dt/2, +dt/2, 0,0]) 

    fig = plt.figure(figsize = (7,7))
    ax = plt.subplot(111)
    im = ax.imshow(racf.T, cmap='afmhot', aspect = 'auto', origin = 'lower',
    # im = ax.imshow(racf, cmap='afmhot', aspect = 'auto', origin = 'lower',
        extent = extent,
        interpolation='bilinear')
    
    np.save(output_dir + 'autocorrelation.npy', racf)
    np.save(output_dir + 'autocorrelation_xlim.npy', [-len(racf[:,0])*dt/2., len(racf[:,0])*dt/2.])
    np.save(output_dir + 'autocorrelation_ylim.npy', [-len(racf[0,:])*dtheta/2, len(racf[0,:])*dtheta/2.])

    plt.contour(Q.T,extent=extent,origin='lower',levels=[0.5], c='purple')
    # plt.contour(Q,extent=extent,origin='lower',levels=[0.5], c='purple')
    # plt.contour((racf > xi_crit).T, extent=extent,origin='lower',levels=[0.5], c='purple')
    ax.set_xlabel(r'$\Delta t [G M/c^3]$')
    ax.set_ylabel(r'$\Delta \mathrm{PA} [{\rm deg}]$')
    ax.set_title('Autocorrelation', fontsize=22)
    ax.set_ylim(-len(racf[0,:])*dtheta/2, len(racf[0,:])*dtheta/2.)
    ax.set_xlim(-len(racf[:,0])*dt/2., len(racf[:,0])*dt/2.)
    #Plot a line with slope equal to the pattern speed
    x_vals = np.arange(0,5*nftot, 1)
    y_vals = pattern_speed * x_vals
    ax.plot(x_vals, y_vals, 'g--', lw=2, alpha=1.0, label='Measured Slope {0:0.2f} [deg / GMc^-3]'.format(pattern_speed))
    colorbar(im)
    ax.legend(loc='best',  bbox_to_anchor=(1.0, 1.2))
    plt.savefig(output_dir + 'Figure6_Autocorrelation.png', bbox_inches = 'tight')
    # plt.show()
    plt.close(fig)

    ############################################################################################################
    ############################## Chapter 4: Output ##############################
    ######## Chapter 4.1: Print or Save Results

    table_row = [file_path, pattern_speed]
    table_name = output_dir + 'pattern_speed.npy'
    np.save(table_name, table_row)

    print("################### Result ###########################")
    print('model:', file_path, "// Pattern Speed:\n{0:0.2f} deg/GMc^-3".format(pattern_speed))