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
from matplotlib.cm import ScalarMappable
import pdb
import argparse
import os
import glob
from utilities import *
colors, titles, labels, mfcs, mss = common()
plt.rcParams["xtick.direction"]="out"
plt.rcParams["ytick.direction"]="out"

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--starmv', type=str, default='none', help='path of starwarps .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('--modelingmv',  type=str, default='none', help='path of modeling .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./gif.gif', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

outpath = args.outpath

paths={}

if args.truthmv!='none':
    paths['truth']=args.truthmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.starmv!='none':
    paths['starwarps']=args.starmv
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv 
if args.ngmv!='none':
    paths['ngmem']=args.ngmv
if args.modelingmv!='none':
    paths['modeling']=args.modelingmv
    
######################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, times, obslist_t, polpaths = process_obs(obs, args, paths)
    
######################################################################
# Set parameters
npix   = 160
fov    = 160 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

# Set the number of subplots
N = len(paths.keys()) 
######################################################################


df = pd.DataFrame(obs.data)
mv={}

for p in paths.keys():
    # Load the movie data and extract images at the scan times
    mvf = eh.movie.load_hdf5(paths[p])
    imlist = []
    for t in times:
        im = mvf.get_image(t).regrid_image(fov, npix)
        imlist.append(im)
    mv[p] = eh.movie.merge_im_list(imlist)


def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

# Create subplots
# Define size/scaling behavior more clearly
width = linear_interpolation(N, 2, 8, 7, 16)
height = linear_interpolation(N, 2, 16, 7, 9)

hspace = linear_interpolation(N, 2, 0.01, 7, 0.1)
wspace = linear_interpolation(N, 2, 0.05, 7, 0.1)

top = linear_interpolation(N, 2, 0.8, 7, 0.7)
bottom = linear_interpolation(N, 2, 0.01, 7, 0.1)
left = linear_interpolation(N, 2, 0.01, 7, 0.005)
right = linear_interpolation(N, 2, 0.8, 7, 0.9)

# Use in plot
fig, ax = plt.subplots(nrows=3, ncols=N, figsize=(width, height))
fig.subplots_adjust(hspace=hspace, wspace=wspace, top=top, bottom=bottom, left=left, right=right)


if N>1:
    for i in range(N):
        ax[0,i].set_xticks([]), ax[0,i].set_yticks([])
        ax[1,i].set_xticks([]), ax[1,i].set_yticks([])
        ax[2,i].set_xticks([]), ax[2,i].set_yticks([])
else:
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[2].set_xticks([]), ax[2].set_yticks([])
    
var={}
imlist={}
varmax=[]
varmin=[]

varQ={}
varmaxQ=[]
varminQ=[]

varU={}
varmaxU=[]
varminU=[]

for p in paths.keys():
    amplist=[]
    amplistQ=[]
    amplistU=[]
    imlist = mv[p].im_list()
    for i in range(len(imlist)):
        im = imlist[i]

        npix = im.xdim   
        U = np.linspace(-10.0e9, 10.0e9, npix)
        V = np.linspace(-10.0e9, 10.0e9, npix)
        UU, VV = np.meshgrid(U, V)
        UV = np.vstack((UU.flatten(), VV.flatten())).T
        vis, visQ, visU, _ = im.sample_uv(UV)
        fft = np.array(vis)
        fftQ = np.array(visQ)
        fftU = np.array(visU)

        # Calculate the amplitude of the FFT
        amp = np.abs(fft).reshape(im.xdim, im.xdim)
        amplist.append(amp)
        
        ampQ = np.abs(fftQ).reshape(im.xdim, im.xdim)
        amplistQ.append(ampQ)
        
        ampU = np.abs(fftU).reshape(im.xdim, im.xdim)
        amplistU.append(ampU)

    # Calculate the variance
    #var[p] = np.std(np.array(amplist), axis=0) ** 2 * 100
    var[p] = np.std(np.array(amplist), axis=0) / np.mean(np.array(amplist), axis=0)
    varmax.append(np.max(var[p]))
    varmin.append(np.min(var[p]))
    
    #varQ[p] = np.std(np.array(amplistQ), axis=0) ** 2 * 100
    varQ[p] = np.std(np.array(amplistQ), axis=0) / np.mean(np.array(amplistQ), axis=0)
    varmaxQ.append(np.max(varQ[p]))
    varminQ.append(np.min(varQ[p]))
    
    #varU[p] = np.std(np.array(amplistU), axis=0) ** 2 * 100
    varU[p] = np.std(np.array(amplistU), axis=0) / np.mean(np.array(amplistU), axis=0)
    varmaxU.append(np.max(varU[p]))
    varminU.append(np.min(varU[p]))

scale = 10
i=0
for p in paths.keys():
    if len(paths.keys())>1:
        a = ax[0,i].imshow(var[p], cmap='binary', extent=[scale, -scale, -scale, scale], vmin=0, vmax=max(varmax))
        aQ = ax[1,i].imshow(varQ[p], cmap='Blues', extent=[scale, -scale, -scale, scale], vmin=0, vmax=max(varmaxQ))
        aU = ax[2,i].imshow(varU[p], cmap='Purples', extent=[scale, -scale, -scale, scale], vmin=0, vmax=max(varmaxU))
        ax[0, i].text(8, 8, f'max = {np.max(var[p]):.3f}', color='red')
        ax[1, i].text(8, 8, f'max = {np.max(varQ[p]):.3f}', color='red')
        ax[2, i].text(8, 8, f'max = {np.max(varU[p]):.3f}', color='red')
        np.save(f'{outpath}_{p}_varI.npy', var[p])
        if p!='resolve':
            np.save(f'{outpath}_{p}_varQ.npy', varQ[p])
            np.save(f'{outpath}_{p}_varU.npy', varU[p])
        # Overlay the visibility points
        ax[0,i].plot(df['u'] / 1e9, df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[0,i].plot(-df['u'] / 1e9, -df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[1,i].plot(df['u'] / 1e9, df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[1,i].plot(-df['u'] / 1e9, -df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[2,i].plot(df['u'] / 1e9, df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[2,i].plot(-df['u'] / 1e9, -df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        # Set axis limits and labels
        ax_lim = 10  # Glambda
        ax[0,i].set_xlim([ax_lim, -ax_lim])
        ax[0,i].set_ylim([-ax_lim, ax_lim])
        ax[0,i].set_title(titles[p])
        ax[1,i].set_xlim([ax_lim, -ax_lim])
        ax[1,i].set_ylim([-ax_lim, ax_lim])
        ax[2,i].set_xlim([ax_lim, -ax_lim])
        ax[2,i].set_ylim([-ax_lim, ax_lim])
        i=i+1
    else:
        a = ax[0].imshow(var[p], cmap='binary', extent=[scale, -scale, -scale, scale], vmin=0, vmax=max(varmax))
        aQ = ax[1].imshow(varQ[p], cmap='Blues', extent=[scale, -scale, -scale, scale], vmin=0, vmax=max(varmaxQ))
        aU = ax[2].imshow(varU[p], cmap='Purples', extent=[scale, -scale, -scale, scale], vmin=0, vmax=max(varmaxU))
        ax[0].text(8, 8, f'max = {np.max(var[p]):.3f}', color='red')
        ax[1].text(8, 8, f'max = {np.max(varQ[p]):.3f}', color='red')
        ax[2].text(8, 8, f'max = {np.max(varU[p]):.3f}', color='red')
        np.save(f'{outpath}_{p}_varI.npy', var[p])
        if p!='resolve':
            np.save(f'{outpath}_{p}_varQ.npy', varQ[p])
            np.save(f'{outpath}_{p}_varU.npy', varU[p])
        # Overlay the visibility points
        ax[0].plot(df['u'] / 1e9, df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[0].plot(-df['u'] / 1e9, -df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[1].plot(df['u'] / 1e9, df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[1].plot(-df['u'] / 1e9, -df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[2].plot(df['u'] / 1e9, df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        ax[2].plot(-df['u'] / 1e9, -df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
        # Set axis limits and labels
        ax_lim = 10  # Glambda
        ax[0].set_xlim([ax_lim, -ax_lim])
        ax[0].set_ylim([-ax_lim, ax_lim])
        ax[0].set_title(titles[p])
        ax[1].set_xlim([ax_lim, -ax_lim])
        ax[1].set_ylim([-ax_lim, ax_lim])
        ax[2].set_xlim([ax_lim, -ax_lim])
        ax[2].set_ylim([-ax_lim, ax_lim])
        i=i+1
        
from mpl_toolkits.axes_grid1 import make_axes_locatable

for i, (artist, label, ticks) in enumerate(zip(
    [a, aQ, aU],
    [r'$\zeta_I$', 
     r'$\zeta_Q$', 
     r'$\zeta_U$'],
    [[0.0, max(varmax)*0.25, max(varmax)*0.5, max(varmax)*0.75, max(varmax)],
     [0.0, max(varmaxQ)*0.25, max(varmaxQ)*0.5, max(varmaxQ)*0.75, max(varmaxQ)],
     [0.0, max(varmaxU)*0.25, max(varmaxU)*0.5, max(varmaxU)*0.75, max(varmaxU)]]
)):
    if N > 1:
        divider = make_axes_locatable(ax[i, N-1])  # Use only the last column
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(artist, cax=cax)
        cbar.set_label(label, fontsize=14)
        cbar.set_ticks(ticks)
    else:
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(artist, cax=cax)
        cbar.set_label(label, fontsize=14)
        cbar.set_ticks(ticks)

plt.savefig(f'{outpath}.png', bbox_inches='tight')