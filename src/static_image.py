######################################################################
# Author: Rohan Dahale, Date: 09 November 2024
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
# Set parameters
npix   = 160
fov    = 160 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

imlistIs = {}
for p in paths.keys():
    im = eh.image.load_fits(paths[p]).regrid_image(fov, npix)
    #im.ivec = im.ivec/im.total_flux()
    if p=='truth':
        if args.scat!='onsky':
            im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS)
    imlistIs[p] =im


def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def static(Is, titles, paths, outpath='./', fov=None, interp='bicubic'):
    num_subplots=len(paths.keys())
    fig, ax = plt.subplots(nrows=1, ncols=len(paths.keys()), figsize=(linear_interpolation(num_subplots, 2, 8, 7, 16),linear_interpolation(num_subplots, 2, 4, 7, 3)))
   #fig.tight_layout()
    fig.subplots_adjust(hspace=linear_interpolation(num_subplots, 2, 0.01, 7, 0.1), wspace=linear_interpolation(num_subplots, 2, 0.05, 7, 0.1), top=linear_interpolation(num_subplots, 2, 0.8, 7, 0.7), bottom=linear_interpolation(num_subplots, 2, 0.01, 7, 0.1), left=linear_interpolation(num_subplots, 2, 0.01, 7, 0.005), right=linear_interpolation(num_subplots, 2, 0.8, 7, 0.9))

    # Set axis limits
    lims = None
    if fov:
        fov  = fov / eh.RADPERUAS
        lims = [fov//2, -fov//2, -fov//2, fov//2]

    # Set colorbar limits
    TBfactor = 3.254e13/(Is[list(paths.keys())[0]].rf**2 * Is[list(paths.keys())[0]].psize**2)/1e9    
    vmax, vmin = max(Is[list(paths.keys())[0]].ivec)*TBfactor, min(Is[list(paths.keys())[0]].ivec)*TBfactor

    
    for i, p in enumerate(Is.keys()):
        if len(Is.keys())>1:
            ax[i].clear() 
            TBfactor = 3.254e13/(Is[p].rf**2 * Is[p].psize**2)/1e9
            im =ax[i].imshow(np.array(Is[p].imarr(pol='I'))*TBfactor, cmap='afmhot_us', interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

            ax[i].set_title(titles[p], fontsize=18)
            ax[i].set_xticks([]), ax[i].set_yticks([])
        else:
            ax.clear() 
            TBfactor = 3.254e13/(Is[p].rf**2 * Is[p].psize**2)/1e9
            im =ax.imshow(np.array(Is[p].imarr(pol='I'))*TBfactor, cmap='afmhot_us', interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

            ax.set_title(titles[p], fontsize=18)
            ax.set_xticks([]), ax.set_yticks([])

    
    ax1 = fig.add_axes([linear_interpolation(num_subplots, 2, 0.82, 7, 0.92), linear_interpolation(num_subplots, 2, 0.025, 7, 0.1), linear_interpolation(num_subplots, 2, 0.035, 7, 0.01), linear_interpolation(num_subplots, 2, 0.765, 7, 0.6)] , anchor = 'E') 
    fig.colorbar(im, cax=ax1, ax=None, label = '$T_B$ ($10^9$ K)', ticklocation='right')
    
    plt.suptitle("Median", y=0.95, fontsize=22)
    # Save Plot
    plt.savefig(f'{outpath}.png', bbox_inches='tight')
    
static(imlistIs, titles, paths, outpath=outpath, fov=fov)
