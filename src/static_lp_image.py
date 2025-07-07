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
from matplotlib.colors import Normalize
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
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .fits')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .fits')
    p.add_argument('--starmv', type=str, default='none', help='path of starwarps .fits')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .fits')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .fits')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .fits')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .fits')
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

imlists = {}
for p in paths.keys():
    im = eh.image.load_fits(paths[p]).regrid_image(fov, npix)
    if p=='truth':
        if args.scat!='onsky':
            im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS)
    imlists[p] =im

def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def static(ims, titles, paths, outpath='./', fov=None, interp='bicubic'):
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
    TBfactor = 3.254e13/(ims[list(paths.keys())[0]].rf**2 * ims[list(paths.keys())[0]].psize**2)/1e9    
    vmax, vmin = max(ims[list(paths.keys())[0]].ivec)*TBfactor, 0 #min(ims['kine'][0].ivec)*TBfactor
    
    polmovies={}
    for i, p in enumerate(ims.keys()):    
        if len(ims[p].qvec) and len(ims[p].uvec) > 0 and p!='starwarps':
            polmovies[p]=ims[p]
                
    for i, p in enumerate(ims.keys()):
        if len(ims.keys())>1:
            ax[i].clear() 
            TBfactor = 3.254e13/(ims[p].rf**2 * ims[p].psize**2)/1e9
            im =ax[i].imshow(np.array(ims[p].imarr(pol='I'))*TBfactor, cmap='binary', interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

            ax[i].set_title(titles[p], fontsize=18)
            ax[i].set_xticks([]), ax[i].set_yticks([])
        else:
            ax.clear() 
            TBfactor = 3.254e13/(ims[p].rf**2 * ims[p].psize**2)/1e9
            im =ax.imshow(np.array(ims[p].imarr(pol='I'))*TBfactor, cmap='binary', interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

            ax.set_title(titles[p], fontsize=18)
            ax.set_xticks([]), ax.set_yticks([])
        
        if p in polmovies.keys():
            self = polmovies[p]
            amp = np.sqrt(self.qvec**2 + self.uvec**2)
            scal=np.max(amp)*0.5
            vx = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)
            vy = ( np.cos(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)
            # tick color will be proportional to mfrac
            mfrac=(amp/self.ivec).reshape(self.xdim, self.ydim)
                
            pcut = 0.1
            mcut = 0.
            skip = 10
            imarr = self.imvec.reshape(self.ydim, self.xdim)
            Imax=max(self.imvec)
            mfrac = np.ma.masked_where(imarr < pcut * Imax, mfrac) 
            #new version with sharper cuts
            mfrac_map=(np.sqrt(self.qvec**2+self.uvec**2)).reshape(self.xdim, self.ydim)
            QUmax=max(np.sqrt(self.qvec**2+self.uvec**2))
            pcut=0.1
            mfrac_m = np.ma.masked_where(mfrac_map < pcut * QUmax , mfrac)
            pcut=0.1
            mfrac_m = np.ma.masked_where(imarr < pcut * Imax, mfrac_m)
            ######
            pixel=self.psize/eh.RADPERUAS #uas
            FOV=pixel*self.xdim
            # generate 2 2d grids for the x & y bounds
            y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                            slice(-FOV/2, FOV/2, pixel)]
            x = np.ma.masked_where(imarr < pcut * Imax, x) 
            y = np.ma.masked_where(imarr < pcut * Imax, y) 
            vx = np.ma.masked_where(imarr < pcut * Imax, vx) 
            vy = np.ma.masked_where(imarr < pcut * Imax, vy) 
            

            cnorm=Normalize(vmin=0.0, vmax=0.5)
            if len(ims.keys())>1:
                tickplot = ax[i].quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
                               mfrac_m[::skip,::skip],
                               headlength=0,
                               headwidth = 1,
                               pivot='mid',
                               width=0.01,
                               cmap='rainbow',
                               norm=cnorm,
                               scale=16)
            else:
                tickplot = ax.quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
                               mfrac_m[::skip,::skip],
                               headlength=0,
                               headwidth = 1,
                               pivot='mid',
                               width=0.01,
                               cmap='rainbow',
                               norm=cnorm,
                               scale=16)
        
        ax1 = fig.add_axes([linear_interpolation(num_subplots, 2, 0.82, 7, 0.92), linear_interpolation(num_subplots, 2, 0.025, 7, 0.1), linear_interpolation(num_subplots, 2, 0.035, 7, 0.01), linear_interpolation(num_subplots, 2, 0.765, 7, 0.6)] , anchor = 'E') 
        cbar = fig.colorbar(tickplot, cmap='rainbow', cax=ax1, pad=0.14,fraction=0.038, orientation="vertical", ticklocation='right') 
        cbar.set_label('$|m|$') 
        
        plt.suptitle("Median", y=0.95, fontsize=22)
        # Save Plot
        plt.savefig(f'{outpath}.png', bbox_inches='tight')

static(imlists, titles, paths, outpath=outpath, fov=fov)
