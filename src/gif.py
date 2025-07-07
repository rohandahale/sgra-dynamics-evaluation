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

obs = eh.obsdata.load_uvfits(args.data)
obs, times, obslist_t, polpaths = process_obs(obs, args, paths)

######################################################################
# Set parameters
npix   = 160
fov    = 160 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

######################################################################
# Adding times where there are gaps and assigning cmap as binary_us in the gaps
dt=[]
for i in range(len(times)-1):
    dt.append(times[i+1]-times[i])
    
mean_dt=np.mean(np.array(dt))

u_times=[]
cmapsl = []
tcolor=[]
for i in range(len(times)-1):
    if times[i+1]-times[i] > mean_dt:
        if i==0:
            u_times.append(times[i+1]-1.1*mean_dt)
        j=0
        while u_times[len(u_times)-1] < times[i+1]-mean_dt:
            if i==0:
                del u_times[i]
            u_times.append(times[i]+j*mean_dt)
            cmapsl.append('afmhot_us')
            tcolor.append('red')
            j=j+1
    else:
        u_times.append(times[i])
        cmapsl.append('afmhot_us')
        tcolor.append('black')

######################################################################

imlistIs = {}
for p in paths.keys():
    mov = eh.movie.load_hdf5(paths[p])
    imlistI = []
    for t in u_times:
        im = mov.get_image(t)
        if p=='truth':
            if args.scat!='onsky':
                im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS).regrid_image(fov, npix)
        im = im.blur_circ(fwhm_i=blur, fwhm_pol=blur).regrid_image(fov, npix)
        #im.ivec=im.ivec/im.total_flux()
        imlistI.append(im)
    imlistIs[p] =imlistI

def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def writegif(movieIs, titles, paths, outpath='./', fov=None, times=[], cmaps=cmapsl, tcolor=tcolor, interp='gaussian', fps=20):
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
    TBfactor = 3.254e13/(movieIs[list(paths.keys())[0]][0].rf**2 * movieIs[list(paths.keys())[0]][0].psize**2)/1e9    
    vmax, vmin = max(movieIs[list(paths.keys())[0]][0].ivec)*TBfactor, min(movieIs[list(paths.keys())[0]][0].ivec)*TBfactor

    def plot_frame(f):
        for i, p in enumerate(movieIs.keys()):
            if len(movieIs.keys())>1:
                ax[i].clear() 
                TBfactor = 3.254e13/(movieIs[p][f].rf**2 * movieIs[p][f].psize**2)/1e9
                im =ax[i].imshow(np.array(movieIs[p][f].imarr(pol='I'))*TBfactor, cmap=cmaps[f], interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

                ax[i].set_title(titles[p], fontsize=18)
                ax[i].set_xticks([]), ax[i].set_yticks([])
            else:
                ax.clear() 
                TBfactor = 3.254e13/(movieIs[p][f].rf**2 * movieIs[p][f].psize**2)/1e9
                im =ax.imshow(np.array(movieIs[p][f].imarr(pol='I'))*TBfactor, cmap=cmaps[f], interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)

                ax.set_title(titles[p], fontsize=18)
                ax.set_xticks([]), ax.set_yticks([])
            
        if f==0:
            ax1 = fig.add_axes([linear_interpolation(num_subplots, 2, 0.82, 7, 0.92), linear_interpolation(num_subplots, 2, 0.025, 7, 0.1), linear_interpolation(num_subplots, 2, 0.035, 7, 0.01), linear_interpolation(num_subplots, 2, 0.765, 7, 0.6)] , anchor = 'E') 
            fig.colorbar(im, cax=ax1, ax=None, label = '$T_B$ ($10^9$ K)', ticklocation='right')
        
        plt.suptitle(f"{u_times[f]:.2f} UT", y=0.95, fontsize=22, color=tcolor[f])

        return fig
    
    def update(f):
        return plot_frame(f)

    ani = animation.FuncAnimation(fig, update, frames=len(u_times), interval=1e3/fps)
    wri = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)

    # Save gif
    ani.save(f'{outpath}.gif', writer=wri, dpi=100)

writegif(imlistIs, titles, paths, outpath=outpath, fov=fov, times=u_times, cmaps=cmapsl, tcolor=tcolor)
