######################################################################
# Author: Rohan Dahale, Date: 14 May 2024
######################################################################

import ehtim as eh
import ehtplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
import pdb
import argparse
import os
import glob

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./gif.gif', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='sct, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

######################################################################
# Plotting Setup
######################################################################
#plt.rc('text', usetex=True)
import matplotlib as mpl
#mpl.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'monospace': ['Computer Modern Typewriter']})
mpl.rcParams['figure.dpi']=300
#mpl.rcParams["mathtext.default"] = 'regular'
#plt.style.use('dark_background')
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18

from matplotlib import font_manager
font_dirs = font_manager.findSystemFonts(fontpaths='./fonts/', fontext="ttf")
#mpl.rc('text', usetex=True)

fe = font_manager.FontEntry(
    fname='./fonts/Helvetica.ttf',
    name='Helvetica')
font_manager.fontManager.ttflist.insert(0, fe) # or append is fine
mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
#####################################################################

# Time average data to 60s
obs = eh.obsdata.load_uvfits(args.data)
obs.add_scans()
obs = obs.avg_coherent(60.0)

# From GYZ: If data used by pipelines is descattered (refractive + diffractive),
# Add 2% error and deblur original data.
if args.scat=='dsct':
    obs = obs.add_fractional_noise(0.02)
    import ehtim.scattering.stochastic_optics as so
    sm = so.ScatteringModel()
    obs = sm.Deblur_obs(obs)

obs.add_scans()
times = []
for t in obs.scans:
    times.append(t[0])
obslist = obs.split_obs()
######################################################################

outpath = args.outpath

paths={}

if args.truthmv!='none':
    paths['truth']=args.truthmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv 
if args.ngmv!='none':
    paths['ngmem']=args.ngmv

# Truncating the times and obslist based on submitted movies
obslist_tn=[]
min_arr=[] 
max_arr=[]
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    min_arr.append(min(mv.times))
    max_arr.append(max(mv.times))
x=np.argwhere(times>max(min_arr))
ntimes=[]
for t in x:
    ntimes.append(times[t[0]])
    obslist_tn.append(obslist[t[0]])
times=[]
obslist_t=[]
y=np.argwhere(min(max_arr)>ntimes)
for t in y:
    times.append(ntimes[t[0]])
    obslist_t.append(obslist_tn[t[0]])
    
######################################################################
# Set parameters
npix   = 128
fov    = 120 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

titles = {  
            'truth'      : 'Truth',
            'kine'       : 'kine',
            'resolve'    : 'resolve',
            'ehtim'      : 'ehtim',
            'doghit'     : 'DoG-HiT',
            'ngmem'      : 'ngMEM'
        }


######################################################################
# Adding times where there are gaps and assigning cmap as binary_us in the gaps
dt=[]
for i in range(len(times)-1):
    dt.append(times[i+1]-times[i])
    
mean_dt=np.mean(np.array(dt))

u_times=[]
cmapsl = []
for i in range(len(times)-1):
    if times[i+1]-times[i] > mean_dt:
        j=0
        while u_times[len(u_times)-1] < times[i+1]-mean_dt:
            u_times.append(times[i]+j*mean_dt)
            cmapsl.append('binary')
            j=j+1
    else:
        u_times.append(times[i])
        cmapsl.append('binary')

######################################################################

imlistIs = {}
for p in paths.keys():
    mov = eh.movie.load_hdf5(paths[p])
    imlistI = []
    for t in u_times:
        im = mov.get_image(t)
        if p=='truth':
            im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS).regrid_image(fov, npix)
        #else:
        im = im.blur_circ(fwhm_i=blur).regrid_image(fov, npix)
        im.ivec=im.ivec/im.total_flux()
        imlistI.append(im)
    imlistIs[p] =imlistI
    #med = np.median(imlistIs[p],axis=0)
    #for i in range(len(imlistIs[p])):
        #imlistIs[p][i]= np.clip(imlistIs[p][i]-med,0,1)
        

def writegif(movieIs, titles, paths, outpath='./', fov=None, times=[], cmaps=cmapsl, interp='gaussian', fps=20):

    fig, ax = plt.subplots(nrows=1, ncols=len(paths.keys()), figsize=(15*len(paths.keys())/7+1,3))
    #fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.7, bottom=0.1, left=0.005, right=0.9)

    # Set axis limits
    lims = None
    if fov:
        fov  = fov / eh.RADPERUAS
        lims = [fov//2, -fov//2, -fov//2, fov//2]

    # Set colorbar limits
    TBfactor = 3.254e13/(movieIs['kine'][0].rf**2 * movieIs['kine'][0].psize**2)/1e9    
    vmax, vmin = max(movieIs['kine'][0].ivec)*TBfactor, min(movieIs['kine'][0].ivec)*TBfactor
    
    polmovies={}
    for i, p in enumerate(movieIs.keys()):
        if len(movieIs[p][0].vvec)> 0:
            polmovies[p]=movieIs[p]
    
    def plot_frame(f):
        for i, p in enumerate(movieIs.keys()):
            ax[i].clear() 
            TBfactor = 3.254e13/(movieIs[p][f].rf**2 * movieIs[p][f].psize**2)/1e9
            im =ax[i].imshow(np.array(movieIs[p][f].imarr(pol='I'))*TBfactor, cmap=cmaps[f], interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)
            
            ax[i].set_title(titles[p], fontsize=18)
            ax[i].set_xticks([]), ax[i].set_yticks([])
            
            if p in polmovies.keys():
                im = polmovies[p][f]

                # Contour level settings
                TBfactor = 3.254e13/(im.rf**2 * im.psize**2)/1e9
                nlevels = 6
                Vmax_cbar=2.0/TBfactor 
                Vmax = Vmax_cbar   
                levels = np.arange(-Vmax*TBfactor, (1+1./nlevels)*Vmax*TBfactor, Vmax*TBfactor/nlevels)
                midindex = int(len(levels)/2-0.5)

                levels = np.delete(levels, [midindex])
                            
                # generate 2 2d grids for the x & y bounds
                pixel=im.psize/eh.RADPERUAS #uas
                FOV=pixel*im.xdim

                # generate 2 2d grids for the x & y bounds
                y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                    slice(-FOV/2, FOV/2, pixel)]
                
                varr = im.vvec.reshape(im.ydim, im.xdim)

                vcontour = ax[i].contour(x, y, np.flip(varr*TBfactor), levels=levels, cmap='bwr', vmin=-Vmax_cbar*TBfactor, vmax=Vmax_cbar*TBfactor, linewidths=2) 
                
            
        if f==0:
            ax1 = fig.add_axes([0.92, 0.1, 0.01, 0.6] , anchor = 'E')     
            fig.colorbar(ScalarMappable(norm=vcontour.norm, cmap=vcontour.cmap), cax=ax1, pad=0.12, fraction=0.046).set_label(label='$T_B$ ($10^9$ K)')

        
        plt.suptitle(f"{u_times[f]:.2f} UT", y=0.95, fontsize=22)

        return fig
    
    def update(f):
        return plot_frame(f)

    ani = animation.FuncAnimation(fig, update, frames=len(u_times), interval=1e3/fps)
    wri = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)

    # Save gif
    ani.save(f'{outpath}.gif', writer=wri, dpi=100)

writegif(imlistIs, titles, paths, outpath=outpath, fov=fov, times=u_times, cmaps=cmapsl)
