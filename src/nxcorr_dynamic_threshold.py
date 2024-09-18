######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import scipy
import argparse
import os
import glob
from utilities import *
colors, titles, labels, mfcs, mss = common()

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv',  type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',   type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',   type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',    type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',   type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')
    return p

# List of parsed arguments
args = create_parser().parse_args()

pathmovt  = args.truthmv
outpath = args.outpath

paths={}
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
    
#################################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, obs_t, obslist_t, splitObs, times, w_norm, equal_w = process_obs_weights(obs, args, paths)

######################################################################

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(28,8), sharex=True)

ax[0].set_ylabel('$\Delta$ nxcorr (I)')
ax[1].set_ylabel('$\Delta$ nxcorr (Q)')
ax[2].set_ylabel('$\Delta$ nxcorr (U)')
ax[3].set_ylabel('$\Delta$ nxcorr (V)')

ax[0].set_xlabel('Time (UTC)')
ax[1].set_xlabel('Time (UTC)')
ax[2].set_xlabel('Time (UTC)')
ax[3].set_xlabel('Time (UTC)')

ax[0].set_ylim(-1.1,1.1)
ax[1].set_ylim(-1.1,1.1)
ax[2].set_ylim(-1.1,1.1)
ax[3].set_ylim(-1.1,1.1)

#ax[0].set_ylim(0,7)
#ax[1].set_ylim(0,7)
#ax[2].set_ylim(0,7)
#ax[3].set_ylim(0,7)

mvt=eh.movie.load_hdf5(pathmovt)
if args.scat!='onsky':
    mvt=mvt.blur_circ(fwhm_x=15*eh.RADPERUAS, fwhm_x_pol=15*eh.RADPERUAS, fwhm_t=0)

mv_nxcorr={}
for p in paths.keys():
    mv_nxcorr[p]=np.zeros(4)

row_labels = ['I','Q','U','V']
table_vals = pd.DataFrame(data=mv_nxcorr, index=row_labels)
        
pollist=['I','Q','U','V']
k=0
for pol in pollist:
    polpaths={}
    for p in paths.keys():
        mv=eh.movie.load_hdf5(paths[p])
        im=mv.im_list()[0]
        
        if pol=='I':
            if len(im.ivec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        elif pol=='Q':
            if len(im.qvec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        elif pol=='U':
            if len(im.uvec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        elif pol=='V':
            if len(im.vvec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        else:
            print('Parse a vaild pol value')

    s=0
    for p in polpaths.keys():
        mv=eh.movie.load_hdf5(polpaths[p])
        imlist_o = [mv.get_image(t) for t in times]
        
        # center the movie with repect to the truth movie frame 0
        mvtruth_image=eh.movie.load_hdf5(pathmovt).im_list()[0]
        shifts = mvtruth_image.align_images(imlist_o)[1] #Shifts only from stokes I
        mean_shift = [int(np.round(np.mean(np.array(shifts)[:, 1]), 0)), int(np.round(np.mean(np.array(shifts)[:, 1]),0))]
        imlist=[]
        for im in imlist_o:
            im2 = im.shift(mean_shift) # Shifts all pol
            imlist.append(im2)
            
        imlistarr=[]
        for im in imlist:
            #im.ivec=im.ivec/im.total_flux()
            imlistarr.append(im.imarr(pol=pol))
        median = np.median(imlistarr,axis=0)
        for im in imlist:
            if pol=='I':
                im.ivec= np.clip(im.imarr(pol=pol)-median,0,1).flatten()
            elif pol=='Q':
                im.qvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='U':
                im.uvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='V':
                im.vvec= np.array(im.imarr(pol=pol)-median).flatten()
    
    
        imlist_t =[mvt.get_image(t) for t in times]
        imlistarr=[]
        for im in imlist_t:
            #im.ivec=im.ivec/im.total_flux()
            #im.qvec=im.qvec/im.total_flux()
            #im.uvec=im.uvec/im.total_flux()
            #im.vvec=im.vvec/im.total_flux()
            imlistarr.append(im.imarr(pol=pol))
        median = np.median(imlistarr,axis=0)
        for im in imlist_t:
            if pol=='I':
                im.ivec= np.clip(im.imarr(pol=pol)-median,0,1).flatten()
            elif pol=='Q':
                im.qvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='U':
                im.uvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='V':
                im.vvec= np.array(im.imarr(pol=pol)-median).flatten()

        nxcorr_t=[]
        nxs_cri=[]
        nxcorr_tab=[]

        i=0
        for im in imlist:
            nxcorr=imlist_t[i].compare_images(im, pol=pol, metric=['nxcorr'], shift=[0,0])
            nxcorr_t.append(nxcorr[0][0])
            nxcorr_tab.append(nxcorr[0][0])
            
            beamparams = obslist_t[i].fit_beam(weighting='uniform')
            nxs_cri.append(get_nxcorr_cri_beam(imlist_t[i], beamparams, pol=pol))
            
            i=i+1
        
        w_ratio = w_norm[pol]/np.max(w_norm[pol])
        w_ncri  = w_ratio*np.array(nxs_cri)
        
        count=0
        diff=np.array(nxcorr_t)-np.array(w_ncri)
        for d in diff:
            if d>0:
                count=count+1
        if count==0:
            table_vals[p][pol]=1000
        else:
            table_vals[p][pol]=np.round(count/len(nxcorr_t)*100,2)
                    
        mc=colors[p]
        alpha = 0.5
        lc=colors[p]
        
        if k==0:
            ax[k].plot(times, nxcorr_t-w_ncri,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
            #ax[k].plot(times, w_ncri, ls='--', lw=1,  color='k', label='Threshold')
        else:
            ax[k].plot(times, nxcorr_t-w_ncri,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha)
            #ax[k].plot(times, w_ncri, ls='--', lw=1,  color='k')
    
        #ax[k].hlines(s+1, xmin=10.5, xmax=14.5, color=colors[p], ls='--', lw=1.5, zorder=0)
        ax[k].hlines(0, xmin=10.5, xmax=14.5, color='k', ls='--', lw=1.5, zorder=0)
        ax[k].yaxis.set_ticklabels([])
        s=s+1
    
    k=k+1
    
table_vals.rename(index={'I':'% Pass (I)'},inplace=True)
table_vals.rename(index={'Q':'% Pass (Q)'},inplace=True)
table_vals.rename(index={'U':'% Pass (U)'},inplace=True)
table_vals.rename(index={'V':'% Pass (V)'},inplace=True)
table_vals.replace(1000, 'Fail', inplace=True)
table_vals.replace(0.000, '-', inplace=True)

table = ax[1].table(cellText=table_vals.values,
                    rowLabels=table_vals.index,
                    colLabels=table_vals.columns,
                    cellLoc='center',
                    loc='bottom',
                    bbox=[-0.1, -0.5, 2.5, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(18)
for c in table.get_children():
    c.set_edgecolor('none')
    c.set_text_props(color='black')
    c.set_facecolor('none')
    c.set_edgecolor('black')
ax[0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(3.5, 1.2), markerscale=5.0)
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)
