##############################################################################################
# Author: Rohan Dahale, Date: 25 March 2024, Version=v0.9
##############################################################################################

import os

#Base Directory of the Project
basedir='/mnt/disks/shared/eht/sgra_dynamics_april11'

# Pipeline and Colors
colors = {   
            'truth'      : 'white',
            'kine'       : 'darkorange',
            'starwarps'  : 'xkcd:azure',
            'ehtim'      : 'forestgreen',
            'doghit'     : 'darkviolet',
            'ngmem'      : 'crimson',
            'resolve'    : 'hotpink'
        }


# Results Directory
resultsdir='results_VL_1'
# Results Directory
subdir='submission_VL_1'

models={
        'crescent'  : 'ring', 
        'ring'      : 'ring', 
        'disk'      : 'non-ring', 
        'edisk'     : 'non-ring',
        'double'    : 'non-ring', 
        'point'     : 'non-ring'
        }


epoch='3601'
band='LO'
cband='HI+LO'
noise='thermal+phasegains'
scat = 'none'   # Options: sct, dsct, none

##############################################################################################
# Directory of the results
##############################################################################################

if not os.path.exists(f'{basedir}/evaluation/{resultsdir}'):
    os.makedirs(f'{basedir}/evaluation/{resultsdir}')
    if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/interpolated_movies'):
        os.makedirs(f'{basedir}/evaluation/{resultsdir}/interpolated_movies')
        for pipe in colors.keys():
            if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/interpolated_movies/{pipe}'):
                os.makedirs(f'{basedir}/evaluation/{resultsdir}/interpolated_movies/{pipe}')
                
    if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/averaged_movies'):
        os.makedirs(f'{basedir}/evaluation/{resultsdir}/averaged_movies')
        for pipe in colors.keys():
            if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/averaged_movies/{pipe}'):
                os.makedirs(f'{basedir}/evaluation/{resultsdir}/averaged_movies/{pipe}')
                
    if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/plots'):
        os.makedirs(f'{basedir}/evaluation/{resultsdir}/plots')

##############################################################################################
# Interpolated Movies, Averaged Movies
##############################################################################################

modelname={}
for m in models.keys():
    modelname[m]={}
    for pipe in colors.keys():
        if pipe=='resolve':
            model=f'{m}_{epoch}_{cband}'
            modelname[m][pipe]=model
        else:
            model=f'{m}_{epoch}_{band}'
            modelname[m][pipe]=model
            
        # Interpolated Movies
        indir=f'{basedir}/{subdir}/{pipe}/{model}/'
        outdir=f'{basedir}/evaluation/{resultsdir}/interpolated_movies/{pipe}/{model}/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        os.system(f'python {basedir}/evaluation/scripts/pipeline/src/hdf5_standardize.py -i {indir} -o {outdir}')
        
        #Average Movies
        outdir=f'{basedir}/evaluation/{resultsdir}/averaged_movies/{pipe}/{model}/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        os.system(f'python {basedir}/evaluation/scripts/pipeline/src/avg_frame.py -i {indir} -o {outdir}')

##############################################################################################
# Chi-squares, closure triangles, ampltitudes, nxcorr, gif, pol net avg, REx
##############################################################################################
for m in models: 
    if scat!='none':       
        pathmov  = f'{basedir}/{subdir}/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1_{scat}.hdf5'
        pathmov2 = f'{basedir}/{subdir}/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1_{scat}.hdf5'
        pathmov3 = f'{basedir}/{subdir}/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1_{scat}.hdf5'
        pathmov4 = f'{basedir}/{subdir}/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1_{scat}.hdf5'
        pathmov5 = f'{basedir}/{subdir}/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1_{scat}.hdf5'
        pathmov6 = f'{basedir}/{subdir}/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1_{scat}.hdf5'
    else:
        pathmov  = f'{basedir}/{subdir}/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1.hdf5'
        pathmov2 = f'{basedir}/{subdir}/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1.hdf5'
        pathmov3 = f'{basedir}/{subdir}/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1.hdf5'
        pathmov4 = f'{basedir}/{subdir}/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1.hdf5'
        pathmov5 = f'{basedir}/{subdir}/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1.hdf5'
        pathmov6 = f'{basedir}/{subdir}/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1.hdf5'
    
   #paths=f'--kinemv {pathmov} --starmv {pathmov2} --ehtmv {pathmov3} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    paths=f'--kinemv {pathmov} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    
    data=f'{basedir}/{subdir}/data/{m}_{epoch}_{band}.uvfits'
    
    pollist=['I', 'Q', 'U', 'V']
    for pol in pollist:
        #CHISQ
        outpath=f'{basedir}/evaluation/{resultsdir}/plots/chisq_{pol}_{modelname[m]["kine"]}'
        if not os.path.exists(outpath+'.png'):
            os.system(f'python {basedir}/evaluation/scripts/pipeline/src/chisq.py -d {data} {paths} -o {outpath} --pol {pol} --scat {scat}')
        
        # CPHASE
        outpath_tri=f'{basedir}/evaluation/{resultsdir}/plots/triangle_{pol}_{modelname[m]["kine"]}'
        if not os.path.exists(outpath_tri+'.png'):
            os.system(f'python {basedir}/evaluation/scripts/pipeline/src/triangles.py -d {data} {paths} -o {outpath_tri} --pol {pol} --scat {scat}')
        
        # AMP
        outpath_amp=f'{basedir}/evaluation/{resultsdir}/plots/amplitude_{pol}_{modelname[m]["kine"]}'
        if not os.path.exists(outpath_amp+'.png'):
            os.system(f'python {basedir}/evaluation/scripts/pipeline/src/amplitudes.py -d {data} {paths} -o {outpath_amp} --pol {pol} --scat {scat}')

    # NXCORR
    if scat!='none':
        pathmovt  = f'{basedir}/{subdir}/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}_{scat}.hdf5'
    else:
        pathmovt  = f'{basedir}/{subdir}/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}.hdf5'

    #paths=f'--truthmv {pathmovt} --kinemv {pathmov} --starmv {pathmov2} --ehtmv {pathmov3} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    paths=f'--truthmv {pathmovt} --kinemv {pathmov} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/nxcorr_{modelname[m]["kine"]}'
    if not os.path.exists(outpath+'.png'):
        os.system(f'python {basedir}/evaluation/scripts/pipeline/src/nxcorr.py --data {data} {paths} -o {outpath} --scat {scat}')
          
    # Stokes I GIF  
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/gif_{modelname[m]["truth"]}'
    if not os.path.exists(outpath+'.gif'):
        os.system(f'python {basedir}/evaluation/scripts/pipeline/src/gif.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    # Stokes P GIF 
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/gif_lp_{modelname[m]["truth"]}'
    if not os.path.exists(outpath+'.gif'):
        os.system(f'python {basedir}/evaluation/scripts/pipeline/src/gif_lp.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    # Stokes V GIF 
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/gif_cp_{modelname[m]["truth"]}'
    if not os.path.exists(outpath+'.gif'):
        os.system(f'python {basedir}/evaluation/scripts/pipeline/src/gif_cp.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    # Pol net, avg 
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/pol_{modelname[m]["kine"]}'
    if not os.path.exists(outpath+'.png'):
        os.system(f'python {basedir}/evaluation/scripts/pipeline/src/pol.py --data {data} {paths} -o {outpath} --scat {scat}')
        
    # REx ring characterization
    if models[m] =='ring':
        outpath =f'{basedir}/evaluation/{resultsdir}/plots/rex_{modelname[m]["kine"]}'
        if not os.path.exists(outpath+'.png'):
            os.system(f'python {basedir}/evaluation/scripts/pipeline_rex_test/src/rex.py --data {data} {paths} -o {outpath}')
          
##############################################################################################