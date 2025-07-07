##############################################################################################
# Author: Rohan Dahale, Date: 23 July 2024, Version=v1.0
##############################################################################################

import os
import glob
import ehtim as eh
codedir=os.getcwd()+'/src'

def evaluation(subdir='./submissions/', resultsdir='./results/', eval_chisq=True, eval_closure_phases=True, eval_amplitudes=True, plot_gifs=True, eval_nxcorr=True, plot_mbreve=True, plot_vis_var=True, eval_rex=True, eval_VIDA_pol=True, eval_VIDA=True, eval_pattern_speed=True, cores=8):
    if not os.path.exists(f'{resultsdir}'):
        os.makedirs(f'{resultsdir}')
    # Dictionary of vida templates available
    modelsvida={
            'crescent'  : 'mring_0_1', 
            'ring'      : 'mring_0_0', 
            'disk'      : 'disk_1', 
            'edisk'     : 'stretchdisk_1',
            'double'    : 'gauss_2', 
            'point'     : 'gauss_2',
            'gaussian'  : 'gauss_1'
            }

    # Crop data and keep between 10.89 UT and 14.05
    fulldatalist=sorted(glob.glob(subdir+'*.uvfits'))
    for d in fulldatalist:
        obs = eh.obsdata.load_uvfits(d)
        #if obs.tstart<10.85 or obs.tstop>14.05:
        #    obs = obs.flag_UT_range(UT_start_hour=10.85, UT_stop_hour=14.05, output='flagged')
        #    obs.tstart, obs.tstop = obs.data['time'][0], obs.data['time'][-1]
        #    obs.save_uvfits(d)
    
    # Find LO band data for all models
    datalist=[]
    for d in fulldatalist:
        if d.find('HI')==-1:
            datalist.append(d)

    for d in datalist:
        dataf = os.path.basename(d)
        datap = subdir + dataf
        #print(dataf)
        
        model = dataf.split('_')[0]
        band =  dataf.split('_')[1]
        scat = dataf.split('_')[2][:-7]#[2]
                
        # Sort reconstructions by pipeline
        #movielist = sorted(glob.glob(subdir+f'{model}_*_{scat}_*.hdf5'))
        movielist = sorted(glob.glob(subdir+f'{model}_*.hdf5'))
        resolve=[]
        kine=[]
        ehtim=[]
        doghit=[]
        ngmem=[]
        truth=[]
        modeling=[]

        for m in movielist:
            n = os.path.basename(m)
            if n.find('resolve')!=-1:
                resolve.append(m)
            elif n.find('kine')!=-1:
                kine.append(m)
            elif n.find('ngmem')!=-1:
                ngmem.append(m)
            elif n.find('doghit')!=-1:
                doghit.append(m)
            elif n.find('ehtim')!=-1:
                ehtim.append(m)
            elif n.find('modeling')!=-1:
                modeling.append(m)
            elif n.find('truth')!=-1 and m.find('HI')==-1:
                truth.append(m)

        movies={
                'truth'  : truth,
                'kine'   : kine,
                'resolve': resolve,
                'doghit' : doghit,
                'ngmem'  : ngmem,
                'ehtim'  : ehtim,
                'modeling': modeling
        }


        if dataf.find('ring')!=-1 and dataf.find('mring')==-1:
            vida_modelname = 'ring'
            modeltype      = 'ring'
            hotspot        = False
        elif dataf.find('disk')!=-1 and dataf.find('edisk')== -1:
            vida_modelname = 'disk'
            modeltype      = 'non-ring'
            hotspot        = False
        elif dataf.find('edisk')!=-1:
            vida_modelname = 'edisk'
            modeltype      = 'non-ring'
            hotspot        = False
        elif dataf.find('double')!=-1:
            vida_modelname = 'double'
            modeltype      = 'non-ring'
            hotspot        = False
        elif dataf.find('point')!=-1:
            vida_modelname = 'point'
            modeltype      = 'non-ring'
            hotspot        = False
        elif dataf.find('mring+hs')!=-1 and dataf.find('ring')!=-1 and dataf.find('mring')!=-1:
            hotspot        = True
            vida_modelname_dynamic = 'gaussian'
            vida_modelname = 'crescent'
            modeltype      = 'ring'
        elif dataf.find('SGRA')!=-1 or dataf.find('sgra')!=-1 or dataf.find('hops') != -1:
            hotspot=True
            vida_modelname_dynamic = 'gaussian'
            vida_modelname = 'crescent'
            modeltype      = 'ring'
        elif dataf.find('mring+hs')==-1 and dataf.find('ring')!=-1:
            hotspot        = True
            vida_modelname_dynamic = 'gaussian'
            vida_modelname = 'crescent'
            modeltype      = 'ring'
        elif dataf.find('grmhd2+hs')!=-1:
            hotspot        = True
            vida_modelname_dynamic = 'gaussian'
            vida_modelname = 'crescent'
            modeltype      = 'ring'
        else:
            vida_modelname = 'crescent'
            modeltype      = 'ring'
            hotspot        = False

        modelname      = dataf[:-7]
        template       = modelsvida[vida_modelname] 

        if dataf.find('sgra') != -1 or dataf.find('SGRA') != -1 or dataf.find('hops') != -1:
            modelname='sgra'
            #scat = 'dsct'

        mvsort = dict()
        for p in movies.keys():
            movie_types = movies[p]
            mvsort[p] = dict()
            for movie in movie_types:
                movief = os.path.basename(movie)
                moviep = subdir + movief
                model = movief.split('_')[0]
                if movief.find(model)!=-1:
                    mvsort[p][model] = moviep   

        model = dataf.split('_')[0]

        print('')
        print(f'Model: {model}')
        print(f'Type:  {modeltype}')
        print(f'VIDA Template: {template}')
        print(f'Results Directory: {resultsdir}')
        print(f'Cores: {cores}')
        print('')
        print(f'Data:  {datap}')
        print(f'Scattering Type:  {scat}')
        
        if modelname!='sgra':
            try:
                truthmv = mvsort['truth'][model]
                print(f'Truth: {truthmv}')
            except:
                truthmv='none'
                print(f'Truth: {truthmv}')

        for p in movies.keys():
            if p=='kine':
                try:
                    kinemv = mvsort['kine'][model]
                    print(f'{p}: {kinemv}')
                except:
                    kinemv='none'
                    print(f'{p}: {kinemv}')
            if p=='resolve':
                try:
                    resmv = mvsort['resolve'][model]
                    print(f'{p}: {resmv}')
                except:
                    resmv='none'
                    print(f'{p}: {resmv}')
            if p=='ehtim':
                try:
                    ehtmv = mvsort['ehtim'][model]
                    print(f'{p}: {ehtmv}')
                except:
                    ehtmv='none'
                    print(f'{p}: {ehtmv}')
            if p=='ngmem':
                try:
                    ngmv = mvsort['ngmem'][model]
                    print(f'{p}: {ngmv}')
                except:
                    ngmv='none'
                    print(f'{p}: {ngmv}')
            if p=='doghit':
                try:
                    dogmv = mvsort['doghit'][model]
                    print(f'{p}: {dogmv}')
                except:
                    dogmv='none'
                    print(f'{p}: {dogmv}')
            if p=='modeling':
                try:
                    modelingmv = mvsort['modeling'][model]
                    print(f'{p}: {modelingmv}')
                except:
                    modelingmv='none'
                    print(f'{p}: {modelingmv}')
        ##############################################################################################
        # Chi-squares, closure triangles, ampltitudes
        ##############################################################################################
        paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
        data=datap

        #pollist=['I', 'Q', 'U']
        #for pol in pollist:
        #########################
        #CHISQ
        #########################
        if eval_chisq:
            outpath=f'{resultsdir}/{model}_{band}_{scat}_chisq'
            if not os.path.exists(outpath+'.png'):
                os.system(f'python {codedir}/chisq.py -d {data} {paths} -o {outpath} --scat {scat}')

            #########################
            # CPHASE
            #########################
            if eval_closure_phases:
                outpath_tri=f'{resultsdir}/{model}_{band}_{scat}_triangle_{pol}'
                if not os.path.exists(outpath_tri+'.png'):
                    os.system(f'python {codedir}/triangles.py -d {data} {paths} -o {outpath_tri} --pol {pol} --scat {scat}')

            #########################
            # AMP
            #########################
            if eval_amplitudes:
                outpath_amp=f'{resultsdir}/{model}_{band}_{scat}_amplitude_{pol}'
                if not os.path.exists(outpath_amp+'.png'):
                    os.system(f'python {codedir}/amplitudes.py -d {data} {paths} -o {outpath_amp} --pol {pol} --scat {scat}')

        ##############################################################################################
        # Static Part of the Movie
        ##############################################################################################
        outpath =f'{resultsdir}/static/'
        if not os.path.exists(f'{outpath}'):
            os.makedirs(f'{outpath}')

        if not os.path.exists(f'{resultsdir}/static/{os.path.basename(kinemv)}'):
            if modelname!='sgra':
                paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                os.system(f'python {codedir}/static.py --data {data} {paths} -o {outpath} --scat {scat}')
            else:
                paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                os.system(f'python {codedir}/static.py --data {data} {paths} -o {outpath} --scat {scat}')
    
        ##############################################################################################
        # Dynamic Part of the Movie
        ##############################################################################################
        outpath =f'{resultsdir}/dynamic/'
        if not os.path.exists(f'{outpath}'):
            os.makedirs(f'{outpath}')

        if not os.path.exists(f'{resultsdir}/dynamic/{os.path.basename(kinemv)}'):
            if modelname!='sgra':
                paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                os.system(f'python {codedir}/dynamic.py --data {data} {paths} -o {outpath} --scat {scat}')
            else:
                paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                os.system(f'python {codedir}/dynamic.py --data {data} {paths} -o {outpath} --scat {scat}')
                
        ##############################################################################################
        # Regrid Full Movie
        ##############################################################################################
        outpath =f'{resultsdir}/static+dynamic/'
        if not os.path.exists(f'{outpath}'):
            os.makedirs(f'{outpath}')

        if not os.path.exists(f'{resultsdir}/static+dynamic/{os.path.basename(kinemv)}'):
            if modelname!='sgra':
                paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                os.system(f'python {codedir}/regrid.py --data {data} {paths} -o {outpath} --scat {scat}')
            else:
                paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                os.system(f'python {codedir}/regrid.py --data {data} {paths} -o {outpath} --scat {scat}')
                
        ##############################################################################################
        # NXCORR
        ##############################################################################################
        if eval_nxcorr:
            if modelname!='sgra':
                paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                #outpath =f'{resultsdir}/{model}_{band}_{scat}_nxcorr_threshold'
                #if not os.path.exists(outpath+'.png'):
                #    os.system(f'python {codedir}/nxcorr_threshold.py --data {data} {paths} -o {outpath} --scat {scat}')

                #outpath =f'{resultsdir}/{model}_{band}_{scat}_nxcorr_static'
                #if not os.path.exists(outpath+'.png'):
                #    os.system(f'python {codedir}/nxcorr_static.py --data {data} {paths} -o {outpath} --scat {scat}')

                #outpath =f'{resultsdir}/{model}_{band}_{scat}_nxcorr_dynamic'
                #if not os.path.exists(outpath+'.png'):
                    #os.system(f'python {codedir}/nxcorr_dynamic.py --data {data} {paths} -o {outpath} --scat {scat}')
                    
                outpath =f'{resultsdir}/{model}_{band}_{scat}_nxcorr_threshold'
                if not os.path.exists(outpath+'.png'):
                    os.system(f'python {codedir}/nxcorr_threshold.py --data {data} {paths} -o {outpath} --scat {scat}')
        
                outpath =f'{resultsdir}/{model}_{band}_{scat}_nxcorr_static_threshold'
                if not os.path.exists(outpath+'.png'):
                    os.system(f'python {codedir}/nxcorr_static_threshold.py --data {data} {paths} -o {outpath} --scat {scat}')
                
                outpath =f'{resultsdir}/{model}_{band}_{scat}_nxcorr_dynamic_threshold'
                if not os.path.exists(outpath+'.png'):
                    os.system(f'python {codedir}/nxcorr_dynamic_threshold.py --data {data} {paths} -o {outpath} --scat {scat}')
                                    
                #outpath =f'{resultsdir}/{model}_{band}_{scat}_rssd'
                #if not os.path.exists(outpath+'.png'):
                #    os.system(f'python {codedir}/rssd.py --data {data} {paths} -o {outpath} --scat {scat}')
                #    
                #outpath =f'{resultsdir}/{model}_{band}_{scat}_rssd_static'
                #if not os.path.exists(outpath+'.png'):
                #    os.system(f'python {codedir}/rssd_static.py --data {data} {paths} -o {outpath} --scat {scat}')
                #    
                #outpath =f'{resultsdir}/{model}_{band}_{scat}_rssd_dynamic'
                #if not os.path.exists(outpath+'.png'):
                #    os.system(f'python {codedir}/rssd_dynamic.py --data {data} {paths} -o {outpath} --scat {scat}')
                    
                outpath =f'{resultsdir}/{model}_{band}_{scat}_weights'
                if not os.path.exists(outpath+'.png'):
                    os.system(f'python {codedir}/weights.py --data {data} {paths} -o {outpath} --scat {scat}')
        ##############################################################################################
        # MBREVE
        ##############################################################################################
        if plot_mbreve:    
            paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}' 
            outpath_mbreve=f'{resultsdir}/{model}_{band}_{scat}_mbreve.png'
            if not os.path.exists(outpath_mbreve):
                os.system(f'python {codedir}/mbreve.py -d {data} {paths} -o {outpath_mbreve} --scat {scat}')

        ##############################################################################################      
        # Stokes I GIF
        ##############################################################################################
        
        if plot_gifs:
            outpath =f'{resultsdir}/{model}_{band}_{scat}_gif'
            if not os.path.exists(outpath+'.gif'):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif.py --data {data} {paths} -o {outpath} --scat {scat}')
                else:
                    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    #paths=f'--kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif.py --data {data} {paths} -o {outpath} --scat {scat}')
        
        ##############################################################################################      
        # Stokes I, P Dynamic Component GIF
        ##############################################################################################
        if plot_gifs:
            outpath =f'{resultsdir}/{model}_{band}_{scat}_dynamic_gif'
            if not os.path.exists(outpath+'.gif'):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif_dynamic.py --data {data} {paths} -o {outpath} --scat {scat}')
                else:
                    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    #paths=f'--kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif_dynamic.py --data {data} {paths} -o {outpath} --scat {scat}')
                    

            outpath =f'{resultsdir}/{model}_{band}_{scat}_dynamic_lp_gif'
            if not os.path.exists(outpath+'.gif'):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}  --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif_dynamic_lp.py --data {data} {paths} -o {outpath} --scat {scat}')
                else:
                    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    #paths=f'--kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif_dynamic_lp.py --data {data} {paths} -o {outpath} --scat {scat}')
    
        ##############################################################################################      
        # Stokes I, P Static Component Plot
        ##############################################################################################
        outpath =f'{resultsdir}/{model}_{band}_{scat}_static'
        if modelname!='sgra':
            truthmvs=os.path.dirname(truthmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+ '/static/' +os.path.basename(truthmv)[:-5]+'.fits'
        else:
            truthmvs='none'
        if not os.path.isfile(truthmvs):
            truthmvs='none'
        kinemvs=os.path.dirname(kinemv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/static/' + os.path.basename(kinemv)[:-5]+'.fits'
        if not os.path.isfile(kinemvs):
            kinemvs='none'
        dogmvs=os.path.dirname(dogmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/static/' + os.path.basename(dogmv)[:-5]+'.fits'
        if not os.path.isfile(dogmvs):
            dogmvs='none'
        ngmvs=os.path.dirname(ngmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/static/' + os.path.basename(ngmv)[:-5]+'.fits'
        if not os.path.isfile(ngmvs):
            ngmvs='none'
        resmvs=os.path.dirname(resmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/static/' + os.path.basename(resmv)[:-5]+'.fits'
        if not os.path.isfile(resmvs):
            resmvs='none'
        ehtmvs=os.path.dirname(ehtmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/static/' + os.path.basename(ehtmv)[:-5]+'.fits'
        if not os.path.isfile(ehtmvs):
            ehtmvs='none'
        modelmvs=os.path.dirname(modelingmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/static/' + os.path.basename(modelingmv)[:-5]+'.fits'
        if not os.path.isfile(modelmvs):
            modelmvs='none'

        if plot_gifs:
            if not os.path.exists(outpath+'.png'):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmvs} --kinemv {kinemvs} --dogmv {dogmvs} --ngmv {ngmvs} --resmv {resmvs} --ehtmv {ehtmvs} --modelingmv {modelmvs}'
                    os.system(f'python {codedir}/static_image.py --data {data} {paths} -o {outpath} --scat {scat}')
                else:
                    paths=f'--kinemv {kinemvs} --dogmv {dogmvs} --ngmv {ngmvs} --resmv {resmvs} --ehtmv {ehtmvs} --modelingmv {modelmvs}'
                    os.system(f'python {codedir}/static_image.py --data {data} {paths} -o {outpath} --scat {scat}')

            outpath =f'{resultsdir}/{model}_{band}_{scat}_static_lp'
            if not os.path.exists(outpath+'.png'):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmvs} --kinemv {kinemvs} --dogmv {dogmvs} --ngmv {ngmvs} --resmv {resmvs} --ehtmv {ehtmvs} --modelingmv {modelmvs}'
                    os.system(f'python {codedir}/static_lp_image.py --data {data} {paths} -o {outpath} --scat {scat}')
                else:
                    paths=f'--kinemv {kinemvs} --dogmv {dogmvs} --ngmv {ngmvs} --resmv {resmvs} --ehtmv {ehtmvs} --modelingmv {modelmvs}'
                    os.system(f'python {codedir}/static_lp_image.py --data {data} {paths} -o {outpath} --scat {scat}')

        ##############################################################################################      
        # Stokes P GIF
        ##############################################################################################
        if plot_gifs:
            outpath =f'{resultsdir}/{model}_{band}_{scat}_gif_lp'
            if not os.path.exists(outpath+'.gif'):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif_lp.py --data {data} {paths} -o {outpath} --scat {scat}')
                else:
                    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    #paths=f'--kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/gif_lp.py --data {data} {paths} -o {outpath} --scat {scat}')

        ##############################################################################################      
        # Stokes V GIF
        ##############################################################################################
        #if plot_gifs:
        #    outpath =f'{resultsdir}/{model}_{band}_{scat}_gif_cp'
        #    if not os.path.exists(outpath+'.gif'):
        #        if modelname!='sgra':
        #            paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
        #            os.system(f'python {codedir}/gif_cp.py --data {data} {paths} -o {outpath} --scat {scat}')
        #        else:
        #            paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
        #            os.system(f'python {codedir}/gif_cp.py --data {data} {paths} -o {outpath} --scat {scat}')

        ##############################################################################################      
        # Visibility Variance
        ##############################################################################################
        if plot_vis_var:
            outpath =f'{resultsdir}/{model}_{band}_{scat}_vis_var'
            if not os.path.exists(outpath+'.png'):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/vis_variance.py --data {data} {paths} -o {outpath} --scat {scat}')
                else:
                    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/vis_variance.py --data {data} {paths} -o {outpath} --scat {scat}')

        ##############################################################################################
        # REx, VIDA ring characterization
        ##############################################################################################  
        if modeltype =='ring':
            if eval_rex:
                outpath =f'{resultsdir}/{model}_{band}_{scat}_rex'
                if not os.path.exists(outpath+'.png') and not os.path.exists(outpath+'_pol.png'):
                    if modelname!='sgra':
                        paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                        os.system(f'python {codedir}/rex.py --data {data}  --scat {scat} {paths} -o {outpath}')
                    else:
                        paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                        os.system(f'python {codedir}/rex.py --data {data} --scat {scat} {paths} -o {outpath}')

            if eval_VIDA_pol:
                outpath=f'{resultsdir}/{model}_{band}_{scat}_vida_pol.png'
                if not os.path.exists(outpath):
                    if modelname!='sgra':
                        paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                        #paths=f'--truthmv {truthmv} --kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                        os.system(f'python {codedir}/vida_pol.py --data {data} {paths} -o {outpath} -c {cores} --scat {scat}')
                        
                        truthcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_truth.csv'
                        kinecsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_kine.csv'
                        rescsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_resolve.csv'
                        ehtcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_ehtim.csv'
                        dogcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_doghit.csv'
                        ngcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_ngmem.csv'
                        modelingcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_modeling.csv'
                        
                        paths=f'--truthcsv {truthcsv} --kinecsv {kinecsv} --rescsv {rescsv} --dogcsv {dogcsv} --ngcsv {ngcsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --modelingcsv {modelingcsv}'
                        outpath =f'{resultsdir}/{model}_{band}_{scat}_vida_pol_all'
                        os.system(f'python {codedir}/merge_csv_vida_pol.py {paths} -o {outpath}')
                        
                    else:
                        paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                        #paths=f'--kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                        os.system(f'python {codedir}/vida_pol.py --data {data} {paths} -o {outpath} -c {cores} --scat {scat}')
                        
                        kinecsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_kine.csv'
                        rescsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_resolve.csv'
                        ehtcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_ehtim.csv'
                        dogcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_doghit.csv'
                        ngcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_ngmem.csv'
                        modelingcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_pol_modeling.csv'
                        
                        paths=f'--kinecsv {kinecsv} --rescsv {rescsv} --dogcsv {dogcsv} --ngcsv {ngcsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --modelingcsv {modelingcsv}'
                        outpath =f'{resultsdir}/{model}_{band}_{scat}_vida_pol_all'
                        os.system(f'python {codedir}/merge_csv_vida_pol.py {paths} -o {outpath}')

        ##############################################################################################
        # VIDA
        ############################################################################################## 
        if eval_VIDA:     
            outpath =f'{resultsdir}/{model}_{band}_{scat}_vida'
            if not os.path.exists(outpath):
                if modelname!='sgra':
                    paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    #paths=f'--truthmv {truthmv} --kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/vida.py --data {data} --model {vida_modelname} --template {template} {paths} -o {outpath} -c {cores} --scat {scat}')
                    
                    truthcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_truth.csv'
                    kinecsv = f'{resultsdir}/{model}_{band}_{scat}_vida_kine.csv'
                    rescsv = f'{resultsdir}/{model}_{band}_{scat}_vida_resolve.csv'
                    ehtcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_ehtim.csv'
                    dogcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_doghit.csv'
                    ngcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_ngmem.csv'
                    modelingcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_modeling.csv'
                        
                    paths=f'--truthcsv {truthcsv} --kinecsv {kinecsv} --rescsv {rescsv} --dogcsv {dogcsv} --ngcsv {ngcsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --modelingcsv {modelingcsv}'
                    outpath =f'{resultsdir}/{model}_{band}_{scat}_vida_all'
                    os.system(f'python {codedir}/merge_csv_vida.py {paths} -o {outpath}')
                        
                else:
                    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv} --modelingmv {modelingmv}'
                    #paths=f'--kinemv {kinemv} --resmv {resmv} --modelingmv {modelingmv}'
                    os.system(f'python {codedir}/vida.py --data {data} --model {vida_modelname} --template {template} {paths} -o {outpath} -c {cores} --scat {scat}')
                    
                    kinecsv = f'{resultsdir}/{model}_{band}_{scat}_vida_kine.csv'
                    rescsv = f'{resultsdir}/{model}_{band}_{scat}_vida_resolve.csv'
                    ehtcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_ehtim.csv'
                    dogcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_doghit.csv'
                    ngcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_ngmem.csv'
                    modelingcsv = f'{resultsdir}/{model}_{band}_{scat}_vida_modeling.csv'
                        
                    paths=f'--kinecsv {kinecsv} --rescsv {rescsv} --dogcsv {dogcsv} --ngcsv {ngcsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --modelingcsv {modelingcsv}'
                    outpath =f'{resultsdir}/{model}_{band}_{scat}_vida_all'
                    os.system(f'python {codedir}/merge_csv_vida.py {paths} -o {outpath}')
                    
        ##############################################################################################
        # Interpolated Movie, Averaged Movie, VIDA Ring, Cylinder
        ##############################################################################################
        if eval_pattern_speed:
            plist=[]
            nlist=[]

            if modelname!='sgra':
                if truthmv!='none':
                    plist.append(truthmv)
                    nlist.append('truth')   
            if kinemv!='none':
                plist.append(kinemv)
                nlist.append('kine')
            if resmv!='none':
                plist.append(resmv)
                nlist.append('resolve')
            if ehtmv!='none':
                plist.append(ehtmv)
                nlist.append('ehtim')
            if dogmv!='none':
                plist.append(dogmv)
                nlist.append('doghit')
            if ngmv!='none':
                plist.append(ngmv)
                nlist.append('ngmem')
            if modelingmv!='none':
                plist.append(modelingmv)
                nlist.append('modeling')

            if modeltype =='ring':
                if not os.path.exists(f'{resultsdir}/patternspeed'):
                    os.makedirs(f'{resultsdir}/patternspeed')

                for i in range(len(plist)):
                    # Interpolated Movies
                    input= plist[i]
                    output=f'{resultsdir}/patternspeed/{os.path.basename(plist[i])}'
                    os.system(f'python {codedir}/hdf5_standardize.py -i {input} -o {output} -d {data}')
                    #Average Movies
                    #input=f'{resultsdir}/patternspeed/{os.path.basename(plist[i])}'
                    #fits=os.path.basename(plist[i])[:-5]+'.fits'
                    #output=f'{resultsdir}/patternspeed/{fits}'
                    #if nlist[i]=='truth':
                    #    os.system(f'python {codedir}/avg_frame.py -i {input} -o {output} --truth --scat {scat}')
                    #else:
                    #    os.system(f'python {codedir}/avg_frame.py -i {input} -o {output} --scat {scat}')
                    # VIDA Ring
                    fits=os.path.basename(plist[i])[:-5]+'.fits'
                    path=f'{resultsdir}/patternspeed/{fits}'
                    outpath = path[:-5]+'.csv'
                    #if not os.path.exists(outpath):    
                    #    os.system(f'julia {codedir}/ring_extractor.jl --in {path} --out {outpath}')
                    #    print(f'{os.path.basename(outpath)} created!')
                    # Cylinder
                    ipathmov=f'{resultsdir}/patternspeed/{os.path.basename(plist[i])}'
                    #ringpath = ipathmov[:-5]+'.csv'
                    outpath  = ipathmov[:-5]
                    #os.system(f'python {codedir}/cylinder.py {ipathmov} {ringpath} {outpath}')
                    os.system(f'python {codedir}/cylinder.py {ipathmov} {outpath}')
                    #os.system(f'python {codedir}/cylinder_xi_crit_optimization_clean.py {ipathmov} {outpath}')
                    
        if hotspot:          
            ##############################################################################################
            # VIDA Dynamic Component
            ##############################################################################################  
            if eval_VIDA:
                if modelname!='sgra':
                    truthmvd=os.path.dirname(truthmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+ '/dynamic/' +os.path.basename(truthmv)
                else:
                    truthmvd='none'
                if not os.path.isfile(truthmvd):
                    truthmvd='none'
                kinemvd=os.path.dirname(kinemv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/dynamic/' + os.path.basename(kinemv)
                print(kinemvd)
                if not os.path.isfile(kinemvd):
                    kinemvd='none'
                dogmvd=os.path.dirname(dogmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/dynamic/' + os.path.basename(dogmv)
                if not os.path.isfile(dogmvd):
                    dogmvd='none'
                ngmvd=os.path.dirname(ngmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/dynamic/' + os.path.basename(ngmv)
                if not os.path.isfile(ngmvd):
                    ngmvd='none'
                resmvd=os.path.dirname(resmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/dynamic/' + os.path.basename(resmv)
                if not os.path.isfile(resmvd):
                    resmvd='none'
                ehtmvd=os.path.dirname(ehtmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/dynamic/' + os.path.basename(ehtmv)
                if not os.path.isfile(ehtmvd):
                    ehtmvd='none'
                modelingmvd=os.path.dirname(modelingmv).replace(os.path.basename(os.path.normpath(subdir)), os.path.basename(os.path.normpath(resultsdir)))+  '/dynamic/' + os.path.basename(modelingmv)
                if not os.path.isfile(modelingmvd):
                    modelingmvd='none'

                template = modelsvida[vida_modelname_dynamic]
                outpath =f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida'
                if not os.path.exists(outpath):
                    if modelname!='sgra':
                        paths=f'--truthmv {truthmvd} --kinemv {kinemvd} --dogmv {dogmvd} --ngmv {ngmvd} --resmv {resmvd} --ehtmv {ehtmvd} --modelingmv {modelingmvd}'
                        #paths=f'--truthmv {truthmvd} --kinemv {kinemvd} --resmv {resmvd} --modelingmv {modelingmvd}'
                        os.system(f'python {codedir}/vida.py --data {data} --model {vida_modelname_dynamic} --template {template} {paths} -o {outpath} -c {cores} --scat {scat}')
                        
                        truthcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_truth.csv'
                        kinecsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_kine.csv'
                        rescsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_resolve.csv'
                        ehtcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_ehtim.csv'
                        dogcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_doghit.csv'
                        ngcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_ngmem.csv'
                        modelingcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_modeling.csv'
                        
                        paths=f'--truthcsv {truthcsv} --kinecsv {kinecsv} --rescsv {rescsv} --dogcsv {dogcsv} --ngcsv {ngcsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --modelingcsv {modelingcsv}'
                        outpath =f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_all'
                        os.system(f'python {codedir}/merge_csv_vida.py {paths} -o {outpath}')
                    
                    else:
                        paths=f'--kinemv {kinemvd} --dogmv {dogmvd} --ngmv {ngmvd} --resmv {resmvd} --ehtmv {ehtmvd} --modelingmv {modelingmvd}'
                        #paths=f'--kinemv {kinemvd} --resmv {resmvd} --modelingmv {modelingmvd}'
                        os.system(f'python {codedir}/vida.py --data {data} --model {vida_modelname_dynamic} {paths} --template {template} -o {outpath} -c {cores} --scat {scat}')
                        
                        kinecsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_kine.csv'
                        rescsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_resolve.csv'
                        ehtcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_ehtim.csv'
                        dogcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_doghit.csv'
                        ngcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_ngmem.csv'
                        modelingcsv = f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_modeling.csv'
                        
                        paths=f'--kinecsv {kinecsv} --rescsv {rescsv} --dogcsv {dogcsv} --ngcsv {ngcsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --modelingcsv {modelingcsv}'
                        outpath =f'{resultsdir}/{model}_{band}_{scat}_dynamic_vida_all'
                        os.system(f'python {codedir}/merge_csv_vida.py {paths} -o {outpath}')

    