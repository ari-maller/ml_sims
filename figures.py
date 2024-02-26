import os
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import optuna

import machine_learn as ml

def tng_min_vals():
    m_dm =np.array([0.45,7.5,40])*1.e6 #dark matter particle mass for 50, 100 and 300
    m_b = np.array([0.08,1.4,11])*1.e6
    m_dmo = np.array([0.55,8.9,103])*1.e6
    epsilon_dm = np.array([0.29,0.74,1.48]) #grav. softening
    return {'m_dm':m_dm,'m_b':m_b,'m_dmo':m_dmo,'epsilon':epsilon_dm}

def sam_paper_sample(df_orig, mass_cut = 1.e9, min_fdisk = 0.0205,
                     fname='tng300_sam_paper.h5', check=False):
    '''only centrals and only logMstar > mass_cut, also default remove 
        low fdisk galaxies that are disky. This can be ignored by setting
        min_fdisk=0'''
    df = df_orig.copy() #don't change original if you need to make other samples
    df.drop(['GalpropRdisk','GalpropRbulge','GalpropMH2', 'GalpropMHI', 
             'GalpropMHII'],axis=1,inplace=True) #only keep Mcold
    df.drop(['GalpropSfrave100myr','GalpropSfrave1gyr','GalpropSfrave20myr'],
            axis=1, inplace=True) #only keep Sfr (instantaneous)
    df.drop(['GalpropX', 'GalpropVx', 'GalpropY', 'GalpropVy', 'GalpropZ',
             'GalpropVz'],axis=1,inplace=True)
    df.drop(['HalopropMdot_eject','HalopropMdot_eject_metal',
             'HalopropMaccdot_metal','HalopropMaccdot_reaccreate_metal'],
             axis=1,inplace=True) #all zero
    df.drop(['GalpropMaccdot','HalopropMaccdot_pristine'],
            axis =1, inplace=True) #mostly zero
    df.drop(['GalpropMdisk'],axis=1,inplace=True) #added not needed
    #make cuts to sample 
    mask1 = (df['GalpropMstar'] > mass_cut) & (df['GalpropSatType']==False)
    #drop fields taht make no sense when only centrals
    df.drop(['GalpropSatType','GalpropRfric','GalpropTsat','HalopropMvir','GalpropMstrip',
             'HalopropMaccdot_radio'],axis=1,inplace=True) #not for centrals
    print(f'Number of central galaxies with mass greater than {mass_cut} = {mask1.sum()}')  
    fdisk = (df['GalpropMstar']+df['GalpropMcold'])/df['GalpropMvir']
    mask2 =  (df['GalpropMbulge']/df['GalpropMstar'] > 0.4) | (fdisk > min_fdisk)
    mask = mask1 & mask2
    print(f'Number fdisk less than {min_fdisk} and B/T less than 0.4 is {mask1.sum() - mask.sum()}')
    df=df[mask].copy()

    if check:
        for field in df.columns.values:
            print(field,df[field].min(),df[field].max())
    if fname:
        df.to_hdf(fname, key='s', mode='w')
    print(f'Dataframe now has the shape {df.shape}')
    return df

def morphology_bins(df,bins=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    bulge_fraction = df['GalpropMbulge']/df['GalpropMstar']
    df['morph_group'] = np.digitize(bulge_fraction,bins) 
    return df

def spin_effective(df):
    df['spin_eff'] = df['HalopropSpin']
    df['spin_eff'][df['HalopropSpin'] < 0.02] = 0.02
    return df

#functions for figures 
def stats_in_bins(df, xfield, yfield, bins):
    N = len(bins)-1
    xvals = np.zeros(N)
    avg = np.zeros(N)
    std = np.zeros(N)
    fields = df.columns.values
    if xfield not in fields:
        print(f"error: {xfield} \n {fields}")
    if yfield not in fields:
        print(f"error: {yfield} \n {fields}")
    for i in range(N):
        mask = (df[xfield] > bins[i]) & (df[xfield] < bins[i+1])
        xvals[i] = np.mean(df[xfield][mask])
        avg[i] = np.mean(df[yfield][mask] )
        std[i] = np.std(df[yfield][mask])
    
    return xvals,avg,std
    
def mass_size_vals(df,bins=[9,9.5,10,10.5,11,11.5], sam=True):
    if sam:
        df['GalpropLogMstar'] = np.log10(df['GalpropMstar'])
        df['GalpropLogRstar'] = np.log10(df['GalpropHalfmassRadius'])
        results = stats_in_bins(df,'GalpropLogMstar','GalpropLogRstar',bins=bins)
    else:
        df['SubhaloLogMstar'] = np.log10(df['SubhaloMstar'])
        df['SubhaloLogRstar'] = np.log10(df['SubhaloRstar'])
        results = stats_in_bins(df,'SubhaloLogMstar','SubhaloLogRstar',bins=bins)
    return results

def mass_size(sim_mMstar,sim_mRstar,sim_stdRstar,
              sam_mMstar,sam_mRstar,sam_stdRstar, save=False,
              label_sim='SIM std',label_sam = 'SAM std'): #figure 1
    '''show mean mass-size relation for sim and sam'''
    plt.plot(sim_mMstar, sim_mRstar, '-', color='green')
    plt.fill_between(sim_mMstar, sim_mRstar - sim_stdRstar, sim_mRstar + sim_stdRstar,
                 color='green', alpha=0.2, label=label_sim)
    plt.plot(sam_mMstar, sam_mRstar, '-', color='darkorange')
    plt.fill_between(sam_mMstar, sam_mRstar - sam_stdRstar, sam_mRstar + sam_stdRstar,
                 color='darkorange', alpha=0.2, label=label_sam)

    plt.title('TNG300 SIM vs SAM: Size vs Mstar')
    plt.xlabel('$ Log_{10} $ Mstar $[M_\odot]$',fontsize='x-large')
    plt.ylabel('$ Log_{10}  R_{50}$ [kpc] ',fontsize='x-large')
    plt.legend(loc='lower right')
    if save:
        plt.savefig('Size_vs_Mstar.pdf')
    else:
        plt.show()


def fig_group_importance(datasets, number_features=5):

    plt.figure(figsize=(20,6))

    for i, _dataset in enumerate(datasets):
        dataset = _dataset.copy()
    
        dataset.loc[1:,'r_sq_score'] = dataset.loc[:,'r_sq_score'].diff()[1:]

        importances = dataset.loc[:number_features -1, 'r_sq_score']
        importances.index = dataset.loc[:number_features-1,'features']
    
    
    for l in [l for l in all_top_n_feats if l not in importances.index]:
        importances[l] = 0
        
    importances.sort_index(inplace=True)
    color_list = cm.Spectral_r(30*i+20)
#     print(color_list)
    plt.bar(np.arange(len(importances))+0.05*(-i), importances, 
        align="center", width=0.2, alpha = 0.5, label = datasets_names[i], 
        color = color_list)
    
    plt.xticks(range(len(list(importances.index))), 
               labels = list(importances.index), rotation=90)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    plt.ylabel(r'Incremental $R^{2}$ score by feature')

    plt.legend(fontsize = 12)  
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.2)
    plt.savefig('group_feature.pdf')

def fig1(df,df_sim):
    '''comparison of sizes in SAM and SIM'''
    sam_mMstar,sam_mRstar,sam_stdRstar = mass_size_vals(df, sam=True)
    sim_mMstar,sim_mRstar,sim_stdRstar = mass_size_vals(df_sim, sam=False)
    mass_size(sam_mMstar,sam_mRstar,sam_stdRstar,
        sim_mMstar,sim_mRstar,sim_stdRstar)

def fig1_alt(df):
    df['log $M_{star}$'] = np.log10(df['GalpropMstar'])
    df['log $R_{50}$'] = np.log10(df['GalpropHalfmassRadius'])
    df['$M_{star} / M_{vir}$'] = df['GalpropMstar'] / df['GalpropMvir']
    df['$R_{50} / R_{vir}$'] = df['GalpropHalfmassRadius']/df['GalpropRhalo']

    g = sns.jointplot(data=df, x = 'log $M_{star}$', y = 'log $R_{50}$',
        kind = 'hist', space=0.0, )
    g.ax_joint.set_xlim([9,11])
    g.ax_joint.set_ylim([-0.2,1.1])
    plt.xlabel('log $M_{star}$', fontsize='x-large')
    plt.ylabel('log $R_{50}$', fontsize='x-large')
    plt.savefig('fig1a.pdf')
    plt.clf()
    g2 = sns.jointplot(data = df, x = '$M_{star} / M_{vir}$', 
        y = '$R_{50} / R_{vir}$', kind='hist', space=0.0)
    g2.ax_joint.set_xlim([0,0.05])
    g2.ax_joint.set_ylim([0,0.03])
    plt.savefig('fig1b.pdf')
#    fig, axs = plt.subplots(ncols=2,figsize = (9,4))
#    axs[0].hist(df['log $M_{star}$'],range=[9,11.5],bins=500)
#    axs[1].hist(df['$M_{star} / M_{vir}$'],range=[0,0.1],bins=500)
#    axs[0].hist(df['log $R_{50}$'],range=[-0.5,1.5],bins=500)
#    axs[1].hist(df['$R_{50} / R_{vir}$'],range=[0,0.05],bins=500)    
#    sns.kdeplot(data=df, x='log $M_{star}$', y = 'log $R_{50}$', alpha=0.3,
#        levels = [0.05,0.25,0.5,0.75,1.0], ax=axs[0], fill=True,
#        bw_adjust=1.5, cut=0.0)
#    sns.kdeplot(data=df, x='$M_{star} / M_{vir}$', y = '$R_{50} / R_{vir}$',
#        alpha=0.3,levels = [0.05,0.25,0.5,0.75,1.0], ax=axs[1], fill=True, 
#        bw_adjust=1.5, cut=0.0)
#    axs[0].set_xlim([9,11.5])
#    axs[0].set_ylim([-0.5,1.5])
#    axs[1].set_xlim([0,0.1])
#    axs[1].set_ylim([0,0.05])
#    plt.show()


def pred_true_plot(df, xrange = None, axis = None, 
        label = None, halo_scale = False, log = False):
    '''default set to raw target, assumes given in log'''
    target='R50'
    logtarget='log(R50)'
    if log:
        df['yp'] = df['y_pred']
        df['yt'] = df['y_true']
    else:
        df['yp'] = 10**df['y_pred']
        df['yt'] = 10**df['y_true'] 
    
    if halo_scale: #TODO instead of Rhalo just 3rd column
        df['yp'] = df['yp'] * df['Rhalo']
        df['yt'] = df['yt'] * df['Rhalo']
        logtarget='log(R_{50}/R_{vir})'
    r2 = ml.R2(df['yp'],df['yt'])
    r2_target = ml.R2(df['y_pred'],df['y_true'])
    mape = ml.mean_absolute_percent_error(df['yp'],df['yt'])
   
    if xrange==None:
        xrange=[np.min(df['yt']),np.max(df['yt'])]
    if axis==None:
        fig, axis = plt.subplots()
    if df.shape[0] > 10000:
        df_s = df.sample(n=10000)
#        axis.scatter(df_s['y_true'],df_s['y_pred'],s=1,marker='.',alpha=0.3,color='green')
        sns.kdeplot(data=df_s, x='yt', y = 'yp', alpha=0.3, gridsize=200, bw_adjust=0.5,
            levels = [0.1,0.25,0.5,1.0], ax=axis, fill=True, cmap='brg')
    else:
        axis.scatter(df['yt'],df['yp'],s=1,marker='.',alpha=0.3,color='green')
    
    axis.plot(xrange,xrange,linestyle='dotted')
    axis.set_xlim(xrange)
    axis.set_ylim(xrange)
    axis.annotate('$R^2_{log{R_{50}}}$ ='+f'{0.01*np.round(100*r2_target):.2f}',(0.1,0.9), xycoords='axes fraction')
    axis.annotate('$R^2_{R_{50}}$ ='+f'{0.01*np.round(100*r2):.2f}',(0.1,0.8), xycoords='axes fraction')
    axis.annotate('MAPE$_{R_{50}}$ ='+f'{np.round(100*mape):.0f}%',(0.1,0.7), xycoords='axes fraction')
    if label:
        axis.set_xlabel('true '+label, fontsize='x-large')
        axis.set_ylabel('predicted '+label, fontsize='x-large')
    else:
        axis.set_xlabel('')
        axis.set_ylabel('')

def get_fnames():
    '''returns dictionary of filenames for different plots based on keywords'''
    fnames_dict={
        'RFdmo':'fit_GalpropHalfmassRadius_RF_dmo_hp-ne175.mdNone.mss4.msl2.mf0.6.csv',
        'RFall':'fit_GalpropHalfmassRadius_RF_hp-ne175.mdNone.mss4.msl2.mf0.6.csv',
        'RFdmo_hs':'fit_GalpropHalfmassRadius_hs_RF_dmo_hs_hp-ne175.mdNone.mss4.msl2.mf0.6.csv',
        'RFall_hs':'fit_GalpropHalfmassRadius_hs_RF_hs_hp-ne175.mdNone.mss4.msl2.mf0.6.csv',
        'SVdmo':'fit_GalpropHalfmassRadius_SV_dmo_hp-krbf.C50.g0.02.e0.1.csv',
        'SVall':'fit_GalpropHalfmassRadius_SV_hp-krbf.C50.g0.02.e0.1.csv',
        'SVdmo_hs':'fit_GalpropHalfmassRadius_hs_SV_dmo_hs_hp-krbf.C50.g0.02.e0.1.csv',
        'SVall_hs':'fit_GalpropHalfmassRadius_hs_SV_hs_hp-krbf.C50.g0.02.e0.1.csv',
        'RFbt1':'fit_GalpropHalfmassRadius_RF_bt1_hp-ne175.mdNone.mss4.msl2.mf0.6.csv',
        'RFbt1_hs':'fit_GalpropHalfmassRadius_hs_RF_bt1_hs_hp-ne175.mdNone.mss4.msl2.mf0.6.csv',
        'SVbt1':'fit_GalpropHalfmassRadius_SV_bt1_hp-krbf.C50.g0.02.e0.1.csv',
        'SVbt1_hs':'fit_GalpropHalfmassRadius_hs_SV_bt1_hs_hp-krbf.C50.g0.02.e0.1.csv'
        }
    return fnames_dict

def fig3(DMO = False):
    'true-predict plots for dark matter only'
    fnames = get_fnames()
    if DMO:
        df_rf = pd.read_csv(fnames['RFdmo'])
        df_sv = pd.read_csv(fnames['SVdmo'])
        df_rf_hs = pd.read_csv(fnames['RFdmo_hs'])
        df_sv_hs = pd.read_csv(fnames['SVdmo_hs'])
        figname = 'fig3.pdf'
    else:
        df_rf = pd.read_csv(fnames['RFall'])
        df_sv = pd.read_csv(fnames['SVall'])
        df_rf_hs = pd.read_csv(fnames['RFall_hs'])
        df_sv_hs = pd.read_csv(fnames['SVall_hs'])
        figname = 'fig4.pdf'
    #convert to r_50
    fig,axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize = (8,8))
    ax=fig.add_subplot(111,frameon=False)
    ax.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('True $R_{50}$', fontsize='xx-large', labelpad = 5)
    ax.set_ylabel('Predicted $R_{50}$', fontsize='xx-large',labelpad = 10)
#    axes[0,0].axis('equal')
#    axes[1,1].axis('equal')
    xr=[0,11]
    pred_true_plot(df_rf, xrange=xr, axis=axes[0,0], 
                   halo_scale = False)
    pred_true_plot(df_sv, xrange=xr, axis=axes[1,0], 
                   halo_scale = False)
    pred_true_plot(df_rf_hs, xrange=xr, axis=axes[0,1], 
                   halo_scale = True)
    pred_true_plot(df_sv_hs, xrange=xr, axis=axes[1,1], 
                   halo_scale = True)
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    plt.savefig(figname)

def fig2alt(scaled=False, kind = 'RF'):
    fnames = get_fnames()

    if scaled:
        xrange = [-4,-1]
        label = '$R_{50}/R_{vir}$'
        zeroD=True
        df_dmo = pd.read_csv(fnames[kind+'dmo_hs'])
        df = pd.read_csv(fnames[kind+'all_hs'])
        name = f'fig2b_{kind}.pdf'
    else:
        label = 'log $R_{50}$'
        xrange = [-0.25,1.75]
        zeroD=False
        df_dmo = pd.read_csv(fnames[kind+'dmo'])
        df = pd.read_csv(fnames[kind+'all'])
        name = f'fig2a_{kind}.pdf'

    fig,axes = plt.subplots(2,1,sharex=True, figsize = (6,8))
    ax=fig.add_subplot(111,frameon=False)
    ax.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('True '+label, fontsize='x-large')
    ax.set_ylabel('Predicted '+label, fontsize='x-large',labelpad = 10)
    axes[0].axis('equal')
    axes[1].axis('equal')
    pred_true_plot(df_dmo, xrange = xrange, axis = axes[0], halo_scale=zeroD )
    pred_true_plot(df, xrange = xrange, axis = axes[1], halo_scale=zeroD)
    plt.subplots_adjust(hspace=0.0)
    plt.savefig(name)

def fig2(scaled = False):
    fnames=get_fnames()

    if scaled:
        xrange = [-2.75,-1.1]
        label = 'log $(R_{50}/R_{vir})$'
        zeroD=True
        df_dmo_rf = pd.read_csv(fnames['RFdmo_hs'])
        df_dmo_sv = pd.read_csv(fnames['SVdmo_hs'])
        df_all_rf = pd.read_csv(fnames['RFall_hs'])
        df_all_sv = pd.read_csv(fnames['SVall_hs'])
        name = 'fig3b.pdf'
    else:
        label = 'log $R_{50}$'
        xrange = [-0.45,1.5]
        zeroD=False
        df_dmo_rf = pd.read_csv(fnames['RFdmo'])
        df_dmo_sv = pd.read_csv(fnames['SVdmo'])
        df_all_rf = pd.read_csv(fnames['RFall'])
        df_all_sv = pd.read_csv(fnames['SVall'])       
        name = f'fig3a.pdf'

    fig,axes = plt.subplots(2,2,sharex=True, sharey=True, figsize = (7,6))
    ax=fig.add_subplot(111,frameon=False)
    ax.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('True '+label, fontsize='x-large')
    ax.set_ylabel('Predicted '+label, fontsize='x-large',labelpad = 15)

#    axes[0,0].axis('equal')
    pred_true_plot(df_dmo_rf, xrange = xrange, axis = axes[0,0], zeroD=zeroD )
#    axes[0,1].axis('equal')
    pred_true_plot(df_dmo_sv, xrange = xrange, axis = axes[0,1], zeroD=zeroD )
#    axes[1,0].axis('equal')
    pred_true_plot(df_all_rf, xrange = xrange, axis = axes[1,0], zeroD=zeroD )
#    axes[1,1].axis('equal')
    axes[1,0].annotate('RF',(0.85,0.1),xycoords='axes fraction')
    pred_true_plot(df_all_sv, xrange = xrange, axis = axes[1,1], zeroD=zeroD )
    axes[1,1].annotate('SV',(0.85,0.1),xycoords='axes fraction')
    plt.subplots_adjust(hspace=0.0,wspace=0)
    plt.savefig(name)

def translate_names(namelist,return_dictionary=False):
    '''translate to paper names'''
    feature_dict={'HalopropSpin':'$\lambda$',
        'HalopropMetal_ejected':'$M_{ejected,z}$', 'HalopropMstar_diffuse':'$M_{diffuse}$',
        'HalopropMass_ejected':'$M_{ejected}$', 'HalopropMaccdot_reaccreate':'$M_{recycle}$',
        'GalpropMstar_merge':'$M_{merge}$', 'GalpropMstar_merge_hs':'$M_{merge} / M_{vir}$',
        'HalopropMhot':'$M_{hot}$', 'HalopropMhot_hs':'$M_{hot} / M_{vir}$',
        'GalpropZcold':'$Z_{cold}$','HalopropZhot':'$Z_{hot}$',
        'GalpropOutflowRate_Mass':'$\dot{M}_{out}$','GalpropMu_merger':'$\mu_{merge}$',
        'GalpropTmerger_major':'$T_{mm}$','GalpropTmerger':'$T_{merge}$',
        'GalpropSfr':'SFR','HalopropMaccdot_pristine':'$\dot{M}_{pristine}$',
        'GalpropMBH':'$M_{BH}$', 'GalpropMBH_hs':'$M_{BH} / M_{vir}$',
        'GalpropMaccdot_radio':'$\dot{M}_{BHr}$','GalpropMaccdot':'$\dot{M}_{BH}$', 
        'GalpropSigmaBulge':'$\sigma_b$','GalpropSigmaBulge_hs':'$\sigma_b/v_{vir}$',
        'GalpropVdisk':'$V_{disk}$', 'GalpropVdisk_hs':'$V_{disk} / V_{vir}$',
        'GalpropMstar':'$M_{star}$', 'GalpropMstar_hs':'$M_{star} / M_{vir}$',
        'GalpropMdisk':'$M_{d}$','GalpropMdisk_hs':'$M_{d} / M_{vir}$',
        'GalpropMbulge':'$M_{b}$','GalpropMbulge_hs':'$M_{b} / M_{vir}$',
        'GalpropMcold':'$M_{cold}$','GalpropMcold_hs':'$M_{cold} / M_{vir}$',
        'GalpropOutflowRate_Metal':'$M_{out,z}$',
        'GalpropMvir':'$M_{vir}$','GalpropVvir':'$V_{vir}$',
        'GalpropRhalo':'$R_{vir}$','GalpropZstar':'$Z_{star}$',
        'HalopropMcooldot':'$\dot{M}_{cool}$','HalopropC_nfw':'$c_{nfw}$',
        'GalpropHalfmassRadius':'$R_{50}$', 'GalpropHalfmassRadius_hs':'$R_{50} / R_{vir}$'}
    if return_dictionary:
        return feature_dict
    elif isinstance(namelist,str):
        return feature_dict(namelist)
    else:
        return [feature_dict[f] for f in namelist]

def add_on_diff(values,features,min_dR2 = 0.015):
    add_on = np.diff(values)
    add_on  = np.insert(add_on,0,values[0])
#    for i in range(len(add_on)):
#        print(i,features[i],add_on[i])
    names = translate_names(features)
    mask = add_on > min_dR2
    return add_on[mask],np.array(names)[mask]

def fig5_and_6(subset = ''):
    '''make a ranked feature plot'''
    figname='fig5.pdf'
    mR2 = [0.014,0.05]
    if subset != '':
        subset='_'+subset
        figname='fig6.pdf'
        mR2 = [0.014,0.008]
    df_RF = pd.read_csv(f'rank_GalpropHalfmassRadius_RF{subset}_hp-ne175.mdNone.mss4.msl2.mf0.6.csv')
    df_SV = pd.read_csv(f'rank_GalpropHalfmassRadius_SV{subset}_hp-krbf.C50.g0.02.e0.1.csv')
    df_RF_hs = pd.read_csv(f'rank_GalpropHalfmassRadius_hs_RF{subset}_hs_hp-ne175.mdNone.mss4.msl2.mf0.6.csv')
    df_SV_hs = pd.read_csv(f'rank_GalpropHalfmassRadius_hs_SV{subset}_hs_hp-krbf.C50.g0.02.e0.1.csv')
    
    add_on_diff_RF, names_RF = add_on_diff(df_RF['add on'], 
        df_RF['feature'].to_list(), min_dR2=mR2[0])
    add_on_diff_SV, names_SV = add_on_diff(df_SV['add on'],
        df_SV['feature'].to_list(), min_dR2=mR2[0])
    add_on_diff_RF_hs, names_RF_hs = add_on_diff(df_RF_hs['add on'],
        df_RF_hs['feature'].to_list(), min_dR2=mR2[1])
    add_on_diff_SV_hs, names_SV_hs = add_on_diff(df_SV_hs['add on'],
        df_SV_hs['feature'].to_list(), min_dR2=mR2[1])

    f,axes = plt.subplots(1,2,sharey=True,figsize=(14,6))
    axes[0].bar(names_RF,add_on_diff_RF,width=0.4,align='edge',label='RF')
    axes[0].bar(names_SV,add_on_diff_SV,width=-0.4,align='edge',label='SV')
    axes[1].bar(names_RF_hs,add_on_diff_RF_hs,width=0.4,align='edge',label='RF')
    axes[1].bar(names_SV_hs,add_on_diff_SV_hs,width=-0.4,align='edge',label='SV')
    axes[0].set_ylabel('$\Delta R^2$', fontsize='x-large')
#    axes[0].set_xlabel('features unscaled', fontsize='x-large')
#    axes[1].set_xlabel('features halo scaled', fontsize='x-large')
    axes[0].legend()
    axes[1].legend()
    plt.setp(axes[0].get_xticklabels(), rotation=45, fontsize='large', ha="right",
        rotation_mode="anchor")
    plt.setp(axes[1].get_xticklabels(), rotation=45, fontsize='large', ha="right",
        rotation_mode="anchor")
    plt.subplots_adjust(wspace=0.0,bottom=0.15)

#    plt.show()
    plt.savefig(figname)

def fig5():
    names = get_fnames()
    df_rf = pd.read_csv(names['RFbt1'])
    df_rf_hs = pd.read_csv(names['RFbt1_hs'])
    df_sv = pd.read_csv(names['SVbt1'])
    df_sv_hs = pd.read_csv(names['SVbt1_hs'])
    fig,axes = plt.subplots(nrows=2,ncols=2,sharey=True,sharex=True)
    xr=[0,6]
    pred_true_plot(df_rf,xrange=xr, axis = axes[0,0], zeroD=False, raw=True )
    pred_true_plot(df_rf_hs, xrange=xr,axis = axes[1,0], zeroD=True, raw=True )
    pred_true_plot(df_sv, xrange=xr, axis = axes[0,1], zeroD=False, raw=True )
    pred_true_plot(df_sv_hs,xrange=xr, axis = axes[1,1], zeroD=True, raw=True )
    plt.subplots_adjust(hspace=0.0,wspace=0.0)
    plt.show()

def figcorr(df, method='spearman', halo_scale=False):
        if halo_scale:
            df = ml.halo_scale(df,type = 'scsam', dmo = False)
        corr = df.corr(method=method)
        c = (corr.sort_values(by='GalpropMvir',axis=1)).sort_values(by='GalpropMvir',axis=0)
        f,ax = plt.subplots(figsize=(15,12))
        fdict = translate_names(5,return_dictionary=True)
        c.rename(mapper=fdict,axis=0,inplace=True)
        c.rename(mapper=fdict,axis=1,inplace=True)
        sns.heatmap(c,ax=ax,cmap='GnBu',annot=True,vmax=1.0,vmin=-1.0,annot_kws={'fontsize':'xx-small'})
        plt.title( method +' correlation matrix')
#        plt.show()
#        plt.savefig(method[0]+'corr.pdf')

def unparam(name):
    if name[0:6] == 'params':
        return name[7:]
    else:
        return name
    
def figoptuna(kind='RF'):
    ''' make figures with the saved optuna study'''
    #firt reload the study
    if kind=='RF':
        study_name = "RF-study"
    else:
        study_name = "SV-study"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    ##output a dataframe of the study
    df = study.trials_dataframe(attrs=['value','duration','params'])
    time = [t.total_seconds() for t in df.duration]
    df['time'] = time
    df.rename(columns=unparam, inplace = True)
    df.drop('duration',axis=1,inplace=True)
    print(df.sort_values(by='value'))

def fig_learning_curve():
#    df1 = pd.read_csv('learning_curve_GalpropHalfmassRadius_hs_RF_hs_hp-ne175.mdNone.mss4.msl2.mf0.6.csv')
#    df2 = pd.read_csv('learning_curve_GalpropHalfmassRadius_hs_SV_N40000_hs_hp-krbf.C50.g0.02.e0.1.csv')
    df1 = pd.read_csv('learning_curve_GalpropHalfmassRadius_RF_hp-ne175.mdNone.mss4.msl2.mf0.6.csv')
    df2 = pd.read_csv('learning_curve_GalpropHalfmassRadius_SV_N40000_hp-krbf.C50.g0.02.e0.1.csv')
    
    fig, ax = plt.subplots()
    ax.fill_between(df1['train sizes'],df1['train score']+df1['train std'], 
        df1['train score']-df1['train std'])
    ax.fill_between(df1['train sizes'],df1['test score']+df1['test std'],
        df1['test score']-df1['test std'])
    ax.fill_between(df2['train sizes'],df2['train score']+df2['train std'], 
        df2['train score']-df2['train std'])
    ax.fill_between(df2['train sizes'],df2['test score']+df2['test std'],
        df2['test score']-df2['test std'])
    
    ax.plot(df1['train sizes'], df1['train score'] , label = 'RF train score')
    ax.plot(df1['train sizes'], df1['test score'],label = 'RF test score')   
    ax.plot(df2['train sizes'], df2['train score'] , label = 'SV train score')
    ax.plot(df2['train sizes'], df2['test score'],label = 'SV test score')    
    
    ax2 = ax.twinx()
    ax2.plot(df1['train sizes'],df1['fit time'],label = 'RF fit time', linestyle='dotted')
    ax2.plot(df2['train sizes'],df2['fit time'],label = 'SV fit time', linestyle='dotted')  
    ax2.set_ylim([0,700])
    ax2.set_ylabel('Time [s]')
    
    ax.set_xlabel('Number of Instances',fontsize='x-large')
    ax.set_ylabel('$R^2$ score',fontsize='x-large')
    ax.set_xlim([0,200000])
    ax.legend(loc = 'lower right')
    ax2.legend(loc = 'lower center')
    plt.savefig('learning_curve.pdf')

def fig_SR(df):
    df['y_true'] = np.log10(df['GalpropHalfmassRadius']/df['GalpropRhalo'])
    df['y_pred'] = np.log10(0.3*df['HalopropSpin'])
    df['Rhalo'] = df['GalpropRhalo']
    fig,axes=plt.subplots(ncols=2,sharey=True,figsize=(6,3))
    pred_true_plot(df,xrange=[0,6],halo_scale=True,axis=axes[0])
    pred_true_plot(df,xrange=[0,6],halo_scale=True,axis=axes[1])
    axes[1].set_xlim([0,6])
    plt.show()

def main(args):
    dropbox_directory = '/Users/ari/Dropbox (City Tech)/data/'

    if args.opt:
        figoptuna(kind=args.opt)

    if args.sample:
        local = os.path.isdir(dropbox_directory)
        if not local:
            dropbox_directory = input("Path to Dropbox directory: ")
        df_orig = pd.read_hdf(dropbox_directory+'tng300-sam.h5')
        df = sam_paper_sample(df_orig)
        df1 = df[df['GalpropMbulge']/df['GalpropMstar'] < 0.1]
        df1.to_hdf('tng300_bt1.h5', key='s', mode='w')

    if args.fig == 0: #in development choice 
        df = pd.read_hdf('tng300_bt1.h5')
        fig_SR(df)
#        df = pd.read_hdf('tng300_sam_paper.h5')
#        figcorr(df, halo_scale=False, method='pearson')
#        plt.savefig('corr_p.pdf')
#        figcorr(df, halo_scale=False, method='spearman')
#        plt.savefig('corr_s.pdf')
#        figcorr(df, halo_scale=True, method='pearson')
#        plt.savefig('corr_p_hs.pdf')
#        figcorr(df, method='spearman') #already halo scaled
#        plt.savefig('corr_s_hs.pdf')

    if args.fig == 1: #make Figure 1
        df = pd.read_hdf('tng300_sam_paper.h5')
#        df_sim = pd.read_hdf(dropbox_directory+'tng300-sim.h5')
#        fig1(df,df_sim)
        fig1_alt(df)
  
    if args.fig == 2:
        fig_learning_curve()

    if args.fig == 3:
        fig3(DMO=True)

    if args.fig == 4:
        fig3()

    if args.fig == 5:
        fig5_and_6()

    if args.fig == 6:
        fig5_and_6(subset='bt1')
    
    if args.fname != 'none':
        df = pd.read_csv(args.fname)
        fig,axis=plt.subplots()
        pred_true_plot(df, axis = axis)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 
        'Makes the data samples and figures for the paper ')
    parser.add_argument('-s', '--sample', default=False, action='store_true',
                        help='just creates the data sample')
    parser.add_argument('-f', '--fig',type = int, default=-1,
                        help = 'number of figure to make')
    parser.add_argument('fname',nargs='?', default='none',
                        help='file to show a true vrs predict plot for')
    parser.add_argument('-o','--opt')
    args = parser.parse_args()
    print(args)
    main(args)


    


