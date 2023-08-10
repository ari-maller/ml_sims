import os
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pandas as pd

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
    df['GalpropLogMstar'] = np.log10(df['GalpropMstar'])
    df['GalpropLogRstar'] = np.log10(df['GalpropHalfmassRadius'])
    df.drop(['GalpropRdisk','GalpropRbulge','GalpropMH2', 'GalpropMHI', 
             'GalpropMHII'],axis=1,inplace=True) #only keep Mcold
    df.drop(['GalpropX', 'GalpropVx', 'GalpropY', 'GalpropVy', 'GalpropZ',
             'GalpropVz'],axis=1,inplace=True)
    df.drop(['HalopropMdot_eject','HalopropMdot_eject_metal',
             'HalopropMaccdot_metal'],axis=1,inplace=True) #all zero
#    df.drop(['HalopropMaccdot_reaccreate_metal']) #mostly zero
    fdisk = (df['GalpropMstar']+df['GalpropMcold'])/df['GalpropMvir']
    mask =  (df['GalpropMbulge']/df['GalpropMstar'] > 0.4) | (fdisk > min_fdisk)
    mask = (df['GalpropMstar'] > mass_cut) & mask
    mask = (df['GalpropSatType']==False) & mask
    df.drop(['GalpropSatType','GalpropRfric','GalpropTsat'],axis=1,inplace=True)
    df.drop(['GalpropLogMstar','GalpropLogRstar'],axis=1,inplace=True)
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
    plt.xlabel('$ Log_{10} $ Mstar $[M_\odot]$')
    plt.ylabel('$ Log_{10}  R_{50}$ [kpc] ')
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

def pred_true_plot(df):
    
def main(args):
    dropbox_directory = '/Users/ari/Dropbox (City Tech)/data/'
    local = os.path.isdir(dropbox_directory)
    if not local:
        dropbox_directory = input("Path to Dropbox directory: ")

    if args.sample or not os.path.isfile('tng300_sam_paper.h5'):
        df_sam = pd.read_hdf(dropbox_directory+'tng300-sam.h5')
        df = sam_paper_sample(df_sam)
    else:
        df = pd.read_hdf('tng300_sam_paper.h5')

#    morphology_bins(df)
#    plt.hist(df['morph_group'])
#    plt.show()

    if args.fig==1: #make Figure 1
        df_sim = pd.read_hdf(dropbox_directory+'tng300-sim.h5')
        fig1(df,df_sim)
  
    if args.fig > 1:
        target = 'GalproNormHalfRadius'
        X_all = df.drop(columns=[target])
        y_all = df.loc[:,target]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 
        'Makes the data samples and figures for the paper ')
    parser.add_argument('-s', '--sample', help='just creates the data samples')
    parser.add_argument('-z','--zeroD', help='create dimensionless sample')
    parser.add_argument('-f', '--fig',type = int, help = 'number of figure to make')
    args = parser.parse_args()
    print(args)
    main(args)


    


