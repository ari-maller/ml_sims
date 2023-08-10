import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.model_selection as skms
from sklearn.ensemble import RandomForestRegressor
import optuna 
import optuna.visualization as ov
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


#scores
def R2(y_pred,y_true):
    y_mean = np.mean(y_true)
    sum_res = np.sum((y_pred - y_true)**2)
    sum_sqr = np.sum((y_true - y_mean)**2)
    return 1.0 - (sum_res/sum_sqr)

def error(y_pred,y_true):
    return y_pred - y_true

def mean_absolute_error(y_pred,y_true):
    return np.mean(np.abs(error(y_pred,y_true)))

def mean_absolute_fractional_error(y_pred,y_true):
    frac_error = (y_pred - y_true) / y_true
    return np.mean(np.abs(frac_error))

def root_mean_sqr_fraction_error(y_pred,y_true):
    frac_error = (y_pred - y_true) / y_true
    return np.sqrt(np.sum(frac_error**2)/len(frac_error))

def predict_true_plot(ypred,ytrue,xrange = None):
    ''''''
    fig, ax = plt.subplots()
    sns.kdeplot(x=ytrue, y = ypred, alpha=0.3,
            levels = [0.1,0.3,0.5,0.7,1.0],ax=ax, linestyles='dashed')
    if xrange==None:
        xrange=[np.min(ytrue),np.max(ytrue)]
    ax.plot(xrange,xrange,linestyle='dotted')
    ax.scatter(ytrue,ypred,s=1,marker='.',alpha=0.3,color='green')
    ax.set_xlim(xrange)
    ax.set_ylim(xrange)
    ax.annotate(f'Test $R^2$ = {0.94}',(0.1,0.9), xycoords='axes fraction')
    ax.axis('equal')
    plt.savefig('predict_true_plot.pdf')
    

def check_range(df_orig):
    '''checks the ranges of the features in a dataframe and replaces 
    with log if they span too large a range. Also sets a lower limit of 
    the mass resolution to prevent log from giving -inf.'''
    df = df_orig.copy()
    for col in df.columns.values:
        high = df.loc[:,col].max()
        low = df.loc[:,col].min()
        if high - low > 30:
            df.loc[:,col] = np.log10(df.loc[:,col]+1.e-8) #rather set with value
    
    return df

def dmo(df, target, type = 'scsam'):
    '''remove all features that are not dark matter, but not the target'''
    features = list(df)
    features.remove(target)
    darkmatter_fields = ['HalopropC_nfw', 'HalopropMvir', 'HalopropSpin',
        'GalpropMvir', 'GalpropRhalo', 'GalpropVvir']
    for field in darkmatter_fields:
        features.remove(field)
    df.drop(features, axis=1, inplace=True)
    return df

def zeroD(df,fname='tng300_sam_paper_0d.h5',type = 'scsam'):
    '''makes a data set dimensionless'''
    def new_name(name):
        spot = name.find('prop')+4
        name[0:spot]+'Norm'+name[spot:]
        return name+'0D'
    
    scales = ['GalpropMvir','GalpropRhalo','GalpropVvir']
    mass_fields=['GalpropMBH', 'GalpropMbulge', 'GalpropMcold', 
                 'GalpropMstar', 'GalpropMstar_merge', 'GalpropMstrip',
                 'GalpropMdisk','HalopropMhot','HalopropMvir']

    size_fields = ['GalpropHalfmassRadius']
    vel_fields = ['GalpropVdisk','GalpropSigmaBulge']
    for field in mass_fields:
        df[new_name(field)] = df[field]/df[scales[0]]
        df.drop(field,axis=1,inplace=True)

    for field in size_fields:
        df[new_name(field)] = df[field]/df[scales[1]]
        df.drop(field,axis=1,inplace=True)    

    for field in vel_fields:
        df[new_name(field)] = df[field]/df[scales[2]]
        df.drop(field,axis=1,inplace=True) 
                         
    for field in scales:
        if field != 'GalpropMvir':
            df.drop(field,axis=1,inplace=True)

    if fname:
        df.to_hdf(fname, key='s', mode='w')
    return df

def run_RF_regressor(X_features, y_target, verbose=False, hyper_params=None):
    '''Run random forest regressor with given hyper parameters'''
    start_time = time.time()
    if hyper_params==None:
        hyper_params = {'n_estimators':100,'max_depth':None,
            'min_samples_split':2,'min_samples_leaf':1,'max_features':1.0}
    regr_RF = RandomForestRegressor(random_state=42,**hyper_params)

    cv = skms.KFold(n_splits=4, shuffle=True, random_state=10) #shuffle?
    scores = skms.cross_validate(regr_RF, X_features, y_target, cv = cv,
                                verbose=verbose, return_train_score=True)
    y_pred = skms.cross_val_predict(regr_RF, X_features, y_target, cv = cv, 
                                verbose=verbose)

    train_score = np.mean(scores['train_score'])
    train_std = np.std(scores['train_score'])
    test_score = np.mean(scores['test_score'])
    test_std = np.std(scores['test_score'])
    print('RF regressor finished. Time elapsed: {:.2f}'.format(time.time() - start_time))
    print('Training score: {:.2f} +/- {:.2f}.'.format(train_score, train_std))
    print('Test score: {:.2f} +/- {:.2f}.'.format(test_score, test_std))
    print('mean absolute fractional error {:.4f}'.format(mean_absolute_fractional_error(y_pred,y_target)))
    print('root square fractional error {:.4f}'.format(root_mean_sqr_fraction_error(y_pred,y_target)))

    return y_pred,scores

def grid_RF_regressor(X_features, y_target, verbose=False):
    '''Grid search optimize random forest regressor'''
    start_time = time.time()
    grid = {'bootstrap': [True],'max_depth': [5, 10, None],
            'min_samples_leaf': [1,2,4], 'n_estimators': [50,100,150]}

    # Grid search of parameters 
    # Note sklearn lacks a grid search without cross validation, so one would 
    # have to use your own loop, it can make a grid with ParameterGrid()
    regr_RF = RandomForestRegressor(random_state=42)
    rfr_grid = skms.GridSearchCV(estimator = regr_RF, param_grid = grid, 
                    cv = skms.KFold(n_splits=5, shuffle=True),
                    verbose = 1, return_train_score=True)
    rfr_grid.fit(X_features, y_target)
    print('Grid search RF regressor finished. Time elapsed: {:.2f}'.format(time.time() - start_time))
    print('Best params, best score:', "{:.4f}".format(rfr_grid.best_score_),
          rfr_grid.best_params_)
    return rfr_grid.best_params_

def optimize_RF_regressor(X_features,y_target,n_trials=50):
    '''use optuna to optimize random forest regressor'''
    start_time = time.time()
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=25)
#        criterion = trial.suggest_categorical("criterion",["squared_error", "poisson"])
        max_depth = trial.suggest_int("max_depth", 5, 15)
        min_samples_split = trial.suggest_int("min_samples_split",2,16,step=2)
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,4)
        max_features = trial.suggest_float('max_features',0.3,1.0)

        regr_RF = RandomForestRegressor(random_state=42,n_estimators=n_estimators,
            max_depth=max_depth,min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, max_features=max_features)
        cv = skms.KFold(n_splits=4, shuffle=True, random_state=10)
        scores = skms.cross_val_score(regr_RF, X_features, y_target, cv = cv)
        return np.mean(scores)
    
    study = optuna.create_study(direction="maximize") #highest R2 score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar = True)
    print('Optimizing RF regressor finished. Time elapsed: {:.2f}'.format(time.time() - start_time))
    print(f'Best trail value: {study.best_trial.values}')
    print(f'Best params: {study.best_trial.params}')
    return(study)

def learning_curve_RF_regressor(X_features,y_target,
                      hyper_params={},plot = True):
    '''return (and plot) a learning curve for the passed features and target'''
    start_time = time.time()
    RF_regress = RandomForestRegressor(random_state=42,**hyper_params)
    train_sizes, train_scores, test_scores, fit_times, score_times = \
        skms.learning_curve(RF_regress, X_features, y_target,return_times = True)

    if plot:
        train = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        fit_time = fit_times.mean(axis=1)
        fig, ax = plt.subplots()
        ax.fill_between(train_sizes,train+train_std,train-train_std)
        ax.fill_between(train_sizes,test+test_std,test-test_std)
        ax.plot(train_sizes,train,label = 'train score')
        ax.plot(train_sizes,test,label = 'test score')   
        ax2 = ax.twinx()
        ax2.plot(train_sizes,fit_time,label = 'fit time',linestyle='dotted')
        ax2.set_ylabel('Time')
        ax.set_xlabel('Number of Galaxies')
        ax.set_ylabel('$R^2$ score')
        plt.legend()
        plt.savefig('learning_curve.pdf')

    print('Learning curve for RF regressor finished. Time elapsed: {:.2f}'.format(time.time() - start_time))
    return train_sizes,train_scores,test_scores


def importance_RF_regressor(X_features,y_target, n_trials=30, 
                            hyper_params={}, plot = True):
    '''Importance ranking with RF regressor'''
    RF_regress = RandomForestRegressor(random_state=42,**hyper_params)
    X_train, X_val, y_train, y_val = skms.train_test_split(X_features, y_target,
         random_state=42)
    model = RF_regress.fit(X_train,y_train)
    print('\n Feature Importance')
    print('Using permutation ranking')
    print(f'All features: {model.score(X_val,y_val)}')
    r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=42)

    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i]  > 0.02:
            print(f"{X_val.columns.values[i]:<8}  {r.importances_mean[i]:.3f}" 
              f" +/- {r.importances_std[i]:.3f}")
            
    print('\n Using impurity decrement')
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    for i in importances.argsort()[::-1]:
        if importances[i] > 0.02:
            print(f"{X_val.columns.values[i]:<8}  {importances[i]:.3f}" 
              f" +/- {std[i]:.3f}")
            
    if plot:
        perm_importances = pd.DataFrame({'permutation':r.importances_mean,
            'mean decrement':importances}, index = X_val.columns.values)
        perm_importances.sort_values(by = ['permutation'], ascending=False,
                                     inplace=True)
        mask = np.logical_or(perm_importances['permutation'] > 0.02, perm_importances['mean decrement'] > 0.02)
        ax = perm_importances[mask].plot.barh(figsize=(8,4))
        plt.subplots_adjust(left=0.25)
        plt.savefig('import.pdf')

def make_filename(args,hyper_params):
    '''filename for saving the predicted and target values'''
    fname = f'fit_{args.target}'
    if args.N!=0:
        fname=fname+f'_N{args.N}'
    if dmo:
        fname+'_dmo'
    if zeroD:
        fname+'_zeroD'
    hp = '_hp-ne{}.md{}.mss{}.msl{}.mf{}'.format(hyper_params['n_estimators'],
        hyper_params['max_depth'], hyper_params['min_samples_split'],
        hyper_params['min_samples_leaf'], hyper_params['max_features'])
    return fname+hp+'.csv'


def main(args):
    df = pd.read_hdf(args.df)
    if args.dmo:
        df = dmo(df,args.target)

    if args.zeroD:
        df = zeroD(df)
        args.target = args.target+'0D'
    else:
        df = check_range(df)

    Nmax = df.shape[0]
    if args.N !=0 and args.N < Nmax:
        df_s = df.sample(n=args.N)
    else:
        df_s = df
    X_features = df_s.drop(columns=[args.target])
    y_target = df_s.loc[:,args.target]

    hyper_params = {'n_estimators':75,'max_depth':None,
        'min_samples_split':8,'min_samples_leaf':4,'max_features':0.9}   
    if args.opt:
        study = optimize_RF_regressor(X_features,y_target,n_trials=50)
        hyper_params = study.best_trial.params
    
    y_pred,scores = run_RF_regressor(X_features,y_target,
            hyper_params = hyper_params)
    results = pd.DataFrame({'y_true':y_target, 'y_pred':y_pred})
    results.to_csv(make_filename(args,hyper_params))

#    if zeroD:
#        r2s = R2(y_pred,y_target)
#        r2 = R2(y_pred*Mvir,y_target*Mvir)
#    else:
#        r2 = r2(y_pred,y_target)
#        r2s = r2(y_pred/X_features.GalpropMvir,y_target/X_features.GalpropMvir)

#    print('R2 unscaled by Mvir: {:3f}'.format(r2))
#    print('R2 scaled by Mvir: {:3f}'.format(r2s))
    if args.lc:
        learning_curve_RF_regressor(X_features,y_target,
                hyper_params = hyper_params, plot = True)
        
    if args.rank:
        importance_RF_regressor(X_features, y_target, n_trials=30, 
                hyper_params = hyper_params, plot = True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run RF regressor on provided dataframe')
    parser.add_argument('df',type=str,
        help = 'Dataframe to run regression on')
    parser.add_argument('target',type=str,
        help = 'The target to predict')
    parser.add_argument('--N',default = 1000, type = int,
        help='Number of instances to use, if 0 will use all')
    parser.add_argument('-z','--zeroD', default=False, action='store_true',
        help='Make the dataset dimensionless by dividing by halo properties')
    parser.add_argument('-d','--dmo', default=False, action='store_true',
        help="Only use dark matter features in modeling, don't use with zeroD")
    parser.add_argument('-o','--opt', default=False, action='store_true',
        help='Run an optimization of the hyper parameters.')
    parser.add_argument('-l','--lc', default=False, action='store_true',
        help='Create a lerning curve for the model.') 
    parser.add_argument('-r','--rank', default=False, action='store_true',
        help='Rank the importance of the features.')   
    args = parser.parse_args()
    main(args)
        