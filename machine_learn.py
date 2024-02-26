import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.model_selection as skms
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from tqdm import tqdm
import optuna 
import optuna.visualization as ov



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

def mean_absolute_percent_error(y_pred,y_true):
    frac_error = (y_pred - y_true) / y_true
    return np.mean(np.abs(frac_error))

def median_absolute_percent_error(y_pred,y_true):
    frac_error = (y_pred - y_true) / y_true
    return np.median(np.abs(frac_error))

def root_mean_sqr_percent_error(y_pred,y_true):
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
    
def check_range(df_orig,verbose=False):
    '''checks the ranges of the features in a dataframe and replaces 
    with log if they span too large a range. Also sets a lower limit of 
    the mass resolution to prevent log from giving -inf.'''
    df = df_orig.copy()
    for col in df.columns.values:
        high = df.loc[:,col].max()
        low = df.loc[:,col].min()
        df.loc[:,col] = np.log10(df.loc[:,col]+1.e-8) #rather set with value
        if verbose:
            print(f'taking log10 of {col}')
    return df

def log_with_zero(df_orig):
    df = df_orig.copy()
    for col in df.columns.values:
        low = np.min(df[col])
        if low == 0.0:
            zero = df[col] == 0.0
            low = np.min(df[col][np.invert(zero)])
            df.loc[zero,col] = low
        df.loc[:,col] = np.log10(df.loc[:,col])
    return df

def dmo(df, target, type = 'scsam'):
    '''remove all features that are not dark matter, but not the target'''
    features = list(df)
    features.remove(target)
    darkmatter_fields = ['HalopropC_nfw', 'HalopropSpin',
        'GalpropMvir', 'GalpropRhalo', 'GalpropVvir', 'GalpropVdisk']
    for field in darkmatter_fields:
        features.remove(field)
    df.drop(features, axis=1, inplace=True)
    return df

def halo_scale(df,type = 'scsam', dmo = False):
    '''makes a data set scaled by the halo properites'''
    def new_name(field):
        return field+'_hs'
    scales = ['GalpropMvir','GalpropRhalo','GalpropVvir']
    mass_fields=['GalpropMBH', 'GalpropMbulge', 'GalpropMcold', 
        'GalpropMstar', 'GalpropMstar_merge','HalopropMhot']
    size_fields = ['GalpropHalfmassRadius']
    vel_fields = ['GalpropVdisk','GalpropSigmaBulge']
    if dmo:
        mass_fields=[]
        vel_fields = ['GalpropVdisk']
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

    return df

def run_regressor(X_features, y_target, verbose=False, hyper_params=None, kind='RF'):
    '''Run RF or SV regressor with given hyper parameters'''
    start_time = time.time()
    if kind=='RF':
        if hyper_params==None:
            hyper_params = {'n_estimators':100,'max_depth':None,
                'min_samples_split':2,'min_samples_leaf':1,'max_features':1.0}
        regr = RandomForestRegressor(random_state=42,**hyper_params)
    elif kind=='SV': #support vectors need to be scaled 
        if hyper_params==None:
            hyper_params = {'kernel':'rbf','C':100, 'gamma':0.1,'epsilon':0.1}
        pipeline = Pipeline(steps=[('normalize', StandardScaler()), ('model', SVR(**hyper_params))])
        regr = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())       

    cv = skms.KFold(n_splits=5, shuffle=True, random_state=10) #shuffle?
    scores = skms.cross_validate(regr, X_features, y_target, cv = cv, n_jobs=5,
                                verbose=verbose, return_train_score=True)
    y_pred = skms.cross_val_predict(regr, X_features, y_target, cv = cv, n_jobs=5)

    train_score = np.mean(scores['train_score'])
    train_std = np.std(scores['train_score'])
    test_score = np.mean(scores['test_score'])
    test_std = np.std(scores['test_score'])
    print('{} regressor finished. Time elapsed: {:.2f} s'.format(kind,time.time() - start_time))
    print('Training score: {:.2f} +/- {:.2f}.'.format(train_score, train_std))
    print('Test score: {:.2f} +/- {:.2f}.'.format(test_score, test_std))
    return y_pred,scores

def grid_regressor(X_features, y_target, verbose=False, kind='RF'):
    '''Grid search optimize random forest or support vector regressor'''
    start_time = time.time()
    if kind == 'RF':
        grid = {'bootstrap': [True],'max_depth': [5, 10, None],
            'min_samples_leaf': [1,2,4], 'n_estimators': [50,100,150]}
        regr = RandomForestRegressor(random_state=42)
    else:
        grid = {'kernel': ['rbf'],'C':[1,10,100], 'gamma':['scale','auto',0.1,0.2],
                'epsilon':[0.05,0.1,0.2]}
        regr = SVR()
    
    # Grid search of parameters 
    # Note sklearn lacks a grid search without cross validation, so one would 
    # have to use your own loop, it can make a grid with ParameterGrid()
    rfr_grid = skms.GridSearchCV(estimator = regr, param_grid = grid, 
                    cv = skms.KFold(n_splits=5, shuffle=True),
                    verbose = 1, return_train_score=True)
    rfr_grid.fit(X_features, y_target)
    print('Grid search {} regressor finished. Time elapsed: {:.2f}'.format(kind,time.time() - start_time))
    print('Best params, best score:', "{:.4f}".format(rfr_grid.best_score_),
          rfr_grid.best_params_)
    return rfr_grid.best_params_

def optimize_RF_regressor2(X_features,y_target,n_trials=50):
    #uses skopt but needs numpy < 1.2. If they update may work
    start_time = time.time()
    X_train, X_test, y_train, y_test = skms.train_test_split(X_features, y_target, 
            train_size=0.75, test_size=.25, random_state=42)
    hyper_params = {
        "n_estimators": (50, 200),
        "max_depth": (5,15),             
        "min_samples_split": (2,16),     
        "min_samples_leaf": (1,4),  
        "max_features": (0.3,1.0)          
        }
    #BayesSearchCV not working, issue with numpy
    opt = BayesSearchCV(RandomForestRegressor(), hyper_params, 
        n_iter=n_trials,cv=3)

    opt.fit(X_train,y_train)
    print('Optimizing RF regressor finished. Time elapsed: {:.2f}'.format(time.time() - start_time))
    print(f'train score: {opt.best_score}')
    print(f'test score: {opt.score(X_test,y_test)}')
    return(opt)

def optimize_regressor(X_features,y_target,n_trials=30, kind='RF'):
    '''use optuna to optimize random forest regressor'''
    start_time = time.time()
    def objective_RF(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 200, step=25)
#        criterion = trial.suggest_categorical("criterion",["squared_error", "poisson"])
#        max_depth = trial.suggest_int("max_depth", 5, 15)
        min_samples_split = trial.suggest_int("min_samples_split",2,8,step=2)
        min_samples_leaf = trial.suggest_int("min_samples_leaf",1,4)
        max_features = trial.suggest_float('max_features',0.3,1.0)

        regr_RF = RandomForestRegressor(random_state=42,n_estimators=n_estimators,
            max_depth=None, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, max_features=max_features)
        cv = skms.KFold(n_splits=3, shuffle=True, random_state=10)
        scores = skms.cross_val_score(regr_RF, X_features, y_target, cv = cv)
        return np.mean(scores)
        
    def objective_SV(trial):
        kernel = trial.suggest_categorical("kernel", ['rbf'])
        C = trial.suggest_int("C", 10, 150)
        gamma = trial.suggest_float("gamma",0.02,0.2)
        epsilon = trial.suggest_float("epsilon",0.02,0.2)
        pipeline = Pipeline(steps=[('normalize', StandardScaler()),
            ('model', SVR(kernel='rbf',C=C, gamma=gamma, epsilon=epsilon))])
        regr_SV = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler()) 
        cv = skms.KFold(n_splits=3, shuffle=True, random_state=10)
        scores = skms.cross_val_score(regr_SV, X_features, y_target, cv = cv)
        return np.mean(scores)
    
    if kind=='RF':
        objective = objective_RF
        study_name = "RF-study"
    else:
        objective = objective_SV
        study_name = "SV-study"
    
  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, 
            storage=storage_name,direction="maximize") #highest R2 score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar = True)
    print('Optimizing {} regressor finished. Time elapsed: {:.2f}'.format(kind, time.time() - start_time))
    print(f'Best trail value: {study.best_trial.values}')
    print(f'Best params: {study.best_trial.params}')
    return(study)

def learning_curve_regressor(X_features,y_target, hyper_params={}, kind='RF', 
        plot = True):
    '''return (and plot) a learning curve for the passed features and target'''
    start_time = time.time()
    if kind=='RF':
        regress = RandomForestRegressor(**hyper_params)
        number = [1000,5000,10000,20000,50000,100000,190000]
    else:
        pipeline = Pipeline(steps=[('normalize', StandardScaler()), ('model', SVR(**hyper_params))])
        regress = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler()) 
#        regress = SVR(**hyper_params)
        number=[1000,5000,10000,15000,20000,24000,28000,30000]
    
    train_sizes, train_scores, test_scores, fit_times, score_times = \
        skms.learning_curve(regress, X_features, y_target,
        return_times = True, train_sizes=number,n_jobs=5)

    train = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    fit_time = fit_times.mean(axis=1)
    df = pd.DataFrame(data={'train sizes':train_sizes,
        'train score':train, 'train std':train_std,
        'test score':test, 'test std':test_std,
        'fit time':fit_time})
    if plot:
        fig, ax = plt.subplots()
        ax.fill_between(train_sizes,train+train_std,train-train_std)
        ax.fill_between(train_sizes,test+test_std,test-test_std)
        ax.plot(train_sizes,train,label = 'train score')
        ax.plot(train_sizes,test,label = 'test score')   
        ax2 = ax.twinx()
        ax2.plot(train_sizes,fit_time,label = 'fit time',linestyle='dotted')
        ax2.set_ylabel('Time [s]')
        ax.set_xlabel('Number of Galaxies')
        ax.set_ylabel('$R^2$ score')
        plt.legend()
        plt.savefig(f'learning_curve_{kind}.pdf')

    print('Learning curve for {} regressor finished. Time elapsed: {:.2f} s.'.format(kind, time.time() - start_time))
    return df


def importance_RF_regressor(X_features,y_target, n_trials=30, 
                            hyper_params={}, plot = True):
    '''Importance ranking with RF regressor'''
    RF_regress = RandomForestRegressor(random_state=42,**hyper_params)
    X_train, X_val, y_train, y_val = skms.train_test_split(X_features, y_target)
    model = RF_regress.fit(X_train,y_train)
    feat_import = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    print('\n Feature Importance')
    print('Using permutation ranking')
    print(f'All features: {model.score(X_val,y_val)}')
    r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=42)
     
    #create a dataframe
    df_rank = pd.DataFrame({'feature':X_val.columns.values,
        'permutation':r.importances_mean,'premutation std':r.importances_std,
        'impurity':feat_import,'impurity std':std})           
    return df_rank

def addon_feature_importance(estimator,X_features,y_target):
    '''cycle through the features to find most important features'''
    start_time = time.time()    
    remaining_features = X_features.columns.tolist()
    rank_features = []
    max_score = skms.cross_val_score(estimator, X_features, y_target, cv=5, n_jobs=5)
    best_score = 0.0
    scores=[]
    stds=[]
    for feature in remaining_features: #loop over first feature 
        x = X_features.loc[:, feature] 
        x = (x.to_numpy()).reshape(-1,1)
        score = skms.cross_val_score(estimator, x, y_target, cv = 3)
        scores.append(score.mean())
        stds.append(score.std())

    n=1
    print(f'Add on Feature Importance \n {n} feature {time.time()-start_time} s')
    i = np.argmax(scores) #find the index for the max score
    best_feature = remaining_features.pop(i) #remove best feature
    score_keeper = [scores[i]]
    std_keeper = [stds[i]]
    rank_features =[best_feature]
    while best_score < max_score.mean() - max_score.std():
        scores=[]
        stds=[]
        for feature in remaining_features:
            use_features = rank_features +[feature]
            x = X_features.loc[:,use_features]
            score = skms.cross_val_score(estimator, x, y_target)
            scores.append(score.mean())
            stds.append(score.std())
            
        n=n+1
        print(f'{n} features {time.time()-start_time} s')
        i = np.argmax(scores) #find the index for the max score
        best_feature = remaining_features.pop(i)
        best_score = scores[i]
        rank_features.append(best_feature)
        score_keeper.append(best_score)
        std_keeper.append(stds[i])

    rank_df = pd.DataFrame({'feature':rank_features,'add on':score_keeper,
                            'add on std':std_keeper})
    return rank_df



def make_filename(args, hyper_params, other=False):
    '''filename for saving the predicted and target values'''
    fname = f'fit_{args.target}_{args.algorithm}'
    if other:
        fname=f'{args.target}_{args.algorithm}'
    if args.subset:
        fname = fname + f'_{args.subset}'
    if args.N!=0:
        fname=fname+f'_N{args.N}'
    if args.dmo:
        fname = fname+'_dmo'
    if args.halo_scale:
        fname = fname+'_hs'
    if args.algorithm=='RF':
        hp = '_hp-ne{}.md{}.mss{}.msl{}.mf{}'.format(hyper_params['n_estimators'],
            hyper_params['max_depth'], hyper_params['min_samples_split'],
            hyper_params['min_samples_leaf'], hyper_params['max_features'])
    else:
        hp = '_hp-k{}.C{}.g{}.e{}'.format(hyper_params['kernel'], 
            hyper_params['C'],hyper_params['gamma'],hyper_params['epsilon'])
    return fname+hp+'.csv'


def main(args):
    #set up sample
    if args.df!='tng300_sam_paper.h5' and args.subset==None:
        print('error: use -s if a subset')
        exit(1)
    df = pd.read_hdf(args.df)
    #save Rhalo if it gets changed, should be more flexible
    Rhalo = df['GalpropRhalo']
    if args.dmo:
        df = dmo(df,args.target)

    if args.halo_scale:
        df = halo_scale(df, dmo = args.dmo)
        args.target = args.target+'_hs'
    else: #need to make general
        df.drop(['GalpropRhalo','GalpropVvir'], axis=1, inplace=True)
    
    df = log_with_zero(df)
    Nmax = df.shape[0]
    if args.N !=0 and args.N < Nmax:
            df_s = df.sample(n=args.N)
    elif args.algorithm == 'SV':
        df_s = df.sample(n=25000)
    else:
        df_s = df

    X_features = df_s.drop(columns=[args.target])
    y_target = df_s.loc[:,args.target]
    Rhalo = Rhalo[X_features.index]
    size=X_features.shape
    print(args.algorithm+f': running with {size[1]} features and {size[0]} instances')

    #now perform machine learning
    if args.algorithm == 'RF':
        hyper_params = {'n_estimators':175,'max_depth':None,
            'min_samples_split':4,'min_samples_leaf':2,'max_features':0.6} 
    else:
        hyper_params = {'kernel':'rbf','C':50, 'gamma':0.02,'epsilon':0.1}
    if args.grid:
        hyper_params = grid_regressor(X_features, y_target, verbose=False, kind=args.kind)
        
    if args.opt:
        study = optimize_regressor(X_features, y_target, kind=args.algorithm)
        hyper_params = study.best_trial.params

    
#    print('Running regressor with these features:\n',
#            X_features.columns.values)
    if not (args.opt or args.grid or args.lc or args.rank):
        y_pred,scores = run_regressor(X_features,y_target,
            hyper_params = hyper_params, kind = args.algorithm)
        if halo_scale:
            print(f'MAPE: {mean_absolute_percent_error((10**y_pred)*Rhalo,(10**y_target)*Rhalo)}')
        else:
            print(f'MAPE: {mean_absolute_percent_error(10**y_pred,10**y_target)}')
        if args.N==0: #only save when N is set to 0
            results = pd.DataFrame({'y_true':y_target, 'y_pred':y_pred, 'Rhalo':Rhalo})
            results.to_csv(make_filename(args,hyper_params))

    if args.lc:
        df_lc = learning_curve_regressor(X_features,y_target,
            hyper_params = hyper_params, kind = args.algorithm, plot = True)
        df_lc.to_csv('learning_curve_'+make_filename(args, hyper_params, other=True))
        
    if args.rank:
        if args.algorithm == 'RF':
            regressor = RandomForestRegressor(random_state=42,**hyper_params)
        else:
            pipeline = Pipeline(steps=[('normalize', StandardScaler()), ('model', SVR(**hyper_params))])
            regressor = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler()) 
        rank_df = addon_feature_importance(regressor, X_features, y_target)
        rank_df.to_csv('rank_'+make_filename(args, hyper_params, other=True))
        print(rank_df)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run RF or SV regressor on provided dataframe.')
    parser.add_argument('df',type=str,
        help = 'Dataframe to run regression on')
    parser.add_argument('target',type=str,
        help = 'The target to predict')
    parser.add_argument('-N','--N',default = 1000, type = int,
        help='Number of instances to use, if 0 will use all')
    parser.add_argument('-a','--algorithm',choices=['RF','SV'], default = 'RF',
        help='Algorithm to use for regression. Choices are: RF - Random Forest or SV - support vectors')    
    parser.add_argument('-hs','--halo_scale', default=False, action='store_true',
        help='Make the dataset dimensionless by dividing by halo properties')
    parser.add_argument('-d','--dmo', default=False, action='store_true',
        help="Only use dark matter features in modeling")
    parser.add_argument('-g','--grid', default=False, action='store_true',
        help='Run a grid search to get optimal hyper parameters.')
    parser.add_argument('-o','--opt', default=False, action='store_true',
        help='Run an optimization of the hyper parameters.')
    parser.add_argument('-l','--lc', default=False, action='store_true',
        help='Create a learning curve for the model.') 
    parser.add_argument('-r','--rank', default=False, action='store_true',
        help='Rank the importance of the features.')   
    parser.add_argument('-s','--subset', type=str, action='store',
        help='This file is a subset of the data, add string to filenames.') 
    args = parser.parse_args()
    main(args)
        