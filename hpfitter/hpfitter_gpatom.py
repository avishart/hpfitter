
import numpy as np
from copy import deepcopy
from .hpfitter import HyperparameterFitter

class HyperparameterFitterGPAtom(HyperparameterFitter):

    def __init__(self,func,optimization_method=None,opt_kwargs={},add_noise_correction=True,**kwargs):
        """ A wrapper for hyperparameter fitter object, so it can be used with ase-GPatom. 
            The hyperparameter fitter object with local and global optimization methods for optimizing the hyperparameters on different objective functions. 
            Parameters:
                func : class
                    A class with the objective function used to optimize the hyperparameters.
                optimization_method : function
                    A function with the optimization method used.
                opt_kwargs : dict 
                    A dictionary with the arguments for the optimization method.
                add_noise_correction : bool
                    Add the noise correction to ratio.
        """
        super().__init__(func=func,optimization_method=optimization_method,opt_kwargs=opt_kwargs)
        self.add_noise_correction=add_noise_correction
        
    def fit(self,X,Y,model,hp=None,pdis=None,**kwargs):
        """ Optimize the hyperparameters 
            Parameters:
                X : (N,D) array
                    Training features with N data points and D dimensions.
                Y : (N,1) array or (N,D+1) array
                    Training targets with or without derivatives with N data points.
                model : Model
                    The Machine Learning Model with kernel and prior that are optimized.
                hp : dict
                    Use a set of hyperparameters to optimize from else the current set is used.
                pdis : dict
                    A dict of prior distributions for each hyperparameter type.
        """
        if hp is None:
            hp=model.hp.copy()
        self.get_noise_correction(X,model,**kwargs)
        theta,parameters=self.hp_to_theta(hp)
        model=self.copy_model(model)
        pdis_new=self.convert_pdis_to_gpatom(pdis)
        self.func.reset_solution()
        sol=self.optimization_method(self.func,theta,parameters,model,X,Y,pdis=pdis_new,**self.opt_kwargs)
        if 'hp' in sol.keys():
            self.get_noise_correction(X,model,**kwargs)
            sol['hp']=self.convert_hp_to_gpatom(sol['hp'],model)
        sol=self.get_full_hp(sol,model)
        return sol
    
    def hp_to_theta(self,hp):
        " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
        hp_new=self.convert_hp_from_gpatom(hp)
        return super().hp_to_theta(hp_new)
    
    def get_full_hp(self,sol,model,**kwargs):
        " Get the full hyperparameter dictionary with hyperparameters that are optimized and are not. "
        sol['full hp']=model.hp.copy()
        sol['full hp'].update(sol['hp'])
        if 'prefactor' in sol['full hp'].keys():
            sol['full hp'].pop('prefactor')
        sol['full hp']['noise']=sol['full hp']['weight']*sol['full hp']['ratio']
        return sol

    def copy_model(self,model):
        " Copy the model and check if the noisefactor is used in the factorization method"
        model=deepcopy(model)
        if 'noisefactor' in model.hp.keys():
            from .objectivefunctions.factorized_likelihood import FactorizedLogLikelihood
            if isinstance(self.func,FactorizedLogLikelihood):
                if model.hp['noisefactor']!=1.0:
                    print('Noisefactor must be 1.0 for the Factorization method')
                    model.hp['noisefactor']=1.0
        return model
    
    def convert_hp_from_gpatom(self,hp):
        " Convert the hyperparameters from GP-atom to the form here. "
        parameters=list(hp.keys())
        hp_new={}
        if 'scale' in parameters:
            hp_new['length']=np.array(np.log(hp['scale'])).reshape(-1)
        if 'weight' in parameters:
            hp_new['prefactor']=np.array(np.log(hp['weight'])).reshape(-1)
        if 'ratio' in parameters:
            ratio=hp['ratio']-self.corr
            if 'noisefactor' in parameters:
                hp_new['noise']=np.array(np.log(ratio*hp['noisefactor'])).reshape(-1)
            else:
                hp_new['noise']=np.array(np.log(ratio)).reshape(-1)
        return hp_new
    
    def convert_hp_to_gpatom(self,hp,model):
        " Convert the hyperparameters from here to the form of GP-atom. "
        parameters=list(hp.keys())
        hp_new={}
        if 'length' in parameters:
            hp_new['scale']=np.array(np.exp(hp['length'])).reshape(-1)
        if 'prefactor' in parameters:
            hp_new['weight']=np.exp(hp['prefactor'][0])
        if 'noise' in parameters:
            ratio=np.exp(hp['noise'][0])+self.corr
            if 'noisefactor' in model.hp.keys():
                hp_new['ratio']=ratio/model.hp['noisefactor']
            else:
                hp_new['ratio']=ratio
        return hp_new
    
    def convert_pdis_to_gpatom(self,pdis,**kwargs):
        " Convert the prior distributions with GPatom hyperparameter names to the form here."
        if pdis is None:
            return pdis
        pdis_new={}
        if 'scale' in pdis:
            pdis_new['length']=pdis['scale'].copy()
        if 'weight' in pdis:
            pdis_new['prefactor']=pdis['weight'].copy()
        if 'ratio' in pdis:
            pdis_new['noise']=pdis['ratio'].copy()
        return pdis_new
    
    def get_noise_correction(self,X,model,**kwargs):
        " Get the noise correction. "
        self.corr=0.0
        if self.add_noise_correction:
            KXX=model.kernel.kernel_matrix(X)
            self.corr=np.sqrt(self.func.get_correction(KXX))
        return self.corr
    
    def __repr__(self):
        return "HyperparameterFitterGPAtom(func={},optimization_method={},opt_kwargs={})".format(self.func.__class__.__name__,self.optimization_method.__name__,self.opt_kwargs)
    