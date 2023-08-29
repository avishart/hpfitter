
import numpy as np
from copy import deepcopy
from .hpfitter import HyperparameterFitter

class HyperparameterFitterGPAtom(HyperparameterFitter):
        
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
        theta,parameters=self.hp_to_theta(hp)
        model=self.copy_model(model)
        self.func.reset_solution()
        sol=self.optimization_method(self.func,theta,parameters,model,X,Y,pdis=pdis,**self.opt_kwargs)
        sol['hp']=self.convert_hp_to_gpatom(sol['hp'],model)
        return sol
    
    def hp_to_theta(self,hp):
        " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
        hp_new=self.convert_hp_from_gpatom(hp)
        return super().hp_to_theta(hp_new)
    
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
            if 'noisefactor' in parameters:
                hp_new['noise']=np.array(np.log(hp['ratio']*hp['noisefactor'])).reshape(-1)
            else:
                hp_new['noise']=np.array(np.log(hp['ratio'])).reshape(-1)
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
            if 'noisefactor' in model.hp.keys():
                hp_new['ratio']=np.exp(hp['noise'][0])/model.hp['noisefactor']
            else:
                hp_new['ratio']=np.exp(hp['noise'][0])
        return hp_new
    
    def __repr__(self):
        return "HyperparameterFitterGPAtom(func={},optimization_method={},opt_kwargs={})".format(self.func.__class__.__name__,self.optimization_method.__name__,self.opt_kwargs)
    