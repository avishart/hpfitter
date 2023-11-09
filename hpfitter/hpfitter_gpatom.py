
import numpy as np
from copy import deepcopy
from .hpfitter import HyperparameterFitter

class HyperparameterFitterGPAtom(HyperparameterFitter):
    def __init__(self,func,optimizer=None,bounds=None,use_update_pdis=False,get_prior_mean=False,use_stored_sols=False,**kwargs):
        """ 
        A wrapper for hyperparameter fitter object, so it can be used with ase-GPatom. 
        Hyperparameter fitter object with an optimizer for optimizing the hyperparameters on different given objective functions. 
        Parameters:
            func : ObjectiveFunction class
                A class with the objective function used to optimize the hyperparameters.
            optimizer : Optimizer class
                A class with the used optimization method.
            bounds : HPBoundaries class
                A class of the boundary conditions of the hyperparameters.
                Most of the global optimizers are using boundary conditions. 
                The bounds in this class will be used for the optimizer and func.
                The bounds have to be with the hyperparameter names used in the objective function.
            use_update_pdis : bool
                Whether to update the prior distributions of the hyperparameters with the given boundary conditions.
            get_prior_mean : bool
                Whether to get the parameters of the prior mean in the solution.
            use_stored_sols : bool
                Whether to store the solutions.
        """
        super().__init__(func,
                         optimizer=optimizer,
                         bounds=bounds,
                         use_update_pdis=use_update_pdis,
                         get_prior_mean=get_prior_mean,
                         use_stored_sols=use_stored_sols,
                         **kwargs)
    
    def get_hyperparams(self,hp,model,**kwargs):
        " Get default hyperparameters if they are not given. "
        if hp is None:
            # Get the hyperparameters from the model
            hp=model.hp.copy()
        # Convert to hyperparameter used in the objective function
        hp=self.convert_hp_from_gpatom(hp)
        # Get the values and hyperparameter names
        theta,parameters=self.hp_to_theta(hp)
        return hp,theta,parameters
    
    def update_pdis(self,pdis,model,X,Y,parameters,**kwargs):
        " Update the prior distributions of the hyperparameters with the boundary conditions. "
        pdis=self.convert_dict_object_to_gpatom(pdis)
        return super().update_pdis(pdis,model,X,Y,parameters,**kwargs)
    
    def get_full_hp(self,sol,model,**kwargs):
        " Get the full hyperparameter dictionary with hyperparameters that are optimized and are not. "
        if 'hp' in sol.keys():
            sol['hp']=self.convert_hp_to_gpatom(sol['hp'],model)
        sol['full hp']=model.hp.copy()
        sol['full hp'].update(sol['hp'])
        if 'prefactor' in sol['full hp'].keys():
            sol['full hp'].pop('prefactor')
        sol['full hp']['noise']=sol['full hp']['weight']*sol['full hp']['ratio']
        return sol

    def copy_model(self,model,**kwargs):
        " Copy the model and check if the noisefactor is not used in the factorization method. "
        model=deepcopy(model)
        if 'noisefactor' in model.hp.keys():
            from .objectivefunctions.factorized_likelihood import FactorizedLogLikelihood
            if isinstance(self.func,FactorizedLogLikelihood):
                if model.hp['noisefactor']!=1.0:
                    raise Exception('Noisefactor must be 1.0 for the Factorization method') 
        return model
    
    def convert_hp_from_gpatom(self,hp,**kwargs):
        " Convert the hyperparameters from GP-atom to the form here. "
        parameters=list(hp.keys())
        hp_new={}
        if 'scale' in parameters:
            hp_new['length']=np.array(np.log(hp['scale'])).reshape(-1)
        if 'weight' in parameters:
            hp_new['prefactor']=np.array(np.log(hp['weight'])).reshape(-1)
        if 'ratio' in parameters:
            hp_new['noise']=np.array(np.log(hp['ratio'])).reshape(-1)
        return hp_new
    
    def convert_hp_to_gpatom(self,hp,model,**kwargs):
        " Convert the hyperparameters from here to the form of GP-atom. "
        parameters=list(hp.keys())
        hp_new={}
        if 'length' in parameters:
            hp_new['scale']=np.array(np.exp(hp['length'])).reshape(-1)
        if 'prefactor' in parameters:
            hp_new['weight']=np.array(np.exp(hp['prefactor'])).reshape(-1)[0]
        if 'noise' in parameters:
            hp_new['ratio']=np.array(np.exp(hp['noise'])).reshape(-1)[0]
        return hp_new

    def convert_dict_object_to_gpatom(self,dict_obj,**kwargs):
        " Convert a dictionary with objects with GPatom hyperparameter names to the form here."
        if dict_obj is None:
            return dict_obj
        dict_obj_new={}
        for key,value in dict_obj.items():
            if key=='scale':
                dict_obj_new['length']=dict_obj['scale'].copy()
            if key=='weight':
                dict_obj_new['prefactor']=dict_obj['weight'].copy()
            if key=='ratio':
                dict_obj_new['noise']=dict_obj['ratio'].copy()
            else:
                dict_obj_new[key]=value.copy()
        return dict_obj_new
