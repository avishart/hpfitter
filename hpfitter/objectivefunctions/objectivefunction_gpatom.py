import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .objectivefunction import ObjectiveFuction

class ObjectiveFuctionGPAtom(ObjectiveFuction):
    
    def convert_hp_to_gpatom(self,hp,model):
        " Convert the hyperparameters from here to the form of GP-atom. "
        parameters=list(hp.keys())
        hp_new={'weight':1.0}
        if 'length' in parameters:
            hp_new['scale']=np.array(np.exp(hp['length'])).reshape(-1)
        if 'prefactor' in parameters:
            # Prefactor is needed for likelihood function evaluations
            hp_new['prefactor']=np.array(hp['prefactor']).reshape(-1)
        else:
            hp_new['prefactor']=np.array(np.log(model.hp['weight'])).reshape(-1)
        if 'noise' in parameters:
            hp_new['ratio']=np.exp(hp['noise'])[0]
        return hp_new
    
    def get_hyperparams(self,model,**kwargs):
        " Get the hyperparameters for the model and the kernel. "
        return model.hp.copy()
    
    def update_model(self,model,hp,**kwargs):
        " Update the hyperparameters of the machine learning model "
        hp_new=self.convert_hp_to_gpatom(hp,model)
        model.set_hyperparams(hp_new)
        return model
    
    def kxx_corr(self,model,X,**kwargs):
        " Get covariance matrix with or without noise correction. "
        # Calculate the kernel with and without noise
        KXX=model.kernel.kernel_matrix(X)
        n_data=len(KXX)
        KXX=self.add_correction(model,KXX,n_data)
        return KXX,n_data
    
    def kxx_reg(self,model,X,Y,**kwargs):
        " Get covariance matrix with regularization. "
        KXX,n_data=self.kxx_corr(model,X)
        Natoms=int(((len(Y.flatten())/len(X))-1)/3)
        KXX_n=model.add_regularization(KXX.copy(),len(X),Natoms)
        return KXX_n,KXX,n_data
        
    def y_prior(self,X,Y,model,L=None,low=None,**kwargs):
        " Update prior and subtract target. "
        Y_p=Y.copy()
        model.prior.update(X,Y_p,L=L,low=low,**kwargs)
        #self.prior.update(self.X, self.Y, self.L)
        if model.use_forces:
            return (Y_p-model.prior.get(X,Y_p,get_derivatives=True)).reshape(-1,1)
        return (Y_p-model.prior.get(X,Y_p,get_derivatives=False))[:,0:1]
    
    def coef_cholesky(self,model,X,Y,**kwargs):
        " Calculate the coefficients by using Cholesky decomposition. "
        # Calculate the kernel with and without noise
        KXX_n,KXX,n_data=self.kxx_reg(model,X,Y)
        # Cholesky decomposition
        L,low=cho_factor(KXX_n)
        # Subtract the prior mean to the training target
        Y_p=self.y_prior(X,Y,model,L=L,low=low)
        # Get the coefficients
        coef=cho_solve((L,low),Y_p,check_finite=False)
        return coef,L,low,Y_p,KXX,n_data

