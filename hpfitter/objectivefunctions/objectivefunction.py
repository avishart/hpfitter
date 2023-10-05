import numpy as np
from scipy.linalg import cho_factor,cho_solve
from numpy.linalg import eigh

class ObjectiveFuction:
    def __init__(self,get_prior_mean=False,**kwargs):
        """ 
        The objective function that is used to optimize the hyperparameters. 
        Parameters:
            get_prior_mean: bool
                Whether to save the parameters of the prior mean in the solution.
        """
        self.reset_solution()
        self.get_prior_mean=get_prior_mean

    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        """ 
        The function call that calculate the objective function. 
        Parameters:
            theta: (H) array of floats
                An array with the hyperparameter values used for the objective function.
            parameters: (H) list of strings
                A list of names of the hyperparameters.
            model: Model
                The Machine Learning Model with kernel and prior that are optimized.
            X: (N,D) array
                Training features with N data points and D dimensions.
            Y: (N,1) array or (N,D+1) array
                Training targets without or with derivatives with N data points.
            pdis: dict
                A dict of prior distributions for each hyperparameter type.
            jac: bool
                Whether to get the derivatives of the objective function wrt. the hyperparameters. 
        Returns:
            float: The objective function value.
            and/or
            (H) array: The derivative of the objective function value wrt. the hyperparameters if jac=True.
        """
        raise NotImplementedError()
    
    def derivative(self,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        raise NotImplementedError()

    def make_hp(self,theta,parameters,**kwargs):
        " Make hyperparameter dictionary from lists"
        theta,parameters=np.array(theta),np.array(parameters)
        parameters_set=sorted(set(parameters))
        hp={para_s:self.numeric_limits(theta[parameters==para_s]) for para_s in parameters_set}
        return hp,parameters_set
    
    def get_hyperparams(self,model,**kwargs):
        " Get the hyperparameters for the model and the kernel. "
        return model.get_hyperparams()
    
    def numeric_limits(self,array,dh=0.1*np.log(np.finfo(float).max)):
        " Replace hyperparameters if they are outside of the numeric limits in log-space. "
        return np.where(-dh<array,np.where(array<dh,array,dh),-dh)
    
    def update(self,model,hp,**kwargs):
        " Update the hyperparameters of the machine learning model "
        model.set_hyperparams(hp)
        return model
    
    def kxx_corr(self,model,X,**kwargs):
        " Get covariance matrix with or without noise correction. "
        # Calculate the kernel with and without noise
        KXX=model.kernel(X,get_derivatives=model.use_derivatives)
        n_data=len(KXX)
        KXX=self.add_correction(model,KXX,n_data)
        return KXX,n_data
    
    def add_correction(self,model,KXX,n_data,**kwargs):
        " Add noise correction to covariance matrix. "
        corr=model.get_correction(np.diag(KXX))
        if corr>0.0:
            KXX[range(n_data),range(n_data)]+=corr
        return KXX
        
    def kxx_reg(self,model,X,**kwargs):
        " Get covariance matrix with regularization. "
        KXX=model.kernel(X,get_derivatives=model.use_derivatives)
        KXX_n=model.add_regularization(KXX,len(X),overwrite=False)
        return KXX_n,KXX,len(KXX)
        
    def y_prior(self,X,Y,model,L=None,low=None,**kwargs):
        " Update prior and subtract target. "
        Y_p=Y.copy()
        model.prior.update(X,Y_p,L=L,low=low,**kwargs)
        if model.use_derivatives:
            return (Y_p-model.prior.get(X,Y_p,get_derivatives=True)).T.reshape(-1,1)
        return (Y_p-model.prior.get(X,Y_p,get_derivatives=False))[:,0:1]
    
    def coef_cholesky(self,model,X,Y,**kwargs):
        " Calculate the coefficients by using Cholesky decomposition. "
        # Calculate the kernel with and without noise
        KXX_n,KXX,n_data=self.kxx_reg(model,X)
        # Cholesky decomposition
        L,low=cho_factor(KXX_n)
        # Subtract the prior mean to the training target
        Y_p=self.y_prior(X,Y,model,L=L,low=low)
        # Get the coefficients
        coef=cho_solve((L,low),Y_p,check_finite=False)
        return coef,L,low,Y_p,KXX,n_data

    def get_eig(self,model,X,Y,**kwargs):
        " Calculate the eigenvalues. " 
        # Calculate the kernel with and without noise
        KXX,n_data=self.kxx_corr(model,X)
        # Eigendecomposition
        try:
            D,U=eigh(KXX)
        except Exception as e:
            import logging
            import scipy.linalg
            logging.error("An error occurred: %s", str(e))
            # More robust but slower eigendecomposition
            D,U=scipy.linalg.eigh(KXX,driver='ev')
        # Subtract the prior mean to the training target
        Y_p=self.y_prior(X,Y,model,D=D,U=U)
        UTY=(np.matmul(U.T,Y_p)).reshape(-1)**2
        return D,U,Y_p,UTY,KXX,n_data
    
    def get_cinv(self,model,X,Y,**kwargs):
        " Get the inverse covariance matrix. "
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        cinv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        return coef,cinv,Y_p,KXX,n_data
    
    def logpriors(self,hp,parameters_set,parameters,pdis=None,jac=False,**kwargs):
        " Log of the prior distribution value for the hyperparameters. "
        if pdis is None:
            return 0.0
        if not jac:
            lprior=0.0
            for para in parameters_set:
                if para in pdis.keys():
                    lprior=lprior+pdis[para].ln_pdf(hp[para])
            if isinstance(lprior,float):
                return lprior
            return lprior.reshape(-1)
        lprior_deriv=np.array([])
        for para in parameters_set:
            if para in pdis.keys():
                lprior_deriv=np.append(lprior_deriv,np.array(pdis[para].ln_deriv(hp[para])).reshape(-1))
            else:
                lprior_deriv=np.append(lprior_deriv,np.zeros((parameters.count(para))))
        return lprior_deriv

    def get_K_inv_deriv(self,K_deriv,KXX_inv,**kwargs):
        " Get the diagonal elements of the matrix product of the inverse and derivative covariance matrix. "
        return np.einsum('ij,dji->d',KXX_inv,K_deriv)
    
    def reset_solution(self):
        " Reset the solution of the optimization in terms of hyperparameters and model. "
        self.sol={'fun':np.inf,'x':np.array([]),'hp':{}}
        return self
    
    def update_solution(self,fun,theta,parameters,model,jac=False,deriv=None,**kwargs):
        """
        Update the solution of the optimization in terms of hyperparameters and model.
        The lowest objective function value is stored togeher with its hyperparameters.
        The prior mean can also be saved if get_prior_mean=True.
        """
        if fun<self.sol['fun']:
            self.sol['fun']=fun
            self.sol['x']=theta.copy()
            self.sol['hp']=self.make_hp(theta,parameters)[0]
            if jac:
                self.sol['jac']=deriv.copy()
            if self.get_prior_mean:
                self.sol['prior']=model.prior.get_parameters()
        return self.sol
    
    def get_solution(self,sol,parameters,model,X,Y,pdis=None,**kwargs):
        " Get the solution of the optimization in terms of hyperparameters and model. "
        if self.sol['fun']<=sol['fun']:
            sol['fun']=self.sol['fun']
            sol['x']=self.sol['x'].copy()
            sol['hp']=self.sol['hp'].copy()
            if 'jac' in self.sol.keys():
                sol['jac']=self.sol['jac'].copy()
            if 'prior' in self.sol.keys():
                sol['prior']=self.sol['prior'].copy()
        else:
            sol['hp']=self.make_hp(sol['x'],parameters)[0]
            # Can also give prior arguments
            if self.get_prior_mean:
                sol['prior']=model.prior.get_parameters()
        return sol
    
    def copy(self):
        " Copy the objective function object. "
        return self.__class__(get_prior_mean=self.get_prior_mean)
    
    def __repr__(self):
        return "{}(get_prior_mean={})".format(self.__class__.__name__,self.get_prior_mean)
    