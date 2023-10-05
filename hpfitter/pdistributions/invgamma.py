import numpy as np
from .pdistributions import Prior_distribution
from scipy.special import loggamma

class Invgamma_prior(Prior_distribution):
    def __init__(self,a=1e-20,b=1e-20,**kwargs):
        """ 
        Inverse-Gamma prior distribution used for each type of hyperparameters in log-space. 
        The inverse-gamma distribution is variable transformed from linear- to log-space.
        If the type of the hyperparameter is multi dimensional (H) it is given in the axis=-1. 
        If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,H) array. 
        Parameters:
            a: float or (H) array
                The shape parameter. 
            b: float or (H) array
                The scale parameter.
        """
        self.update(a=a,b=b)
    
    def ln_pdf(self,x):
        if self.nosum:
            return self.lnpre-2*self.a*x-self.b*np.exp(-2*x)
        return np.sum(self.lnpre-2*self.a*x-self.b*np.exp(-2*x),axis=-1)
    
    def ln_deriv(self,x):
        return -2*self.a+2*self.b*np.exp(-2*x)
    
    def update(self,a=None,b=None,**kwargs):
        if a is not None:
            if isinstance(a,(float,int)):
                self.a=a
            else:
                self.a=np.array(a).reshape(-1)
        if b is not None:
            if isinstance(b,(float,int)):
                self.b=b
            else:
                self.b=np.array(b).reshape(-1)
        self.lnpre=self.a*np.log(self.b)-loggamma(self.a)
        if isinstance(self.a,(float,int)) and isinstance(self.b,(float,int)):
            self.nosum=True
        else:
            self.nosum=False
        return self
            
    def mean_var(self,mean,var):
        mean,var=np.exp(mean),np.exp(2*np.sqrt(var))
        min_v=mean-np.sqrt(var)*2
        return self.update(a=min_v,b=min_v)
    
    def min_max(self,min_v,max_v):
        b=np.exp(2*min_v)
        return self.update(a=b,b=b)
    
    def copy(self):
        return self.__class__(a=self.a,b=self.b)
    
    def __str__(self):
        return 'Inverse-Gamma({},{})'.format(self.a,self.b)
    
    def __repr__(self):
        return 'Invgamma_prior({},{})'.format(self.a,self.b)
