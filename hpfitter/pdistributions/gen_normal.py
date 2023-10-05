import numpy as np
from .pdistributions import Prior_distribution

class Gen_normal_prior(Prior_distribution):
    def __init__(self,mu=0.0,s=10.0,v=2,**kwargs):
        """ 
        Independent Generalized Normal prior distribution used for each type of hyperparameters in log-space. 
        If the type of the hyperparameter is multi dimensional (H) it is given in the axis=-1. 
        If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,H) array. 
        Parameters:
            mu: float or (H) array
                The mean of the generalized normal distribution. 
            std: float or (H) array
                The scale of the generalized normal distribution.
            v: float or (H) array
                The shape or magnitude of the generalized normal distribution.
        """
        self.update(mu=mu,s=s,v=v)
    
    def ln_pdf(self,x):
        if self.nosum:
            return -((x-self.mu)/self.s)**(2*self.v)-np.log(self.s)+np.log(0.52)
        return np.sum(-((x-self.mu)/self.s)**(2*self.v)-np.log(self.s)+np.log(0.52),axis=-1)
    
    def ln_deriv(self,x):
        return (-(2*self.v)*((x-self.mu)**(2*self.v-1)))/(self.s**(2*self.v))
    
    def update(self,mu=None,s=None,v=None,**kwargs):
        if mu is not None:
            if isinstance(mu,(float,int)):
                self.mu=mu
            else:
                self.mu=np.array(mu).reshape(-1)
        if s is not None:
            if isinstance(s,(float,int)):
                self.s=s
            else:
                self.s=np.array(s).reshape(-1)
        if v is not None:
            if isinstance(v,(float,int)):
                self.v=v
            else:
                self.v=np.array(v).reshape(-1)
        if isinstance(self.mu,(float,int)) and isinstance(self.s,(float,int)) and isinstance(self.v,(float,int)):
            self.nosum=True
        else:
            self.nosum=False
        return self
            
    def mean_var(self,mean,var):
        return self.update(mu=mean,s=np.sqrt(var/0.32))
    
    def min_max(self,min_v,max_v):
        mu=(max_v+min_v)/2.0
        return self.update(mu=mu,s=np.sqrt(2/0.32)*(max_v-mu))
    
    def copy(self):
        return self.__class__(mu=self.mu,s=self.s,v=self.v)
    
    def __str__(self):
        return 'Gen_normal_prior({},{},{})'.format(self.mu,self.s,self.v)

    def __repr__(self):
        return 'Generalized-normal({},{},{})'.format(self.mu,self.s,self.v)
