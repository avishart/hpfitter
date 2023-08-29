import numpy as np
from .pdistributions import Prior_distribution

class Gen_normal_prior(Prior_distribution):
    def __init__(self,mu=0.0,s=10.0,v=2,**kwargs):
        'Generalized normal distribution'
        self.update(mu=mu,s=s,v=v)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        if self.nosum:
            return -((x-self.mu)/self.s)**(2*self.v)-np.log(self.s)+np.log(0.52)
        return np.sum(-((x-self.mu)/self.s)**(2*self.v)-np.log(self.s)+np.log(0.52),axis=-1)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return (-(2*self.v)*((x-self.mu)**(2*self.v-1)))/(self.s**(2*self.v))
    
    def update(self,mu=None,s=None,v=None):
        'Update the parameters of distribution function'
        if mu!=None:
            self.mu=mu
        if s!=None:
            self.s=s
        if v!=None:
            self.v=v
        self.nosum=isinstance(self.mu,(float,int))
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        return self.update(mu=mean,s=np.sqrt(var/0.32))
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        mu=(max_v+min_v)/2.0
        return self.update(mu=mu,s=np.sqrt(2/0.32)*(max_v-mu))
    
    def copy(self):
        " Copy the prior distribution of the hyperparameter. "
        return self.__class__(mu=self.mu,s=self.s,v=self.v)
    
    def __str__(self):
        return 'Gen_normal_prior({},{},{})'.format(self.mu,self.s,self.v)

    def __repr__(self):
        return 'Generalized-normal({},{},{})'.format(self.mu,self.s,self.v)
