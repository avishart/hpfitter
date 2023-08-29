import numpy as np
from .pdistributions import Prior_distribution

class Uniform_prior(Prior_distribution):
    def __init__(self,start=-18.0,end=18.0,prob=1.0,**kwargs):
        'Uniform distribution'
        self.update(start=start,end=end,prob=prob)
    
    def ln_pdf(self,x):
        'Log of probability density function'
        ln_0=-np.log(np.nan_to_num(np.inf))
        if self.nosum:
            return np.where(x>=self.start,np.where(x<=self.end,np.log(self.prob),ln_0),ln_0)
        return np.sum(np.where(x>=self.start,np.where(x<=self.end,np.log(self.prob),ln_0),ln_0),axis=-1)
    
    def ln_deriv(self,x):
        'The derivative of the log of the probability density function as respect to x'
        return 0.0*x
    
    def update(self,start=None,end=None,prob=None):
        'Update the parameters of distribution function'
        if start!=None:
            self.start=start
        if end!=None:
            self.end=end
        if prob!=None:
            self.prob=prob
        self.nosum=isinstance(self.prob,(float,int))
        return self
            
    def mean_var(self,mean,var):
        'Obtain the parameters of the distribution function by the mean and variance values'
        std=np.sqrt(var)
        return self.update(start=mean-4.0*std,end=mean+4.0*std,prob=1/(8.0*std))
    
    def min_max(self,min_v,max_v):
        'Obtain the parameters of the distribution function by the minimum and maximum values'
        return self.update(start=min_v,end=max_v,prob=1.0/(max_v-min_v))
    
    def copy(self):
        " Copy the prior distribution of the hyperparameter. "
        return self.__class__(start=self.start,end=self.start,prob=self.prob)
    
    def __str__(self):
        return 'Uniform({},{})'.format(self.start,self.end)
    
    def __repr__(self):
        return 'Uniform_prior({},{},{})'.format(self.start,self.end,self.prob)
