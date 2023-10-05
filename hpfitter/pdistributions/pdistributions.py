import numpy as np

def make_pdis(model,parameters,X,Y,bounds=None,prior_dis=None,**kwargs):
    " Make prior distribution for hyperparameters from educated guesses in log space. "
    # Make boundary conditions for updating the prior distributions
    if bounds is None:
        # Use strict educated guesses for the boundary conditions if not given
        from ..hpboundary.strict import StrictBoundaries
        bounds=StrictBoundaries(log=True,use_prior_mean=True)
    # Update boundary conditions to the data
    bounds.update_bounds(model,X,Y,parameters)
    # Make prior distributions for hyperparameters from boundary conditions
    pdis={}
    for para,bound in bounds.get_bounds().items():
        if prior_dis is None or para not in pdis.keys():
            # Use Normal prior distribution as default
            from .normal import Normal_prior
            pdis[para]=Normal_prior().min_max(bound[:,0],bound[:,1])
        else:
            # Use given prior distributions to update them 
            pdis[para]=prior_dis[para].min_max(bound[:,0],bound[:,1])
    return pdis


class Prior_distribution:
    def __init__(self,**kwargs):
        """ 
        Prior probability distribution used for each type of hyperparameters in log-space. 
        If the type of the hyperparameter is multi dimensional (H) it is given in the axis=-1. 
        If multiple values (M) of the hyperparameter(/s) are calculated simultaneously it has to be in a (M,H) array. 
        """
        
    def pdf(self,x):
        """ 
        Probability density function.
        Parameter:
            x: float or (M,H) array
                x is the hyperparameter value used by the prior distribution.
                x can be a float if the prior distribution only consider 
                a single hyperparameter of that type.
                x can be a (1,H) array if the prior distribution consider 
                H hyperparameter of that type.  
                x can be a (M,H) array if the prior distribution consider 
                H hyperparameter of that type with M different values.  
        Returns: 
            float: Value of the probability density function.
            or
            (M) array: M values of the probability density function if M different values is given.
        """
        return np.exp(self.ln_pdf(x))
    
    def deriv(self,x):
        " The derivative of the probability density function as respect to x. "
        return self.pdf(x)*self.ln_deriv(x)
    
    def ln_pdf(self,x):
        """ 
        Log of probability density function.
        Parameter:
            x: float or (M,H) array
                x is the hyperparameter value used by the prior distribution.
                x can be a float if the prior distribution only consider 
                a single hyperparameter of that type.
                x can be a (1,H) array if the prior distribution consider 
                H hyperparameter of that type.  
                x can be a (M,H) array if the prior distribution consider 
                H hyperparameter of that type with M different values.  
        Returns: 
            float: Value of the log of probability density function.
            or
            (M) array: M values of the log of probability density function if M different values is given.
        """
        raise NotImplementedError()
    
    def ln_deriv(self,x):
        " The derivative of the log of the probability density function as respect to x. "
        raise NotImplementedError()
    
    def update(self,**kwargs):
        " Update the parameters of distribution function. "
        raise NotImplementedError()
            
    def mean_var(self,mean,var):
        " Obtain the parameters of the distribution function by the mean and variance values. "
        raise NotImplementedError()
    
    def min_max(self,min_v,max_v):
        " Obtain the parameters of the distribution function by the minimum and maximum values. "
        raise NotImplementedError()
    
    def copy(self):
        " Copy the prior distribution of the hyperparameter. "
        return self.__class__()
    
    def __str__(self):
        return 'Prior()'

    def __repr__(self):
        return 'Prior_distribution()'
