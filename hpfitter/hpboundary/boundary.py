import numpy as np

class HPBoundaries:
    def __init__(self,bounds_dict={},scale=1.0,log=True,**kwargs):
        """ 
        Boundary conditions for the hyperparameters.
        A dictionary with boundary conditions of the hyperparameters can be given as an argument.
        Machine precisions are used as boundary conditions for the hyperparameters not given in the dictionary.
        Parameters:
            bounds_dict: dict
                A dictionary with boundary conditions as numpy (H,2) arrays with two columns for each type of hyperparameter.
            scale: float
                Scale the boundary conditions.
            log: bool
                Whether to use hyperparameters in log-scale or not.
        """
        self.initial_parameters(bounds_dict)
        self.scale=scale
        self.log=log

    def update_bounds(self,model,X,Y,parameters,**kwargs):
        """ 
        Create and update the boundary conditions for the hyperparameters. 
        Parameters:
            model: Model
                The Machine Learning Model with kernel and prior that are optimized.
            X: (N,D) array
                Training features with N data points and D dimensions.
            Y: (N,1) array or (N,D+1) array
                Training targets with or without derivatives with N data points.
            parameters: (H) list of strings
                A list of names of the hyperparameters.
        Returns:
            self: The object itself.
        """
        # Update parameters
        self.get_parameters_set(parameters)
        # Make bounds (Parameters and keys in bounds_dict have to be the same)
        self.bounds_dict=self.make_bounds(model,X,Y,parameters,self.parameters_set)
        return self

    def get_bounds(self,array=False,**kwargs):
        """ 
        Get the boundary conditions of the hyperparameters. 
        Parameters:
            array: bool
                Whether to get an array or a dictionary as output.
        Returns:
            (H,2) array: The boundary conditions as an array if array=True.
            or
            dict: A dictionary of the boundary conditions. 
        """
        if array:
            return np.concatenate([self.bounds_dict[para] for para in self.parameters_set],axis=0)
        return self.bounds_dict.copy()
    
    def get_hp(self,array=False,**kwargs):
        """ 
        Get the guess of the hyperparameters. 
        The mean of the boundary conditions in log-space is used as the guess. 
        Parameters:
            array: bool
                Whether to get an array or a dictionary as output.
        Returns:
            (H) array: The guesses of the hyperparameters as an array if array=True.
            or
            dict: A dictionary of the guesses of the hyperparameters. 
        """
        if self.log:
            if array:
                return np.concatenate([np.mean(self.bounds_dict[para],axis=1) for para in self.parameters_set])
            return {para:np.mean(bound,axis=1) for para,bound in self.bounds_dict.items()}
        if array:
            return np.concatenate([np.exp(np.mean(np.log(self.bounds_dict[para]),axis=1)) for para in self.parameters_set])
        return {para:np.exp(np.mean(np.log(bound),axis=1)) for para,bound in self.bounds_dict.items()}
    
    def make_lines(self,ngrid=80,**kwargs):
        """ 
        Make grid in each dimension of the hyperparameters from the boundary conditions.
        Parameters:
            ngrid: int or (H) list
                A integer or a list with number of grid points in each dimension.
        Returns:
            (H,) list: A list with grid points for each (H) hyperparameters.
        """
        bounds=self.get_bounds(array=True)
        if isinstance(ngrid,(int,float)):
            ngrid=[int(ngrid)]*len(bounds)
        return [np.linspace(bound[0],bound[1],ngrid[b]) for b,bound in enumerate(bounds)]
    
    def make_single_line(self,parameter,ngrid=80,i=0,**kwargs):
        """ 
        Make grid in one dimension of the hyperparameters from the boundary conditions.
        Parameters:
            parameters: str
                A string of the hyperparameter name.
            ngrid: int
                A integer with number of grid points in each dimension.
            i: int
                The index of the hyperparameter used if multiple hyperparameters of the same type exist. 
        Returns: 
            (ngrid) array: A grid of ngrid points for the given hyperparameter.
        """
        if not isinstance(ngrid,(int,float)):
            ngrid=ngrid[int(self.parameters.index(parameter)+i)]
        bound=self.bounds_dict[parameter][i]
        return np.linspace(bound[0],bound[1],int(ngrid))
    
    def sample_thetas(self,npoints=50,**kwargs):
        """ 
        Sample hyperparameters from the boundary conditions. 
        Parameters:
            npoints: int
                Number of points to sample.   
        Returns:
            (npoints,H) array: An array with sampled hyperparameters. 
        """
        bounds=self.get_bounds(array=True)
        return np.random.uniform(low=bounds[:,0],high=bounds[:,1],size=(int(npoints),len(bounds)))
    
    def make_bounds(self,model,X,Y,parameters,parameters_set,**kwargs):
        " Make the boundary conditions with educated guesses of the length-scale hyperparamer(s). "
        eps_lower,eps_upper=self.get_boundary_limits()
        bounds={}
        for para in parameters_set:
            if para in self.bounds_dict:
                bounds[para]=self.bounds_dict[para].copy()
            else:
                bounds[para]=np.array([[eps_lower,eps_upper]]*parameters.count(para))
        return bounds
    
    def initial_parameters(self,bounds_dict,**kwargs):
        " Make and store the hyperparameter types and the dictionary with hyperparameter bounds. "
        self.bounds_dict={key:np.array(value) for key,value in bounds_dict.items()}
        self.parameters_set=sorted(bounds_dict.keys())
        if 'correction' in self.parameters_set:
            self.parameters_set.remove('correction')
        self.parameters=sum([[para]*len(bounds_dict[para]) for para in self.parameters_set],[])
        return self
    
    def get_parameters_set(self,parameters,**kwargs):
        " Get and store the hyperparameters types. "
        parameters=list(parameters)
        if 'correction' in parameters:
            parameters.remove('correction')
        self.parameters=sorted(parameters)
        self.parameters_set=sorted(set(parameters))
        return self.parameters_set
    
    def get_boundary_limits(self,**kwargs):
        " Get the machine precision limits for the hyperparameters. "
        eps_lower=10*np.sqrt(2.0*np.finfo(float).eps)/self.scale
        if self.log:
            eps_lower=np.log(eps_lower)
            return eps_lower,-eps_lower
        return eps_lower,1.0/eps_lower
    
    def copy(self):
        " Copy the object. "
        return self.__class__(bounds_dict=self.bounds_dict,scale=self.scale,log=self.log)
    
    def __repr__(self):
        return "HPBoundaries(bounds_dict={},scale={},log={})".format(self.bounds_dict,self.scale,self.log)
        