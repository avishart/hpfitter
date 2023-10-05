import numpy as np
from .boundary import HPBoundaries

class VariableTransformation(HPBoundaries):
    def __init__(self,var_dict={},bounds=None,s=0.14,eps=np.finfo(float).eps,**kwargs):
        """ 
        Make variable transformation of hyperparameters into an interval of (0,1). 
        A dictionary of mean and standard deviation values are used to make Logistic transformations.
        Boundary conditions can be used to calculate the variable transformation parameters. 
        Parameters:
            var_dict: dict
                A dictionary with the variable transformation parameters (mean,std) for each hyperparameter.
            bounds: Boundary condition class 
                A Boundary condition class that make the boundaries of the hyperparameters. 
                The boundaries are used to calculate the variable transformation parameters. 
            s: float
                The scale parameter in a Logistic distribution. 
                It determines how large part of the distribution that is within the boundaries. 
                s=0.5*p/(ln(p)-ln(1-p)) with p being the quantile that the boundaries constitute.
            eps: float
                The first value of a grid in the variable transformed hyperparameter space.
                The last value of a grid is 1.0-eps. 
        """
        self.initial_parameters(var_dict)
        if bounds is None:
            from .strict import StrictBoundaries
            self.bounds=StrictBoundaries(bounds_dict={},scale=1.0,log=True,use_prior_mean=True)
        else:
            self.bounds=bounds.copy()
        if self.bounds.log==False:
            raise Exception('The Variable Transformation need to use boundary conditions in the log-scale!')
        self.s=s
        self.eps=eps

    def update_bounds(self,model,X,Y,parameters,**kwargs):
        """ 
        Create and update the boundary conditions for the hyperparameters. 
        Therefore the variable transformation parameters are also updated. 
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
        # Update the parameters used
        self.get_parameters_set(parameters)
        # Update the boundary conditions and get them
        self.bounds.update_bounds(model,X,Y,parameters)
        self.bounds_dict=self.bounds.get_bounds(array=False)
        # Update the variable transformation parameters
        for para,bound in self.bounds_dict.items():
            self.var_dict[para]={'mean':np.mean(bound,axis=1),
                                 'std':self.s*np.abs(bound[:,1]-bound[:,0])}
        return self
    
    def get_variable_transformation_parameters(self,array=False,**kwargs):
        """ 
        Get the variable transformation parameters. 
        Parameters:
            array: bool
                Whether to get an array for the mean and std or a dictionary as output.
        Returns:
            dict: A dictionary of the variable transformation parameters. 
            If array=True, a dictionary with mean and std is given instead.
        """
        if array:
            var_dict_array={}
            var_dict_array['mean']=np.concatenate([self.var_dict[para]['mean'] for para in self.parameters_set],axis=0)
            var_dict_array['std']=np.concatenate([self.var_dict[para]['std'] for para in self.parameters_set],axis=0)
            return var_dict_array
        return self.var_dict.copy()
    
    def transformation(self,hp,array=False,**kwargs):
        """ 
        Transform the hyperparameters with the variable transformation to get a dictionary. 
        Parameters:
            hp: dict
                The dictionary of the hyperparameters
            array: bool
                Whether to get an array or a dictionary as output.
        Returns:
            (H) array: The variable transformed hyperparameters as an array if array=True.
            or
            dict: A dictionary of the variable transformed hyperparameters. 
        """
        if array:
            return np.concatenate([self.transform(theta,self.var_dict[para]['mean'],self.var_dict[para]['std']) for para,theta in hp.items()])
        return {para:self.transform(theta,self.var_dict[para]['mean'],self.var_dict[para]['std']) for para,theta in hp.items()}
    
    def reverse_trasformation(self,t,array=False,**kwargs):
        """ 
        Transform the variable transformed hyperparameters back to the hyperparameters dictionary.
        Parameters:
            t: dict
                The dictionary of the variable transformed hyperparameters
            array: bool
                Whether to get an array or a dictionary as output.
        Returns:
            (H) array: The retransformed hyperparameters as an array if array=True.
            or
            dict: A dictionary of the retransformed hyperparameters. 
        """
        if array:
            return np.concatenate([self.retransform(ti,self.var_dict[para]['mean'],self.var_dict[para]['std']) for para,ti in t.items()])
        return {para:self.retransform(ti,self.var_dict[para]['mean'],self.var_dict[para]['std']) for para,ti in t.items()}
    
    def get_bounds(self,array=False,transformed=False,**kwargs):
        """ 
        Get the boundary conditions of the hyperparameters. 
        Parameters:
            array: bool
                Whether to get an array or a dictionary as output.
            transformed: bool
                If transformed=True, the boundaries is in variable transformed space.
                If transformed=False, the boundaries is transformed back to hyperparameter space.
        Returns:
            (H,2) array: The boundary conditions as an array if array=True.
            or
            dict: A dictionary of the boundary conditions. 
        """
        # Get the bounds in the variable transformed space 
        if transformed:
            if array:
                return np.full((len(self.parameters),2),[self.eps,1.00-self.eps])
            return {para:np.full((len(self.bounds_dict[para]),2),[self.eps,1.00-self.eps]) for para in self.parameters_set}
        # Get the bounds in the hyperparameter space 
        return self.bounds.get_bounds(array=array)
    
    def get_hp(self,array=False,transformed=False,**kwargs):
        """ 
        Get the guess of the hyperparameters. 
        The mean of the boundary conditions in log-space is used as the guess. 
        Parameters:
            array: bool
                Whether to get an array or a dictionary as output.
            transformed: bool
                If transformed=True, the hyperparameters is in variable transformed space.
                If transformed=False, the hyperparameters is transformed back to hyperparameter space.
        Returns:
            (H) array: The guesses of the hyperparameters as an array if array=True.
            or
            dict: A dictionary of the guesses of the hyperparameters. 
        """
        # Get the hyperparameter guess in the variable transformed space (so in the middle 0.5)
        if transformed:
            if array:
                return np.full((len(self.parameters)),0.50)
            return {para:np.full((len(self.bounds_dict[para])),0.50) for para in self.parameters_set}
        # Get the hyperparameter guess in the hyperparameter space 
        return self.bounds.get_hp(array=array)

    def make_lines(self,ngrid=80,transformed=False,**kwargs):
        """ 
        Make grid in each dimension of the hyperparameters from the boundary conditions.
        Parameters:
            ngrid: int or (H) list
                A integer or a list with number of grid points in each dimension.
            transformed: bool
                If transformed=True, the grid is in variable transformed space.
                If transformed=False, the grid is transformed back to hyperparameter space.
        Returns:
            (H,) list: A list with grid points for each (H) hyperparameters.
        """
        # Get the number of parameters
        n_parameters=len(self.parameters)
        # Make sure that a list of number grid points is used
        if isinstance(ngrid,(int,float)):
            ngrid=[int(ngrid)]*n_parameters
        # The grid is made within the variable transformed hyperparameters
        if transformed:
            return [np.linspace(self.eps,1.00-self.eps,ngrid[i]) for i in range(n_parameters)]
        # The grid is made within the variable transformed hyperparameters and then transformed back
        var_dict_array=self.get_variable_transformation_parameters(array=True)
        lines=[]
        for i,(vt_mean,vt_std) in enumerate(zip(var_dict_array['mean'],var_dict_array['std'])):
            t_line=np.linspace(self.eps,1.00-self.eps,ngrid[i])
            lines.append(self.retransform(t_line,vt_mean,vt_std))
        return lines
    
    def make_single_line(self,parameter,ngrid=80,i=0,transformed=False,**kwargs):
        """ 
        Make grid in each dimension of the hyperparameters from the boundary conditions.
        Parameters:
            parameters: str
                A string of the hyperparameter name.
            ngrid: int
                A integer with number of grid points in each dimension.
            i: int
                The index of the hyperparameter used if multiple hyperparameters of the same type exist.
            transformed: bool
                If transformed=True, the grid is in variable transformed space.
                If transformed=False, the grid is transformed back to hyperparameter space.
        Returns: 
            (ngrid) array: A grid of ngrid points for the given hyperparameter.
        """
        # Make sure that a int of number grid points is used
        if not isinstance(ngrid,(int,float)):
            ngrid=ngrid[int(self.parameters.index(parameter)+i)]
        # The grid is made within the variable transformed hyperparameters
        t_line=np.linspace(self.eps,1.00-self.eps,ngrid)
        if transformed:
            return t_line
        # The grid is transformed back to hyperparameter space
        return self.retransform(t_line,self.var_dict[parameter]['mean'][i],self.var_dict[parameter]['std'][i])
    
    def sample_thetas(self,npoints=50,transformed=False,**kwargs):
        """ 
        Sample hyperparameters from the variable transformed hyperparameter space.
        The sampled variable transformed hyperparameters are then transformed back to the hyperparameter space. 
        Parameters:
            npoints: int
                Number of points to sample.
            transformed: bool
                If transformed=True, the samples are in variable transformed space.
                If transformed=False, the samples are transformed back to hyperparameter space.
        Returns:
            (npoints,H) array: An array with sampled hyperparameters. 
        """
        # Sample the hyperparameters from the variable transformed hyperparameter space
        samples=np.random.uniform(low=self.eps,high=1.00-self.eps,size=(npoints,len(self.parameters)))
        # The samples are made within the variable transformed hyperparameters
        if transformed:
            return samples
        # The samples are transformed back to hyperparameter space
        var_dict_array=self.get_variable_transformation_parameters(array=True)
        for i,(vt_mean,vt_std) in enumerate(zip(var_dict_array['mean'],var_dict_array['std'])):
            samples[:,i]=self.retransform(samples[:,i],vt_mean,vt_std)
        return samples
    
    def transform(self,theta,vt_mean,vt_std,**kwargs):
        " Transform the hyperparameters with the variable transformation. "
        return 1.0/(1.0+np.exp(-(theta-vt_mean)/vt_std))

    def retransform(self,ti,vt_mean,vt_std,**kwargs):
        " Transform the variable transformed hyperparameters back to the hyperparameters. "
        return self.numeric_limits(vt_std*np.log(ti/(1.00-ti))+vt_mean)

    def numeric_limits(self,value,dh=0.1*np.log(np.finfo(float).max)):
        " Replace hyperparameters if they are outside of the numeric limits in log-space. "
        return np.where(-dh<value,np.where(value<dh,value,dh),-dh)
    
    def initial_parameters(self,var_dict,**kwargs):
        " Make and store the hyperparameter types and the dictionary with transformation parameters. "
        self.var_dict={key:np.array(value) for key,value in var_dict.items()}
        self.parameters_set=sorted(var_dict.keys())
        if 'correction' in self.parameters_set:
            self.parameters_set.remove('correction')
        self.parameters=sum([[para]*len(var_dict[para]) for para in self.parameters_set],[])
        return self

    def copy(self):
        " Copy the object. "
        return self.__class__(var_dict=self.var_dict,bounds=self.bounds,s=self.s,eps=self.eps)
    
    def __repr__(self):
        return "VariableTransformation(var_dict={},bounds={},s={},eps={})".format(self.var_dict,self.bounds,self.s,self.eps)
    