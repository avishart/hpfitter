import numpy as np

def theta_to_hp(theta,parameters):
    " Transform a list of values and a list of parameter categories to  a dictionary of hyperparameters " 
    return {para_s:theta[np.array(parameters)==para_s] for para_s in sorted(set(parameters))}

def hp_to_theta(hp):
    " Transform a dictionary of hyperparameters to a list of values and a list of parameter categories " 
    parameters_set=sorted(hp.keys())
    theta=sum([list(hp[para]) for para in parameters_set],[])
    parameters=sum([[para]*len(hp[para]) for para in parameters_set],[])
    return np.array(theta),parameters

def make_grid(lines,maxiter=5000):
    "Make a grid in multi-dimensions from a list of 1D grids in each dimension"
    lines=np.array(lines)
    if len(lines.shape)<2:
        lines=lines.reshape(1,-1)
    #Number of combinations
    combi=1
    for i in [len(line) for line in lines]:
        combi*=i
    if combi<maxiter:
        maxiter=combi
    #If there is a low probability to find grid points randomly the entire grid are calculated
    if (1-(maxiter/combi))<0.99:
        X=lines[0].reshape(-1,1)
        lines=lines[1:]
        for line in lines:
            dim_X=len(X)
            X=np.concatenate([X]*len(line),axis=0)
            X=np.concatenate([X,np.sort(np.concatenate([line.reshape(-1)]*dim_X,axis=0)).reshape(-1,1)],axis=1)
        return np.random.permutation(X)[:maxiter]
    #Randomly sample the grid points
    X=np.array([np.random.choice(line,size=maxiter) for line in lines]).T
    X=np.unique(X,axis=0)
    while len(X)<maxiter:
        x=np.array([np.random.choice(line,size=1) for line in lines]).T
        X=np.append(X,x,axis=0)
        X=np.unique(X,axis=0)
    return X[:maxiter]

def anneal_var_trans(x,fun,hyper_var,parameters,model,X,Y,pdis=None,jac=False):
    " Object function called for simulated annealing, where hyperparameter transformation. "
    x=np.where(x<1.0,np.where(x>0.0,x,hyper_var.eps),1.00-hyper_var.eps)
    theta=hyper_var.reverse_trasformation(theta_to_hp(x,parameters),array=True)
    return fun.function(theta,parameters,model,X,Y,pdis,jac)

def calculate_list_values(line,fun,*args,**kwargs):
    " Calculate a list of values with a function. "
    return np.array([fun(theta,*args,**kwargs) for theta in line])

def find_minimas(xvalues,fvalues,i_min,len_l,multiple_min=True,theta_index=0,xtol=None,ftol=None,**kwargs):
    " Find all the local minimums and their indicies or just the global minimum and then check convergence. "
    # Investigate multiple minimums
    if multiple_min:
        # Find local minimas for middel part of line 
        i_minimas=np.where((fvalues[1:-1]<fvalues[:-2])&(fvalues[2:]>fvalues[1:-1]))[0]+1
        # Check if the function values for the minimums are within the tolerance
        if len(i_minimas):
            i_keep=abs(fvalues[i_minimas+1]+fvalues[i_minimas-1]-2.0*fvalues[i_minimas])>=ftol*(1.0+abs(fvalues[i_minimas]))
            i_minimas=i_minimas[i_keep]
        # Find local minimas for end parts of line
        if fvalues[0]-fvalues[1]<-ftol:
            i_minimas=np.append([1],i_minimas)
        if fvalues[-1]-fvalues[-2]<-ftol:
            i_minimas=np.append(i_minimas,[len_l-2])
        # Check the distances in the local minimas
        if len(i_minimas):
            if theta_index is None:
                i_keep=np.linalg.norm(xvalues[i_minimas+1]-xvalues[i_minimas-1],axis=1)>=xtol*(1.0+np.linalg.norm(xvalues[i_minimas],axis=1))
            else:
                i_keep=abs(xvalues[i_minimas+1,theta_index]-xvalues[i_minimas-1,theta_index])>=xtol*(1.0+abs(xvalues[i_minimas,theta_index]))
            i_minimas=i_minimas[i_keep]
        # Sort the indicies after function value sizes
        if len(i_minimas)>1:
            i_sort=np.argsort(fvalues[i_minimas])
            i_minimas=i_minimas[i_sort]
        return i_minimas
    # Investigate the global minimum
    i_minimas=[]
    if i_min==0:
        # Check if the function values are converged if the endpoints is the minimum value
        if fvalues[0]-fvalues[1]<-ftol:
            i_minimas=[i_min+1]
    elif i_min==int(len_l-1):
        # Check if the function values are converged if the endpoints is the minimum value
        if fvalues[-1]-fvalues[-2]<-ftol:
            i_minimas=[i_min-1]
    else:
        # Check if the function values are converged
        if abs(fvalues[i_min+1]+fvalues[i_min-1]-2.0*fvalues[i_min])>=ftol*(1.0+abs(fvalues[i_min])):
            i_minimas=[i_min]
    # Check if the distance in the local minimum is converged
    if len(i_minimas):
        i_minima=i_minimas[0]
        if not np.linalg.norm(xvalues[i_minima+1]-xvalues[i_minima-1])>=xtol*(1.0+np.linalg.norm(xvalues[i_minima])):
            i_minimas=[]
    return np.array(i_minimas)
