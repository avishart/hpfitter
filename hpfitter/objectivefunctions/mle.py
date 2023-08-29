import numpy as np
from scipy.linalg import cho_solve
from .objectivefunction_gpatom import ObjectiveFuctionGPAtom

class MaximumLogLikelihood(ObjectiveFuctionGPAtom):
    def __init__(self,modification=False,**kwargs):
        """ Maximum log-likelihood objective function as a function of the hyperparameters. 
            The prefactor hyperparameter is calculated from an analytical expression. """
        super().__init__(**kwargs)
        self.modification=modification
    
    def function(self,theta,parameters,model,X,Y,pdis=None,jac=False,**kwargs):
        " The objective function value. "
        hp,parameters_set=self.make_hp(theta,parameters)
        model=self.update(model,hp)
        coef,L,low,Y_p,KXX,n_data=self.coef_cholesky(model,X,Y)
        prefactor2=np.matmul(Y_p.T,coef).item(0)/n_data
        nlp=0.5*n_data*(1+np.log(2.0*np.pi)+np.log(prefactor2))+np.sum(np.log(np.diagonal(L)))
        nlp=nlp.item(0)-self.logpriors(hp,parameters_set,parameters,pdis,jac=False)
        if jac:
            deriv=self.derivative(hp,parameters_set,parameters,model,X,KXX,L,low,coef,prefactor2,n_data,pdis,**kwargs) 
            self.update_solution(nlp,theta,parameters,model,jac=jac,deriv=deriv,Y_p=Y_p,coef=coef,n_data=n_data)
            return nlp,deriv
        self.update_solution(nlp,theta,parameters,model,jac=jac,Y_p=Y_p,coef=coef,n_data=n_data)
        return nlp
    
    def derivative(self,hp,parameters_set,parameters,model,X,KXX,L,low,coef,prefactor2,n_data,pdis,**kwargs):
        " The derivative of the objective function wrt. the hyperparameters. "
        nlp_deriv=np.array([])
        KXX_inv=cho_solve((L,low),np.identity(n_data),check_finite=False)
        for para in parameters_set:
            if para=='prefactor':
                nlp_deriv=np.append(nlp_deriv,np.zeros((len(hp[para]))))
                continue
            K_deriv=model.get_gradients(X,[para],KXX=KXX)[para]
            K_deriv_cho=self.get_K_inv_deriv(K_deriv,KXX_inv)
            nlp_deriv=np.append(nlp_deriv,(-(0.5*np.matmul(coef.T,np.matmul(K_deriv,coef)).reshape(-1))/prefactor2)+0.5*K_deriv_cho)
        nlp_deriv=nlp_deriv-self.logpriors(hp,parameters_set,parameters,pdis,jac=True)
        return nlp_deriv
    
    def update_solution(self,fun,theta,parameters,model,jac=False,get_prior=False,deriv=None,Y_p=None,coef=None,n_data=None,**kwargs):
        " Update the solution of the optimization in terms of hyperparameters and model. "
        if fun<self.sol['fun']:
            hp,parameters_set=self.make_hp(theta,parameters)
            prefactor2=np.matmul(Y_p.T,coef).item(0)/n_data
            if self.modification:
                prefactor2=(n_data/(n_data-len(theta)))*prefactor2 if n_data-len(theta)>0 else prefactor2
            hp.update(dict(prefactor=np.array([0.5*np.log(prefactor2)])))
            self.sol['x']=np.array(sum([list(np.array(hp[para]).reshape(-1)) for para in parameters_set],[]))
            self.sol['hp']=hp.copy()
            self.sol['fun']=fun
            if jac:
                self.sol['jac']=deriv.copy()
            if get_prior:
                self.sol['prior']=model.prior.get_parameters()
        return self.sol
    
    def copy(self):
        " Copy the objective function object. "
        return self.__class__(modification=self.modification)
