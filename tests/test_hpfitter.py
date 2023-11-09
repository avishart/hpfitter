import unittest
import numpy as np
from gpatom.gpfp.fingerprint import CartesianCoordFP

def create_h2_atoms(gridsize=200,seed=1):
    " Generate the trial data set of H2 ASE atoms with EMT "
    from ase import Atoms
    from ase.calculators.emt import EMT
    z_list=np.linspace(0.2,4.0,gridsize)
    atoms_list=[]
    energies,forces=[],[]
    for z in z_list:
        h2=Atoms('H2',positions=np.array([[0.0,0.0,0.0],[z,0.0,0.0]]))
        h2.center(vacuum=10.0)
        h2.calc=EMT()
        energies.append(h2.get_potential_energy())
        forces.append(h2.get_forces().reshape(-1))
        atoms_list.append(h2)
    np.random.seed(seed)
    i_perm=np.random.permutation(list(range(len(atoms_list))))
    atoms_list=[atoms_list[i] for i in i_perm]
    return atoms_list,np.array(energies).reshape(-1,1)[i_perm],np.array(forces)[i_perm]

def make_train_test_set(x,f,g,tr=20,te=20,use_derivatives=True):
    " Genterate the training and test sets "
    x_tr,f_tr,g_tr=x[:tr],f[:tr],g[:tr]
    x_te,f_te,g_te=x[tr:tr+te],f[tr:tr+te],g[tr:tr+te]
    if use_derivatives:
        f_tr=np.concatenate([f_tr.reshape(tr,1),g_tr.reshape(tr,-1)],axis=1)
        f_te=np.concatenate([f_te.reshape(te,1),g_te.reshape(te,-1)],axis=1)
    return x_tr,f_tr,x_te,f_te

def get_fingerprints(atoms_list,make_fp=CartesianCoordFP):
    " Transform the ASE atoms into the fingerprints. "
    FPs=[]
    for i in range(len(atoms_list)):
        fp=make_fp(atoms_list[i])
        FPs.append(fp)    
    return FPs

def check_minima(sol,FPs,f_tr,gp,pdis=None,dstep=1e-3):
    " Check if thee solution is a minimum. "
    from hpfitter.objectivefunctions.likelihood import LogLikelihood
    from hpfitter.optimizers.optimizer import FunctionEvaluation
    from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
    # Construct optimizer
    hpfitter=HyperparameterFitterGPAtom(func=LogLikelihood(),optimizer=FunctionEvaluation(jac=False),get_prior_mean=False)
    # Get hyperparameter solution
    hp0=sol['hp'].copy()
    is_minima=True
    # Iterate over all hyperparameters
    for para,value in hp0.items():
        # Get function value of minimum 
        hp_test=hp0.copy()
        sol0=hpfitter.fit(FPs,f_tr,gp,hp=hp_test,pdis=pdis)
        # Get function value of a larger hyperparameter 
        hp_test[para]=value+dstep
        sol1=hpfitter.fit(FPs,f_tr,gp,hp=hp_test,pdis=pdis)
        # Get function value of a larger hyperparameter 
        hp_test[para]=value-dstep
        sol2=hpfitter.fit(FPs,f_tr,gp,hp=hp_test,pdis=pdis)
        # Check if it is a minimum
        if sol0['fun']>sol1['fun'] or sol0['fun']>sol2['fun']:
            is_minima=False
    return is_minima

def check_ratio_unchanged(gp,sol):
    " Check if the ratio is unchanged. "
    return gp.hp['ratio']==sol['full hp']['ratio']


class TestHpfitterGPatom(unittest.TestCase):
    """ Test if the Hyperparameter Fitter can be used for the Gaussian Process from GP-atom. """
    
    def test_line_factorization(self):
        " Line search with factorization method with optimization of scale, weight, and ratio. "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import GoldenSearch
        from hpfitter.objectivefunctions.factorized_likelihood import FactorizedLogLikelihood 
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=1e-3,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=80
        # Construct optimizer
        line_optimizer=GoldenSearch(optimize=True,multiple_min=True,parallel=False)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500)
        hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),optimizer=optimizer,get_prior_mean=False)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize
        hp=dict(ratio=1e-4,scale=0.5,weight=2.0)
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.509)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-58.105)<1.0)
        self.assertTrue(abs(sol['hp']['ratio']-0.220)<1e-2)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp)
        self.assertTrue(is_minima)
    

    def test_prior_mean(self):
        " Line search with factorization method with optimization of scale, weight, and ratio, but where the prior mean parameters are extracted. "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import GoldenSearch
        from hpfitter.objectivefunctions.factorized_likelihood import FactorizedLogLikelihood 
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=1e-3,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=80
        # Construct optimizer
        line_optimizer=GoldenSearch(optimize=True,multiple_min=True,parallel=False)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500)
        hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),optimizer=optimizer,get_prior_mean=True)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize
        hp=dict(ratio=1e-4,scale=0.5,weight=2.0)
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.509)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-58.105)<1.0)
        self.assertTrue(abs(sol['hp']['ratio']-0.220)<1e-2)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp)
        self.assertTrue(is_minima)

    def test_pdis(self):
        " Line search with factorization method with optimization of scale, weight, and ratio, but where the prior distributions are used for the hyperparameters. "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import GoldenSearch
        from hpfitter.objectivefunctions.factorized_likelihood import FactorizedLogLikelihood 
        from hpfitter.pdistributions.normal import Normal_prior
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=1e-3,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=80
        # Construct optimizer
        line_optimizer=GoldenSearch(optimize=True,multiple_min=True,parallel=False)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500)
        hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),optimizer=optimizer,get_prior_mean=False)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize
        hp=dict(ratio=1e-4,scale=0.5,weight=2.0)
        # Make the prior distributions of the hyperparameters
        pdis=dict(scale=Normal_prior(mu=0.0,std=1.0),ratio=Normal_prior(mu=-6.0,std=2.0))
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp,pdis=pdis)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-686.620)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.544)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-70.083)<1.0)
        self.assertTrue(abs(sol['hp']['ratio']-0.179)<1e-2)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp,pdis=pdis)
        self.assertTrue(is_minima)

    def test_bounds(self):
        " Line search with factorization method with optimization of scale, weight, and ratio. Different boundary conditions of the hyperparameters are tested. "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import GoldenSearch
        from hpfitter.objectivefunctions.factorized_likelihood import FactorizedLogLikelihood 
        from hpfitter.hpboundary.boundary import HPBoundaries
        from hpfitter.hpboundary.educated import EducatedBoundaries
        from hpfitter.hpboundary.hptrans import VariableTransformation
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=1e-3,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=80
        # Construct optimizer
        line_optimizer=GoldenSearch(optimize=True,multiple_min=True,parallel=False)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500)
        # Make boundary conditions
        bounds_list=[VariableTransformation(),
                     EducatedBoundaries(),
                     HPBoundaries(),
                     VariableTransformation(bounds=HPBoundaries()),
                     HPBoundaries(bounds_dict=dict(length=np.array([[-3.0,3.0]]),noise=np.array([[-12.0,1.0]]),prefactor=np.array([[-5.0,5.0]])))]
        for index,bounds in enumerate(bounds_list):
            with self.subTest(bounds=bounds):
                hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),
                                                    optimizer=optimizer,
                                                    get_prior_mean=False,
                                                    bounds=bounds)
                # Make fingerprints of training set
                FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
                # Define hyperparameters to optimize
                hp=dict(ratio=1e-4,scale=0.5,weight=2.0)
                # Optimize hyperparameters
                sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
                # Test the solution deviation
                self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
                self.assertTrue(abs(sol['hp']['scale']-0.509)<1e-2)
                self.assertTrue(abs(sol['hp']['weight']-58.105)<1.0)
                self.assertTrue(abs(sol['hp']['ratio']-0.220)<1e-2)
                # Check if it is a minimum
                is_minima=check_minima(sol,FPs,f_tr,gp)
                self.assertTrue(is_minima)

    def test_line_mle(self):
        " Line search with factorization method with optimization of scale and weight. "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import GoldenSearch
        from hpfitter.objectivefunctions.mle import MaximumLogLikelihood 
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Set the ratio as the same as the result before due to fix noise ratio
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=0.2203,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=80
        # Construct optimizer
        line_optimizer=GoldenSearch(optimize=True,multiple_min=True,parallel=False)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),optimizer=optimizer,get_prior_mean=False)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize (do not optimize ratio)
        hp=dict(scale=0.5,weight=2.0)
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.509)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-58.111)<1.0)
        # Check if the ratio is unchanged
        ratio_unchanged=check_ratio_unchanged(gp,sol)
        self.assertTrue(ratio_unchanged)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp)
        self.assertTrue(is_minima)

    def test_line_mle_finegrid(self):
        " Line search with a parallelizable method with factorization method with optimization of scale and weight (this is not parallelized). "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import FineGridSearch
        from hpfitter.objectivefunctions.mle import MaximumLogLikelihood 
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Set the ratio as the same as the result before due to fix noise ratio
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=0.2203,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=80
        # Construct optimizer
        line_optimizer=FineGridSearch(optimize=True,multiple_min=True,loops=5,ngrid=ngrid,parallel=False)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500,parallel=False)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),optimizer=optimizer,get_prior_mean=False)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize (do not optimize ratio)
        hp=dict(scale=0.5,weight=2.0)
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.508)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-58.061)<1.0)
        # Check if the ratio is unchanged
        ratio_unchanged=check_ratio_unchanged(gp,sol)
        self.assertTrue(ratio_unchanged)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp)
        self.assertTrue(is_minima)

    def test_line_mle_parallel(self):
        " Line search with a parallelizable method with factorization method with optimization of scale and weight (this is parallelized). "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import FineGridSearch
        from hpfitter.objectivefunctions.mle import MaximumLogLikelihood 
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Set the ratio as the same as the result before due to fix noise ratio
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=0.2203,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=80
        # Construct optimizer
        line_optimizer=FineGridSearch(optimize=True,multiple_min=True,loops=5,ngrid=ngrid,parallel=True)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500,parallel=True)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),optimizer=optimizer,get_prior_mean=False)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize (do not optimize ratio)
        hp=dict(scale=0.5,weight=2.0)
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.509)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-58.062)<1.0)
        # Check if the ratio is unchanged
        ratio_unchanged=check_ratio_unchanged(gp,sol)
        self.assertTrue(ratio_unchanged)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp)
        self.assertTrue(is_minima)

    def test_line_mle_fast(self):
        " Fastest line search with the parallelized method with factorization method with optimization of scale and weight (this is parallelized). "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.globaloptimizer import FactorizedOptimizer
        from hpfitter.optimizers.linesearcher import FineGridSearch
        from hpfitter.objectivefunctions.mle import MaximumLogLikelihood 
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Set the ratio as the same as the result before due to fix noise ratio
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=0.2203,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # The cpu number to use times the number of grid points pr cpu (default is 80) 
        ngrid=40
        # Construct optimizer
        line_optimizer=FineGridSearch(optimize=True,multiple_min=False,loops=5,ngrid=ngrid,parallel=True)
        optimizer=FactorizedOptimizer(line_optimizer=line_optimizer,ngrid=ngrid,maxiter=500,parallel=True)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),optimizer=optimizer,get_prior_mean=False)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize (do not optimize ratio)
        hp=dict(scale=0.5,weight=2.0)
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.509)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-58.062)<1.0)
        # Check if the ratio is unchanged
        ratio_unchanged=check_ratio_unchanged(gp,sol)
        self.assertTrue(ratio_unchanged)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp)
        self.assertTrue(is_minima)

    def test_local_mle(self):
        " Local optimization of scale and weight. "
        # Set a random state 
        np.random.seed(1)
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.localoptimizer import ScipyOptimizer
        from hpfitter.objectivefunctions.likelihood import LogLikelihood 
        # Need another prior mean for working
        from hpfitter.means.mean import Prior_mean 
        # Create the data set
        x,f,g=create_h2_atoms()
        ## Whether to learn from the derivatives
        use_derivatives=True
        x_tr,f_tr,x_te,f_te=make_train_test_set(x,f,g,tr=20,te=1,use_derivatives=use_derivatives)
        # Initial hyperparameters of the GP
        ## Set the ratio as the same as the result before due to fix noise ratio
        ## Noisefactor needs to be 1.0 for the Factorization method
        hp=dict(scale=1.0,weight=1.0,ratio=0.2203,noisefactor=1.0)
        # Construct gp (it needs another prior mean than default)
        gp=GaussianProcess(hp=hp,prior=Prior_mean(),kerneltype='sqexp',use_forces=True,parallelkernel=False)
        # Construct optimizer
        optimizer=ScipyOptimizer(maxiter=5000,jac=False,method='l-bfgs-b',use_bounds=False,tol=1e-12)
        # Construct the hyperparameter fitter
        hpfitter=HyperparameterFitterGPAtom(func=LogLikelihood(),optimizer=optimizer)
        # Make fingerprints of training set
        FPs=get_fingerprints(x_tr,make_fp=CartesianCoordFP)
        # Define hyperparameters to optimize (do not optimize ratio)
        hp=dict(scale=0.5,weight=2.0)
        # Optimize hyperparameters
        sol=hpfitter.fit(FPs,f_tr,gp,hp=hp)
        # Test the solution deviation
        self.assertTrue(abs(sol['fun']-681.487)<1e-2) 
        self.assertTrue(abs(sol['hp']['scale']-0.509)<1e-2)
        self.assertTrue(abs(sol['hp']['weight']-58.064)<1.0)
        # Check if the ratio is unchanged
        ratio_unchanged=check_ratio_unchanged(gp,sol)
        self.assertTrue(ratio_unchanged)
        # Check if it is a minimum
        is_minima=check_minima(sol,FPs,f_tr,gp)
        self.assertTrue(is_minima)


if __name__ == '__main__':
    unittest.main()

