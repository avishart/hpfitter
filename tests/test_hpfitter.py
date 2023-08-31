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

class TestHpfitterGPatom(unittest.TestCase):
    """ Test if the Hyperparameter Fitter can be used for the Gaussian Process from GP-atom. """

    def test_line_factorization(self):
        " Line search with factorization method with optimization of scale, weight, and ratio. "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import run_golden
        from hpfitter.optimizers.global_opt import line_search_scale
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
        opt_kwargs=dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,ngrid=ngrid,optimize=True,multiple_min=True)
        hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),add_noise_correction=True,
                                            optimization_method=line_search_scale,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_noise_correction(self):
        " Line search with factorization method with optimization of scale, weight, and ratio, but where the noise correction is not added. "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import run_golden
        from hpfitter.optimizers.global_opt import line_search_scale
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
        opt_kwargs=dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,ngrid=ngrid,optimize=True,multiple_min=True)
        hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),add_noise_correction=False,
                                            optimization_method=line_search_scale,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_prior_mean(self):
        " Line search with factorization method with optimization of scale, weight, and ratio, but where the prior mean parameters are extracted. "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import run_golden
        from hpfitter.optimizers.global_opt import line_search_scale
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
        opt_kwargs=dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,ngrid=ngrid,optimize=True,multiple_min=True)
        hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(get_prior_mean=True),add_noise_correction=False,
                                            optimization_method=line_search_scale,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_pdis(self):
        " Line search with factorization method with optimization of scale, weight, and ratio, but where the prior distributions are used for the hyperparameters. "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import run_golden
        from hpfitter.optimizers.global_opt import line_search_scale
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
        opt_kwargs=dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,ngrid=ngrid,optimize=True,multiple_min=True)
        hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),add_noise_correction=False,
                                            optimization_method=line_search_scale,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_bounds(self):
        " Line search with factorization method with optimization of scale, weight, and ratio. Different boundary conditions of the hyperparameters are tested. "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import run_golden
        from hpfitter.optimizers.global_opt import line_search_scale
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
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,ngrid=ngrid,optimize=True,multiple_min=True)
        test_opt_kwargs=[dict(bounds=None,hptrans=True,use_bounds=True),
                         dict(bounds=None,hptrans=False,use_bounds=True),
                         dict(bounds=None,hptrans=False,use_bounds=False),
                         dict(bounds=None,hptrans=True,use_bounds=False),
                         dict(bounds=np.array([[-3.0,3.0],[-12.0,1.0],[-5.0,5.0]]))]
        for index,opt_kwarg in enumerate(test_opt_kwargs):
            with self.subTest(opt_kwarg=opt_kwarg):
                hpfitter=HyperparameterFitterGPAtom(func=FactorizedLogLikelihood(),
                                                    optimization_method=line_search_scale,
                                                    opt_kwargs=dict(maxiter=500,jac=False,local_run=run_golden,**opt_kwarg,local_kwargs=local_kwargs))
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

    def test_line_mle(self):
        " Line search with factorization method with optimization of scale and weight. "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import run_golden
        from hpfitter.optimizers.global_opt import line_search_scale
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
        opt_kwargs=dict(local_run=run_golden,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,ngrid=ngrid,optimize=True,multiple_min=True)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),add_noise_correction=True,
                                            optimization_method=line_search_scale,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_line_mle_finegrid(self):
        " Line search with a parallelizable method with factorization method with optimization of scale and weight (this is not parallelized). "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import fine_grid_search
        from hpfitter.optimizers.global_opt import line_search_scale
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
        opt_kwargs=dict(local_run=fine_grid_search,ngrid=ngrid,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,loops=5,iterloop=ngrid,optimize=True,multiple_min=True)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),add_noise_correction=True,
                                            optimization_method=line_search_scale,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_line_mle_parallel(self):
        " Line search with a parallelizable method with factorization method with optimization of scale and weight (this is parallelized). "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import fine_grid_search
        from hpfitter.optimizers.mpi_global_opt import line_search_scale_parallel
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
        opt_kwargs=dict(local_run=fine_grid_search,ngrid=ngrid,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,loops=5,iterloop=ngrid,optimize=True,multiple_min=True)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),add_noise_correction=True,
                                            optimization_method=line_search_scale_parallel,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_line_mle_fast(self):
        " Fastest line search with the parallelized method with factorization method with optimization of scale and weight (this is parallelized). "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.functions import calculate_list_values
        from hpfitter.optimizers.local_opt import fine_grid_search
        from hpfitter.optimizers.mpi_global_opt import line_search_scale_parallel
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
        opt_kwargs=dict(local_run=fine_grid_search,ngrid=ngrid,bounds=None,hptrans=True,use_bounds=True)
        local_kwargs=dict(fun_list=calculate_list_values,tol=1e-5,loops=5,iterloop=ngrid,optimize=True,multiple_min=False)
        hpfitter=HyperparameterFitterGPAtom(func=MaximumLogLikelihood(),add_noise_correction=True,
                                            optimization_method=line_search_scale_parallel,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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

    def test_local_mle(self):
        " Local optimization of scale and weight. "
        # Import from gpatom
        from gpatom.gpfp.gp import GaussianProcess
        # Import the hpfitter
        from hpfitter.hpfitter_gpatom import HyperparameterFitterGPAtom
        from hpfitter.optimizers.local_opt import scipy_opt
        from hpfitter.optimizers.global_opt import local_optimize
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
        opt_kwargs=dict(local_run=scipy_opt)
        local_kwargs=dict(tol=1e-12,method='L-BFGS-B')
        hpfitter=HyperparameterFitterGPAtom(func=LogLikelihood(),add_noise_correction=True,
                                            optimization_method=local_optimize,
                                            opt_kwargs=dict(maxiter=500,jac=False,**opt_kwargs,local_kwargs=local_kwargs))
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


if __name__ == '__main__':
    unittest.main()

