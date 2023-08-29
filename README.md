A wrapper for the ObjectiveFuction class and HyperparameterFitter have been made, so the same optimizers and objective functions can be used for the gaussianprocess and the GP-atom. 

The wrapper for the ObjectiveFuction class is ObjectiveFuctionGPAtom. However, the objective functions need to inherit the ObjectiveFuctionGPAtom class instead of the ObjectiveFuction class.
The wrapper for the HyperparameterFitter class is HyperparameterFitterGPAtom.
Hard coded changes were required for the Educated_guess class (educated.py) due to missing functions in the kernel.
The prior mean from the GP-atom had to be changed, since it used the Cholesky matrix (L), which is not avavilable in the Factorization method. A constant prior mean is used here.
