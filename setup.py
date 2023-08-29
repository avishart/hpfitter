from setuptools import setup, find_packages
# Setting up
setup( 
    name="hpfitter", 
    version="1.0.0",
    author="Andreas Vishart",
    author_email="<alyvi@dtu.dk>",
    description="Hyperparameter fitter for the Gaussian Process from GP-atom",
    long_description="Hyperparameter fitter for the Gaussian Process from GP-atom",
    packages=find_packages(),
    install_requires=['numpy>=1.20.3','scipy>=1.8.0','ase>=3.22.1','ase-gpatom'], 
    extras_require={'optional':['mpi4py>=3.0.3']},
    test_suite='tests',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.7'
)
