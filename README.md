Parameters of FCNN-based XC potential and codes to call them (.py).

PySCF, Pytorch, tqdm and Optimized Einsum packages are required to run the example codes.

Compile the sh.f90 file to get the spherical function.

f2py -c --f90flags='-fopenmp' -lgomp -llapack sh.f90 -m sh_fort

mc.npy is the Maxwell-Cartesian spherical harmonic (MCSH) stencils.

example.py indicates how to run KS-DFT/FCNN procedure via our FCNN model. (In this case, we use hydrogen fluoride molecule as an example, and B3LYP electron density as initial guess.)
