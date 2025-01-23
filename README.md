Parameters of FCNN-based XC potential and example codes to call them (.py).

PySCF Pytorch and tqdm packages are required to run the example codes.

Compile the sh.f90 file to get the spherical function.

  f2py -c --f90flags='-fopenmp' -lgomp -llapack sh.f90 -m sh_fort
