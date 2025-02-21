import numpy as np
from pyscf import dft

def _eval_rho_on_grids(mol, coords, dm, na):
    n_samples = len(coords) // (na * na * na)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
    rho = rho.reshape([4, n_samples, int(na*na*na)]).astype(np.float32)
    return rho