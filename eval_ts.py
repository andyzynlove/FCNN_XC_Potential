import numpy as np
from pyscf import dft
from opt_einsum import contract

def _eval_ts_on_grids(mol, coords, mo_coeff, mo_occ, na):
    n_samples = len(coords) // (na * na * na)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)[1:]
    ts = 0.5 * contract('xgp,pi,xgq,qi->g', ao, mo_coeff[:, mo_occ>0], ao, mo_coeff[:, mo_occ>0])
    ts = ts.reshape([n_samples, int(na*na*na)]).astype(np.float32)
    return ts