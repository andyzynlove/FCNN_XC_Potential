import numpy as np
from pyscf import scf, dft, cc
from relaxed_ccsd import *

def comput_rhodiff(mol, dm1, dm2):
    grids = dft.gen_grid.Grids(mol)
    grids.build()
    coords = grids.coords
    weights = grids.weights
    ao = dft.numint.eval_ao(mol, coords, deriv=0)
    rho_1 = dft.numint.eval_rho(mol, ao, dm1)
    rho_2 = dft.numint.eval_rho(mol, ao, dm2)
    weights = grids.weights
    diff = rho_2 - rho_1
    rho2_square = rho_2**2
    rho1_square = rho_1**2
    diffrho_square = diff**2
    f0 = np.sum(diffrho_square*weights)
    f2 = np.sum(rho2_square*weights)
    f1 = np.sum(rho1_square*weights)
    I = f0/(f1 + f2)
    return I

#Relaxed CCSD#
def ccsd_r(mol):
    mf_ccsd_r = scf.RHF(mol)
    mf_ccsd_r.kernel()
    mycc_r = cc.CCSD(mf_ccsd_r)
    mycc_r.kernel()
    dm_ccsd_1 = mycc_r.make_rdm1()
    dm_ccsd_r = cc_rrdm1(mycc_r)
    return dm_ccsd_r