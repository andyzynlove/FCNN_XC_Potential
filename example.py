import numpy as np
from pyscf import gto, dft
from grids import *
from load_model import *
from get_vc import *
from get_vxc import *
from run_scf import *
from utility import *

einsum = np.einsum

# definition of molecule (using HF molecule as example) #
atom="""H    +0.45    0.0    0.0	
        F    -0.45    0.0    0.0"""
basis = "aug-cc-pVQZ"
mol = gto.M(atom=atom, basis=basis, verbose=0)

# definition of initial electron density (using B3LYP as example) #
mf = dft.RKS(mol)
mf.xc = 'b3lypg'
mf.kernel()
c_init = mf.mo_coeff
mo_occ_init = mf.mo_occ
dm_init = mf.make_rdm1()

# definition of grids #
grids = dft.gen_grid.Grids(mol)
grids.prune = None
grids.level = 2
grids.build()
grids = Grids(grids.coords, grids.weights)

# loading NN model parameters #
model = LoadModel()
model.cuda()

# SCF process #
scf_max_iter = 150

print('SCF Begin')
dm_nn, vc_nn, vc_real_space_nn, vxc_nn, mo_energy_nn, mo_coeff_nn, mo_occ_nn = scf_run(model, mol, dm_init, grids, c_init, mo_occ_init, scf_max_iter)
print('SCF End')
print('\n')

# CCSD reference density#
dm_ccsd = ccsd_r(mol)
print('Relaxed CCSD electron density has been calculated')
print('\n')

# I value #
I_b3lyp_ccsd = comput_rhodiff(mol, dm_init, dm_ccsd)
I_nn_ccsd = comput_rhodiff(mol, dm_nn, dm_ccsd)
print('I_value')
print('B3LYP/CCSD:      ', '{0:18.10e}'.format(I_b3lyp_ccsd), ';', 'NN/B3LYP:      ', '{0:18.10e}'.format(I_nn_ccsd))
print('\n')

# Dipole momnet #
dip_b3lyp = mf.dip_moment(dm = dm_init, verbose=0)
dip_ccsd = mf.dip_moment(dm = dm_ccsd, verbose=0)
dip_nn = mf.dip_moment(dm = dm_nn, verbose=0)
print('Dipole momnet (Debye)')
print('B3LYP:      ', '{0:.5f}'.format(dip_b3lyp[0]), ';', 'CCSD:      ', '{0:.5f}'.format(dip_ccsd[0]), ';', 'NN:      ', '{0:.5f}'.format(dip_nn[0]))
