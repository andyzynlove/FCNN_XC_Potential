import numpy as np
from pyscf import gto, scf, dft, cc
from grids import *
from load_model import *
from get_vc import *
from get_vxc import *

einsum = np.einsum

# definition of molecule (using HF molecule as example) #
atom="""H    +0.45    0.0    0.0	
        F    -0.45    0.0    0.0"""
basis = "aug-cc-pVQZ"
mol = gto.M(atom=atom, basis=basis, verbose=0)

# definition of initial electron density (using CCSD as example) #
mf = scf.RHF(mol)
mf.kernel()
c = mf.mo_coeff
mo_occ = mf.mo_occ
mycc = cc.CCSD(mf)
ecc, t1, t2 = mycc.kernel()
dm_mo = mycc.make_rdm1()
dm_ao = einsum('pi,ij,qj->pq', c, dm_mo, c.conj())

# definition of grids #
grids = dft.gen_grid.Grids(mol)
grids.prune = None
grids.level = 2
grids.build()
grids = Grids(grids.coords, grids.weights)

# loading NN model parameters #
model = LoadModel()
model.cuda()

# get the exchange-corrlation potential matrix #
vc, vc_real_space = get_vc(model, mol, dm_ao, grids, c, mo_occ)
vxc = get_vxc(mol, dm_ao, vc)

print(vc_real_space)