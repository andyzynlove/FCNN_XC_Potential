from functools import reduce
from pyscf import scf
from get_vc import *
import numpy as np

class DIIS:
    def __init__(self, S, diis_space):

        eig, Z = np.linalg.eigh(S)
        S12 = 1./np.sqrt(eig)
        self.S = S
        self.O = reduce(np.dot, (Z, np.diag(S12), Z.T))
        self.diis_space = diis_space
        self.norb = len(S[0])
        self.ems = np.zeros((self.diis_space, self.norb, self.norb))
        self.pms = np.zeros((self.diis_space, self.norb, self.norb))
        self.tall = self.t_1 = self.t_2 = self.t_3 = 0.

    def extrapolate(self, iteration, fock, dm):

        if iteration <= 1 or self.diis_space < 2:
            return fock, 0.0

        for k in range(1, min(iteration, self.diis_space))[::-1]:
            self.ems[k] = self.ems[k-1]
            self.pms[k] = self.pms[k-1]

        em = reduce(np.dot, (fock, dm, self.S))
        em -= em.T
        self.ems[0] = reduce(np.dot, (self.O.T, em, self.O))
        self.pms[0] = fock[:]
        idx = np.abs(self.ems[0]).argmax()
        diis_err = np.abs(np.ravel(self.ems[0])[idx])

        nb = min(iteration, self.diis_space)-1
        B = -1.*np.ones((nb+1, nb+1))
        B[nb, nb] = 0.
        B[:nb, :nb] = np.einsum('aij,bji->ab', self.ems[:nb, :, :], self.ems[:nb, :, :],optimize='greedy')
        A = np.zeros(nb+1)
        A[nb] = -1.
        C = np.linalg.solve(B, A)

        newfock = np.zeros_like(fock)
        for i, c in enumerate(C[:-1]):
            newfock += c*self.pms[i]

        return newfock, diis_err
    
def scf_run(model, mol, dm_init, grids, c_init, mo_occ_init, scf_max_iter):
    S = mol.intor('int1e_ovlp_sph')
    T = mol.intor('cint1e_kin_sph')
    vn = mol.intor('cint1e_nuc_sph')
    H = T + vn
    mf = scf.RHF(mol)
    mf.kernel()
    zdiis = DIIS(S, 40)
    dm = dm_init
    c = c_init
    mo_occ = mo_occ_init
    for cycle in range(1, scf_max_iter):
        dm_old = dm
        vc, vc_real_space = get_vc(model, mol, dm, grids, c, mo_occ)
        mr = scf.RKS(mol)
        mr.xc = 'b3lypg'
        V0 = mr.get_veff(mol, dm)
        Fock = H + V0 + vc
        Fock = scf.hf.level_shift(S, dm*.5, Fock, 20)
        Fock, diis_e = zdiis.extrapolate(cycle, Fock, dm)
        e, c = scf.hf.eig(Fock, S)
        mo_occ = mf.get_occ(e, c)
        dm = mf.make_rdm1(c, mo_occ)
        ddm = dm_old - dm
        dm_e = np.max(np.abs(ddm))
        dm_converged = dm_e < 1e-6
        diis_converged = diis_e < 1e-3
        print(cycle, dm_e, diis_e)
        e[mo_occ==0] -= 20
        converged = dm_converged and diis_converged
        if cycle  == scf_max_iter - 1:
            print('----------SCF Not Converged!----------')
            break
            
        elif converged and cycle > 1:
            print('------------SCF Completed!------------')
            break
        
    J = scf.hf.get_jk(mol, dm)[0]
    vxc = V0 + vc - J
    dm_nn = dm
    mo_energy = e
    mo_coeff = c
    mo_occ = mo_occ
    return dm_nn, vc, vc_real_space, vxc, mo_energy, mo_coeff, mo_occ