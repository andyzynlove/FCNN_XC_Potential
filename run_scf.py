from functools import reduce
from pyscf import scf
from get_vc import *
import numpy as np

class DIIS:
    def __init__(self, overlap_matrix, diis_space):
        eigenvalues, eigenvectors = np.linalg.eigh(overlap_matrix)
        sqrt_inv_eigenvalues = 1. / np.sqrt(eigenvalues)
        self.overlap_matrix = overlap_matrix
        self.ortho_matrix = reduce(np.dot, (eigenvectors, np.diag(sqrt_inv_eigenvalues), eigenvectors.T))
        self.diis_space = diis_space
        self.n_orb = len(overlap_matrix[0])
        self.error_matrices = np.zeros((self.diis_space, self.n_orb, self.n_orb))
        self.fock_matrices = np.zeros((self.diis_space, self.n_orb, self.n_orb))

    def extrapolate(self, iteration, fock, dm):
        if iteration <= 1 or self.diis_space < 2:
            return fock, 0.0
        for k in range(1, min(iteration, self.diis_space))[::-1]:
            self.error_matrices[k] = self.error_matrices[k-1]
            self.fock_matrices[k] = self.fock_matrices[k-1]
        error_matrix = reduce(np.dot, (fock, dm, self.overlap_matrix))
        error_matrix -= error_matrix.T
        self.error_matrices[0] = reduce(np.dot, (self.ortho_matrix.T, error_matrix, self.ortho_matrix))
        self.fock_matrices[0] = fock.copy()
        max_index = np.abs(self.error_matrices[0]).argmax()
        diis_error = np.abs(np.ravel(self.error_matrices[0])[max_index])
        n_prev = min(iteration, self.diis_space) - 1
        b_matrix = -1. * np.ones((n_prev+1, n_prev+1))
        b_matrix[n_prev, n_prev] = 0.
        b_matrix[:n_prev, :n_prev] = np.einsum('aij,bji->ab', self.error_matrices[:n_prev, :, :], self.error_matrices[:n_prev, :, :], optimize='greedy')
        a_vector = np.zeros(n_prev+1)
        a_vector[n_prev] = -1.
        c_vector = np.linalg.solve(b_matrix, a_vector)
        new_fock = np.zeros_like(fock)
        for i, coeff in enumerate(c_vector[:-1]):
            new_fock += coeff * self.fock_matrices[i]

        return new_fock, diis_error
    
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
