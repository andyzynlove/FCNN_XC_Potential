import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from pyscf import dft
from eval_rho import _eval_rho_on_grids
from gen_SHdescriptor import MCSH_density, Generate_S
from eval_ts import _eval_ts_on_grids
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

einsum = np.einsum
BLK_SIZE = 1000

def _eval_vc_from_nn(model, rho, cuda):
    inputs = torch.from_numpy(rho)
    inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = model(inputs)
    if cuda: outputs = outputs.cpu()
    vc = outputs.data.numpy().reshape(-1)    
    return vc

def eval_vc_on_grids(model, mol, dm, grids, c, mo_occ):
    # Calculate real-space correction potential #
    coords = grids.coords
    extended_coords = grids.extended_coords
    na = 7
    na3 = na * na * na
    total_size = len(coords)
    assert(total_size * na3 == len(extended_coords))
    n_blk = total_size // BLK_SIZE
    res = total_size - BLK_SIZE * n_blk
    print('Evaluate xc potential on grids. Block size: %d Total: %d Number of blocks: %d Residual: %d' % 
            (BLK_SIZE, total_size, n_blk, res))
    wc = np.empty(total_size)
    with tqdm(total=total_size) as pbar:
        for i in range(n_blk):
            idx = slice(BLK_SIZE * i, BLK_SIZE * (i + 1))
            ext_idx = slice(BLK_SIZE * i * na3, BLK_SIZE * (i + 1) * na3)
            rho = _eval_rho_on_grids(mol, extended_coords[ext_idx], dm, na)
            SHdescriptors_rho = MCSH_density(rho[0], BLK_SIZE, na)
            grad_s = Generate_S(rho[0], rho[1], rho[2], rho[3], BLK_SIZE, na3)
            SHdescriptors_rho_grad = MCSH_density(grad_s, BLK_SIZE, na)
            ts = _eval_ts_on_grids(mol, extended_coords[ext_idx], c, mo_occ, na)
            SHdescriptors_kinetic_rho = MCSH_density(ts, BLK_SIZE, na)
            SHdescriptors = np.concatenate((SHdescriptors_rho, SHdescriptors_rho_grad, SHdescriptors_kinetic_rho), axis=1).astype(np.float32)
            wc[idx] = _eval_vc_from_nn(model, SHdescriptors, cuda='True')
            pbar.update(BLK_SIZE)
        if res > 0:
            rho = _eval_rho_on_grids(mol, extended_coords[-res*na3:], dm, na)
            SHdescriptors_rho = MCSH_density(rho[0], res, na)
            grad_s = Generate_S(rho[0], rho[1], rho[2], rho[3], res, na3)
            SHdescriptors_rho_grad = MCSH_density(grad_s, res, na)
            ts = _eval_ts_on_grids(mol, extended_coords[-res*na3:], c, mo_occ, na)
            SHdescriptors_kinetic_rho = MCSH_density(ts, res, na)
            SHdescriptors = np.concatenate((SHdescriptors_rho, SHdescriptors_rho_grad, SHdescriptors_kinetic_rho), axis=1).astype(np.float32)
            wc[-res:] = _eval_vc_from_nn(model, SHdescriptors, cuda='True')
            pbar.update(res)
    return wc

def get_vc(load_model, mol, dm, grids, c, mo_occ):
    weights = grids.weights
    wc = eval_vc_on_grids(load_model, mol, dm, grids, c, mo_occ)
    total_size = len(grids.original_coords)
    n_blk = total_size // BLK_SIZE
    res = total_size - BLK_SIZE * n_blk
    vc_matrix = np.zeros(dm.shape, dtype=dm.dtype) 
    ao_loc = mol.ao_loc_nr()
    shls_slice = (0, mol.nbas)
    for i in range(n_blk):
        idx = slice(BLK_SIZE * i, BLK_SIZE * (i + 1))
        ao = dft.numint.eval_ao(mol, grids.original_coords[idx], deriv=0)
        n_grids, n_ao = ao.shape
        wv = weights[idx] * wc[idx] * 0.5
        aow = einsum('pi,p->pi', ao, wv)
        vc_matrix += dft.numint._dot_ao_ao(mol, ao, aow, np.ones((n_grids, mol.nbas), dtype=np.int8), shls_slice, ao_loc) 
    if res > 0:
        ao = dft.numint.eval_ao(mol, grids.original_coords[-res:], deriv=0)
        n_grids, n_ao = ao.shape
        wv = weights[-res:] * wc[-res:] * 0.5
        aow = einsum('pi,p->pi', ao, wv)
        vc_matrix += dft.numint._dot_ao_ao(mol, ao, aow, np.ones((n_grids, mol.nbas), dtype=np.int8), shls_slice, ao_loc)   
    vc_matrix = vc_matrix + vc_matrix.T
    vc_matrix = -vc_matrix
    wc = -wc
    return vc_matrix, wc
