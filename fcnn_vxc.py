import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pyscf import gto, scf, dft
from opt_einsum import contract
import sh_fort
from tqdm import tqdm
einsum = np.einsum
BLK_SIZE = 1000

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.rho_type = "mGGA"
        self.ln = nn.LayerNorm(256, eps= 1e-5, elementwise_affine=True)
        self.fc1 = nn.Linear(39, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = 0.1*torch.log(x + 1e-7)
        x = F.tanh(self.fc1(x))
        x = F.elu(self.fc2(self.ln(x)))
        x = F.elu(self.fc3(self.ln(x)))
        x = F.elu(self.fc4(self.ln(x)))
        x = F.elu(self.fc5(self.ln(x)))
        x = F.elu(self.fc6(self.ln(x)))
        x = F.elu(self.fc7(self.ln(x)))
        x = self.fc8(self.ln(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ExtendModel(nn.Module):
    def __init__(self):
        super(ExtendModel, self).__init__()
        self.model = DNN()

    def load_model(self, load_path):
        self.model = nn.DataParallel(self.model)
        model_path = os.path.dirname(__file__) + '/fcnn_vxc'
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, x):
        x = self.model.forward(x)
        return x

# definition of vxc #
def _eval_vc_from_nn(model, rho, cuda):
    inputs = torch.from_numpy(rho)
    inputs = inputs.cuda()
    inputs = Variable(inputs)
    outputs = model(inputs)
    if cuda: outputs = outputs.cpu()
    vc = outputs.data.numpy().reshape(-1)    
    return vc

def _eval_rho_on_grids(mol, coords, dm, na):
    n_samples = len(coords) // (na * na * na)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)
    rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
    rho = rho.reshape([4, n_samples, int(na*na*na)]).astype(np.float32)
    return rho

def _eval_ts_on_grids(mol, coords, mo_coeff, mo_occ, na):
    n_samples = len(coords) // (na * na * na)
    ao = dft.numint.eval_ao(mol, coords, deriv=1)[1:]
    ts = 0.5 * contract('xgp,pi,xgq,qi->g', ao, mo_coeff[:, mo_occ>0], ao, mo_coeff[:, mo_occ>0])
    ts = ts.reshape([n_samples, int(na*na*na)]).astype(np.float32)
    return ts

def Generate_S(rho, rhox, rhoy, rhoz, nblk ,na3):
    grad_s = np.zeros([nblk, na3])
    grad_s = sh_fort.generate_s(rho, rhox, rhoy, rhoz, nblk, na3)
    return grad_s

def MCSH_density(rho, nblk, na):
    file_path = os.path.dirname(__file__) + '/mc.npy'
    SH_dicts = np.load(file_path, allow_pickle=True).item()
    r_list = [0.05, 0.10, 0.15]
    na3 =int(na*na*na)
    SHdescriptors = np.zeros([nblk, 13])
    for n in range(nblk):
        index_center = int((na3-1)/2)
        SHdescriptors[n,0] = rho[n, index_center]
        p = 1

    for r in r_list:
        n_cube_dim = int(round(r/0.05)*2 + 1)
        index_shift = int((na-1)/2 -(n_cube_dim-1)/2)
        stencils = SH_dicts["0_1"][str(r)][0]
        stentmp = np.array(stencils)
        stentmp = stentmp.reshape(-1)
        ncb3 = int(n_cube_dim*n_cube_dim*n_cube_dim)
        SHdescriptors[:,p] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        p = p + 1

        stencils = SH_dicts["1_1"][str(r)][0]
        stentmp = np.array(stencils[0])
        stentmp = stentmp.reshape(-1)
        SHtmp = np.zeros([3, nblk])
        SHtmp[0] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        stentmp = np.array(stencils[1])
        stentmp = stentmp.reshape(-1)
        SHtmp[1] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        stentmp = np.array(stencils[2])
        stentmp = stentmp.reshape(-1)
        SHtmp[2] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        SHdescriptors[:,p] = sh_fort.merge_shd(SHtmp[0], SHtmp[1], SHtmp[2], nblk)
        p = p + 1

        stencils = SH_dicts["2_1"][str(r)][0]
        stentmp = np.array(stencils[0])
        stentmp = stentmp.reshape(-1)
        SHtmp = np.zeros([3, nblk])
        SHtmp[0] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        stentmp = np.array(stencils[1])
        stentmp = stentmp.reshape(-1)
        SHtmp[1] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        stentmp = np.array(stencils[2])
        stentmp = stentmp.reshape(-1)
        SHtmp[2] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        SHdescriptors[:,p] = sh_fort.merge_shd(SHtmp[0], SHtmp[1], SHtmp[2], nblk)
        p = p + 1

        stencils = SH_dicts["2_2"][str(r)][0]
        stentmp = np.array(stencils[0])
        stentmp = stentmp.reshape(-1)
        SHtmp = np.zeros([3, nblk])
        SHtmp[0] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        stentmp = np.array(stencils[1])
        stentmp = stentmp.reshape(-1)
        SHtmp[1] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        stentmp = np.array(stencils[2])
        stentmp = stentmp.reshape(-1)
        SHtmp[2] = sh_fort.mcsh_r(rho, stentmp, n_cube_dim, na, index_shift, nblk, na3, ncb3)
        SHdescriptors[:,p] = sh_fort.merge_shd(SHtmp[0], SHtmp[1], SHtmp[2], nblk)
        p = p + 1

    return SHdescriptors

class Grids(object):
    def __init__(self, coords, weights):
        self._coords = coords
        self.original_coords = coords.copy() 
        self.weights = weights
        self.a = 0.566915 # Cube length
        self.na = 7       # Cube point  
        self.coords = self._coords.copy() 
        self._generate_offset()
        self._extend_coords()

    def _generate_offset(self):
        na3 = self.na * self.na * self.na
        offset = np.empty([na3, 3])
        dd = 1. / (self.na - 1)
        p = 0
        for i in range(self.na):
            for j in range(self.na):
                for k in range(self.na):
                    offset[p][0] = -0.5 + dd * i
                    offset[p][1] = -0.5 + dd * j
                    offset[p][2] = -0.5 + dd * k
                    p += 1
        self.offset = offset * self.a

    def _extend_coords(self):
        na = self.na
        na3 = na * na * na
        extended_coords = np.empty([len(self.coords)*na3, 3])
        p = 0
        for i, c in enumerate(self.coords):
            extended_coords[p:p+na3] = c + self.offset
            p += na3
        self.extended_coords = extended_coords

def eval_xc_on_grids(mol, dm, grids, c, mo_occ):
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

def eval_vc_matrix(mol, dm, grids, c, mo_occ):
    weights = grids.weights
    wc = eval_xc_on_grids(mol, dm, grids, c, mo_occ)
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
    return vc_matrix, wc

def eval_vxc_matrix(mol, dm, vc):
    nelec, exc_guide, vxc_guide = dft.numint.nr_vxc(mol, dft.gen_grid.Grids(mol), 'b3lypg', dm)
    vxc_tot = vxc_guide + vc
    return vxc_tot


# definition of molecule (Using HF molecule as example) #
atom="""H	+0.45   0	0	
        F	-0.45   0	0"""
basis = "aug-cc-pVQZ"
mol = gto.M(atom=atom, basis=basis, verbose=0)
mr = scf.RKS(mol)
mr.grids.prune = None
mr.grids.level = 2
mr.grids.build()

# generate grids #
grids = Grids(mr.grids.coords, mr.grids.weights)

# loading NN model parameters #
model = ExtendModel()
model.cuda()

# Get the XC potential matrix (Using 'minao--A superposition of atomic densities technique' as initial electron density) #
mf = scf.RHF(mol)
mf.init_guess = 'minao'
mf.kernel()
dm = mf.make_rdm1()
c = mf.mo_coeff
mo_occ = mf.mo_occ
vc, wc = eval_vc_matrix(mol, dm, grids, c, mo_occ)
vxc = eval_vxc_matrix(mol, dm, vc)