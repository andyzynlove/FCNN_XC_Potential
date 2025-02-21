import numpy as np
import sh_fort
import os

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