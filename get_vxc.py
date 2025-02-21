from pyscf import dft

def get_vxc(mol, dm, vc):
    nelec, exc_guide, vxc_guide = dft.numint.nr_vxc(mol, dft.gen_grid.Grids(mol), 'b3lypg', dm)
    vxc = vxc_guide + vc
    return vxc