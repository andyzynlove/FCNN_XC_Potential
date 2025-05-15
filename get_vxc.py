from pyscf import scf

def get_vxc(mol, dm, vc):
    mr = scf.RKS(mol)
    mr.xc = 'b3lypg'
    v0 = mr.get_veff(mol, dm)
    veff = v0 + vc
    J = scf.hf.get_jk(mol, dm)[0]
    vxc = veff - J
    return vxc