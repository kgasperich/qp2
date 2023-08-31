
#!/usr/bin/env python

import numpy as np
import scipy
from ezfio import ezfio
import itertools
import functools

@functools.lru_cache()
def doublefactorial(n):
    if n>=2:
        return n * doublefactorial(n-2)
    else:
        return 1

@functools.lru_cache()
def cartnorm(nxyz):
    return np.product([doublefactorial(2*i-1) for i in nxyz])

def cartnorm_str(s):
    return cartnorm(tuple(s.count(i) for i in 'xyz'))

@functools.lru_cache()
def gms_cart_order(l):
    def sort1(s):
        # return single xyz string in gamess order
        return ''.join(sorted((i*s.count(i) for i in 'xyz'), key=lambda x:(-len(x),x)))

    # unsorted, unordered xyz strings
    sxyz0 = map(''.join, itertools.combinations_with_replacement('xyz', r=l))

    # unsorted, ordered xyz strings
    sxyz1 = map(sort1, sxyz0)

    # sorted, ordered xyz strings
    return sorted(sxyz1, key=lambda s: (sorted((-s.count(i) for i in 'xyz')), s))

def pyscf_cart_order(l):
    return sorted(map(''.join, itertools.combinations_with_replacement('xyz', r=l)))

def pyscf_to_gms_order(l):
    xg = gms_cart_order(l)
    xp = pyscf_cart_order(l)
    m = np.zeros((len(xp),len(xg)))
    #p2g = [(xp.index(''.join(sorted(xi))),i) for i,xi in enumerate(xg)]
    for i,xi in enumerate(xg):
        m[i,xp.index(''.join(sorted(xi)))] = 1
    return m.T

def debug_pyscf_to_gms_order(l):
    xg = gms_cart_order(l)
    xp = pyscf_cart_order(l)
    print(xg)
    print(xp)
    m = np.zeros((len(xp),len(xg)))
    #p2g = [(xp.index(''.join(sorted(xi))),i) for i,xi in enumerate(xg)]
    for i,xi in enumerate(xg):
        m[i,xp.index(''.join(sorted(xi)))] = 1
    print(m)
    return m


def lm_cart_rescale(l):
    '''
    rescaling of certain functions within shells
    these arise due to different normalization factors for the different Cartesian functions
    a Cartesian GTO:
       x^(n_x) y^(n_y) z^(n_z) exp(-a r^2)
    is relatively normalized (i.e. relative to other function with the same shell)
    by a factor of:
       product((2*n_k-1)!! for n_k in (n_x, n_y, n_z))
    '''
    if l<=1:
        return np.eye(shsze(l,False))
    else:
        n0 = cartnorm_str('x'*l)
        return np.diag(np.sqrt([cartnorm_str(si)/n0 for si in gms_cart_order(l)])) * (2.0)**(-0.5*l)


# map xyz string to format of ezfio.ao_basis_ao_power
def xyznum(xyzstr):
    return [xyzstr.count(i) for i in 'xyz']
def aopow3(l):
    return list(map(xyznum,map(''.join, itertools.combinations_with_replacement('xyz',r=l))))

# number of AOs in cart/sph shell
def nao_l_cart(l):
    return ((l+1)*(l+2))//2
def nao_l_sph(l):
    return 2*l+1


def cart2sph_coeff(mol, normalized='sp', ct={}):
    from pyscf import gto
    '''Transformation matrix that transforms Cartesian GTOs to spherical
    GTOs for all basis functions

    Kwargs:
        normalized : string or boolean
            How the Cartesian GTOs are normalized.  Except s and p functions,
            Cartesian GTOs do not have the universal normalization coefficients
            for the different components of the same shell.  The value of this
            argument can be one of 'sp', 'all', None.  'sp' means the Cartesian s
            and p basis are normalized.  'all' means all Cartesian functions are
            normalized.  None means none of the Cartesian functions are normalized.
            The default value 'sp' is the convention used by libcint library.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvtz')
    >>> c = mol.cart2sph_coeff()
    >>> s0 = mol.intor('int1e_ovlp_sph')
    >>> s1 = c.T.dot(mol.intor('int1e_ovlp_cart')).dot(c)
    >>> print(abs(s1-s0).sum())
    >>> 4.58676826646e-15
    '''
    ld = { l : ct[l] if l in ct else None for l in range(12)}

    
    c2s_l = [gto.cart2sph(l, c_tensor = ld[l], normalized=normalized) for l in range(12)]
    c2s = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for n in range(mol.bas_nctr(ib)):
            c2s.append(c2s_l[l])
    return scipy.linalg.block_diag(*c2s)

##################################################################

def bas_from_mbas(mbas):
    """
    input: mbas, dict
    dict containing pyscf basis info

    output: bas, dict
    dict containing QP2 basis info (for basis module)
    """
    from collections import Counter
    outbas = {}

    # TODO: verify that QP2 needs bas funcs grouped by atom
    assert(sorted(mbas['atom']) == mbas['atom'])

    natom = max(mbas['atom'])+1


    outbas['sh_prim'] = sum([[i]*j for i,j in zip(mbas['nprim'],mbas['nctr'])],[])
    outbas['sh_num'] = sum(mbas['nctr'])
    outbas['sh_i'] = sum([[i+1]*j for i,j in enumerate(outbas['sh_prim'])],[])
    outbas['sh_l'] = sum([[i]*j for i,j in zip(mbas['l'],mbas['nctr'])],[])
    outbas['nucl_idx'] = sum([[i+1]*j for i,j in zip(mbas['atom'],mbas['nctr'])],[])
    nuc_sh_count = Counter(outbas['nucl_idx'])
    outbas['nucl_sh_num'] = [nuc_sh_count[i+1] for i in range(natom)]
    outbas['primnum'] = sum(outbas['sh_prim'])

    outbas['coef'] = []
    outbas['expo'] = []
    for i,(nprim,nctr) in enumerate(zip(mbas['nprim'],mbas['nctr'])):
        for ictr in range(nctr):
            outbas['coef'].extend(mbas['ctr_coef'][i][:,ictr].tolist())
            outbas['expo'].extend(mbas['exp'][i].tolist())


    return outbas

def aobas_from_mbas(mbas):
    """
    input: mbas, dict
    dict containing pyscf basis info

    output: aobas, dict
    dict containing QP2 ao_basis info (for ao_basis module)
    """
    from collections import Counter
    outbas = {}

    # TODO: verify that QP2 needs bas funcs grouped by atom
    assert(sorted(mbas['atom']) == mbas['atom'])
    primnum_max = max(mbas['nprim'])

    natom = max(mbas['atom'])+1

    sh_nl = sum([[nao_l_cart(l)] * nc for l,nc in zip(mbas['l'],mbas['nctr'])],[])
    outbas['num'] = sum(sh_nl)

    sh_prim = sum([[i]*j for i,j in zip(mbas['nprim'],mbas['nctr'])],[])
    outbas['primnum'] = sum([[i]*j for i,j in zip(sh_prim,sh_nl)],[])

    outbas['power'] = sum([aopow3(l)*nc for l,nc in zip(mbas['l'],mbas['nctr'])],[])
    outbas['nucl'] = sum([[i+1]*nc*nao_l_cart(l) for i,nc,l in zip(mbas['atom'],mbas['nctr'],mbas['l'])],[])

    outbas['expo'] = np.zeros((primnum_max, outbas['num']))
    outbas['coef'] = np.zeros((primnum_max, outbas['num']))


    istart=0
    for i,(nprim,nctr,l) in enumerate(zip(mbas['nprim'],mbas['nctr'],mbas['l'])):
        for ctr in range(nctr):
            outbas['coef'][:nprim,istart:istart+nao_l_cart(l)] = np.tile(mbas['ctr_coef'][i][:,ctr],(nao_l_cart(l),1)).T
            outbas['expo'][:nprim,istart:istart+nao_l_cart(l)] = np.tile(mbas['exp'][i],(nao_l_cart(l),1)).T
            istart += nao_l_cart(l)

    outbas['coef'] = outbas['coef'].tolist()
    outbas['expo'] = outbas['expo'].tolist()
    outbas['power'] = np.array(outbas['power'],dtype=int).T.tolist()
    
    return outbas

def mbas_from_mol(mol):
    """
    input: mol, pyscf Mol object

    output: mbas, dict
    dict containing pyscf basis info
    """
    mbas = {}
    mbas['nbas'] =     mol.nbas
    mbas['nprim'] =    [mol.bas_nprim(i)     for i in range(mbas['nbas'])]
    mbas['l'] =        [mol.bas_angular(i)   for i in range(mbas['nbas'])]
    mbas['ctr_coef'] = [mol.bas_ctr_coeff(i) for i in range(mbas['nbas'])]
    mbas['nctr'] =     [mol.bas_nctr(i)      for i in range(mbas['nbas'])]
    mbas['exp'] =      [mol.bas_exp(i)       for i in range(mbas['nbas'])]
    mbas['atom'] =     [mol.bas_atom(i)      for i in range(mbas['nbas'])]
    return mbas

def save_mol_to_ezfio(mol,ezpath):
    """
    input:
        mol, pyscf Mol object
        Mol object containing basis info to be written to ezfio

        ezpath, str
        path to ezfio directory where basis data will be written
    """
    mbas = mbas_from_mol(mol)
    aobas = aobas_from_mbas(mbas)
    bas = bas_from_mbas(mbas)



    basname = mol.basis if isinstance(mol.basis,str) else 'custom'

    ezfio.set_file(ezpath)
    
    natom = mol.natm
    
    ezfio.set_nuclei_nucl_num(natom)
    ezfio.set_nuclei_nucl_coord(mol.atom_coords('B').T.tolist())
    ezfio.set_nuclei_nucl_charge(mol.atom_charges().tolist())
    ezfio.set_nuclei_nucl_label([mol.atom_symbol(i) for i in range(natom)])


    ezfio.set_ao_basis_ao_num(aobas['num'])
    ezfio.set_ao_basis_ao_prim_num(aobas['primnum'])
    ezfio.set_ao_basis_ao_coef(aobas['coef'])
    ezfio.set_ao_basis_ao_expo(aobas['expo'])
    ezfio.set_ao_basis_ao_nucl(aobas['nucl'])
    ezfio.set_ao_basis_ao_power(aobas['power'])
    
    ezfio.set_ao_basis_ao_basis(basname)
    ezfio.set_ao_basis_ao_cartesian(False)
    ezfio.set_ao_basis_ao_normalized(True)
    ezfio.set_ao_basis_primitives_normalized(True)

    ezfio.set_basis_prim_num(bas['primnum'])
    ezfio.set_basis_nucleus_shell_num(bas['nucl_sh_num'])
    ezfio.set_basis_shell_num(bas['sh_num'])
    ezfio.set_basis_prim_coef(bas['coef'])
    ezfio.set_basis_prim_expo(bas['expo'])
    ezfio.set_basis_basis_nucleus_index(bas['nucl_idx'])
    ezfio.set_basis_shell_ang_mom(bas['sh_l'])
    ezfio.set_basis_shell_index(bas['sh_i'])
    ezfio.set_basis_shell_prim_num(bas['sh_prim'])

    ezfio.set_basis_typ('Gaussian')
    ezfio.set_basis_basis(basname)

    na,nb = mol.nelec
    ezfio.set_electrons_elec_alpha_num(na)
    ezfio.set_electrons_elec_beta_num(nb)


def get_c2s_norm_l(l):
    """
    cartesian function normalization factors in pyscf order
    """
    if l<=1:
        #return np.eye(2*l+1,dtype=float)
        return None
    cart_str = pyscf_cart_order(l)
    mfac = list(map(cartnorm_str, pyscf_cart_order(l)))
    return np.diag(np.sqrt(4*np.pi/doublefactorial(2*l+1) * np.array(mfac)))

def get_c2s_norm(lmax=9):
    """
    dict of cartesian norm factors for multiple l
    pass to `cart2sph_coeff` as arg ct
    """
    return {l:get_c2s_norm_l(l) for l in range(2,lmax)}

def save_mos_to_ezfio(mf,ezpath):
    """
    input:
        mol, pyscf Mol object
        Mol object containing basis info to be written to ezfio

        ezpath, str
        path to ezfio directory where basis data will be written
    """
    mbas = mbas_from_mol(mf.mol)
    aobas = aobas_from_mbas(mbas)
    bas = bas_from_mbas(mbas)

    ezfio.set_file(ezpath)

    _, nmo = mf.mo_coeff.shape

    if mf.mol.cart:
        coef_pyscf_cart = mf.mo_coeff
        xyz = [i[3] for i in mf.mol.ao_labels(fmt=False)]

        def fnorm(s):
            if len(s) <=1:
                return 1
            else:
                return np.sqrt(4*np.pi/doublefactorial(2*len(s)+1) * cartnorm_str(s))
        cartnorm = np.diag([fnorm(i) for i in xyz])
        coef_qp2_cart = cartnorm @ coef_pyscf_cart
    else:
        c2s = cart2sph_coeff(mf.mol,ct=get_c2s_norm())
        #s2c = np.linalg.inv(c2s.T @ c2s) @ c2s.T

        coef_pyscf_sph = mf.mo_coeff
        #coef_pyscf_cart = s2c.T @ coef_pyscf_sph
        coef_qp2_cart = c2s @ coef_pyscf_sph

    ezfio.set_mo_basis_mo_num(nmo)
    ezfio.set_mo_basis_mo_coef(coef_qp2_cart.T.tolist())
    return

def save_pyscf_to_ezfio(mf, ezpath):
    """
    save basis and mo coefs to ezfio
    """
    save_mol_to_ezfio(mf.mol, ezpath)
    save_mos_to_ezfio(mf, ezpath)
    return

