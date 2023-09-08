#!/usr/bin/env python

import numpy as np
import scipy
from ezfio import ezfio_obj
import itertools
import functools

from collections import defaultdict


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

    ezf = ezfio_obj()

    ezf.set_file(ezpath)
    
    natom = mol.natm
    
    ezf.set_nuclei_nucl_num(natom)
    ezf.set_nuclei_nucl_coord(mol.atom_coords('B').T.tolist())
    ezf.set_nuclei_nucl_charge(mol.atom_charges().tolist())
    ezf.set_nuclei_nucl_label([mol.atom_symbol(i) for i in range(natom)])


    ezf.set_ao_basis_ao_num(aobas['num'])
    ezf.set_ao_basis_ao_prim_num(aobas['primnum'])
    ezf.set_ao_basis_ao_coef(aobas['coef'])
    ezf.set_ao_basis_ao_expo(aobas['expo'])
    ezf.set_ao_basis_ao_nucl(aobas['nucl'])
    ezf.set_ao_basis_ao_power(aobas['power'])
    
    ezf.set_ao_basis_ao_basis(basname)
    ezf.set_ao_basis_ao_cartesian(False)
    ezf.set_ao_basis_ao_normalized(True)
    ezf.set_ao_basis_primitives_normalized(True)

    ezf.set_basis_prim_num(bas['primnum'])
    ezf.set_basis_nucleus_shell_num(bas['nucl_sh_num'])
    ezf.set_basis_shell_num(bas['sh_num'])
    ezf.set_basis_prim_coef(bas['coef'])
    ezf.set_basis_prim_expo(bas['expo'])
    ezf.set_basis_basis_nucleus_index(bas['nucl_idx'])
    ezf.set_basis_shell_ang_mom(bas['sh_l'])
    ezf.set_basis_shell_index(bas['sh_i'])
    ezf.set_basis_shell_prim_num(bas['sh_prim'])

    ezf.set_basis_typ('Gaussian')
    ezf.set_basis_basis(basname)

    na,nb = mol.nelec
    ezf.set_electrons_elec_alpha_num(na)
    ezf.set_electrons_elec_beta_num(nb)


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

    ezf = ezfio_obj()
    ezf.set_file(ezpath)

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

    ezf.set_mo_basis_mo_num(nmo)
    ezf.set_mo_basis_mo_coef(coef_qp2_cart.T.tolist())
    return

def save_pyscf_to_ezfio(mf, ezpath):
    """
    save basis and mo coefs to ezfio
    """
    save_mol_to_ezfio(mf.mol, ezpath)
    save_mos_to_ezfio(mf, ezpath)
    return

##################################################################

def pyint_to_u64(i,n64):
    """
    non-negative python int (arbitrary size) to np array of uint64
    """
    b=i.to_bytes(n64*8,'little')
    return np.frombuffer(b,dtype='uint64')

def make_u64_psidet(inp):
   if isinstance(inp, np.ndarray) and x.dtype in [np.uint64, np.int64]:
       # if already a np array, no work to do
       return inp.view('uint64').copy()
   else:
       try:
           # directly from qp2 should already be int64
           return np.array(inp,dtype='int64').view('uint64').copy()
       except OverflowError:
           pass
       # ensure non-negative (should be either uint64 or arbitrary-sized int)
       assert(np.min(inp) >= 0)
       try:
           return np.array(inp,dtype='uint64')
       except OverflowError:
           # each spindet should be single arbitrary-sized int
           # dims of inp should be [ndet, nspin]
           ndet, nspin = np.shape(inp)
           maxval = np.max(inp)
           nbit = maxval.bit_length()
           n64 = ((nbit-1)//64)+1
           assert(n64>1)

           return np.array([pyint_to_u64(i,n64) for i in np.ravel(inp)],dtype='uint64').reshape(ndet,nspin,n64)

def u64_1d_to_int(a):
    """
    a: 1d np array of uint64
    """
    return int.from_bytes(a.tobytes(),byteorder='little')

def u64_array_to_int(a):
    """
    a: array of uint64 with dims [ndet, nspin, n64]
    """
    return np.array([[ int.from_bytes(sdet.tobytes(),byteorder='little') for sdet in det] for det in a],dtype=object)


class PsiDet:
    def __init__(self,inp):
        self.psidet_u64 = inp

    @property
    def psidet_u64(self):
        return self._psidet_u64

    @psidet_u64.setter
    def psidet_u64(self, inp):
        self._psidet_u64 = make_u64_psidet(inp)


    def to_i64(self):
        return self._psidet_u64.view('int64')

    def to_int(self):
        return u64_array_to_int(self._psidet_u64)

    def to_bits(self):
        ndet,nspin,nint = self._psidet_u64.shape
        return np.unpackbits(self._psidet_u64.view('uint8'),bitorder='little').reshape(ndet,nspin,-1)

    def unique_alpha(self):
        # return set([tuple(i) for i in self._psidet_u64[:,0,:]])
        return sorted(set([tuple(i) for i in self._psidet_u64[:,0,:]]), key=lambda x: tuple(reversed(x)))

    def unique_beta(self):
        # return set([tuple(i) for i in self._psidet_u64[:,1,:]])
        return sorted(set([tuple(i) for i in self._psidet_u64[:,1,:]]), key=lambda x: tuple(reversed(x)))
    
    def make_bilinear(self):
        self.ndet = len(self.psidet_u64)
        self.sorted_a_unique = self.unique_alpha()
        self.sorted_b_unique = self.unique_beta()
        self.n_alpha_unique = len(self.sorted_a_unique)
        self.n_beta_unique  = len(self.sorted_b_unique)
        self.a_unique_map = {d: i for i,d in enumerate(self.sorted_a_unique)}
        self.b_unique_map = {d: i for i,d in enumerate(self.sorted_b_unique)}
        
        tmp_rows = np.zeros(self.ndet,dtype=np.int64) - 1
        tmp_cols = np.zeros(self.ndet,dtype=np.int64) - 1
        to_sort = np.zeros(self.ndet,dtype=np.int64) - 1
        
        
        for k,dab in enumerate(self.psidet_u64):
            i = self.a_unique_map[tuple(dab[0])]
            j = self.b_unique_map[tuple(dab[1])]
            tmp_rows[k] = i
            tmp_cols[k] = j
            to_sort[k] = self.n_alpha_unique * j + i
        
        self.bilinear_order = np.argsort(to_sort)
        self.bilinear_rows = tmp_rows[self.bilinear_order]
        self.bilinear_cols = tmp_cols[self.bilinear_order]
        self.bilinear_order_reverse = np.argsort(self.bilinear_order)
        
        
        self.bilinear_cols_loc = np.zeros(self.n_beta_unique + 1, dtype=np.int64) - 1
        l = self.bilinear_cols[0]
        self.bilinear_cols_loc[l] = 0
        for k in range(1,self.ndet):
            if self.bilinear_cols[k] == self.bilinear_cols[k-1]:
                continue
            else:
                l = self.bilinear_cols[k]
                self.bilinear_cols_loc[l] = k
            if self.bilinear_cols[k] < 0:
                raise
        self.bilinear_cols_loc[self.n_beta_unique] = self.ndet
        
        # tmp_rows = np.zeros(self.ndet,dtype=np.int64) - 1
        # tmp_cols = np.zeros(self.ndet,dtype=np.int64) - 1
        # to_sort = np.zeros(self.ndet,dtype=np.int64) - 1
        
        # for k,dab in enumerate(self.psidet_u64):
        #     tmp_cols[k] = self.bilinear_cols[k]
        #     tmp_rows[k] = self.bilinear_rows[k]
        #     i = tmp_cols[k]
        #     j = tmp_rows[k]
        #     to_sort[k] = self.n_beta_unique * j + i
            
        # self.bilinear_transp_order = np.argsort(to_sort)
        # self.bilinear_transp_rows = tmp_rows[self.bilinear_transp_order]
        # self.bilinear_transp_cols = tmp_cols[self.bilinear_transp_order]
        # self.bilinear_transp_order_reverse = np.argsort(self.bilinear_transp_order)
        
        # tmp_rows = np.zeros(self.ndet,dtype=np.int64) - 1
        # tmp_cols = np.zeros(self.ndet,dtype=np.int64) - 1
        # to_sort = np.zeros(self.ndet,dtype=np.int64) - 1
        
        # for k,dab in enumerate(self.psidet_u64):
            
        #     i = self.b_unique_map[tuple(dab[1])]
        #     j = self.a_unique_map[tuple(dab[0])]
        #     tmp_rows[k] = i
        #     tmp_cols[k] = j
        #     to_sort[k] = self.n_beta_unique * j + i
            
        # self.bilinear_transp_order = np.argsort(to_sort)
        # self.bilinear_transp_rows = tmp_rows[self.bilinear_transp_order]
        # self.bilinear_transp_cols = tmp_cols[self.bilinear_transp_order]
        # self.bilinear_transp_order_reverse = np.argsort(self.bilinear_transp_order)
        
        tmp_rows = np.zeros(self.ndet,dtype=np.int64) - 1
        tmp_cols = np.zeros(self.ndet,dtype=np.int64) - 1
        to_sort = np.zeros(self.ndet,dtype=np.int64) - 1
        
        for k,dab in enumerate(self.psidet_u64):
            
            i = self.b_unique_map[tuple(dab[1])]
            j = self.a_unique_map[tuple(dab[0])]
            tmp_rows[k] = j
            tmp_cols[k] = i
            to_sort[k] = self.n_beta_unique * j + i
            
        self.bilinear_transp_order = np.argsort(to_sort)
        self.bilinear_transp_rows = tmp_rows[self.bilinear_transp_order]
        self.bilinear_transp_cols = tmp_cols[self.bilinear_transp_order]
        self.bilinear_transp_order_reverse = np.argsort(self.bilinear_transp_order)
        
        
        self.bilinear_transp_rows_loc = np.zeros(self.n_alpha_unique + 1, dtype=np.int64) - 1
        l = self.bilinear_transp_rows[0]
        self.bilinear_transp_rows_loc[l] = 0
        for k in range(1,self.ndet):
            if self.bilinear_transp_rows[k] == self.bilinear_transp_rows[k-1]:
                continue
            else:
                l = self.bilinear_transp_rows[k]
                self.bilinear_transp_rows_loc[l] = k

        self.bilinear_transp_rows_loc[self.n_alpha_unique] = self.ndet
        
        
        
        
        
        



def long_int_to_uint64(i):
    res = []
    while i:
        res.append(i%(1<<64))
        i >>= 64
    return res


def u64int(i):
    i = int(i)
    return np.uint64(i if i >= 0 else 2**64 + i)

def int_to_qpdet(sdet, nint):
    return [(np.int64(np.uint64((sdet >> (64*i)) % (1<<64)))) for i in range(nint)]

def ints_to_qpdet(detab, nint):
    return [int_to_qpdet(sdet, nint) for sdet in detab]

def det_to_bits(detsab):
    """
    input: detsab, array with shape (ndet, nspin, nint)
    """
    psidet = np.array(detsab,dtype=np.uint64)
    ndet, nspin, nint = psidet.shape
    psibits = np.unpackbits(psidet.view('uint8'),bitorder='little').reshape(ndet,nspin,-1)
    return psibits

def qp2_det_to_uint64(detab):
    """
    input: detab, pair of lists of int
    [[a1,a2,...,aN],[b1,b2,...,bN]]
    """
    return list(map(lambda x: list(map(u64int, x)), detab))

def qp2_det_to_uint64(detab):
    return np.array(detab,dtype=np.int64).view('uint64')

def spindet_to_pyint(sdet_list):
    """
    input: sdet, list of uint64
    [a1,a2,...,aN]

    convert from 64-bit chunks of orbs to arbitrary-length int
    input is not validated.  Should already be uint64

    >>> spindet_to_pyint([1])
    1
    >>> spindet_to_pyint([1,1])
    18446744073709551617
    >>> spindet_to_pyint([1,2])
    36893488147419103233
    >>> spindet_to_pyint([2,1])
    18446744073709551618
    >>> spindet_to_pyint([1,1,1])
    340282366920938463481821351505477763073

    """

    sdet = 0
    for i, ai in enumerate(sdet_list):
        sdet += int(ai) << (64 * i)
    return sdet


def det_to_pyint(detab):
    """
    input: detab, pair of lists of uint64
    [[a1,a2,...,aN],[b1,b2,...,bN]]
    """

    return list(map(spindet_to_pyint, detab))


def qp2_det_to_pyint(detab):
    return det_to_pyint(qp2_det_to_uint64(detab))



def get_n64_from_norb(norb):
    return ((norb-1)//64)+1


def get_psi(ezpath):
    ezf = ezfio_obj()
    ezf.set_file(ezpath)

    c0 = ezf.get_determinants_psi_coef()
    d0 = ezf.get_determinants_psi_det()
    d1 = PsiDet(d0)

    return c0,d1

def get_hpmask_u64(d1,d2):
    """
    d1,d2 each [nspin, nint]
    """
    return np.bitwise_xor(d1,d2)
    #return np.unpackbits(np.bitwise_xor(d1,d2),bitorder='little').reshape(2,-1)

def get_is_single(d1,d2):
    return np.sum(i.bit_count() for i in np.bitwise_xor(d1,d2).ravel())


def make_tdm(psi1,psi2,norb=None):

    c1,pd1 = psi1
    c2,pd2 = psi2

    d1 = pd1.psidet_u64
    d2 = pd2.psidet_u64

    nd1, nspin,  nint  = d1.shape
    nd2, nspin2, nint2 = d2.shape
    assert(nspin==nspin2)
    assert(nint==nint2)
    if norb==None:
        norb = 64 * nint

    tdm1 = np.zeros((2,norb,norb),dtype=float)

    #for (ci,di),(cj,dj) in itertools.product(zip(c1[0],d1),zip(c2[0],d2)):
    for (ci,di) in zip(c1[0],d1):
        for (cj,dj) in zip(c2[0],d2):
            hpu64 = get_hpmask_u64(di,dj)
            if np.unpackbits(hpu64.view('uint8'),bitorder='little').sum() == 2:
                h_u64 = np.bitwise_and(hpu64,di)
                p_u64 = np.bitwise_and(hpu64,dj)
                hspin, hidx = np.argwhere(np.unpackbits(h_u64.view('uint8'),bitorder='little').reshape(2,-1)).ravel()
                pspin, pidx = np.argwhere(np.unpackbits(p_u64.view('uint8'),bitorder='little').reshape(2,-1)).ravel()
                assert(hspin == pspin)

                phase = 1  #TODO: compute phase
                tdm1[hspin, hidx, pidx] += phase * ci * cj
    return tdm1

def make_tdm_bits(psi1,psi2,norb=None):

    c1,pd1 = psi1
    c2,pd2 = psi2

    d1 = pd1.to_bits()
    d2 = pd2.to_bits()

    nd1, nspin,  nbit  = d1.shape
    nd2, nspin2, nbit2 = d2.shape
    assert(nspin==nspin2)
    assert(nbit==nbit2)
    if norb==None:
        norb = nbit

    tdm1 = np.zeros((2,norb,norb),dtype=float)

    #for (ci,di),(cj,dj) in itertools.product(zip(c1[0],d1),zip(c2[0],d2)):
    for (ci,di) in zip(c1[0],d1):
        for (cj,dj) in zip(c2[0],d2):
            hpbits = np.logical_xor(di,dj)
            if np.count_nonzero(hpbits) == 2:
                h_b = np.logical_and(hpbits,di)
                p_b = np.logical_and(hpbits,dj)
                hspin, hidx = np.argwhere(h_b).ravel()
                pspin, pidx = np.argwhere(p_b).ravel()
                assert(hspin == pspin)

                phase = 1  #TODO: compute phase
                tdm1[hspin, hidx, pidx] += phase * ci * cj
    return tdm1

def make_tdm_sorted(psi1,psi2,norb=None):

    c1,pd1 = psi1
    c2,pd2 = psi2
    pd1.make_bilinear()
    pd2.make_bilinear()

    d1 = pd1.to_bits()
    d2 = pd2.to_bits()

    nd1, nspin,  nbit  = d1.shape
    nd2, nspin2, nbit2 = d2.shape
    assert(nspin==nspin2)
    assert(nbit==nbit2)
    if norb==None:
        norb = nbit

    tdm1 = np.zeros((2,norb,norb),dtype=float)

    # alpha singles
    for ib1,b1 in enumerate(pd1.sorted_b_unique):
        ib2 = pd2.b_unique_map.get(b1,-1)
        if ib2<0:
            continue #b1 not in wf2
        itot1_0 = pd1.bilinear_cols_loc[ib1]
        itot1_1 = pd1.bilinear_cols_loc[ib1+1]
        itot2_0 = pd2.bilinear_cols_loc[ib2]
        itot2_1 = pd2.bilinear_cols_loc[ib2+1]
        
        for itot1 in range(itot1_0,itot1_1):
            a1 = pd1.sorted_a_unique[pd1.bilinear_rows[itot1]]
            for itot2 in range(itot2_0,itot2_1):
                a2 = pd2.sorted_a_unique[pd2.bilinear_rows[itot2]]
                hpmask_u64 = np.bitwise_xor(a1,a2)
                hpbits = np.unpackbits(hpmask_u64.view('uint8'),bitorder='little')
                if hpbits.sum() == 2:
                    hmask_u64 = np.bitwise_and(a1,hpmask_u64)
                    pmask_u64 = np.bitwise_and(a2,hpmask_u64)
                    hbits = np.unpackbits(hmask_u64.view('uint8'),bitorder='little')
                    pbits = np.unpackbits(pmask_u64.view('uint8'),bitorder='little')
                    hidx = np.argwhere(hbits)[0,0]
                    pidx = np.argwhere(pbits)[0,0]
                    tdm1[0,hidx,pidx] += c1[0][pd1.bilinear_order[itot1]] * c2[0][pd2.bilinear_order[itot2]]
    # beta singles
    for ia1,a1 in enumerate(pd1.sorted_a_unique):
        ia2 = pd2.a_unique_map.get(a1,-1)
        if ia2<0:
            continue #a1 not in wf2
        itot1_0 = pd1.bilinear_transp_rows_loc[ia1]
        itot1_1 = pd1.bilinear_transp_rows_loc[ia1+1]
        itot2_0 = pd2.bilinear_transp_rows_loc[ia2]
        itot2_1 = pd2.bilinear_transp_rows_loc[ia2+1]
        
        for itot1 in range(itot1_0,itot1_1):
            b1 = pd1.sorted_b_unique[pd1.bilinear_transp_cols[itot1]]
            for itot2 in range(itot2_0,itot2_1):
                b2 = pd2.sorted_b_unique[pd2.bilinear_transp_cols[itot2]]
                hpmask_u64 = np.bitwise_xor(b1,b2)
                hpbits = np.unpackbits(hpmask_u64.view('uint8'),bitorder='little')
                if hpbits.sum() == 2:
                    hmask_u64 = np.bitwise_and(b1,hpmask_u64)
                    pmask_u64 = np.bitwise_and(b2,hpmask_u64)
                    hbits = np.unpackbits(hmask_u64.view('uint8'),bitorder='little')
                    pbits = np.unpackbits(pmask_u64.view('uint8'),bitorder='little')
                    hidx = np.argwhere(hbits)[0,0]
                    pidx = np.argwhere(pbits)[0,0]
                    tdm1[1,hidx,pidx] += c1[0][pd1.bilinear_transp_order[itot1]] * c2[0][pd2.bilinear_transp_order[itot2]]
                    
    return tdm1


def get_hfdet(nmo,nab):
    na,nb = nab
    nint = get_n64_from_norb(nmo)
    det8 = np.zeros((1,2,nint*64),dtype=np.uint8)
    for i,ni in enumerate((na,nb)):
        det8[:,i,:ni] += 1

    det = np.packbits(det8,bitorder='little').view('int64')


    return det.reshape(1,2,-1)

def get_hfdet_from_ezfio(ezpath):
    ezf = ezfio_obj()
    ezf.set_file(ezpath)
    nmo = ezf.get_mo_basis_mo_num()
    na = ezf.get_electrons_elec_alpha_num()
    nb = ezf.get_electrons_elec_beta_num()

    return get_hfdet(nmo,(na,nb))


def make_det(nmo,aocc,bocc):
    nint = get_n64_from_norb(nmo)
    det8 = np.zeros((1,2,nint*64),dtype=np.uint8)
    for i,iocc in enumerate((aocc,bocc)):
        for iorb in iocc:
            det8[0,i,iorb] = 1

    det = np.packbits(det8,bitorder='little').view('int64')


    return det.reshape(1,2,-1)


def apply_hp(det0,hplist):
    """
    hplist: [(idx_i,spin_i, is_part_i), ...] for each particle/hole
        is_part: False/True for hole/particle
    """
    ndet,nspin,nint = det0.shape
    assert(ndet==1)
    assert(nspin==2)
    nmo = nint*64

    abh = [[],[]]
    abp = [[],[]]

    for idx,spin,is_part in hplist:
        if is_part:
            abp[spin].append(idx)
        else:
            abh[spin].append(idx)

    pmask = make_det(nmo, abp[0], abp[1])
    hmask = make_det(nmo, abh[0], abh[1])

    assert(np.all(np.bitwise_and(hmask,det0) == hmask))
    assert(np.all(np.bitwise_and(np.bitwise_not(det0),pmask) == pmask))
    hpmask = np.bitwise_or(hmask,pmask)

    return np.bitwise_xor(hpmask,det0)



def set_1det_exc(ezpath,hplist):
    """
    hplist: [(idx_i,spin_i, is_part_i), ...] for each particle/hole
    is_part: False/True for hole/particle
    """
    hfdet = get_hfdet_from_ezfio(ezpath)
    newdet = apply_hp(hfdet,hplist)

    ezf = ezfio_obj()
    ezf.set_file(ezpath)
    ezf.set_determinants_n_det(1)
    ezf.set_determinants_psi_det(newdet.tolist())
    ezf.set_determinants_psi_coef([[1]])
    ezf.set_determinants_n_det_qp_edit(1)
    ezf.set_determinants_psi_det_qp_edit(newdet.tolist())
    ezf.set_determinants_psi_coef_qp_edit([[1]])
    ezf.set_determinants_read_wf(True)

    return


def save_1det_to_ezfio(mf, ezpath, hplist = None, n_det_max = 10000):
    nao, nmo = mf.mo_coeff.shape
    N_int = ((nmo-1) // 64) + 1

    hfdet = get_hfdet(nmo, mf.mol.nelec)

    save_pyscf_to_ezfio(mf, ezpath)

    ezf = ezfio_obj()
    ezf.set_file(ezpath)
    ezf.set_determinants_bit_kind(8)
    ezf.set_determinants_n_int(N_int)
    ezf.set_determinants_n_states(1)
    ezf.set_determinants_n_det_max(n_det_max)
    ezf.set_determinants_selection_factor(0.2)
    if hplist==None:
        qpdet = ints_to_qpdet(hfdet, N_int)
    else:
        qpdet = apply_hp(hfdet, hplist)

    ezf.set_determinants_mo_label('None')
    ezf.set_determinants_n_det(1)
    ezf.set_determinants_n_det_qp_edit(1)
    ezf.set_determinants_psi_det(qpdet)
    ezf.set_determinants_psi_det_qp_edit(qpdet)
    ezf.set_determinants_psi_coef([[1]])
    ezf.set_determinants_psi_coef_qp_edit([[1]])
    ezf.set_determinants_read_wf(True)

    return

def set_ormas_ezfio(ezpath, ormas_info):
    nspace, min_e, max_e, mstart = ormas_info

    ezf = ezfio_obj()
    ezf.set_file(ezpath)
    ezf.set_bitmask_do_ormas(True)
    ezf.set_bitmask_ormas_n_space(nspace)
    ezf.set_bitmask_ormas_min_e(min_e)
    ezf.set_bitmask_ormas_max_e(max_e)
    ezf.set_bitmask_ormas_mstart(mstart)
    return
    

def gen_core_ormas_atom(mf,hsym,psym):
    from pyscf.symm.basis import _SO3_ID2SYMB
    orblist = [(i,occ,symid,_SO3_ID2SYMB[symid]) for i,(occ,symid) in enumerate(zip(mf.mo_occ,mf.orbsym))]
    virtlist = [ i for i in orblist if i[1]==0]
    symb_to_orbidx = defaultdict(list)
    for i,occ,_,symb in orblist:
        symb_to_orbidx[symb,round(occ)].append(i)
    #print(symb_to_orbidx)
    #print(psym)
    hole_idx = min(symb_to_orbidx[hsym,2]) # lowest doubly-occ orb of this sym
    part_idx = min(symb_to_orbidx[psym,0]) # lowest unocc orb of this sym

    hplist = [(hole_idx,0,False),(part_idx,0,True)]

    nelec = mf.mol.nelec
    nelec_tot = sum(nelec)

    ormas_nspace = 2
    ormas_min_e = (0, nelec_tot - 1)
    ormas_max_e = (1, nelec_tot)
    ormas_mstart = (1,2)
    ormas_info = (ormas_nspace, ormas_min_e, ormas_max_e, ormas_mstart)

    return hplist, ormas_info

def save_ormas_ezfio(mf,hsym,psym,ezpath,n_det_max=10000):
    hplist, ormas_info = gen_core_ormas_atom(mf,hsym,psym)
    save_1det_to_ezfio(mf,ezpath,hplist=hplist,n_det_max=n_det_max)
    set_ormas_ezfio(ezpath,ormas_info)
    return

def save_ground_ezfio(mf,ezpath,n_det_max=10000):
    save_1det_to_ezfio(mf,ezpath,n_det_max=n_det_max)
    return


