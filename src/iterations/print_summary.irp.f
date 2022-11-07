subroutine print_summary(e_,pt2_data,pt2_data_err,n_det_,n_configuration_,n_st,s2_)
  use selection_types
  implicit none
  BEGIN_DOC
! Print the extrapolated energy in the output
  END_DOC

  integer, intent(in)            :: n_det_, n_configuration_, n_st
  double precision, intent(in)   :: e_(n_st), s2_(n_st)
  type(pt2_type)  , intent(in)   :: pt2_data, pt2_data_err
  integer                        :: i, k
  integer                        :: N_states_p
  character*(9)                  :: pt2_string
  character*(512)                :: fmt

  if (do_pt2) then
    pt2_string = '        '
  else
    pt2_string = '(approx)'
  endif

  N_states_p = min(N_det_,n_st)

  print *, ''
  print '(A,I12)',  'Summary at N_det = ', N_det_
  print '(A)',      '-----------------------------------'
  print *, ''

  write(fmt,*) '(''# ============'',', N_states_p, '(1X,''=============================''))'
  write(*,fmt)
  write(fmt,*) '(13X,', N_states_p, '(6X,A7,1X,I6,10X))'
  write(*,fmt) ('State',k, k=1,N_states_p)
  write(fmt,*) '(''# ============'',', N_states_p, '(1X,''=============================''))'
  write(*,fmt)
  write(fmt,*) '(A13,', N_states_p, '(1X,F14.8,15X))'
  write(*,fmt) '# E          ', e_(1:N_states_p)
  if (N_states_p > 1) then
    write(*,fmt) '# Excit. (au)', e_(1:N_states_p)-e_(1)
    write(*,fmt) '# Excit. (eV)', (e_(1:N_states_p)-e_(1))*27.211396641308d0
  endif
  write(fmt,*) '(A13,', 2*N_states_p, '(1X,F14.8))'
  write(*,fmt) '# PT2 '//pt2_string, (pt2_data % pt2(k), pt2_data_err % pt2(k), k=1,N_states_p)
  write(*,fmt) '# rPT2'//pt2_string, (pt2_data % rpt2(k), pt2_data_err % rpt2(k), k=1,N_states_p)
  write(*,'(A)') '#'
  write(*,fmt) '# E+PT2      ', (e_(k)+pt2_data % pt2(k),pt2_data_err % pt2(k), k=1,N_states_p)
  write(*,fmt) '# E+rPT2     ', (e_(k)+pt2_data % rpt2(k),pt2_data_err % rpt2(k), k=1,N_states_p)
  if (N_states_p > 1) then
    write(*,fmt) '# Excit. (au)', ( (e_(k)+pt2_data % pt2(k)-e_(1)-pt2_data % pt2(1)), &
      dsqrt(pt2_data_err % pt2(k)*pt2_data_err % pt2(k)+pt2_data_err % pt2(1)*pt2_data_err % pt2(1)), k=1,N_states_p)
    write(*,fmt) '# Excit. (eV)', ( (e_(k)+pt2_data % pt2(k)-e_(1)-pt2_data % pt2(1))*27.211396641308d0, &
      dsqrt(pt2_data_err % pt2(k)*pt2_data_err % pt2(k)+pt2_data_err % pt2(1)*pt2_data_err % pt2(1))*27.211396641308d0, k=1,N_states_p)
  endif
  write(fmt,*) '(''# ============'',', N_states_p, '(1X,''=============================''))'
  write(*,fmt)
  print *,  ''

  print *,  'N_det             = ', N_det_
  print *,  'N_states          = ', n_st
  if (s2_eig) then
    print *,  'N_cfg             = ', N_configuration_
    if (only_expected_s2) then
      print *,  'N_csf             = ', N_csf
    endif
  endif
  print *,  ''

  do k=1, N_states_p
    print*,'* State ',k
    print *,  '< S^2 >         = ', s2_(k)
    print *,  'E               = ', e_(k)
    print *,  'Variance        = ', pt2_data % variance(k), ' +/- ', pt2_data_err % variance(k)
    print *,  'PT norm         = ', dsqrt(pt2_data % overlap(k,k)), ' +/- ', 0.5d0*dsqrt(pt2_data % overlap(k,k)) * pt2_data_err % overlap(k,k) / (pt2_data % overlap(k,k))
    print *,  'PT2             = ', pt2_data % pt2(k), ' +/- ', pt2_data_err % pt2(k)
    print *,  'rPT2            = ', pt2_data % rpt2(k), ' +/- ', pt2_data_err % rpt2(k)
    print *,  'E+PT2 '//pt2_string//' = ', e_(k)+pt2_data % pt2(k), ' +/- ', pt2_data_err % pt2(k)
    print *,  'E+rPT2'//pt2_string//' = ', e_(k)+pt2_data % rpt2(k), ' +/- ', pt2_data_err % rpt2(k)
    print *,  ''
  enddo

  print *,  '-----'
  if(n_st.gt.1)then
    print *, 'Variational Energy difference (au | eV)'
    do i=2, N_states_p
      print*,'Delta E = ', (e_(i) - e_(1)), &
        (e_(i) - e_(1)) * 27.211396641308d0
    enddo
    print *,  '-----'
    print*, 'Variational + perturbative Energy difference (au | eV)'
    do i=2, N_states_p
      print*,'Delta E = ', (e_(i)+ pt2_data % pt2(i) - (e_(1) + pt2_data % pt2(1))), &
        (e_(i)+ pt2_data % pt2(i) - (e_(1) + pt2_data % pt2(1))) * 27.211396641308d0
    enddo
    print *,  '-----'
    print*, 'Variational + renormalized perturbative Energy difference (au | eV)'
    do i=2, N_states_p
      print*,'Delta E = ', (e_(i)+ pt2_data % rpt2(i) - (e_(1) + pt2_data % rpt2(1))), &
        (e_(i)+ pt2_data % rpt2(i) - (e_(1) + pt2_data % rpt2(1))) * 27.211396641308d0
    enddo
  endif

  call print_energy_components_summary()

end subroutine

subroutine print_energy_components_summary()
  implicit none
  BEGIN_DOC
! Prints the different components of the energy.
  END_DOC
  integer, save :: ifirst = 0
  double precision :: Vee, Ven, Vnn, Vecp, T, f
  integer  :: i,j,k

  Vnn = nuclear_repulsion

  print *, 'Energy components'
  print *, '================='
  print *, ''
  do k=1,N_states

    Ven  = 0.d0
    Vecp = 0.d0
    T    = 0.d0

    do j=1,mo_num
      do i=1,mo_num
        f = one_e_dm_mo_alpha(i,j,k) + one_e_dm_mo_beta(i,j,k)
        Ven  = Ven  + f * mo_integrals_n_e(i,j)
        Vecp = Vecp + f * mo_pseudo_integrals(i,j)
        T    = T    + f * mo_kinetic_integrals(i,j)
      enddo
    enddo
    Vee = psi_energy(k) - Ven - Vecp - T
    
    if (ifirst == 0) then
      ifirst = 1
      print *, 'Vnn  : Nucleus-Nucleus   potential energy'
      print *, 'Ven  : Electron-Nucleus  potential energy'
      print *, 'Vee  : Electron-Electron potential energy'
      print *, 'Vecp : Potential energy of the pseudo-potentials'
      print *, 'T    : Electronic kinetic energy'
      print *, ''
    endif

    print *, 'State ', k
    print *, '---------'
    print *, ''
    print *, 'Vnn  = ', Vnn
    print *, 'Ven  = ', Ven
    print *, 'Vee  = ', Vee
    print *, 'Vecp = ', Vecp
    print *, 'T    = ', T
    print *, ''
  enddo

  print *, ''

end

