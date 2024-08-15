program save_natorb_block
  implicit none
  BEGIN_DOC
! Save natural |MOs| into the |EZFIO|.
!
! This program reads the wave function stored in the |EZFIO| directory,
! extracts the corresponding natural orbitals and setd them as the new
! |MOs|.
!
! If this is a multi-state calculation, the density matrix that produces
! the natural orbitals is obtained from an average of the density
! matrices of each state with the corresponding
! :option:`determinants state_average_weight`
  END_DOC
  PROVIDE nucl_coord
  read_wf = .True.
  touch read_wf
  PROVIDE psi_det
  call routine
end
subroutine routine
  implicit none

  integer(bit_kind), allocatable  :: tmp_det(:,:,:) !(N_int,2,ndet)
  double precision, allocatable :: tmp_coef(:,:)
  double precision :: c_02, c_i
  integer :: ndet_save,i
  ! psi_det will be invalidated when MOs are rotated
  ! save first dets to use as starting guess for next round of CIPSI
  ! much easier than trying to make ref_bitmask that will fit all ORMAS constraints
  
  ndet_save=N_det
  c_02 = (psi_coef_sorted(1,1)*psi_coef_sorted(1,1))
  do i=1,N_det
    c_i = psi_coef_sorted(i,1)
    if (c_i*c_i < (c_02 - 1.0d-7)) then
      ndet_save = i-1
      exit
    endif
  enddo
  allocate(tmp_det(N_int, 2, ndet_save), tmp_coef(ndet_save,N_states))

  tmp_det = psi_det_sorted(:,:,:ndet_save)
  tmp_coef = psi_coef_sorted(:ndet_save,:)

  call save_natural_mos_block

  call save_wavefunction_general(ndet_save,min(N_states,ndet_save),tmp_det,size(tmp_coef,1),tmp_coef)
  call ezfio_set_mo_two_e_ints_io_mo_two_e_integrals('None')
  call ezfio_set_mo_one_e_ints_io_mo_one_e_integrals('None')
  call ezfio_set_mo_one_e_ints_io_mo_integrals_kinetic('None')
  call ezfio_set_mo_one_e_ints_io_mo_integrals_n_e('None')
  call ezfio_set_mo_one_e_ints_io_mo_integrals_pseudo('None')
  deallocate(tmp_det,tmp_coef)
end

