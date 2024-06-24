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

  integer(bit_kind)  :: tmp_det(N_int,2)

  ! psi_det will be invalidated when MOs are rotated
  ! save first det to use as starting guess for next round of CIPSI
  ! much easier than trying to make ref_bitmask that will fit all ORMAS constraints
  tmp_det = psi_det(:,:,1)
  call save_natural_mos_block
  psi_det(:,:,1) = tmp_det
  call save_first_determinant
  call ezfio_set_mo_two_e_ints_io_mo_two_e_integrals('None')
  call ezfio_set_mo_one_e_ints_io_mo_one_e_integrals('None')
  call ezfio_set_mo_one_e_ints_io_mo_integrals_kinetic('None')
  call ezfio_set_mo_one_e_ints_io_mo_integrals_n_e('None')
  call ezfio_set_mo_one_e_ints_io_mo_integrals_pseudo('None')
end
