program append_states
  implicit none
 BEGIN_DOC
! Program that computes the one body density on the |MO| and |AO| basis
! for $\alpha$ and $\beta$ electrons from the wave function
! stored in the |EZFIO| directory, and then saves it into the
! :ref:`module_aux_quantities`.
!
! Then, the global variable :option:`aux_quantities data_one_e_dm_alpha_mo`
! and :option:`aux_quantities data_one_e_dm_beta_mo` (and the corresponding for |AO|)
! will automatically ! read this density in the next calculation. 
! This can be used to perform damping on the density in |RSDFT| calculations (see
! :ref:`module_density_for_dft`).
 END_DOC
  read_wf = .True.
  touch read_wf
  call routine_append_states

end

subroutine routine_append_states
  use bitmasks
  implicit none
  BEGIN_DOC
  ! routine called by :c:func:`append_states`
  END_DOC
 
  integer :: i,j,ndet_to_add,ndet_total,nstates_total,iint,istate,ndet_new,nstates_new
  logical, external :: is_in_wavefunction
  integer, external :: get_index_in_psi_det_sorted_bit

  PROVIDE psi_det psi_coef
  PROVIDE data_psi_det_new data_psi_coef_new
 
  double precision, allocatable  :: coef_buf_existing(:,:), coef_buf_new(:,:), coef_buf_total(:,:)
  integer(bit_kind), allocatable :: det_buf_new(:,:,:), det_buf_total(:,:,:)

  ndet_new = data_n_det_new
  nstates_new = data_n_states_new
  nstates_total = N_states + data_n_states_new

  allocate(coef_buf_existing(N_det, nstates_new), &
           coef_buf_new(ndet_new, nstates_new), &
           det_buf_new(N_int, 2, ndet_new))

  coef_buf_existing = 0.d0
  coef_buf_new = 0.d0

  ndet_to_add = 0

  do j=1,ndet_new
    if (is_in_wavefunction(data_psi_det_new(1,1,j),N_int)) then
      i = get_index_in_psi_det_sorted_bit(data_psi_det_new(1,1,j),N_int)
      do istate=1,nstates_new
        coef_buf_existing(i, istate) = data_psi_coef_new(j, istate)
      enddo
    else
      ndet_to_add += 1
      do istate=1,nstates_new
        coef_buf_new(ndet_to_add, istate) = data_psi_coef_new(j, istate)
      enddo
      do iint=1,N_int
        det_buf_new(iint,1,ndet_to_add) = data_psi_det_new(iint,1,j)
        det_buf_new(iint,2,ndet_to_add) = data_psi_det_new(iint,2,j)
      enddo
    endif
  enddo

  ndet_total = N_det + ndet_to_add

  allocate(coef_buf_total(ndet_total, nstates_total), &
           det_buf_total(N_int, 2, ndet_total))

  coef_buf_total = 0.d0

  do i=1,N_det
    do istate=1,N_states
      coef_buf_total(i,istate) = psi_coef_sorted_bit(i,istate)
    enddo
    do istate=1,nstates_new
      coef_buf_total(i,istate + N_states) = coef_buf_existing(i, istate)
    enddo
    do iint=1,N_int
      det_buf_total(iint,1,i) = psi_det_sorted_bit(iint,1,i)
      det_buf_total(iint,2,i) = psi_det_sorted_bit(iint,2,i)
    enddo
  enddo
  do i=1,ndet_to_add
    j=N_det + i
    do istate=1,N_states
      coef_buf_total(j,istate) = 0.d0
    enddo
    do istate=1,nstates_new
      coef_buf_total(j,istate + N_states) = coef_buf_new(i, istate)
    enddo
    do iint=1,N_int
      det_buf_total(iint,1,j) = det_buf_new(iint,1,i)
      det_buf_total(iint,2,j) = det_buf_new(iint,2,i)
    enddo
  enddo

  call save_wavefunction_general(ndet_total, nstates_total, det_buf_total,ndet_total,coef_buf_total)

  deallocate(coef_buf_existing, &
             coef_buf_new, &
             det_buf_new, &
             coef_buf_total, &
             det_buf_total)

end
