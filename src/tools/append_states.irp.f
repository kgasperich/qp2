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
  ! routine called by :c:func:`save_one_e_dm`
  END_DOC
 
  integer :: i,j,ndet_to_add,ndet_total,nstates_total,iint,istate
  logical :: j_done
  integer*8, external :: det_search_key
  integer*8 :: i_key,j_key

  PROVIDE psi_det psi_coef
  PROVIDE data_psi_det_new data_psi_coef_new
 
  double precision, allocatable :: psi_coef_sorted_new(:,:), psi_coef_total(:,:)
  integer(bit_kind), allocatable :: psi_det_sorted_new(:,:,:), psi_det_total(:,:,:)

  ndet_total = N_det + data_n_det_new
  nstates_total = N_states + data_n_states_new
 
  allocate(psi_coef_sorted_new(data_n_det_new, data_n_states_new), psi_det_sorted_new(N_int,2,data_n_det_new))

  allocate(psi_coef_total(ndet_total, nstates_total), psi_det_total(N_int,2,ndet_total))
 
  call sort_dets_by_det_search_key(data_n_det_new, data_psi_det_new, data_psi_coef_new, data_n_det_new, psi_det_sorted_new, psi_coef_sorted_new, data_n_states_new)


  j_done = .False.
  j=1
  j_key = det_search_key(psi_det_sorted_new(1,1,j),N_int)
  ndet_to_add=0
  do i=1,N_det
    i_key = det_search_key(psi_det_sorted_bit(1,1,i),N_int)
    do while ((j_key <= i_key).and.(j <= data_n_det_new))
      ndet_to_add += 1
      do iint=1,N_int
        psi_det_total(iint,1,ndet_to_add) = psi_det_sorted_new(iint,1,j)
        psi_det_total(iint,2,ndet_to_add) = psi_det_sorted_new(iint,2,j)
      enddo
      if (j_key == i_key) then
        do istate=1,N_states
          psi_coef_total(ndet_to_add,istate) = psi_coef_sorted_bit(i,istate)
        enddo
      else
        do istate=1,N_states
          psi_coef_total(ndet_to_add,istate) = 0.d0
        enddo
      endif
      do istate=1,data_n_states_new
        psi_coef_total(ndet_to_add,N_states+istate) = psi_coef_sorted_new(j,istate)
      enddo
      j += 1
      if (j > data_n_det_new) exit
      j_key = det_search_key(psi_det_sorted_new(1,1,j),N_int)
    enddo

    ndet_to_add += 1
    do iint=1,N_int
      psi_det_total(iint,1,ndet_to_add) = psi_det_sorted_bit(iint,1,i)
      psi_det_total(iint,2,ndet_to_add) = psi_det_sorted_bit(iint,2,i)
    enddo
    do istate=1,N_states
      psi_coef_total(ndet_to_add,istate) = psi_coef_sorted_bit(i,istate)
    enddo
    do istate=1,data_n_states_new
      psi_coef_total(ndet_to_add,N_states+istate) = 0.d0
    enddo
  enddo

  do while (j <= data_n_det_new)
    ndet_to_add += 1
    do iint=1,N_int
      psi_det_total(iint,1,ndet_to_add) = psi_det_sorted_bit(iint,1,j)
      psi_det_total(iint,2,ndet_to_add) = psi_det_sorted_bit(iint,2,j)
    enddo
    do istate=1,N_states
      psi_coef_total(ndet_to_add,istate) = 0.d0
    enddo
    do istate=1,data_n_states_new
      psi_coef_total(ndet_to_add,N_states+istate) = psi_coef_sorted_new(j,istate)
    enddo
    j += 1
  enddo

  call save_wavefunction_general(ndet_to_add, nstates_total, psi_det_total(:,:,:ndet_to_add),ndet_total,psi_coef_total) 
end
