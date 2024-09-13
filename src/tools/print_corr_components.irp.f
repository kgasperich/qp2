program print_corr_components
  implicit none
 BEGIN_DOC
! print dynamic and nondynamic correlation
! refs:
!  E. Ramos-Cordoba, P. Salvador, E. Matito. "Separation of dynamic and nondynamic correlation" Phys. Chem. Chem. Phys., 2016, 18, 24015
!  E. Ramos-Cordoba, E. Matito. "Local Descriptors of Dynamic and Nondynamic Correlation" JCTC 2017 13 (6), 2705-2711
 END_DOC
  read_wf = .True.
  touch read_wf
  call routine_print_corr_components

end

subroutine routine_print_corr_components
 implicit none
 BEGIN_DOC
 ! routine called by :c:func:`print_corr_components`
 END_DOC
 PROVIDE one_e_dm_mo_alpha one_e_dm_mo_beta

 double precision, allocatable :: no_occs_ab(:,:,:), tmp_vals(:), tmp_vecs(:,:), corr_d(:,:), corr_nd(:,:)
 integer :: norb, nst, istate, iorb, ispin
 double precision :: o_i, a_i

 norb = size(one_e_dm_mo_alpha,1)
 nst = size(one_e_dm_mo_alpha,3)

 allocate(no_occs_ab(2,norb,nst))
 allocate(corr_d(2,nst), corr_nd(2,nst))

 no_occs_ab = 0.d0
 corr_d = 0.d0
 corr_nd = 0.d0



 allocate(tmp_vals(norb), tmp_vecs(norb,norb))
 do istate=1,nst
   call lapack_diagd(tmp_vals, tmp_vecs, one_e_dm_mo_alpha(:,:,istate), norb, norb)
   do iorb=1,norb
     no_occs_ab(1,iorb,istate) = tmp_vals(iorb)
   enddo
   call lapack_diagd(tmp_vals, tmp_vecs, one_e_dm_mo_beta(:,:,istate), norb, norb)
   do iorb=1,norb
     no_occs_ab(2,iorb,istate) = tmp_vals(iorb)
   enddo
 enddo
 deallocate(tmp_vals, tmp_vecs)

 do istate=1,nst
   do ispin=1,2
     do iorb=1,norb
       o_i = no_occs_ab(ispin,iorb,istate)
       a_i = o_i * (1.d0 - o_i)
       corr_d(ispin,istate) += 0.25d0 * (dsqrt(a_i) - 2.d0 * a_i)
       corr_nd(ispin,istate) += 0.5d0 *  a_i
     enddo
   enddo
 enddo

 character(*), parameter :: fmt2 = '(A14, F12.4, F12.4, F12.4)'
 character(*), parameter :: fmt1 = '(14X, A12, A12, A12)'
 do istate = 1, nst
     print '(A,I3)', 'State ', istate
     !print '(A)', '                       alpha        beta         total'
     write (*, fmt1), 'alpha','beta','total'
     
     ! Use the format string variable
     write (*, fmt2) 'dynamic      ', corr_d(1,istate), corr_d(2,istate), corr_d(1,istate) + corr_d(2,istate)
     write (*, fmt2) 'nondynamic   ', corr_nd(1,istate), corr_nd(2,istate), corr_nd(1,istate) + corr_nd(2,istate)
     write (*, fmt2) 'total        ', corr_d(1,istate) + corr_nd(1,istate), corr_d(2,istate) + corr_nd(2,istate), &
                           corr_d(1,istate) + corr_d(2,istate) + corr_nd(1,istate) + corr_nd(2,istate)
 
     !print *,*  ! Blank line for readability
 end do



end
