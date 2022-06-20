program spectral_density
    implicit none
    BEGIN_DOC
    ! Program that calculates the spectral density.
    END_DOC

    logical                         :: force_reads, finished
    integer                         :: N_det_read, N_test, i, j, nnz, nnz_max
    integer, allocatable            :: idx_arr(:)
    integer(bit_kind), allocatable  :: psi_det_read(:,:,:)
    double precision , allocatable  :: psi_coef_read(:,:)
    complex*16       , allocatable  :: psi_coef_complex_read(:,:)

    call ezfio_get_determinants_n_det(N_det_read)
    N_det = N_det_read
    if (N_det == N_det_read) then 
        allocate(psi_det_read(N_int, 2, N_det))
        call ezfio_get_determinants_psi_det(psi_det_read)
        psi_det = psi_det_read
        deallocate(psi_det_read)
    end if

    call form_sparse_dH(finished)
    ! call test_unique_looping(finished)
    print *, finished

    ! nnz_max = ceiling(sqrt(real(nnz_max_per_row)))

    ! print *, nnz_max, nnz_max_per_row

    ! allocate(idx_arr(nnz_max))
    ! do i = 1, N_det
    !     idx_arr = 0
    !     call get_sparse_columns(i, idx_arr, nnz, nnz_max)
    !     print *, '-------------------'
    !     write(*, '(A12, I10, A12, I10)'), 'det idx:', idx_arr(1),&
    !                                       'nnz:', nnz
    !     do j = 1, nnz+1
    !         write(*, '(I10, I10)'), j, idx_arr(j)
    !     end do
    ! end do
    ! deallocate(idx_arr)

    ! idx_arr = get_sparse_columns(5,1,elec_alpha_num+1)
    ! do i = 1, size(idx_arr,1)
    !     write(*,'(I10, I10)'), i , idx_arr(i)
    ! end do
    ! if (is_complex) then 
    !     allocate(psi_coef_complex_read(N_det, N_states))
    !     call ezfio_get_determinants_psi_coef_complex(psi_coef_complex_read)
    !     psi_coef_complex = psi_coef_complex_read
    !     deallocate(psi_coef_complex_read)

    !     force_reads = size(psi_coef_complex, 1) == N_det_read .and.&
    !     size(psi_det, 3)  == N_det_read
    !     if(force_reads) then
    !         if(spectral_density_calc_A) then
    !             call ezfio_set_spectral_density_spectral_density_A_complex(spectral_density_A_complex)
    !             if(write_greens_f) then
    !                 call ezfio_set_spectral_density_greens_A_complex(greens_A_complex)
    !             end if
    !         end if 
    !         if(spectral_density_calc_R) then
    !             call ezfio_set_spectral_density_spectral_density_R_complex(spectral_density_R_complex)
    !             if(write_greens_f) then
    !                 call ezfio_set_spectral_density_greens_R_complex(greens_R_complex)
    !             end if
    !         end if
    !     end if
    ! else
    !     allocate(psi_coef_read(N_det, N_states))
    !     call ezfio_get_determinants_psi_coef(psi_coef_read)
    !     psi_coef = psi_coef_read
    !     deallocate(psi_coef_read)

    !     force_reads = size(psi_coef, 1) == N_det_read .and.&
    !     size(psi_det, 3)  == N_det_read
        
    !     if(force_reads) then
    !         if(spectral_density_calc_A) then
    !             call ezfio_set_spectral_density_spectral_density_A(spectral_density_A)
    !             if(write_greens_f) then
    !                 call ezfio_set_spectral_density_greens_A(greens_A)
    !             end if
    !         end if 
    !         if(spectral_density_calc_R) then
    !             call ezfio_set_spectral_density_spectral_density_R(spectral_density_R)
    !             if(write_greens_f) then
    !                 call ezfio_set_spectral_density_greens_R(greens_R)
    !             end if
    !         end if
    !     end if
    ! end if
    
end

BEGIN_PROVIDER [double precision, spectral_density_A, (greens_omega_N)]
    implicit none

    double precision :: pi

    pi = acos(-1.0)
    spectral_density_A = (-1.0/pi) * aimag(greens_A)

END_PROVIDER

BEGIN_PROVIDER [double precision, spectral_density_R, (greens_omega_N)]
    implicit none

    double precision :: pi

    pi = acos(-1.0)
    spectral_density_R = (-1.0/pi) * aimag(greens_R)

END_PROVIDER

BEGIN_PROVIDER [double precision, spectral_density_A_complex, (greens_omega_N)]
    implicit none

    double precision :: pi

    pi = acos(-1.0)
    spectral_density_A_complex = (-1.0/pi) * aimag(greens_A_complex)

END_PROVIDER

BEGIN_PROVIDER [double precision, spectral_density_R_complex, (greens_omega_N)]
    implicit none

    double precision :: pi

    pi = acos(-1.0)
    spectral_density_R_complex = (-1.0/pi) * aimag(greens_R_complex)

END_PROVIDER


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!  Unit tests !!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! BEGIN_PROVIDER [integer, cfraction_test]
!     implicit none
!     use cfraction

!     !!!! Continued fraction tests
!     ! Real tests
!     double precision, allocatable    :: a_r(:), b_r(:)
!     double precision                 :: a0_r, phi, phi_ref, pi, pi_ref
!     integer                          :: i, N

!     N = 97
!     allocate(a_r(N), b_r(N))

!     a_r = 1
!     b_r = 1
!     a0_r = 1

!     print *, "Calculating phi directly"
!     phi_ref = (1.d0+sqrt(5.d0))/2.d0
!     print *, phi_ref

!     print *, "Calculating phi with continued fraction expansion"
!     phi = cfraction_r(a0_r, a_r, b_r, N)
!     print *, phi

!     print *, "Calculating pi directly"
!     pi_ref = acos(-1.d0)
!     print *, pi_ref

!     print *, "Calculating pi with continued fraction expansion"
!     a0_r = 3
!     b_r = (/7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2,&
!             2, 1, 84, 2, 1, 1, 15, 3, 13, 1, 4, 2, 6, 6, 99, 1, 2, 2, 6,&
!             3, 5, 1, 1, 6, 8, 1, 7, 1, 2, 3, 7, 1, 2, 1, 1, 12, 1, 1, 1,&
!             3, 1, 1, 8, 1, 1, 2, 1, 6, 1, 1, 5, 2, 2, 3, 1, 2, 4, 4, 16,&
!             1, 161, 45, 1, 22, 1, 2, 2, 1, 4, 1, 2, 24, 1, 2, 1, 3, 1,&
!             2, 1/)

!     pi = cfraction_r(a0_r, a_r, b_r, N)
!     print *, pi

!     deallocate(a_r, b_r)

!     ! Complex tests
!     complex*16, allocatable     :: a_c(:), b_c(:)
!     complex*16                  :: a0_c, ref_val, test_val, x

!     N = 100

!     allocate(a_c(N), b_c(N))

!     x = (2.3, -1.4) ! random value
!     print *, x
!     a0_c = (1.d0, 0.d0)
!     a_c(1) =  x
!     b_c(1) = 1

!     do i = 2, N
!         a_c(i) = -(i-1)*x
!         b_c(i) = i + x
!     enddo

!     print *, "Calculating reference exp(x)"
!     ref_val = exp(x)
!     print *, ref_val

!     print *, "Calculating test exp(x) with complex continued fraction"
!     test_val = cfraction_c(a0_c, a_c, b_c, N)
!     print *, test_val
!     deallocate(a_c, b_c)

!     cfraction_test = 1
! END_PROVIDER

! BEGIN_PROVIDER [double precision, lanczos_alpha, (lanczos_N)]
! &BEGIN_PROVIDER [double precision, lanczos_beta, (lanczos_N)]
! &BEGIN_PROVIDER [double precision, lanczos_basis, (lanczos_N, lanczos_N)]
! &BEGIN_PROVIDER [double precision, lanczos_tri_H, (lanczos_N, lanczos_N)]
! &BEGIN_PROVIDER [double precision, lanczos_int_Q, (lanczos_N, lanczos_N)]
! &BEGIN_PROVIDER [double precision, lanczos_Q, (lanczos_N, lanczos_N)]

!     implicit none

!     !!! Real tests
!     integer                       :: k, sze, i, j, ii, N_tests
!     sze = lanczos_N
!     k = lanczos_N

!     double precision              :: H(lanczos_N,lanczos_N), tH(lanczos_N,lanczos_N), uu(lanczos_N,lanczos_N), Q(lanczos_N,lanczos_N), u(lanczos_N), alpha(lanczos_N), beta(lanczos_N)
!     double precision              :: dnrm2, err, Q_int(lanczos_N, lanczos_N)

!     N_tests = lanczos_n_tests

!     do ii = 1, N_tests

!         H = 0
!         tH = 0
!         Q = 0
!         u = 1.0
!         u(1) = 2.0
!         u = u / dnrm2(sze, u, 1)

!         H = lanczos_test_H_r(:,:,ii)

!         call lanczos_tridiag_reortho_rb(H, u, uu, alpha, beta, k, sze)

!         ! form tridiagonal matrix, and check to make sure Q vectors are calculated correctly
        
!         tH = 0 
!         do i = 1, sze
!             tH(i,i) = alpha(i)
!             if (i < sze) then 
!                 tH(i+1,i) = beta(i+1)
!                 tH(i,i+1) = beta(i+1)
!             end if
!         end do

!         lanczos_tri_H = tH
!         lanczos_basis = uu


!         call ordered_dgemm(sze, uu, tH, Q, Q_int, 'N', 'T', 'N', 'N')

!         lanczos_int_Q = Q_int
!         lanczos_Q = Q

!         err = 0
!         do i = 1, sze
!             do j = 1, sze
!                 err += abs(Q(i,j)-H(i,j))
!             end do
!         end do

!         print *, "error iter ", ii, err/(sze*sze)

        
!     end do

!     lanczos_alpha = alpha
!     lanczos_beta = beta

! END_PROVIDER


! BEGIN_PROVIDER [double precision, lanczos_alpha_complex, (lanczos_N)]
! &BEGIN_PROVIDER [double precision, lanczos_beta_complex, (lanczos_N)]
! &BEGIN_PROVIDER [complex*16, lanczos_basis_complex, (lanczos_N,lanczos_N)]
! &BEGIN_PROVIDER [complex*16, lanczos_tri_H_complex, (lanczos_N, lanczos_N)]
! &BEGIN_PROVIDER [complex*16, lanczos_int_Q_complex, (lanczos_N, lanczos_N)]
! &BEGIN_PROVIDER [complex*16, lanczos_Q_complex, (lanczos_N, lanczos_N)]

!     implicit none

!     !!! Complex tests
!     integer                       :: k, sze, i, j, ii, N_tests
!     N_tests = lanczos_n_tests
!     sze = lanczos_N
!     k = lanczos_N

!     complex*16              :: H(lanczos_N,lanczos_N), tH(lanczos_N,lanczos_N), uu(lanczos_N,lanczos_N), Q(lanczos_N,lanczos_N), u(lanczos_N), Q_int(lanczos_N, lanczos_N)
!     double precision        :: alpha(lanczos_N), beta(lanczos_N), dznrm2, err

!     N_tests = lanczos_n_tests

!     do ii = 1, N_tests
        
!         H = (0.0, 0.0)
!         tH = (0.0, 0.0)
!         Q = (0.0, 0.0)
!         u = (1.0, 0.0)
!         u(1) = (2.0, 0.0)
!         u = u / dznrm2(sze, u, 1)

!         H = cmplx(lanczos_test_H_c(1,:,:,ii),&
!                   lanczos_test_H_c(2,:,:,ii),&
!                   kind=16)

!         call lanczos_tridiag_reortho_cb(H, u, uu, alpha, beta, k, sze)

!         lanczos_basis_complex = uu

!         tH = 0 
!         do i = 1, sze
!             tH(i,i) = alpha(i)
!             if (i < sze) then 
!                 tH(i+1,i) = beta(i+1)
!                 tH(i,i+1) = beta(i+1)
!             end if
!         end do

!         lanczos_tri_H_complex = tH

!         call ordered_zgemm(sze, uu, tH, Q, Q_int, 'N', 'C', 'N', 'N')
        
!         lanczos_Q_complex = Q
!         lanczos_int_Q_complex = Q_int

!         err = 0
!         do i = 1, sze
!             do j = 1, sze
!                 err += abs(Q(i,j)-H(i,j))
!             end do
!         end do

!         print *, "error iter ", ii, err/(sze*sze)

!     end do

        
!     lanczos_alpha_complex = alpha
!     lanczos_beta_complex = beta

! END_PROVIDER


! subroutine ordered_dgemm(sze, basis, tH, Q, Q_int, t0, t1, t2, t3)
!     implicit none

!     character, intent(in)            :: t0, t1, t2, t3
!     integer, intent(in)              :: sze
!     double precision, intent(in)     :: tH(sze,sze), basis(sze,sze)
!     double precision, intent(inout)  :: Q(sze,sze), Q_int(sze, sze)


!     call dgemm(t0, t1, sze, sze, sze, 1.d0, &
!                 tH, size(tH, 1), &
!                 basis, size(basis, 1), 0.d0, &
!                 Q_int, size(Q_int, 1))

!     call dgemm(t2, t3, sze, sze, sze, 1.d0, &
!                 basis, size(basis, 1), &
!                 Q_int, size(Q_int, 1), 0.d0, & 
!                 Q, size(Q, 1))

! end

! subroutine ordered_zgemm(sze, basis, tH, Q, Q_int, t0, t1, t2, t3)
!     implicit none

!     character, intent(in)            :: t0, t1, t2, t3
!     integer, intent(in)              :: sze
!     complex*16, intent(in)           :: tH(sze,sze), basis(sze,sze)
!     complex*16, intent(inout)        :: Q(sze,sze), Q_int(sze, sze)


!     call zgemm(t0, t1, sze, sze, sze, (1.d0, 0.d0), &
!                 tH, size(tH, 1), &
!                 basis, size(basis, 1), (0.d0, 0.d0), &
!                 Q_int, size(Q_int, 1))

!     call zgemm(t2, t3, sze, sze, sze, (1.d0, 0.d0), &
!                 basis, size(basis, 1), &
!                 Q_int, size(Q_int, 1), (0.d0, 0.d0), & 
!                 Q, size(Q, 1))

! end