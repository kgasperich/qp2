program rotate_mos_iju
  implicit none
  BEGIN_DOC
  ! Rotates molecular orbitals i and j to eliminate C[mu,i]
  ! $(c*\phi_i + s*\phi_j )$ and
  ! $(-s*\phi_i + c*\phi_j )$.
  END_DOC
  integer                        :: mu,iorb,jorb
  integer                        :: i,j
  double precision               :: theta, c, s
  double precision, allocatable  :: mo_coef_tmp(:,:)

  read(5,*)iorb,jorb,mu

  allocate(mo_coef_tmp(ao_num,mo_num))
  mo_coef_tmp = mo_coef

  theta = atan(-mo_coef(mu,iorb),mo_coef(mu,jorb))
  c = cos(theta)
  s = sin(theta)
  do i = 1, ao_num
    mo_coef(i,iorb) = ( c*mo_coef_tmp(i,iorb) + s*mo_coef_tmp(i,jorb) )
    mo_coef(i,jorb) = ( -s*mo_coef_tmp(i,iorb) + c*mo_coef_tmp(i,jorb) )
  enddo

  touch mo_coef
  call save_mos
  
end
