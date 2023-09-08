program print_dipole
 implicit none
 read_wf = .True.
 TOUCH read_wf


 call print_ao_spread
 call print_ao_dipole
 call print_mo_spread
 call print_mo_dipole

end

subroutine printmat(A,n1,n2,name,thresh)
  implicit none

  double precision, intent(in) :: A(n1,n2)
  integer, intent(in)          :: n1,n2
  character(len=*), intent(in) :: name
  double precision, intent(in) :: thresh

  integer :: i,j
  integer :: i_unit_output, getUnitAndOpen
  double precision :: v

  character*(128) :: output

  output = trim(ezfio_filename)//'/.'//name
  i_unit_output = getUnitAndOpen(output,'w')

  do i=1,n1
    do j=1,n2
      v = A(i,j)
      if (dabs(v).gt.thresh) then
        write(i_unit_output,*) i, j, v
      endif
    enddo
  enddo
end



subroutine print_ao_spread
 implicit none
 PROVIDE ao_spread_x ao_spread_y ao_spread_z
 double precision :: thr
 thr = 1E-14

 call printmat(ao_spread_x, ao_num, ao_num, 'ao_spread_x', thr)
 call printmat(ao_spread_y, ao_num, ao_num, 'ao_spread_y', thr)
 call printmat(ao_spread_z, ao_num, ao_num, 'ao_spread_z', thr)
end

subroutine print_ao_dipole
 implicit none
 PROVIDE ao_dipole_x ao_dipole_y ao_dipole_z
 double precision :: thr
 thr = 1E-14

 call printmat(ao_dipole_x, ao_num, ao_num, 'ao_dipole_x', thr)
 call printmat(ao_dipole_y, ao_num, ao_num, 'ao_dipole_y', thr)
 call printmat(ao_dipole_z, ao_num, ao_num, 'ao_dipole_z', thr)
end

subroutine print_mo_spread
 implicit none
 PROVIDE mo_spread_x mo_spread_y mo_spread_z
 double precision :: thr
 thr = 1E-14

 call printmat(mo_spread_x, mo_num, mo_num, 'mo_spread_x', thr)
 call printmat(mo_spread_y, mo_num, mo_num, 'mo_spread_y', thr)
 call printmat(mo_spread_z, mo_num, mo_num, 'mo_spread_z', thr)
end

subroutine print_mo_dipole
 implicit none
 PROVIDE mo_dipole_x mo_dipole_y mo_dipole_z
 double precision :: thr
 thr = 1E-14

 call printmat(mo_dipole_x, mo_num, mo_num, 'mo_dipole_x', thr)
 call printmat(mo_dipole_y, mo_num, mo_num, 'mo_dipole_y', thr)
 call printmat(mo_dipole_z, mo_num, mo_num, 'mo_dipole_z', thr)
end
