logical function is_the_hole_in_det(key_in,ispin,i_hole)
  use bitmasks
  ! returns true if the electron ispin is absent from i_hole
 implicit none
 integer, intent(in) :: i_hole,ispin
 integer(bit_kind), intent(in) :: key_in(N_int,2)
 integer(bit_kind) :: key_tmp(N_int)
 integer(bit_kind) :: itest(N_int)
 integer :: i,j,k
 do i = 1, N_int
  itest(i) = 0_bit_kind
 enddo
 k = shiftr(i_hole-1,bit_kind_shift)+1
 j = i_hole-shiftl(k-1,bit_kind_shift)-1
 itest(k) = ibset(itest(k),j)
 j = 0
 do i = 1, N_int
  key_tmp(i) = iand(itest(i),key_in(i,ispin))
  j += popcnt(key_tmp(i))
 enddo
 if(j==0)then
  is_the_hole_in_det = .True.
 else
  is_the_hole_in_det = .False.
 endif

end

logical function is_the_particl_in_det(key_in,ispin,i_particl)
  use bitmasks
  ! returns true if the electron ispin is absent from i_particl
 implicit none
 integer, intent(in) :: i_particl,ispin
 integer(bit_kind), intent(in) :: key_in(N_int,2)
 integer(bit_kind) :: key_tmp(N_int)
 integer(bit_kind) :: itest(N_int)
 integer :: i,j,k
 do i = 1, N_int
  itest(i) = 0_bit_kind
 enddo
 k = shiftr(i_particl-1,bit_kind_shift)+1
 j = i_particl-shiftl(k-1,bit_kind_shift)-1
 itest(k) = ibset(itest(k),j)
 j = 0
 do i = 1, N_int
  key_tmp(i) = iand(itest(i),key_in(i,ispin))
  j += popcnt(key_tmp(i))
 enddo
 if(j==0)then
  is_the_particl_in_det = .False.
 else
  is_the_particl_in_det = .True.
 endif

end
BEGIN_PROVIDER [integer, dim_list_ionized_core_orb]
  implicit none
  BEGIN_DOC
  ! dimensions for the allocation of list_core.
  ! it is at least 1
  END_DOC
   dim_list_ionized_core_orb = max(n_ionized_core_orb,1)
END_PROVIDER


 BEGIN_PROVIDER [ integer, list_ionized_core        , (dim_list_ionized_core_orb) ]
&BEGIN_PROVIDER [ integer, list_ionized_core_reverse, (mo_num) ]
  implicit none
  BEGIN_DOC
  ! List of MO indices which are in the core.
  END_DOC
  integer                        :: i, n
  list_ionized_core = 0
  list_ionized_core_reverse = 0

  n=0
  do i = 1, mo_num
    !if(mo_class(i) == 'Core')then
    if(i <= n_ionized_core_orb)then
      n += 1
      list_ionized_core(n) = i
      list_ionized_core_reverse(i) = n
    endif
  enddo
  print *,  'Ionized Core MOs:'
  print *,  list_ionized_core(1:n_ionized_core_orb)

END_PROVIDER


 BEGIN_PROVIDER [ integer(bit_kind), ionized_core_bitmask , (N_int,2) ]
&BEGIN_PROVIDER [ integer, n_int_ionized_core_max ]
  implicit none
  BEGIN_DOC
  ! Bitmask identifying the ionized core MOs
  END_DOC
  integer :: i,ispin
  ionized_core_bitmask  = 0_bit_kind
  if(n_ionized_core_orbs > 0)then
    call list_to_bitstring( ionized_core_bitmask(1,1), list_ionized_core, n_ionized_core_orb, N_int)
    call list_to_bitstring( ionized_core_bitmask(1,2), list_ionized_core, n_ionized_core_orb, N_int)
  endif
  n_int_ionized_core_max=0
  do i=1,N_int
    do ispin=1,2
      if (popcnt(ionized_core_bitmask(i,ispin))>0) then
        n_int_ionized_core_max=i
      endif
    enddo
  enddo

 END_PROVIDER


subroutine ab_holes_in_ionized_core(key_in,ab_holes)
  use bitmasks
  ! returns holes of each spin in ionized core orbs of key_in
  implicit none
  integer(bit_kind), intent(in) :: key_in(N_int,2)
  integer, intent(out) :: ab_holes(2)
  integer(bit_kind) :: key_tmp
  integer :: i,ispin
  ab_holes = 0
  do i = 1, n_int_ionized_core_max
    do ispin = 1,2
      key_tmp = iand(ionized_core_bitmask(i,ispin),not(key_in(i,ispin)))
      ab_holes(ispin) += popcnt(key_tmp)
    enddo
  enddo
end

integer function n_holes_in_ionized_core(key_in)
  use bitmasks
  ! returns holes of each spin in ionized core orbs of key_in
  implicit none
  integer(bit_kind), intent(in) :: key_in(N_int,2)
  integer :: i,ispin
  n_holes_in_ionized_core = 0
  do i = 1, n_int_ionized_core_max
    do ispin = 1,2
      n_holes_in_ionized_core += popcnt(iand(ionized_core_bitmask(i,ispin),not(key_in(i,ispin))))
    enddo
  enddo
end

logical function det_allowed_ionized_core(key_in)
  use bitmasks
  ! returns .True. if key_in has the proper number of holes in the core
  implicit none
  integer(bit_kind), intent(in) :: key_in(N_int,2)
  det_allowed_ionized_core = (n_holes_in_ionized_core(key_in) == n_core_holes)
end
