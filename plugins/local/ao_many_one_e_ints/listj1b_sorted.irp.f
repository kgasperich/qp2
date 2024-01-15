
! ---

 BEGIN_PROVIDER [integer, List_comb_thr_b2_size, (ao_num, ao_num)]
&BEGIN_PROVIDER [integer, max_List_comb_thr_b2_size]

  implicit none
  integer          :: i_1s, i, j, ipoint
  integer          :: list(ao_num)
  double precision :: coef,beta,center(3),int_env
  double precision :: r(3),weight,dist

  List_comb_thr_b2_size = 0
  print*,'List_env1s_size = ',List_env1s_size

  do i = 1, ao_num
    do j = i, ao_num
      do i_1s = 1, List_env1s_size
        coef        = List_env1s_coef(i_1s)
        if(dabs(coef).lt.thrsh_cycle_tc) cycle
        beta        = List_env1s_expo(i_1s)
        beta = max(beta,1.d-12)
        center(1:3) = List_env1s_cent(1:3,i_1s)
        int_env = 0.d0
        do ipoint = 1, n_points_extra_final_grid
          r(1:3) = final_grid_points_extra(1:3,ipoint)
          weight = final_weight_at_r_vector_extra(ipoint)
          dist  = ( center(1) - r(1) )*( center(1) - r(1) )
          dist += ( center(2) - r(2) )*( center(2) - r(2) )
          dist += ( center(3) - r(3) )*( center(3) - r(3) )
          int_env += dabs(aos_in_r_array_extra_transp(ipoint,i) * aos_in_r_array_extra_transp(ipoint,j))*dexp(-beta*dist) * weight
        enddo
        if(dabs(coef)*dabs(int_env).gt.thrsh_cycle_tc)then
          List_comb_thr_b2_size(j,i) += 1
        endif
      enddo
    enddo 
  enddo

  do i = 1, ao_num
    do j = 1, i-1
      List_comb_thr_b2_size(j,i) = List_comb_thr_b2_size(i,j)
    enddo
  enddo
  do i = 1, ao_num
    list(i) = maxval(List_comb_thr_b2_size(:,i))
  enddo

  max_List_comb_thr_b2_size = maxval(list) 
  print*, ' max_List_comb_thr_b2_size = ',max_List_comb_thr_b2_size
 
END_PROVIDER 

! ---

 BEGIN_PROVIDER [ double precision, List_comb_thr_b2_coef, (  max_List_comb_thr_b2_size,ao_num,ao_num)]
&BEGIN_PROVIDER [ double precision, List_comb_thr_b2_expo, (  max_List_comb_thr_b2_size,ao_num,ao_num)]
&BEGIN_PROVIDER [ double precision, List_comb_thr_b2_cent, (3,max_List_comb_thr_b2_size,ao_num,ao_num)]
&BEGIN_PROVIDER [ double precision, ao_abs_comb_b2_env   , (  max_List_comb_thr_b2_size,ao_num,ao_num)]

  implicit none
  integer :: i_1s,i,j,ipoint,icount
  double precision :: coef,beta,center(3),int_env
  double precision :: r(3),weight,dist

  ao_abs_comb_b2_env = 10000000.d0
  do i = 1, ao_num
    do j = i, ao_num
      icount = 0
      do i_1s = 1, List_env1s_size
        coef        = List_env1s_coef  (i_1s)
        if(dabs(coef).lt.thrsh_cycle_tc)cycle
        beta        = List_env1s_expo  (i_1s)
        center(1:3) = List_env1s_cent(1:3,i_1s)
        int_env = 0.d0
        do ipoint = 1, n_points_extra_final_grid
          r(1:3) = final_grid_points_extra(1:3,ipoint)
          weight = final_weight_at_r_vector_extra(ipoint)
          dist  = ( center(1) - r(1) )*( center(1) - r(1) )
          dist += ( center(2) - r(2) )*( center(2) - r(2) )
          dist += ( center(3) - r(3) )*( center(3) - r(3) )
          int_env += dabs(aos_in_r_array_extra_transp(ipoint,i) * aos_in_r_array_extra_transp(ipoint,j))*dexp(-beta*dist) * weight
        enddo
        if(dabs(coef)*dabs(int_env).gt.thrsh_cycle_tc)then
          icount += 1
          List_comb_thr_b2_coef(icount,j,i) = coef
          List_comb_thr_b2_expo(icount,j,i) = beta
          List_comb_thr_b2_cent(1:3,icount,j,i) = center(1:3)
          ao_abs_comb_b2_env(icount,j,i) = int_env
        endif
      enddo
    enddo 
  enddo

  do i = 1, ao_num
    do j = 1, i-1
      do icount = 1, List_comb_thr_b2_size(j,i)
        List_comb_thr_b2_coef(icount,j,i) = List_comb_thr_b2_coef(icount,i,j)
        List_comb_thr_b2_expo(icount,j,i) = List_comb_thr_b2_expo(icount,i,j)
        List_comb_thr_b2_cent(1:3,icount,j,i) = List_comb_thr_b2_cent(1:3,icount,i,j)
      enddo
    enddo
  enddo
 
END_PROVIDER 

! ---

 BEGIN_PROVIDER [integer, List_comb_thr_b3_size, (ao_num,ao_num)]
&BEGIN_PROVIDER [integer, max_List_comb_thr_b3_size]

  implicit none
  integer :: i_1s,i,j,ipoint
 integer :: list(ao_num)
  double precision :: coef,beta,center(3),int_env
  double precision :: r(3),weight,dist

  List_comb_thr_b3_size = 0
  print*,'List_env1s_square_size = ',List_env1s_square_size
  do i = 1, ao_num
    do j = 1, ao_num
      do i_1s = 1, List_env1s_square_size
        coef        = List_env1s_square_coef  (i_1s)
        beta        = List_env1s_square_expo  (i_1s)
        center(1:3) = List_env1s_square_cent(1:3,i_1s)
        if(dabs(coef).lt.thrsh_cycle_tc)cycle
        int_env = 0.d0
        do ipoint = 1, n_points_extra_final_grid
          r(1:3) = final_grid_points_extra(1:3,ipoint)
          weight = final_weight_at_r_vector_extra(ipoint)
          dist  = ( center(1) - r(1) )*( center(1) - r(1) )
          dist += ( center(2) - r(2) )*( center(2) - r(2) )
          dist += ( center(3) - r(3) )*( center(3) - r(3) )
          int_env += dabs(aos_in_r_array_extra_transp(ipoint,i) * aos_in_r_array_extra_transp(ipoint,j))*dexp(-beta*dist) * weight
        enddo
        if(dabs(coef)*dabs(int_env).gt.thrsh_cycle_tc) then
          List_comb_thr_b3_size(j,i) += 1
        endif
      enddo
    enddo 
  enddo

  do i = 1, ao_num
    list(i) = maxval(List_comb_thr_b3_size(:,i))
  enddo

  max_List_comb_thr_b3_size = maxval(list) 
  print*, ' max_List_comb_thr_b3_size =  ',max_List_comb_thr_b3_size
 
END_PROVIDER 

! ---

 BEGIN_PROVIDER [double precision, List_comb_thr_b3_coef, (   max_List_comb_thr_b3_size,ao_num,ao_num)]
&BEGIN_PROVIDER [double precision, List_comb_thr_b3_expo, (   max_List_comb_thr_b3_size,ao_num,ao_num)]
&BEGIN_PROVIDER [double precision, List_comb_thr_b3_cent, (3, max_List_comb_thr_b3_size,ao_num,ao_num)]
&BEGIN_PROVIDER [double precision, ao_abs_comb_b3_env   , (   max_List_comb_thr_b3_size,ao_num,ao_num)]

  implicit none
  integer :: i_1s,i,j,ipoint,icount
  double precision :: coef,beta,center(3),int_env
  double precision :: r(3),weight,dist

  ao_abs_comb_b3_env = 10000000.d0
  do i = 1, ao_num
    do j = 1, ao_num
      icount = 0
      do i_1s = 1, List_env1s_square_size
        coef        = List_env1s_square_coef  (i_1s)
        beta        = List_env1s_square_expo  (i_1s)
        beta = max(beta,1.d-12)
        center(1:3) = List_env1s_square_cent(1:3,i_1s)
        if(dabs(coef).lt.thrsh_cycle_tc)cycle
        int_env = 0.d0
        do ipoint = 1, n_points_extra_final_grid
          r(1:3) = final_grid_points_extra(1:3,ipoint)
          weight = final_weight_at_r_vector_extra(ipoint)
          dist  = ( center(1) - r(1) )*( center(1) - r(1) )
          dist += ( center(2) - r(2) )*( center(2) - r(2) )
          dist += ( center(3) - r(3) )*( center(3) - r(3) )
          int_env += dabs(aos_in_r_array_extra_transp(ipoint,i) * aos_in_r_array_extra_transp(ipoint,j))*dexp(-beta*dist) * weight
        enddo
        if(dabs(coef)*dabs(int_env).gt.thrsh_cycle_tc)then
          icount += 1
          List_comb_thr_b3_coef(icount,j,i) = coef
          List_comb_thr_b3_expo(icount,j,i) = beta
          List_comb_thr_b3_cent(1:3,icount,j,i) = center(1:3)
          ao_abs_comb_b3_env(icount,j,i) = int_env
        endif
      enddo
    enddo 
  enddo

END_PROVIDER 

! ---

