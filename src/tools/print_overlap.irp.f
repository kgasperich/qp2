program print_overlap
 implicit none
 BEGIN_DOC
 ! print ao overlap
 ! for debugging
 END_DOC
 call run
end

subroutine run
 implicit none
 integer :: i
 print*,'START overlap'
 do i=1,ao_num
   print '(1000E25.15)' , ao_overlap(i,:)
 enddo
 print*,'END overlap'
end
