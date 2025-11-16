!==============================================================================!
  program Main_Call_Amgx
!------------------------------------------------------------------------------!
!   Compilation is explained in C++ source
!------------------------------------------------------------------------------!
!----------------------------------[Modules]-----------------------------------!
  use iso_c_binding
!------------------------------------------------------------------------------!
  implicit none
!------------------------------[Local parameters]------------------------------!
  integer, parameter :: NX  = 100
  integer, parameter :: NY  = 100
  integer, parameter :: NZ  = 100
  integer, parameter :: N   = NX * NY * NZ
  integer, parameter :: OFF = -1
!-----------------------------------[Locals]-----------------------------------!
  integer, allocatable :: a_row_host(:)
  integer, allocatable :: a_col_host(:)
  integer, allocatable :: a_dia_host(:)
  real(8), allocatable :: a_val_host(:)
  real(8)              :: diag
  integer              :: neigh(6)
  integer              :: i, j, k, r, c, i_cel, nnz, pos
!==============================================================================!

  ! Count non-zero entries
  nnz = N + 2 * (  (NX - 1) * NY * NZ     &
                 + NX * (NY - 1) * NZ     &
                 + NX * NY * (NZ - 1)  )

  ! Allocate memory for host arrays
  allocate(a_row_host(N+1));
  allocate(a_col_host(nnz));
  allocate(a_dia_host(N));
  allocate(a_val_host(nnz));

  ! Fill the matrix entries
  pos = 1
  do c = 1, N

    k = (c-1) / (NX * NY) + 1
    r = mod(c-1, NX * NY)
    j = r / NX            + 1
    i = mod(r, NX)        + 1

    ! Work out neighbours from smallest to biggest
    neigh(1) = OFF;  if(k > 1)  neigh(1) = c - NX * NY
    neigh(2) = OFF;  if(j > 1)  neigh(2) = c - NX
    neigh(3) = OFF;  if(i > 1)  neigh(3) = c - 1
    neigh(4) = OFF;  if(i < NX) neigh(4) = c + 1
    neigh(5) = OFF;  if(j < NY) neigh(5) = c + NX
    neigh(6) = OFF;  if(k < NZ) neigh(6) = c + NX * NY

    ! Initialize value of diagonal entry
    diag = 0.0;

    ! Store the pointer to the beginning of a row.
    ! (First time it reaches this line, c .eq. 1 and
    ! pos .eq. 1 meaning initialization will be fine)
    a_row_host(c) = pos;

    ! Browse through neighbours smaller than c
    do i_cel = 1, 3
      if(neigh(i_cel) .ne. OFF) then
        a_col_host(pos) = neigh(i_cel)
        a_val_host(pos) = -1.0
        diag = diag + 1.0
        pos = pos + 1
      end if
    end do

    ! Central position
    a_col_host(pos) = c
    a_dia_host(c)   = pos
    pos             = pos + 1

    ! Browse through neighbours larger than c
    do i_cel = 4, 6
      if(neigh(i_cel) .ne. OFF) then
        a_col_host(pos) = neigh(i_cel)
        a_val_host(pos) = -1.0
        diag = diag + 1.0
        pos = pos + 1
      end if
    end do

    ! Update the main diagonal
    a_val_host(a_dia_host(c)) = diag

  end do

  ! Final entry in a_row
  a_row_host(N+1) = pos

  call call_amgx(NX, NY, NZ, N, nnz,  &
                 a_row_host, a_col_host, a_dia_host)

  end program
