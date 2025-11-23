set -x

# Compile little CUDA C helpers
nvc++ -c Cuda_C_Helpers.cpp -o cuda_c_helpers.o -acc -I/home/niceno/Development/AMGX/include

# Compile Fortran interface to AMGX
nvc++ -c Calling_Amgx.cpp -o fortran_amgx_interface.o -acc -I/home/niceno/Development/AMGX/include

# Compile the main program in Fortran
nvfortran Main_Call_Amgx.f90 -o Process cuda_c_helpers.o fortran_amgx_interface.o -acc -c++libs -I/home/niceno/Development/AMGX/include -L/home/niceno/Development/AMGX/build -L/$NVHPC_ROOT/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib -L/$NVHPC_ROOT/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib -lcublas -lcusparse -lcusolver -lnvJitLink -lcudart -lamgxsh

# To run it, you will also have to set:
# export LD_LIBRARY_PATH=/home/niceno/Development/AMGX/build:/$NVHPC_ROOT/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib:/$NVHPC_ROOT/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
