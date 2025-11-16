/*------------------------------------------------------------------------------
   I compiled this file with:

nvc++ Calling_Amgx.cpp -o Calling_Amgx                                                    \
-I/home/niceno/Development/AMGX/include                                           \
-L/home/niceno/Development/AMGX/build -lamgxsh                                    \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib        \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib/  \
-lcublas -lcusparse -lcusolver -lnvJitLink

   To run in, you will also have to set:

export LD_LIBRARY_PATH=/home/niceno/Development/AMGX/build:/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
------------------------------------------------------------------------------*/
#include <cstdio>
#include <cstdlib>
#include <amgx_c.h>

/*============================================================================*/
static void check_amgx(AMGX_RC rc, const char *where) {
/*----------------------------------------------------------------------------*/

  if (rc != AMGX_RC_OK) {
    std::fprintf(stderr, "AMGX error in %s: rc = %d\n", where, rc);
    std::exit(EXIT_FAILURE);
  }
}

/*============================================================================*/
int main() {
/*----------------------------------------------------------------------------*/

  std::printf("Hello from Calling_Amgx!\n\n");
  std::printf("I go a step furhter from previously.  I am compiled with the\n");
  std::printf("same libraries as before, but I form a system matrix and\n");
  std::printf("right hand side and unknow vectors on the host.\n");
  std::printf("\nThe AMGX functions I am using are still the same:\n");
  std::printf("- AMGX_initialize\n");
  std::printf("- AMGX_get_api_version\n");
  std::printf("- AMGX_finalize\n\n");

  // Initialize AMGX core
  check_amgx(AMGX_initialize(), "AMGX_initialize");

  // Query API version
  int api_major = 0;
  int api_minor = 0;
  check_amgx(AMGX_get_api_version(&api_major, &api_minor),
             "AMGX_get_api_version");

  std::printf("AMGX API version: %d.%d\n", api_major, api_minor);

  // Create matrix and two vectors
  const int NX  = 10;
  const int NY  = 10;
  const int NZ  = 10;
  const int N   = NX * NY * NZ;
  const int OFF = -1;
  int neigh[6];
  int    * a_row;
  int    * a_col;
  int    * a_dia;
  double * a_val;
  double * x;
  double * b;

  // Count non-zero entries
  int cnt = 0;
  for(int i = 0; i < NX; i++)
    for(int j = 0; j < NY; j++)
      for(int k = 0; k < NZ; k++) {
        if(k > 0)    cnt++;
        if(j > 0)    cnt++;
        if(i > 0)    cnt++;
        cnt++;               // for diagonal
        if(i < NX-1) cnt++;
        if(j < NY-1) cnt++;
        if(k < NZ-1) cnt++;
      }

  // Allocate memory
  a_row = new int   [N+1];
  a_col = new int   [cnt];
  a_dia = new int   [N];
  a_val = new double[cnt];
  x     = new double[N];
  b     = new double[N];

  // Fill the matrix entries
  int pos = 0;
  for(int k = 0; k < NZ; k++)
    for(int j = 0; j < NY; j++)
      for(int i = 0; i < NX; i++) {

        // Central index
        int c = i + NX * (j + NY * k);

        // Work out neighbours from smallest to biggest
        neigh[0] = OFF;  if(k > 0)    neigh[0] = c - NX * NY;
        neigh[1] = OFF;  if(j > 0)    neigh[1] = c - NX;
        neigh[2] = OFF;  if(i > 0)    neigh[2] = c - 1;
        neigh[3] = OFF;  if(i < NX-1) neigh[3] = c + 1;
        neigh[4] = OFF;  if(j < NY-1) neigh[4] = c + NX;
        neigh[5] = OFF;  if(k < NZ-1) neigh[5] = c + NX * NY;

        // Initialize valud of diagonal entry
        double diag = 0.0;

        // Store the pointer to the beginning of a row.
        // (First time it reaches this line, c == 0 and
        // pos == 0 meaning initialization will be fine)
        a_row[c] = pos;

        // Browse through neighbours smaller than c
        for(int n = 0; n <= 2; n++)
          if(neigh[n] != OFF) {
            a_col[pos] = neigh[n];
            a_val[pos] = -1.0;
            diag += 1.0;
            pos = pos + 1;
          }
        // Central position
        a_col[pos] = c;
        a_dia[c]   = pos;
        pos        = pos + 1;
        // Browse through neighbours larger than c
        for(int n = 3; n <= 5; n++)
          if(neigh[n] != OFF) {
            a_col[pos] = neigh[n];
            a_val[pos] = -1.0;
            diag += 1.0;
            pos = pos + 1;
          }
        // Update the main diagonal
        a_val[a_dia[c]] = diag;

  }

  // Final entry in a_row
  a_row[N] = pos;

  // Free the memory
  delete [] a_row;
  delete [] a_col;
  delete [] a_dia;
  delete [] a_val;
  delete [] x;
  delete [] b;

  // Finalize
  check_amgx(AMGX_finalize(), "AMGX_finalize");

  std::printf("\nGoodbye from Calling_Amgx.\n");

  return 0;
}

