/*------------------------------------------------------------------------------
   I compiled this file with:

nvc++ Calling_Amgx.cpp -o Calling_Amgx                                                    \
-I/home/niceno/Development/AMGX/include                                           \
-L/home/niceno/Development/AMGX/build -lamgxsh                                    \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib        \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib/  \
-lcublas -lcusparse -lcusolver -lnvJitLink

To run it, you will also have to set:

export LD_LIBRARY_PATH=/home/niceno/Development/AMGX/build:                       \
/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib:    \
/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib:         \
$LD_LIBRARY_PATH
------------------------------------------------------------------------------*/
#include <cstdio>
#include <cstdlib>
#include <amgx_c.h>

/*============================================================================*/
static void check_amgx(AMGX_RC rc, const char *where) {
/*----------------------------------------------------------------------------*/

  if (rc != AMGX_RC_OK) {
    fprintf(stderr, "AMGX error in %s: rc = %d\n", where, rc);
    exit(EXIT_FAILURE);
  }
}

/*============================================================================*/
int main() {
/*----------------------------------------------------------------------------*/

  printf("Hello from Calling_Amgx!\n");
  printf("I go an important small step furhter from Step_006 and in\n");
  printf("addition to copying the system (matrix and vectors) to the\n");
  printf("device, I also solve them.\n");
  printf("\nThe AMGX functions I am using are:\n");
  printf("- AMGX_initialize\n");
  printf("- AMGX_get_api_version\n");
  printf("- AMGX_config_create\n");
  printf("- AMGX_resources_create_simple\n");
  printf("- AMGX_matrix_create\n");
  printf("- AMGX_vector_create\n");
  printf("- AMGX_matrix_upload_all\n");
  printf("- AMGX_vector_upload\n");
  printf("- AMGX_solver_create                (new)\n");
  printf("- AMGX_solver_setup                 (new)\n");
  printf("- AMGX_solver_solve                 (new)\n");
  printf("- AMGX_solver_get_status            (new)\n");
  printf("- AMGX_solver_get_iterations_number (new)\n");
  printf("- AMGX_vector_download              (new)\n");
  printf("- AMGX_solver_destroy               (new)\n");
  printf("- AMGX_matrix_destroy\n");
  printf("- AMGX_vector_destroy\n");
  printf("- AMGX_resources_destroy\n");
  printf("- AMGX_config_destroy\n");
  printf("- AMGX_finalize\n\n");
  printf("and these AMGX's data types:\n");
  printf("- AMGX_config_handle\n");
  printf("- AMGX_resources_handle\n");
  printf("- AMGX_matrix_handle\n");
  printf("- AMGX_vector_handle\n");
  printf("- AMGX_solver_handle               (new)\n\n");

  /////////////////////////////
  //                         //
  //   AMGX Initialization   //
  //                         //
  /////////////////////////////

  // Initialize AMGX core
  check_amgx(AMGX_initialize(), "AMGX_initialize");

  // Query API version
  int api_major = 0;
  int api_minor = 0;
  check_amgx(AMGX_get_api_version(&api_major, &api_minor),
             "AMGX_get_api_version");

  printf("AMGX API version: %d.%d\n", api_major, api_minor);

  ///////////////////////////////////////////////////////
  //                                                   //
  //   Create linear system of equations on the host   //
  //                                                   //
  ///////////////////////////////////////////////////////

  // Create matrix and two vectors
  const int NX  = 100;
  const int NY  = 100;
  const int NZ  = 100;
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

        // Initialize value of diagonal entry
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

  // Fill the vector entries
  for (int k = 0; k < NZ; k++)
    for (int j = 0; j < NY; j++)
      for (int i = 0; i < NX; i++) {
        // Central index
        int c = i + NX * (j + NY * k);
        x[c] = 0.0;
        b[c] = 0.0;
        if(i == 0)    b[c] = -0.1;
        if(i == NX-1) b[c] = +0.1;
      }

  ///////////////////////////////////////
  //                                   //
  //   Copy system to AMGX workspace   //
  //                                   //
  ///////////////////////////////////////
  AMGX_resources_handle rsrc  = nullptr;
  AMGX_config_handle    cfg   = nullptr;
  AMGX_matrix_handle    A_dev = nullptr;
  AMGX_vector_handle    x_dev = nullptr;
  AMGX_vector_handle    b_dev = nullptr;
  AMGX_solver_handle    slv   = nullptr;

  // Simple JSON config: PCG + Jacobi
  const char *cfg_str =
    "{\n"
    "  \"config_version\": 2,\n"
    "  \"solver\": {\n"
    "    \"solver\": \"PCG\",\n"
    "    \"preconditioner\": {\n"
    "      \"solver\": \"BLOCK_JACOBI\"\n"
    "    },\n"
    "    \"max_iters\": 100,\n"
    "    \"tolerance\": 1e-6,\n"
    "    \"norm\": \"L2\",\n"
    "    \"monitor_residual\": 1,\n"
    "    \"print_solve_stats\": 1,\n"
    "    \"obtain_timings\": 1\n"
    "  }\n"
    "}";

  check_amgx(AMGX_config_create(&cfg, cfg_str), "AMGX_config_create");

  // Simple resources: single GPU, device 0
  check_amgx(AMGX_resources_create_simple(&rsrc, cfg),
             "AMGX_resources_create_simple");

  // Create matrix and vectors on device
  const AMGX_Mode mode = AMGX_mode_dDDI;
  check_amgx(AMGX_matrix_create(&A_dev, rsrc, mode),
             "AMGX_matrix_create");

  // Create device vectors (same mode as matrix)
  check_amgx(AMGX_vector_create(&x_dev, rsrc, mode),
             "AMGX_vector_create(x_dev)");
  check_amgx(AMGX_vector_create(&b_dev, rsrc, mode),
             "AMGX_vector_create(b_dev)");

  // Upload CSR matrix to AMGX (device)
  check_amgx(AMGX_matrix_upload_all(
               A_dev, N, cnt, 1, 1, a_row, a_col, a_val, nullptr),
               "AMGX_matrix_upload_all");

  printf("Matrix uploaded to AMGX.\n");

  // Upload host vectors to device
  check_amgx(AMGX_vector_upload(x_dev, N, 1, x), "AMGX_vector_upload(x_dev)");
  check_amgx(AMGX_vector_upload(b_dev, N, 1, b), "AMGX_vector_upload(b_dev)");
  printf("Vectors x and b uploaded to AMGX.\n");

  ///////////////////////////////////
  //                               //
  //   Create and run the solver   //
  //                               //
  ///////////////////////////////////

  check_amgx(AMGX_solver_create(&slv, rsrc, mode, cfg),
             "AMGX_solver_create");

  check_amgx(AMGX_solver_setup(slv, A_dev),
             "AMGX_solver_setup");

  check_amgx(AMGX_solver_solve(slv, b_dev, x_dev),
             "AMGX_solver_solve");

  AMGX_SOLVE_STATUS status;
  check_amgx(AMGX_solver_get_status(slv, &status),
             "AMGX_solver_get_status");
  printf("Solve status: %d (0=SUCCESS)\n", status);

  int iters = 0;
  check_amgx(AMGX_solver_get_iterations_number(slv, &iters),
             "AMGX_solver_get_iterations_number");
  printf("Number of iterations: %d\n", iters);

  // Download solution back to host
  double *x_sol = new double[N];
  check_amgx(AMGX_vector_download(x_dev, x_sol),
             "AMGX_vector_download");

  printf("First 10 solution entries:\n");
  for (int i = 0; i < 10 && i < N; i++) {
    printf("  x[%d] = %e\n", i, x_sol[i]);
  }

  delete [] x_sol;

  // Free the memory
  delete [] a_row;
  delete [] a_col;
  delete [] a_dia;
  delete [] a_val;
  delete [] x;
  delete [] b;

  ///////////////////////////
  //                       //
  //   AMGX Finalization   //
  //                       //
  ///////////////////////////

  // Destroy AMGX objects
  check_amgx(AMGX_solver_destroy(slv),      "AMGX_solver_destroy");
  check_amgx(AMGX_matrix_destroy(A_dev),    "AMGX_matrix_destroy");
  check_amgx(AMGX_vector_destroy(x_dev),    "AMGX_vector_destroy");
  check_amgx(AMGX_vector_destroy(b_dev),    "AMGX_vector_destroy");
  check_amgx(AMGX_resources_destroy(rsrc),  "AMGX_resources_destroy");
  check_amgx(AMGX_config_destroy(cfg),      "AMGX_config_destroy");

  // Finalize
  check_amgx(AMGX_finalize(), "AMGX_finalize");

  printf("\nGoodbye from Calling_Amgx.\n");

  return 0;
}

