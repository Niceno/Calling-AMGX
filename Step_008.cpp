/*------------------------------------------------------------------------------
   I compiled this file with:

nvc++ Calling_Amgx.cpp -o Calling_Amgx -acc                                               \
-I/home/niceno/Development/AMGX/include                                           \
-L/home/niceno/Development/AMGX/build -lamgxsh                                    \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib        \
-L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib/  \
-lcublas -lcusparse -lcusolver -lnvJitLink -lcudart

To run it, you will also have to set:

export LD_LIBRARY_PATH=/home/niceno/Development/AMGX/build:                       \
/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/math_libs/12.3/targets/x86_64-linux/lib:    \
/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/12.3/targets/x86_64-linux/lib:         \
$LD_LIBRARY_PATH
------------------------------------------------------------------------------*/
#include <cstdio>
#include <cstdlib>
#include <amgx_c.h>
#include <cuda_runtime.h>

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
  int    * a_row_host;
  int    * a_col_host;
  int    * a_dia_host;
  double * a_val_host;
  int    * a_row;
  int    * a_col;
  int    * a_dia;
  double * a_val;
  double * x;
  double * b;

  // Count non-zero entries
  int cnt = N + 2 * (  (NX - 1) * NY * NZ
                     + NX * (NY - 1) * NZ
                     + NX * NY * (NZ - 1)  );

  // Allocate memory
  a_row_host = new int   [N+1];
  a_col_host = new int   [cnt];
  a_dia_host = new int   [N];
  a_val_host = new double[cnt];
  cudaMalloc(&a_row, (N+1) * sizeof(int));
  cudaMalloc(&a_col,  cnt  * sizeof(int));
  cudaMalloc(&a_dia,  N    * sizeof(int));
  cudaMalloc(&a_val,  cnt  * sizeof(double));
  cudaMalloc(&x,      N    * sizeof(double));
  cudaMalloc(&b,      N    * sizeof(double));

  // Fill the matrix entries
  int pos = 0;
  for(int c = 0; c < N; c++) {

    int k = c / (NX * NY);
    int r = c % (NX * NY);
    int j = r / NX;
    int i = r % NX;

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
    a_row_host[c] = pos;

    // Browse through neighbours smaller than c
    for(int n = 0; n <= 2; n++)
      if(neigh[n] != OFF) {
        a_col_host[pos] = neigh[n];
        a_val_host[pos] = -1.0;
        diag += 1.0;
        pos = pos + 1;
      }

    // Central position
    a_col_host[pos] = c;
    a_dia_host[c]   = pos;
    pos             = pos + 1;

    // Browse through neighbours larger than c
    for(int n = 3; n <= 5; n++)
      if(neigh[n] != OFF) {
        a_col_host[pos] = neigh[n];
        a_val_host[pos] = -1.0;
        diag += 1.0;
        pos = pos + 1;
      }

    // Update the main diagonal
    a_val_host[a_dia_host[c]] = diag;
  }

  // Final entry in a_row
  a_row_host[N] = pos;

  // Copy the entire matrix to the device
  cudaMemcpy(a_row, a_row_host, (N+1) * sizeof(int),    cudaMemcpyHostToDevice);
  cudaMemcpy(a_col, a_col_host,  cnt  * sizeof(int),    cudaMemcpyHostToDevice);
  cudaMemcpy(a_dia, a_dia_host,  N    * sizeof(int),    cudaMemcpyHostToDevice);
  cudaMemcpy(a_val, a_val_host,  cnt  * sizeof(double), cudaMemcpyHostToDevice);

  // Fill the vector x
  #pragma acc parallel loop deviceptr(x)
  for(int c = 0; c < N; c++) {
    x[c] = 0.0;
  }

  // Fill the vector b
  #pragma acc parallel loop deviceptr(b)
  for (int c = 0; c < N; ++c) {
    int r = c % (NX * NY);
    int i = r % NX;

    double val = 0.0;
    if(i == 0)     val = -0.1;
    if(i == NX-1)  val = +0.1;

    b[c] = val;
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
  check_amgx(AMGX_matrix_create(&A_dev, rsrc, mode), "AMGX_matrix_create");

  // Create device vectors (same mode as matrix)
  check_amgx(AMGX_vector_create(&x_dev, rsrc, mode), "AMGX_vector_create(x)");
  check_amgx(AMGX_vector_create(&b_dev, rsrc, mode), "AMGX_vector_create(b)");

  // Upload CSR matrix to AMGX (device)
  check_amgx(AMGX_matrix_upload_all(
               A_dev, N, cnt, 1, 1, a_row, a_col, a_val, nullptr),
               "AMGX_matrix_upload_all");

  printf("Matrix uploaded to AMGX.\n");

  // Upload host vectors to device
  check_amgx(AMGX_vector_upload(x_dev, N, 1, x), "AMGX_vector_upload(x)");
  check_amgx(AMGX_vector_upload(b_dev, N, 1, b), "AMGX_vector_upload(b)");
  printf("Vectors x and b uploaded to AMGX.\n");

  ///////////////////////////////////
  //                               //
  //   Create and run the solver   //
  //                               //
  ///////////////////////////////////

  check_amgx(AMGX_solver_create(&slv, rsrc, mode, cfg), "AMGX_solver_create");
  check_amgx(AMGX_solver_setup(slv, A_dev),             "AMGX_solver_setup");
  check_amgx(AMGX_solver_solve(slv, b_dev, x_dev),      "AMGX_solver_solve");

  AMGX_SOLVE_STATUS status;
  check_amgx(AMGX_solver_get_status(slv, &status), "AMGX_solver_get_status");
  printf("Solve status: %d (0=SUCCESS)\n", status);

  int iters = 0;
  check_amgx(AMGX_solver_get_iterations_number(slv, &iters),
             "AMGX_solver_get_iterations_number");
  printf("Number of iterations: %d\n", iters);

  // Download solution back to host
  double * x_sol = new double[N];
  check_amgx(AMGX_vector_download(x_dev, x_sol),
             "AMGX_vector_download");

  printf("First 10 solution entries:\n");
  for (int i = 0; i < 10 && i < N; i++) {
    printf("  x[%d] = %e\n", i, x_sol[i]);
  }

  delete [] x_sol;

  // Free the memory
  delete [] a_row_host;
  delete [] a_col_host;
  delete [] a_dia_host;
  delete [] a_val_host;
  cudaFree(a_row);
  cudaFree(a_col);
  cudaFree(a_dia);
  cudaFree(a_val);
  cudaFree(x);
  cudaFree(b);

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

