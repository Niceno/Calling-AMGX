#include <cstdio>
#include <amgx_c.h>
#include <cuda_runtime.h>

/*============================================================================*/
static void check_amgx(AMGX_RC rc, const char * where) {
/*----------------------------------------------------------------------------*/

  if(rc != AMGX_RC_OK) {
    fprintf(stderr, "AMGX error in %s: rc = %d\n", where, rc);
    exit(EXIT_FAILURE);
  }
}

/*============================================================================*/
extern "C" int call_amgx_(const int    & N,
                          const int    & nnz,
                          void **      a_row_dev_ptr,         // size: N+1
                          void **      a_col_dev_ptr,         // size: nnz
                          void **      a_val_dev_ptr,         // size: nnz
                          void **      x_dev_ptr,             // size: N
                          void **      b_dev_ptr) {           // size: N
/*----------------------------------------------------------------------------*/

  printf("Hello from Calling_Amgx!\n\n");
  printf("Everything works as it supposed to.  Fortan side has everything\n");
  printf("on the device and passes it all to C++ for solving it.\n");
  printf("\nThe AMGX functions I am using are:\n");
  printf("- AMGX_initialize\n");
  printf("- AMGX_get_api_version\n");
  printf("- AMGX_config_create\n");
  printf("- AMGX_resources_create_simple\n");
  printf("- AMGX_matrix_create\n");
  printf("- AMGX_vector_create\n");
  printf("- AMGX_matrix_upload_all\n");
  printf("- AMGX_vector_upload\n");
  printf("- AMGX_solver_create\n");
  printf("- AMGX_solver_setup\n");
  printf("- AMGX_solver_solve\n");
  printf("- AMGX_solver_get_status\n");
  printf("- AMGX_solver_get_iterations_number\n");
  printf("- AMGX_vector_download\n");
  printf("- AMGX_solver_destroy\n");
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
  printf("- AMGX_solver_handle\n\n");

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

  /////////////////////////////////////////////////////////////////////////
  //                                                                     //
  //   Connect to the linear system of equations from the Fortran side   //
  //                                                                     //
  /////////////////////////////////////////////////////////////////////////

  // Connect the local poitners to what you have received from the Fortran side
  int    * a_row = static_cast <int *>    (* a_row_dev_ptr);
  int    * a_col = static_cast <int *>    (* a_col_dev_ptr);
  double * a_val = static_cast <double *> (* a_val_dev_ptr);
  double * x     = static_cast <double *> (* x_dev_ptr);
  double * b     = static_cast <double *> (* b_dev_ptr);

  // Correct the indices of integer-based matrix pointers
  #pragma acc parallel loop deviceptr(a_row)
  for(int c = 0; c <= N; c++) {
    a_row[c]--;
  }
  #pragma acc parallel loop deviceptr(a_col)
  for(int i = 0; i < nnz; i++) {
    a_col[i]--;
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
               A_dev, N, nnz, 1, 1, a_row, a_col, a_val, nullptr),
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
  cudaFree(a_row);
  cudaFree(a_col);
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

