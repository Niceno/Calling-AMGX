/*------------------------------------------------------------------------------
   I compiled this file with:

nvc++ Calling_Amgx.cpp -o Calling_Amgx                                            \
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
  std::printf("I go a step furhter from Step_001.  I am compiled with AMGX\n");
  std::printf("library libamgxsh and, in addition, to all other libraries it\n");
  std::printf("depends on which includes cublas, cusparse cusolver and\n");
  std::printf("nvJitLink.\n");
  std::printf("\nI also initialize, get API version and finalize AMGX.\n");
  std::printf("\nThe AMGX functions I am using are:\n");
  std::printf("- AMGX_initialize      (new)\n");
  std::printf("- AMGX_get_api_version (new)\n");
  std::printf("- AMGX_finalize        (new)\n\n");

  // Initialize AMGX core
  check_amgx(AMGX_initialize(), "AMGX_initialize");

  // Query API version
  int api_major = 0;
  int api_minor = 0;
  check_amgx(AMGX_get_api_version(&api_major, &api_minor),
             "AMGX_get_api_version");

  std::printf("AMGX API version: %d.%d\n", api_major, api_minor);

  // Finalize
  check_amgx(AMGX_finalize(), "AMGX_finalize");

  std::printf("\nGoodbye from Calling_Amgx.\n");

  return 0;
}

