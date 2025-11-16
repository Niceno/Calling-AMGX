/*------------------------------------------------------------------------------
   I compiled this file with:

nvc++ Step_001.cpp -o Step_001           \
-I/home/niceno/Development/AMGX/include  \
/home/niceno/Development/AMGX/build/libamgx.a
------------------------------------------------------------------------------*/

#include <cstdio>

/*============================================================================*/
int main() {
/*----------------------------------------------------------------------------*/

  printf("Hello from Step_001!\n\n");
  printf("I do noting useful.  I am simply compiled with libamgx.a library\n");
  printf("which later even proved to be the wrong library to get linked to.\n");


  printf("\nGoodbye from Step_002.\n");

  return 0;
}
