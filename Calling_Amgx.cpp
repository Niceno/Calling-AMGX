/*------------------------------------------------------------------------------
   I compiled this file with:

nvc++ Calling_Amgx.cpp -o Calling_Amgx           \
-I/home/niceno/Development/AMGX/include  \
/home/niceno/Development/AMGX/build/libamgx.a
------------------------------------------------------------------------------*/

#include <cstdio>

/*============================================================================*/
int main() {
/*----------------------------------------------------------------------------*/

  printf("Hello from Calling_Amgx!\n\n");
  printf("I do noting useful.  I am simply compiled with libamgx.a library\n");
  printf("which later even proved to be the wrong library to get linked to.\n");


  printf("\nGoodbye from Calling_Amgx.\n");

  return 0;
}
