#include <stdio.h>

#include "functions.h"

int main()
{

  /* SECTION 1
     Uncomment the code below in this section once you have written print_bits.
  */

  int x;
  printf("Enter an integer >");
  scanf("%d", &x);
  print_bits(x);

  /* SECTION 2
     Uncomment the code below in this section once you have written int_multiply.
  */

  int y, z;
  printf("Enter two integers (to multiply) > ");
  scanf("%d %d", &y, &z);

  printf("%d * %d = %ld\n", y, z, int_multiply(y,z));
  printf("Checking, result should be %ld\n", ((long) y) * ((long) z));

  /* SECTION 3
     Uncomment the code below in this section once you have written float_multiply.
  */

  float a, b;
  printf("Enter two floating point numbers(to multiply) > ");
  scanf("%f %f", &a, &b);

  printf("%f * %f = %f\n", a, b, float_multiply(a,b));
  printf("Checking, result should be %f\n", a*b);

  /* SECTION 4
     Uncomment the code below in this section once you have written float_multiply.
  */

  #include "exponent.h"

  int p, q, r;

  printf("Enter three integers (for exponent calculation) >");
  scanf("%d %d %d", &p, &q, &r);
  printf("%d^%d + %d^%d = %d\n", p, r, q, r, exponent(p,q,r));

  int prodp = 1;
  int prodq = 1;
  int i;
  for(i=0; i < r; i++) {
    prodp *= p;
    prodq *= q;
  }
  printf("Checking, result should be %d\n", prodp + prodq);
}

