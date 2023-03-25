#include <stdio.h>

#include "functions.h"

int main()
{

  /* Test 1: print_bits */

  int x;
  printf("Enter an integer >");
  scanf("%d", &x);
  print_bits(x);

  /* Test 2: int_multiply */

  int y, z;
  printf("Enter two integers (to multiply) > ");
  scanf("%d %d", &y, &z);

  printf("%d * %d = %ld\n", y, z, int_multiply(y,z));
  printf("Checking, result should be %ld\n", ((long) y) * ((long) z));

  /* Test 3: float_multiply. */

  float a, b;
  printf("Enter two floating point numbers(to multiply) > ");
  scanf("%f %f", &a, &b);

  printf("%f * %f = %f\n", a, b, float_multiply(a,b));
  printf("Checking, result should be %f\n", a*b);

  /* Test 4: float_multiply. */

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

