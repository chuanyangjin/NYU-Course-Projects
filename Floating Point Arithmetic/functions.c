#include <stdio.h> // to use printf and scanf
#define MASK8 0xFF
#define MASK23 0x7FFFFF

void print_bits(int x)
{
    // iterate i from the index of the leftmost bit
    // (in this case 31) down to 0
    for (int i = (sizeof(int) * 8) - 1; i >= 0; i--) {
        // determine the value of the ith bit of x by shifting the mask
        // the mask to left by i and bit wise-AND'ing with 1.
        if (x & (0x1 << i)) {
            printf("1");
        }
        else {
            printf("0");
        }
    }
}

long int_multiply(int x, int y) {
    long lx = (long)x;
    long ly = (long)y;
    long r = 0;
    for (int i = 0; i <= (sizeof(long) * 8) - 1; i++) {
        // determine the value of the ith bit of x by shifting the mask
        // the mask to left by i and bit wise-AND'ing with 1.
        if ((ly >> i) & (long)1) {
            r = r + (lx << i);
        }
    }
    return r;
}

float float_multiply(float a, float b) {
    unsigned int val_a = *((unsigned int*)&a);
    unsigned int val_b = *((unsigned int*)&b);
    if ((val_a == 0) || (val_b == 0)) {
        return 0.0;
    }

    // extract the exponent
    unsigned int a_eb = (val_a >> 23) & MASK8;
    unsigned int b_eb = (val_b >> 23) & MASK8;

    // extract the mantissa
    unsigned int a_mb = val_a & MASK23;
    unsigned int b_mb = val_b & MASK23;

    // compute the exponent
    unsigned int exponent = a_eb + b_eb - 127;

    // compute the mantissa
    unsigned int a_m = ((unsigned int)1 << 23) | a_mb;
    unsigned int b_m = ((unsigned int)1 << 23) | b_mb;
    unsigned long ml = (unsigned long)int_multiply((unsigned long)a_m, (unsigned long)b_m);
    ml = ml >> 23;
    if ((ml >> 24) & 1) {
        ml = ml >> 1;
        exponent = exponent + 1;
    }
    unsigned int m = (unsigned int)ml;

    // extract and compute the sign
    unsigned int sign = ((val_a >> 31) & 1) ^ ((val_b >> 31) & 1);

    // construct the result
    unsigned int result = m & MASK23;
    result = result | (exponent << 23);
    result = result | (sign << 31);
    return *((float*)&result);
}