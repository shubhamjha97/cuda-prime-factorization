#include <stdio.h>
#include <math.h>

void printUnsignedInteger(unsigned int x) {
    printf("%u ", x);
}

int main(int argc, char *argv[]) {
    unsigned int n;
    sscanf(argv[1], "%u", &n);

    while(!(n%2)) {
        printUnsignedInteger(2);
        n/=2;
    }

    int i_limit = ceil(sqrt(n));
    for(int i=3; i<=i_limit; i+=2) {
        while(!(n%i)) {
            printUnsignedInteger(i);
            n /= i;
        }
    }

    if(n>2) { printUnsignedInteger(n); }

    return 0;
}
