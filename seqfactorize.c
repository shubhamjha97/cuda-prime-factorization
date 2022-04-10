#include <stdio.h>
#include <math.h>

void printInteger(unsigned int x) {
    printf("%d ", x);
}

int main(int argc, char *argv[]) {
    unsigned int n;
    sscanf(argv[1], "%d", &n);

    while(!(n%2)) {
        printInteger(2);
        n/=2;
    }

    int i_limit = ceil(sqrt(n));
    for(int i=3; i<=i_limit; i+=2) {
        while(!(n%i)) {
            printInteger(i);
            n /= i;
        }
    }

    if(n>2) { printInteger(n); }

    return 0;
}
