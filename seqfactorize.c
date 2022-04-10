#include <stdio.h>
#include <math.h>

void output(unsigned int x) {
    printf("%d ", x);
}

int main(int argc, char *argv[]) {
    unsigned int n;
    sscanf(argv[1], "%d", &n);

    while(!(n%2)) {
        output(2);
        n/=2;
    }

    int i_limit = ceil(sqrt(n));
    for(int i=3; i<=i_limit; i+=2) {
        while(!(n%i)) {
            output(i);
            n /= i;
        }
    }

    if(n>2) { output(n); }

    return 0;
}
