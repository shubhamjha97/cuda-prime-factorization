# Prime factorization algorithm implemented in CUDA

This is a two pass implementation of the prime factorization algorithm. The passes do the following:
1. Use Sieve of Eratosthenes algorithm to find all print numbers less than sqrt(n).
2. For each prime number, check if it is a factor of n and how many times it divides n.

Space and time complexity: O(sqrt(n))

### Optimizations
- Used a bitset data structure for representing the sieve