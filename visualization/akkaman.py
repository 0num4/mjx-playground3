import sys
import time
# from numba import jit
# from python import numpy as np

# 再帰回数の上限を開放しておきます.
# sys.setrecursionlimit(200000000)


def ackermann(m: int, n: int) -> int:
    # print(np.random.rand())
    if m == 0:
        return n + 1
    elif n == 0:
        return ackermann(m - 1, 1)
    else:
        return ackermann(m - 1, ackermann(m, n - 1))


m = 3
n = 11

t0 = time.time()
result = ackermann(m, n)
t1 = time.time()

print(f"ackermann({m}, {n}) = {result}")
print(f"elapsed time: {int((t1 - t0) * 1000)} [ms]")