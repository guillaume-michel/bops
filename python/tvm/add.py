import tvm
import numpy as np

# Global declarations of environment.

tgt_host="llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"

#n = tvm.convert(1024)
n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = tvm.create_schedule(C.op)

bx, tx = s[C].split(C.op.axis[0], factor=64)

print(tvm.lower(s, [A, B, C], simple_mode=True))

# fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")

# ctx = tvm.context(tgt, 0)

# n = 1024
# a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
# b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
# c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
# fadd(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

# print(fadd.get_source())
