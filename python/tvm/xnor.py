import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime
import topi
import numpy as np
import time
import matplotlib.pyplot as plt

repeat = 10
fac_range = 32
fac_stride = 2

tgt_host="llvm"
tgt="llvm"

# batch
n = tvm.var("n")
# bitplane
b = tvm.var("b")
# input neurones/64
chw = tvm.var("chw")
# output neurones
m = tvm.var("m")

X = tvm.placeholder((n, b, chw,), dtype='uint64', name='X')
W = tvm.placeholder((b, m, chw,), dtype='uint64', name='W')
ichw = tvm.reduce_axis((0, chw), name="ichw")

Y = tvm.compute((n, b, m,),
                lambda nn, bb, mm: tvm.sum(tvm.popcount(X[nn, bb, ichw] ^ W[bb, mm, ichw]),
                                           axis=ichw),
                name="Y")

data_x = np.zeros(fac_range//fac_stride)
data_y = np.zeros(fac_range//fac_stride)

for counter, fac in enumerate(range(2, fac_range + fac_stride, fac_stride)):
    print('-'*80)
    s = tvm.create_schedule(Y.op)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    mo, mi = s[Y].split(Y.op.axis[2], factor=fac)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    s[Y].reorder(Y.op.axis[0], Y.op.axis[1], mo, ichw, mi)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    s[Y].unroll(mi)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    s[Y].parallel(Y.op.axis[0])
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    # fused = s[Y].fuse(Y.op.axis[0], Y.op.axis[1])
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    # s[Y].parallel(fused)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    # bx, tx = s[Y].split(Y.op.axis[0], factor=64)
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    # fused = s[Y].fuse(Y.op.axis[0], Y.op.axis[1], Y.op.axis[2])
    # print(tvm.lower(s, [X, W, Y], simple_mode=True))

    # print('-'*80)
    # with tvm.target.create("llvm -mattr=+sse,+sse2,+sse3,+sse4"):
    #     sg = topi.generic.schedule_reduce(Y)
    #     print(tvm.lower(sg, [X, W, Y], simple_mode=True))

    dense = tvm.build(s, [X, W, Y], tgt, target_host=tgt_host, name="dense")

    ctx = tvm.cpu()

    n_dim = 60000
    b_dim = 8
    chw_dim = 16
    m_dim = 128
    x_nd = np.random.uniform(size=(n_dim,b_dim,chw_dim)).astype(X.dtype)
    w_nd = np.random.uniform(size=(b_dim,m_dim,chw_dim)).astype(W.dtype)
    y_nd = np.ones((n_dim,b_dim,m_dim), dtype=Y.dtype)
    x = tvm.nd.array(x_nd, ctx)
    w = tvm.nd.array(w_nd, ctx)
    y = tvm.nd.array(y_nd, ctx)

    start = time.time()
    for _ in range(repeat):
        dense(x, w, y)
    end = time.time()

    print('dense execution time:', (end-start)/repeat, 's for fac: ', fac)
    data_x[counter] = fac
    data_y[counter] = (end-start)/repeat

fig = plt.figure()
plt.plot(data_x, data_y)
plt.show()

#print(dense.get_source())
