import tensorflow as tf
import numpy as np
import time
from numpy.random import randn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline

# Author: Dougal J. Sutherland
def cholesky_unblocked(X, size = None):
    if size is None:
        # need to have an explicit shape for X
        shape = X.get_shape()
        shape.assert_has_rank(2)
        m = shape[0].value
        assert m is not None
        assert shape[1].value == m
    else:
        m = size

    A = tf.matrix_band_part(X, -1, 0)
    cond = lambda j, A: j < m
    def body(j, A):
        s = A[j, :j]
        s = A[j, :j]
        s2 = tf.matmul(tf.expand_dims(s, 0), tf.expand_dims(s, 1))[0, 0]
        Ajj = tf.sqrt(A[j, j] - s2)

        col = A[j+1:m, j]
        col -= tf.matmul(A[j+1:m, :j], tf.expand_dims(A[j, :j], 1))[:, 0]
        col /= Ajj

        A = tf.concat(0, [
            A[:j],
            tf.concat(1, [
                A[j:, :j],
                tf.expand_dims(tf.concat(0, [tf.expand_dims(Ajj, 0), col]), 1),
                A[j:, j+1:]
            ])])
        A.set_shape([m, m])
        return j + 1, A
    return tf.while_loop( cond, body, (tf.constant(0, tf.int32), A))[1]


def add_top_padding(X, n, m):
    """
        Adds n rows of padding to the top of X.
        m equals the number of columns of X.
    """
    padding = tf.zeros((n, m), dtype=X.dtype)
    return tf.concat(0, [padding, X])



# Author: Vincent Dutordoir
def cholesky_blocked(A, nb = 200):
    """
        nb - block size, A.shape[0] % nb == 0
    """
    n = A.get_shape()[0].value
    j = 0
    # make A a lower triangular matrix
    A = tf.matrix_band_part(A, -1, 0)

    def body(A, j):
        A11 = tf.slice(A, [j, j], [nb, nb])
        A21 = tf.slice(A, [j + nb, j], [n - j - nb, nb])
        A22 = tf.slice(A, [j + nb, j + nb], [n - j - nb, n - j - nb])

        A11 = cholesky_unblocked(A11, nb)
        A21 = tf.matmul(A21, tf.matrix_inverse(tf.transpose(A11)))
        A22 = A22 - tf.matmul(A21, tf.transpose(A21))

        B_left = tf.concat(0, [A11, A21])
        B_right = add_top_padding(A22, nb, n - j - nb)
        B = tf.concat(1, [B_left, B_right])

        A_left = tf.slice(A, [0, 0], [n, j])
        A_right = add_top_padding(B, j, n - j)
        A = tf.concat(1, [A_left, A_right])

        # TODO remove this line
        A.set_shape([n, n])

        return A, j + nb

    def condition(A, j):
        return j < n

    return tf.while_loop(condition, body, [A, j])[0]


#
# main
#

def positive_definite_tensor(N):
    A = np.cov(randn(N, 3*N))
    A = tf.Variable(A, dtype="float64")
    return A

def test(N):
    A = positive_definite_tensor(N)

    A_chol_blocked = cholesky_blocked(A, 2)
    A_chol = tf.cholesky(A)

    with tf.Session() as s:
        tf.initialize_all_variables().run()

        A_chol_blocked = s.run(A_chol_blocked)
        A_chol = s.run(A_chol)

        print "blocked: \n", A_chol_blocked
        print "tf: \n", A_chol


def benchmark_blocked(block_sizes, matrix_order, plot):
    times = []
    BNs = []
    for BN in block_sizes:
        print "\nmatrix size: ", BN
        A = positive_definite_tensor(matrix_order)

        A_chol_blocked = cholesky_blocked(A, BN)

        with tf.Session() as s:
            tf.initialize_all_variables().run()

            t0 = time.time()
            result = s.run(A_chol_blocked)
            t1 = time.time()
            print "time: ", t1 - t0
            times += [t1 - t0]

        BNs += [BN]

    if plot:
        fig = plt.figure(0)
        fig.canvas.set_window_title('Block size')
        plt.plot(BNs, times, '--o', color='r')
        plt.xlabel('BN')
        plt.ylabel('time (s)')
        plt.show()


def benchmark_implementations(matrix_orders, plot, trace):
    implementations = {'tf': {'tensor_fn': tf.cholesky, 'durations': [], 'color': 'g'},
                       'blocked': {'tensor_fn': cholesky_blocked , 'durations': [], 'color': 'r'},
                       'unblocked': {'tensor_fn': cholesky_unblocked, 'durations': [], 'color': 'b'}}

    for N in matrix_orders:
        print "\nmatrix size: ", N
        A = positive_definite_tensor(N)
        BN = 200

        with tf.Session() as s:
            for impl, data in implementations.iteritems():
                print impl
                tensor = data['tensor_fn'](A)
                tf.initialize_all_variables().run()
                if trace:
                    run_metadata = tf.RunMetadata()
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    s.run(tensor, options=options, run_metadata=run_metadata)

                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open(impl + '_cholesky.json', 'w') as f:
                        f.write(ctf)
                else:
                    t0 = time.time()
                    result = s.run(tensor)
                    t1 = time.time()
                    data['durations'].append(t1 - t0)
                    print data['durations'][-1]



    if not trace and plot:
        for impl, data in implementations.iteritems():
            plt.semilogy(matrix_orders, data['durations'], color=data['color'], label=impl)

        plt.legend(loc='best')
        plt.xlabel('N')
        plt.ylabel('time (s)')
        plt.savefig('benchmark_cholesky_implementations.png')

# ###########
# MAIN
# ###########

# benchmark_blocked([1, 50, 100, 200, 400, 500], 2000, True)
# test(2)
benchmark_implementations([1000, 2000, 3000, 4000, 5000, 6000, 7000], True, False)
